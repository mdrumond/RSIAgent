/** Bounded, independently reviewed admission; approval is not promotion. */
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile,rename} from 'node:fs/promises';
import {join} from 'node:path';
import {prepareQaSources,summaryProtocol,validateSummary,summaryItems} from './skill-evidence-summary.mjs';
import {frozenEvidencePacket,scopedQaGroups} from './skill-review-evidence.mjs';
import {reopenMissingVotes,runPanelRound} from './skill-qa-panel-run.mjs';
import {projectSummaries,learningProofEvidence} from './project-summary.mjs';
const reservedThrough=state=>Math.max(state.elapsedSeconds??0,state.inFlight?.reservedThrough??0,...Object.values(state.inFlightOperations??{}).map(op=>op.reservedThrough??0));
const digest=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
export const qaTopics=['action','mechanism','example','evidence','objective'];
export const qaFindingKinds=['answer','draft','evidence','advisory'];
const panelBudgets={answer:300,review:600,audit:120};
const phaseSeconds=300;
const findings=value=>(value.issues??[]).filter(issue=>issue?.kind!=='advisory');
function validateQaFindings(value){
 for(const issue of value.issues??[]){
  if(issue?.kind===undefined)continue;
  if(!qaFindingKinds.includes(issue.kind))throw Error('Unknown QA finding kind: '+issue.kind);
  if(issue.kind==='answer'&&!qaTopics.some(topic=>issue.path==='answers.'+topic+'.text'))
   throw Error('Answer finding requires an exact answers.<topic>.text path');
 }
 if(value.verdict==='approved'&&findings(value).length)throw Error('Approval cannot retain material findings');
 if(value.verdict==='revise'&&!findings(value).length)throw Error('Revision requires a material finding');
 return value;
}
export function validateQaAnswers(value,refs){
 if(!value?.answers)throw Error('QA requires five structured answers');
 for(const topic of qaTopics){const answer=value.answers[topic];
  if(!answer?.text?.trim()||!Array.isArray(answer.refs)||answer.refs.some(ref=>!refs.has(ref)))throw Error('QA answer requires text and actual evidence references: '+topic);
 }
 return value;
}
export function validateQaVerdict(value){
 if(!['approved','revise','learn','inconclusive'].includes(value?.verdict)||!value.reason?.trim())throw Error('QA requires a verdict and reason');
 if(value.verdict==='revise'&&(!Array.isArray(value.issues)||!value.issues.length))throw Error('Revision requires material issues');
 if(value.verdict==='learn'&&(!Array.isArray(value.learningRequests)||!value.learningRequests.length))throw Error('Learning verdict requires learning requests');
 return value;
}
export function validateSummaryTargets(value,items=[]){
 const ids=new Set(items.map(item=>item.id));
 for(const issue of value.issues??[])if(issue.summaryItemId!==undefined&&!ids.has(issue.summaryItemId))throw Error('Unknown summaryItemId: '+issue.summaryItemId);
 return value;
}
export function qaIdentity(input,proposal=input.proposal){
 return digest({changes:proposal.changes,memoryRefs:proposal.memoryRefs,baseline:input.entries,objective:input.objective??null,target:input.target,evidence:input.groups,memoryEvidence:input.memoryEvidence??null,learning:input.learning??[],learningEvidence:input.learningEvidence??[],retrievalEvidenceHash:input.retrievalEvidenceHash??null,evaluationContract:input.evaluationContract??null});
}
export async function runProposalQa(input,ops){
 const concurrency=input.qaConcurrency??input.config?.skill_qa_concurrency??1;
 if(!Number.isSafeInteger(concurrency)||concurrency<1)throw Error('QA concurrency must be a positive integer');
 const requestedPolicy=input.qaPolicy,failureFirst=input.evaluationContract?.workflow==='failure-first-v1';
 if(requestedPolicy!==undefined&&!['shared','split','panel'].includes(requestedPolicy))throw Error('Unknown QA policy: '+requestedPolicy);
 if(['split','panel'].includes(requestedPolicy)&&!failureFirst)throw Error('Split QA requires a failure-first evaluation contract');
 const entered=(ops.now??Date.now)(),summarized=typeof ops.qaSummarize==='function'&&typeof ops.qaAuditSummary==='function';
 input={...input,projectSummaries:input.projectSummaries??await projectSummaries(input.config,input.groups),learningProofEvidence:input.learningProofEvidence??await learningProofEvidence(input.learning,input.learningEvidence)};
 const projectFactsIdentity=digest(input.projectSummaries),catalogue=summarized?await prepareQaSources(input):null,proofProtocol=summarized?summaryProtocol:'claim-proofs-v1';
 await mkdir(input.directory,{recursive:true});let file=join(input.directory,'qa.json');
 let state;try{state=JSON.parse(await readFile(file,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
 const identity=qaIdentity(input),now=ops.now??Date.now;

 if(state&&state.format!==2){const legacy=state;file=join(input.directory,'qa-v2.json');try{state=JSON.parse(await readFile(file,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;state={format:2,protocol:'skill-qa-v2',inputIdentity:identity,startedAt:now(),seconds:300,elapsedSeconds:legacy.elapsedSeconds??0,draft:input.proposal,rounds:[],resumeFrom:join(input.directory,'qa.json')};}}
 if(!state&&input.resumeFrom){const legacy=JSON.parse(await readFile(input.resumeFrom,'utf8'));state={format:2,protocol:'skill-qa-v2',inputIdentity:identity,startedAt:now(),seconds:300,elapsedSeconds:Math.max(input.priorElapsedSeconds??0,reservedThrough(legacy)),draft:input.proposal,rounds:[],resumeFrom:input.resumeFrom};}
 if(state&&state.inputIdentity!==identity)throw Error('QA input changed; use a new proposal QA directory');
 if(state?.qaPolicy&&requestedPolicy&&state.qaPolicy!==requestedPolicy)throw Error('Retained QA policy changed');
 const policy=state?.qaPolicy??(state?'shared':requestedPolicy??(failureFirst?'split':'shared'));
 const panel=policy==='panel',split=policy!=='shared',qaLanes=panel?Object.keys(panelBudgets):['answer','review'];
 const panelConcurrency=input.qaPanelConcurrency??5;
 if(panel&&(!Number.isSafeInteger(panelConcurrency)||panelConcurrency<1||panelConcurrency>5))throw Error('Panel concurrency must be between one and five');
 if(panel&&state&&state.panelAuditEvidenceIdentity!==digest(input.panelAuditEvidence??[]))throw Error('Panel audit sources changed on resume');
 if(input.frozenQa){
  if(!panel||input.frozenQa.inputIdentity!==identity||digest(input.frozenQa.draft)!==digest(input.proposal))throw Error('Frozen answers belong to a different draft or QA input');
  if(state&&state.frozenQaIdentity!==digest(input.frozenQa))throw Error('Frozen QA inputs changed on resume');
 }
 if(state?.frozenQaIdentity&&!input.frozenQa)throw Error('Resume requires the original frozen QA inputs');
 state??={format:2,qaPolicy:policy,proofProtocol,projectFactsIdentity,...(summarized?{sourceIdentity:catalogue.identity}:{}),protocol:'skill-qa-v2',inputIdentity:identity,startedAt:now(),seconds:panel?1020:split?2*phaseSeconds:phaseSeconds,elapsedSeconds:input.priorElapsedSeconds??0,draft:input.proposal,rounds:[]};
 if(state&&(state.proofProtocol!==proofProtocol||state.projectFactsIdentity!==projectFactsIdentity||summarized&&state.sourceIdentity!==catalogue.identity)){const upgrade={verdict:state.verdict??null,rounds:state.rounds??[],projectFactsIdentity:state.projectFactsIdentity??null,summary:state.summary??null,sourceIdentity:state.sourceIdentity??null};state={...state,proofUpgrade:upgrade,proofUpgrades:[...(state.proofUpgrades??[]),upgrade],rounds:[],summary:undefined,draft:state.proposal??state.draft??input.proposal};for(const key of ['verdict','approval','proposal','reason','reasonKind','retryable','error'])delete state[key];}
 state.qaPolicy=policy;input={...input,qaPolicy:policy};
 if(split)state.phaseSeconds??=panel?{...panelBudgets}:{answer:phaseSeconds,review:phaseSeconds};
 if(panel)state.panelAuditEvidenceIdentity=digest(input.panelAuditEvidence??[]);
 if(input.frozenQa)state.frozenQaIdentity=digest(input.frozenQa);
 state.proofProtocol=proofProtocol;state.projectFactsIdentity=projectFactsIdentity;if(summarized)state.sourceIdentity=catalogue.identity;
 // Reserve a bounded slice before provider work. A hard loss charges that slice,
 // never paused wall time; graceful completion releases the unused reservation.
 const begun=entered,charged=reservedThrough(state);
 const phaseClocks=Object.fromEntries(qaLanes.map(lane=>[lane,{active:0,started:0,spent:Math.max(state.phaseElapsed?.[lane]??0,
  ...Object.values(state.inFlightOperations??{}).filter(op=>op.lane===lane).map(op=>op.reservedPhaseThrough??0))}]));
 const phaseElapsed=lane=>{
  const phase=phaseClocks[lane];
  return phase.spent+(phase.active?Math.max(0,(now()-phase.started)/1000):0);
 };
 delete state.inFlight;delete state.inFlightOperations;
 const elapsed=()=>charged+Math.max(0,(now()-begun)/1000);
 let writes=Promise.resolve();
 const persist=()=>{state.elapsedSeconds=elapsed();if(split)state.phaseElapsed=Object.fromEntries(qaLanes.map(lane=>[lane,phaseElapsed(lane)]));const snapshot=JSON.stringify(state,null,2)+'\n';writes=writes.then(async()=>{const tmp=file+'.tmp';await writeFile(tmp,snapshot);await rename(tmp,file);});return writes;};
 const remaining=lane=>Math.max(0,Math.min(state.seconds-elapsed(),split&&lane?state.phaseSeconds[lane]-phaseElapsed(lane):Infinity));
 if(panel&&input.retryMissingPanelVotes&&state.verdict==='inconclusive'&&remaining('review')>0&&reopenMissingVotes(state.rounds.at(-1)))state.retryable=true;
 if(state.verdict&&!state.retryable)return state;
 if(state.retryable){delete state.verdict;delete state.reason;delete state.error;delete state.reasonKind;delete state.retryable;}
 const stop=new AbortController(),signal=AbortSignal.any([stop.signal,...(input.signal?[input.signal]:[])]),waiting=[];
 let active=0,failedLane;
 const acquire=async()=>{signal.throwIfAborted();if(active<concurrency)active++;else await new Promise(resolve=>waiting.push(resolve));};
 const release=()=>{const next=waiting.shift();if(next)next();else active--;};
 const parallel=async(values,work)=>{
  let cursor=0,failed;
  const workers=Array.from({length:Math.min(concurrency,values.length)},async()=>{
   while(cursor<values.length&&!signal.aborted){const index=cursor++;try{await work(values[index],index);}catch(error){failed??=error;stop.abort(error);}}
  });
  await Promise.allSettled(workers);if(failed)throw failed;signal.throwIfAborted();
 };
 const call=async(name,payload,key,independent=false)=>{
  if(!independent)await acquire();let heartbeat,started;
  const lane=name==='qaReview'?'review':name==='qaPanelAudit'?'audit':'answer',phase=phaseClocks[lane];
  try{
   signal.throwIfAborted();if(remaining(lane)<=0)throw new DOMException('QA '+lane+' elapsed allocation exhausted','TimeoutError');
   const seconds=Math.min(remaining(lane),input.seconds??(panel?state.phaseSeconds[lane]:phaseSeconds)),deadline=AbortSignal.timeout(Math.max(1,Math.ceil(seconds*1000))),bounded=AbortSignal.any([signal,deadline]);
   if(split){started=now();if(!phase.active)phase.started=started;phase.active++;}
   const directory=join(input.directory,'sessions',state.protocol,identity,state.proofProtocol,projectFactsIdentity,...(summarized?[catalogue.identity]:[]),key);await mkdir(directory,{recursive:true});
   const reserve=()=>{(state.inFlightOperations??={})[key]={stage:name,reservedThrough:Math.min(state.seconds,elapsed()+5),...(split?{lane,reservedPhaseThrough:Math.min(state.phaseSeconds[lane],phaseElapsed(lane)+5)}:{})};return persist();};
   await reserve();let checkpointError;
   heartbeat=setInterval(()=>{void reserve().catch(error=>{checkpointError=error;});},1000);
   const result=await ops[name]({...input,...payload,directory,seconds,signal:bounded});if(checkpointError)throw checkpointError;
   if(split){
    signal.throwIfAborted();
    if(deadline.aborted||(now()-started)/1000>seconds)throw new DOMException('QA '+lane+' elapsed allocation exhausted','TimeoutError');
   }
   return result;
  }catch(error){if(!independent){failedLane??=lane;stop.abort(error);}throw error;}
  finally{
   clearInterval(heartbeat);
   if(started!==undefined){const spent=phaseElapsed(lane);phase.active--;if(!phase.active)phase.spent=spent;}
   if(state.inFlightOperations){delete state.inFlightOperations[key];if(!Object.keys(state.inFlightOperations).length)delete state.inFlightOperations;}
   try{await persist();}finally{if(!independent)release();}
  }
 };

 const repairSummary=round=>{
  if(!summarized||round.summaryRequeued||round.review?.verdict!=='revise')return;
  const items=catalogue.pages.flatMap(page=>{const value=state.summary?.pages?.[page.id]?.value;return value?summaryItems(value,page.id):[];});
  validateSummaryTargets(round.review,items);
  const issues=(split?findings(round.review):round.review.issues??[]).filter(issue=>issue.summaryItemId!==undefined),pages=new Set(issues.map(issue=>items.find(item=>item.id===issue.summaryItemId).pageId));
  for(const pageId of pages){const row=state.summary.pages[pageId],previous=summaryItems(row.value,pageId),priorAudits=previous.map(item=>({itemId:item.id,...state.summary.audits[item.id]}));
   const feedback=issues.filter(issue=>previous.some(item=>item.id===issue.summaryItemId)).map(issue=>({itemId:issue.summaryItemId,verdict:'revise',reason:round.review.reason,issues:[issue]}));
   row.revisions.push({source:'final-review',value:row.value,audits:feedback,priorAudits,review:round.review});row.value=null;
   for(const item of previous)delete state.summary.audits[item.id];
  }
  round.summaryRequeued=true;
 };
 let proposal=state.rounds.at(-1)?.proposal??state.draft;
 try{
  while(remaining()>0){
   let round=state.rounds.at(-1);
   if(!round){round={proposal};state.rounds.push(round);await persist();}
   const index=state.rounds.length-1;
   if(summarized&&!round.review){
    const summary=state.summary??={pages:{},audits:{}};
    await parallel(catalogue.pages,async page=>{
     let row=summary.pages[page.id]??={revisions:[],value:null};
     while(remaining()>0){
      if(!row.value){
       const result=await call('qaSummarize',{sourceCatalogue:catalogue,page,previousReview:row.revisions.at(-1)?.audits??[],validate:value=>validateSummary(value,catalogue)},`summary-${page.id}-${row.revisions.length}`);
       row.value=validateSummary(result,catalogue);await persist();
      }
      const items=[...new Map(summaryItems(row.value,page.id).map(item=>[item.id,item])).values()];
      await parallel(items,async item=>{
       if(!summary.audits[item.id])summary.audits[item.id]=await call('qaAuditSummary',{sourceCatalogue:catalogue,item,validate:value=>{if(!['supported','revise','unknown'].includes(value.verdict)||!value.reason?.trim())throw Error('Summary audit needs a verdict and reason');return value;}},`summary-audit-${item.id}${row.revisions.length?'-revision-'+row.revisions.length:''}`);
       await persist();
      });
      const audits=items.map(item=>({itemId:item.id,...summary.audits[item.id]}));
      if(audits.every(a=>a.verdict==='supported'))break;
      row.revisions.push({value:row.value,audits});row.value=null;await persist();
     }
     if(!row.value)throw Error('QA elapsed allocation exhausted before summary support');
    });
    input={...input,sourceCatalogue:catalogue,evidenceSummary:{items:catalogue.pages.flatMap(p=>summaryItems(summary.pages[p.id].value,p.id))}};
   }
   if(!round.answers){
    const artifactRefs=summarized?[]:(await frozenEvidencePacket(scopedQaGroups(input.groups,proposal))).artifacts.filter(a=>a.available).map(a=>a.id);
    const refs=new Set([...artifactRefs,...(catalogue?.records??[]).map(r=>r.reference),...input.groups.flatMap(g=>[...g.occurrences,...(g.linkedDiagnostics??[])].map(o=>o.reference)),...input.entries.map(e=>e.id),...input.projectSummaries.flatMap(s=>[...(s.facts??[]),...(s.references??[])].map(r=>r.reference)),...input.learningProofEvidence.filter(r=>r.available!==false&&typeof r.text==='string').map(r=>r.reference),...(proposal.memoryRefs??[]),...(input.learning??[]).flatMap(([,record])=>(record.result?.evidence??[]).flatMap(e=>e.reference?[e.reference]:[]))]);
    const validate=value=>{
     validateQaAnswers(value,refs);
     if(round.answerCorrection&&['changes','memoryRefs','reason'].some(key=>Object.hasOwn(value,key)))throw Error('Answer-only correction must preserve the proposal');
     if(value.memoryRefs?.some(ref=>!(input.proposal.memoryRefs??[]).includes(ref)))throw Error('QA revisions cannot introduce unretrieved memory references');
     if(value.reason!==undefined&&(!value.reason?.trim()||!value.claims?.some(c=>c.path==='reason'&&c.text===value.reason)))throw Error('Revised proposal reason requires its exact source-bound claim');
     if(value.changes||value.reason!==undefined){const revised=input.validateProposal({...proposal,...(value.reason!==undefined?{reason:value.reason}:{}),changes:value.changes??proposal.changes,...(value.memoryRefs?{memoryRefs:value.memoryRefs}:{})});return {...value,changes:revised.changes};}
     return value;
    };
    const frozen=index===0&&input.frozenQa;
    if(frozen){
     proposal=input.validateProposal(frozen.proposal);
     if(!Array.isArray(frozen.claims)||!frozen.claims.length)throw Error('Frozen answers need their accepted proof claims');
    }
    const result=frozen?validate({answers:frozen.answers,claims:frozen.claims}):validate(await call('qaAnswer',{proposal,previousReview:state.rounds[index-1]?.review,answerCorrection:round.answerCorrection,validate},`${index}-qaAnswer`));
    if(result.changes||result.reason!==undefined)proposal=input.validateProposal({...proposal,...(result.reason!==undefined?{reason:result.reason}:{}),changes:result.changes??proposal.changes,...(result.memoryRefs?{memoryRefs:result.memoryRefs}:{})});
    round.proposal=proposal;round.answers=result.answers;if(result.claims)round.claims=result.claims;await persist();
   }
   if(!round.review){const validate=value=>{validateQaVerdict(value);if(split)validateQaFindings(value);validateSummaryTargets(value,input.evidenceSummary?.items??[]);if(value.learningRequests?.length)input.validateProposal({...round.proposal,learningRequests:value.learningRequests});return value;};
    const payload={proposal:round.proposal,answers:round.answers,answerClaims:round.claims,previousReview:state.rounds[index-1]?.review,validate};
    round.review=panel?await runPanelRound({round,payload,index,call,persist,now,signal,concurrency:panelConcurrency}):validate(await call('qaReview',payload,`${index}-qaReview`));await persist();}
   if(remaining()<=0)throw Error('QA elapsed allocation exhausted');
   if(round.review.verdict==='revise'){
    const blocking=findings(round.review),answerOnly=split&&blocking.length>0&&blocking.every(issue=>issue?.kind==='answer');
    if(split&&blocking.some(issue=>issue?.kind==='evidence')){
     state.verdict='inconclusive';state.reason=round.review.reason;break;
    }
    repairSummary(round);
    proposal=round.proposal;
    if(answerOnly){
     if(state.answerCorrectionUsed||remaining('answer')<=0||remaining('review')<=0){
      state.verdict='inconclusive';state.reason=round.review.reason;break;
     }
     state.answerCorrectionUsed=true;
     state.rounds.push({proposal,answerCorrection:{findings:blocking,previousAnswers:round.answers,previousClaims:round.claims??[]}});
    }else{
     if(panel||input.allowWriterRevision){state.verdict='inconclusive';state.reason=round.review.reason;if(input.allowWriterRevision)state.revisionFeedback=panel?JSON.stringify({reason:round.review.reason,issues:blocking}):round.review.reason;break;}
     state.rounds.push({proposal});
    }
    await persist();continue;
   }
   state.verdict=round.review.verdict;state.reason=round.review.reason;
   if(panel&&state.verdict==='inconclusive'&&(round.panel.auditError||round.review.missingSlots.length>=3))state.reasonKind='infrastructure';
   state.proposal=round.proposal;state.learningRequests=round.review.learningRequests??[];
   if(state.verdict==='approved'){
    if(summarized)for(const key of ['pages','audits'])state.summary[key]=Object.fromEntries(Object.entries(state.summary[key]).sort(([a],[b])=>a.localeCompare(b)));
    state.approval=summarized?digest({proposal:qaIdentity(input,round.proposal),sources:catalogue.identity,summary:state.summary}):qaIdentity(input,round.proposal);
   }
   break;
  }
  if(!state.verdict){state.verdict='inconclusive';state.reason='qa-elapsed';}
 }catch(error){
  if(input.signal?.aborted){await persist();throw error;}
  state.verdict='inconclusive';state.reason=remaining()<=0||error.name==='TimeoutError'?'qa-elapsed':'qa-execution-failed';state.reasonKind='infrastructure';state.retryable=remaining(failedLane)>0;state.error={name:error.name,message:error.message,...(error.reviewSession?{reviewSession:error.reviewSession}:{})};
 }
 state.reasonKind??=state.reason==='qa-elapsed'?'infrastructure':'scientific';state.retryable??=false;await persist();return state;
}
