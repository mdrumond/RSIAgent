/** Bounded, independently reviewed admission; approval is not promotion. */
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile,rename} from 'node:fs/promises';
import {join} from 'node:path';
import {prepareQaSources,summaryProtocol,validateSummary,summaryItems} from './skill-evidence-summary.mjs';
import {projectSummaries,learningProofEvidence} from './project-summary.mjs';
const reservedThrough=state=>Math.max(state.elapsedSeconds??0,state.inFlight?.reservedThrough??0,...Object.values(state.inFlightOperations??{}).map(op=>op.reservedThrough??0));
const digest=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
export const qaTopics=['action','mechanism','example','evidence','objective'];
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
export function qaIdentity(input,proposal=input.proposal){
 return digest({changes:proposal.changes,memoryRefs:proposal.memoryRefs,baseline:input.entries,objective:input.objective??null,target:input.target,evidence:input.groups,memoryEvidence:input.memoryEvidence??null,learning:input.learning??[],learningEvidence:input.learningEvidence??[],retrievalEvidenceHash:input.retrievalEvidenceHash??null,evaluationContract:input.evaluationContract??null});
}
export async function runProposalQa(input,ops){
 const concurrency=input.qaConcurrency??input.config?.skill_qa_concurrency??1;
 if(!Number.isSafeInteger(concurrency)||concurrency<1)throw Error('QA concurrency must be a positive integer');
 const entered=(ops.now??Date.now)(),summarized=typeof ops.qaSummarize==='function'&&typeof ops.qaAuditSummary==='function';
 input={...input,projectSummaries:input.projectSummaries??await projectSummaries(input.config,input.groups),learningProofEvidence:input.learningProofEvidence??await learningProofEvidence(input.learning,input.learningEvidence)};
 const projectFactsIdentity=digest(input.projectSummaries),catalogue=summarized?await prepareQaSources(input):null,proofProtocol=summarized?summaryProtocol:'claim-proofs-v1';
 await mkdir(input.directory,{recursive:true});let file=join(input.directory,'qa.json');
 let state;try{state=JSON.parse(await readFile(file,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
 const identity=qaIdentity(input),now=ops.now??Date.now;

 if(state&&state.format!==2){const legacy=state;file=join(input.directory,'qa-v2.json');try{state=JSON.parse(await readFile(file,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;state={format:2,protocol:'skill-qa-v2',inputIdentity:identity,startedAt:now(),seconds:300,elapsedSeconds:legacy.elapsedSeconds??0,draft:input.proposal,rounds:[],resumeFrom:join(input.directory,'qa.json')};}}
 if(!state&&input.resumeFrom){const legacy=JSON.parse(await readFile(input.resumeFrom,'utf8'));state={format:2,protocol:'skill-qa-v2',inputIdentity:identity,startedAt:now(),seconds:300,elapsedSeconds:Math.max(input.priorElapsedSeconds??0,reservedThrough(legacy)),draft:input.proposal,rounds:[],resumeFrom:input.resumeFrom};}
 if(state&&state.inputIdentity!==identity)throw Error('QA input changed; use a new proposal QA directory');
 state??={format:2,proofProtocol,projectFactsIdentity,...(summarized?{sourceIdentity:catalogue.identity}:{}),protocol:'skill-qa-v2',inputIdentity:identity,startedAt:now(),seconds:300,elapsedSeconds:input.priorElapsedSeconds??0,draft:input.proposal,rounds:[]};
 if(state&&(state.proofProtocol!==proofProtocol||state.projectFactsIdentity!==projectFactsIdentity||summarized&&state.sourceIdentity!==catalogue.identity)){const upgrade={verdict:state.verdict??null,rounds:state.rounds??[],projectFactsIdentity:state.projectFactsIdentity??null,summary:state.summary??null,sourceIdentity:state.sourceIdentity??null};state={...state,proofUpgrade:upgrade,proofUpgrades:[...(state.proofUpgrades??[]),upgrade],rounds:[],summary:undefined,draft:state.proposal??state.draft??input.proposal};for(const key of ['verdict','approval','proposal','reason','reasonKind','retryable','error'])delete state[key];}
 state.proofProtocol=proofProtocol;state.projectFactsIdentity=projectFactsIdentity;if(summarized)state.sourceIdentity=catalogue.identity;
 // Reserve a bounded slice before provider work. A hard loss charges that slice,
 // never paused wall time; graceful completion releases the unused reservation.
 const begun=entered,charged=reservedThrough(state);
 delete state.inFlight;delete state.inFlightOperations;
 const elapsed=()=>charged+Math.max(0,(now()-begun)/1000);
 let writes=Promise.resolve();
 const persist=()=>{state.elapsedSeconds=elapsed();const snapshot=JSON.stringify(state,null,2)+'\n';writes=writes.then(async()=>{const tmp=file+'.tmp';await writeFile(tmp,snapshot);await rename(tmp,file);});return writes;};
 const remaining=()=>Math.max(0,state.seconds-elapsed());
 if(state.verdict&&!state.retryable)return state;
 if(state.retryable){delete state.verdict;delete state.reason;delete state.error;delete state.reasonKind;delete state.retryable;}
 const stop=new AbortController(),signal=AbortSignal.any([stop.signal,...(input.signal?[input.signal]:[])]),waiting=[];
 let active=0;
 const acquire=async()=>{signal.throwIfAborted();if(active<concurrency)active++;else await new Promise(resolve=>waiting.push(resolve));};
 const release=()=>{const next=waiting.shift();if(next)next();else active--;};
 const parallel=async(values,work)=>{
  let cursor=0,failed;
  const workers=Array.from({length:Math.min(concurrency,values.length)},async()=>{
   while(cursor<values.length&&!signal.aborted){const index=cursor++;try{await work(values[index],index);}catch(error){failed??=error;stop.abort(error);}}
  });
  await Promise.allSettled(workers);if(failed)throw failed;signal.throwIfAborted();
 };
 const call=async(name,payload,key)=>{
  await acquire();let heartbeat;
  try{
   signal.throwIfAborted();if(remaining()<=0)throw Error('QA elapsed allocation exhausted');
   const seconds=Math.min(remaining(),input.seconds??300),bounded=AbortSignal.any([signal,AbortSignal.timeout(Math.max(1,Math.ceil(seconds*1000)))]);
   const directory=join(input.directory,'sessions',state.protocol,identity,state.proofProtocol,projectFactsIdentity,...(summarized?[catalogue.identity]:[]),key);await mkdir(directory,{recursive:true});
   const reserve=()=>{(state.inFlightOperations??={})[key]={stage:name,reservedThrough:Math.min(state.seconds,elapsed()+5)};return persist();};
   await reserve();let checkpointError;
   heartbeat=setInterval(()=>{void reserve().catch(error=>{checkpointError=error;});},1000);
   const result=await ops[name]({...input,...payload,directory,seconds,signal:bounded});if(checkpointError)throw checkpointError;return result;
  }catch(error){stop.abort(error);throw error;}
  finally{
   clearInterval(heartbeat);
   if(state.inFlightOperations){delete state.inFlightOperations[key];if(!Object.keys(state.inFlightOperations).length)delete state.inFlightOperations;}
   try{await persist();}finally{release();}
  }
 };

 let proposal=state.rounds.at(-1)?.proposal??state.draft;
 try{
  while(remaining()>0){
   const round=state.rounds.at(-1)?.review?.verdict==='revise'||!state.rounds.length?{proposal}:state.rounds.at(-1);
   if(round!==state.rounds.at(-1)){state.rounds.push(round);await persist();}
   const index=state.rounds.length-1;
   if(summarized){
    const summary=state.summary??={pages:{},audits:{}};
    await parallel(catalogue.pages,async page=>{
     let row=summary.pages[page.id]??={revisions:[],value:null};
     while(remaining()>0){
      if(!row.value){
       const result=await call('qaSummarize',{sourceCatalogue:catalogue,page,previousReview:row.revisions.at(-1)?.audits??[],validate:value=>validateSummary(value,catalogue)},`summary-${page.id}-${row.revisions.length}`);
       row.value=validateSummary(result,catalogue);await persist();
      }
      const items=summaryItems(row.value,page.id);
      await parallel(items,async item=>{
       if(!summary.audits[item.id])summary.audits[item.id]=await call('qaAuditSummary',{sourceCatalogue:catalogue,item,validate:value=>{if(!['supported','revise','unknown'].includes(value.verdict)||!value.reason?.trim())throw Error('Summary audit needs a verdict and reason');return value;}},`summary-audit-${item.id}`);
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
    const refs=new Set([...(catalogue?.records??[]).map(r=>r.reference),...input.groups.flatMap(g=>[...g.occurrences,...(g.linkedDiagnostics??[])].map(o=>o.reference)),...input.entries.map(e=>e.id),...input.projectSummaries.flatMap(s=>[...(s.facts??[]),...(s.references??[])].map(r=>r.reference)),...input.learningProofEvidence.filter(r=>r.available!==false&&typeof r.text==='string').map(r=>r.reference),...(proposal.memoryRefs??[]),...(input.learning??[]).flatMap(([,record])=>(record.result?.evidence??[]).flatMap(e=>e.reference?[e.reference]:[]))]);
    const validate=value=>{
     validateQaAnswers(value,refs);
     if(value.memoryRefs?.some(ref=>!(input.proposal.memoryRefs??[]).includes(ref)))throw Error('QA revisions cannot introduce unretrieved memory references');
     if(value.reason!==undefined&&(!value.reason?.trim()||!value.claims?.some(c=>c.path==='reason'&&c.text===value.reason)))throw Error('Revised proposal reason requires its exact source-bound claim');
     if(value.changes||value.reason!==undefined){const revised=input.validateProposal({...proposal,...(value.reason!==undefined?{reason:value.reason}:{}),changes:value.changes??proposal.changes,...(value.memoryRefs?{memoryRefs:value.memoryRefs}:{})});return {...value,changes:revised.changes};}
     return value;
    };
    const result=validate(await call('qaAnswer',{proposal,previousReview:state.rounds[index-1]?.review,validate},`${index}-qaAnswer`));
    if(result.changes||result.reason!==undefined)proposal=input.validateProposal({...proposal,...(result.reason!==undefined?{reason:result.reason}:{}),changes:result.changes??proposal.changes,...(result.memoryRefs?{memoryRefs:result.memoryRefs}:{})});
    round.proposal=proposal;round.answers=result.answers;if(result.claims)round.claims=result.claims;await persist();
   }
   if(!round.review){const validate=value=>{validateQaVerdict(value);if(value.learningRequests?.length)input.validateProposal({...round.proposal,learningRequests:value.learningRequests});return value;};
    round.review=validate(await call('qaReview',{proposal:round.proposal,answers:round.answers,answerClaims:round.claims,validate},`${index}-qaReview`));await persist();}
   if(remaining()<=0)throw Error('QA elapsed allocation exhausted');
   if(round.review.verdict==='revise'){proposal=round.proposal;continue;}
   state.verdict=round.review.verdict;state.reason=round.review.reason;
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
  state.verdict='inconclusive';state.reason=remaining()<=0||error.name==='TimeoutError'?'qa-elapsed':'qa-execution-failed';state.reasonKind='infrastructure';state.retryable=remaining()>0;state.error={name:error.name,message:error.message};
 }
 state.reasonKind??=state.reason==='qa-elapsed'?'infrastructure':'scientific';state.retryable??=false;await persist();return state;
}
