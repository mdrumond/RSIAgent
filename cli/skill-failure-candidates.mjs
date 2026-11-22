/** Evidence-reviewed instruction candidates; only the tested immutable version can publish. */
import assert from 'node:assert/strict';
import {mkdir,readFile,writeFile,rename} from 'node:fs/promises';
import {setTimeout as delay} from 'node:timers/promises';
import {join,resolve} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {Type} from 'typebox';
import {guidanceEntries} from '../controller/skill-review.mjs';
import {stages} from '../controller/workflow.mjs';
import {applyProposal,writeVersion} from '../controller/skill-evolution.mjs';
import {skillInput} from './skill-microeval.mjs';
import {claimEvidence} from './project-summary.mjs';
import {frozenEvidencePacket} from './skill-review-evidence.mjs';
import {reviewSession,nativeQaAnswer,nativeQaReview,nativeQaPanelAudit} from './skill-review-native.mjs';
import {qaIdentity,runProposalQa} from './skill-review-qa.mjs';
import {panelCorrectionSources,replayPanelDecision} from './skill-qa-panel-run.mjs';
import {validateReviewProposal} from './skill-review.mjs';
import {publishSkillVersion} from './skills.mjs';
import {readConfig} from './instance.mjs';
import {focusedBaselineDecision,pilotWinner,validatePilotTrial,validatePilotManifest} from './skill-failure-pilot.mjs';

const hash=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const textHash=value=>createHash('sha256').update(value).digest('hex');
const load=async file=>{try{return JSON.parse(await readFile(file,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;return null;}};
const save=async(file,value)=>{await mkdir(join(file,'..'),{recursive:true});const temporary=file+'.'+randomUUID();await writeFile(temporary,JSON.stringify(value,null,2));await rename(temporary,file);};
const scopes=['global-kdb',...stages];
const unsubmittedKinds=['no-submit','rejected-submissions'];
const assertInstruction=value=>{assert(typeof value==='string'&&value.trim()&&value.trim().split(/\s+/).length<=150&&!value.includes('```'),'A short prose instruction of at most 150 words is required');};
const selector=Type.Object({reference:Type.String({minLength:1}),startLine:Type.Optional(Type.Integer({minimum:1})),endLine:Type.Optional(Type.Integer({minimum:1})),segment:Type.Optional(Type.Integer({minimum:1})),field:Type.Optional(Type.String())},{additionalProperties:false});
const schema=Type.Object({text:Type.String({minLength:1}),scope:Type.Union(scopes.map(value=>Type.Literal(value))),memoryRefs:Type.Array(Type.String({minLength:1}),{minItems:1}),claims:Type.Array(Type.Object({path:Type.Literal('text'),kind:Type.Union([Type.Literal('observation'),Type.Literal('hypothesis')]),proofs:Type.Array(selector,{minItems:1})},{additionalProperties:false}),{minItems:1})},{additionalProperties:false});
const memoryEntries=retrievals=>[...new Map(retrievals.flatMap(record=>(record.entries??[]).map(entry=>({...entry,ref:entry.ref??`${entry.collection}:${entry.entry_id??entry.chunk_id??entry.id}`}))).map(entry=>[entry.ref,entry])).values()];
async function writerHealthy(written){
 if(!written.session)return;
 for(const line of (await readFile(written.session,'utf8')).split('\n').filter(Boolean)){
  const message=JSON.parse(line).message;if(message?.role!=='toolResult'||!message.isError||!message.toolName?.startsWith('mcp__kdb__'))continue;
  const text=JSON.stringify(message.content);assert(!/timed out|timeout|transport|connection|unavailable/i.test(text),'Writer memory transport failed; instruction evidence is infrastructure-contaminated');
 }
}
function reviewGroups(item){return item.evidence?.groups??[];}
const designIdentity=({base,baseline,family,results,identities,item})=>hash({base,baseline,family,results,identities,origin:{id:item.id,groupIds:item.groupIds,evidence:item.evidence,classifications:item.preparation?.classifications}});
function designEvidence(item,task,baselines,qualification){
 const groups=structuredClone(reviewGroups(item));
 assert(groups.some(group=>group.occurrences?.length),'Original failure evidence is required');
 const outcomes=baselines.map(result=>({correct:result.correct,mechanismMatch:result.mechanismMatch,sourceSha256:result.sourceSha256,failure:result.failure}));
 groups[0].linkedDiagnostics=[...(groups[0].linkedDiagnostics??[]),...outcomes.map((data,index)=>({
  reference:'failure-first:baseline:'+index,type:'isolated-baseline-verification',data,artifacts:baselines[index].artifacts??[],
 }))];
 const development=Object.fromEntries(['id','prompt','files','caseIds'].filter(key=>task[key]!==undefined).map(key=>[key,task[key]]));
 const proofRecords=outcomes.map((data,index)=>({reference:'baseline-'+index,text:JSON.stringify(data)}));
 if(qualification){
  const {referencePassed,negativeFailed,calibrationPassed}=qualification;
  proofRecords.push({reference:'development-controls',text:JSON.stringify({referencePassed,negativeFailed,calibrationPassed})});
 }
 return {groups,development,proofRecords,evaluationContract:{workflow:'failure-first-v1',development,baselineOutcomes:proofRecords}};
}
async function checkVersion(config,candidate){
 const effective=await skillInput(join(config.root,'skills','versions',candidate.version));
 assert.equal(effective.text,candidate.text,'Candidate guidance changed after review');assert.equal(effective.sha256,candidate.skillSha256,'Candidate source changed after review');
 assert.equal(textHash(candidate.text),candidate.textSha256,'Candidate text identity mismatch');return candidate;
}
/** Reconstruct the panel vote before opening device trials for its exact proposal. */
export function verifyPanelApproval(qa,proposal){
 assert.equal(qa?.qaPolicy,'panel','Expected panel QA');
 assert.equal(qa.verdict,'approved','Panel QA must approve the candidate');
 const round=qa.rounds?.at(-1);
 assert.deepEqual(round?.proposal,proposal,'Panel reviewed a different proposal');
 const decision=replayPanelDecision(round.panel);
 assert.equal(decision.verdict,'approved','Retained panel votes do not approve the candidate');
 assert.deepEqual(round.panel.decision,decision,'Retained panel decision changed');
}

async function writerCall(directory,seconds,signal,fn,now){
 signal?.throwIfAborted();
 const file=join(directory,'writer-spend.json'),prior=await load(file)??{seconds:180,elapsedSeconds:0};
 assert(!prior.running,'Interrupted writer elapsed custody is unknown');const remaining=Math.min(seconds,prior.seconds-prior.elapsedSeconds);if(remaining<=0)return {providerTimeout:'elapsed'};
 const started=now();let writes=Promise.resolve();const snapshot=running=>({...prior,elapsedSeconds:prior.elapsedSeconds+(now()-started)/1000,running});
 await save(file,snapshot(true));const timer=setInterval(()=>{writes=writes.then(()=>save(file,snapshot(true)));},1000);
 try{
  const bounded=AbortSignal.any([...(signal?[signal]:[]),AbortSignal.timeout(Math.max(1,Math.ceil(remaining*1000)))]);
  bounded.throwIfAborted();
  return {written:await fn({seconds:remaining,signal:bounded})};
 }catch(error){
  const terminal=error?.reviewSession;
  signal?.throwIfAborted();
  if(unsubmittedKinds.includes(terminal?.kind)||terminal?.kind==='provider-error'&&!terminal.retryableTimeout)return {unavailable:terminal.kind};
  if(terminal?.kind==='elapsed'||terminal?.retryableTimeout===true&&terminal.kind==='provider-error')return {providerTimeout:terminal.kind,observedTimeout:terminal.retryableTimeout===true};
  throw error;
 }finally{clearInterval(timer);await writes;await save(file,snapshot(false));}
}

// QA may revise the instruction while preserving the additive change under review.
export function validateAdditiveDesignChange(changes,changeId){
 if(changes?.length!==1)throw Error('Expected exactly one change in changes; received '+(changes?.length??0));
 const change=changes[0],required={id:changeId,action:'upsert',critical:false,negative:false};
 const mismatches=Object.entries(required).filter(([key,value])=>change?.[key]!==value)
  .map(([key,value])=>'changes[0].'+key+' must be '+JSON.stringify(value)+'; received '+(JSON.stringify(change?.[key])??'missing'));
 if(mismatches.length)throw Error('Preserve the reviewed change fields: '+mismatches.join('. '));
}

/** Called only after five clean baseline attempts establish the reproduced struggle. */
export function createFailureDesign({config,item,knowledgeConfig=config},dependencies={}){
 const ops={session:reviewSession,qa:runProposalQa,qaAnswer:nativeQaAnswer,qaReview:nativeQaReview,qaPanelAudit:nativeQaPanelAudit,now:Date.now,...dependencies};
 const design=async input=>{
  assert(Number.isFinite(input.remainingSeconds)&&input.remainingSeconds>0,'Positive remaining experiment elapsed allowance required');
  assert.equal(item.objective,'correctness','Failure-first instruction design currently supports correctness only');
  const focused=input.mode==='focused-correctness';
  const exactFailures=input.baselineResults?.filter(row=>row.correct===false&&row.mechanismMatch).length;
  const fullPasses=input.baselineResults?.filter(row=>row.correct===true).length;
  assert(input.baselineResults?.length===5&&!input.baselineResults.some(row=>row.infrastructure)&&exactFailures>=(focused?1:3)&&(!focused||fullPasses<=3),focused?'Baseline evidence does not meet the writing gate':'Five clean baselines and three matching failures must precede writing');
  const base=item.base??config.skills.activeVersion;assert.equal(base,config.skills.activeVersion,'Instruction base changed');
  const manifest=await load(join(input.directory,'..','..','manifest.json')),identities=input.identities??manifest?.identities;assert(identities&&['model','runtime','memory'].every(key=>identities[key]!==undefined),'Frozen model, runtime and memory identities required');
  const baseline=await skillInput(join(config.root,'skills','versions',base)),identity=designIdentity({base,baseline:baseline.sha256,family:input.family,results:input.baselineResults,identities,item});
  assert.equal(input.baseline??manifest?.baseline,baseline.text,'Tested baseline guidance differs from the active skill');
  const perspectives=['robust decisions','diagnostic reasoning','implementation complexity'];
  const panel=input.qaPolicy==='panel';
  if(input.panelAuditPolicy!==undefined||input.panelAssertions!==undefined)assert(panel&&input.panelAuditPolicy==='assertions','Assertion audits require the panel policy');
  if(panel)assert(focused&&['principle-a','principle-b','principle-c'].includes(input.reviewCandidate),'Select one focused candidate for panel review');
  else assert(!input.reviewCandidate&&!input.frozenQa,'Frozen candidate review requires the panel policy');
  const laneIds=perspectives.map((_,index)=>'principle-'+String.fromCharCode(97+index));
  if(input.revisionCandidates)assert(!panel&&input.revisionFeedback&&Array.isArray(input.revisionCandidates)&&input.revisionCandidates.length&&input.revisionCandidates.every(id=>laneIds.includes(id)),'Invalid redesign lanes');
  const designInput={identity,...(input.revisionCandidates?{revisionCandidates:input.revisionCandidates,revisionFeedback:input.revisionFeedback}:{}),...(panel?{qaPolicy:'panel',reviewCandidate:input.reviewCandidate,...(input.panelAuditPolicy?{panelAuditPolicy:input.panelAuditPolicy,panelAssertionsIdentity:hash(input.panelAssertions??null)}:{}),frozenQaSha256:input.frozenQa?hash(input.frozenQa):null,
   panelAuditEvidenceSha256:hash(input.panelAuditEvidence??[]),revisionFeedback:input.revisionFeedback??null}:{})};
  const inputFile=join(input.directory,'design-input.json'),priorInput=await load(inputFile);
  if(priorInput)assert.deepEqual(priorInput,designInput,'Instruction evidence changed');else await save(inputFile,designInput);
  const registry=join(input.directory,'approved-design.json'),cached=await load(registry),dispositionFile=join(input.directory,'design-disposition.json'),disposition=await load(dispositionFile);
  async function restoreCandidate(candidate){
   if(panel){
    assert.equal(candidate.id,input.reviewCandidate,'Retained candidate differs from panel selection');
    const qa=await load(join(input.directory,candidate.id,'qa','qa.json'));
    verifyPanelApproval(qa,candidate.proposal);
    assert.equal(qa.approval,candidate.qa.approval,'Retained candidate panel approval changed');
   }
   return checkVersion(config,candidate);
  }
  if(cached){assert.equal(cached.identity,identity,'Instruction evidence changed');return Promise.all(cached.candidates.map(restoreCandidate));}
  if(disposition){assert.equal(disposition.identity,identity,'Instruction evidence changed');const {identity:_,...result}=disposition;return result;}
  const entries=await guidanceEntries(config.root,base);
  const {groups,development,proofRecords,evaluationContract}=designEvidence(item,input.family.development,input.baselineResults,focused?input.qualification:undefined);
  const refs=groups.flatMap(group=>[...group.occurrences,...(group.linkedDiagnostics??[])].map(row=>row.reference));
  const started=ops.now(),left=()=>Math.max(0,input.remainingSeconds-(ops.now()-started)/1000),sessionConfig={...knowledgeConfig,project_evidence_root:config.root};
  input.signal?.throwIfAborted();
  const primaryRecords=claimEvidence({packet:await frozenEvidencePacket(groups)});
  let panelRevisionSources=[];
  if(panel&&input.revisionFeedback){
   const feedback=JSON.parse(input.revisionFeedback);
   assert(Array.isArray(feedback.issues),'Panel revision must use the verified QA findings');
   panelRevisionSources=panelCorrectionSources(feedback.issues,input.panelAuditEvidence??[]);
  }
  const writerProofRecords=[...primaryRecords,...proofRecords,...panelRevisionSources];
  const pending=perspectives.flatMap((perspective,index)=>{
   const id='principle-'+String.fromCharCode(97+index);
   if(panel&&id!==input.reviewCandidate||input.revisionCandidates&&!input.revisionCandidates.includes(id))return [];
   return [runCandidate(id,perspective)];
  });
  async function runCandidate(id,perspective){
   const directory=join(input.directory,id),candidateFile=join(directory,'approved-candidate.json');
   const retained=await load(candidateFile);if(retained){assert.equal(retained.identity,identity,'Candidate evidence changed');return await restoreCandidate(retained.candidate);}
   await mkdir(directory,{recursive:true});const resultFile=join(directory,'writer-result.json');let written=await load(resultFile);
   const validate=(value,retrievals)=>{assertInstruction(value.text);assert(scopes.includes(value.scope),'Unknown instruction scope');const known=new Set(memoryEntries(retrievals).map(entry=>entry.ref));assert(value.memoryRefs?.length&&value.memoryRefs.every(ref=>known.has(ref)),'Supporting memory must use retrieved collection:chunk_id references. Allowed: '+JSON.stringify([...known]));return value;};
   if(input.frozenQa)assert(written,'Frozen panel replay requires its retained writer result');
   if(!written){
    const previous=await load(join(directory,'writer','termination.json'));
    if(unsubmittedKinds.includes(previous?.kind)||previous?.kind==='provider-error'&&!previous.retryableTimeout)return {id,unavailable:previous.kind};
    let outcome,pause=1;do{outcome=await writerCall(directory,Math.min(180,left()),input.signal,limits=>ops.session({...limits,directory:join(directory,'writer'),config:sessionConfig,knowledge:true,proofRequired:true,selectorOnly:true,resultSchema:schema,proofRecords:writerProofRecords,evidenceGroups:groups,
     prompt:{...(input.revisionFeedback?{revisionFeedback:input.revisionFeedback}:{}),task:'Derive one broadly applicable instruction from the reproduced failure and baseline outcomes, considering '+perspective+'. Search relevant successful memory. State a general principle and when it applies, leaving diagnosis, algorithms and testing choices to the actor. Do not prescribe the exact repair, include code, benchmark identifiers, case constants or claims of universal compiler causes. Use at most 150 words. Return text,scope,memoryRefs,claims. text contains the broad instruction. memoryRefs must use collection:chunk_id from actual KDB results; claims.path is text, with source selectors only.',development,baselineOutcomes:proofRecords,...(panelRevisionSources.length?{revisionSources:panelRevisionSources}:{}),entries},validate}),ops.now);
    if(outcome.unavailable)return {id,unavailable:outcome.unavailable};
    if(!outcome.providerTimeout)break;
    const spent=await load(join(directory,'writer-spend.json'));
    if(outcome.providerTimeout==='elapsed'||Math.min(left(),spent.seconds-spent.elapsedSeconds)<=pause)return {id,providerTimeout:outcome.providerTimeout,observedTimeout:outcome.observedTimeout===true};
    await delay(pause*1000,undefined,{signal:input.signal});pause=Math.min(60,pause*2);
   }while(left()>0);
   if(!outcome.written)return {id,providerTimeout:outcome.providerTimeout,observedTimeout:outcome.observedTimeout===true};
    written=outcome.written;validate(written.value,written.retrievals??[]);await save(resultFile,written);
   }
   else validate(written.value,written.retrievals??[]);
   await writerHealthy(written);
   const memoryEvidence=memoryEntries(written.retrievals??[]),changeId='failure-first-'+hash([identity,id]).slice(0,24);
   const proposal={action:'update',reason:'Instruction grounded in reproduced baseline failures',groupIds:item.groupIds,memoryRefs:written.value.memoryRefs,changes:[{action:'upsert',id:changeId,scope:written.value.scope,text:written.value.text,critical:false,negative:false,evidenceRefs:refs,response:'actor-change'}]};
   const validateProposal=value=>{validateAdditiveDesignChange(value.changes,changeId);validateReviewProposal(value,groups,item.preparation?.classifications??{});assertInstruction(value.changes[0].text);assert(value.memoryRefs.every(ref=>memoryEvidence.some(entry=>entry.ref===ref)),'QA cannot introduce unretrieved memory');return value;};
   let qa,pause=1;do{qa=await ops.qa({directory:join(directory,'qa'),config:sessionConfig,proposal:validateProposal(proposal),groups,entries,objective:'correctness',target:config.target,memoryEvidence,proposalDirectory:join(directory,'writer'),validateProposal,
    ...(panel?{qaPolicy:'panel',panelAuditPolicy:input.panelAuditPolicy,panelAssertions:input.panelAssertions,frozenQa:input.frozenQa,qaPanelConcurrency:input.qaPanelConcurrency,panelAuditEvidence:input.panelAuditEvidence,panelRevisionSources,retryMissingPanelVotes:input.retryMissingPanelVotes}:{}),allowWriterRevision:focused&&!input.revisionFeedback,evaluationContract,seconds:input.panelAuditPolicy==='assertions'?left():Math.min(panel?1020:600,left()),signal:AbortSignal.any([...(input.signal?[input.signal]:[]),AbortSignal.timeout(Math.max(1,Math.ceil(left()*1000)))])},{qaAnswer:async request=>{const response=await ops.qaAnswer(request);return response?.value??response;},qaReview:async request=>{const response=await ops.qaReview(request);return panel?response:response?.value??response;},qaPanelAudit:ops.qaPanelAudit});
   if(panel||qa.reasonKind!=='infrastructure'||!qa.retryable||!qa.error?.reviewSession?.retryableTimeout||Math.min(left(),qa.seconds-qa.elapsedSeconds)<=pause)break;
   await delay(pause*1000,undefined,{signal:input.signal});pause=Math.min(60,pause*2);
   }while(left()>0);
   if(qa.verdict!=='approved')throw Object.assign(Error(qa.reason??'Instruction review rejected'),{lane:id,code:qa.reasonKind==='infrastructure'?'FAILURE_DESIGN_INFRASTRUCTURE':'FAILURE_DESIGN_REJECTED',...(qa.revisionFeedback?.trim()?{revisionFeedback:panel?qa.revisionFeedback:id+': '+qa.revisionFeedback}:{} )});
   assert(qa.approval&&left()>0,'QA approval and remaining experiment time required');const reviewed=validateProposal(qa.proposal);
   if(panel)verifyPanelApproval(qa,reviewed);
   const updated=applyProposal(entries,reviewed.changes,{allowPruning:false,history:[]}),version=await writeVersion(config.root,updated,join(config.root,'skills','versions',base)),effective=await skillInput(join(config.root,'skills','versions',version));
   const candidate={id,instruction:reviewed.changes[0].text,scope:reviewed.changes[0].scope,version,text:effective.text,textSha256:textHash(effective.text),skillSha256:effective.sha256,base,baselineSha256:baseline.sha256,proposal:reviewed,qa:{verdict:qa.verdict,approval:qa.approval,...(panel?{policy:'panel'}:{})},evidence:written.value.claims.flatMap(claim=>claim.proofs)};
   await save(candidateFile,{identity,candidate});return candidate;
  }
  const settled=await Promise.allSettled(pending),failures=settled.filter(row=>row.status==='rejected');
  input.signal?.throwIfAborted();
  const unexpected=failures.find(row=>!['FAILURE_DESIGN_INFRASTRUCTURE','FAILURE_DESIGN_REJECTED'].includes(row.reason.code));
  if(unexpected)throw unexpected.reason;
  const writers=settled.flatMap(row=>row.status==='fulfilled'&&(row.value.providerTimeout||row.value.unavailable)?[{id:row.value.id,kind:row.value.providerTimeout??row.value.unavailable}]:[]);
  const candidates=settled.flatMap(row=>row.status==='fulfilled'&&!row.value.providerTimeout&&!row.value.unavailable?[row.value]:[]);
  if(candidates.length){
   const unique=[...new Map(candidates.map(candidate=>[candidate.instruction,candidate])).values()];
   await save(registry,{identity,candidates:unique});return unique;
  }
  const revisions=failures.filter(row=>row.reason.code==='FAILURE_DESIGN_REJECTED'&&row.reason.revisionFeedback);
  if(focused&&!input.revisionFeedback&&revisions.length){
   const result={status:'rejected',reason:revisions[0].reason.message,revisionFeedback:revisions.map(row=>row.reason.revisionFeedback).join('\n'),...(!panel?{revisionCandidates:revisions.map(row=>row.reason.lane)}:{}),...(writers.length?{writers}:{})};
   await save(dispositionFile,{identity,...result});return result;
  }
  const infra=failures.find(row=>row.reason.code==='FAILURE_DESIGN_INFRASTRUCTURE');
  if(infra&&panel)throw infra.reason;
  if(infra){const result={status:'infrastructure',infrastructure:true,correct:null,reason:infra.reason.message};await save(dispositionFile,{identity,...result});return result;}
  if(writers.length){
   const onlyTimeouts=settled.every(row=>row.status==='fulfilled'&&row.value.observedTimeout);
   const result={status:'infrastructure',infrastructure:true,correct:null,errorCode:onlyTimeouts?'FAILURE_DESIGN_PROVIDER_TIMEOUT':'FAILURE_DESIGN_UNAVAILABLE',reason:onlyTimeouts?'Instruction writer provider request timed out':'No writer produced a review-approved instruction',writers};
   await save(dispositionFile,{identity,...result});return result;
  }
  if(failures.length){const feedback=failures.map(row=>row.reason.revisionFeedback).filter(Boolean);const result={status:'rejected',reason:failures[0].reason.message,...(focused&&!input.revisionFeedback&&feedback.length?{revisionFeedback:feedback.join('\n')}: {})};await save(dispositionFile,{identity,...result});return result;}
  throw Error('Design completed without a lane disposition');
 };
 design.canResume=async op=>{
  const input=await load(join(op.directory,'design-input.json'));
  if(!input)return false;
  const ids=['principle-a','principle-b','principle-c'];
  if(input.qaPolicy==='panel'&&!ids.includes(input.reviewCandidate))return false;
  if(input.revisionCandidates&&(!Array.isArray(input.revisionCandidates)||!input.revisionCandidates.length||!input.revisionCandidates.every(id=>ids.includes(id))))return false;
  for(const id of input.qaPolicy==='panel'?[input.reviewCandidate]:(input.revisionCandidates??ids)){
   const lane=join(op.directory,id),spend=await load(join(lane,'writer-spend.json')),draft=await load(join(lane,'writer-result.json'));
   if(spend?spend.running!==false:!(input.frozenQaSha256&&draft))return false;
   if(!draft){
    const termination=await load(join(lane,'writer','termination.json'));
    if(!['elapsed','provider-error','cancelled','submitted',...unsubmittedKinds].includes(termination?.kind))return false;
   }
   for(const name of ['qa.json','qa-v2.json']){
    const qa=await load(join(lane,'qa',name));
    if(qa?.inFlight||Object.keys(qa?.inFlightOperations??{}).length)return false;
   }
  }
  return true;
 };
 return design;
}

function trials(rows,task){assert(rows?.length===5,'Every arm requires five trials');return rows.map(row=>{assert(!row.infrastructure,'Infrastructure cannot establish skill benefit');return validatePilotTrial(structuredClone(row),task);});}

function completeOperation(report,directory,id,kind){
 const operation=report.operations?.[id],path=join(directory,'operations',id);
 assert(operation?.status==='complete'&&operation.kind===kind&&operation.result!==undefined,'Missing complete '+kind+' operation: '+id);
 assert.equal(resolve(operation.directory),resolve(path),'Operation directory changed: '+id);
 return operation;
}

async function focusedPublicationEvidence(report,manifest,family,directory,{requireTransfer=true}={}){
 assert.equal(report.manifestSha256,hash(manifest),'Frozen focused pilot manifest or fixture changed');
 assert(Object.values(report.operations).every(operation=>operation.status==='complete'),'Focused pilot operations remain unresolved');
 const tasks=[family.development,...family.transfer];
 assert(new Set(tasks.map(task=>hash(task.files))).size===3&&new Set(family.transfer.map(task=>task.changeType)).size===2&&family.transfer.every(task=>task.changeType&&task.relation),'Two independent transfer fixtures required');
 const qualification=async task=>{
  const key=family.id+'--'+task.id,retained=report.qualifications[key];
  assert(task.controls?.calibration?.files&&retained?.qualified===true&&retained.referencePassed===true&&retained.negativeFailed===true&&retained.calibrationPassed===true,'Focused task controls must qualify: '+task.id);
  assert.equal(report.checkerHashes?.[key],textHash(await readFile(task.checker)),'Frozen checker changed: '+task.id);
  assert.deepEqual(retained,completeOperation(report,directory,'qualify--'+key,'qualify').result,'Qualification operation result changed');
  return retained;
 };
 const boundTrials=(rows,phase,task,arm)=>{
  const prefix=[phase,family.id,task.id,arm].join('--');
  return trials(rows,task).map((row,index)=>{
   const operation=completeOperation(report,directory,prefix+'--'+index,'trial');
   assert.deepEqual(row,validatePilotTrial(operation.result,task),'Trial operation result changed: '+prefix+'--'+index);
   return row;
  });
 };
 const developmentQualification=await qualification(family.development);
 const screen=boundTrials(report.screens[family.id],'screen',family.development,'baseline');
 assert.equal(focusedBaselineDecision(screen),'admitted','Development baseline did not qualify for focused writing');
 assert.deepEqual(trials(report.development.baseline,family.development),screen,'Focused development baseline must reuse the screened cohort');
 const development={baseline:screen};
 for(const candidate of report.candidates)development[candidate.id]=boundTrials(report.development[candidate.id],'development',family.development,candidate.id);
 const winner=pilotWinner(development,report.candidates);
 assert(winner&&hash(winner)===hash(report.winner),'Development winner changed');
 for(const task of requireTransfer?family.transfer:[]){
  await qualification(task);
  const rows=report.confirmation[task.id];
  assert(rows,'Missing focused transfer comparison');
  const baseline=boundTrials(rows.baseline,'transfer-screen',task,'baseline');
  assert.equal(focusedBaselineDecision(baseline),'admitted','Transfer baseline did not qualify for a candidate: '+task.id);
  const candidate=boundTrials(rows[winner.id],'confirmation',task,winner.id);
  assert(pilotWinner({baseline,[winner.id]:candidate},[winner]),'Both transfers must improve by two successes without control regression');
 }
 return {winner,developmentQualification};
}

async function focusedDesignOperation(report,family,directory,identity){
 const first=completeOperation(report,directory,'design--'+family.id,'design');
 let accepted=first;
 if(report.operations['redesign--'+family.id]){
  const feedback=first.result?.revisionFeedback;
  assert(first.result?.status==='rejected'&&typeof feedback==='string'&&feedback.trim(),'Redesign requires retained QA feedback');
  assert.deepEqual(await load(join(first.directory,'design-disposition.json')),{identity,...first.result},'Redesign QA disposition changed');
  accepted=completeOperation(report,directory,'redesign--'+family.id,'design');
 }
 assert.deepEqual(accepted.result,report.candidates,'Reviewed design operation result changed');
 return accepted.directory;
}

async function requireFocusedApproval({config,item,base,family,screen,qualification,designDirectory,identity,winner}){
 const directory=join(designDirectory,winner.id);
 const candidate=await load(join(directory,'approved-candidate.json'));
 assert.deepEqual(candidate,{identity,candidate:winner},'Reviewed candidate receipt changed');
 const qaDirectory=join(directory,'qa');
 const initial=await load(join(qaDirectory,'qa.json'));
 const qa=initial?.format===2?initial:await load(join(qaDirectory,'qa-v2.json'));
 assert(qa?.format===2&&qa.verdict==='approved'&&qa.proofProtocol==='claim-proofs-v1'&&!qa.retryable&&!qa.inFlight&&!Object.keys(qa.inFlightOperations??{}).length,'Terminal focused QA approval required');
 if(qa.qaPolicy==='panel'||winner.qa.policy==='panel')verifyPanelApproval(qa,winner.proposal);
 assert.deepEqual(qa.proposal,winner.proposal,'QA approved proposal changed');
 assert.equal(qa.rounds?.at(-1)?.review?.verdict,'approved','Final QA review must approve the candidate');
 assert.deepEqual(qa.rounds.at(-1).proposal,winner.proposal,'Final QA review approved a different proposal');
 const writer=await load(join(directory,'writer-result.json'));
 assert(writer?.value,'Retained writer evidence required for focused QA');
 const {groups,evaluationContract}=designEvidence(item,family.development,screen,qualification);
 const entries=await guidanceEntries(config.root,base),memoryEvidence=memoryEntries(writer.retrievals??[]);
 const expected=qaIdentity({groups,entries,objective:item.objective,target:config.target,memoryEvidence,evaluationContract},winner.proposal);
 assert.equal(qa.approval,expected,'Focused QA approval changed');
 assert.equal(winner.qa.approval,expected,'Winner differs from the focused QA approval');
}

/** Validate the same development and QA evidence used for activation. */
export async function verifyFocusedSearchableApproval({config,item,result,manifest,family,directory}){
 assert(manifest.mode==='focused-correctness'&&item.objective==='correctness','Focused correctness evidence required');
 const base=item.base??config.skills.activeVersion;
 const baseline=await skillInput(join(config.root,'skills','versions',base));
 assert.equal(manifest.baseline,baseline.text,'Tested baseline guidance changed');
 assert(result.candidates?.length>=1&&result.candidates.length<=3,'Reviewed candidates required');
 const {winner,developmentQualification}=await focusedPublicationEvidence(result,manifest,family,directory,{requireTransfer:false});
 await checkVersion(config,winner);
 assert.equal(winner.base,base);
 assert.equal(winner.baselineSha256,baseline.sha256);
 const screen=result.screens[family.id];
 const identity=designIdentity({base,baseline:baseline.sha256,family:{id:family.id,development:family.development,evidence:family.evidence??[]},results:screen,identities:manifest.identities,item});
 const designDirectory=await focusedDesignOperation(result,family,directory,identity);
 const registry=await load(join(designDirectory,'approved-design.json'));
 assert.equal(registry?.identity,identity,'Design evidence identity changed');
 assert.deepEqual(registry.candidates,result.candidates,'Reviewed candidate cohort changed');
 assert(registry.candidates.some(candidate=>hash(candidate)===hash(winner)),'Winner differs from its reviewed version');
 await requireFocusedApproval({config,item,base,family,screen,qualification:developmentQualification,designDirectory,identity,winner});
 return designDirectory;
}

/** Recheck the persisted evidence and exact reviewed version before the existing atomic publisher. */
export async function publishFailureWinner({config,item,result,directory,signal,recoverOnly=false},dependencies={}){
 const ops={readConfig,publish:publishSkillVersion,record:async request=>{
  const {recordStorageSkill}=await import('../controller/storage-knowledge.mjs');return recordStorageSkill(request.config,request.candidate,{serialized:true,baseVersion:request.base,promote:true,reason:'failure-first-transfer-improvement',signal:request.signal});
 },...dependencies};
 const saved=await load(join(directory,'pilot.json')),manifest=validatePilotManifest(await load(join(directory,'manifest.json')));assert(saved&&hash(saved)===hash(result),'Only the persisted complete pilot result can publish');
 const base=item.base??config.skills.activeVersion,baseline=await skillInput(join(config.root,'skills','versions',base));assert.equal(manifest.baseline,baseline.text,'Tested baseline guidance differs from the active skill');
 assert.equal(item.objective,'correctness');assert(result.status==='complete'&&result.promising===true&&result.winner,'Transfer-confirmed result required');
 const focused=manifest.mode==='focused-correctness';
 const family=manifest.families.find(row=>row.id===result.selectedFamily);assert(family,'Qualified family required');
 assert(result.candidates.length>=1&&result.candidates.length<=3&&new Set(result.candidates.map(candidate=>candidate.id)).size===result.candidates.length,'Reviewed candidates required');
 let winner,developmentQualification;
 if(focused){({winner,developmentQualification}=await focusedPublicationEvidence(result,manifest,family,directory));}
 else{
  const qualified=result.qualifications[family.id];assert(qualified?.qualified&&qualified.referencePassed&&qualified.negativeFailed,'Qualified family required');
  const screen=trials(result.screens[family.id],family.development);assert(screen.filter(row=>!row.correct&&row.mechanismMatch).length>=3,'Baseline struggle was not established');
  const development=Object.fromEntries([['baseline',trials(result.development.baseline,family.development)],...result.candidates.map(candidate=>[candidate.id,trials(result.development[candidate.id],family.development)])]);
  winner=pilotWinner(development,result.candidates);assert(winner&&hash(winner)===hash(result.winner),'Development winner changed');
  for(const task of family.transfer){const rows=result.confirmation[task.id];assert(rows&&pilotWinner({baseline:trials(rows.baseline,task),[winner.id]:trials(rows[winner.id],task)},[winner]),'Both transfers must improve by two successes without control regression');}
 }
 assert.equal(family.transfer.length,2);assert(winner.qa?.verdict==='approved'&&winner.qa.approval,'Reviewed candidate required');await checkVersion(config,winner);
 assert.equal(winner.base,base);assert.equal(baseline.sha256,winner.baselineSha256,'Tested baseline changed');
 const designId=designIdentity({base,baseline:winner.baselineSha256,family:{id:family.id,development:family.development,evidence:family.evidence??[]},results:result.screens[family.id],identities:manifest.identities,item});
 const designDirectory=focused?await focusedDesignOperation(result,family,directory,designId):result.operations['design--'+family.id]?.directory;
 assert(designDirectory,'Missing reviewed design operation');const registry=await load(join(designDirectory,'approved-design.json'));
 assert.equal(registry?.identity,designId,'Design evidence identity changed');
 assert.deepEqual(registry.candidates,result.candidates,'Reviewed candidate cohort changed');
 assert(registry.candidates.some(candidate=>hash(candidate)===hash(winner)),'Winner differs from its reviewed version');
 if(focused)await requireFocusedApproval({config,item,base,family,screen:result.screens[family.id],qualification:developmentQualification,designDirectory,identity:designId,winner});
 const publicationFile=join(directory,'publication.json'),identity=hash({study:result.identity,result:hash(result),base,version:winner.version,text:winner.textSha256,storage:Boolean(config.storage)}),prior=await load(publicationFile);if(prior)assert.equal(prior.identity,identity,'Publication evidence changed');
 const record=async()=>{if(config.storage)await ops.record({config,candidate:winner.version,base,signal});};
 const current=await ops.readConfig(config.root);if(current.skills.activeVersion!==base){if(current.skills.activeVersion===winner.version&&prior){if(!prior.recordComplete)await record();const recovered={...prior,published:true,memoryRecorded:Boolean(config.storage),recordComplete:true,recovered:true};await save(publicationFile,recovered);return recovered;}return {published:false,reason:'stale-base'};}
 if(recoverOnly)return {published:false,reason:'no-prior-publication'};
 signal?.throwIfAborted();await save(publicationFile,{identity,base,candidate:winner.version,published:false});const published=await ops.publish(config.root,{active:base,candidate:winner.version,signal});
 if(published)await record();const receipt={identity,base,candidate:winner.version,published:Boolean(published),memoryRecorded:Boolean(published&&config.storage),recordComplete:Boolean(published),reason:published?'transfer-improvement':'stale-base'};await save(publicationFile,receipt);return receipt;
}
