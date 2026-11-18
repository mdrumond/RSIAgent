/** Evidence-reviewed instruction candidates; only the tested immutable version can publish. */
import assert from 'node:assert/strict';
import {mkdir,readFile,writeFile,rename} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {Type} from 'typebox';
import {guidanceEntries} from '../controller/skill-review.mjs';
import {stages} from '../controller/workflow.mjs';
import {applyProposal,writeVersion} from '../controller/skill-evolution.mjs';
import {skillInput} from './skill-microeval.mjs';
import {reviewSession,nativeQaAnswer,nativeQaReview} from './skill-review-native.mjs';
import {runProposalQa} from './skill-review-qa.mjs';
import {validateReviewProposal} from './skill-review.mjs';
import {publishSkillVersion} from './skills.mjs';
import {readConfig} from './instance.mjs';
import {pilotWinner,validatePilotTrial,validatePilotManifest} from './skill-failure-pilot.mjs';

const hash=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const textHash=value=>createHash('sha256').update(value).digest('hex');
const load=async file=>{try{return JSON.parse(await readFile(file,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;return null;}};
const save=async(file,value)=>{await mkdir(join(file,'..'),{recursive:true});const temporary=file+'.'+randomUUID();await writeFile(temporary,JSON.stringify(value,null,2));await rename(temporary,file);};
const scopes=['global-kdb',...stages];
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
async function checkVersion(config,candidate){
 const effective=await skillInput(join(config.root,'skills','versions',candidate.version));
 assert.equal(effective.text,candidate.text,'Candidate guidance changed after review');assert.equal(effective.sha256,candidate.skillSha256,'Candidate source changed after review');
 assert.equal(textHash(candidate.text),candidate.textSha256,'Candidate text identity mismatch');return candidate;
}
async function writerCall(directory,seconds,signal,fn,now){
 signal?.throwIfAborted();
 const file=join(directory,'writer-spend.json'),prior=await load(file)??{seconds:180,elapsedSeconds:0};
 assert(!prior.running,'Interrupted writer elapsed custody is unknown');const remaining=Math.min(seconds,prior.seconds-prior.elapsedSeconds);assert(remaining>0,'Writer elapsed allowance exhausted');
 const started=now();let writes=Promise.resolve();const snapshot=running=>({...prior,elapsedSeconds:prior.elapsedSeconds+(now()-started)/1000,running});
 await save(file,snapshot(true));const timer=setInterval(()=>{writes=writes.then(()=>save(file,snapshot(true)));},1000);
 try{const bounded=AbortSignal.any([...(signal?[signal]:[]),AbortSignal.timeout(Math.max(1,Math.ceil(remaining*1000)))]);bounded.throwIfAborted();return await fn({seconds:remaining,signal:bounded});}
 finally{clearInterval(timer);await writes;await save(file,snapshot(false));}
}

/** Called only after five clean baseline attempts establish the reproduced struggle. */
export function createFailureDesign({config,item,knowledgeConfig=config},dependencies={}){
 const ops={session:reviewSession,qa:runProposalQa,qaAnswer:nativeQaAnswer,qaReview:nativeQaReview,now:Date.now,...dependencies};
 return async input=>{
  assert(Number.isFinite(input.remainingSeconds)&&input.remainingSeconds>0,'Positive remaining experiment elapsed allowance required');
  assert.equal(item.objective,'correctness','Failure-first instruction design currently supports correctness only');
  assert(input.baselineResults?.length===5&&!input.baselineResults.some(row=>row.infrastructure)&&input.baselineResults.filter(row=>row.correct===false&&row.mechanismMatch).length>=3,'Five clean baselines and three matching failures must precede writing');
  const base=item.base??config.skills.activeVersion;assert.equal(base,config.skills.activeVersion,'Instruction base changed');
  const identities=input.identities??(await load(join(input.directory,'..','..','manifest.json')))?.identities;assert(identities&&['model','runtime','memory'].every(key=>identities[key]!==undefined),'Frozen model, runtime and memory identities required');
  const baseline=await skillInput(join(config.root,'skills','versions',base)),identity=designIdentity({base,baseline:baseline.sha256,family:input.family,results:input.baselineResults,identities,item});
  const inputFile=join(input.directory,'design-input.json'),priorInput=await load(inputFile);if(priorInput)assert.equal(priorInput.identity,identity,'Instruction evidence changed');else await save(inputFile,{identity});
  const registry=join(input.directory,'approved-design.json'),cached=await load(registry),dispositionFile=join(input.directory,'design-disposition.json'),disposition=await load(dispositionFile);
  if(cached){assert.equal(cached.identity,identity,'Instruction evidence changed');return Promise.all(cached.candidates.map(candidate=>checkVersion(config,candidate)));}
  if(disposition){assert.equal(disposition.identity,identity,'Instruction evidence changed');return {status:disposition.status,reason:disposition.reason};}
  const entries=await guidanceEntries(config.root,base),groups=structuredClone(reviewGroups(item));assert(groups.some(group=>group.occurrences?.length),'Original failure evidence is required');
  // QA must cite the new trusted baseline observations, not infer them from historical project outcomes.
  groups[0].linkedDiagnostics=[...(groups[0].linkedDiagnostics??[]),...input.baselineResults.map((result,index)=>({reference:'failure-first:baseline:'+index,type:'isolated-baseline-verification',data:{correct:result.correct,mechanismMatch:result.mechanismMatch,sourceSha256:result.sourceSha256,failure:result.failure},artifacts:result.artifacts??[]}))];
  const refs=groups.flatMap(group=>[...group.occurrences,...(group.linkedDiagnostics??[])].map(row=>row.reference));
  const started=ops.now(),left=()=>Math.max(0,input.remainingSeconds-(ops.now()-started)/1000),sessionConfig={...knowledgeConfig,project_evidence_root:config.root};
  // The development descriptor is deliberately projected: controls and private checker paths are not writer input.
  const development=Object.fromEntries(['id','prompt','files','caseIds'].filter(key=>input.family.development[key]!==undefined).map(key=>[key,input.family.development[key]]));
  const proofRecords=input.baselineResults.map((result,index)=>({reference:'baseline-'+index,text:JSON.stringify({correct:result.correct,mechanismMatch:result.mechanismMatch,sourceSha256:result.sourceSha256,failure:result.failure})}));
  const stop=new AbortController();let infrastructureCause;input={...input,signal:AbortSignal.any([stop.signal,...(input.signal?[input.signal]:[])])};
  const pending=['robust decisions','diagnostic reasoning','implementation complexity'].map(async(perspective,index)=>{try{
   const id='principle-'+String.fromCharCode(97+index),directory=join(input.directory,id),candidateFile=join(directory,'approved-candidate.json');
   const retained=await load(candidateFile);if(retained){assert.equal(retained.identity,identity,'Candidate evidence changed');return checkVersion(config,retained.candidate);}
   await mkdir(directory,{recursive:true});const resultFile=join(directory,'writer-result.json');let written=await load(resultFile);
   const validate=(value,retrievals)=>{assertInstruction(value.text);assert(scopes.includes(value.scope),'Unknown instruction scope');const known=new Set(memoryEntries(retrievals).map(entry=>entry.ref));assert(value.memoryRefs?.length&&value.memoryRefs.every(ref=>known.has(ref)),'Supporting memory must use retrieved collection:chunk_id references. Allowed: '+JSON.stringify([...known]));return value;};
   if(!written){written=await writerCall(directory,Math.min(180,left()),input.signal,limits=>ops.session({...limits,directory:join(directory,'writer'),config:sessionConfig,knowledge:true,proofRequired:true,selectorOnly:true,resultSchema:schema,proofRecords,evidenceGroups:groups,
     prompt:{task:'Derive one broadly applicable instruction from the reproduced failure and baseline outcomes, considering '+perspective+'. Search relevant successful memory. State a general principle and when it applies, leaving diagnosis, algorithms and testing choices to the actor. Do not prescribe the exact repair, include code, benchmark identifiers, case constants or claims of universal compiler causes. Use at most 150 words. Return text,scope,memoryRefs,claims. text contains the broad instruction. memoryRefs must use collection:chunk_id from actual KDB results; claims.path is text, with source selectors only.',development,baselineOutcomes:proofRecords,entries},validate}),ops.now);validate(written.value,written.retrievals??[]);await save(resultFile,written);}
   else validate(written.value,written.retrievals??[]);
   await writerHealthy(written);
   const memoryEvidence=memoryEntries(written.retrievals??[]),changeId='failure-first-'+hash([identity,id]).slice(0,24);
   const proposal={action:'update',reason:'Instruction grounded in reproduced baseline failures',groupIds:item.groupIds,memoryRefs:written.value.memoryRefs,changes:[{action:'upsert',id:changeId,scope:written.value.scope,text:written.value.text,critical:false,negative:false,evidenceRefs:refs,response:'actor-change'}]};
   const validateProposal=value=>{validateReviewProposal(value,groups,item.preparation?.classifications??{});assert(value.changes?.length===1&&value.changes[0].id===changeId&&value.changes[0].action==='upsert'&&value.changes[0].critical===false&&value.changes[0].negative===false,'Only one additive instruction may be reviewed');assertInstruction(value.changes[0].text);assert(value.memoryRefs.every(ref=>memoryEvidence.some(entry=>entry.ref===ref)),'QA cannot introduce unretrieved memory');return value;};
   const qa=await ops.qa({directory:join(directory,'qa'),config:sessionConfig,proposal:validateProposal(proposal),groups,entries,objective:'correctness',target:config.target,memoryEvidence,proposalDirectory:join(directory,'writer'),validateProposal,
    evaluationContract:{workflow:'failure-first-v1',development,baselineOutcomes:proofRecords},seconds:Math.min(300,left()),signal:AbortSignal.any([...(input.signal?[input.signal]:[]),AbortSignal.timeout(Math.max(1,Math.ceil(left()*1000)))])},{qaAnswer:ops.qaAnswer,qaReview:ops.qaReview});
   if(qa.verdict!=='approved')throw Object.assign(Error(qa.reason??'Instruction review rejected'),{code:qa.reasonKind==='infrastructure'?'FAILURE_DESIGN_INFRASTRUCTURE':'FAILURE_DESIGN_REJECTED'});
   assert(qa.approval&&left()>0,'QA approval and remaining experiment time required');const reviewed=validateProposal(qa.proposal),updated=applyProposal(entries,reviewed.changes,{allowPruning:false,history:[]}),version=await writeVersion(config.root,updated,join(config.root,'skills','versions',base)),effective=await skillInput(join(config.root,'skills','versions',version));
   const candidate={id,instruction:reviewed.changes[0].text,scope:reviewed.changes[0].scope,version,text:effective.text,textSha256:textHash(effective.text),skillSha256:effective.sha256,base,baselineSha256:baseline.sha256,proposal:reviewed,qa:{verdict:qa.verdict,approval:qa.approval},evidence:written.value.claims.flatMap(claim=>claim.proofs)};
   await save(candidateFile,{identity,candidate});return candidate;
  }catch(error){if(error.code!=='FAILURE_DESIGN_REJECTED'){infrastructureCause??=error;stop.abort(infrastructureCause);}throw error;}});
  const settled=await Promise.allSettled(pending),failures=settled.filter(row=>row.status==='rejected');if(infrastructureCause)throw infrastructureCause;
  if(failures.length){const result={status:'rejected',reason:failures[0].reason.message};await save(dispositionFile,{identity,...result});return result;}const candidates=settled.map(row=>row.value);
  assert(new Set(candidates.map(candidate=>candidate.instruction)).size===3,'Three distinct instruction candidates required');await save(registry,{identity,candidates});return candidates;
 };
}

function trials(rows,task){assert(rows?.length===5,'Every arm requires five trials');return rows.map(row=>{assert(!row.infrastructure,'Infrastructure cannot establish skill benefit');return validatePilotTrial(structuredClone(row),task);});}
/** Recheck the persisted evidence and exact reviewed version before the existing atomic publisher. */
export async function publishFailureWinner({config,item,result,directory,signal},dependencies={}){
 const ops={readConfig,publish:publishSkillVersion,record:async request=>{
  const {recordStorageSkill}=await import('../controller/storage-knowledge.mjs');return recordStorageSkill(request.config,request.candidate,{serialized:true,baseVersion:request.base,promote:true,reason:'failure-first-transfer-improvement',signal:request.signal});
 },...dependencies};
 const saved=await load(join(directory,'pilot.json')),manifest=validatePilotManifest(await load(join(directory,'manifest.json')));assert(saved&&hash(saved)===hash(result),'Only the persisted complete pilot result can publish');
 assert.equal(item.objective,'correctness');assert(result.status==='complete'&&result.promising===true&&result.winner,'Transfer-confirmed result required');
 const family=manifest.families.find(row=>row.id===result.selectedFamily),qualified=result.qualifications[family?.id];assert(family&&qualified?.qualified&&qualified.referencePassed&&qualified.negativeFailed,'Qualified family required');
 const screen=trials(result.screens[family.id],family.development);assert(screen.filter(row=>!row.correct&&row.mechanismMatch).length>=3,'Baseline struggle was not established');
 assert(result.candidates.length===3&&new Set(result.candidates.map(candidate=>candidate.id)).size===3,'Three reviewed candidates required');
 const development=Object.fromEntries([['baseline',trials(result.development.baseline,family.development)],...result.candidates.map(candidate=>[candidate.id,trials(result.development[candidate.id],family.development)])]);
 const winner=pilotWinner(development,result.candidates);assert(winner&&hash(winner)===hash(result.winner),'Development winner changed');
 for(const task of family.transfer){const rows=result.confirmation[task.id];assert(rows&&pilotWinner({baseline:trials(rows.baseline,task),[winner.id]:trials(rows[winner.id],task)},[winner]),'Both transfers must improve by two successes without control regression');}
 assert.equal(family.transfer.length,2);assert(winner.qa?.verdict==='approved'&&winner.qa.approval,'Reviewed candidate required');await checkVersion(config,winner);
 const base=item.base??config.skills.activeVersion;assert.equal(winner.base,base);assert.equal((await skillInput(join(config.root,'skills','versions',base))).sha256,winner.baselineSha256,'Tested baseline changed');
 const designDirectory=result.operations['design--'+family.id]?.directory;assert(designDirectory,'Missing reviewed design operation');const registry=await load(join(designDirectory,'approved-design.json'));
 assert.equal(registry?.identity,designIdentity({base,baseline:winner.baselineSha256,family:{id:family.id,development:family.development,evidence:family.evidence??[]},results:result.screens[family.id],identities:manifest.identities,item}),'Design evidence identity changed');
 assert(registry.candidates.some(candidate=>hash(candidate)===hash(winner)),'Winner differs from its reviewed version');
 const publicationFile=join(directory,'publication.json'),identity=hash({study:result.identity,result:hash(result),base,version:winner.version,text:winner.textSha256,storage:Boolean(config.storage)}),prior=await load(publicationFile);if(prior)assert.equal(prior.identity,identity,'Publication evidence changed');
 const record=async()=>{if(config.storage)await ops.record({config,candidate:winner.version,base,signal});};
 const current=await ops.readConfig(config.root);if(current.skills.activeVersion!==base){if(current.skills.activeVersion===winner.version&&prior){if(!prior.recordComplete)await record();const recovered={...prior,published:true,memoryRecorded:Boolean(config.storage),recordComplete:true,recovered:true};await save(publicationFile,recovered);return recovered;}return {published:false,reason:'stale-base'};}
 signal?.throwIfAborted();await save(publicationFile,{identity,base,candidate:winner.version,published:false});const published=await ops.publish(config.root,{active:base,candidate:winner.version,signal});
 if(published)await record();const receipt={identity,base,candidate:winner.version,published:Boolean(published),memoryRecorded:Boolean(published&&config.storage),recordComplete:Boolean(published),reason:published?'transfer-improvement':'stale-base'};await save(publicationFile,receipt);return receipt;
}
