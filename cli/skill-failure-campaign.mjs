/** Campaign admission for the failure-first experiment; legacy comparisons stay frozen. */
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {existsSync} from 'node:fs';
import {DatabaseSync} from 'node:sqlite';
import {State} from '../controller/state.mjs';
import {failureRoutingRevision} from '../controller/skill-failure-triage.mjs';
import {SkillBacklog} from '../controller/skill-backlog.mjs';
import {enqueueLearningRequest} from '../controller/skill-review.mjs';
import {attentionRevision} from '../controller/failure-attention.mjs';
import {processIdentity,ownerStopped} from '../controller/lifecycle.mjs';
import {readConfig} from './instance.mjs';
import {skillInput} from './skill-microeval.mjs';
import {skillModelIdentity,skillConfig} from '../controller/skill-context.mjs';
import {runFailurePilot} from './skill-failure-pilot.mjs';
import {failureTiming} from './skill-failure-protocol.mjs';
import {searchableMemoryFingerprint} from './skill-searchable.mjs';

const hash=x=>createHash('sha256').update(JSON.stringify(x)).digest('hex');
const optional=async path=>{try{return JSON.parse(await readFile(path,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;return null;}};
export const failureFirstEnabled=config=>config.skills?.failureFirst?.enabled===true;
const open=config=>{const state=new State(join(config.root,'state.sqlite'));return {state,backlog:new SkillBacklog(state,{target:config.target})};};
const value=(db,key)=>table(db,'control')?JSON.parse(db.prepare('SELECT value FROM control WHERE key=?').get(key)?.value??'null'):null;
const table=(db,name)=>!!db.prepare("SELECT 1 FROM sqlite_master WHERE name=?").get(name);
const queuedFamily=backlog=>backlog.list({status:'queued',workflow:'failure-first-v1',objective:'correctness',limit:1})[0];
function requeueUnwitnessedFamilies(backlog){
 backlog.atomic(()=>{
  const queued=backlog.list({status:'queued',workflow:'failure-first-v1',objective:'correctness',limit:100});
  for(const item of queued){
   const classifications=item.preparation?.classifications;
   if(!classifications||item.preparation.failureRoutingRevision===failureRoutingRevision)continue;
   backlog.revise(item.id,{status:'deferred',reason:'Queued classification requires runtime-boundary review'});
   for(const id of item.groupIds)backlog.reclassify(id);
  }
 });
}


export function failureMemoryGeneration(config,state){
 const cutoff=table(state.db,'events')?state.db.prepare('SELECT max(sequence) n FROM events WHERE attempt IS NOT NULL').get().n:null;
 const generated=table(state.db,'generation_events')?state.db.prepare('SELECT max(sequence) n FROM generation_events').get().n:null;
 let generation=null;const path=join(config.root,'knowledge.sqlite');
 if(existsSync(path)){const db=new DatabaseSync(path,{readOnly:true});try{if(table(db,'active_generation'))generation=db.prepare('SELECT revision FROM active_generation').get()?.revision;}finally{db.close();}}
 const admissions=searchableMemoryFingerprint(state.db);
 return hash({base:config.skills.activeVersion,generation,cutoff,generated,attention:attentionRevision(state.db),...(admissions?{admissions}:{})});
}

export async function failureCampaignStatus(config){
 const {state,backlog}=open(config);
 try{
  const active=backlog.meta('failure-first-current'),run=active&&backlog.run(active),cp=run?.checkpoint;
  const pilot=cp&&await optional(join(cp.directory,'pilot','pilot.json'));
  const isolation=cp?.phase==='isolate'&&await optional(join(cp.directory,'family','isolation.json'));
  const operations=Object.values(pilot?.operations??{}),running=operations.filter(o=>o.status==='running');
  const screen=Object.values(pilot?.screens??{}).flat(),last=backlog.meta('failure-first-status');
  return {...backlog.status(),...last,workflow:'failure-first-v1',current:run?{id:run.id,itemId:run.item_id,phase:cp.phase,
   elapsedSeconds:run.elapsed,allocatedSeconds:run.seconds,directory:cp.directory}:null,
   activeWork:run?{kind:'failure-first',stage:running[0]?.kind??cp.phase,id:run.id}:null,
   operations:running.map(o=>({id:o.id,elapsedSeconds:o.elapsedSeconds,allocatedSeconds:o.allocatedSeconds})),
   ...(isolation?{isolation:{elapsedSeconds:isolation.elapsedSeconds,allocatedSeconds:failureTiming.isolationSeconds,timings:isolation.timings}}:{}),
   screen:{completed:screen.length,passed:screen.filter(r=>r.correct===true).length,matchingFailures:screen.filter(r=>r.correct===false&&r.mechanismMatch).length},
   resources:cp?.slots??value(state.db,'skill-background-resources')??[],pending:!!run,unresolved:!!cp?.unresolved,
   hasWork:!config.skills.tuningDisabled&&(!!run||backlog.status().dirtyFailures>0||!!queuedFamily(backlog))};
 }finally{state.close();}
}

export async function failureCampaignSettled(config,dependencies={}){
 const {state,backlog}=open(config);
 try{const id=backlog.meta('failure-first-current'),run=id&&backlog.run(id);
  if(run?.owner&&!await ownerStopped(run.owner))return false;
  if(!run?.checkpoint?.runtimeDirectory)return true;
  const result=await (dependencies.settled??(await import('./skill-failure-runtime.mjs')).failureRuntimeSettled)({config,directory:run.checkpoint.runtimeDirectory,signal:AbortSignal.timeout(90000)});
  return result.closed===true;
 }finally{state.close();}
}

/** One ranked family owns the complete two-hour ledger and all child operation handles. */
export async function runCampaignFailureStep(config,{slots,memory,remainingSeconds,signal},dependencies={}){
 config=skillConfig(config);if(!failureFirstEnabled(config)||!['a3','a5'].includes(config.target))throw Error('Failure-first workflow on A3 or A5 must be explicitly enabled');
 if(slots?.length!==4)throw Error('Failure-first startup requires four reserved healthy devices');
 const {state,backlog}=open(config),now=dependencies.now??Date.now;
 const invokedAt=now();
 const campaignLeft=()=>Math.max(0,remainingSeconds-(now()-invokedAt)/1000);
 const knowledgeConfig={...memory?.config,project_evidence_root:config.root};
 const ops={
  analyze:async args=>(await import('./skill-backlog-analysis.mjs')).analyzeBacklog(args,{classify:async input=>(await import('./skill-review-native.mjs')).nativeClassifyBatch({...input,config:knowledgeConfig})}),
  family:async args=>(await import('./skill-failure-families.mjs')).prepareFailureFamily(args,{session:async input=>(await import('./skill-review-native.mjs')).reviewSession({...input,config:knowledgeConfig})}),
  runtime:async args=>(await import('./skill-failure-runtime.mjs')).createFailureRuntime(args),
  design:async args=>(await import('./skill-failure-candidates.mjs')).createFailureDesign(args),
  admit:async args=>(await import('./skill-searchable.mjs')).admitSearchableSkill(args),
  publish:async args=>(await import('./skill-failure-candidates.mjs')).publishFailureWinner(args),
  pilot:runFailurePilot,model:skillModelIdentity,input:skillInput,...dependencies};
 let run,cp,timer,runtime,closed=true,save;
 try{
  let ingestion;const until=now()+1000;do{ingestion=backlog.ingest({limit:100});}while(ingestion.progress&&now()<until);
  const existing=backlog.list({status:'running',limit:1})[0];
  if(!existing)requeueUnwitnessedFamilies(backlog);
  if(!existing&&!queuedFamily(backlog)){
   await ops.analyze({config,backlog,directory:join(config.root,'skills','priority','failure-first-analysis',hash(backlog.meta('queue-epoch'))),
    classificationOnly:true,seconds:120,remainingSeconds:Math.min(120,campaignLeft()),limit:2,signal});
  }
  const item=existing??queuedFamily(backlog);
  if(!item)return {status:'idle',reason:'No eligible correctness family'};
  if(item.objective!=='correctness'||item.proposal?.action==='update')throw Error('Legacy proposal cannot enter failure-first screening');
  const retained=existing&&backlog.run(existing.id);
  if(retained?.checkpoint?.phase==='admit-searchable'){
   const directory=join(retained.checkpoint.directory,'pilot');
   const result=await optional(join(directory,'pilot.json'));
   const admissionInput={config,item:retained.snapshot,directory,result};
   const elapsed=()=>Math.max(retained.elapsed,(now()-retained.checkpoint.startedAt)/1000);
   const finishRecovery=(outcome)=>backlog.atomic(()=>{
    const spent=elapsed();
    const status=outcome.admission&&spent<retained.seconds?'accepted':'inconclusive';
    backlog.checkpoint(retained.id,{elapsed:spent,checkpoint:retained.checkpoint});
    backlog.finish(retained.id,{status,result:outcome});
    backlog.meta('failure-first-current',null);
    backlog.meta('failure-first-status',{status,reason:outcome.reason,at:now(),lastDirectory:retained.checkpoint.directory});
    return {status,...outcome};
   });
   const recovered=await ops.admit({...admissionInput,signal,recoverOnly:true});
   if(recovered?.admitted)return finishRecovery({reason:'searchable-admission-recovered',searchable:true,admission:recovered});
   if(recovered?.reason!=='no-prior-admission')throw Error(recovered?.reason??'Searchable skill admission recovery refused');
   const remaining=Math.min(retained.seconds-elapsed(),campaignLeft());
   if(remaining<=0)return finishRecovery({reason:'Experiment elapsed allowance exhausted before searchable admission'});
   const deadline=AbortSignal.any([...(signal?[signal]:[]),AbortSignal.timeout(Math.max(1,Math.floor(remaining*1000)))]);
   deadline.throwIfAborted();
   const admission=await ops.admit({...admissionInput,signal:deadline});
   if(!admission?.admitted)throw Error(admission?.reason??'Searchable skill admission refused');
   return finishRecovery({reason:'searchable-development-improvement',searchable:true,result,admission});
  }
  if(retained?.checkpoint?.phase==='publish'){
   const saved=await optional(join(retained.checkpoint.directory,'pilot','pilot.json'));
   const elapsedAtResume=Math.max(retained.elapsed,(now()-retained.checkpoint.startedAt)/1000);
   const remaining=Math.min(retained.seconds-elapsedAtResume,campaignLeft());
   const alreadyChanged=(await readConfig(config.root)).skills.activeVersion!==retained.base;
   if(!alreadyChanged&&remaining<=0)throw Error('Experiment elapsed allowance exhausted before publication recovery');
   const deadline=alreadyChanged?signal:AbortSignal.any([...(signal?[signal]:[]),AbortSignal.timeout(Math.max(1,Math.floor(remaining*1000)))]);
   const publication=await ops.publish({config,item:retained.snapshot,result:saved,directory:join(retained.checkpoint.directory,'pilot'),signal:deadline,recoverOnly:alreadyChanged});
   if(!publication?.published)throw Error(publication?.reason??'Publication recovery refused');
   const elapsed=Math.max(retained.elapsed,(now()-retained.checkpoint.startedAt)/1000),status=elapsed<retained.seconds?'accepted':'inconclusive';
   backlog.checkpoint(retained.id,{elapsed,checkpoint:retained.checkpoint});backlog.finish(retained.id,{status,result:{reason:'publication-recovered',published:true,publication}});backlog.meta('failure-first-current',null);
   return {status,published:true,reason:'publication-recovered'};
  }
  if(item.base!==config.skills.activeVersion){
   if(existing)throw Error('Frozen experiment baseline changed; preserve its original run');
   backlog.revise(item.id,{base:config.skills.activeVersion,status:'queued'});return {status:'reclassified',reason:'Active skill changed before screening'};
  }
  const owner=await processIdentity();
  if(campaignLeft()<=0)throw Error('Campaign elapsed allowance exhausted');
  const claimed=backlog.claim({base:config.skills.activeVersion,itemId:item.id,seconds:Math.min(failureTiming.experimentSeconds,campaignLeft()),workflow:'failure-first-v1',owner});
  if(!claimed)return {status:'idle',reason:'Family already completed'};
  run=claimed.run;cp=run.checkpoint??{workflow:'failure-first-v1',startedAt:now(),directory:join(config.root,'skills','priority','failure-first',run.id),slots};
  if(cp.workflow!=='failure-first-v1'||hash(cp.slots)!==hash(slots))throw Error('Retained experiment requires its exact original resource bindings');
  const resumedAt=now(),retainedElapsed=Math.max(run.elapsed,(resumedAt-cp.startedAt)/1000);
  const elapsed=()=>retainedElapsed+(now()-resumedAt)/1000;
  const remaining=()=>Math.max(0,Math.min(run.seconds-elapsed(),campaignLeft()));
  save=()=>{backlog.checkpoint(run.id,{elapsed:elapsed(),checkpoint:cp});};
  backlog.meta('failure-first-current',run.id);await mkdir(cp.directory,{recursive:true});
  const phase=name=>{cp.phase=name;save();backlog.meta('failure-first-status',{status:'running',phase:name,at:now()});};
  const finish=(status,result)=>backlog.atomic(()=>{
   save();
   backlog.finish(run.id,{status,result});
   backlog.meta('failure-first-current',null);
   backlog.meta('failure-first-status',{status,reason:result.reason,...(result.stage?{stage:result.stage}:{}),at:now(),lastDirectory:cp.directory,
    ...(result.disposition?{disposition:result.disposition}:{}),...(result.learningRequest?{learningRequest:result.learningRequest}:{})});
   return {status,...result};
  });
  const failure=new AbortController();save();timer=setInterval(()=>{try{save();}catch(error){failure.abort(error);}},1000);
  if(remaining()<=0)throw Error('Experiment elapsed allowance exhausted');
  const active=AbortSignal.any([failure.signal,...(signal?[signal]:[]),AbortSignal.timeout(Math.max(1,Math.ceil(remaining()*1000)))]),cleanup=AbortSignal.timeout(Math.max(1,Math.ceil((remaining()+90)*1000)));
  phase('isolate');
  const priorFamily=await optional(join(cp.directory,'family-ready.json'));
  const openRuntime=async()=>{
   if(runtime)return runtime;
   cp.runtimeDirectory=join(cp.directory,'runtime');save();
   runtime=await ops.runtime({config,directory:cp.runtimeDirectory,publicDirectory:join(cp.directory,'public'),slots:cp.slots,knowledge:memory.knowledge,hiddenDirectories:[memory.config.root],signal:active,stopSignal:cleanup});
   return runtime;
  };
  const isolation=new AbortController();
  let referenceError,identity;
  const familyDirectory=join(cp.directory,'family');
  const checkReferences=async({family,seconds,signal})=>{
   if(referenceError)throw referenceError;
   const started=performance.now();
   try{
    const adapter=await openRuntime();
    identity??=await adapter.identify({seconds:Math.min(120,seconds),signal});
    return await adapter.checkReferences({family,directory:join(cp.directory,'submission-checks',randomUUID()),
     cacheScope:familyDirectory,seconds:Math.min(remaining(),seconds-(performance.now()-started)/1000),signal});
   }catch(error){
    if(error.code!=='FAILURE_REFERENCE_REJECTED'){
     referenceError=error;
     isolation.abort(error);
    }
    throw error;
   }
  };
  let prepared;
  try{
   prepared=await ops.family({config,backlog,item:run.snapshot,directory:familyDirectory,seconds:Math.min(failureTiming.isolationSeconds,remaining()),signal:AbortSignal.any([active,isolation.signal]),checkReferences});
  }catch(error){
   if(referenceError)throw referenceError;
   active.throwIfAborted();
   const disposition=error.reviewSession?.kind;
   if(config.target!=='a5'||!['no-submit','rejected-submissions'].includes(disposition))throw error;
   prepared={status:'unsubmitted',disposition:{blocker:disposition,route:'hold',verification:null,replay:'unverified',
    checks:[],passedChecks:[],nextQuestion:null},reason:disposition==='no-submit'
    ?'Family author completed without submitting a validated family'
    :error.message};
  }
  if(referenceError&&(prepared.status!=='expired'||referenceError.failureStage!==prepared.stage))throw referenceError;
  if(prepared.status!=='ready'){
   if(runtime){
    closed=await runtime.close();runtime=null;
    if(!closed)throw Error('Submission reference execution remains unresolved');
   }
   if(prepared.status==='expired')return finish('inconclusive',{reason:'stage-elapsed',stage:prepared.stage,detail:prepared.reason,timings:prepared.timings});
   const disposition=prepared.disposition;
   if(disposition?.route==='phase1'){
    return backlog.atomic(()=>{
     const {verification,nextQuestion}=disposition;
     const learningRequest=enqueueLearningRequest(state,{workflow:'failure-first-v1',target:config.target,workKind:'phase1',
      learningTarget:'device',backlogId:run.item_id,priority:run.snapshot.priority,groupIds:run.snapshot.groupIds,
      question:nextQuestion,hypothesis:prepared.reason,verification,
      requiredEvidence:'Use the archived workload and verification '+verification.reference+'. Define an explicit PyTorch reference for each affected stage or output. Document the source of each expected behavior. Use small cases that distinguish the plausible interpretations and show the stage outputs and verifier result.'});
     return finish('learning',{reason:'phase1-learning-requested',detail:prepared.reason,disposition,learningRequest});
    });
   }
   return finish('inconclusive',{reason:prepared.status==='unsubmitted'?'family-unsubmitted':'needs-evidence',detail:prepared.reason,timings:prepared.timings,...(disposition?{disposition}:{})});
  }
  if(priorFamily&&hash(priorFamily.family)!==hash(prepared.family))throw Error('Frozen failure family changed');
  if(!priorFamily)await writeFile(join(cp.directory,'family-ready.json'),JSON.stringify(prepared));
  await openRuntime();
  phase('qualify');
  const modelIdentities=async current=>({actor:await ops.model(current.actor),review:await ops.model(current.skill_writer_actor??current.actor)});
  const memoryIdentity={snapshot:memory.knowledge.identity,attentionRevision:memory.knowledge.attentionRevision??null};
  let manifest=await optional(join(cp.directory,'experiment.json'));
  if(!manifest){const baseline=await ops.input(join(config.root,'skills','versions',run.base));identity??=await runtime.identify({seconds:Math.min(120,remaining()),signal:active});
   const focused=[prepared.family.development,...prepared.family.transfer].every(task=>task.controls?.calibration?.files);
   manifest={...(focused?{mode:'focused-correctness'}:{}),baseline:baseline.text,identities:{model:await modelIdentities(config),runtime:identity,memory:memoryIdentity,skills:baseline.sha256,source:config.skills.failureFirst.sourceRevision,dependencies:hash(await readFile(new URL('../npm-shrinkwrap.json',import.meta.url),'utf8'))},families:[prepared.family]};
   cp.initialElapsedSeconds=elapsed();await writeFile(join(cp.directory,'experiment.json'),JSON.stringify(manifest));save();}
  const currentConfig=skillConfig(await readConfig(config.root));
  if(currentConfig.skills.activeVersion!==run.base||hash(manifest.identities.memory)!==hash(memoryIdentity)||hash(manifest.identities.model)!==hash(await modelIdentities(currentConfig))||manifest.identities.skills!==(await ops.input(join(config.root,'skills','versions',run.base))).sha256||manifest.identities.source!==config.skills.failureFirst.sourceRevision||manifest.identities.dependencies!==hash(await readFile(new URL('../npm-shrinkwrap.json',import.meta.url),'utf8')))throw Error('Frozen experiment baseline, model, machinery or memory changed');
  const design=await ops.design({config,item:run.snapshot,knowledgeConfig});phase('evaluate');
  const result=await ops.pilot({directory:join(cp.directory,'pilot'),manifest,seconds:run.seconds,trialSeconds:failureTiming.actorSeconds,verificationSeconds:failureTiming.verificationSeconds,concurrency:4,initialElapsedSeconds:cp.initialElapsedSeconds,signal:active},
   {qualify:runtime.qualify,trial:runtime.trial,settle:runtime.settle,design});
  closed=await runtime.close();runtime=null;if(!closed)throw Error('Experiment execution remains unresolved');
  if(result.status!=='complete')throw Error(result.reason??'Incomplete experiment');
  if(!result.promising){
   if(config.target!=='a5'||manifest.mode!=='focused-correctness'||!result.winner)
    return finish('rejected',{reason:result.reason,result});
   phase('admit-searchable');
   const admission=await ops.admit({config,item:run.snapshot,directory:join(cp.directory,'pilot'),result,signal:active});
   if(!admission?.admitted)throw Error(admission?.reason??'Searchable skill admission refused');
   return finish(remaining()>0?'accepted':'inconclusive',{reason:'searchable-development-improvement',searchable:true,result,admission});
  }
  if(config.target!=='a5'&&manifest.mode==='focused-correctness')
   return finish('inconclusive',{reason:'Focused correctness confirmation retained for review',result});
  phase('publish');const publication=await ops.publish({config,item:run.snapshot,result,directory:join(cp.directory,'pilot'),signal:active});
  if(!publication?.published)throw Error(publication?.reason??'Validated skill publication refused');
  return finish('accepted',{reason:'protected-transfer-improvement',result,publication});
 }catch(error){
  if(cp&&save){cp.unresolved=!closed;save();}
  throw error;
 }finally{
  clearInterval(timer);if(runtime){closed=await runtime.close().catch(()=>false);cp.unresolved=!closed;save();}
  state.close();
 }
}
