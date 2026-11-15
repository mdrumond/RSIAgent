/** Campaign-local failure backlog; one retained hypothesis owns the background slot. */
import {createHash} from 'node:crypto';
import {mkdir,readFile} from 'node:fs/promises';
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {State} from '../controller/state.mjs';
import {SkillBacklog} from '../controller/skill-backlog.mjs';
import {ownerStopped,processIdentity} from '../controller/lifecycle.mjs';
import {readConfig} from './instance.mjs';
import {publishSkillVersion} from './skills.mjs';
import {analyzeBacklog,prioritizeBacklog,prepareHypothesis} from './skill-backlog-analysis.mjs';
import {evaluateHypothesis,hypothesisEvaluationSettled,freezeHypothesisProtocol} from './skill-hypothesis-evaluation.mjs';
import {runMicrolearning,microlearningSettled,pendingMicrolearning} from './skill-microlearning.mjs';
import {resolveHypothesisCases,qualifiedSuites} from './skill-priority-cases.mjs';
import {refreshCohort,cohortStatus,hasCohortCandidates,claimCohort,attachRetainedCohort,admissibleContinuations,retireStaleContinuations} from './skill-cohort.mjs';
const caseReadiness=async args=>(await import('./skill-priority-cases.mjs')).hypothesisCaseReadiness?.(args)??{ready:false};

const executionDirectory=checkpoint=>checkpoint?.evaluationDirectory??checkpoint?.qualificationDirectory;
async function evaluationReport(checkpoint){
 if(!checkpoint?.evaluationDirectory)return null;
 try{return JSON.parse(await readFile(join(checkpoint.evaluationDirectory,'micro','results.json'),'utf8'));}
 catch(error){if(error.code!=='ENOENT')throw error;return null;}
}
const interruptedReport=report=>Boolean(report?.runningSample||report?.pairedReplay?.runningPair||report?.status==='interrupted');
async function recoveryRetirement(run){
 if(!run?.checkpoint?.evaluationDirectory||!run.checkpoint.evaluation?.binding?.environment||!run.owner||!await ownerStopped(run.owner))return false;
 return Boolean(run.checkpoint.unresolved||interruptedReport(await evaluationReport(run.checkpoint)));
}
const evaluationContract=cases=>({target:cases.target,objective:cases.objective,binding:cases.binding,...(cases.protocol?{protocol:cases.protocol}:{}),tasks:cases.tasks.map(task=>{
 const result={...task};for(const key of ['checker','pairedChecker'])if(task[key])result[key]=createHash('sha256').update(readFileSync(task[key])).digest('hex');return result;
})});
async function recordCandidate(config,prepared,options){
 if(config.storage)await (await import('../controller/storage-knowledge.mjs')).recordStorageSkill(config,prepared.candidate,
  {serialized:true,baseVersion:prepared.active,...options});
}
const open=config=>{const state=new State(join(config.root,'state.sqlite'));return {state,backlog:new SkillBacklog(state,{target:config.target})};};
const learningFingerprint=state=>state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='skill_learning_requests'").get()
 ?JSON.stringify(state.db.prepare("SELECT id,status,json_extract(data,'$.findings') findings FROM skill_learning_requests WHERE status IN ('verified','inconclusive') ORDER BY id").all()):'[]';
export async function prioritySkillStatus(config){
 const {state,backlog}=open(config);
 try{
  const status=backlog.status(),items=['queued','learning','needs-review','user-required'].flatMap(s=>backlog.list({status:s,limit:20})).sort((a,b)=>b.priority-a.priority),
   active=backlog.list({status:'running',limit:1})[0],run=active&&backlog.run(active.id);
  const learningWork=backlog.meta('microlearning-active'),learningQueue=pendingMicrolearning(backlog);
  const changed=backlog.meta('learning-seen')!==learningFingerprint(state),cohort=cohortStatus(backlog),continuations=admissibleContinuations(backlog,config.skills.activeVersion);
  const queued=status.items.filter(i=>['queued','needs-review'].includes(i.status)).reduce((sum,i)=>sum+i.n,0);
  return {...status,cohort,activeWork:learningWork?{kind:learningWork.request.workKind,stage:learningWork.stage,id:learningWork.id,elapsedSeconds:learningWork.elapsed,allocatedSeconds:learningWork.seconds}:active?{kind:'hypothesis',stage:run?.checkpoint?.evaluationDirectory?'evaluation':run?.checkpoint?.prepared?'qa':'cases',id:active.id}:null,learningQueued:learningQueue.length,continuationQueued:continuations.length,lastLearning:backlog.meta('microlearning-last'),queued,evaluationStatus:active||learningWork?'running':queued?'deferred':'idle',pending:Boolean(active||learningWork),resource:run?.checkpoint?.resource??learningWork?.resource,unresolved:Boolean(run?.checkpoint?.unresolved||learningWork?.stage==='settlement'),
   analysisError:backlog.meta('analysis-error'),rankingError:backlog.meta('ranking-error'),
   current:active?{id:active.id,objective:active.objective,priority:active.priority,elapsedSeconds:run.elapsed,allocatedSeconds:run.seconds,
    candidate:run.checkpoint?.prepared?.candidate,evaluationDirectory:run.checkpoint?.evaluationDirectory}:null,
   backlog:items.map(({id,skill,objective,status,priority,reason})=>({id,skill,objective,status,priority,reason})),
   hasWork:!config.skills?.tuningDisabled&&(status.ingestion.initialPending||status.ingestion.freshPending||!status.backfillComplete||status.dirtyFailures>0||changed||backlog.meta('learning-pending')||Boolean(active||learningWork||cohort?.counts.pending)||continuations.length>0||learningQueue.some(request=>request.workKind!=='suite')||items.some(i=>['queued','needs-review'].includes(i.status)))};
 }finally{state.close();}
}
export async function prioritySkillsSettled(config,dependencies={}){
 const {state,backlog}=open(config);
 try{
  const item=backlog.list({status:'running',limit:1})[0],run=item&&backlog.run(item.id);
  if(!await microlearningSettled(config,backlog.meta('microlearning-active'),dependencies))return false;
  if(!run)return true;
  if(executionDirectory(run.checkpoint))return await (dependencies.settled??hypothesisEvaluationSettled)({directory:executionDirectory(run.checkpoint),config,resource:run.checkpoint.resource,...run.checkpoint.evaluation,retirePrePayload:await recoveryRetirement(run)});
  // instanceStopped independently checks the writer identity. Preparation has no device command.
  return true;
 }finally{state.close();}
}
export async function runPrioritySkills(config,values,dependencies={}){
 const ops={analyze:analyzeBacklog,prioritize:prioritizeBacklog,prepare:prepareHypothesis,evaluate:evaluateHypothesis,
  learn:runMicrolearning,caseReadiness,resolveCases:resolveHypothesisCases,publish:publishSkillVersion,settled:hypothesisEvaluationSettled,now:()=>performance.now(),...dependencies};
 const {state,backlog}=open(config),directory=join(config.root,'skills','priority');await mkdir(directory,{recursive:true});
 const campaignEnd=ops.now()+values.remainingSeconds*1000,remainingCampaign=()=>Math.max(0,(campaignEnd-ops.now())/1000);
 let selected,checkpoint,timer,started,prior;
 try{
  config=await readConfig(config.root);
  const ingestionEnd=ops.now()+Math.min(1000,remainingCampaign()*1000);let ingestion;
  do{ingestion=backlog.ingest({limit:100});}while(ingestion.progress&&ops.now()<ingestionEnd);
  // Classification is shared work. Proposal writing is subsequently charged to its hypothesis.
  if(backlog.meta('microlearning-active'))return await ops.learn({config,backlog,...values});
  retireStaleContinuations(backlog,config.skills.activeVersion);
  await wakeQualifiedSuiteWaiters({config,backlog});
  const retained=backlog.list({status:'running',limit:1})[0];
  if(retained)attachRetainedCohort(backlog,backlog.run(retained.id));
  const cohort=refreshCohort(backlog);
  if(!retained&&(!cohort||cohort.status==='complete')&&(cohort?.status==='complete'||!hasCohortCandidates(backlog,config.skills.activeVersion))){
   const admittedLearning=learningFingerprint(state);
   if(backlog.meta('learning-seen')!==admittedLearning){backlog.meta('learning-scan',null);backlog.meta('learning-pending',true);}
   // A bounded analysis can produce useful items before its last group times out.
   // Preserve that progress and existing priorities instead of starving evaluation.
   for(const [name,operation,args]of [['analysis',ops.analyze,{directory:join(directory,'analysis'),seconds:Math.min(120,remainingCampaign()),limit:10}],
    ['ranking',ops.prioritize,{directory:join(directory,'ranking'),seconds:Math.min(60,remainingCampaign())}]]){
    if((backlog.meta(name+'-after')??0)>Date.now())continue;
    try{await operation({config,backlog,signal:values.signal,...args});backlog.meta(name+'-error',null);}
    catch(error){values.signal?.throwIfAborted();backlog.meta(name+'-error',{at:new Date().toISOString(),message:error.message});backlog.meta(name+'-after',Date.now()+60000);}
   }
   backlog.meta('learning-seen',admittedLearning);
   for(const item of backlog.list({status:'needs-review'}))backlog.revise(item.id,{status:'queued'});
  }
  values.signal?.throwIfAborted();
  const owner=await processIdentity();
  if(retained){
   const old=backlog.run(retained.id);
   if(old.owner&&!await ownerStopped(old.owner)&&JSON.stringify(old.owner)!==JSON.stringify(owner))return {status:'blocked',reason:'Retained hypothesis owner remains live'};
   const persistedReport=await evaluationReport(old.checkpoint);
   if(old.checkpoint?.unresolved||interruptedReport(persistedReport)||persistedReport?.settledInterruption){
    const cp=old.checkpoint,elapsed=old.elapsed+(cp.runningAt?Math.max(0,Date.now()-cp.runningAt)/1000:0);
    cp.runningAt=Date.now();backlog.checkpoint(old.id,{elapsed,checkpoint:cp});
    const closed=await ops.settled({directory:executionDirectory(cp),config,resource:cp.resource,...cp.evaluation,retirePrePayload:await recoveryRetirement(old)}),charged=elapsed+Math.max(0,Date.now()-cp.runningAt)/1000;
    if(closed){delete cp.unresolved;delete cp.runningAt;}else{cp.unresolved=true;cp.runningAt=Date.now();}
    backlog.checkpoint(old.id,{elapsed:charged,checkpoint:cp});
    if(!closed)return {status:'blocked',reason:'Retained evaluation execution remains unresolved'};
    const interrupted=await evaluationReport(cp);
    if(interrupted?.settledInterruption)return backlog.finish(old.id,{status:'inconclusive',result:{reason:'execution-interrupted',decision:{objective:old.snapshot.objective,promote:false,reason:'execution-interrupted'},report:interrupted}});
   }
   if(old.base!==config.skills.activeVersion){
    if(executionDirectory(old.checkpoint)&&!await ops.settled({directory:executionDirectory(old.checkpoint),config,resource:old.checkpoint.resource,...old.checkpoint.evaluation}))return {status:'blocked',reason:'Stale-baseline execution remains unresolved'};
    if(old.checkpoint?.publicationApproved&&old.checkpoint.prepared?.candidate===config.skills.activeVersion){
     await recordCandidate(config,old.checkpoint.prepared,{promote:true,reason:'recovered-promotion',signal:values.signal});
     return backlog.finish(old.id,{status:'accepted',result:{reason:'recovered-promotion',published:true}});
    }
    backlog.finish(old.id,{status:'inconclusive',result:{reason:'active-skill-changed'}});backlog.rebase(retained.id,config.skills.activeVersion);
    return {status:'needs-review'};
   }
  }
  selected=retained?backlog.claim({base:config.skills.activeVersion,owner}):await claimCohort(backlog,config,owner,ops.caseReadiness);
  if(!selected){
   const requests=pendingMicrolearning(backlog).filter(request=>request.workKind!=='suite');
   return requests.length?await ops.learn({config,backlog,...values,remainingSeconds:remainingCampaign(),requestIds:requests.map(r=>r.id)}):{status:'idle',reason:'No applicable cohort comparison or bounded factual learning'};
  }
  attachRetainedCohort(backlog,selected.run);
  const {run}=selected,item=run.snapshot,home=join(directory,'hypotheses',run.id);await mkdir(home,{recursive:true});
  checkpoint={...run.checkpoint,resource:values.resource};
  prior=run.elapsed+(checkpoint.runningAt?Math.max(0,Date.now()-checkpoint.runningAt)/1000:0);
  if(!checkpoint.proposalCharged){prior+=Math.max(0,item.preparation?.proposalElapsedSeconds??item.evidence?.proposalElapsedSeconds??0);checkpoint.proposalCharged=true;}
  started=ops.now();
  const spent=()=>prior+(ops.now()-started)/1000;
  const save=()=>{if(checkpoint.runningAt)checkpoint.runningAt=Date.now();return backlog.checkpoint(run.id,{elapsed:spent(),checkpoint});};
  const left=()=>Math.max(0,Math.min(run.seconds-spent(),remainingCampaign()));
  checkpoint.runningAt=Date.now();save();timer=setInterval(save,1000);
  if(left()<=30){delete checkpoint.runningAt;save();return backlog.finish(run.id,{status:'inconclusive',result:{reason:'hypothesis-elapsed'}});}
  const budget=AbortSignal.timeout(Math.max(1,Math.ceil(left()*1000))),signal=values.signal?AbortSignal.any([values.signal,budget]):budget;
  const finish=(status,result)=>{
   delete checkpoint.runningAt;save();
   const report=result?.report,diagnostics=report?.results?.filter(r=>r.correct!==true).map(r=>({projectId:r.projectId,run:r.run,arm:r.arm,correct:r.correct,timedOut:r.timedOut,
    error:r.verification?.error,output:String(r.verification?.output??'').slice(-2000),stderr:String(r.verification?.error??'').slice(-2000)}));
   const feedback=['rejected','inconclusive'].includes(status)&&report?.results?.length?{hypothesisId:item.id,groupIds:item.groupIds,objective:item.objective,
    evaluation:{path:join(home,'cases','selected.json')},previousEvidence:{memoryEvidence:item.evidence?.memoryEvidence??[],learningIdentity:item.evidence?.learningIdentity??null},previousProposal:checkpoint.prepared?.proposal??item.proposal,reason:result.decision?.reason??result.reason,diagnostics,
    coverage:report.results.length,metrics:result.decision?.tasks??result.decision?.passes}:null;
   const path=join(home,'feedback.json');
   if(feedback){const bytes=JSON.stringify(feedback,null,2)+'\n';try{writeFileSync(path,bytes,{flag:'wx'});}catch(error){if(error.code!=='EEXIST'||readFileSync(path,'utf8')!==bytes)throw error;}}
   return backlog.atomic(()=>{const done=backlog.finish(run.id,{status,result});if(feedback)state.event(null,'skill-hypothesis-rejected',{...feedback,evidence:{path}});return done;});
  };
  const freezeCases=async cases=>{const {protocol}=await freezeHypothesisProtocol({directory:join(home,'evaluation'),target:cases.target,binding:cases.binding});return {...cases,...(protocol?{protocol}:{})};};
  const qualificationDirectory=join(home,'cases','qualification');
  checkpoint.qualificationDirectory??=qualificationDirectory;save();
  const beforeQualification=async directory=>{checkpoint.qualificationDirectory=directory;save();};
  const finishCases=async cases=>{
   if(!await ops.settled({directory:checkpoint.qualificationDirectory,config,resource:values.resource})){checkpoint.unresolved=true;save();return {status:'blocked',reason:'Case qualification execution remains unresolved'};}
   return finish(cases.status==='needs-learning'?'learning':'inconclusive',cases);
  };
  let cases=await ops.resolveCases({config,backlog,item,beforeQualification,proposal:checkpoint.prepared?.status==='ready'?checkpoint.prepared.proposal??item.proposal:item.proposal,directory:join(home,'cases'),qualificationDirectory,
   resource:values.resource,signal,seconds:left(),event:(type,data)=>state.event(null,type,data)});
  if(cases.status!=='ready')return await finishCases(cases);
  cases=await freezeCases(cases);
  satisfySuiteRequests(backlog,item.id,cases);
  const contractKey=createHash('sha256').update(JSON.stringify(evaluationContract(cases))).digest('hex');
  if(!checkpoint.prepared||checkpoint.prepared.retryable||checkpoint.reviewedContract!==contractKey){
   const previous=checkpoint.prepared,qaElapsed=previous?.qa?.elapsedSeconds??previous?.elapsedSeconds??checkpoint.priorQaElapsedSeconds??0;
   checkpoint.prepared=await ops.prepare({config,backlog,item,directory:join(home,'prepare-v2-'+contractKey),signal,remainingSeconds:left(),evaluationContract:evaluationContract(cases),priorElapsedSeconds:qaElapsed});
   checkpoint.reviewedContract=contractKey;save();
  }
  if(checkpoint.prepared.status!=='ready'){
   if(checkpoint.prepared.retryable)return finish('inconclusive',checkpoint.prepared);
   return finish(checkpoint.prepared.status==='learning'?'learning':'inconclusive',checkpoint.prepared);
  }
  const reviewedCases=JSON.stringify(evaluationContract(cases));
  cases=await ops.resolveCases({config,backlog,item,beforeQualification,proposal:checkpoint.prepared.proposal??item.proposal,directory:join(home,'cases'),qualificationDirectory,
   resource:values.resource,signal,seconds:left(),event:(type,data)=>state.event(null,type,data)});
  if(cases.status!=='ready')return await finishCases(cases);
  cases=await freezeCases(cases);
  if(JSON.stringify(evaluationContract(cases))!==reviewedCases&&checkpoint.prepared.proposal){
   checkpoint.prepared=await ops.prepare({config,backlog,item:{...item,proposal:checkpoint.prepared.proposal},directory:join(home,'final-contract'),signal,remainingSeconds:left(),evaluationContract:evaluationContract(cases),priorElapsedSeconds:checkpoint.prepared.qa?.elapsedSeconds??checkpoint.prepared.elapsedSeconds??0});checkpoint.reviewedContract=createHash('sha256').update(JSON.stringify(evaluationContract(cases))).digest('hex');save();
   if(checkpoint.prepared.status!=='ready')return finish(checkpoint.prepared.status==='learning'?'learning':'inconclusive',checkpoint.prepared);
   let finalCases=await ops.resolveCases({config,backlog,item,beforeQualification,proposal:checkpoint.prepared.proposal??item.proposal,directory:join(home,'cases'),qualificationDirectory,resource:values.resource,signal,seconds:left()});
   if(finalCases.status!=='ready')return await finishCases(finalCases);
   finalCases=await freezeCases(finalCases);
   if(JSON.stringify(evaluationContract(finalCases))!==JSON.stringify(evaluationContract(cases)))return finish('learning',{reason:'Final proposal requires a different qualified contract'});
  }
  await recordCandidate(config,checkpoint.prepared,{reason:'awaiting-evaluation',signal});
  checkpoint.evaluationDirectory=join(home,'evaluation');checkpoint.evaluation={binding:cases.binding,resource:values.resource};save();
  const result=await ops.evaluate({directory:checkpoint.evaluationDirectory,config,objective:item.objective,target:cases.target,
   tasks:cases.tasks,binding:cases.binding,skills:{active:join(config.root,'skills','versions',checkpoint.prepared.active),candidate:join(config.root,'skills','versions',checkpoint.prepared.candidate)},
   remainingSeconds:left(),resource:values.resource,signal});
  if(!await ops.settled({directory:checkpoint.evaluationDirectory,config,...checkpoint.evaluation})){checkpoint.unresolved=true;save();return {status:'blocked',reason:'Evaluation execution remains unresolved'};}
  delete checkpoint.unresolved;
  if(signal.aborted||left()<=0)return finish('inconclusive',{...result,reason:'hypothesis-elapsed'});
  if(result.decision?.promote){
   checkpoint.publicationApproved=true;save();
   const published=await ops.publish(config.root,{active:checkpoint.prepared.active,candidate:checkpoint.prepared.candidate,signal});
   await recordCandidate(config,checkpoint.prepared,{promote:Boolean(published),reason:result.decision.reason,signal});
   return finish(published?'accepted':'inconclusive',{...result,published:Boolean(published)});
  }
  await recordCandidate(config,checkpoint.prepared,{promote:false,reason:result.decision?.reason,signal});
  return finish(result.decision?.reason==='inconclusive'?'inconclusive':'rejected',result);
 }catch(error){
  if(selected){
   const settled=await prioritySkillsSettled(config,{settled:ops.settled});
   if(checkpoint){if(settled)delete checkpoint.runningAt;else{checkpoint.unresolved=true;checkpoint.runningAt=Date.now();}
    backlog.checkpoint(selected.run.id,{elapsed:prior+(ops.now()-started)/1000,checkpoint});}
   const promoted=checkpoint?.publicationApproved&&checkpoint.prepared?.candidate===(await readConfig(config.root)).skills.activeVersion;
   if(!values.signal?.aborted&&settled&&!promoted)backlog.finish(selected.run.id,{status:'inconclusive',result:{reason:error.message}});
  }
  throw error;
 }finally{clearInterval(timer);refreshCohort(backlog);state.close();}
}

/** Explicit one-time procedural migration; claimed foreground projects and old run records stay intact. */
export async function migratePriorityThroughput(config,{procedure,base=config.skills.activeVersion}={}){
 if(!procedure?.trim())throw Error('Audited procedure description required');
 const {state,backlog}=open(config);try{return backlog.atomic(()=>{
  if(backlog.meta('throughput-migration'))return backlog.meta('throughput-migration');
  const queue=[],tagged=[],seen=new Set();
  for(const {id}of backlog.db.prepare("SELECT id FROM skill_backlog_runs ORDER BY rowid DESC").all()){
   const run=backlog.run(id);if(seen.has(run.item_id))continue;seen.add(run.item_id);
   if(!['learning','inconclusive'].includes(run.status))continue;
   if(run.base!==base||run.elapsed>=run.seconds||run.checkpoint?.evaluationDirectory||run.result?.report?.results?.length)continue;
   let qualification;try{qualification=JSON.parse(readFileSync(join(run.checkpoint?.qualificationDirectory??join(config.root,'skills','priority','hypotheses',run.id,'cases','qualification'),'micro','results.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
   if(qualification?.runningSample||qualification?.results?.length)continue;
   const requestId=run.result?.learningRequest,row=requestId&&backlog.db.prepare("SELECT data,status FROM skill_learning_requests WHERE id=?").get(requestId),request=row&&JSON.parse(row.data);
   const missingCases=Boolean(run.status==='learning'&&row?.status==='pending'&&request?.backlogId===run.item_id&&request.question?.startsWith('Qualify an experiment for '));
   if(!missingCases&&!['qa-elapsed','qa-execution-failed'].includes(run.result?.reason))continue;
   if(missingCases){request.workKind='suite';request.familyKey=createHash('sha256').update(JSON.stringify([config.target,run.snapshot.objective,run.snapshot.skill,run.snapshot.proposal?.changes])).digest('hex');
    backlog.db.prepare("UPDATE skill_learning_requests SET data=? WHERE id=? AND status='pending'").run(JSON.stringify(request),requestId);tagged.push(requestId);}
   queue.push({id,procedure,base,missingCases});
  }
  const audit={procedure,base,at:new Date().toISOString(),tagged,continuations:queue.map(q=>q.id)};
  backlog.atomic(()=>{backlog.meta('procedural-continuations',queue);backlog.meta('throughput-migration',audit);state.event(null,'skill-throughput-migrated',audit);});return audit;});
 }finally{state.close();}
}

function satisfySuiteRequests(backlog,itemId,cases){
 if(!backlog.db.prepare("SELECT 1 FROM sqlite_master WHERE name='skill_learning_requests'").get())return;
 for(const row of backlog.db.prepare("SELECT id,data FROM skill_learning_requests WHERE status='pending'").all()){
  const request=JSON.parse(row.data);if(request.workKind!=='suite'||request.backlogId!==itemId&&!request.backlogIds?.includes(itemId))continue;
  backlog.db.prepare("UPDATE skill_learning_requests SET status='satisfied',data=? WHERE id=? AND status='pending'").run(JSON.stringify({...request,satisfiedBy:{kind:'qualified-catalogue',identity:cases.identity??cases.proposalIdentity,at:new Date().toISOString()}}),row.id);
 }
}
/** A new qualified catalogue permits one relevance retry, using the original hypothesis allowance. */
export async function wakeQualifiedSuiteWaiters({config,backlog}){
 const suites=(await qualifiedSuites({config})).filter(s=>s.target==='cpu'||s.target===config.target&&JSON.stringify(s.binding?.environment)===JSON.stringify(config.environment?.options));
 const identity=createHash('sha256').update(JSON.stringify(suites.map(s=>s.qualification.fingerprint).sort())).digest('hex');
 if(!suites.length||backlog.meta('qualified-suite-wake')===identity)return [];
 const queue=backlog.meta('procedural-continuations')??[],existing=new Set(queue.map(q=>q.id)),seen=new Set(),added=[];
 if(backlog.db.prepare("SELECT 1 FROM sqlite_master WHERE name='skill_learning_requests'").get())for(const {id}of backlog.db.prepare('SELECT id FROM skill_backlog_runs ORDER BY rowid DESC').all()){
  const run=backlog.run(id);if(seen.has(run.item_id))continue;seen.add(run.item_id);
  if(run.status!=='learning'||run.base!==config.skills.activeVersion||run.elapsed>=run.seconds||run.checkpoint?.evaluationDirectory||run.result?.report?.results?.length||!run.result?.learningRequest||existing.has(id)||
   !suites.some(s=>s.objective===run.snapshot.objective))continue;
  const item=backlog.get(run.item_id);if(!item||item.fingerprint!==run.snapshot.fingerprint)continue;
  const row=backlog.db.prepare("SELECT data FROM skill_learning_requests WHERE id=? AND status='pending'").get(run.result.learningRequest),request=row&&JSON.parse(row.data);
  if(!request||request.workKind!=='suite'||request.backlogId!==run.item_id||!request.question?.startsWith('Qualify an experiment for '))continue;
  let qualification;try{qualification=JSON.parse(readFileSync(join(run.checkpoint?.qualificationDirectory??join(config.root,'skills','priority','hypotheses',run.id,'cases','qualification'),'micro','results.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
  if(qualification?.runningSample||qualification?.results?.length)continue;
  queue.push({id,base:run.base,missingCases:true,procedure:'Qualified catalogue '+identity+' published; recheck semantic relevance'});added.push(id);
 }
 backlog.atomic(()=>{backlog.meta('procedural-continuations',queue);backlog.meta('qualified-suite-wake',identity);backlog.db.prepare("INSERT INTO events(type,data,created) VALUES('skill-catalogue-woke-waiters',?,?)").run(JSON.stringify({identity,runs:added}),new Date().toISOString());});
 return added;
}
