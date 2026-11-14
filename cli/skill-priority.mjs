/** Campaign-local failure backlog; one retained hypothesis owns the background slot. */
import {mkdir} from 'node:fs/promises';
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {State} from '../controller/state.mjs';
import {SkillBacklog} from '../controller/skill-backlog.mjs';
import {ownerStopped,processIdentity} from '../controller/lifecycle.mjs';
import {readConfig} from './instance.mjs';
import {publishSkillVersion} from './skills.mjs';
import {analyzeBacklog,prioritizeBacklog,prepareHypothesis} from './skill-backlog-analysis.mjs';
import {evaluateHypothesis,hypothesisEvaluationSettled} from './skill-hypothesis-evaluation.mjs';
import {resolveHypothesisCases} from './skill-priority-cases.mjs';

const executionDirectory=checkpoint=>checkpoint?.evaluationDirectory??checkpoint?.qualificationDirectory;
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
  const changed=backlog.meta('learning-seen')!==learningFingerprint(state);
  const queued=status.items.filter(i=>['queued','needs-review'].includes(i.status)).reduce((sum,i)=>sum+i.n,0);
  return {...status,queued,evaluationStatus:active?'running':queued?'deferred':'idle',pending:Boolean(active),resource:run?.checkpoint?.resource,unresolved:Boolean(run?.checkpoint?.unresolved),
   analysisError:backlog.meta('analysis-error'),rankingError:backlog.meta('ranking-error'),
   current:active?{id:active.id,objective:active.objective,priority:active.priority,elapsedSeconds:run.elapsed,allocatedSeconds:run.seconds,
    candidate:run.checkpoint?.prepared?.candidate,evaluationDirectory:run.checkpoint?.evaluationDirectory}:null,
   backlog:items.map(({id,skill,objective,status,priority,reason})=>({id,skill,objective,status,priority,reason})),
   hasWork:!config.skills?.tuningDisabled&&(status.ingestion.initialPending||status.ingestion.freshPending||!status.backfillComplete||status.dirtyFailures>0||changed||backlog.meta('learning-pending')||Boolean(active)||items.some(i=>['queued','needs-review'].includes(i.status)))};
 }finally{state.close();}
}
export async function prioritySkillsSettled(config,dependencies={}){
 const {state,backlog}=open(config);
 try{
  const item=backlog.list({status:'running',limit:1})[0],run=item&&backlog.run(item.id);
  if(!run)return true;
  if(executionDirectory(run.checkpoint))return await (dependencies.settled??hypothesisEvaluationSettled)({directory:executionDirectory(run.checkpoint),config,resource:run.checkpoint.resource,...run.checkpoint.evaluation});
  // instanceStopped independently checks the writer identity. Preparation has no device command.
  return true;
 }finally{state.close();}
}
export async function runPrioritySkills(config,values,dependencies={}){
 const ops={analyze:analyzeBacklog,prioritize:prioritizeBacklog,prepare:prepareHypothesis,evaluate:evaluateHypothesis,
  resolveCases:resolveHypothesisCases,publish:publishSkillVersion,settled:hypothesisEvaluationSettled,now:()=>performance.now(),...dependencies};
 const {state,backlog}=open(config),directory=join(config.root,'skills','priority');await mkdir(directory,{recursive:true});
 const campaignEnd=ops.now()+values.remainingSeconds*1000,remainingCampaign=()=>Math.max(0,(campaignEnd-ops.now())/1000);
 let selected,checkpoint,timer,started,prior;
 try{
  config=await readConfig(config.root);
  const ingestionEnd=ops.now()+Math.min(1000,remainingCampaign()*1000);let ingestion;
  do{ingestion=backlog.ingest({limit:100});}while(ingestion.progress&&ops.now()<ingestionEnd);
  // Classification is shared work. Proposal writing is subsequently charged to its hypothesis.
  const retained=backlog.list({status:'running',limit:1})[0];
  if(!retained){
   if(backlog.meta('learning-seen')!==learningFingerprint(state)){backlog.meta('learning-scan',null);backlog.meta('learning-pending',true);}
   // A bounded analysis can produce useful items before its last group times out.
   // Preserve that progress and existing priorities instead of starving evaluation.
   for(const [name,operation,args]of [['analysis',ops.analyze,{directory:join(directory,'analysis'),seconds:Math.min(120,remainingCampaign()),limit:10}],
    ['ranking',ops.prioritize,{directory:join(directory,'ranking'),seconds:Math.min(60,remainingCampaign())}]]){
    if((backlog.meta(name+'-after')??0)>Date.now())continue;
    try{await operation({config,backlog,signal:values.signal,...args});backlog.meta(name+'-error',null);}
    catch(error){values.signal?.throwIfAborted();backlog.meta(name+'-error',{at:new Date().toISOString(),message:error.message});backlog.meta(name+'-after',Date.now()+60000);}
   }
   backlog.meta('learning-seen',learningFingerprint(state));
   for(const item of backlog.list({status:'needs-review'}))backlog.revise(item.id,{status:'queued'});
  }
  values.signal?.throwIfAborted();
  const owner=await processIdentity();
  if(retained){
   const old=backlog.run(retained.id);
   if(old.owner&&!await ownerStopped(old.owner)&&JSON.stringify(old.owner)!==JSON.stringify(owner))return {status:'blocked',reason:'Retained hypothesis owner remains live'};
   if(old.checkpoint?.unresolved){
    const cp=old.checkpoint,elapsed=old.elapsed+(cp.runningAt?Math.max(0,Date.now()-cp.runningAt)/1000:0);
    cp.runningAt=Date.now();backlog.checkpoint(old.id,{elapsed,checkpoint:cp});
    if(!await ops.settled({directory:executionDirectory(cp),config,resource:cp.resource,...cp.evaluation}))return {status:'blocked',reason:'Retained evaluation execution remains unresolved'};
    delete cp.unresolved;delete cp.runningAt;backlog.checkpoint(old.id,{elapsed,checkpoint:cp});
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
  selected=backlog.claim({base:config.skills.activeVersion,owner});
  if(!selected)return {status:'idle',backlog:backlog.status()};
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
  if(!checkpoint.prepared){
   checkpoint.prepared=await ops.prepare({config,backlog,item,directory:join(home,'prepare'),signal,remainingSeconds:left()});save();
  }
  if(checkpoint.prepared.status!=='ready')return finish(checkpoint.prepared.status==='learning'?'learning':'inconclusive',checkpoint.prepared);
  await recordCandidate(config,checkpoint.prepared,{reason:'awaiting-evaluation',signal});
  checkpoint.qualificationDirectory=join(home,'cases','qualification');save();
  const cases=await ops.resolveCases({config,backlog,item,directory:join(home,'cases'),qualificationDirectory:checkpoint.qualificationDirectory,
   resource:values.resource,signal,seconds:left(),event:(type,data)=>state.event(null,type,data)});
  if(cases.status!=='ready')return finish('learning',cases);
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
 }finally{clearInterval(timer);state.close();}
}
