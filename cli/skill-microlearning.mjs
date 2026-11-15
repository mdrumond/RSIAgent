/** One bounded child on the campaign's already reserved background resource. */
import {mkdir,readFile,writeFile,cp,symlink} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {join,resolve,sep} from 'node:path';
import {State} from '../controller/state.mjs';
import {processIdentity,ownerStopped} from '../controller/lifecycle.mjs';
import {refillProjects} from './project-generation.mjs';
import {importProject} from './instance.mjs';
import {operate,instanceStopped} from './runtime.mjs';
import {ensureKnowledgeService} from './knowledge.mjs';
import {freezeGroups} from './skill-backlog-analysis.mjs';
import {qualifyLearningSuite} from './skill-priority-cases.mjs';
import {hypothesisEvaluationSettled} from './skill-hypothesis-evaluation.mjs';
import {createKdbBridge} from '../integrations/kdb-extension.mjs';
const has=db=>db.prepare("SELECT 1 FROM sqlite_master WHERE name='skill_learning_requests'").get();
export const smallLearning=r=>r.intervention!=='user-required'&&(r.workKind==='suite'||r.workKind==='microlearning');
export function pendingMicrolearning(backlog){
 if(!has(backlog.db))return [];
 return backlog.db.prepare("SELECT id,data FROM skill_learning_requests WHERE status='pending'").all().map(row=>({...JSON.parse(row.data),id:row.id}))
  .filter(smallLearning).sort((a,b)=>(backlog.get(b.backlogId)?.priority??b.priority??0)-(backlog.get(a.backlogId)?.priority??a.priority??0));
}
export function validateArchivalInspection(value,{refs,request,retrievals=[]}){
 const known=new Set([...refs,...retrievals.flatMap(record=>(record.entries??[]).flatMap(entry=>{
  const collection=entry.collection??record.arguments?.collection,id=entry.entry_id??entry.chunk_id??entry.id;
  return collection&&id?[`${collection}:${id}`]:[];
 }))]);
 if(!['resolved','experiment'].includes(value?.status)||!value.answer?.trim()||!Array.isArray(value.refs)||value.refs.some(ref=>!known.has(ref))||value.status==='resolved'&&(!value.refs.length||request.workKind==='suite'))throw Error('Cited archival facts or a required experiment decision expected');
 return value;
}
export function microlearningConfig(config,{directory,request,resource,seconds}){
 if(!resource?.id)throw Error('Microlearning requires the admitted background resource');
 const picked=Object.fromEntries(['actor','curriculum_actor','python','kernel_language','environment','local_execution','delivery','postprocessing','knowledge','pause_grace_seconds'].filter(k=>config[k]!==undefined).map(k=>[k,config[k]]));
 const cpu=request.learningTarget==='cpu';
 if(cpu){delete picked.kernel_language;picked.environment={module:fileURLToPath(new URL('../integrations/environments/index.mjs',import.meta.url)),options:{mode:'local',localResourceId:resource.id}};}
 if(picked.environment?.module)picked.environment={...picked.environment,module:resolve(config.root,picked.environment.module)};
 return {...picked,format:1,root:directory,target:cpu?'cpu':config.target,mode:'phase1',concurrency:1,resource_ids:[resource.id],actor_seconds:seconds,curriculum_seconds:seconds,
  campaign_plan:[{id:'microlearning',phase:'phase1',new_projects:1}],phase_targets:{phase1:1},
  curriculum:{generation:{enabled:true,topics:[request.question],preprompt:'Investigate only this bounded learning question. Preserve independent reference and negative controls; report actual observations: '+JSON.stringify(request)}},
  campaign_seconds:seconds,skills:{activeVersion:config.skills.activeVersion,background:false,tuningDisabled:true,review:{enabled:false},priority:{enabled:false}},recovery:false};
}
export async function microlearningSettled(config,work,dependencies={}){
 if(!work)return true;
 if(work.owner&&!await ownerStopped(work.owner))return false;
 let child;try{child=JSON.parse(await readFile(join(work.directory,'instance.json'),'utf8'));}catch(e){if(e.code==='ENOENT')return true;throw e;}
 if(work.qualificationDirectory&&!await (dependencies.settled??hypothesisEvaluationSettled)({directory:work.qualificationDirectory,config,resource:work.resource}))return false;
 return (dependencies.stopped??instanceStopped)(child);
}
export async function runMicrolearning({config,backlog,resource,remainingSeconds,signal},dependencies={}){
 const ops={qualify:qualifyLearningSuite,settled:hypothesisEvaluationSettled,inspect:async input=>(await import('./skill-review-native.mjs')).reviewSession(input),construct:refillProjects,importProject,operate,stopped:instanceStopped,ensureKnowledgeService,createKdbBridge,now:()=>performance.now(),...dependencies};
 let work=backlog.meta('microlearning-active');
 if(work?.owner&&!await ownerStopped(work.owner))return {status:'blocked',reason:'Microlearning owner remains live'};
 if(!work){
  const request=pendingMicrolearning(backlog)[0];if(!request)return {status:'idle'};
  work={id:request.id,request,directory:join(config.root,'skills','priority','learning',request.id),elapsed:0,seconds:900,stage:'construction',resource};
  const changed=backlog.atomic(()=>{if(backlog.meta('microlearning-active'))return 0;const changed=backlog.db.prepare("UPDATE skill_learning_requests SET status='background',draft=? WHERE id=? AND status='pending'").run('background:'+request.id,request.id).changes;if(changed)backlog.meta('microlearning-active',work);return changed;});
  if(!changed)return {status:'deferred'};
 }
 if(work.resource.id!==resource.id)return {status:'blocked',reason:'Retained learning owns a different resource'};
 if(!await microlearningSettled(config,work,{stopped:ops.stopped,settled:ops.settled}))return {status:'blocked',reason:'Learning child execution remains unresolved'};
 work.owner=await processIdentity();work.elapsed+=(work.runningAt?Math.max(0,Date.now()-work.runningAt)/1000:0);
 const begun=ops.now(),prior=work.elapsed,spent=()=>prior+(ops.now()-begun)/1000;
 const save=()=>{work.elapsed=spent();work.runningAt=Date.now();backlog.meta('microlearning-active',work);};
 const left=()=>Math.max(0,Math.min(work.seconds-spent(),remainingSeconds-(ops.now()-begun)/1000));
 await mkdir(work.directory,{recursive:true});save();const timer=setInterval(save,1000);
 const budget=AbortSignal.timeout(Math.max(1,Math.ceil(left()*1000))),active=signal?AbortSignal.any([signal,budget]):budget;
 let child;
 try{
  try{child=JSON.parse(await readFile(join(work.directory,'instance.json'),'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
  if(left()<=0){
   if(child&&!await ops.stopped(child))return {status:'blocked',reason:'Expired learning child remains unresolved'};
   work.elapsed=spent();delete work.runningAt;delete work.owner;backlog.meta('microlearning-last',{...work,status:'inconclusive',reason:'microlearning-elapsed'});backlog.meta('microlearning-active',null);
   backlog.db.prepare("UPDATE skill_learning_requests SET status='inconclusive' WHERE id=?").run(work.id);return {status:'inconclusive',reason:'microlearning-elapsed'};
  }
  child??=microlearningConfig(config,{directory:work.directory,request:work.request,resource,seconds:Math.max(.001,left())});
  const version=child.skills.activeVersion,source=resolve(config.root,'skills','versions',version),destination=resolve(work.directory,'skills','versions',version);
  if(!source.startsWith(resolve(config.root,'skills','versions')+sep)||destination.startsWith(source+sep))throw Error('Invalid pinned skill version');
  await mkdir(join(work.directory,'skills','versions'),{recursive:true});
  if(!work.skillsPinned){await cp(source,destination,{recursive:true,errorOnExist:false});work.skillsPinned=true;save();}
  const url=await ops.ensureKnowledgeService(config,{signal:active});
  const receipt=JSON.parse(await readFile(join(config.root,'knowledge-service.json'),'utf8'));
  await writeFile(join(work.directory,'knowledge-service.json'),JSON.stringify({...receipt,url,ownership:'borrowed',borrowedFrom:config.root}));
  await symlink(join(config.root,'machinery'),join(work.directory,'machinery'),'dir').catch(e=>{if(e.code!=='EEXIST')throw e;});
  await symlink(join(config.root,'knowledge.sqlite'),join(work.directory,'knowledge.sqlite')).catch(e=>{if(e.code!=='EEXIST')throw e;});
  await writeFile(join(work.directory,'instance.json'),JSON.stringify(child));
  if(!work.inspection){
   work.stage='inspection';save();
   const groups=await freezeGroups(config.root,(work.request.groupIds??[]).map(id=>backlog.failure(id,{limit:2})).filter(Boolean),join(work.directory,'inspection','frozen'));
   const refs=new Set(groups.flatMap(g=>[...(g.occurrences??[]),...(g.linkedDiagnostics??[])].map(o=>o.reference)));
   const inspection=await ops.inspect({directory:join(work.directory,'inspection'),config,signal:active,seconds:Math.min(90,left()),knowledge:true,evidenceGroups:groups,
    prompt:{task:'Inspect the supplied archived diagnostics and indexed knowledge first. Return {status:resolved|experiment,answer,refs:[]}. resolved is allowed only for an archival factual question fully established by cited supplied diagnostics or actual KDB retrievals. Constructing a new suite or establishing device correctness/performance requires experiment. Do not invent device evidence.',request:work.request,groups},
    validate:(value,retrievals=[])=>validateArchivalInspection(value,{refs,request:work.request,retrievals})});
   work.inspection=inspection.value??inspection;save();
   if(work.inspection.status==='resolved'){
    for(const group of groups){const row=backlog.db.prepare('SELECT data FROM skill_backlog_failures WHERE id=?').get(group.id);
     backlog.db.prepare('UPDATE skill_backlog_failures SET data=?,dirty=1 WHERE id=?').run(JSON.stringify({...JSON.parse(row.data),archivalInspection:{...work.inspection,directory:join(work.directory,'inspection')}}),group.id);}
    backlog.db.prepare("UPDATE skill_learning_requests SET status='inconclusive',data=? WHERE id=?").run(JSON.stringify({...work.request,archivalInspection:work.inspection}),work.id);
    work.elapsed=spent();delete work.runningAt;delete work.owner;backlog.meta('microlearning-last',{...work,status:'archival-resolved'});backlog.meta('microlearning-active',null);
    return {status:'archival-resolved',inspection:work.inspection};
   }
  }
  const state=new State(join(work.directory,'state.sqlite'));
  try{
   const projects=state.db.prepare('SELECT id FROM projects').all();if(projects.length>1)throw Error('Microlearning accepts exactly one project');
   let spec=projects.length?state.project(projects[0].id).spec:null;
   if(!spec&&left()>0){
    const bridge=ops.createKdbBridge({url,attemptId:work.id,agentDir:join(work.directory,'agent')});
    const drafts=await ops.construct({config:child,state,bridge,concurrency:1,path:join(work.directory,'drafts'),signal:active});
    if(drafts[0]){spec=await ops.importProject(work.directory,{...drafts[0],phase:'phase1',campaign_entry:'microlearning',actor_seconds:Math.max(.001,left())},{signal:active});state.add(spec);}
   }
   if(spec&&left()>0){
    child.projects=[spec];child.curriculum={phase1:[spec.id],generation:{enabled:false}};child.actor_seconds=Math.min(spec.actor_seconds,left());
    await writeFile(join(work.directory,'instance.json'),JSON.stringify(child));work.stage='execution';save();
    const attempts=state.db.prepare("SELECT a.outcome,EXISTS(SELECT 1 FROM events e WHERE e.attempt=a.id AND e.type='postprocess-completed') AS processed FROM attempts a").all();
    if(!attempts.length||attempts.some(a=>a.outcome===null||!a.processed))await ops.operate(attempts.length?'resume':'wave',child,{signal:active,'no-recovery':true});
   }
   work.stage='settlement';save();
   if(!await ops.stopped(child))return {status:'blocked',reason:'Learning child execution remains unresolved'};
   const passed=state.db.prepare("SELECT a.id,a.project FROM attempts a WHERE a.outcome='passed' AND EXISTS(SELECT 1 FROM events e WHERE e.attempt=a.id AND e.type='postprocess-completed') ORDER BY a.rowid DESC LIMIT 1").get();
   const data={...work.request,findings:passed?{instanceRoot:work.directory,projectId:passed.project,attemptId:passed.id,outcome:'passed'}:{instanceRoot:work.directory,outcome:'inconclusive'}};
   let verified=Boolean(passed),qualification;
   if(passed&&work.request.workKind==='suite'){
    work.stage='qualification';work.qualificationDirectory=join(work.directory,'qualification');save();
    qualification=left()>0?await ops.qualify({config,backlog,request:data,directory:work.qualificationDirectory,resource,signal:active,seconds:left()}):{status:'inconclusive',reason:'microlearning-elapsed'};
    if(!await ops.settled({directory:work.qualificationDirectory,config,resource})){work.stage='settlement';save();return {status:'blocked',reason:'Suite qualification remains unresolved'};}
    verified=qualification.status==='qualified';data.qualification={status:qualification.status,reason:qualification.reason};
   }
   backlog.db.prepare('UPDATE skill_learning_requests SET status=?,data=? WHERE id=?').run(verified?'verified':'inconclusive',JSON.stringify(data),work.id);
   work.elapsed=spent();delete work.runningAt;delete work.owner;backlog.meta('microlearning-last',{...work,status:verified?'verified':'inconclusive',reason:qualification?.reason});backlog.meta('microlearning-active',null);
   return {status:verified?'verified':'inconclusive',findings:data.findings,qualification};
  }finally{state.close();}
 }catch(error){work.stage='settlement';save();throw error;}finally{clearInterval(timer);if(backlog.meta('microlearning-active')){save();delete work.owner;if(child&&await ops.stopped(child)&&(!work.qualificationDirectory||await ops.settled({directory:work.qualificationDirectory,config,resource})))delete work.runningAt;backlog.meta('microlearning-active',work);}}
}
