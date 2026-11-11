/** Campaign adapter for short CPU skill comparisons; old comparisons stay immutable. */
import {readFile,writeFile,mkdir,rename} from 'node:fs/promises';
import {join} from 'node:path';
import {randomUUID} from 'node:crypto';
import {State} from '../controller/state.mjs';
import {ownerStopped} from '../controller/lifecycle.mjs';
import {evaluateNonInferiority} from '../controller/skill-noninferiority.mjs';
import {runMicroeval} from './skill-microeval.mjs';
import {tasks} from './microtasks/tasks.mjs';
import {withSkillLock} from './evolution-lock.mjs';
import {readConfig,updateConfig} from './instance.mjs';
const read=async path=>{try{return JSON.parse(await readFile(path,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;return null;}};
const save=async(path,value)=>{const tmp=path+'.'+randomUUID();await writeFile(tmp,JSON.stringify(value,null,2)+'\n',{mode:0o600});await rename(tmp,path);};
export function settings(config) {
 const value={seconds:90,totalSeconds:900,runs:2,margin:.05,...config.skills?.microEvaluation};
 if(!Number.isInteger(value.runs)||value.runs<2||!Number.isFinite(value.margin)||value.margin<0||value.margin>1||
  !Number.isFinite(value.seconds)||value.seconds<=0||!Number.isFinite(value.totalSeconds)||value.totalSeconds<=0)
  throw Error('Invalid micro evaluation policy');
 return value;
}
export const projectIds=()=>tasks.map(t=>t.id);
export function decide(results,ids,runs,config) {
 return evaluateNonInferiority(results,{projectIds:ids,runs,margin:settings(config).margin});
}
export async function settled({instanceRoot}) {
 const report=await read(join(instanceRoot,'micro','results.json'));
 return !report||(!report.runningSample&&(!report.owner||await ownerStopped(report.owner)));
}
export async function compare({pending,signal},dependencies={}) {
 const config=pending.config,policy=settings(config),root=pending.evaluationRoot;
 await mkdir(root,{recursive:true});
 const plan=pending.projectIds.flatMap(projectId=>Array.from({length:pending.runs},(_,run)=>(run%2?['candidate','active']:['active','candidate'])
  .map(arm=>({projectId,run,arm,instanceRoot:root})))).flat();
 const checkpoint=async report=>save(join(root,'round.json'),{kind:'micro',status:report.status==='complete'?'completed':report.status==='elapsed'?'exhausted':report.status,
  allocatedSeconds:pending.evaluationSeconds,elapsedSeconds:report.elapsedSeconds??0,...(report.owner?{runningAt:Date.now()}:{}),
  samples:plan.map(sample=>{const result=report.results.find(r=>r.projectId===sample.projectId&&r.run===sample.run&&r.arm===sample.arm);return {...sample,...(result?{result}: {})};})});
 try {
  const report=await runMicroeval({directory:join(root,'micro'),skills:{active:join(pending.sourceInstanceRoot,'skills','versions',pending.active),candidate:join(pending.sourceInstanceRoot,'skills','versions',pending.candidate)},
   tasks,profile:config.actor,python:config.python,runs:pending.runs,seconds:policy.seconds,totalSeconds:pending.evaluationSeconds,signal,resume:true,shadow:false},
   {...dependencies,checkpoint});
  await checkpoint(report);return await read(join(root,'round.json'));
 }catch(error){error.evaluationStopped=await settled({instanceRoot:root});throw error;}
}

// The caller must pause the campaign. This operation never rewrites project rows,
// session identities, results, elapsed budgets or the previous comparison files.
export async function restartMicroEvaluation(config,{settings:requested={}}={}) {
 const {instanceStopped}=await import('./runtime.mjs');
 const {backgroundSkillsSettled}=await import('./skill-background.mjs');
 if(!await backgroundSkillsSettled(config)||!await instanceStopped(config))throw Error('Stop campaign and evaluation executions before restarting A/B');
 return withSkillLock(config,async()=>{
  const current=await readConfig(config.root),ledger=new State(join(config.root,'state.sqlite'));
  try {
   const get=key=>JSON.parse(ledger.db.prepare('SELECT value FROM control WHERE key=?').get(key)?.value??'null');
   if(!get('pause'))throw Error('Pause campaign before restarting A/B');
   for(const key of ['owner','supervisor-owner','memory-writer'])if(get(key)&&!await ownerStopped(get(key)))throw Error('Campaign writer remains live');
   if(ledger.db.prepare("SELECT 1 FROM sqlite_master WHERE name='requests'").get())for(const request of ledger.db.prepare("SELECT status,owner FROM requests WHERE status IN ('queued','running')").all())
    if(request.status==='queued'||!await ownerStopped(request.owner))throw Error('Background request remains live');
   const file=join(config.root,'skills','evolution.json'),before=await readFile(file),evolution=JSON.parse(before);
   const candidate=evolution.pending?.candidate??evolution.candidate;
   if(!/^[A-Za-z0-9-]+$/.test(candidate??'')||evolution.candidateBase!==current.skills.activeVersion)throw Error('Current candidate required');
   await readFile(join(config.root,'skills','versions',candidate,'skill-map.json'));
   const policy=settings({skills:{microEvaluation:requested}}),at=new Date().toISOString(),id=randomUUID();
   const archives=join(config.root,'skills','restarts');await mkdir(archives,{recursive:true});
   await save(join(archives,id+'.json'),{at,reason:'operator-restart-micro-evaluation',evolution});
   const nextConfig={...current,skills:{...current.skills,microEvaluation:policy,runsPerArm:policy.runs,runtimeModule:'machinery/cli/micro-evaluation.mjs'}};
   const remaining=Math.max(0,(get('campaign-budget')?.seconds??Infinity)-(get('campaign-budget')?.elapsed??0));
   if(!remaining)throw Error('Campaign elapsed allocation exhausted');
   evolution.pending={active:current.skills.activeVersion,candidate,projects:new Set((evolution.history??[]).filter(t=>t.kind==='original'&&t.completed).map(t=>t.projectId)).size,
    projectIds:projectIds(),runs:policy.runs,evaluationSeconds:Math.min(policy.totalSeconds,remaining),evaluationRoot:join(config.root,'skills','evaluations',randomUUID()),
    sourceInstanceRoot:config.root,config:nextConfig,allocations:[],...(evolution.pending?.resource?{resource:evolution.pending.resource}:{})};
   evolution.evaluationStatus='deferred';evolution.restartedEvaluations=[...(evolution.restartedEvaluations??[]),{id,at,archive:join(archives,id+'.json')}];
   await updateConfig(config.root,()=>nextConfig,{activate:()=>save(file,evolution),rollback:()=>writeFile(file,before)});
   return {restarted:true,candidate,evaluationRoot:evolution.pending.evaluationRoot,archived:join(archives,id+'.json')};
  }finally{ledger.close();}
 });
}
