import { spawn } from 'node:child_process';
import { open, mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import { State } from '../controller/state.mjs';
import { processIdentity, ownerStopped, attemptStopped } from '../controller/lifecycle.mjs';
import { nextWave, admitJob, recordOutcome, RecoveryQueue,pendingOutcomes,outcomeRecorded } from '../controller/recovery.mjs';
import { readConfig,updateConfig } from './instance.mjs';

const self=fileURLToPath(import.meta.url);
const policy=config=>({mode:config.mode,inferenceRecovery:config.recovery!==false,
  maxFailedRounds:config.failed_recovery_rounds??5,investigationSeconds:config.investigation_seconds??9600});
const stateFor=root=>new State(join(root,'state.sqlite'));
function control(state) {
  state.db.exec('CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY,value TEXT NOT NULL)');
  return {
    get:key=>JSON.parse(state.db.prepare('SELECT value FROM control WHERE key=?').get(key)?.value??'null'),
    set:(key,value)=>state.db.prepare('INSERT OR REPLACE INTO control(key,value) VALUES (?,?)').run(key,JSON.stringify(value))
  };
}
async function environment(config,path,event,exclude,resource,signal) {
  const module=await import(pathToFileURL(resolve(config.root,config.environment.module)));
  return module.createEnvironment(config.environment.options,{instancePath:path,event,exclude,resource,signal});
}
export async function operate(command,config,options={}) {
  if(options.signal?.aborted)throw new Error('Wave interrupted');
  config={...config,mode:options.mode==='revisit'?config.mode:(options.mode??config.mode),
    recovery:options['no-recovery']?false:config.recovery};
  if(!['phase1','phase2','inference'].includes(config.mode))throw new Error('Unknown execution mode');
  if(options.detach && command!=='stop') {
    await mkdir(join(config.root,'logs'),{recursive:true});
    const log=await open(join(config.root,'logs','controller.log'),'a',0o600);
    const child=spawn(process.execPath,[self,command,config.root,JSON.stringify({...options,detach:false,signal:undefined})],
      {detached:true,stdio:['ignore',log.fd,log.fd]});
    child.unref();await log.close();return {pid:child.pid,submitted:true};
  }
  const state=stateFor(config.root), ctl=control(state);
  let owned=false,watcher;
  const schedulerAbort=new AbortController();
  const abort=()=>{ctl.set('stop',true);schedulerAbort.abort(new Error('Wave interrupted'));};
  options.signal?.addEventListener('abort',abort,{once:true});
  try {
    if(command==='stop') {
      ctl.set('stop',true);
      for(const event of state.events(null).filter(e=>e.type==='evaluation-started')) {
        try{await operate('stop',await readConfig(event.data.instanceRoot));}
        catch(error){if(error.code!=='ENOENT')state.event(null,'evaluation-stop-error',{message:error.message});}
      }
      const {requests}=await import('./background.mjs');
      requests(state);
      state.db.prepare("UPDATE requests SET status='failed',error='Stopped before execution' WHERE status='queued'").run();
      for(const request of requests(state).filter(r=>r.status==='running')) {
        const owner=request.owner;
        if(owner?.start && owner?.boot && !await ownerStopped(owner))process.kill(owner.pid,'SIGTERM');
      }
      return {stop_requested:true};
    }
    const active=ctl.get('owner');
    if(active && !await ownerStopped(active))throw new Error('Instance controller is already running');
    const owner=await processIdentity();
    state.db.exec('BEGIN IMMEDIATE');
    try {
      if(JSON.stringify(ctl.get('owner'))!==JSON.stringify(active))throw new Error('Instance ownership changed');
      ctl.set('owner',owner);ctl.set('stop',false);state.db.exec('COMMIT');owned=true;
    }catch(error){state.db.exec('ROLLBACK');throw error;}
    watcher=setInterval(()=>{if(ctl.get('stop'))schedulerAbort.abort(new Error('Wave interrupted'));},100);
    if(options.signal?.aborted)abort();
    if((await readConfig(config.root)).mode!==config.mode)await updateConfig(config.root,current=>({...current,mode:config.mode}));
    for(const project of config.projects)if(!state.project(project.id))state.add(project);
    const wave=(ctl.get('wave')??0)+(command==='wave'?1:0);ctl.set('wave',wave);
    const queue=new RecoveryQueue(state,policy(config));
    if(queue.lessons().length) {
      try {
        const {ensureKnowledgeService}=await import('./knowledge.mjs');
        const {deliverLessons}=await import('./memory.mjs');
        await deliverLessons({config,state,queue,url:await ensureKnowledgeService(config,{signal:options.signal}),signal:options.signal});
      }catch(error){state.event(null,'memory-update-error',{message:error.message});}
    }
    const finalize=pendingOutcomes(state,policy(config));
    const jobs=[];
    for(const attempt of finalize)if(await ownerStopped(attempt.owner))
      jobs.push({attempt,projectId:attempt.project,resume:true,finalize:true});
    jobs.push(...(command==='resume'
      ? state.db.prepare('SELECT * FROM attempts WHERE outcome IS NULL').all().map(attempt=>({attempt,projectId:attempt.project,resume:true}))
      : nextWave(state,{wave,...policy(config)}).filter(job=>{
          const selected=config.curriculum?.[config.mode];
          return Array.isArray(selected)?selected.includes(job.projectId):(!job.project.spec.phase || job.project.spec.phase===config.mode);
        })));
    const access=await environment(config,join(config.root,'scheduler'),(type,data)=>state.event(null,type,data),async(_label,fn)=>fn(),undefined,schedulerAbort.signal);
    const running=new Set();
    const requested=options.concurrency??config.concurrency??1;
    const concurrency=requested==='auto'?Math.max(1,jobs.length):Number(requested);
    if(!Number.isSafeInteger(concurrency)||concurrency<1)throw new Error('Invalid concurrency');
    const launch=async(job,resource)=>{
      const admitted=job.resume?{...JSON.parse(await readFile(join(config.root,'attempts',job.attempt.id,'job.json'),'utf8')),...job}:admitJob(state,{projectId:job.projectId,wave,owner,resourceId:resource?.id,...policy(config)});
      const attempt=admitted.attempt;
      const latest=await readConfig(config.root);
      let skillVersion=admitted.skillVersion??latest.skills.activeVersion;
      if(!job.resume && admitted.kind!=='original') {
        try{
          const {candidateVersion}=await import('../controller/skill-evolution.mjs');
          skillVersion=candidateVersion(JSON.parse(await readFile(join(config.root,'skills','evolution.json'),'utf8')),skillVersion);
        }
        catch(error){if(error.code!=='ENOENT')throw error;}
      }
      if(!job.resume)state.event(attempt.id,'skills-selected',{version:skillVersion,kind:admitted.kind});
      const path=join(config.root,'attempts',attempt.id);await mkdir(path,{recursive:true});
      await writeFile(join(path,'job.json'),JSON.stringify({...admitted,wave:admitted.wave??wave,options:{...admitted.options,...options,signal:undefined},mode:admitted.mode??config.mode,skillVersion,resource:resource??admitted.resource}),{mode:0o600});
      const log=await open(join(path,'worker.log'),'a',0o600);
      const child=spawn(process.execPath,[self,'worker',config.root,attempt.id],{stdio:['ignore',log.fd,log.fd]});
      let complete=false;
      child.once('exit',()=>{complete=true;});
      child.once('error',error=>{state.event(attempt.id,'worker-spawn-error',{message:error.message});complete=true;});
      await log.close();
      while(!complete)await delay(50);
      const outcome=state.attempt(attempt.id).outcome;
      if(outcome)state.db.prepare('DELETE FROM resource_leases WHERE attempt=?').run(attempt.id);
      return {attempt:attempt.id,outcome};
    };
    const results=[];
    try {
    for(const job of jobs) {
      if(ctl.get('stop'))break;
      while(running.size>=concurrency)await Promise.race(running);
      if(job.finalize) {
        const pending=launch(job).then(value=>results.push(value));running.add(pending);
        pending.then(()=>running.delete(pending),()=>running.delete(pending));continue;
      }
      let resources=await access.resources({purpose:(job.project?.spec??state.project(job.projectId)?.spec)?.performance_goal?'performance':(job.project?.spec?.purpose??'ordinary'),target:config.target});
      const prior=job.resume?JSON.parse(await readFile(join(config.root,'attempts',job.attempt.id,'job.json'),'utf8')).resource:null;
      const choose=()=>resources.find(r=>r.available && (!config.resource_ids || config.resource_ids.includes(r.id)) && (job.resume?r.id===prior?.id:
        !state.db.prepare('SELECT 1 FROM resource_leases WHERE resource=?').get(r.id)));
      let resource=choose();
      while(!resource && running.size && !ctl.get('stop')) {
        await Promise.race(running);
        resources=await access.resources({purpose:(job.project?.spec??state.project(job.projectId)?.spec)?.performance_goal?'performance':(job.project?.spec?.purpose??'ordinary'),target:config.target});
        resource=choose();
      }
      if(!resource)continue;
      if(ctl.get('stop'))break;
      const pending=launch(job,resource).then(value=>results.push(value));
      running.add(pending);pending.then(()=>running.delete(pending),()=>running.delete(pending));
    }
    await Promise.all(running);
    }catch(error){
      try{ctl.set('stop',true);}finally{await Promise.allSettled([...running]);}
      throw error;
    }
    return {wave,results};
  } finally {clearInterval(watcher);options.signal?.removeEventListener('abort',abort);if(owned)ctl.set('owner',null);state.close();}
}

// A campaign only advances waves; every actor still uses operate and the same worker.
export async function campaign(config,options={}) {
  const waves=[],started=performance.now();
  const seconds=config.campaign_seconds;
  if(seconds!==undefined && (!Number.isFinite(seconds)||seconds<=0))throw new Error('Invalid campaign elapsed allocation');
  const abort=new AbortController(),stop=()=>abort.abort();
  options.signal?.addEventListener('abort',stop,{once:true});
  if(options.signal?.aborted)abort.abort();
  const timer=seconds===undefined?null:setTimeout(stop,seconds*1000);
  try {
    while(!abort.signal.aborted) {
      const current=await readConfig(config.root),state=stateFor(config.root);
      let stopped;
      try {
        stopped=control(state).get('stop');
      }finally{state.close();}
      if(stopped && waves.length)return {waves,reason:'stopped'};
      const result=await operate('wave',current,{...options,detach:false,signal:abort.signal});
      waves.push(result);
      if(!result.results.length) {
        const ledger=stateFor(config.root);
        try {
          const open=ledger.db.prepare('SELECT 1 FROM attempts WHERE outcome IS NULL LIMIT 1').get();
          return {waves,reason:open?'requires-resume':'complete-or-blocked'};
        }finally{ledger.close();}
      }
    }
    return {waves,reason:seconds!==undefined && (performance.now()-started)/1000>=seconds?'elapsed':'stopped'};
  }finally{clearTimeout(timer);options.signal?.removeEventListener('abort',stop);}
}

async function worker(root,id) {
  let config=await readConfig(root);const state=stateFor(root),attempt=state.attempt(id);
  const path=join(root,'attempts',id),job=JSON.parse(await readFile(join(path,'job.json'),'utf8'));
  config={...config,mode:job.mode??config.mode,recovery:job.options['no-recovery']?false:config.recovery};
  let env,bridge,url,stopping,interrupted=false,claimed=false;
  const abort=new AbortController(),ctl=control(state);
  const signal=()=>{
    if(interrupted)return;interrupted=true;abort.abort();process.exitCode=130;
    stopping=(async()=>{if(env)for(const e of state.events(id).filter(e=>e.type==='execution-handle')) {
      try{await env.stop(e.data.handle);}catch(error){state.event(id,'stop-observation-error',{message:error.message});}
    }})();
  };
  process.on('SIGTERM',signal);process.on('SIGINT',signal);
  const watcher=setInterval(()=>{if(ctl.get('stop'))signal();},100);
  async function finish() {
    const current=state.attempt(id);
    if(!outcomeRecorded(state,id,policy(config))) {
      const {findingsFor}=await import('./findings.mjs');
      const findings=await findingsFor({config,job:{...job,attempt},path,state,bridge,
        passed:current.outcome==='passed',signal:abort.signal});
      recordOutcome(state,{attemptId:id,wave:job.wave,passed:current.outcome==='passed',...findings,...policy(config)});
    }
    if(!url)url=await (await import('./knowledge.mjs')).ensureKnowledgeService(config,{signal:abort.signal});
    await (await import('./memory.mjs')).deliverLessons({config,state,queue:new RecoveryQueue(state,policy(config)),url,signal:abort.signal});
    state.event(id,'memory-updated',{});
    if(config.mode!=='phase1' && !config.skills?.tuningDisabled) {
      try {
        await (await import('./skills.mjs')).run(config,{signal:abort.signal,resource:job.resource,
          task:{id,projectId:attempt.project,kind:job.kind,completed:true,phase:config.mode,events:state.events(id)}});
        state.event(id,'skills-updated',{});
      }catch(error){if(abort.signal.aborted)throw error;state.event(id,'skill-update-error',{message:error.message});}
    }
    state.event(id,'postprocess-completed',{});
  }
  try {
    if(ctl.get('stop'))signal();
    const me=await processIdentity();
    if(!job.resume) {
      state.db.prepare('UPDATE attempts SET owner=? WHERE id=?').run(JSON.stringify(me),id);claimed=true;
    }
    env=await environment(config,path,(type,data)=>state.event(id,type,data),async(_label,fn)=>fn(),job.resource,abort.signal);
    if(job.finalize) {
      if(!await ownerStopped(attempt.owner))throw new Error('Previous postprocessor remains live');
      state.db.prepare('UPDATE attempts SET owner=? WHERE id=? AND owner=?').run(JSON.stringify(me),id,attempt.owner);claimed=true;
    }else if(job.resume){await state.resume(id,a=>attemptStopped(state,a,env),me);claimed=true;}
    if(interrupted)throw new Error('Worker interrupted');
    url=await (await import('./knowledge.mjs')).ensureKnowledgeService(config,{signal:abort.signal});
    const {createKdbBridge}=await import('../integrations/kdb-extension.mjs');
    bridge=createKdbBridge({url,attemptId:id,agentDir:path,target:config.target,event:(type,data)=>state.event(id,type,data)});
    const base=state.project(attempt.project).spec;
    let project={...base,actor_seconds:job.actor_seconds};
    if(!job.finalize && job.kind==='investigation') {
      const file=join(path,'project.json');
      if(job.resume)project=JSON.parse(await readFile(file,'utf8'));
      else {
        project=await (await import('./curriculum.mjs')).investigate({config,job,base,path,state,bridge,signal:abort.signal});
        await writeFile(file,JSON.stringify(project),{mode:0o600});
      }
    }
    if(!job.finalize) {
      const directory=join(root,'skills','versions',job.skillVersion);
      const skills=JSON.parse(await readFile(join(directory,'skill-map.json'),'utf8'));
      for(const scope in skills)skills[scope]=resolve(directory,skills[scope]);
      state.event(id,'task-constructed',{goal:project.goal,kind:job.kind,actor_seconds:project.actor_seconds});
      await (await import('../controller/pi.mjs')).runAttempt({state,attempt:state.attempt(id),project,attemptPath:path,
        signal:abort.signal,settleProject:job.kind!=='investigation',
        environmentFactory:ctx=>environment(config,path,ctx.event,ctx.exclude,job.resource,ctx.signal),
        ...await (await import('../controller/model.mjs')).modelFromProfile(config.actor),
        skillMap:skills,knowledge:bridge.knowledge,extensions:[bridge.extension]});
    }
    if(!interrupted)await finish();
  }catch(error) {
    state.event(id,'attempt-error',{message:error.message,stage:'worker'});
    if(!interrupted && claimed && !state.attempt(id).outcome) {
      const {operationsStopped}=await import('../controller/lifecycle.mjs');
      if(await operationsStopped(state,state.attempt(id),env??{})) {
        state.finish(id,'failed',{settleProject:job.kind!=='investigation'});
        try{await finish();}catch(failure){state.event(id,'postprocess-error',{message:failure.message});}
      }
    }
    throw error;
  }finally {
    clearInterval(watcher);await stopping;
    process.off('SIGTERM',signal);process.off('SIGINT',signal);state.close();
  }
}

if(process.argv[1]===self) {
  const [command,root,arg]=process.argv.slice(2);
  const operation=command==='worker'?worker(root,arg):readConfig(root).then(c=>(command==='campaign'?campaign:operate)(...(command==='campaign'?[c,JSON.parse(arg)]:[command,c,JSON.parse(arg)])));
  operation.then(value=>console.log(JSON.stringify(value))).catch(error=>{console.error(error.stack);process.exitCode=1;});
}
