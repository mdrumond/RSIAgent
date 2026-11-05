import { spawn } from 'node:child_process';
import { open, mkdir, readFile, writeFile, rename } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { join, resolve } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import { selectAdmission } from '../controller/dispatch.mjs';
import {campaignBudget} from '../controller/campaign-budget.mjs';
import { campaignProgress } from '../controller/progress.mjs';
import { State } from '../controller/state.mjs';
import { processIdentity, ownerStopped, attemptStopped, settleFailure, reconcileFailure, failureSettled } from '../controller/lifecycle.mjs';
import { nextWave, admitJob, recordOutcome, RecoveryQueue,pendingOutcomes,outcomeRecorded } from '../controller/recovery.mjs';
import { readConfig,updateConfig,importProject } from './instance.mjs';
import {CampaignStorage} from '../controller/storage.mjs';

const self=fileURLToPath(import.meta.url);
const policy=config=>({mode:config.mode,inferenceRecovery:config.recovery!==false,
  maxFailedRounds:config.failed_recovery_rounds??5,investigationSeconds:config.investigation_seconds??9600});
const stateFor=root=>new State(join(root,'state.sqlite'));
async function writeJob(path,value) {
  const temporary=path+'.'+randomUUID()+'.tmp';
  await writeFile(temporary,JSON.stringify(value),{mode:0o600,flag:'wx'});
  await rename(temporary,path);
}
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
export async function instanceStopped(config) {
  const state=stateFor(config.root),ctl=control(state);
  try {
    for(const key of ['owner','skill-writer']) {
      const owner=ctl.get(key);if(owner&&!await ownerStopped(owner))return false;
    }
    for(const row of state.db.prepare('SELECT owner FROM attempts').all())
      if(!await ownerStopped(row.owner))return false;
    return true;
  } finally {state.close();}
}

export async function operate(command,config,options={}) {
  if(options.signal?.aborted)throw new Error('Wave interrupted');
  config={...config,mode:options.mode==='revisit'?config.mode:(options.mode??config.mode),
    recovery:options['no-recovery']?false:config.recovery};
  if(!['phase1','phase2','inference'].includes(config.mode))throw new Error('Unknown execution mode');
  const state=stateFor(config.root), ctl=control(state);
  if(command==='resume'&&ctl.get('operation')==='campaign')options={...options,continuous:true};
  let owned=false,watcher,campaignClock,budgetTimer,budgetWatcher;
  const schedulerAbort=new AbortController();
  const backgroundAbort=new AbortController();
  const abort=()=>{ctl.set('stop',true);schedulerAbort.abort(new Error('Wave interrupted'));backgroundAbort.abort();};
  options.signal?.addEventListener('abort',abort,{once:true});
  try {
    if(command==='pause') {
      ctl.set('pause-at',Date.now());ctl.set('pause',true);
      return {pause_requested:true,grace_seconds:config.pause_grace_seconds??30};
    }
    if(command==='stop') {
      ctl.set('stop',true);
      for(const event of state.events(null).filter(e=>e.type==='evaluation-started')) {
        try{await operate('stop',await readConfig(event.data.instanceRoot));}
        catch(error){if(error.code!=='ENOENT')state.event(null,'evaluation-stop-error',{message:error.message});}
      }
      const {requests}=await import('./background.mjs');
      requests(state);
      state.db.prepare("UPDATE requests SET status='failed',error='Stopped before execution' WHERE status='queued'").run();
      for(const request of state.db.prepare("SELECT owner FROM requests WHERE status='running'").iterate()) {
        const owner=JSON.parse(request.owner);
        if(owner?.start&&owner?.boot&&!await ownerStopped(owner)) {
          await (await import('./background.mjs')).stopRequest(owner);
        }
      }
      return {stop_requested:true};
    }
    if(command!=='resume'&&ctl.get('pause'))return {reason:'paused'};
    for(const attempt of state.db.prepare('SELECT id,owner FROM attempts WHERE outcome IS NULL AND owner IS NOT NULL').iterate()) {
      const identity=JSON.parse(attempt.owner);
      if(identity?.pid&&identity.start&&identity.boot&&!await ownerStopped(identity))throw new Error('Previous worker remains active; observe or stop it before starting another controller: '+attempt.id);
    }
    const active=ctl.get('owner');
    if(active && !await ownerStopped(active))throw new Error('Instance controller is already running');
    const owner=await processIdentity();
    state.db.exec('BEGIN IMMEDIATE');
    try {
      if(JSON.stringify(ctl.get('owner'))!==JSON.stringify(active))throw new Error('Instance ownership changed');
      ctl.set('owner',owner);
      if(command!=='resume')ctl.set('operation',options.continuous?'campaign':'wave');
      ctl.set('stop',false);ctl.set('pause',false);state.db.exec('COMMIT');owned=true;
    }catch(error){state.db.exec('ROLLBACK');throw error;}
    if(options.continuous&&config.campaign_seconds!==undefined) {
      const cycle=command==='resume'?ctl.get('campaign-budget')?.cycle??randomUUID():randomUUID();
      campaignClock=campaignBudget(state,{seconds:config.campaign_seconds,cycle,boot:owner.boot});
      if(campaignClock.remaining<=0)return {reason:'elapsed'};
      const expire=()=>{
        if(campaignClock.remaining>0)budgetTimer=setTimeout(expire,Math.max(1,Math.ceil(campaignClock.remaining*1000)));
        else {campaignClock.stop();abort();}
      };
      budgetTimer=setTimeout(expire,Math.ceil(campaignClock.remaining*1000));
      budgetWatcher=setInterval(()=>campaignClock.checkpoint(),1000);
    }
    watcher=setInterval(()=>{
      if(ctl.get('stop')){schedulerAbort.abort(new Error('Wave interrupted'));backgroundAbort.abort();}
      if(ctl.get('pause')){campaignClock?.stop();backgroundAbort.abort(new Error('Instance paused'));}
    },100);
    if(options.signal?.aborted)abort();
    if((await readConfig(config.root)).mode!==config.mode)await updateConfig(config.root,current=>({...current,mode:config.mode}));
    for(const project of config.projects)if(!state.project(project.id))state.add(project);
    let wave=(ctl.get('wave')??0)+(command==='wave'?1:0);ctl.set('wave',wave);
    const queue=new RecoveryQueue(state,policy(config));
    if(config.storage) {
      const url=await (await import('./knowledge.mjs')).ensureKnowledgeService(config,{signal:schedulerAbort.signal});
      await (await import('../controller/storage-knowledge.mjs')).activateStorageBaseline(config,{state,url,signal:schedulerAbort.signal});
    }
    // Unknown failed execution is reconciled, never replayed through the actor.
    for(const attempt of state.db.prepare('SELECT * FROM attempts WHERE outcome IS NOT NULL').all()) {
      if(failureSettled(state,attempt.id) || !await ownerStopped(attempt.owner))continue;
      try {
        const path=join(config.root,'attempts',attempt.id);
        const job=JSON.parse(await readFile(join(path,'job.json'),'utf8'));
        const previous=await environment(config,path,(type,data)=>state.event(attempt.id,type,data),
          async(_label,fn)=>fn(),job.resource,schedulerAbort.signal);
        await reconcileFailure(state,attempt.id,previous);
      }catch(error){state.event(attempt.id,'settlement-error',{message:error.message});}
    }
    // A prior scheduler may have died before its worker-exit waiter released a lease.
    for(const attempt of state.db.prepare(`SELECT a.*,l.resource FROM resource_leases l
      JOIN attempts a ON a.id=l.attempt WHERE a.outcome IS NOT NULL`).all()) {
      schedulerAbort.signal.throwIfAborted();
      try {
        if(!failureSettled(state,attempt.id)||!queue.outcomeRecorded(attempt.id)||!await ownerStopped(attempt.owner))continue;
        const path=join(config.root,'attempts',attempt.id);
        const job=JSON.parse(await readFile(join(path,'job.json'),'utf8'));
        const previous=await environment(config,path,(type,data)=>state.event(attempt.id,type,data),
          async(_label,fn)=>fn(),job.resource,schedulerAbort.signal);
        if(await attemptStopped(state,attempt,previous)) {
          state.db.prepare('DELETE FROM resource_leases WHERE attempt=? AND resource=?').run(attempt.id,attempt.resource);
          state.event(attempt.id,'resource-lease-reclaimed',{resource:attempt.resource});
        }
      }catch(error){state.event(attempt.id,'lease-reconciliation-error',{message:error.message});}
    }
    if(queue.lessons().length) {
      try {
        const {ensureKnowledgeService}=await import('./knowledge.mjs');
        const {deliverLessons}=await import('./memory.mjs');
        await deliverLessons({config,state,queue,url:await ensureKnowledgeService(config,{signal:schedulerAbort.signal}),signal:schedulerAbort.signal});
      }catch(error){state.event(null,'memory-update-error',{message:error.message});}
    }
    const finalize=pendingOutcomes(state,policy(config));
    const jobs=[];
    for(const attempt of finalize)if(await ownerStopped(attempt.owner))
      jobs.push({attempt,projectId:attempt.project,resume:true,finalize:true});
    jobs.push(...(command==='resume'
      ? state.db.prepare('SELECT * FROM attempts WHERE outcome IS NULL').all()
          .map(attempt=>({attempt,projectId:attempt.project,resume:true}))
      : nextWave(state,{wave,...policy(config)}).filter(job=>{
          const selected=config.curriculum?.[config.mode];
          return Array.isArray(selected)?selected.includes(job.projectId):(!job.project.spec.phase || job.project.spec.phase===config.mode);
        })));
    for(const job of jobs)if(job.resume) {
      const manifest=JSON.parse(await readFile(join(config.root,'attempts',job.attempt.id,'job.json'),'utf8'));
      Object.assign(job,{...manifest,...job});
    }
    let access=await environment(config,join(config.root,'scheduler'),(type,data)=>state.event(null,type,data),async(_label,fn)=>fn(),undefined,schedulerAbort.signal);
    let environmentRevision=JSON.stringify(config.environment);
    const running=new Set();
    const requested=options.concurrency??config.concurrency??1;
    const concurrency=requested==='auto'?Math.max(1,jobs.length):Number(requested);
    if(!Number.isSafeInteger(concurrency)||concurrency<1)throw new Error('Invalid concurrency');
    const contain=async(attempt,resource,kind,message,unclaimed=false)=>{
      const current=state.attempt(attempt.id);
      let stopped=unclaimed && current.owner===JSON.stringify(owner);
      if(!stopped) {
        try {stopped=await ownerStopped(current.owner);}
        catch(error){state.event(attempt.id,'owner-observation-error',{message:error.message});}
      }
      if(!stopped) {
        state.event(attempt.id,'worker-containment-blocked',{reason:'Current owner is not confirmed stopped'});
        return false;
      }
      let previous;
      try {
        previous=await environment(config,join(config.root,'attempts',attempt.id),
          (type,data)=>state.event(attempt.id,type,data),async(_label,fn)=>fn(),resource,schedulerAbort.signal);
      }catch(error){state.event(attempt.id,'settlement-error',{message:error.message});}
      if(state.attempt(attempt.id).owner!==current.owner) {
        state.event(attempt.id,'worker-containment-blocked',{reason:'Owner changed during closure observation'});
        return false;
      }
      // A recovery investigation does not settle the original project as passed.
      return settleFailure(state,attempt.id,previous,{phase:'controller',kind,message});
    };
    const launch=async(job,resource)=>{
      let attempt,assignedResource,child,path;
      try {
      const prepare=async()=>{
      let pinnedMode=job.mode??config.mode;
      if(job.origin_attempt&&!job.resume)pinnedMode=JSON.parse(await readFile(join(config.root,'attempts',job.origin_attempt,'job.json'),'utf8')).mode;
      const admitted=job.resume?{...JSON.parse(await readFile(join(config.root,'attempts',job.attempt.id,'job.json'),'utf8')),...job}:admitJob(state,{projectId:job.projectId,wave,owner,resourceId:resource?.id,...policy({...config,mode:pinnedMode})});
      admitted.mode=pinnedMode;
      attempt=admitted.attempt;
      assignedResource=resource??admitted.resource;
      if(job.origin_attempt&&!job.resume) {
        const parent=JSON.parse(await readFile(join(config.root,'attempts',job.origin_attempt,'job.json'),'utf8'));
        admitted.mode=parent.mode;
      }
      if(job.failure_pattern) {
        const {claimFailureInvestigation}=await import('../controller/failure-patterns.mjs');
        if(claimFailureInvestigation(state,job.failure_pattern))admitted.failure_pattern=job.failure_pattern;
      }
      path=join(config.root,'attempts',attempt.id);await mkdir(path,{recursive:true});
      // Retain a finalization manifest even if subsequent launch preparation fails.
      await writeJob(join(path,'job.json'),{...admitted,wave:admitted.wave??wave,
        options:{...admitted.options,...options,signal:undefined},mode:admitted.mode??config.mode,
        skillVersion:admitted.skillVersion??config.skills.activeVersion,resource:assignedResource,
        deferFinalization:options.continuous&&!job.finalize});
      const latest=await readConfig(config.root);
      if(latest.storage) {
        const mapped=state.db.prepare("SELECT revision,decision FROM storage_skill_versions WHERE version=?").get(latest.skills.activeVersion);
        const active=state.db.prepare("SELECT revision FROM storage_revisions WHERE name='skills'").get();
        if(!mapped||mapped.decision!=='promoted'||mapped.revision!==active?.revision)
          await (await import('../controller/storage-knowledge.mjs')).initializeStorageKnowledge(latest,state);
      }
      let skillVersion=admitted.skillVersion??latest.skills.activeVersion;
      if(!job.resume && admitted.kind!=='original') {
        try{
          const {candidateVersion}=await import('../controller/skill-evolution.mjs');
          skillVersion=candidateVersion(JSON.parse(await readFile(join(config.root,'skills','evolution.json'),'utf8')),skillVersion);
        }
        catch(error){if(error.code!=='ENOENT')throw error;}
      }
      if(!job.resume)state.event(attempt.id,'skills-selected',{version:skillVersion,kind:admitted.kind});
      const storage=CampaignStorage.open(state,config.root);
      if(storage) {
        storage.attempt(attempt.id,{phase:admitted.mode??config.mode,kind:admitted.kind,
          parent:job.origin_attempt??null,wave:admitted.wave??wave,
          skillRevision:storage.db.prepare("SELECT revision FROM storage_revisions WHERE name='skills'").get()?.revision??skillVersion});
        await storage.register(join(path,'worker.log'),{attempt:attempt.id,project:attempt.project,kind:'worker-log',role:'controller'});
      }
      await writeJob(join(path,'job.json'),{...admitted,wave:admitted.wave??wave,options:{...admitted.options,...options,signal:undefined},mode:admitted.mode??config.mode,skillVersion,resource:assignedResource,
        deferFinalization:options.continuous&&!job.finalize});
      return {argv:[process.execPath,self,'worker',config.root,attempt.id],cwd:config.root,
        env:{...process.env,...assignedResource?.env},
        stdout:join(path,'worker.log'),stderr:join(path,'worker.log')};
      };
      let exit;
      await prepare();
      const log=await open(join(path,'worker.log'),'a',0o600);
      child=spawn(process.execPath,[self,'worker',config.root,attempt.id],{stdio:['ignore',log.fd,log.fd],env:{...process.env,...assignedResource?.env}});
      const exited=new Promise(resolve=>{
        child.once('exit',(code,signal)=>resolve({code,signal}));
        child.once('error',error=>resolve({message:error.message}));
      });
      await log.close();exit=await exited;
      state.event(attempt.id,'worker-exited',exit);
      // Intentional stop preserves resumability and the lease.
      if(!ctl.get('stop') && !ctl.get('pause') && !state.attempt(attempt.id).outcome) {
        const closed=await contain(attempt,assignedResource,'worker-exit',JSON.stringify(exit),!job.resume &&
          (!child.pid || child.exitCode!==null || child.signalCode!==null));
        if(closed)state.db.prepare('DELETE FROM resource_leases WHERE attempt=?').run(attempt.id);
      }
      // Even an outcome is not proof that detached execution has stopped.
      if(state.attempt(attempt.id).outcome) {
        const previous=await environment(config,path,(type,data)=>state.event(attempt.id,type,data),
          async(_label,fn)=>fn(),assignedResource,schedulerAbort.signal);
        if(await attemptStopped(state,state.attempt(attempt.id),previous) && failureSettled(state,attempt.id))
          state.db.prepare('DELETE FROM resource_leases WHERE attempt=?').run(attempt.id);
      }
      return {attempt:attempt.id,outcome:state.attempt(attempt.id).outcome};
      }catch(error) {
        if(!attempt) {
          state.event(null,'job-error',{project:job.projectId,message:error.message});
          return {project:job.projectId,error:error.message};
        }
        const childStopped=!child || !child.pid || child.exitCode!==null || child.signalCode!==null;
        const closed=await contain(attempt,assignedResource,'worker-boundary',error.message,!job.resume && childStopped);
        if(closed && childStopped)
          state.db.prepare('DELETE FROM resource_leases WHERE attempt=?').run(attempt.id);
        return {attempt:attempt.id,outcome:state.attempt(attempt.id).outcome,error:error.message};
      }

    };
    const results=[],activeJobs=new Map(),startedJobs=new Set();
    const unavailable=new Set();let resourceWait;
    const keyFor=job=>job.attempt?`${job.finalize?'finalize':'attempt'}:${job.attempt.id}`:
      `${job.kind}:${job.projectId}:${job.kind==='original'?'':job.eligible_wave}`;
    let maintenance,nextMaintenance=0,generation,nextGeneration=0,progressSnapshot,storageLaunch;
    const compress=()=>{
      if(!config.storage||storageLaunch)return;
      storageLaunch=import('./storage-maintenance.mjs').then(({queueCompression})=>queueCompression(config))
        .catch(error=>state.event(null,'storage-maintenance-error',{message:error.message})).finally(()=>{storageLaunch=null;});
    };
    let freshSinceRecovery=ctl.get('fresh-since-recovery')??0;
    const track=(job,operation)=>{
      const key=keyFor(job);startedJobs.add(key);
      const pending=Promise.resolve().then(operation).then(value=>{
        if(value.deferred)startedJobs.delete(key);else results.push(value);
        if(!value.deferred)compress();
      });
      running.add(pending);activeJobs.set(pending,job);
      pending.finally(()=>{running.delete(pending);activeJobs.delete(pending);
        if(job.finalize)setTimeout(()=>startedJobs.delete(key),5000).unref();}).catch(()=>{});
    };
    const start=async(job,resource)=>{
      track(job,()=>launch(job,resource));
      if(!job.finalize) {
        freshSinceRecovery=job.kind==='original'?freshSinceRecovery+1:0;
        ctl.set('fresh-since-recovery',freshSinceRecovery);
      }
      return true;
    };
    try {
    for(;;) {
      if(ctl.get('stop') || ctl.get('pause') || schedulerAbort.signal.aborted)break;
      const latest=await readConfig(config.root);
      config={...latest,mode:config.mode,recovery:options['no-recovery']?false:latest.recovery};
      for(const project of config.projects)if(!state.project(project.id))state.add(project);
      const progress=campaignProgress(state,config),snapshot=JSON.stringify(progress);
      if(snapshot!==progressSnapshot){ctl.set('progress',progress);progressSnapshot=snapshot;}
      if(options.continuous&&progress.nextMode) {
        await updateConfig(config.root,current=>({...current,mode:progress.nextMode}));
        state.event(null,'phase-advanced',{from:config.mode,to:progress.nextMode});
        config={...config,mode:progress.nextMode};continue;
      }
      if(options.continuous&&!progress.phaseComplete&&config.mode==='phase1'&&config.curriculum?.generation?.enabled&&!generation&&Date.now()>=nextGeneration) {
        const generationConfig=config;
        generation=(async()=>{
          try {
            const {ensureKnowledgeService}=await import('./knowledge.mjs');
            const {createKdbBridge}=await import('../integrations/kdb-extension.mjs');
            const path=join(config.root,'curriculum','generation');await mkdir(path,{recursive:true});
            const bridge=createKdbBridge({url:await ensureKnowledgeService(generationConfig,{signal:backgroundAbort.signal}),agentDir:path,target:generationConfig.target,
              event:(type,data)=>state.event(null,type,{...data,role:'generation'})});
            const specs=await (await import('./project-generation.mjs')).refillPhase1({config:generationConfig,state,bridge,
              concurrency:Math.max(1,ctl.get('actor-capacity')??1),path,signal:backgroundAbort.signal});
            for(const spec of specs) {
              backgroundAbort.signal.throwIfAborted();
              if(state.project(spec.id))continue;
              const imported=await importProject(config.root,spec);
              await updateConfig(config.root,(current,ledger)=>{
                if(!ledger.project(imported.id))ledger.add(imported);
                return {...current,projects:[...current.projects.filter(p=>p.id!==imported.id),imported],
                  curriculum:{...current.curriculum,phase1:[...new Set([...(current.curriculum?.phase1??[]),imported.id])]}};
              });
            }
          }catch(error){if(!backgroundAbort.signal.aborted)state.event(null,'generation-error',{message:error.message});}
          finally{nextGeneration=Date.now()+5000;generation=null;}
        })();
      }
      if(options.continuous&&!maintenance&&Date.now()>=nextMaintenance&&queue.lessons().length) {
        maintenance=(async()=>{
          try {
            const {ensureKnowledgeService}=await import('./knowledge.mjs');
            const {deliverLessons}=await import('./memory.mjs');
            await deliverLessons({config,state,queue,url:await ensureKnowledgeService(config,{signal:backgroundAbort.signal}),signal:backgroundAbort.signal});
          }catch(error){state.event(null,'memory-update-error',{message:error.message});}
          finally{nextMaintenance=Date.now()+5000;maintenance=null;}
        })();
      }
      const revision=JSON.stringify(config.environment);
      if(revision!==environmentRevision) {
        access=await environment(config,join(config.root,'scheduler'),(type,data)=>state.event(null,type,data),async(_label,fn)=>fn(),undefined,schedulerAbort.signal);
        environmentRevision=revision;
      }
      const requestedNow=options.concurrency??config.concurrency??1;
      const limit=requestedNow==='auto'?Math.max(1,config.projects.length):Number(requestedNow);
      if(!Number.isSafeInteger(limit)||limit<1)throw new Error('Invalid concurrency');
      const selected=config.curriculum?.[config.mode];
      const admittedIds=new Set([...activeJobs.values()].map(keyFor));
      const maintenanceJobs=[];
      if(options.continuous)for(const attempt of state.db.prepare(`SELECT a.* FROM attempts a
        WHERE outcome IS NOT NULL AND NOT EXISTS(SELECT 1 FROM events e
          WHERE e.attempt=a.id AND e.type='postprocess-completed')`).all()) {
        if(!failureSettled(state,attempt.id)||!await ownerStopped(attempt.owner))continue;
        const manifest=JSON.parse(await readFile(join(config.root,'attempts',attempt.id,'job.json'),'utf8'));
        maintenanceJobs.push({...manifest,attempt,projectId:attempt.project,resume:true,finalize:true});
      }
      const candidates=command==='resume' && !options.continuous?jobs:
        [...maintenanceJobs,...jobs.filter(job=>job.resume),...nextWave(state,{wave:options.continuous?wave+1:wave,...policy({...config,mode:'phase1'})})];
      for(const job of candidates)if(['investigation','retry'].includes(job.kind)&&!job.mode&&job.origin_attempt)
        job.mode=JSON.parse(await readFile(join(config.root,'attempts',job.origin_attempt,'job.json'),'utf8')).mode;
      const eligible=candidates.filter(job=>{
        const key=keyFor(job);
        if(job.attempt&&[...activeJobs.values()].some(active=>active.attempt?.id===job.attempt.id))return false;
        if(['investigation','retry'].includes(job.kind)&&!policy({...config,mode:job.mode??config.mode}).inferenceRecovery&&job.mode==='inference')return false;
        if(options.continuous&&progress.phaseComplete&&job.kind==='original'&&!job.finalize&&!job.resume)return false;
        if(job.finalize&&state.queryEvents(job.attempt.id,{types:['postprocess-completed'],limit:1}).length)return false;
        if(['investigation','retry'].includes(job.kind)&&config.mode!=='phase1'&&!config.skills?.tuningDisabled&&
          state.db.prepare(`SELECT 1 FROM attempts a WHERE project=? AND outcome IS NOT NULL
            AND NOT EXISTS(SELECT 1 FROM events e WHERE e.attempt=a.id AND e.type='postprocess-completed') LIMIT 1`).get(job.projectId))return false;
        return !startedJobs.has(key)&&!admittedIds.has(key) && (job.finalize || ['investigation','retry'].includes(job.kind) ||
          (Array.isArray(selected)?selected.includes(job.projectId):(!job.project?.spec.phase || job.project.spec.phase===config.mode)));
      });
      const finalizer=eligible.find(job=>job.finalize);
      if(finalizer && ![...activeJobs.values()].some(job=>job.finalize)) {
        if(await start(finalizer))continue;
        await Promise.race([...running,delay(250)]);continue;
      }
      const activeActors=[...activeJobs.values()].filter(job=>!job.finalize);
      const job=selectAdmission(eligible.filter(job=>!job.finalize&&!unavailable.has(keyFor(job))),{running:activeActors,concurrency:limit,freshSinceRecovery});
      if(!job) {
        if(!running.size&&!options.continuous&&!resourceWait)break;
        if(options.continuous&&progress.done&&!running.size&&!maintenance&&!generation&&
          !eligible.some(job=>job.finalize||['investigation','retry'].includes(job.kind)))break;
        ctl.set('wait',resourceWait??{reason:running.size?'actor-capacity':'inputs-or-recovery',at:new Date().toISOString()});
        await Promise.race([...running,delay(250,undefined,{signal:schedulerAbort.signal})]);unavailable.clear();continue;
      }
      if(job.kind==='investigation') {
        const {failureInvestigations}=await import('../controller/failure-patterns.mjs');
        const recent=state.db.prepare(`SELECT a.id,a.project,a.outcome FROM attempts a
          WHERE outcome IS NOT NULL AND EXISTS(SELECT 1 FROM events e WHERE e.attempt=a.id AND e.type='postprocess-completed')
          ORDER BY (SELECT MAX(sequence) FROM events e WHERE e.attempt=a.id AND e.type='postprocess-completed') DESC LIMIT 10`).all().reverse();
        for(const attempt of recent){attempt.target=ctl.get('target-pool')??config.target;attempt.failures=state.failures(attempt.id,{order:'desc',limit:20}).map(e=>e.data);}
        const claimed=state.queryEvents(null,{types:['failure-investigation-claimed'],order:'desc',limit:1000}).map(e=>e.data);
        job.failure_pattern=failureInvestigations(recent,claimed).find(pattern=>pattern.attempts.some(a=>a.project===job.projectId));
      }
      let resources;
      try {
        resources=await access.resources({purpose:(job.project?.spec??state.project(job.projectId)?.spec)?.performance_goal?'performance':(job.project?.spec?.purpose??'ordinary'),target:config.target});
        resourceWait=null;
      }catch(error){
        unavailable.add(keyFor(job));
        resourceWait={reason:error.code==='MEMORY_ENFORCEMENT_UNAVAILABLE'?'memory-enforcement':'resources',
          code:error.code,message:error.message,project:job.projectId,at:new Date().toISOString()};
        ctl.set('wait',resourceWait);continue;
      }
      ctl.set('actor-capacity',Math.min(resources.filter(r=>r.available).length,limit));
      if(resources[0]?.metadata?.pool)ctl.set('target-pool',resources[0].metadata.pool);
      const prior=job.resume?JSON.parse(await readFile(join(config.root,'attempts',job.attempt.id,'job.json'),'utf8')).resource:null;
      const choose=()=>resources.find(r=>r.available && (!config.resource_ids || config.resource_ids.includes(r.id)) && (job.resume?r.id===prior?.id:
        !state.db.prepare('SELECT 1 FROM resource_leases WHERE resource=?').get(r.id)));
      const resource=choose();
      if(!resource) {
        unavailable.add(keyFor(job));
        ctl.set('wait',{reason:'resources',project:job.projectId,at:new Date().toISOString()});
        continue;
      }
      if(ctl.get('stop'))break;
      if(options.continuous && job.kind!=='original' && job.eligible_wave>wave){wave=job.eligible_wave;ctl.set('wave',wave);}
      ctl.set('wait',null);
      if(!await start(job,resource))await Promise.race([...running,delay(250)]);
    }
    await Promise.all(running);
    if(maintenance)await maintenance;
    if(generation)await generation;
    compress();if(storageLaunch)await storageLaunch;
    }catch(error){
      const interrupted=schedulerAbort.signal.aborted;
      if(!interrupted)state.event(null,'scheduler-error',{message:error.message});
      try{ctl.set('stop',true);}finally{await Promise.allSettled([...running]);}
      if(maintenance)await maintenance;
      if(generation)await generation;
      if(!interrupted)throw error;
    }
    return {wave,results};
  } finally {clearInterval(watcher);clearInterval(budgetWatcher);clearTimeout(budgetTimer);campaignClock?.stop();options.signal?.removeEventListener('abort',abort);if(owned)ctl.set('owner',null);state.close();}
}

// A campaign only advances waves; every actor still uses operate and the same worker.
export async function campaign(config,options={}) {
  const waves=[];
  const seconds=config.campaign_seconds;
  if(seconds!==undefined && (!Number.isFinite(seconds)||seconds<=0))throw new Error('Invalid campaign elapsed allocation');
  const abort=new AbortController(),stop=()=>abort.abort();
  options.signal?.addEventListener('abort',stop,{once:true});
  if(options.signal?.aborted)abort.abort();
  try {
    const result=await operate(options.resume?'resume':'wave',await readConfig(config.root),{...options,continuous:true,detach:false,signal:abort.signal});
    waves.push(result);
    const ledger=stateFor(config.root);
    let stopped,paused,elapsed;
    try{stopped=control(ledger).get('stop');paused=control(ledger).get('pause');elapsed=control(ledger).get('campaign-budget')?.elapsed;}finally{ledger.close();}
    return {waves,reason:seconds!==undefined&&elapsed>=seconds?'elapsed':
      paused?'paused':abort.signal.aborted||stopped?'stopped':'complete'};
  }finally{options.signal?.removeEventListener('abort',stop);}
}

export async function resume(config,options={}) {
  const state=stateFor(config.root);
  let continuous;
  try{continuous=control(state).get('operation')==='campaign';}finally{state.close();}
  return continuous?campaign(config,{...options,resume:true}):operate('resume',config,options);
}

async function worker(root,id) {
  let config=await readConfig(root);const state=stateFor(root),attempt=state.attempt(id);
  const path=join(root,'attempts',id),job=JSON.parse(await readFile(join(path,'job.json'),'utf8'));
  config={...config,mode:job.mode??config.mode,recovery:job.options['no-recovery']?false:config.recovery};
  let env,bridge,url,stopping,interrupted=false,claimed=false,actorRunning=false;
  const abort=new AbortController(),ctl=control(state);
  const signal=()=>{
    if(interrupted)return;interrupted=true;abort.abort();process.exitCode=130;
    stopping=(async()=>{if(env)for(const e of state.events(id).filter(e=>e.type==='execution-handle')) {
      try{await env.stop(e.data.handle);}catch(error){state.event(id,'stop-observation-error',{message:error.message});}
    }})();
  };
  process.on('SIGTERM',signal);process.on('SIGINT',signal);
  const grace=config.pause_grace_seconds??30;
  if(!Number.isFinite(grace)||grace<0)throw new Error('Invalid pause grace elapsed allocation');
  const watcher=setInterval(()=>{if(ctl.get('stop') || (ctl.get('pause')&&
    (!actorRunning||Date.now()-(ctl.get('pause-at')??0)>=grace*1000)))signal();},100);
  async function finish() {
    const current=state.attempt(id);
    if(!failureSettled(state,id))return;
    if(current.outcome==='failed' && !state.failures(id,{limit:1}).length)
      state.recordFailure(id,{phase:'actor',kind:'failed-outcome',message:'Actor completed without acceptance'});
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
          task:{id,projectId:attempt.project,kind:job.kind,completed:true,phase:config.mode,events:state.queryEvents(id,{types:['skill-read','knowledge-retrieved','guidance-contradicted','failure-recorded',
            'verification','rejected','stage-completed'],limit:1000,order:'desc'}).reverse()}});
        state.event(id,'skills-updated',{});
      }catch(error){state.event(id,'skill-update-error',{message:error.message});throw error;}
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
    const storage=CampaignStorage.open(state,config.root);
    const pinned=storage?.db.prepare('SELECT knowledge_generation FROM storage_attempts WHERE attempt=?').get(id);
    bridge=createKdbBridge({url,attemptId:id,agentDir:path,target:config.target,generation:pinned?.knowledge_generation,
      recordRetrieval:storage?entry=>storage.retrieval({...entry,attempt:id}):undefined,event:(type,data)=>state.event(id,type,data)});
    const base=state.project(attempt.project).spec;
    let project={...base,actor_seconds:job.actor_seconds};
    if(!job.finalize && job.kind==='investigation') {
      const file=join(path,'project.json');
      let constructed;
      try{constructed=JSON.parse(await readFile(file,'utf8'));}
      catch(error){if(error.code!=='ENOENT')throw error;}
      if(constructed)project=constructed;
      else {
        if(state.attempt(id).session || state.events(id).some(e=>e.type==='task-constructed'))
          throw new Error('Constructed investigation project is missing; cannot reconstruct an actor attempt');
        project=await (await import('./curriculum.mjs')).investigate({config,job,base,path,state,bridge,signal:abort.signal});
        await writeFile(file,JSON.stringify(project),{mode:0o600});
      }
    }
    if(!job.finalize) {
      const directory=join(root,'skills','versions',job.skillVersion);
      const skills=JSON.parse(await readFile(join(directory,'skill-map.json'),'utf8'));
      for(const scope in skills)skills[scope]=resolve(directory,skills[scope]);
      state.event(id,'task-constructed',{goal:project.goal,kind:job.kind,actor_seconds:project.actor_seconds});
      actorRunning=true;
      try {await (await import('../controller/pi.mjs')).runAttempt({state,attempt:state.attempt(id),project,attemptPath:path,
        signal:abort.signal,settleProject:job.kind!=='investigation',
        shouldPause:()=>Boolean(ctl.get('pause')),onPause:signal,
        environmentFactory:ctx=>environment(config,path,ctx.event,ctx.exclude,job.resource,ctx.signal),
        ...await (await import('../controller/model.mjs')).modelFromProfile(config.actor),
        skillMap:skills,knowledge:bridge.knowledge,extensions:[bridge.extension]});
      }finally{actorRunning=false;}
    }
    if(!interrupted) {
      if(state.attempt(id).outcome==='failed')
        await settleFailure(state,id,env,{phase:'actor',kind:'failed-outcome',
          message:'Actor completed without acceptance',settleProject:job.kind!=='investigation'});
      if(!job.deferFinalization)await finish();
    }
  }catch(error) {
    state.event(id,'attempt-error',{message:error.message,stage:'worker'});
    if(!interrupted && claimed) {
      if(!state.attempt(id).outcome || (state.attempt(id).outcome==='failed' && !failureSettled(state,id))) {
        await settleFailure(state,id,env,{message:error.message,settleProject:job.kind!=='investigation'});
      }else state.recordFailure(id,{phase:'postprocess',kind:'exception',message:error.message});
      if(failureSettled(state,id)&&!job.deferFinalization) {
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
