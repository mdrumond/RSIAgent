#!/usr/bin/env node
import {parseArgs} from 'node:util';
import {readFile,mkdir,writeFile,rename,appendFile,statfs,stat} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {randomUUID} from 'node:crypto';
import {setTimeout as delay} from 'node:timers/promises';
import {State} from '../controller/state.mjs';
import {campaignElapsed} from '../controller/campaign-budget.mjs';
import {processIdentity,ownerStopped} from '../controller/lifecycle.mjs';
import {readConfig} from './instance.mjs';
import {qualificationGate,campaignEntryConfig} from './campaign-manifest.mjs';
import {generationSettings} from '../controller/progress.mjs';
import {assertPackAvailable} from './storage.mjs';
import {memorySettings,memoryDecision,sampleCampaignMemory,writeMemoryControl,requestMemoryPause,releaseKnowledgeService} from './campaign-memory.mjs';

const GiB=1024**3,ownerKey='supervisor-owner';
const writerKeys=['owner','skill-writer','memory-writer','storage-maintenance','knowledge-startup'];
const validOwner=owner=>Number.isInteger(owner?.pid)&&owner.pid>0&&typeof owner.start==='string'&&owner.start&&typeof owner.boot==='string'&&owner.boot;

export async function claimSupervisor(config,owner,{stopped=ownerStopped}={}) {
  if(!validOwner(owner))throw new Error('Supervisor process identity required');
  if(!(await stat(join(config.root,'state.sqlite'))).isFile())throw new Error('Existing campaign ledger required');
  const state=new State(join(config.root,'state.sqlite')),token=JSON.stringify(owner);
  try {
    state.db.exec('CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
    const previous=state.db.prepare('SELECT value FROM control WHERE key=?').get(ownerKey)?.value??null;
    if(previous&&JSON.parse(previous)&&(!validOwner(JSON.parse(previous))||!await stopped(JSON.parse(previous))))
      throw new Error('Previous supervisor remains live or unresolved');
    state.db.exec('BEGIN IMMEDIATE');
    try {
      await assertPackAvailable(state);
      if((state.db.prepare('SELECT value FROM control WHERE key=?').get(ownerKey)?.value??null)!==previous)
        throw new Error('Supervisor ownership changed');
      state.db.prepare('INSERT OR REPLACE INTO control VALUES (?,?)').run(ownerKey,token);state.db.exec('COMMIT');
    }catch(error){state.db.exec('ROLLBACK');throw error;}
  }finally{state.close();}
  return async()=>{
    const state=new State(join(config.root,'state.sqlite'));
    try{state.db.prepare('DELETE FROM control WHERE key=? AND value=?').run(ownerKey,token);}finally{state.close();}
  };
}

// Restart checks only LOCAL owners. Native resume owns remote observation and retains fences.
export async function localOwnersStopped(config,{stopped=ownerStopped}={}) {
  const state=new State(join(config.root,'state.sqlite'));let observable=false,unknown=false,activeWork=false;
  try {
    const inspect=async(owner,work=false)=>{
      if(!owner)return;
      if(!validOwner(owner)){unknown=true;return;}
      if(!await stopped(owner)){observable=true;if(work)activeWork=true;}
    };
    if(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get())
      for(const {key,value} of state.db.prepare('SELECT key,value FROM control').all())
        if(writerKeys.includes(key))await inspect(JSON.parse(value),!['owner','knowledge-startup'].includes(key));
    for(const row of state.db.prepare('SELECT owner,outcome FROM attempts').all()) {
      if(!row.owner&&row.outcome===null)unknown=true;
      await inspect(row.owner?JSON.parse(row.owner):null,true);
    }
    if(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='requests'").get())
      for(const row of state.db.prepare('SELECT kind,status,owner FROM requests').all()) {
        if(!row.owner&&['queued','running'].includes(row.status))unknown=true;
        await inspect(row.owner?JSON.parse(row.owner):null,!['campaign','resume'].includes(row.kind));
      }
    return {stopped:!observable&&!unknown,observable,unknown,activeWork};
  }finally{state.close();}
}

// Foreground work only: perpetual A/B and controller/health poll telemetry must
// not disguise a campaign with no runnable originals. Curriculum elapsed is a
// bounded construction checkpoint, not the campaign wall clock.
export async function campaignActivity(state,{stopped=ownerStopped}={}) {
  const table=name=>Boolean(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name=?").get(name));
  const milestone=state.db.prepare(`SELECT MAX(sequence) n FROM events WHERE type IN
    ('job-admitted','stage-completed','postprocess-completed','campaign-entry-started','campaign-entry-completed')`).get().n;
  const projects=state.db.prepare('SELECT COUNT(*) n FROM projects').get().n;
  const lessons=table('recovery_lessons')?state.db.prepare('SELECT COUNT(*) n FROM recovery_lessons WHERE recorded=1').get().n:0;
  const drafts=table('generation_drafts')?state.db.prepare(`SELECT id,
    json_extract(data,'$.status') status,
    MIN(COALESCE(json_extract(data,'$.elapsed'),0),COALESCE(json_extract(data,'$.seconds'),0)) elapsed
    FROM generation_drafts ORDER BY id`).all():[];
  const controller=table('control')?state.db.prepare("SELECT value FROM control WHERE key='owner'").get()?.value:null;
  let active=false;
  for(const {owner} of state.db.prepare(`SELECT owner FROM attempts a WHERE outcome IS NULL OR
      NOT EXISTS (SELECT 1 FROM events e WHERE e.attempt=a.id AND e.type='postprocess-completed')`).all()) {
    // Admission initially belongs to the controller until the worker claims it.
    // A stranded admission is not evidence of an active actor or device compile.
    if(!owner||owner===controller)continue;
    const identity=JSON.parse(owner);
    if(validOwner(identity)&&!await stopped(identity)){active=true;break;}
  }
  return {signature:JSON.stringify({milestone,projects,lessons,drafts}),active};
}

async function nativeSnapshot(config,cli) {
  const status=await cli(['status','--instance',config.root]);
  const state=new State(join(config.root,'state.sqlite'));
  try {
    const control=state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get();
    const value=key=>control?JSON.parse(state.db.prepare('SELECT value FROM control WHERE key=?').get(key)?.value??'null'):null;
    return {status,activity:await campaignActivity(state),budget:value('campaign-budget'),operation:value('operation'),
      paused:Boolean(value('pause')),memoryPause:value('memory-pause'),operatorStopped:Boolean(value('operator-stop')),
      attempts:state.db.prepare('SELECT COUNT(*) n FROM attempts').get().n,
      leaseBlocked:state.db.prepare("SELECT data,created FROM events WHERE type='device-lease-blocked' ORDER BY sequence DESC LIMIT 1").get()};
  }finally{state.close();}
}
async function diskBytes(config) {const value=await statfs(config.root);return value.bavail*value.bsize;}
async function load(config) {
  try{return JSON.parse(await readFile(join(config.root,'.supervisor','state.json'),'utf8'));}
  catch(error){if(error.code==='ENOENT')return null;throw error;}
}
async function save(config,report) {
  const root=join(config.root,'.supervisor');await mkdir(root,{recursive:true,mode:0o700});
  const temp=join(root,randomUUID()+'.json');await writeFile(temp,JSON.stringify(report,null,2)+'\n',{mode:0o600});
  await rename(temp,join(root,'status.json'));
  const checkpoint=join(root,randomUUID()+'.json');await writeFile(checkpoint,JSON.stringify(report)+'\n',{mode:0o600});
  await rename(checkpoint,join(root,'state.json'));
  const signature=JSON.stringify([report.state,report.reason,report.memory?.limit]);
  if(save.last?.root!==root||save.last.signature!==signature||report.at-save.last.at>=60000) {
    await appendFile(join(root,'events.jsonl'),JSON.stringify(report)+'\n',{mode:0o600});save.last={root,signature,at:report.at};
  }
}

export async function superviseCampaign({instance,cohort,qualification,sourceRevision,signal},dependencies={}) {
  const {read=readConfig,cli:providedCli,identity=processIdentity,claim=claimSupervisor,
    restart=localOwnersStopped,closed:providedClosed,
    space=diskBytes,now=()=>Number(process.hrtime.bigint())/1e6,wait=delay,
    persist=save,load:restore=load,snapshot:providedSnapshot,
    memorySample=sampleCampaignMemory,memoryControl=writeMemoryControl,memoryPause=requestMemoryPause,
    memoryRelease=releaseKnowledgeService}=dependencies;
  qualificationGate(qualification);
  if(sourceRevision!==qualification.expected.sourceRevision)throw new Error('Supervisor source revision does not match qualification');
  const config=await read(resolve(instance));
  if(!cohort||config.campaign_cohort!==cohort||config.campaign_source_revision!==sourceRevision||
    !config.campaign_plan||!(config.campaign_seconds>0))throw new Error('Qualified fresh campaign binding required');
  const cli=providedCli??(await import(pathToFileURL(join(config.root,'machinery','cli','main.mjs')))).main;
  const nativeClosed=providedClosed?null:(await import(pathToFileURL(join(config.root,'machinery','cli','runtime.mjs')))).instanceStopped;
  const closed=providedClosed??(async config=>await nativeClosed(config)&&(await localOwnersStopped(config)).stopped);
  const snapshot=providedSnapshot??(config=>nativeSnapshot(config,cli));
  const owner=await identity(),release=await claim(config,owner);
  let blockedSince=null,nextReconcile=0,lastLaunch=null,report,memory,progressWatch;
  const startedWall=Date.now();
  const settings=config.local_memory?memorySettings(config.local_memory):null,known=new Map();
  const publish=async(state,reason,extra={})=>{
    report={format:1,at:now(),observedAt:new Date().toISOString(),owner,state,reason,sourceRevision,
      ...(progressWatch?{progressWatch:{...progressWatch,idleElapsed:Math.max(0,now()-progressWatch.since)}}:{}),
      blockedSince,blockedElapsed:blockedSince===null?0:Math.max(0,now()-blockedSince),...(memory?{memory}:{}),...extra};
    await persist(config,report);return report;
  };
  const pause=async reason=>{
    try{await cli(['pause','--instance',config.root]);}
    catch(error){return publish('attention-needed',reason,{checkpointPending:true,errorType:error.name});}
    const until=now()+((config.pause_grace_seconds??30)+60)*1000;
    for(;;) {
      try{if(await closed(config))break;}
      catch(error){return publish('attention-needed',reason,{checkpointPending:true,errorType:error.name});}
      if(now()>=until)return publish('attention-needed',reason,{checkpointPending:true});
      await publish('checkpointing',reason);await wait(1000);
    }
    return publish(reason==='complete'?'complete':reason==='signal'?'paused':'attention-needed',reason,{checkpointPending:false});
  };
  try {
    const prior=await restore(config);
    if(settings&&prior?.sourceRevision===sourceRevision&&prior.memory) {
      memory={...prior.memory,lowSince:null};
      if(prior.owner?.boot!==owner.boot){memory.highSince=null;memory.lastReduction=null;memory.phaseAt=now();}
      await memoryControl(config,memory);
    }
    // An explicit checkpoint ends the running watchdog window. A resumed campaign
    // gets a fresh window; running/crashed supervision retains its elapsed stall.
    const checkpointed=['checkpointing','paused','attention-needed','cooling','complete'].includes(prior?.state);
    // Explicit resume after a minimum-cap pause gets a fresh admission decision,
    // including after a qualified source upgrade. Do not thaw a cooling token.
    let recheckMinimum=checkpointed&&prior.reason==='memory-at-minimum'&&prior.memory?.limit===1&&!prior.memory.token;
    let minimumRecoveryPending=false;
    if(prior?.sourceRevision===sourceRevision&&Number.isFinite(prior.blockedSince)&&!checkpointed)
      blockedSince=prior.owner?.boot===owner.boot?Math.min(prior.blockedSince,now()):now()-(prior.blockedElapsed??0);
    if(prior?.sourceRevision===sourceRevision&&prior.progressWatch&&!checkpointed) {
      const previous=prior.progressWatch;
      progressWatch={...previous,since:prior.owner?.boot===owner.boot
        ?Math.min(previous.since,now()):now()-(previous.idleElapsed??0)};
    }
    for(;;) {
      if(signal?.aborted)return await pause('signal');
      try {
        const current=await read(config.root);
        if(current.campaign_cohort!==cohort||current.campaign_source_revision!==sourceRevision||current.campaign_seconds!==config.campaign_seconds)
          return await pause('campaign-binding-changed');
        const view=await snapshot(config),status=view.status;
        if(view.leaseBlocked&&Date.parse(view.leaseBlocked.created)>=startedWall)return await pause('device-lease-blocked');
        const used=campaignElapsed(view.budget,{boot:owner.boot,now});
        const latestRequest=status.requests?.find(request=>request.id===lastLaunch?.request)??
          status.requests?.find(request=>['campaign','resume'].includes(request.kind));
        if(status.progress?.done)return await pause('complete');
        if(used>=config.campaign_seconds)return await pause('elapsed');
        if(view.paused&&(!memory?.token||view.memoryPause!==memory.token))return await pause('operator-pause');
        if(view.operatorStopped||(status.controller?.stopRequested&&latestRequest?.status==='passed'))return await pause('operator-stop');
        if(settings) {
          const minimumWait=status.wait?.reason==='memory-concurrency-minimum';
          if(!minimumWait)minimumRecoveryPending=false;
          // A completed resume may have survived a supervisor crash before its acknowledgement.
          if(memory?.resumePending&&!view.paused&&status.controller?.state==='running') {
            if(progressWatch)progressWatch.since=now();
            memory={...memory,token:null,resumePending:false,recovered:true,lowSince:null,highSince:null,lastReduction:null};
            await memoryControl(config,memory);await publish('running','memory-recovered');
          }
          let measured;
          try{measured=await memorySample(config,owner,{known});}
          catch(error){memory={...memory,limit:memory?.limit??Number(current.concurrency??4),blocked:true};await memoryControl(config,memory);throw error;}
          if(recheckMinimum&&!view.paused&&status.controller?.state==='running'&&
              measured.processCount>0&&measured.rssBytes<settings.lowBytes&&Number(current.concurrency??4)>=2) {
            // Retain one background slot and permit one foreground worker,
            // rather than immediately restoring the full configured maximum.
            memory={...memory,limit:2,recovered:true};
            recheckMinimum=false;minimumRecoveryPending=true;
          }
          memory={...memoryDecision(memory,{rssBytes:measured.rssBytes,now:now(),maximum:Number(current.concurrency??4)},settings),...measured};
          if(minimumWait&&!(minimumRecoveryPending&&memory.limit>1&&memory.action==='run')&&memory.action!=='emergency') {
            memory={...memory,limit:1,blocked:true};await memoryControl(config,memory);
            return await pause('memory-at-minimum');
          }
          await memoryControl(config,memory);
          if(memory.action==='emergency')return await pause('memory-emergency');
          if(['checkpoint','exhausted'].includes(memory.action)) {
            const exhausted=memory.action==='exhausted';
            memory.token=randomUUID();memory.phaseAt=now();await publish('checkpointing','memory-pressure');
            await memoryPause(config,memory.token);
            const until=now()+((config.pause_grace_seconds??30)+60)*1000;
            while(!await closed(config)) {
              if(now()>=until)return await publish('attention-needed','memory-checkpoint-timeout',{checkpointPending:true});
              await wait(settings.pollMs);const sample=await memorySample(config,owner,{known});
              memory={...memory,...sample};await publish('checkpointing','memory-pressure');
              if(sample.rssBytes>=settings.hardBytes)return await publish('attention-needed','memory-emergency',{checkpointPending:true});
            }
            try{await memoryRelease(config);}
            catch(error){return await publish('attention-needed','memory-service-release',{checkpointPending:true,errorType:error.name});}
            if(exhausted)return await publish('attention-needed','memory-at-minimum',{checkpointPending:false});
            memory.lowSince=null;if(progressWatch)progressWatch.since=now();await publish('cooling','memory-pressure');continue;
          }
          if(memory.token) {
            if(progressWatch)progressWatch.since=now();
            if(!view.paused||view.memoryPause!==memory.token)return await pause('memory-pause-ownership');
            if(now()-(memory.phaseAt??now())>=1800000)return await publish('attention-needed','memory-recovery-timeout');
            if(memory.resumePending) {
              const candidates=(status.requests??[]).filter(request=>request.kind==='resume'&&!(memory.priorRequests??[]).includes(request.id));
              const request=memory.resumeRequest?(status.requests??[]).find(request=>request.id===memory.resumeRequest):candidates.length===1?candidates[0]:null;
              if(!request)return await publish('attention-needed','memory-resume-unresolved');
              memory.resumeRequest=request.id;
              if(['failed','passed'].includes(request.status))return await publish('attention-needed','memory-resume-failed');
              await publish('observing','memory-resume',{request:request.id});
            }else if(memory.action==='resume') {
              if(!await closed(config))return await publish('attention-needed','memory-ownership',{checkpointPending:true});
              // A restored cooling checkpoint may precede service shutdown; recheck before every resume.
              try{await memoryRelease(config);}
              catch(error){return await publish('attention-needed','memory-service-release',{checkpointPending:true,errorType:error.name});}
              const ready=await snapshot(config);
              if(ready.operatorStopped)return await pause('operator-stop');
              if(!ready.paused||ready.memoryPause!==memory.token)return await pause('operator-pause');
              // Persist intent before dispatch. Observe a retained queued request after interruption, never replay it.
              memory.resumePending=true;memory.priorRequests=(status.requests??[]).map(request=>request.id);memory.phaseAt=now();
              await publish('launching','memory-resume');
              lastLaunch=await cli(['resume','--instance',config.root,'--detach','--memory-pause-token',memory.token]);
              memory.resumeRequest=lastLaunch?.request;
              await publish('observing','memory-resume',{request:memory.resumeRequest});
            }else await publish('cooling','memory-pressure');
            await wait(settings.pollMs);continue;
          }
        }
        if(status.controller?.state!=='running'&&latestRequest?.status==='failed'&&blockedSince===null)blockedSince=now();
        const available=await space(config),started=Boolean(view.budget?.cycle||view.operation==='campaign'||view.attempts||lastLaunch||latestRequest);
        if(!Number.isFinite(available)||available<0)throw new Error('Disk availability unknown');
        if(available<(started?16:64)*GiB) {
          if(started)return await pause('low-disk');
          return await publish('attention-needed','initial-disk',{availableBytes:available});
        }
        if(status.controller?.state==='unknown')throw new Error('Unresolved controller ownership');
        if(status.controller?.state==='running')blockedSince=null;
        if(blockedSince!==null&&now()-blockedSince>=1800000)return await pause('infrastructure-or-ownership');
        if(status.controller?.state==='running') {
          const activity=view.activity??{signature:JSON.stringify(status.progress??{}),
            active:(status.attempts??[]).some(attempt=>attempt.outcome===null)};
          const entry=status.progress?.currentEntry;
          const policy=entry?campaignEntryConfig(current,entry.id):current;
          const waitingExternal=status.wait?.reason==='waiting-inputs'&&entry?.remainingAdmissions>0&&
            !generationSettings(policy,entry.phase).enabled;
          if(!progressWatch||activity.active||waitingExternal||activity.signature!==progressWatch.signature)
            progressWatch={signature:activity.signature,since:now()};
          if(now()-progressWatch.since>=1800000) {
            return await pause(entry?.remainingAdmissions===0&&entry.remainingOriginals>0
              ?'exhausted-admissions-unsettled-originals':'campaign-no-progress');
          }
          await publish('running',status.wait?.reason??'active',{progress:{currentEntry:status.progress?.currentEntry?.id??null,
            finishedOriginals:status.progress?.finishedOriginals??0,target:status.progress?.target??null},elapsedSeconds:used,availableBytes:available});
        }else if(now()>=nextReconcile) {
          nextReconcile=now()+300000;
          const ownership=await restart(config);
          if(ownership.observable) {
            if(ownership.activeWork)blockedSince=null;
            else if(blockedSince===null)blockedSince=now();
            await publish('observing',ownership.activeWork?'local-workers':'controller-startup');
          }
          else if(!ownership.stopped)throw new Error('Unresolved local ownership');
          else {
            if(view.attempts&&view.operation!=='campaign'&&!view.budget?.cycle)return await publish('attention-needed','not-a-campaign');
            const command=view.budget?.cycle||view.operation==='campaign'?'resume':'campaign';
            await publish('launching',command,{elapsedSeconds:used});
            lastLaunch=await cli([command,'--instance',config.root,'--detach']);
            await publish('observing','controller-request',{request:lastLaunch?.request??null,elapsedSeconds:used});
          }
        }
      }catch(error) {
        if(blockedSince===null)blockedSince=now();
        await publish('waiting','infrastructure-or-ownership',{errorType:error.name,blockedSeconds:(now()-blockedSince)/1000});
        if(now()-blockedSince>=1800000)return await pause('infrastructure-or-ownership');
      }
      try{await wait(settings?.pollMs??60000,undefined,{signal});}catch(error){if(!signal?.aborted)throw error;}
    }
  }catch(error){try{await cli(['pause','--instance',config.root]);}catch{}throw error;}
  finally{await release();}
}

if(import.meta.main) {
  const {values}=parseArgs({options:{instance:{type:'string'},cohort:{type:'string'},qualification:{type:'string'},revision:{type:'string'}}});
  const abort=new AbortController(),stop=()=>abort.abort();process.on('SIGTERM',stop);process.on('SIGINT',stop);
  Promise.resolve().then(async()=>{
    if(!values.instance||!values.cohort||!values.qualification||!values.revision)throw new Error('Instance, cohort, qualification and revision required');
    return superviseCampaign({instance:values.instance,cohort:values.cohort,
      qualification:JSON.parse(await readFile(values.qualification,'utf8')),sourceRevision:values.revision,signal:abort.signal});
  }).then(result=>console.log(JSON.stringify(result))).catch(error=>{console.error(JSON.stringify({error:error.code??error.name}));process.exitCode=1;})
    .finally(()=>{process.off('SIGTERM',stop);process.off('SIGINT',stop);});
}
