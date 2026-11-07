import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm,readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {setTimeout as delay} from 'node:timers/promises';
import {State} from '../../controller/state.mjs';
import {CampaignStorage} from '../../controller/storage.mjs';
import {processIdentity} from '../../controller/lifecycle.mjs';
import {withPackCapture} from '../../cli/storage.mjs';
import {readConfig} from '../../cli/instance.mjs';
import {operate} from '../../cli/runtime.mjs';
import {setup} from './driver_setup.mjs';
import {workloads} from '../../cli/campaign-manifest.mjs';
import {superviseCampaign,claimSupervisor,localOwnersStopped} from '../../cli/campaign-supervisor.mjs';

function qualification() {
  const expected={sourceRevision:'frozen-source'},receipts=[];
  for(const target of ['a3','a5']) {
    expected[target]={};
    for(const workload of workloads) {
      expected[target][workload]={sourceHash:'source',caseHash:'cases',verifierHash:'verifier',independentCheckHash:'checker',runtimeFingerprint:target,caseIds:[1]};
      receipts.push({format:1,qualified:true,target,workload,sourceHash:'source',caseHash:'cases',verifierHash:'verifier',
        independentChecks:{workload,passed:true,checkerHash:'checker'},runtime:{fingerprint:target},
        cases:[{id:1,passed:true,referenceHash:'reference',outputHash:'output',maxAbsError:0}],
        timing:{passed:true,samples:[{id:1,seconds:[.1]}]}});
    }
  }
  return {expected,receipts,localValidation:{passed:true,sourceRevision:'frozen-source'}};
}
const options=()=>({instance:'/virtual',cohort:'new',sourceRevision:'frozen-source',qualification:qualification()});
const fresh=()=>({status:{controller:{state:'stopped'},progress:{done:false}},budget:null,operation:null,attempts:0});
const done=()=>({status:{controller:{state:'stopped'},progress:{done:true}},budget:{cycle:'same-cycle',elapsed:9},operation:'campaign',attempts:1});
function fixture(views,started=0) {
  let clock=started,index=0,released=0;
  const calls=[],reports=[],config={root:'/virtual',campaign_cohort:'new',campaign_source_revision:'frozen-source',campaign_seconds:2592000,campaign_plan:[{id:'only'}]};
  const deps={read:async()=>structuredClone(config),identity:async()=>({pid:1,start:'start',boot:'boot'}),
    claim:async()=>async()=>released++,snapshot:async()=>structuredClone(views[Math.min(index++,views.length-1)]),
    cli:async argv=>{calls.push(argv);return {request:'request-'+calls.length};},
    restart:async()=>({stopped:true}),closed:async()=>true,space:async()=>100*1024**3,
    now:()=>clock,wait:async ms=>{clock+=ms;if(clock>4000000)throw new Error('Test did not reach a terminal state');},
    load:async()=>null,persist:async(_config,report)=>reports.push(structuredClone(report))};
  return {deps,calls,reports,config,released:()=>released};
}

test('fresh supervision submits once; observable controller/request ownership prevents duplicates',async()=>{
  const running={...fresh(),status:{controller:{state:'running'},progress:{done:false}}};
  const f=fixture([fresh(),running,done()]);
  assert.equal((await superviseCampaign(options(),f.deps)).state,'complete');
  assert.deepEqual(f.calls.map(call=>call[0]),['campaign','pause']);assert.equal(f.released(),1);
  const observed=fixture([fresh(),done()]);observed.deps.restart=async()=>({stopped:false,observable:true});
  await superviseCampaign(options(),observed.deps);
  assert.deepEqual(observed.calls.map(call=>call[0]),['pause']);
});

test('dead local owners resume the same native cycle while retained A/B handles remain untouched',async()=>{
  const retained={...fresh(),budget:{cycle:'same-cycle',elapsed:123,running:true},operation:'campaign',attempts:2,
    pendingAB:{handle:'remote:target:job:retained',reserved:true}};
  const before=structuredClone(retained),f=fixture([retained,done()]);
  await superviseCampaign(options(),f.deps);
  assert.deepEqual(f.calls.map(call=>call[0]),['resume','pause']);
  assert.deepEqual(retained,before);assert.equal(f.reports.find(report=>report.state==='launching').elapsedSeconds,123);
  assert.ok(f.calls.every(call=>!call.includes('config')));
});

test('explicit stop survives a failed campaign request while internal failures remain resumable',async t=>{
  const root=await mkdtemp(join(tmpdir(),'supervisor-operator-stop-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  new State(join(root,'state.sqlite')).close();
  await operate('stop',{root,mode:'phase1'});
  const state=new State(join(root,'state.sqlite'));
  let operatorStopped;
  try{operatorStopped=JSON.parse(state.db.prepare("SELECT value FROM control WHERE key='operator-stop'").get().value);}
  finally{state.close();}
  const failed={...fresh(),operatorStopped,budget:{cycle:'same',elapsed:3},operation:'campaign',
    status:{controller:{state:'stopped',stopRequested:true},progress:{done:false},
      requests:[{id:'failed',kind:'campaign',status:'failed'}]}};
  const stopped=fixture([failed]);
  assert.equal((await superviseCampaign(options(),stopped.deps)).reason,'operator-stop');
  assert.deepEqual(stopped.calls.map(call=>call[0]),['pause']);
  const crashed=fixture([{...failed,operatorStopped:false},done()]);
  await superviseCampaign(options(),crashed.deps);
  assert.deepEqual(crashed.calls.map(call=>call[0]),['resume','pause']);
});

test('unknown ownership and repeated native resume failures become attention after30 elapsed minutes',async()=>{
  for(const failure of ['unknown','failed-request']) {
    const view=fresh();view.operation='campaign';view.budget={cycle:'same',elapsed:3};
    if(failure==='unknown')view.status.controller.state='unknown';
    else view.status.requests=[{id:'failed',kind:'resume',status:'failed'}];
    const f=fixture([view]);const result=await superviseCampaign(options(),f.deps);
    assert.equal(result.state,'attention-needed');assert.equal(result.reason,'infrastructure-or-ownership');
    if(failure==='unknown')assert.equal(f.calls.some(call=>call[0]==='resume'),false);
    assert.ok(f.reports.at(-1).at>=1800000);
  }
});

test('supervisor restart retains elapsed infrastructure wait, while a recovered controller clears it',async()=>{
  const prior={sourceRevision:'frozen-source',owner:{boot:'boot'},blockedSince:0,blockedElapsed:1800000};
  const unknown=fresh();unknown.status.controller.state='unknown';
  const f=fixture([unknown],1800000);f.deps.load=async()=>prior;
  const result=await superviseCampaign(options(),f.deps);
  assert.equal(result.state,'attention-needed');assert.equal(result.at,1800000);
  const running={...fresh(),status:{controller:{state:'running'},progress:{done:false}}};
  const recovered=fixture([running,done()],1800000);recovered.deps.load=async()=>prior;
  assert.equal((await superviseCampaign(options(),recovered.deps)).state,'complete');
  assert.equal(recovered.reports.find(report=>report.state==='running').blockedSince,null);
});

test('startup can need attention after30minutes without misclassifying a known long-running actor',async()=>{
  const startup=fixture([fresh()]);startup.deps.restart=async()=>({stopped:false,observable:true,activeWork:false});
  assert.equal((await superviseCampaign(options(),startup.deps)).state,'attention-needed');
  const actor={...fresh(),operation:'campaign',attempts:1,budget:{cycle:'same',elapsed:1}};
  const working=fixture([...Array(32).fill(actor),done()]);
  working.deps.restart=async()=>({stopped:false,observable:true,activeWork:true});
  assert.equal((await superviseCampaign(options(),working.deps)).state,'complete');
  assert.ok(working.reports.at(-1).at>1800000);
  assert.deepEqual(working.calls.map(call=>call[0]),['pause']);
});

test('disk guards prevent launch below64GiB and checkpoint a running campaign below16GiB',async()=>{
  const initial=fixture([fresh()]);initial.deps.space=async()=>63*1024**3;
  assert.equal((await superviseCampaign(options(),initial.deps)).reason,'initial-disk');assert.deepEqual(initial.calls,[]);
  const active={...fresh(),budget:{cycle:'same',elapsed:1},operation:'campaign'};
  const f=fixture([active]);f.deps.space=async()=>15*1024**3;
  assert.equal((await superviseCampaign(options(),f.deps)).reason,'low-disk');
  assert.deepEqual(f.calls.map(call=>call[0]),['pause']);
});

test('completed campaigns and exhausted native allocations never drain or refill A/B',async()=>{
  for(const view of [done(),{...fresh(),budget:{cycle:'same',elapsed:2592000},operation:'campaign'}]) {
    const f=fixture([view]);await superviseCampaign(options(),f.deps);
    assert.deepEqual(f.calls.map(call=>call[0]),['pause']);assert.equal(f.released(),1);
  }
});

test('signal checkpoints before release; failed closure remains explicitly pending',async()=>{
  for(const settled of [true,false]) {
    const f=fixture([fresh()]),abort=new AbortController();abort.abort();
    f.deps.closed=async()=>settled;
    const result=await superviseCampaign({...options(),signal:abort.signal},f.deps);
    assert.equal(result.checkpointPending,!settled);assert.equal(f.released(),1);
    assert.deepEqual(f.calls.map(call=>call[0]),['pause']);
  }
});

test('stale local/source bindings fail before claiming or starting a process',async()=>{
  for(const change of [o=>o.qualification.localValidation.passed=false,o=>o.sourceRevision='other-source']) {
    const f=fixture([fresh()]),o=options();change(o);let claimed=false;f.deps.claim=async()=>{claimed=true;};
    await assert.rejects(superviseCampaign(o,f.deps));assert.equal(claimed,false);assert.deepEqual(f.calls,[]);
  }
});

test('exclusive PID/start/boot claims block competing watchdogs, packing and cleanup but not stopped ownership',async t=>{
  const root=await mkdtemp(join(tmpdir(),'supervisor-ownership-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  await new CampaignStorage(state,root).initialize({id:'fixture'});
  const config={root},me=await processIdentity();
  state.add({id:'closed',goal:'fixture',actor_seconds:1});const attempt=state.admit('closed',{...me,start:'old'});
  state.finish(attempt.id,'passed');state.event(attempt.id,'postprocess-completed',{});
  const release=await claimSupervisor(config,me);
  await assert.rejects(claimSupervisor(config,me),/Previous supervisor/);
  await assert.rejects(withPackCapture(config,async()=>assert.fail('live watchdog packed')),/stopped writers/);
  const python=process.env.PYTHON??process.env.DRIVER_PYTHON??'python3';
  const code=`from ascend_tools.storage_cli import Storage\nimport sys\ns=Storage(sys.argv[1])\ntry:\n s.plan({})\nexcept ValueError:\n print('blocked')\nelse:\n raise AssertionError('live watchdog cleanup allowed')\ns.quiescent({'attempt':sys.argv[2]})\n`;
  assert.equal((await promisify(execFile)(python,['-c',code,root,attempt.id],{env:{...process.env,PYTHONPATH:process.cwd()}})).stdout.trim(),'blocked');
  await release();let captured=false;await withPackCapture(config,async()=>{captured=true;});assert.equal(captured,true);
  const staleRelease=await claimSupervisor(config,{...me,start:'old-start'}),currentRelease=await claimSupervisor(config,me);
  await staleRelease();assert.deepEqual(JSON.parse(state.db.prepare("SELECT value FROM control WHERE key='supervisor-owner'").get().value),me);
  await currentRelease();
  state.db.prepare('INSERT INTO control VALUES (?,?)').run('supervisor-owner',JSON.stringify({pid:me.pid,start:5,boot:me.boot}));
  await assert.rejects(claimSupervisor(config,me),/Previous supervisor/);
  state.db.prepare("DELETE FROM control WHERE key='supervisor-owner'").run();
  state.db.prepare('INSERT INTO control VALUES (?,?)').run('owner',JSON.stringify(me));
  assert.equal((await localOwnersStopped(config)).observable,true);
  state.db.prepare("UPDATE control SET value='null' WHERE key='owner'").run();
  state.db.prepare('INSERT INTO control VALUES (?,?)').run('skill-background-resource',JSON.stringify({handle:'remote:target:job:pending'}));
  assert.equal((await localOwnersStopped(config)).stopped,true);
  state.db.exec("CREATE TABLE requests(kind TEXT,status TEXT,owner TEXT);INSERT INTO requests VALUES('campaign','queued',NULL)");
  assert.equal((await localOwnersStopped(config)).unknown,true);
});

test('real public CLI supervision completes a scripted local project without changing its native budget cycle',{timeout:60000},async t=>{
  const f=await setup(t,'phase2',['unrelated']),original=await readConfig(f.root),q=qualification();
  const input=join(f.root,'supervision-config.json');
  await writeFile(input,JSON.stringify({mode:'phase1',campaign_cohort:'new',
    campaign_source_revision:'frozen-source',campaign_seconds:60,skills:{...original.skills,tuningDisabled:true},
    campaign_plan:[{id:'only',phase:'phase1',new_projects:1}],projects:original.projects.map(project=>({...project,phase:'phase1',campaign_entry:'only'}))}));
  await f.run('config','--config',input);
  const admission=new State(join(f.root,'state.sqlite'));
  admission.db.exec('CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
  admission.db.prepare('INSERT OR REPLACE INTO control VALUES (?,?)').run('memory-admission',JSON.stringify({limit:1,blocked:true}));
  const pending=superviseCampaign({...options(),instance:f.root,qualification:q,signal:t.signal},
    {space:async()=>100*1024**3,wait:(ms,value,opts)=>delay(Math.min(ms,30),value,opts)});
  const entryDeadline=Date.now()+15000;
  while(!admission.queryEvents(null,{types:['campaign-entry-started'],limit:1}).length) {
    if(Date.now()>entryDeadline)throw Error('Runtime did not start the campaign entry');await delay(25);
  }
  await delay(150);assert.equal(admission.db.prepare('SELECT count(*) n FROM attempts').get().n,0);
  admission.db.prepare('UPDATE control SET value=? WHERE key=?').run(JSON.stringify({limit:1,blocked:false}),'memory-admission');admission.close();
  const result=await pending;
  assert.equal(result.state,'complete');
  const status=await f.run('status');assert.equal(status.progress.done,true);assert.equal(status.attempts.length,1);
  const ledger=new State(join(f.root,'state.sqlite'));
  try{const budget=JSON.parse(ledger.db.prepare("SELECT value FROM control WHERE key='campaign-budget'").get().value);
    assert.equal(budget.seconds,60);assert.ok(budget.elapsed>0);assert.ok(budget.cycle);
    assert.equal(ledger.db.prepare("SELECT value FROM control WHERE key='supervisor-owner'").get(),undefined);
  }finally{ledger.close();}
  assert.equal(JSON.parse(await readFile(join(f.root,'.supervisor','status.json'),'utf8')).state,'complete');f.pass();
});

test('orphan workers keep charging a running native clock without modifying its checkpoint',async()=>{
  const allocation=30*86400,elapsed=allocation-9*3600;
  for(const [boot,running,gap,reason] of [['boot',true,9*3600,'elapsed'],
    ['boot',true,8*3600,'complete'],['old-boot',true,9*3600,'complete'],['boot',false,9*3600,'complete']]) {
    const view={...fresh(),operation:'campaign',attempts:1,
      budget:{cycle:'same',seconds:allocation,elapsed,monotonic:0,boot,running}};
    const before=structuredClone(view.budget),f=fixture([view,done()],gap*1000);
    f.deps.restart=async()=>({stopped:false,observable:true,activeWork:true});
    f.deps.wait=async()=>{};
    assert.equal((await superviseCampaign(options(),f.deps)).reason,reason);
    assert.deepEqual(view.budget,before);
    assert.deepEqual(f.calls.map(call=>call[0]),['pause']);
  }
});

test('running controller cannot reset repeated disk failures, including across supervisor restart',async()=>{
  const running={...fresh(),status:{controller:{state:'running'},progress:{done:false}}};
  for(const invalid of [false,true]) {
    const f=fixture([running],120000);
    f.deps.load=async()=>({sourceRevision:'frozen-source',owner:{boot:'boot'},blockedSince:0,blockedElapsed:120000});
    f.deps.space=async()=>{if(invalid)return NaN;throw new Error('disk probe unavailable');};
    const result=await superviseCampaign(options(),f.deps);
    assert.equal(result.state,'attention-needed');assert.equal(result.reason,'infrastructure-or-ownership');
    assert.equal(result.at,1800000);assert.equal(result.blockedSince,0);
    assert.equal(result.checkpointPending,false);assert.deepEqual(f.calls.map(call=>call[0]),['pause']);
  }
  const recovered=fixture([running,done()],32*60000);
  recovered.deps.load=async()=>({sourceRevision:'frozen-source',owner:{boot:'boot'},blockedSince:0,blockedElapsed:32*60000});
  assert.equal((await superviseCampaign(options(),recovered.deps)).state,'complete');
  assert.equal(recovered.reports.find(report=>report.state==='running').blockedSince,null);
});

function memoryFixture({rss=29,prior,operator=false,settled=true,queuedPolls=0,operatorPending=false}={}) {
  const f=fixture([fresh()]);f.config.concurrency=4;
  f.config.local_memory={lowBytes:20,softBytes:24,checkpointBytes:28,hardBytes:32,pollMs:2,reduceMs:30,highMs:120,coolMs:6};
  let paused=Boolean(prior?.token),token=prior?.token,resumed=false,releases=0,samples=0,acknowledged=0,queued=0;
  const controls=[];
  const budget={cycle:'retained-cycle',elapsed:123};
  f.deps.snapshot=async()=>{
    if(resumed&&queued++>=queuedPolls){paused=false;token=null;acknowledged++;}
    return {budget,operation:'campaign',attempts:2,paused,memoryPause:operator||operatorPending&&resumed?'operator':token,operatorStopped:false,
      status:{controller:{state:paused?'stopped':'running'},progress:{done:acknowledged>1},
        requests:resumed?[{id:'same-request',kind:'resume',status:'running'}]:[]}};
  };
  f.deps.memorySample=async()=>({rssBytes:paused||resumed?10:rss,processCount:1,largest:[],sample:++samples});
  f.deps.memoryControl=async(_config,state)=>controls.push(structuredClone(state));
  f.deps.memoryPause=async(_config,value)=>{paused=true;token=value;};
  f.deps.memoryRelease=async()=>releases++;
  f.deps.closed=async()=>settled;
  f.deps.cli=async argv=>{f.calls.push(argv);if(argv[0]==='resume')resumed=true;return {request:'same-request'};};
  if(prior)f.deps.load=async()=>({sourceRevision:'frozen-source',owner:{boot:'boot'},memory:prior});
  return {...f,controls,budget,releases:()=>releases,samples:()=>samples};
}

test('memory checkpoint releases services then resumes retained attempts and elapsed allocation at a lower cap',async()=>{
  const f=memoryFixture();
  const result=await superviseCampaign(options(),f.deps);
  assert.equal(result.reason,'complete');assert.equal(f.releases(),1);
  assert.deepEqual(f.calls.map(call=>call[0]),['resume','pause']);
  assert.ok(f.reports.some(report=>report.state==='cooling'));
  assert.equal(result.memory.limit,3);assert.equal(result.memory.recovered,true);
  assert.deepEqual(f.budget,{cycle:'retained-cycle',elapsed:123});
  assert.ok(f.controls.some(control=>control.blocked));assert.equal(f.controls.at(-1).blocked,false);
});

test('guard restart restores reduced cap and own pause; operator pause takes precedence',async()=>{
  const prior={limit:2,token:'retained-pause',lowSince:0};
  const f=memoryFixture({prior});
  assert.equal((await superviseCampaign(options(),f.deps)).reason,'complete');
  assert.equal(f.reports.at(-1).memory.limit,2);assert.equal(f.calls[0][0],'resume');
  const operator=memoryFixture({prior,operator:true});
  assert.equal((await superviseCampaign(options(),operator.deps)).reason,'operator-pause');
  assert.deepEqual(operator.calls.map(call=>call[0]),['pause']);assert.equal(operator.samples(),0);
});

test('memory emergency, recurring pressure at one and unresolved checkpoint never automatically resume',async()=>{
  const emergency=memoryFixture({rss:32});
  assert.equal((await superviseCampaign(options(),emergency.deps)).reason,'memory-emergency');
  const minimum=memoryFixture({rss:29,prior:{limit:1,recovered:true}});
  assert.equal((await superviseCampaign(options(),minimum.deps)).reason,'memory-at-minimum');assert.equal(minimum.releases(),1);
  const unresolved=memoryFixture({settled:false});
  const result=await superviseCampaign(options(),unresolved.deps);
  assert.equal(result.reason,'memory-checkpoint-timeout');assert.equal(result.checkpointPending,true);
  for(const f of [emergency,minimum,unresolved])assert.equal(f.calls.some(call=>call[0]==='resume'),false);
});


test('queued memory resume is observed until native acknowledgement, and a newer operator pause wins',async()=>{
  const f=memoryFixture({queuedPolls:5});
  assert.equal((await superviseCampaign(options(),f.deps)).reason,'complete');
  assert.equal(f.calls.filter(call=>call[0]==='resume').length,1);
  assert.ok(f.reports.filter(report=>report.reason==='memory-resume').length>=5);
  const operator=memoryFixture({queuedPolls:5,operatorPending:true});
  assert.equal((await superviseCampaign(options(),operator.deps)).reason,'operator-pause');
  assert.equal(operator.calls.filter(call=>call[0]==='resume').length,1);
});

test('unreclaimed memory cool-down is bounded without repeatedly resuming',async()=>{
  const f=memoryFixture({prior:{limit:2,token:'owned',phaseAt:0}});
  f.deps.memorySample=async()=>({rssBytes:23,processCount:1,largest:[]});
  let clock=0;f.deps.now=()=>clock;f.deps.wait=async()=>{clock+=600000;};
  assert.equal((await superviseCampaign(options(),f.deps)).reason,'memory-recovery-timeout');
  assert.equal(f.calls.some(call=>call[0]==='resume'),false);
});

test('failed memory observation blocks admissions and retains infrastructure failure elapsed time',async()=>{
  const f=memoryFixture();f.deps.memorySample=async()=>{throw Error('proc unavailable');};
  let clock=0;f.deps.now=()=>clock;f.deps.wait=async()=>{clock+=600000;};
  const result=await superviseCampaign(options(),f.deps);
  assert.equal(result.reason,'infrastructure-or-ownership');assert.ok(f.controls.length>=3);
  assert.ok(f.controls.every(control=>control.blocked));assert.equal(f.calls.some(call=>call[0]==='resume'),false);
});


test('unconfirmed knowledge-service release prevents automatic recovery',async()=>{
  const f=memoryFixture();f.deps.memoryRelease=async()=>{throw Error('service still alive');};
  const result=await superviseCampaign(options(),f.deps);
  assert.equal(result.reason,'memory-service-release');assert.equal(result.checkpointPending,true);
  assert.equal(f.calls.some(call=>call[0]==='resume'),false);
});
