import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { syncBuiltinESMExports } from 'node:module';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir, homedir } from 'node:os';
import { join, dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { createCplClient, createEnvironmentTools } from '../../integrations/environments/index.mjs';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { State } from '../../controller/state.mjs';
import { ActorClock } from '../../controller/budget.mjs';
import { attemptStopped, processIdentity } from '../../controller/lifecycle.mjs';
import { createEnvironment } from '../../integrations/environments/ascend.mjs';
import {reserveDevice} from '../../integrations/environments/device-leases.mjs';

// Real foundation parser runs against a CPU npu-smi wire-format fixture. No NPU evidence.
async function fixture(t, health = {}) {
  const instancePath = await mkdtemp(join(tmpdir(), 'ascend-resources-'));
  t.after(() => rm(instancePath, { recursive: true, force: true }));
  const bin = join(instancePath, 'bin'), source = join(instancePath, 'project');
  await mkdir(bin); await mkdir(source);
  const instructionFile = join(instancePath, 'user-instructions.md');
  await writeFile(instructionFile, 'User-selected Ascend instructions');
  const telemetry = join(instancePath, 'devices.json');
  const rows = { 0: {}, 1: { status: 'ALARM' }, 2: { busy: true, memory: 49 }, 3: { timeout: true } };
  await writeFile(telemetry, JSON.stringify(rows));
  await writeFile(join(bin, 'npu-smi'), `#!/usr/bin/env python3
import json,sys,os,time
args=sys.argv; device=int(args[args.index('-i')+1]); query=args[args.index('-t')+1]
row=json.load(open(os.environ['ASCEND_TEST_TELEMETRY']))[str(device)]
if row.get('timeout'): time.sleep(2)
outputs={'board':f'NPU ID: {device}\\nChip Name: ${health.product==='a3'?'Ascend910C':'Ascend950'}',
'health':'Health Status: '+row.get('status','OK')+'\\nError Code: 0',
'usages':f'HBM Capacity(MB): 1000\\nHBM Usage Rate(%): {row.get("memory",0)}\\nAicore Usage Rate(%): 0\\nAivector Usage Rate(%): 0\\nNPU Utilization(%): 0\\nAicube Usage Rate(%): 0',
'proc-mem':'Process id: 123' if row.get('busy') else 'No process in device.',
'current-fault-event':'Fault Event Count: 0'}
print(outputs[query])
`, { mode: 0o700 });
  const calls = [], events = [], commands = [], jobs = new Map(), admissions=new Map();
  let runWorkload = false, onSync = async () => {}, running = false, queued = 0;
  let waitSignal,releaseProof,transformAdmission;
  let enabled = [0,1,2,3], disconnect = false, sequence = 0, clockSkew = 0, hideSubmission = false, lastProbe;
  const client = async (args,{signal}={}) => {
    calls.push(args);
    if (args[0] === 'npu') return { state: 'completed', profiles: [{ target: 'fixture', pool: instancePath, devices: enabled.map(id => ({ id, enabled: true })) }] };
    if (args[0] === 'capabilities') return { state: 'available' };
    if (args[0] === 'preflight') return { state: 'completed' };
    if (args[0] === 'upload') return { state: 'completed' };
    if (args[0] === 'download') { let value=admissions.get(args[2]) ?? [...admissions].find(([key])=>key.endsWith('/'+args[2]))?.[1] ?? ''; if(value&&transformAdmission&&args[2].endsWith('/admission.json'))value=JSON.stringify(transformAdmission(JSON.parse(value))); await writeFile(args[3],value); return { state: 'completed' }; }
    if (args[0] === 'run') {
      const handle = `remote:fixture:job:${++sequence}`, file = args[args.indexOf('--file')+1];
      let content = '';
      if (file.endsWith('probe.sh')) {
        const out = await promisify(execFile)('bash', [file], { env: { ...process.env, PATH: bin+':'+process.env.PATH, ASCEND_TEST_TELEMETRY: telemetry } });
        const response = JSON.parse(out.stdout); response.observed_at += clockSkew;
        content = JSON.stringify(response);
      }
      if (file.endsWith('sync.sh')) await onSync();
      let exit = 0;
      if (file.endsWith('command.sh')) {
        const script = await readFile(file,'utf8');
        const config = JSON.parse(script.slice(script.lastIndexOf(" '{")+2,-2).replaceAll("'\\''","'"));
        const inner=JSON.parse(config.command.slice(config.command.lastIndexOf(" '{")+2,-1).replaceAll("'\\''","'"));
        if(inner.source)commands.push({devices:inner.devices,env:config.env});
        let admission={token:inner.lease.token,decision:inner.source?'admitted':'reconciled',...(inner.source&&releaseProof!==undefined?{released:releaseProof}:{})};
        if(runWorkload){
          const receipt=join(instancePath,'host-admission-'+sequence);
          try { await promisify(execFile)('bash',['-c',config.command],{env:{...process.env,...config.env,RSI_LEASE_RECEIPT:receipt,PATH:bin+':'+process.env.PATH,ASCEND_TEST_TELEMETRY:telemetry}}); }
          catch(error) { exit=error.code; }
          try{admission=JSON.parse(await readFile(receipt,'utf8'));}catch{admission=null;}
        }
        if(admission)admissions.set(config.env.RSI_LEASE_RECEIPT,JSON.stringify(admission));
      }
      jobs.set(handle, {content,exit});
      if (file.endsWith('probe.sh')) lastProbe = { handle, receipt:args[args.indexOf('--receipt-file')+1] };
      if (hideSubmission) throw new Error('lost native reply and receipt not yet visible');
      if (args.includes('--receipt-file')) await writeFile(args[args.indexOf('--receipt-file')+1],JSON.stringify({handle}));
      return { handle, state: 'queued' };
    }
    if(args[0]==='observe' && queued>0){queued--;return {state:'queued'};}
    if(args[0]==='observe' && waitSignal && args.includes('--wait'))return waitSignal(signal);
    if (args[0] === 'observe') return { state: disconnect ? 'observation-unavailable' : running ? 'running' : 'completed', exit: jobs.get(args[1])?.exit ?? 0 };
    if (args[0] === 'logs') return { state: 'completed', content: jobs.get(args[1])?.content };
    throw new Error('unexpected CPL operation '+args[0]);
  };
  const context = { instancePath, leaseDirectory: join(instancePath,'leases'), client, sdk: { createLocalBashOperations: () => ({}) },
    event: async (type,data) => events.push({ type,...data }), exclude: async (_,fn) => fn() };
  const options = { mode: 'cpl', target: 'fixture', runtime: 'registered', instructionFile,
    workspacePath: instancePath, remoteDirectory: 'owned', pollMs: 0, probeSeconds: 3,
    health: { product: 'a5', chip: null, query_timeout_seconds: .1, ...health } };
  const environment = await createEnvironment(options, context);
  return { environment, options, context, source, calls, events, commands, rows, telemetry,
    transformAdmission:fn=>{transformAdmission=fn;},setReleaseProof:value=>{releaseProof=value;}, setEnabled: ids => { enabled = ids; }, disconnect: value => { disconnect = value; }, setClockSkew: seconds => { clockSkew = seconds; }, hideSubmission:value=>{hideSubmission=value;}, lastProbe:()=>lastProbe,
    queued:value=>{queued=value;}, wait:fn=>{waitSignal=fn;},runWorkload: () => {runWorkload=true;}, onSync: fn => {onSync=fn;}, running: value => {running=value;},
    update: () => writeFile(telemetry, JSON.stringify(rows)) };
}

for(const failure of ['closed','closed-at-stop','not-admitted','unknown'])test(`health setup proof: ${failure}`,async t=>{
  const f=await fixture(t),reason={stage:'ssh_preflight',message:'open failed'};
  const abort=new AbortController(),hasHandle=failure.startsWith('closed');
  const environment=await createEnvironment(f.options,{...f.context,signal:abort.signal,client:async(args,settings)=>{
    if(failure==='not-admitted'&&args[0]==='run')return {admitted:false,reason};
    if(args[0]==='observe'){
      if(failure==='closed-at-stop'){
        if(!args.includes('--wait'))return {state:'running'};
        abort.abort();
      }
      return failure==='unknown'?{state:'observation-unavailable'}:{state:'failed',exit:null,reason};
    }
    return f.context.client(args,settings);
  }});
  await assert.rejects(environment.inspectDevices(),error=>{
    if(failure==='unknown'){assert.equal(error.closure,undefined);return true;}
    assert.equal(error.code,hasHandle?'SETUP_CLOSED':'CPL_NOT_ADMITTED');
    assert.equal(error.closure.closed,true);assert.equal(error.closure.payloadStarted,false);
    assert.deepEqual(error.closure.handles,hasHandle?[f.lastProbe().handle]:[]);
    assert.match(error.message,/open failed/);return true;
  });
  assert.equal(f.calls.some(a=>a[0]==='run'&&a[a.indexOf('--file')+1].endsWith('command.sh')),false);
});

test('fresh parser-backed discovery filters permission, busy, Alarm, and unresponsive devices', async t => {
  const f = await fixture(t);
  const first = await f.environment.inspectDevices({ target: 'fixture' });
  assert.deepEqual(first.filter(row => row.available).map(row => row.metadata.physicalDevice), [0]);
  assert.ok(first.every(row => row.id.startsWith('ascend-') && row.metadata.healthSha256.length === 64));
  assert.ok(first.every(row => !Object.hasOwn(row,'env')),'CPL rows must not mask controller workers');
  f.rows[0].busy = true; await f.update();
  assert.equal((await f.environment.inspectDevices()).some(row => row.available), false);
  f.setEnabled([]);
  assert.deepEqual(await f.environment.inspectDevices(), []);
  assert.deepEqual(await f.environment.inspectDevices({ target: 'other' }), []);
});

test('ordinary Alarm/busy opt-ins retain policy, parallel availability stays idle, timing is healthy idle', async t => {
  const f = await fixture(t, { allow_alarm: true, allow_busy: true });
  const ordinary = await f.environment.inspectDevices();
  assert.deepEqual(ordinary.filter(row => row.available).map(row => row.metadata.physicalDevice), [0,1]);
  assert.equal(ordinary.find(row => row.metadata.physicalDevice === 2).metadata.admitted, true);
  f.rows[2].memory = 50; await f.update();
  const full = await f.environment.inspectDevices();
  assert.equal(full.find(row => row.metadata.physicalDevice === 2).metadata.admitted, false);
  const timing = await f.environment.inspectDevices({ purpose: 'timing' });
  assert.deepEqual(timing.filter(row => row.available).map(row => row.metadata.physicalDevice), [0]);
});

test('scheduler slots carry no mask; every command freshly selects and fixes its device', async t => {
  const f = await fixture(t, {allow_alarm:true});
  const slots = await f.environment.resources();
  assert.equal(f.calls.some(args => args[0] === 'run'), false);
  assert.ok(slots.every(row => row.metadata.kind === 'host-slot' && !row.env && row.metadata.physicalDevice === undefined));
  const bound = await createEnvironment(f.options, {...f.context,resource:slots[0]});
  await bound.prepare({project:{id:'p',source:f.source},attempt:{id:'a'}});
  for (const physical of [0,1]) {
    if (physical) { f.rows[0].busy=true; await f.update(); }
    const result = await bound.execute('kernel-command',f.source,{timeout:3,resourceId:slots[0].id,
      env:{ASCEND_PHYSICAL_DEVICE:'7',ASCEND_RT_VISIBLE_DEVICES:'7'}});
    assert.equal(result.settled,true);
    const run = f.calls.filter(args=>args[0]==='run'&&args[args.indexOf('--file')+1].endsWith('command.sh')&&args.includes('--npu')).at(-1);
    assert.equal(run[run.indexOf('--npu')+1],String(physical));
    const script=await readFile(run[run.indexOf('--file')+1],'utf8');
    const config=JSON.parse(script.slice(script.lastIndexOf(" '{")+2,-2).replaceAll("'\\''","'"));
    for(const key of ['ASCEND_PHYSICAL_DEVICE','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES']) assert.equal(config.env[key],String(physical));
    assert.ok(config.timeout>2.5 && config.timeout<=3);
  }
});

for(const refusal of ['busy','disabled']) test(`unavailable ${refusal} never runs payload`,async t=>{
  const f=await fixture(t); f.setEnabled(refusal==='disabled'?[]:[0]);
  f.rows[0].busy=true;await f.update();
  const abort=new AbortController();
  const bound=await createEnvironment(f.options,{...f.context,event:async(type,data)=>{
    await f.context.event(type,data);if(type==='resource-probe-completed')abort.abort();
  }});
  await assert.rejects(bound.execute('must-not-run',f.source,{signal:abort.signal}));
  assert.equal(f.calls.some(a=>a[0]==='run'&&a[a.indexOf('--file')+1].endsWith('command.sh')),false);
});

test('unknown admission probe holds the same handle without a closed setup claim or payload', async t => {
  const f = await fixture(t), lease = (await f.environment.inspectDevices()).find(row => row.available);
  const bound = await createEnvironment(f.options, {...f.context, resource: lease});
  f.disconnect(true);
  const before = f.calls.length;
  for (let call = 0; call < 2; call++) await assert.rejects(bound.execute('must-not-run', f.source), error => {
    assert.equal(error.closure, undefined); assert.match(error.message, /unresolved/); return true;
  });
  const runs = f.calls.slice(before).filter(args => args[0] === 'run');
  assert.equal(runs.length, 1);
  assert.ok(f.calls.slice(before).filter(args => args[0] === 'observe').every(args => args[1] === f.lastProbe().handle));
  assert.equal(f.calls.slice(before).some(args => args[0] === 'upload'), false);
});

test('probe disconnect persists exact handle across restart and never resubmits unknown work', async t => {
  const f = await fixture(t);
  f.disconnect(true);
  await assert.rejects(f.environment.inspectDevices(), /unresolved/);
  const handle = f.events.find(row => row.type === 'execution-handle').handle;
  const restarted = await createEnvironment(f.options,f.context);
  await assert.rejects(restarted.inspectDevices(), /unresolved/);
  assert.equal(f.calls.filter(args => args[0] === 'run').length, 1);
  assert.ok(f.calls.filter(args => args[0] === 'observe').every(args => args[1] === handle));
  f.disconnect(false);
  assert.equal((await restarted.inspectDevices()).filter(row => row.available).length, 1);
  assert.equal(f.calls.filter(args => args[0] === 'run').length, 1);
});

test('actual user instruction file is required rather than fabricated defaults', async t => {
  const f = await fixture(t);
  await assert.rejects(createEnvironment({ ...f.options, instructionFile:undefined },f.context), /instructionFile/);
  await assert.rejects(createEnvironment({ ...f.options, instructionFile:join(f.source,'missing') },f.context), /ENOENT/);
});


test('remote wall-clock skew does not replace local elapsed freshness', async t => {
  const f = await fixture(t); f.setClockSkew(3600);
  const resources = await f.environment.inspectDevices();
  assert.equal(resources.filter(row => row.available).length, 1);
  assert.ok(resources[0].metadata.observedAt > resources[0].metadata.checkedAt + 3500);
});


test('explicit project device affinity is a recorded filter and never falls back', async t => {
  const f = await fixture(t, { allow_alarm: true });
  const bound = await createEnvironment({ ...f.options, deviceAffinity:[1] },f.context);
  const rows = await bound.inspectDevices();
  assert.deepEqual(rows.filter(row => row.available).map(row => row.metadata.physicalDevice), [1]);
  assert.deepEqual(rows[0].metadata.deviceAffinity,[1]);
  assert.ok(f.events.some(row => row.type === 'resource-policy' && row.deviceAffinity?.[0] === 1));
  f.setEnabled([0]); assert.deepEqual(await bound.inspectDevices(),[]);
});


for (const failure of ['policy','unknown']) test(`core probe-only reconciliation: ${failure}`, async t => {
  const f=await fixture(t);
  const lease=(await f.environment.inspectDevices()).find(row=>row.available);
  const file=join(f.context.instancePath,'state.sqlite');
  let state=new State(file),depth=0,now=0,exclusions=0;
  t.after(()=>state.close());
  state.add({id:'p',goal:'numeric',actor_seconds:30,source:f.source,verifier:{files:[],command:'true'}});
  const attempt=state.admit('p',{pid:2147483647,start:'gone',boot:'gone'});
  state.checkpoint(attempt.id,2);
  state.event(attempt.id,'execution-starting',{operation:'probe-only',startElapsed:2});
  const clock=new ActorClock(30,2,()=>now);
  const context={...f.context,resource:lease,
    exclude:(label,fn)=>label==='admission'?clock.exclude(label,fn):clock.exclude(label,async()=>{
      depth++;exclusions++;now+=10000;
      assert.equal(clock.elapsed,2);
      try{return await fn();}finally{
        now+=10000;assert.equal(clock.elapsed,2);depth--;
      }
    }),
    event:async(type,data)=>{
      if(type==='execution-submitting' && data.accountingExcluded)assert.ok(depth>0);
      state.event(attempt.id,type,{...data,operation:'probe-only'});
    }};
  const bound=await createEnvironment(f.options,context);
  if(failure==='policy')f.setEnabled([]);
  if(failure==='busy'){f.rows[0].busy=true;await f.update();}
  if(failure==='unknown')f.hideSubmission(true);
  await assert.rejects(bound.execute('must-not-run',f.source));
  state.event(attempt.id,'execution-unresolved',{operation:'probe-only',startElapsed:2,handle:null});
  const submitted=f.calls.filter(args=>args[0]==='run').length;
  if(failure==='unknown'){
    assert.equal(await attemptStopped(state,state.attempt(attempt.id),bound),false);
    const probe=f.lastProbe();
    await writeFile(probe.receipt,JSON.stringify({handle:probe.handle}));
    f.disconnect(true);
    assert.equal(await attemptStopped(state,state.attempt(attempt.id),bound),false);
    f.disconnect(false);f.hideSubmission(false);
  }
  state.close();state=new State(file);
  const restarted=await createEnvironment(f.options,context);
  const resumed=await state.resume(attempt.id,a=>attemptStopped(state,a,restarted),await processIdentity());
  assert.equal(resumed.id,attempt.id);assert.equal(resumed.elapsed,2);
  assert.equal(f.calls.filter(args=>args[0]==='run').length,submitted);
  assert.ok(exclusions>0);assert.equal(depth,0);assert.equal(clock.elapsed,2);
  now+=750;assert.equal(clock.elapsed,2.75);
  const reconciliation=state.events(attempt.id).find(row=>row.type==='execution-reconciled');
  assert.equal(reconciliation.data.executionSeconds,0);
});

test('simultaneous calls cannot exclude the active command interval from shared ActorClock',async t=>{
  const f=await fixture(t);let now=0;const clock=new ActorClock(30,0,()=>now);
  const bound=await createEnvironment(f.options,{...f.context,exclude:(label,fn)=>clock.exclude(label,fn)});
  await bound.prepare({project:{id:'p',source:f.source},attempt:{id:'queue'}});
  let release,started;const executing=new Promise(resolve=>{started=resolve;});
  const wait=new Promise(resolve=>{release=resolve;});
  const client=f.context.client;
  const other=await createEnvironment(f.options,{...f.context,exclude:(label,fn)=>clock.exclude(label,fn),client:async(args,settings)=>{
    if(args[0]==='observe'&&args[1]===payload){started();await wait;}
    const result=await client(args,settings);
    if(args[0]==='run'&&args.includes('--npu'))payload=result.handle;
    return result;
  }});let payload;
  await other.prepare({project:{id:'p',source:f.source},attempt:{id:'clock'}});
  const first=other.execute('one',f.source,{timeout:3});await Promise.race([executing,first.then(result=>{throw new Error('Unexpected early result '+JSON.stringify(result));})]);
  now+=1000;
  await assert.rejects(other.execute('two',f.source),error=>error.code==='CONCURRENT_DEVICE_COMMAND');
  now+=1000;assert.equal(clock.elapsed,0);release();assert.equal((await first).exitCode,0);
  assert.equal(clock.elapsed,2,'active payload is charged on settlement');
});

test('workload guard after sync refuses changed health before arbitrary Bash', async t => {
  const f=await fixture(t),lease=(await f.environment.inspectDevices()).find(row=>row.available);
  const bound=await createEnvironment(f.options,{...f.context,resource:lease});
  await bound.prepare({project:{id:'p',source:f.source},attempt:{id:'post-sync'}});
  f.runWorkload();
  const marker=join(f.context.instancePath,'executed');
  f.onSync(async()=>{ f.rows[0].busy=true; await f.update(); });
  assert.equal((await bound.execute(`touch '${marker}'`,f.source,{timeout:3})).exitCode,125);
  await assert.rejects(readFile(marker),{code:'ENOENT'});
  f.rows[0].busy=false; await f.update();
  f.onSync(async()=>{});
  assert.equal((await bound.execute(`touch '${marker}'`,f.source,{timeout:3})).exitCode,0);
  assert.equal(await readFile(marker,'utf8'),'');
});

test('explicit observation allowance bounds pending-probe wait and retains same handle',async t=>{
  const f=await fixture(t),lease=(await f.environment.inspectDevices()).find(row=>row.available);
  const bound=await createEnvironment({...f.options,pollMs:500,probeObservationSeconds:.02},{...f.context,resource:lease});
  f.disconnect(true);
  await assert.rejects(bound.execute('true',f.source,{timeout:3}),/unresolved/);
  const handle=f.lastProbe().handle,submitted=f.calls.filter(a=>a[0]==='run').length;
  f.disconnect(false);f.running(true);
  f.wait(signal=>delay(500,undefined,{signal}).then(()=>({state:'running'})));
  const started=performance.now();
  await assert.rejects(bound.execute('true',f.source,{timeout:.02}),/unresolved/);
  assert.ok(performance.now()-started<200,'current caller must not wait original probe allocation');
  assert.equal(f.calls.filter(a=>a[0]==='run').length,submitted);
  assert.equal(f.lastProbe().handle,handle);
  f.running(false);
  await bound.prepare({project:{id:'p',source:f.source},attempt:{id:'pending'}});
  assert.equal((await bound.execute('true',f.source,{timeout:3})).exitCode,0);
  assert.equal(f.calls.filter(a=>a[0]==='run'&&a[a.indexOf('--file')+1].endsWith('probe.sh')).length,2);
});


async function stalePending(f) {
  const path=join(dirname(dirname(f.lastProbe().receipt)),'pending.json');
  const value=JSON.parse(await readFile(path,'utf8'));
  value.startedMonotonic=(process.hrtime.bigint()-10_000_000_000n).toString();
  delete value.freshnessMonotonic;
  await writeFile(path,JSON.stringify(value));return path;
}

test('native SDK stale pending closure retains evidence then one fresh probe admits workload',async t=>{
  const f=await fixture(t),lease=(await f.environment.inspectDevices()).find(row=>row.available);
  let depth=0,now=0,exclusions=0;const clock=new ActorClock(30,2,()=>now);
  const bound=await createEnvironment(f.options,{...f.context,resource:lease,
    exclude:(label,fn)=>label==='admission'?clock.exclude(label,fn):clock.exclude(label,async()=>{
      depth++;exclusions++;now+=10000;
      try{return await fn();}finally{now+=10000;depth--;}
    })});
  f.disconnect(true);await assert.rejects(bound.execute('must-not-run',f.source,{timeout:3}),/unresolved/);
  const old=f.lastProbe().handle,pending=await stalePending(f);
  const runs=f.calls.filter(a=>a[0]==='run').length;
  f.disconnect(false);f.runWorkload();f.running(true);
  f.wait(async()=>{await delay(20);f.running(false);return {state:'completed',exit:0};});
  await bound.prepare({project:{id:'p',source:f.source},attempt:{id:'stale'}});
  const marker=join(f.context.instancePath,'accepted');
  const [tool]=await createEnvironmentTools(bound,f.source);
  const result=await tool.execute('cpu',{command:`touch '${marker}'`,timeout:3});
  assert.notEqual(result.isError,true);assert.equal(await readFile(marker,'utf8'),'');
  assert.equal(f.calls.filter(a=>a[0]==='run'&&a.includes('--npu')).length,1);
  await assert.rejects(readFile(join(f.context.instancePath,'.ascend-command-lease.json')),{code:'ENOENT'});
  assert.equal(f.calls.filter(a=>a[0]==='run'&&a[a.indexOf('--file')+1].endsWith('probe.sh')).length,3);
  const completed=f.events.find(e=>e.type==='resource-probe-completed'&&e.handle===old);
  assert.ok(f.calls.some(a=>a[0]==='observe'&&a[1]===old&&a.includes('--wait')));
  assert.equal(completed.executionSeconds,0);assert.equal(completed.accountingExcluded,true);
  assert.equal(JSON.parse(await readFile(completed.observation,'utf8')).handle,old);
  assert.ok(f.events.some(e=>e.type==='resource-probe-stale'&&e.handle===old));
  await assert.rejects(readFile(pending),{code:'ENOENT'});
  assert.ok(exclusions>0);assert.equal(clock.elapsed,2);assert.equal(depth,0);
  now+=750;assert.equal(clock.elapsed,2.75);
});

test('confirmed queued time does not age measurements; completion never resets freshness',async t=>{
  const f=await fixture(t);f.disconnect(true);await assert.rejects(f.environment.inspectDevices(),/unresolved/);
  const old=f.lastProbe().handle;await stalePending(f);
  f.disconnect(false);f.queued(1);
  assert.ok((await f.environment.inspectDevices()).some(row=>row.available));
  assert.equal(f.calls.filter(a=>a[0]==='run').length,1);
  assert.equal(f.events.find(e=>e.type==='resource-probe-completed').handle,old);
  assert.equal(f.events.some(e=>e.type==='resource-probe-stale'),false);
});

test('probe cancellation retains receipt and never submits replacement',async t=>{
  const f=await fixture(t),lease=(await f.environment.inspectDevices()).find(row=>row.available);
  const bound=await createEnvironment(f.options,{...f.context,resource:lease});
  const abort=new AbortController();f.running(true);
  f.wait(async()=>{abort.abort();return {state:'observation-unavailable'};});
  const count=f.calls.filter(a=>a[0]==='run').length;
  await assert.rejects(bound.execute('must-not-run',f.source,{timeout:3,signal:abort.signal}),/abort/i);
  assert.equal(f.calls.filter(a=>a[0]==='run').length,count+1);
  const pending=JSON.parse(await readFile(join(dirname(dirname(f.lastProbe().receipt)),'pending.json'),'utf8'));
  assert.equal(pending.handle,f.lastProbe().handle);
  assert.ok(f.calls.some(a=>a[0]==='observe'&&a.includes('--wait')));
});


test('standard CPL client cancellation closes only its local observer process',async t=>{
  const root=await mkdtemp(join(tmpdir(),'probe-observer-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const executable=join(root,'observer'),identity=join(root,'pid');
  await writeFile(executable,`#!/usr/bin/env python3
import os,time
open(${JSON.stringify(identity)},'w').write(str(os.getpid()))
time.sleep(10)
`,{mode:0o700});
  const instructions=join(root,'observer-instructions.md');
  await writeFile(instructions,'Local observer fixture instructions');
  const read=fs.readFile,skill=join(homedir(),'.agents/skills/remote-access/SKILL.md');
  let instructionReads=0;
  const mocked=t.mock.method(fs,'readFile',async(path,...args)=>{
    if(path===skill){instructionReads++;return read(instructions,...args);}
    return read(path,...args);
  });
  syncBuiltinESMExports();
  try {
    const abort=new AbortController(),pending=createCplClient(executable)(['observe','same-handle','--wait'],{signal:abort.signal});
    let pid;const deadline=performance.now()+2000;
    while(!pid&&performance.now()<deadline){try{pid=Number(await readFile(identity,'utf8'));}catch{}await delay(5);}
    assert.ok(pid);abort.abort();await assert.rejects(pending,{name:'AbortError'});
    let stopped=false;
    while(!stopped&&performance.now()<deadline){try{process.kill(pid,0);}catch(error){if(error.code==='ESRCH')stopped=true;else throw error;}await delay(5);}
    assert.equal(stopped,true);
    assert.equal(instructionReads,1);
  }finally{mocked.mock.restore();syncBuiltinESMExports();}
});

test('unknown payload holds pool/device and exact handle across binding restart',async t=>{
  const f=await fixture(t,{allow_alarm:true});let payloadHandle,resolved=false;
  const recoveryClient=async(args,settings)=>{
    if(!resolved&&args[0]==='observe'&&args[1]===payloadHandle)return {state:'observation-unavailable'};
    return f.context.client(args,settings);
  };
  const bound=await createEnvironment(f.options,{...f.context,client:async(args,settings)=>{
    if(args[0]==='observe'&&args[1]===payloadHandle)return {state:'observation-unavailable'};
    const result=await f.context.client(args,settings);
    if(args[0]==='run'&&args[args.indexOf('--file')+1].endsWith('command.sh'))payloadHandle=result.handle;
    return result;
  }});
  await bound.prepare({project:{id:'p',source:f.source},attempt:{id:'held'}});
  assert.equal((await bound.execute('unknown-payload',f.source,{timeout:3})).settled,false);
  const held=JSON.parse(await readFile(join(f.context.instancePath,'.ascend-command-lease.json'),'utf8'));
  assert.equal(held.physicalDevice,0);
  assert.equal(f.events.some(e=>e.type==='device-lease-released'&&e.token===held.token),false);
  assert.ok(JSON.parse(await readFile(held.receipt,'utf8')).handles.includes(payloadHandle));
  const restarted=await createEnvironment(f.options,{...f.context,client:recoveryClient});
  await assert.rejects(restarted.execute('no-replay',f.source),/lease unresolved/);
  assert.equal(f.events.some(e=>e.type==='device-lease-released'&&e.token===held.token),false);
  const other=await createEnvironment({...f.options,runtime:'other-runtime'},
    {...f.context,instancePath:join(f.context.instancePath,'other-controller')});
  await other.prepare({project:{id:'p',source:f.source},attempt:{id:'other'}});
  assert.equal((await other.execute('other-device',f.source,{timeout:3})).settled,true);
  assert.equal(f.events.filter(e=>e.type==='device-lease-acquired').at(-1).physicalDevice,1);
  resolved=true;assert.equal((await restarted.observe(payloadHandle)).state,'stopped');
  await assert.rejects(readFile(held.receipt),{code:'ENOENT'});
  assert.equal(f.events.filter(e=>e.type==='device-lease-released'&&e.token===held.token).length,1);
  assert.equal(await restarted.reconcileLease(),true);
});

test('production CPL profile without canonical pool fails closed',async t=>{
  const f=await fixture(t);
  const bound=await createEnvironment(f.options,{...f.context,client:async()=>({state:'completed',profiles:[{target:'fixture',devices:[{id:0,enabled:true}]}]})});
  await assert.rejects(bound.resources(),/actual alias-group pool/);
});

for(const failure of ['unknown-observe','legacy-unknown-observe','lost-handle-event'])test(`target reconciliation ${failure} resumes same control handle without replay`,async t=>{
  const f=await fixture(t);let controlHandle,broken=true;
  const client=async(args,settings)=>{
    if(broken&&failure.endsWith('unknown-observe')&&args[0]==='observe'&&args[1]===controlHandle)return {state:'observation-unavailable'};
    const result=await f.context.client(args,settings);
    if(args[0]==='run'&&!args.includes('--npu')&&args[args.indexOf('--file')+1].endsWith('command.sh'))controlHandle=result.handle;
    return result;
  };
  const bound=await createEnvironment(f.options,{...f.context,client,event:async(type,data)=>{
    await f.context.event(type,data);
    if(broken&&failure==='lost-handle-event'&&type==='execution-handle'&&data.handle===controlHandle)throw new Error('lost control handle event');
  }});
  await bound.prepare({project:{id:'p',source:f.source},attempt:{id:'control'}});
  if(failure==='lost-handle-event')await assert.rejects(bound.execute('payload-once',f.source,{timeout:3}),/lost control/);
  else assert.equal((await bound.execute('payload-once',f.source,{timeout:3})).settled,false);
  const runs=f.calls.filter(a=>a[0]==='run').length;
  if(failure==='legacy-unknown-observe'){
    const pointer=JSON.parse(await readFile(join(f.context.instancePath,'.ascend-command-lease.json'),'utf8'));
    const held=JSON.parse(await readFile(pointer.receipt,'utf8'));
    const stable=held.controlDirectory,legacy=join(dirname(pointer.receipt),'control');
    await fs.rename(stable,legacy);
    for(const entry of await fs.readdir(join(legacy,'.executions'),{withFileTypes:true})){
      if(!entry.isDirectory())continue;
      const file=join(legacy,'.executions',entry.name,'execution.json');
      let row;try{row=JSON.parse(await readFile(file,'utf8'));}catch(error){if(error.code==='ENOENT')continue;throw error;}
      if(row.receiptFile?.startsWith(stable+'/'))row.receiptFile=legacy+row.receiptFile.slice(stable.length);
      await writeFile(file,JSON.stringify(row));
    }
    for(const row of held.submissions)if(row.receiptFile?.startsWith(stable+'/'))row.receiptFile=legacy+row.receiptFile.slice(stable.length);
    delete held.controlDirectory;await writeFile(pointer.receipt,JSON.stringify(held));
  }
  broken=false;const restarted=await createEnvironment(f.options,{...f.context,client});
  assert.deepEqual(await Promise.all([restarted.reconcileLease(),restarted.reconcileLease()]),[true,true]);
  assert.equal(f.calls.filter(a=>a[0]==='run').length,runs);
  assert.ok(f.calls.some(a=>a[0]==='observe'&&a[1]===controlHandle));
  await assert.rejects(readFile(join(f.context.instancePath,'.ascend-command-lease.json')),{code:'ENOENT'});
});

test('lease telemetry pairs actual release with ownership and emits one wait transition across polls',async t=>{
  const f=await fixture(t);f.setEnabled([0]);
  const held=await reserveDevice(f.context.instancePath,0,f.context.leaseDirectory);
  t.after(async()=>{try{await held.release();}catch(error){if(error.code!=='ENOENT')throw error;}});
  let polls=0;
  const environment=await createEnvironment({...f.options,leasePollMs:1},{...f.context,client:async(args,settings)=>{
    const result=await f.context.client(args,settings);
    if(args[0]==='logs'&&++polls===3)await held.release();
    return result;
  },event:async(type,data)=>{
    if(type==='device-lease-released') {
      await assert.rejects(readFile(data.receipt),{code:'ENOENT'});
      await assert.rejects(readFile(join(f.context.instancePath,'.ascend-command-lease.json')),{code:'ENOENT'});
    }
    await f.context.event(type,data);
  }});
  await environment.prepare({project:{source:f.source},attempt:{id:'lease-events'}});
  assert.equal((await environment.execute('one-payload',f.source,{timeout:3})).settled,true);
  const acquired=f.events.filter(e=>e.type==='device-lease-acquired');
  const released=f.events.filter(e=>e.type==='device-lease-released');
  assert.equal(acquired.length,1);assert.equal(released.length,1);
  assert.ok(acquired[0].token);
  assert.deepEqual(released[0],{...acquired[0],type:'device-lease-released'});
  assert.deepEqual(f.events.filter(e=>e.type==='device-wait'),[
    {type:'device-wait',pool:f.context.instancePath,waiting:true,reason:'idle-devices'},
    {type:'device-wait',pool:f.context.instancePath,waiting:false},
  ]);
  assert.ok(polls>=3);
});


for(const product of ['a3','a5'])test(`${product} reserved physical device is exclusive to marked slot zero`,async t=>{
  const f=await fixture(t,{product,chip:product==='a3'?0:null});
  f.rows[2]={};await f.update();
  const options={...f.options,deviceAffinity:[0,2],backgroundDevice:2};
  const scheduler=await createEnvironment(options,{...f.context,instancePath:join(f.context.instancePath,'scheduler')});
  const slots=await scheduler.resources();
  assert.equal(slots.length,4);assert.deepEqual(slots.map(row=>row.metadata.hostSlot),[0,1,2,3]);
  assert.deepEqual(slots.map(row=>row.metadata.backgroundSlot),[true,false,false,false]);
  assert.deepEqual((await scheduler.inspectDevices()).map(row=>row.metadata.physicalDevice),[0,2]);
  const background=await createEnvironment(options,{...f.context,resource:slots[0],instancePath:join(f.context.instancePath,'background')});
  const foreground=await createEnvironment(options,{...f.context,resource:slots[1],instancePath:join(f.context.instancePath,'foreground')});
  for(const [environment,physical] of [[background,2],[foreground,0]]) {
    assert.deepEqual((await environment.resources()).map(row=>row.id),slots.map(row=>row.id));
    assert.deepEqual((await environment.inspectDevices()).map(row=>row.metadata.physicalDevice),[physical]);
    await environment.prepare({project:{source:f.source},attempt:{id:product+'-'+physical}});
    assert.equal((await environment.execute('true',f.source,{timeout:5})).settled,true);
    const payload=f.commands.at(-1);assert.deepEqual(payload.devices,[physical]);
    assert.equal(payload.env.ASCEND_PHYSICAL_DEVICE,String(physical));
    assert.equal(payload.env.ASCEND_RT_VISIBLE_DEVICES,String(physical));
  }
  f.rows[2]={status:'ALARM'};await f.update();
  assert.equal((await background.inspectDevices()).some(row=>row.available),false);
  assert.deepEqual((await foreground.inspectDevices()).filter(row=>row.available).map(row=>row.metadata.physicalDevice),[0]);
  f.setEnabled([0,1,3]);
  const previous=f.commands.length;
  await assert.rejects(background.execute('must-not-fall-back',f.source,{timeout:5}),/No enabled devices/);
  assert.equal(f.commands.length,previous);
  const disabled=await scheduler.resources();
  assert.deepEqual(disabled.map(row=>[row.id,row.metadata.hostSlot,row.available]),
    slots.map((row,index)=>[row.id,index,index!==0]));
  assert.deepEqual((await foreground.inspectDevices()).filter(row=>row.available).map(row=>row.metadata.physicalDevice),[0]);
  assert.equal((await foreground.execute('foreground-while-background-disabled',f.source,{timeout:5})).settled,true);
  assert.deepEqual(f.commands.at(-1).devices,[0]);
  assert.equal(f.commands.at(-1).env.ASCEND_RT_VISIBLE_DEVICES,'0');
  f.setEnabled([]);
  assert.deepEqual((await scheduler.resources()).map(row=>[row.id,row.available]),[[slots[0].id,false]]);
  f.setEnabled([0,2]);
  assert.deepEqual((await scheduler.resources()).map(row=>[row.id,row.available]),slots.slice(0,2).map(row=>[row.id,true]));
});

test('physical background routing requires explicit scalar membership and valid worker slot',async t=>{
  const f=await fixture(t);
  for(const settings of [{backgroundDevice:0},{deviceAffinity:[0,2],backgroundDevice:[0,2]},
    {deviceAffinity:[0,2],backgroundDevice:1},{deviceAffinity:[0,0],backgroundDevice:0}])
    await assert.rejects(createEnvironment({...f.options,...settings},f.context),/backgroundDevice/);
  await assert.rejects(createEnvironment({...f.options,deviceAffinity:[0,2],backgroundDevice:2},
    {...f.context,resource:{id:'legacy'}}),/host-slot metadata/);
  assert.equal((await f.environment.resources()).some(row=>'backgroundSlot' in row.metadata),false);
});

test('stopped prior reservation is reconciled through native closure and target fence before actor admission',async t=>{
  const f=await fixture(t);f.setEnabled([0]);
  const held=await reserveDevice(f.context.instancePath,0,f.context.leaseDirectory,{instancePath:'/deleted/old-attempt'});
  await held.update({owner:{...held.record.owner,start:'old-process'},handles:['remote:fixture:job:prior']});
  await f.environment.prepare({project:{source:f.source},attempt:{id:'new'}});
  assert.equal((await f.environment.execute('new-payload',f.source,{timeout:5})).settled,true);
  assert.ok(f.calls.some(a=>a[0]==='observe'&&a[1]==='remote:fixture:job:prior'));
  assert.equal(f.events.filter(e=>e.type==='device-lease-reconciled').length,1);
  assert.equal(f.events.filter(e=>e.type==='device-lease-acquired').length,1);
});

test('legacy reservation with no owner proof reports actionable blockage without an endless probe loop',async t=>{
  const f=await fixture(t);f.setEnabled([0]);
  const held=await reserveDevice(f.context.instancePath,0,f.context.leaseDirectory);await held.update({owner:null});
  await f.environment.prepare({project:{source:f.source},attempt:{id:'new'}});
  await assert.rejects(f.environment.execute('must-not-run',f.source,{timeout:5}),error=>error.code==='DEVICE_LEASE_BLOCKED');
  const blocked=f.events.find(e=>e.type==='device-lease-blocked');
  assert.equal(blocked.reservations[0].receipt,held.receipt);assert.equal(blocked.reservations[0].reason,'owner-unknown');
  assert.equal(f.events.filter(e=>e.type==='device-lease-acquired').length,0);
  assert.equal(f.calls.filter(a=>a[0]==='run'&&a[a.indexOf('--file')+1].endsWith('probe.sh')).length,1);
});

test('retaining native fence-control receipts preserves the actor source staging cache',async t=>{
  const f=await fixture(t);f.setEnabled([0]);
  await f.environment.prepare({project:{source:f.source},attempt:{id:'source-cache'}});
  assert.equal((await f.environment.execute('first',f.source,{timeout:5})).settled,true);
  assert.equal((await f.environment.execute('second',f.source,{timeout:5})).settled,true);
  const stages=f.events.filter(e=>e.type==='source-staged'&&!e.purpose);
  assert.equal(stages.length,2);assert.equal(stages[0].reused,false);assert.equal(stages[1].reused,true);
});

test('reclaimed token closure lets its original checkpoint resume without touching a new reservation',async t=>{
  const f=await fixture(t);f.setEnabled([0]);
  const held=await reserveDevice(f.context.instancePath,0,f.context.leaseDirectory,{instancePath:f.context.instancePath});
  await held.update({owner:{...held.record.owner,start:'old-process'},handles:['remote:fixture:job:prior']});
  await writeFile(join(f.context.instancePath,'.ascend-command-lease.json'),JSON.stringify({receipt:held.receipt,token:held.token,pool:held.record.pool,physicalDevice:0}));
  const other=await createEnvironment(f.options,{...f.context,instancePath:join(f.context.instancePath,'other-controller')});
  assert.equal((await other.reconcileReservation(0)).closed,true);
  const fresh=await reserveDevice(f.context.instancePath,0,f.context.leaseDirectory);assert.ok(fresh);
  assert.notEqual(fresh.token,held.token);
  assert.equal(await f.environment.reconcileLease(),true);
  await assert.rejects(readFile(join(f.context.instancePath,'.ascend-command-lease.json')),{code:'ENOENT'});
  assert.equal(JSON.parse(await readFile(fresh.receipt,'utf8')).token,fresh.token);
});

for(const unknown of [false,true])test(`failed control admission download (${unknown?'unknown':'closed'} transfer) retains its native handle without replay`,async t=>{
  const f=await fixture(t);let broken=true,controlHandle;
  const client=async(args,settings)=>{
    if(args[0]==='capabilities')return {state:'available',backend:'ssh'};
    if(broken&&args[0]==='download'&&args[2].includes('/lease-control/')&&args[2].endsWith('/admission.json')){
      if(unknown)throw new Error('native transfer acknowledgement lost');
      return {state:'failed',exit:255};
    }
    const result=await f.context.client(args,settings);
    if(args[0]==='run'&&!args.includes('--npu')&&args[args.indexOf('--file')+1].endsWith('command.sh'))controlHandle=result.handle;
    return result;
  };
  const environment=await createEnvironment(f.options,{...f.context,client});
  await environment.prepare({project:{source:f.source},attempt:{id:'download-retry'}});
  assert.equal((await environment.execute('payload-once',f.source,{timeout:5})).settled,false);
  assert.ok(controlHandle);
  const runs=f.calls.filter(a=>a[0]==='run').length;
  const restarted=await createEnvironment(f.options,{...f.context,client});
  assert.equal(await restarted.reconcileLease(),false);
  let migrated=false;
  assert.equal((await restarted.reconcileReservation(0,{stopped:async()=>true,
    migrateClosedControl:async()=>{migrated=true;return true;}})).closed,false);
  assert.equal(migrated,false,'an existing native journal is not a legacy migration');
  assert.equal(f.calls.filter(a=>a[0]==='run').length,runs);
  broken=false;const recovered=await restarted.reconcileLease();
  assert.equal(recovered,!unknown);
  if(unknown){const pointer=JSON.parse(await readFile(join(f.context.instancePath,'.ascend-command-lease.json'),'utf8'));
    assert.equal(JSON.parse(await readFile(pointer.receipt,'utf8')).controlHandle,controlHandle);}
  assert.equal(f.calls.filter(a=>a[0]==='run').length,runs);
});

test('control-started with an unknown payload submission is held instead of redispatched',async t=>{
  const f=await fixture(t),held=await reserveDevice(f.context.instancePath,0,f.context.leaseDirectory);
  await held.update({controlStarted:true});
  await held.event('execution-submitting',{submission:'unknown-control',kind:'run',controlPayload:true,accountingExcluded:true});
  await writeFile(join(f.context.instancePath,'.ascend-command-lease.json'),JSON.stringify({receipt:held.receipt,token:held.token,pool:held.record.pool,physicalDevice:0}));
  assert.equal(await f.environment.reconcileLease(),false);
  assert.equal(f.calls.filter(a=>a[0]==='run').length,0);
  assert.equal(JSON.parse(await readFile(held.receipt,'utf8')).controlStarted,true);
});

test('explicit closed legacy-control migration requires an absent journal and native closure',async t=>{
  const f=await fixture(t),held=await reserveDevice(f.context.instancePath,0,f.context.leaseDirectory);
  await held.update({owner:null,controlStarted:true,controlHandle:'remote:fixture:job:legacy-control'});
  let checked=false;
  const result=await f.environment.reconcileReservation(0,{stopped:async()=>true,
    migrateClosedControl:async record=>{checked=true;return record.token===held.token&&record.controlHandle==='remote:fixture:job:legacy-control';}});
  assert.equal(result.closed,true);assert.equal(checked,true);
  assert.ok(f.calls.some(a=>a[0]==='observe'&&a[1]==='remote:fixture:job:legacy-control'));
});


for(const setup of ['none','closed','terminal','unknown','live'])test(`pre-payload control setup ${setup} retains closure gates before host reconciliation`,async t=>{
  const f=await fixture(t),held=await reserveDevice(f.context.instancePath,0,f.context.leaseDirectory);
  await held.update({owner:{...held.record.owner,start:'stopped-owner'},controlStarted:true,controlHandle:null});
  if(setup!=='none'){
    await held.event('execution-submitting',{submission:'setup',kind:'upload',accountingExcluded:true,purpose:'device-lease-control'});
    if(setup==='closed')await held.event('execution-submitted',{submission:'setup',closed:true});
    if(['terminal','live'].includes(setup))await held.event('execution-submitted',{
      submission:'setup',closed:false,handle:'remote:fixture:transfer:old-setup'});
  }
  const environment=await createEnvironment(f.options,{...f.context,client:async(args,settings)=>{
    if(setup==='live'&&args[0]==='observe'&&args[1]==='remote:fixture:transfer:old-setup')return {state:'running'};
    return f.context.client(args,settings);
  }});
  const result=await environment.reconcileReservation(0);
  const fenced=['unknown','live'].includes(setup);
  assert.equal(result.closed,!fenced);
  const controls=f.calls.filter(args=>args[0]==='run'&&args[args.indexOf('--file')+1].endsWith('command.sh'));
  assert.equal(controls.length,fenced?0:1);
  if(fenced){
    assert.equal(result.reason,setup==='unknown'?'submission-unknown':'execution-unresolved');
    const retained=JSON.parse(await readFile(held.receipt,'utf8'));
    assert.equal(retained.controlStarted,true);assert.equal(retained.controlHandle,null);
  }else{
    await assert.rejects(readFile(held.receipt),{code:'ENOENT'});
    assert.ok(f.events.some(e=>e.type==='execution-submitting'&&e.purpose==='device-lease-control'&&e.controlPayload));
  }
});

for(const terminalState of ['failed','completed','observation-unavailable'])test(`cleanup with missing receipt and native ${terminalState} preserves payload identity`,async t=>{
  const f=await fixture(t);let failedControl,failedAdmission;
  const client=async(args,settings)=>{
    if(args[0]==='observe' && args[1]===failedControl)
      return {state:terminalState,...(terminalState==='completed'?{exit:0}:{reason:{message:'open failed'}})};
    if(args[0]==='download' && (args[2]===failedAdmission||failedAdmission?.endsWith('/'+args[2])))return {state:'failed',exit:1,handle:'remote:fixture:transfer:missing-admission'};
    const result=await f.context.client(args,settings);
    if(!failedControl && args[0]==='run' && !args.includes('--npu') && args[args.indexOf('--file')+1].endsWith('command.sh')){
      failedControl=result.handle;
      const script=await readFile(args[args.indexOf('--file')+1],'utf8');
      const configuration=JSON.parse(script.slice(script.lastIndexOf(" '{")+2,-2).replaceAll("'\\''","'"));
      failedAdmission=configuration.env.RSI_LEASE_RECEIPT;
    }
    return result;
  };
  const environment=await createEnvironment(f.options,{...f.context,client});
  await environment.prepare({project:{source:f.source},attempt:{id:'failed-cleanup'}});
  const result=await environment.execute('payload-once',f.source,{timeout:5});
  assert.equal(result.settled,terminalState==='failed');
  assert.equal(result.exitCode,0);
  if(terminalState==='failed'){
    assert.equal(f.commands.length,1,'cleanup retry does not rerun payload');
    const [entry]=await fs.readdir(join(f.context.leaseDirectory,'.closed'));
    const closed=JSON.parse(await readFile(join(f.context.leaseDirectory,'.closed',entry,'lease.json'),'utf8'));
    assert.ok(closed.handles.includes(failedControl));
    assert.notEqual(closed.controlHandle,failedControl);
    return;
  }
  const pointer=JSON.parse(await readFile(join(f.context.instancePath,'.ascend-command-lease.json'),'utf8'));
  const retained=JSON.parse(await readFile(pointer.receipt,'utf8'));
  const payloadHandle=retained.payloadHandle;
  assert.ok(payloadHandle);assert.notEqual(payloadHandle,failedControl);
  assert.ok(retained.handles.includes(failedControl),'failed cleanup remains observable in the journal');
  const restarted=await createEnvironment(f.options,{...f.context,client});
  const first=await restarted.reconcileLease();
  // A live/unknown control or a completed control with unavailable receipt must
  // keep its identity. Only a positively failed cleanup is replaceable.
  assert.equal(first,false);
  const current=JSON.parse(await readFile(pointer.receipt,'utf8'));
  assert.equal(current.controlHandle,failedControl);
  assert.equal(f.calls.filter(a=>a[0]==='run'&&!a.includes('--npu')&&a[a.indexOf('--file')+1].endsWith('command.sh')).length,1);
});

test('repeated positively failed cleanup stops at elapsed observation allowance without replaying payload',async t=>{
  const f=await fixture(t),controls=new Set(),receipts=new Set();
  const client=async(args,settings)=>{
    if(args[0]==='observe'&&controls.has(args[1]))return {state:'failed',exit:null,reason:{message:'open failed'}};
    if(args[0]==='download'&&[...receipts].some(path=>path===args[2]||path.endsWith('/'+args[2])))return {state:'failed',exit:1,handle:'remote:fixture:transfer:missing-admission'};
    const result=await f.context.client(args,settings);
    if(args[0]==='run'&&!args.includes('--npu')&&args[args.indexOf('--file')+1].endsWith('command.sh')){
      controls.add(result.handle);
      const script=await readFile(args[args.indexOf('--file')+1],'utf8');
      const configuration=JSON.parse(script.slice(script.lastIndexOf(" '{")+2,-2).replaceAll("'\\''","'"));
      receipts.add(configuration.env.RSI_LEASE_RECEIPT);
    }
    return result;
  };
  const environment=await createEnvironment({...f.options,probeObservationSeconds:1},{...f.context,client});
  await environment.prepare({project:{source:f.source},attempt:{id:'cleanup-allowance'}});
  const result=await environment.execute('payload-once',f.source,{timeout:5});
  assert.equal(result.settled,false);assert.equal(result.exitCode,0);
  assert.equal(f.commands.length,1);
  assert.ok(controls.size>=2,'retry happens before elapsed allowance is exhausted');
  const pointer=JSON.parse(await readFile(join(f.context.instancePath,'.ascend-command-lease.json'),'utf8'));
  const retained=JSON.parse(await readFile(pointer.receipt,'utf8'));
  for(const handle of controls)assert.ok(retained.handles.includes(handle));
});

for(const stage of ['payload-observe','control-prepare','control-observe','admission-read','upload-parent','transfer'])
test(`one independent cleanup deadline bounds a hung ${stage} RPC and retains its fence`,async t=>{
  const f=await fixture(t),controls=new Set(),parents=new Set(),receipts=new Set();
  const heldTransfer='remote:fixture:transfer:held-cleanup-upload';
  let payload,observations=0,aborted=0,blocking=true,freshCancellation=0;
  const {createServer,connect}=await import('node:net'),sockets=new Set();
  const server=createServer(socket=>{sockets.add(socket);socket.on('close',()=>sockets.delete(socket));});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  t.after(async()=>{for(const socket of sockets)socket.destroy();await new Promise(resolve=>server.close(resolve));});
  const hang=signal=>{
    assert.ok(signal,'cleanup must pass its independent signal into every RPC');
    return new Promise((_,reject)=>{
      // A real open transport keeps the event loop alive, as a CPL request does.
      const socket=connect(server.address().port,'127.0.0.1');socket.on('error',reject);
      const stop=()=>{aborted++;socket.destroy();reject(signal.reason);};
      if(signal.aborted)stop();else signal.addEventListener('abort',stop,{once:true});
    });
  };
  const client=async(args,{signal}={})=>{
    if(stage==='upload-parent'&&args[0]==='capabilities'&&f.commands.length)return {state:'available',backend:'ssh'};
    if(blocking&&stage==='transfer'&&args[0]==='upload'&&f.commands.length)
      return {state:'running',handle:heldTransfer};
    if(blocking&&args[0]==='observe'&&(parents.has(args[1])||args[1]===heldTransfer))return hang(signal);
    if(blocking&&args[0]==='cancel'&&(parents.has(args[1])||args[1]===heldTransfer)) {
      freshCancellation++;return hang(signal);
    }
    if(blocking&&stage==='payload-observe'&&args[0]==='observe'&&args[1]===payload&&++observations>1)return hang(signal);
    if(blocking&&stage==='control-prepare'&&args[0]==='capabilities'&&f.commands.length)return hang(signal);
    if(blocking&&stage==='control-observe'&&args[0]==='observe'&&controls.has(args[1]))return hang(signal);
    if(blocking&&stage==='admission-read'&&args[0]==='download'&&[...receipts].some(path=>path===args[2]||path.endsWith('/'+args[2])))return hang(signal);
    const result=await f.context.client(args,{signal});
    if(stage==='upload-parent'&&args[0]==='run'&&args[args.indexOf('--file')+1].endsWith('upload-parent.sh'))parents.add(result.handle);
    if(args[0]==='run'&&args[args.indexOf('--file')+1].endsWith('command.sh')) {
      if(args.includes('--npu'))payload=result.handle;
      else {
        controls.add(result.handle);
        const script=await readFile(args[args.indexOf('--file')+1],'utf8');
        const configuration=JSON.parse(script.slice(script.lastIndexOf(" '{")+2,-2).replaceAll("'\\''","'"));
        receipts.add(configuration.env.RSI_LEASE_RECEIPT);
      }
    }
    return result;
  };
  const environment=await createEnvironment({...f.options,probeObservationSeconds:1},{...f.context,client});
  await environment.prepare({project:{source:f.source},attempt:{id:'bounded-cleanup'}});
  const start=performance.now(),result=await environment.execute('payload-once',f.source,{timeout:5});
  assert.equal(result.settled,false);assert.equal(result.exitCode,0);assert.ok(aborted>0);
  assert.equal(freshCancellation,0,'cleanup cancellation must not start a fresh RPC budget after its deadline');
  assert.ok(performance.now()-start<5000,'a hung cleanup must not receive fresh per-RPC observation windows');
  assert.equal(f.commands.length,1);assert.equal(f.events.some(event=>event.type==='device-lease-released'),false);
  const pointer=JSON.parse(await readFile(join(f.context.instancePath,'.ascend-command-lease.json'),'utf8'));
  const retained=JSON.parse(await readFile(pointer.receipt,'utf8'));
  assert.ok(retained.handles.includes(payload));
  for(const handle of controls)assert.ok(retained.handles.includes(handle));
  for(const handle of parents)assert.ok(retained.handles.includes(handle));
  if(stage==='transfer')assert.ok(retained.handles.includes(heldTransfer));
  blocking=false;
  if(stage==='admission-read') {
    assert.ok(retained.submissions.some(row=>!row.closed&&!row.handle),'an interrupted unacknowledged transfer remains unknown');
    assert.equal(await environment.reconcileLease(),false,'transport recovery is not proof that the unacknowledged transfer closed');
  } else assert.equal(await environment.reconcileLease(),true,'an expired control context must not poison a later observation');
  assert.equal(f.commands.length,1,'observing late completion must never replay the payload');
});

test('admitted actor command receives persistent scratch and runtime cache defaults',async t=>{
  const f=await fixture(t);
  const bound=await createEnvironment(f.options,f.context);
  await bound.prepare({project:{id:'setup',source:f.source},attempt:{id:'setup'}});
  f.runWorkload();
  const resultFile=join(f.context.instancePath,'actor-setup.json');
  const command=`python3 -c 'import json,os,pathlib; p=pathlib.Path(os.environ["RSI_SCRATCH_DIR"])/"probe"; p.write_text("retained"); json.dump({k:os.environ[k] for k in ["RSI_SCRATCH_DIR","RSI_SETUP_REPORT","TRITON_CACHE_DIR"]},open("${resultFile}","w"))'`;
  assert.equal((await bound.execute(command,f.source,{timeout:5})).exitCode,0);
  const paths=JSON.parse(await readFile(resultFile,'utf8'));
  assert.equal(JSON.parse(await readFile(paths.RSI_SETUP_REPORT,'utf8')).logical_device,0);
  assert.equal((await bound.execute('test "$(cat "$RSI_SCRATCH_DIR/probe")" = retained',f.source,{timeout:5})).exitCode,0);
  assert.equal(paths.RSI_SCRATCH_DIR.startsWith(f.options.workspacePath),true);
});

for(const released of [true,false,'true'])test('only a positive released receipt skips remote fence reconciliation: '+released,async t=>{
 const f=await fixture(t);f.setEnabled([0]);f.setReleaseProof(released);
 await f.environment.prepare({project:{source:f.source},attempt:{id:'release'}});
 const result=await f.environment.execute('true',f.source,{timeout:10});assert.equal(result.settled,true);
 const controls=f.events.filter(e=>e.type==='execution-submitting'&&e.controlPayload);
 assert.equal(controls.length,released===true?0:1);
 assert.ok(f.events.some(e=>e.type==='device-lease-released'));
});

test('released payload evidence cannot close an unresolved owned handle',async t=>{
 const f=await fixture(t);f.setEnabled([0]);f.setReleaseProof(true);let reconcile=false;
 const environment=await createEnvironment(f.options,{...f.context,exclude:async(label,fn)=>{const result=await fn();if(label==='admission'&&result?.result?.handle){reconcile=true;f.disconnect(true);}return result;}});
 await environment.prepare({project:{source:f.source},attempt:{id:'unknown-release'}});
 const result=await environment.execute('true',f.source,{timeout:10});assert.equal(reconcile,true);assert.equal(result.settled,false);
 assert.equal(f.events.some(e=>e.type==='device-lease-released'),false);
 f.disconnect(false);assert.equal(await environment.reconcileLease(),true);
 assert.equal(f.events.filter(e=>e.type==='execution-submitting'&&e.controlPayload).length,0);
});

test('a released receipt from another token still requires the existing host reconciliation',async t=>{
 const f=await fixture(t);f.setEnabled([0]);f.setReleaseProof(true);let reconciling=false;
 f.transformAdmission(value=>reconciling&&value.released===true?{...value,token:'other-owner'}:value);
 const environment=await createEnvironment(f.options,{...f.context,exclude:async(label,fn)=>{const result=await fn();if(label==='admission'&&result?.result?.handle)reconciling=true;return result;}});
 await environment.prepare({project:{source:f.source},attempt:{id:'foreign-release'}});
 const result=await environment.execute('true',f.source,{timeout:10});assert.equal(result.settled,true);
 assert.equal(f.events.filter(e=>e.type==='execution-submitting'&&e.controlPayload).length,1);
 assert.ok(f.events.some(e=>e.type==='device-lease-released'));
});
