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
outputs={'board':f'NPU ID: {device}\\nChip Name: Ascend950',
'health':'Health Status: '+row.get('status','OK')+'\\nError Code: 0',
'usages':f'HBM Capacity(MB): 1000\\nHBM Usage Rate(%): {row.get("memory",0)}\\nAicore Usage Rate(%): 0\\nAivector Usage Rate(%): 0\\nNPU Utilization(%): 0\\nAicube Usage Rate(%): 0',
'proc-mem':'Process id: 123' if row.get('busy') else 'No process in device.',
'current-fault-event':'Fault Event Count: 0'}
print(outputs[query])
`, { mode: 0o700 });
  const calls = [], events = [], jobs = new Map();
  let runWorkload = false, onSync = async () => {}, running = false, queued = 0;
  let waitSignal;
  let enabled = [0,1,2,3], disconnect = false, sequence = 0, clockSkew = 0, hideSubmission = false, lastProbe;
  const client = async (args,{signal}={}) => {
    calls.push(args);
    if (args[0] === 'npu') return { state: 'completed', profiles: [{ target: 'fixture', devices: enabled.map(id => ({ id, enabled: true })) }] };
    if (args[0] === 'capabilities') return { state: 'available' };
    if (args[0] === 'preflight') return { state: 'completed' };
    if (args[0] === 'upload') return { state: 'completed' };
    if (args[0] === 'download') { await writeFile(args[3], ''); return { state: 'completed' }; }
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
      if (file.endsWith('command.sh') && runWorkload) {
        const script = await readFile(file,'utf8');
        const config = JSON.parse(script.slice(script.lastIndexOf(" '{")+2,-2).replaceAll("'\\''","'"));
        try { await promisify(execFile)('bash',['-c',config.command],{env:{...process.env,...config.env,PATH:bin+':'+process.env.PATH,ASCEND_TEST_TELEMETRY:telemetry}}); }
        catch(error) { exit = error.code; }
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
  const context = { instancePath, client, sdk: { createLocalBashOperations: () => ({}) },
    event: async (type,data) => events.push({ type,...data }), exclude: async (_,fn) => fn() };
  const options = { mode: 'cpl', target: 'fixture', runtime: 'registered', instructionFile,
    workspacePath: '/registry/workspace', remoteDirectory: 'owned', pollMs: 0, probeSeconds: 3,
    health: { product: 'a5', chip: null, query_timeout_seconds: .1, ...health } };
  const environment = await createEnvironment(options, context);
  return { environment, options, context, source, calls, events, rows, telemetry,
    setEnabled: ids => { enabled = ids; }, disconnect: value => { disconnect = value; }, setClockSkew: seconds => { clockSkew = seconds; }, hideSubmission:value=>{hideSubmission=value;}, lastProbe:()=>lastProbe,
    queued:value=>{queued=value;}, wait:fn=>{waitSignal=fn;},runWorkload: () => {runWorkload=true;}, onSync: fn => {onSync=fn;}, running: value => {running=value;},
    update: () => writeFile(telemetry, JSON.stringify(rows)) };
}

test('fresh parser-backed discovery filters permission, busy, Alarm, and unresponsive devices', async t => {
  const f = await fixture(t);
  const first = await f.environment.resources({ target: 'fixture' });
  assert.deepEqual(first.filter(row => row.available).map(row => row.metadata.physicalDevice), [0]);
  assert.ok(first.every(row => row.id.startsWith('ascend-') && row.metadata.healthSha256.length === 64));
  f.rows[0].busy = true; await f.update();
  assert.equal((await f.environment.resources()).some(row => row.available), false);
  f.setEnabled([]);
  assert.deepEqual(await f.environment.resources(), []);
  assert.deepEqual(await f.environment.resources({ target: 'other' }), []);
});

test('ordinary Alarm/busy opt-ins retain policy, parallel availability stays idle, timing is healthy idle', async t => {
  const f = await fixture(t, { allow_alarm: true, allow_busy: true });
  const ordinary = await f.environment.resources();
  assert.deepEqual(ordinary.filter(row => row.available).map(row => row.metadata.physicalDevice), [0,1]);
  assert.equal(ordinary.find(row => row.metadata.physicalDevice === 2).metadata.admitted, true);
  f.rows[2].memory = 50; await f.update();
  const full = await f.environment.resources();
  assert.equal(full.find(row => row.metadata.physicalDevice === 2).metadata.admitted, false);
  const timing = await f.environment.resources({ purpose: 'timing' });
  assert.deepEqual(timing.filter(row => row.available).map(row => row.metadata.physicalDevice), [0]);
});

for (const physical of [0, 1]) test(`lease ${physical}: assigned opaque lease rechecks and forwards physical declaration, logical-zero mask and elapsed remainder`, async t => {
  const f = await fixture(t, { allow_alarm: true });
  const lease = (await f.environment.resources()).find(row => row.available && row.metadata.physicalDevice === physical);
  const bound = await createEnvironment(f.options, { ...f.context, resource: lease });
  await bound.prepare({ project: { id:'p', source:f.source }, attempt:{ id:'a' } });
  const result = await bound.execute('kernel-command', f.source, { timeout:3, env:{ ASCEND_PHYSICAL_DEVICE:'7', ASCEND_RT_VISIBLE_DEVICES:'7' } });
  assert.equal(result.settled, true);
  const run = f.calls.find(args => args[0] === 'run' && args[args.indexOf('--file')+1].endsWith('command.sh'));
  assert.equal(run[run.indexOf('--npu')+1], String(physical));
  assert.equal(run[run.indexOf('--runtime')+1], 'registered');
  const script = await readFile(run[run.indexOf('--file')+1], 'utf8');
  const config = JSON.parse(script.slice(script.lastIndexOf(" '{")+2,-2).replaceAll("'\\''","'"));
  assert.equal(config.env.ASCEND_PHYSICAL_DEVICE, String(physical));
  assert.equal(config.env.ASCEND_RT_VISIBLE_DEVICES, String(physical));
  assert.equal(config.env.ASCEND_VISIBLE_DEVICES, String(physical));
  assert.ok(config.timeout > 2.8 && config.timeout <= 3);
  await assert.rejects(bound.execute('work',f.source,{ resourceId:'0' }), /opaque|Assigned/);
  f.rows[physical].busy = true; await f.update();
  await assert.rejects(bound.execute('work',f.source), /admission refused/);
});

test('probe disconnect persists exact handle across restart and never resubmits unknown work', async t => {
  const f = await fixture(t);
  f.disconnect(true);
  await assert.rejects(f.environment.resources(), /unresolved/);
  const handle = f.events.find(row => row.type === 'execution-handle').handle;
  const restarted = await createEnvironment(f.options,f.context);
  await assert.rejects(restarted.resources(), /unresolved/);
  assert.equal(f.calls.filter(args => args[0] === 'run').length, 1);
  assert.ok(f.calls.filter(args => args[0] === 'observe').every(args => args[1] === handle));
  f.disconnect(false);
  assert.equal((await restarted.resources()).filter(row => row.available).length, 1);
  assert.equal(f.calls.filter(args => args[0] === 'run').length, 1);
});

test('actual user instruction file is required rather than fabricated defaults', async t => {
  const f = await fixture(t);
  await assert.rejects(createEnvironment({ ...f.options, instructionFile:undefined },f.context), /instructionFile/);
  await assert.rejects(createEnvironment({ ...f.options, instructionFile:join(f.source,'missing') },f.context), /ENOENT/);
});


test('remote wall-clock skew does not replace local elapsed freshness', async t => {
  const f = await fixture(t); f.setClockSkew(3600);
  const resources = await f.environment.resources();
  assert.equal(resources.filter(row => row.available).length, 1);
  assert.ok(resources[0].metadata.observedAt > resources[0].metadata.checkedAt + 3500);
});


test('explicit project device affinity is a recorded filter and never falls back', async t => {
  const f = await fixture(t, { allow_alarm: true });
  const bound = await createEnvironment({ ...f.options, deviceAffinity:[1] },f.context);
  const rows = await bound.resources();
  assert.deepEqual(rows.filter(row => row.available).map(row => row.metadata.physicalDevice), [1]);
  assert.deepEqual(rows[0].metadata.deviceAffinity,[1]);
  assert.ok(f.events.some(row => row.type === 'resource-policy' && row.deviceAffinity?.[0] === 1));
  f.setEnabled([0]); assert.deepEqual(await bound.resources(),[]);
});


for (const failure of ['policy','busy','unknown']) test(`core probe-only reconciliation: ${failure}`, async t => {
  const f=await fixture(t);
  const lease=(await f.environment.resources()).find(row=>row.available);
  const file=join(f.context.instancePath,'state.sqlite');
  let state=new State(file),depth=0,now=0,exclusions=0;
  t.after(()=>state.close());
  state.add({id:'p',goal:'numeric',actor_seconds:30,source:f.source,verifier:{files:[],command:'true'}});
  const attempt=state.admit('p',{pid:2147483647,start:'gone',boot:'gone'});
  state.checkpoint(attempt.id,2);
  state.event(attempt.id,'execution-starting',{operation:'probe-only',startElapsed:2});
  const clock=new ActorClock(30,2,()=>now);
  const context={...f.context,resource:lease,
    exclude:(label,fn)=>clock.exclude(label,async()=>{
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

test('waiting behind another execution still consumes the caller timeout',async t=>{
  const f=await fixture(t),lease=(await f.environment.resources()).find(row=>row.available);
  const bound=await createEnvironment(f.options,{...f.context,resource:lease});
  await bound.prepare({project:{id:'p',source:f.source},attempt:{id:'queue'}});
  const first=bound.execute('one',f.source,{timeout:3});
  const queued=assert.rejects(bound.execute('two',f.source,{timeout:.01}),/elapsed/);
  assert.equal((await first).exitCode,0);await queued;
  assert.equal(f.calls.filter(args=>args[0]==='run'&&args[args.indexOf('--file')+1].endsWith('command.sh')).length,1);
});


test('workload guard after sync refuses changed health before arbitrary Bash', async t => {
  const f=await fixture(t),lease=(await f.environment.resources()).find(row=>row.available);
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
  const f=await fixture(t),lease=(await f.environment.resources()).find(row=>row.available);
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
  const f=await fixture(t),lease=(await f.environment.resources()).find(row=>row.available);
  let depth=0,now=0,exclusions=0;const clock=new ActorClock(30,2,()=>now);
  const bound=await createEnvironment(f.options,{...f.context,resource:lease,
    exclude:(label,fn)=>clock.exclude(label,async()=>{
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
  assert.equal(f.calls.filter(a=>a[0]==='run').length,runs+3);
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
  const f=await fixture(t);f.disconnect(true);await assert.rejects(f.environment.resources(),/unresolved/);
  const old=f.lastProbe().handle;await stalePending(f);
  f.disconnect(false);f.queued(1);
  assert.ok((await f.environment.resources()).some(row=>row.available));
  assert.equal(f.calls.filter(a=>a[0]==='run').length,1);
  assert.equal(f.events.find(e=>e.type==='resource-probe-completed').handle,old);
  assert.equal(f.events.some(e=>e.type==='resource-probe-stale'),false);
});

test('probe cancellation retains receipt and never submits replacement',async t=>{
  const f=await fixture(t),lease=(await f.environment.resources()).find(row=>row.available);
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
