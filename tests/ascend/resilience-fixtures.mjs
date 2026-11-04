import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {performance} from 'node:perf_hooks';
import {schedule,target,serve,connect} from './resilience/fixtures.mjs';
import {runCases,Blocked} from './resilience/runner.mjs';
import {cases} from './resilience/workspace.case.mjs';
import {caseClient} from './resilience/transport.case.mjs';
import {createEnvironment} from '../../integrations/environments/index.mjs';

async function directory(t){const root=await mkdtemp(join(tmpdir(),'resilience-fixture-'));
  t.after(()=>rm(root,{recursive:true,force:true}));return root;}
function deadlineClock(t){
  let now=0;
  t.mock.method(performance,'now',()=>now);
  t.mock.timers.enable({apis:['setTimeout']});
  return {elapse:milliseconds=>{now+=milliseconds;},timers:t.mock.timers};
}
test('seeded burst schedules replay and abort delayed requests',async()=>{
  const collect=async seed=>{const s=schedule({seed,profile:'bursty'});
    for(let i=0;i<4;i++)await s.at('readiness');
    assert.ok(s.events.every(e=>Number.isFinite(e.observedMilliseconds)&&e.observedMilliseconds>=0));
    return s.events.map(({observedMilliseconds,...chosen})=>chosen);};
  assert.deepEqual(await collect(1),await collect(1));
  assert.notDeepEqual(await collect(1),await collect(2));
  const abort=new AbortController();abort.abort();
  await assert.rejects(schedule({profile:'slow'}).at('upload',{},abort.signal),{name:'AbortError'});
});
test('lost reply after receipt retains one real job across observer connections',async t=>{
  const root=await mkdtemp(join(tmpdir(),'resilience-receipt-'));
  const s=schedule({faults:[{point:'run.after-receipt',occurrence:1,disconnect:true}]});
  const backend=await target({root,scheduler:s}),server=await serve(backend);
  t.after(async()=>{await server.close();await rm(root,{recursive:true,force:true});});
  const script=join(root,'command.sh'),receipt=join(root,'receipt.json');
  await writeFile(script,'printf retained; sleep 0.1\n');
  await assert.rejects(connect(server.url)(['run','fixture','--file',script,'--receipt-file',receipt]));
  const {handle}=JSON.parse(await readFile(receipt,'utf8'));
  await Promise.all([...backend.jobs.values()].map(j=>j.done));
  const observer=connect(server.url);
  assert.deepEqual(await observer(['observe',handle]),{state:'completed',exit:0});
  assert.equal((await observer(['logs',handle])).content,'retained');
  assert.equal(backend.calls.filter(c=>c[0]==='run').length,1);
});
test('actual SDK local edit reaches separate target with same-handle observation',async t=>{
  const root=await directory(t);
  const report=await runCases(cases,{output:join(root,'report'),tier:'local',cleanupSeconds:0,deadlineSeconds:15});
  assert.equal(report.status,'passed');assert.equal(report.results[0].evidence.dispatches,1);
});
test('runner preserves blocked/failed outcomes, cleanup and does not retry',async t=>{
  const root=await directory(t);let calls=0,closed=0;
  const report=await runCases([
    {id:'blocked',tier:'local',run:async()=>{throw new Blocked('Missing fixture credential');}},
    {id:'failed',tier:'local',run:async c=>{calls++;c.onCleanup(()=>closed++);throw Error('Actual failure');}},
  ],{output:join(root,'report'),tier:'local',cleanupSeconds:0,deadlineSeconds:5});
  assert.equal(report.status,'failed');assert.deepEqual(report.results.map(r=>r.status),['blocked','failed']);
  assert.equal(calls,1);assert.equal(closed,1);
  assert.deepEqual(JSON.parse(await readFile(join(root,'report','report.json'),'utf8')).results,report.results);
});
test('round deadline aborts current work and leaves subsequent work incomplete',async t=>{
  const root=await directory(t),clock=deadlineClock(t);let entered=false,second=false,closed=false;
  const report=await runCases([
    {id:'slow',tier:'local',run:async c=>{
      entered=true;c.onCleanup(()=>{closed=true;});assert.equal(c.signal.aborted,false);
      const cancelled=new Promise(resolve=>c.signal.addEventListener('abort',resolve,{once:true}));
      clock.elapse(80);clock.timers.tick(80);await cancelled;c.signal.throwIfAborted();
    }},
    {id:'next',tier:'local',run:async()=>{second=true;}},
  ],{output:join(root,'report'),tier:'local',cleanupSeconds:0,deadlineSeconds:.08});
  assert.equal(entered,true);assert.equal(report.status,'incomplete');assert.equal(second,false);assert.equal(closed,true);
  assert.deepEqual(report.results.map(r=>r.status),['incomplete','incomplete']);
});
test('all tier cannot pass with missing required tiers and unknown cases never run',async t=>{
  const root=await directory(t);let called=false;
  const list=[{id:'one',tier:'local',run:async()=>{called=true;}}];
  await assert.rejects(runCases(list,{caseId:'missing',output:join(root,'missing')}));assert.equal(called,false);
  const report=await runCases(list,{output:join(root,'report'),cleanupSeconds:0,deadlineSeconds:5});
  assert.equal(report.status,'incomplete');assert.equal(called,true);
  assert.equal(report.results.filter(r=>r.status==='incomplete').length,2);
});
test('blocking CPU work cannot pass after its elapsed deadline',async t=>{
  const root=await directory(t),clock=deadlineClock(t);let iterations=0,closed=false;
  const report=await runCases([{id:'cpu',tier:'local',run:async c=>{
    c.onCleanup(()=>{closed=true;});assert.equal(c.signal.aborted,false);
    const end=performance.now()+40;
    while(performance.now()<end){iterations++;clock.elapse(1);}
    // No timer callback can run during this synchronous CPU work.
    assert.equal(c.signal.aborted,false);
  }}],{output:join(root,'report'),tier:'local',cleanupSeconds:0,deadlineSeconds:.02});
  assert.ok(iterations>0);assert.ok(performance.now()>20);assert.equal(closed,true);
  assert.equal(report.status,'incomplete');assert.equal(report.results[0].status,'incomplete');
});
test('public command discovers real case modules without circular imports',async t=>{
  const root=await directory(t),entry=new URL('./resilience/cli.mjs',import.meta.url).pathname;
  const {stdout}=await promisify(execFile)(process.execPath,[entry,'--tier','local','--case','workspace-snapshots',
    '--deadline-seconds','310','--output',join(root,'report')]);
  const report=JSON.parse(stdout);assert.equal(report.status,'passed');
  assert.equal(report.results[0].evidence.nativeBashEdit,true);
});
test('public command accepts a relative report directory from outside the checkout',async t=>{
  const root=await directory(t),entry=new URL('./resilience/cli.mjs',import.meta.url).pathname;
  const {stdout}=await promisify(execFile)(process.execPath,[entry,'--tier','local','--case','workspace-snapshots',
    '--deadline-seconds','310','--output','relative/report'],{cwd:root});
  const report=JSON.parse(stdout);assert.equal(report.status,'passed');
  assert.equal(report.output,join(root,'relative','report'));assert.ok(report.revision);
  assert.equal(JSON.parse(await readFile(join(report.output,'report.json'),'utf8')).status,'passed');
});
for(const phase of ['preflight','upload'])test(`case cancellation reaches native ${phase} without caller settings`,async t=>{
  const root=await directory(t),abort=new AbortController();let dispatched=false;
  const source=join(root,'source');await mkdir(source);await writeFile(join(source,'answer.py'),'value=42\n');
  const calls=[];
  const scoped=caseClient({signal:abort.signal,readInstructions:async()=>{},native:async(args,{signal})=>{
    calls.push(args[0]);
    if(args[0]===phase){queueMicrotask(()=>abort.abort());await delay(1000,undefined,{signal});}
    if(args[0]==='capabilities')return {state:'available'};
    if(args[0]==='run')dispatched=true;
    return {state:'completed'};
  }});
  const environment=await createEnvironment({mode:'cpl',target:'fixture',workspacePath:root,remoteDirectory:'owned'},
    {instancePath:root,client:scoped.client,event:async()=>{},exclude:async(_,fn)=>fn()});
  const prepare=()=>environment.prepare({project:{source,id:'p'},attempt:{id:'a'}});
  if(phase==='preflight')await assert.rejects(prepare());
  else {await prepare();await assert.rejects(environment.execute('python3 answer.py',source,{signal:abort.signal}));}
  assert.ok(calls.includes(phase));
  assert.equal(dispatched,false);
});
test('cleanup preserves unknown evidence and still stops subsequent owned work',async t=>{
  const root=await directory(t),abort=new AbortController();abort.abort();
  const stopped=[],scoped=caseClient({native:async()=>({}),signal:abort.signal,readInstructions:async()=>{}});
  scoped.handles.add('unknown');scoped.handles.add('running');
  await assert.rejects(scoped.close({observe:async h=>({state:h==='unknown'?'unknown':'running'}),
    stop:async h=>{stopped.push(h);return {state:'stopped'};}},root));
  assert.deepEqual(stopped,['running']);
  const proof=JSON.parse(await readFile(join(root,'closure.json'),'utf8'));
  assert.deepEqual(proof.map(o=>o.state),['unknown','stopped']);
});
test('cleanup has a separate bounded signal after the case is cancelled',async t=>{
  const root=await directory(t),abort=new AbortController();abort.abort();
  const scoped=caseClient({signal:abort.signal,readInstructions:async()=>{},native:async(_,{signal})=>{
    await delay(1000,undefined,{signal});return {state:'completed'};
  }});
  scoped.handles.add('retained');
  await assert.rejects(scoped.close({observe:async h=>scoped.client(['observe',h])},root,{milliseconds:10}));
  assert.equal(JSON.parse(await readFile(join(root,'closure.json'),'utf8'))[0].state,'unknown');
});
test('cleanup receives the remaining round deadline and preserves its failure',async t=>{
  const root=await directory(t),clock=deadlineClock(t);let entered=false,aborted=false;
  const report=await runCases([{id:'cleanup',tier:'local',run:async c=>{
    entered=true;
    c.onCleanup(async({signal,remainingSeconds})=>{
      assert.equal(signal.aborted,false);assert.equal(remainingSeconds,.04);
      const cancelled=new Promise(resolve=>signal.addEventListener('abort',resolve,{once:true}));
      clock.elapse(40);clock.timers.tick(40);await cancelled;
      aborted=signal.aborted;signal.throwIfAborted();
    });
  }}],{output:join(root,'report'),tier:'local',deadlineSeconds:.04,cleanupSeconds:0});
  assert.equal(entered,true);assert.equal(aborted,true);assert.equal(report.status,'failed');
  assert.equal(report.results[0].cleanupErrors.length,1);
});
