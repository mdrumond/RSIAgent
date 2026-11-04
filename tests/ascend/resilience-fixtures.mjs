import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {schedule,target,serve,connect} from './resilience/fixtures.mjs';
import {runCases,Blocked} from './resilience/runner.mjs';
import {cases} from './resilience/workspace.case.mjs';

async function directory(t){const root=await mkdtemp(join(tmpdir(),'resilience-fixture-'));
  t.after(()=>rm(root,{recursive:true,force:true}));return root;}
test('seeded burst schedules replay and abort delayed requests',async()=>{
  const collect=async seed=>{const s=schedule({seed,profile:'bursty'});
    for(let i=0;i<4;i++)await s.at('readiness');return s.events;};
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
  const root=await directory(t);let second=false,closed=false;
  const report=await runCases([
    {id:'slow',tier:'local',run:async c=>{c.onCleanup(()=>{closed=true;});await delay(1000,undefined,{signal:c.signal});}},
    {id:'next',tier:'local',run:async()=>{second=true;}},
  ],{output:join(root,'report'),tier:'local',cleanupSeconds:0,deadlineSeconds:.08});
  assert.equal(report.status,'incomplete');assert.equal(second,false);assert.equal(closed,true);
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
  const root=await directory(t);
  const report=await runCases([{id:'cpu',tier:'local',run:async()=>{
    const end=performance.now()+40;while(performance.now()<end){};
  }}],{output:join(root,'report'),tier:'local',cleanupSeconds:0,deadlineSeconds:.02});
  assert.equal(report.status,'incomplete');assert.equal(report.results[0].status,'incomplete');
});
test('public command discovers real case modules without circular imports',async t=>{
  const root=await directory(t),entry=new URL('./resilience/cli.mjs',import.meta.url).pathname;
  const {stdout}=await promisify(execFile)(process.execPath,[entry,'--tier','local','--case','workspace-snapshots',
    '--deadline-seconds','310','--output',join(root,'report')]);
  const report=JSON.parse(stdout);assert.equal(report.status,'passed');
  assert.equal(report.results[0].evidence.nativeBashEdit,true);
});
