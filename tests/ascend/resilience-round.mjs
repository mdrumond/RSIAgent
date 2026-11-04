import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm,readFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {runCases} from './resilience/runner.mjs';

async function output(t){const dir=await mkdtemp(join(tmpdir(),'resilience-round-'));
  t.after(()=>rm(dir,{recursive:true,force:true}));return dir;}
test('local failure prevents provider and remote dispatch despite registration order',async t=>{
  const called=[],make=(id,tier)=>({id,tier,async run(){called.push(id);if(tier==='local')throw Error('oracle failed');}});
  const report=await runCases([make('remote','remote'),make('paid','deepseek'),make('cpu','local')],
    {output:await output(t),deadlineSeconds:10,cleanupSeconds:1});
  assert.deepEqual(called,['cpu']);assert.equal(report.status,'failed');
  assert.deepEqual(report.results.map(r=>r.status),['failed','blocked','blocked']);
});
test('missing mandatory local scenario blocks paid work and is incomplete',async t=>{
  let submitted=false;
  const report=await runCases([{id:'cpu',tier:'local',run:async()=>{}},
    {id:'paid',tier:'deepseek',run:async()=>{submitted=true;}},{id:'remote',tier:'remote',run:async()=>{}}],
  {output:await output(t),deadlineSeconds:10,cleanupSeconds:1,requiredCases:[{id:'missing',tier:'local'}]});
  assert.equal(submitted,false);assert.equal(report.status,'incomplete');
  assert.equal(report.results.find(r=>r.id==='missing').status,'incomplete');
});
test('external interruption retains cleanup evidence and never starts next case',async t=>{
  const abort=new AbortController(),dir=await output(t);let next=false,closed=false;
  const report=await runCases([{id:'active',tier:'local',async run(ctx){
    ctx.onCleanup(async({signal,remainingSeconds})=>{
      assert.equal(signal.aborted,false);assert.ok(remainingSeconds>0);closed=true;
    });
    const active=JSON.parse(await readFile(join(dir,'report.json'),'utf8'));
    assert.equal(active.activeCase,'active');abort.abort(Error('operator stop'));ctx.signal.throwIfAborted();
  }},{id:'next',tier:'local',run:async()=>{next=true;}}],
  {output:dir,tier:'local',signal:abort.signal,deadlineSeconds:10,cleanupSeconds:1});
  assert.equal(closed,true);assert.equal(next,false);assert.equal(report.status,'incomplete');
  assert.equal(JSON.parse(await readFile(join(dir,'report.json'),'utf8')).activeCase,undefined);
});
test('passing prerequisites execute all tiers and report mandatory coverage',async t=>{
  const called=[],requiredCases=['local','deepseek','remote'].map(tier=>({id:tier,tier}));
  const report=await runCases([...requiredCases].reverse().map(item=>({...item,run:async()=>{called.push(item.id);}})),
    {output:await output(t),requiredCases,deadlineSeconds:10,cleanupSeconds:1});
  assert.equal(report.status,'passed');assert.deepEqual(called,['local','deepseek','remote']);
  assert.deepEqual(report.coverage.completed,requiredCases.map(c=>c.id));
});
