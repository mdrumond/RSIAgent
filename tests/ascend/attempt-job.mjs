import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,access,chmod} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {runtimeStatus} from '../../controller/runtime-status.mjs';
import {writeAttemptJob,publishAttemptJob,materializeAttemptJob,readAttemptJob,availableAttemptJob} from '../../cli/attempt-job.mjs';
import {operate} from '../../cli/runtime.mjs';

async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'attempt-job-')),state=new State(join(root,'state.sqlite'));
  t.after(async()=>{state.close();await rm(root,{recursive:true,force:true});});
  state.add({id:'p',goal:'retained original',actor_seconds:90});
  const attempt=state.admit('p',{pid:2147483647,start:'old',boot:'source'});
  state.checkpoint(attempt.id,17.25,'retained-session');
  const job={attempt:{...state.attempt(attempt.id)},projectId:'p',kind:'retry',origin_attempt:'original',wave:21,
    mode:'phase1',actor_seconds:90,skillVersion:'accepted',options:{continuous:true},
    resource:{id:'host-slot',metadata:{pool:'retained-pool',hostSlot:0}}};
  return {root,state,attempt,job,file:join(root,'attempts',attempt.id,'job.json')};
}

test('missing and stale file mirrors recover exact trusted options and preserve attempt elapsed state',async t=>{
  const f=await fixture(t);await writeAttemptJob(f.state,f.root,f.attempt.id,f.job);
  const before=f.state.attempt(f.attempt.id),bytes=await readFile(f.file,'utf8');
  await rm(f.file);assert.deepEqual(await readAttemptJob(f.state,f.root,f.attempt.id),f.job);
  assert.equal(await readFile(f.file,'utf8'),bytes);assert.deepEqual(f.state.attempt(f.attempt.id),before);
  const next={...f.job,resource:{id:'another-slot',metadata:{pool:'retained-pool',hostSlot:1}}};
  f.state.db.exec('BEGIN IMMEDIATE');publishAttemptJob(f.state,f.attempt.id,next);f.state.db.exec('COMMIT');
  // A crash before file publication must not revive the old resource assignment.
  assert.equal(JSON.parse(await readFile(f.file,'utf8')).resource.id,'host-slot');
  assert.deepEqual(await readAttemptJob(f.state,f.root,f.attempt.id),next);
  assert.equal(JSON.parse(await readFile(f.file,'utf8')).resource.id,'another-slot');
  assert.deepEqual(f.state.attempt(f.attempt.id),before);
});

test('failed atomic file publication retains the trusted checkpoint and database rollback retains its old value',async t=>{
  const f=await fixture(t);await mkdir(f.file,{recursive:true});
  await assert.rejects(writeAttemptJob(f.state,f.root,f.attempt.id,f.job));
  await rm(f.file,{recursive:true});
  assert.deepEqual(await readAttemptJob(f.state,f.root,f.attempt.id),f.job);
  f.state.db.exec('BEGIN IMMEDIATE');publishAttemptJob(f.state,f.attempt.id,{...f.job,mode:'phase2'});f.state.db.exec('ROLLBACK');
  assert.equal((await readAttemptJob(f.state,f.root,f.attempt.id)).mode,'phase1');
});

test('checkpoint materialization failures block only the affected attempt and retry the same trusted bytes',async t=>{
  if(process.getuid?.()===0){t.skip('root bypasses directory write permissions');return;}
  const f=await fixture(t);await writeAttemptJob(f.state,f.root,f.attempt.id,f.job);await rm(f.file);
  const directory=join(f.root,'attempts',f.attempt.id),before=f.state.attempt(f.attempt.id);
  await chmod(directory,0o555);
  try{
    assert.equal(await availableAttemptJob(f.state,f.root,f.attempt.id),null);
    assert.equal(runtimeStatus(f.state).metadataBlocked[0].reason,'checkpoint-materialization-failed');
    assert.deepEqual(f.state.attempt(f.attempt.id),before);
  }finally{await chmod(directory,0o700);}
  assert.deepEqual(await availableAttemptJob(f.state,f.root,f.attempt.id),f.job);
  assert.deepEqual(runtimeStatus(f.state).metadataBlocked,[]);
});

test('a portable destination never recovers pre-binding metadata but accepts its retained destination file',async t=>{
  const f=await fixture(t);await writeAttemptJob(f.state,f.root,f.attempt.id,f.job);
  f.state.event(f.attempt.id,'portable-bound',{binding:'destination'});await rm(f.file);
  await assert.rejects(readAttemptJob(f.state,f.root,f.attempt.id),e=>e.reason==='checkpoint-precedes-binding');
  const destination={...f.job,resource:{id:'destination-slot'}};
  await materializeAttemptJob(f.root,f.attempt.id,destination);
  assert.deepEqual(await readAttemptJob(f.state,f.root,f.attempt.id),destination);
  await writeAttemptJob(f.state,f.root,f.attempt.id,destination);await rm(f.file);
  assert.deepEqual(await readAttemptJob(f.state,f.root,f.attempt.id),destination);
});

test('legacy missing metadata stays visibly blocked without fabricating work, releasing custody or repeating failures',async t=>{
  const f=await fixture(t);new RecoveryQueue(f.state);
  f.state.db.prepare('INSERT INTO resource_leases VALUES (?,?)').run('reserved',f.attempt.id);
  const before=f.state.attempt(f.attempt.id);
  for(let i=0;i<4;i++)assert.equal(await availableAttemptJob(f.state,f.root,f.attempt.id),null);
  assert.equal(f.state.queryEvents(f.attempt.id,{types:['attempt-metadata-unavailable']}).length,1);
  assert.equal(f.state.failures(f.attempt.id).length,1);assert.deepEqual(f.state.attempt(f.attempt.id),before);
  assert.equal(f.state.db.prepare('SELECT attempt FROM resource_leases').get().attempt,f.attempt.id);
  assert.equal(runtimeStatus(f.state).metadataBlocked[0].attempt,f.attempt.id);
  await assert.rejects(access(f.file));
  await materializeAttemptJob(f.root,f.attempt.id,f.job);
  assert.deepEqual(await availableAttemptJob(f.state,f.root,f.attempt.id),f.job);
  assert.deepEqual(runtimeStatus(f.state).metadataBlocked,[]);
});

test('scheduler isolates missing finalizer metadata instead of aborting unrelated scheduling',async t=>{
  const f=await fixture(t);f.state.finish(f.attempt.id,'failed');
  f.state.event(f.attempt.id,'failure-settlement',{closed:true});
  const environment=join(f.root,'environment.mjs');
  await writeFile(environment,'export function createEnvironment(){return {resources:async()=>[]};}');
  const config={format:1,root:f.root,mode:'phase1',projects:[],concurrency:1,environment:{module:environment,options:{}}};
  await writeFile(join(f.root,'instance.json'),JSON.stringify(config));
  const before=f.state.attempt(f.attempt.id),result=await operate('wave',config);
  assert.deepEqual(result.results,[]);assert.deepEqual(f.state.attempt(f.attempt.id),before);
  assert.equal(runtimeStatus(f.state).metadataBlocked[0].reason,'no-trusted-checkpoint');
  assert.equal(f.state.queryEvents(f.attempt.id,{types:['postprocess-completed','scheduler-error']}).length,0);
});
