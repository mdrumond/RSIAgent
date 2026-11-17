import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,rm,access} from 'node:fs/promises';
import {join} from 'node:path';
import {setup,rows} from './driver_setup.mjs';
import {State} from '../../controller/state.mjs';
import {readAttemptJob} from '../../cli/attempt-job.mjs';

test('public CLI completes unrelated native work with a missing legacy finalizer and recovers its own published job',{timeout:90000},async t=>{
  const f=await setup(t,'phase2',['unrelated']);
  const state=new State(join(f.root,'state.sqlite'));let legacy,before;
  try{
    state.add({id:'legacy-blocked',goal:'Retain real failed history',actor_seconds:60});
    legacy=state.admit('legacy-blocked',{pid:2147483647,start:'old',boot:'source'});
    state.checkpoint(legacy.id,12.75,null);state.finish(legacy.id,'failed');
    state.db.prepare("UPDATE projects SET frozen=1 WHERE id='legacy-blocked'").run();
    state.event(legacy.id,'failure-settlement',{closed:true});before=state.attempt(legacy.id);
  }finally{state.close();}
  await f.run('wave');
  const status=await f.run('status');
  assert.equal(status.projects.find(p=>p.id==='unrelated').status,'passed');
  assert.equal(status.metadataBlocked[0].attempt,legacy.id);
  assert.equal(status.metadataBlocked[0].reason,'no-trusted-checkpoint');
  assert.equal(rows(f.root,`SELECT count(*) n FROM events WHERE attempt='${legacy.id}' AND type='attempt-metadata-unavailable'`)[0].n,1);
  const native=rows(f.root,"SELECT * FROM attempts WHERE project='unrelated'")[0];
  assert.ok(rows(f.root,`SELECT 1 FROM events WHERE attempt='${native.id}' AND type='postprocess-completed'`).length);
  assert.ok(rows(f.root,`SELECT 1 FROM events WHERE attempt='${native.id}' AND type='execution-handle'`).length);
  const file=join(f.root,'attempts',native.id,'job.json'),original=await readFile(file,'utf8');
  await rm(file);
  const restored=new State(join(f.root,'state.sqlite'));
  try{
    assert.deepEqual(await readAttemptJob(restored,f.root,native.id),JSON.parse(original));
    assert.deepEqual(restored.attempt(legacy.id),before);
  }finally{restored.close();}
  assert.equal(await readFile(file,'utf8'),original);
  await assert.rejects(access(join(f.root,'attempts',legacy.id,'job.json')));
  f.pass();
});
