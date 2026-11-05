import test from 'node:test';
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {setup,rows,until} from './driver_setup.mjs';

test('pause grace closes command and preserves the same attempt, session and elapsed allocation for resume',{timeout:45000},async t=>{
  const f=await setup(t,'interrupt',['interrupt'],{install:false});
  const input=join(f.root,'pause-settings.json');
  await writeFile(input,JSON.stringify({pause_grace_seconds:.2}));await f.run('config','--config',input);
  const running=f.start('wave');
  await until(()=>rows(f.root,"SELECT 1 FROM events WHERE type='execution-handle'").length);
  const [before]=rows(f.root,'SELECT * FROM attempts');
  const paused=await f.run('pause');assert.equal(paused.pause_requested,true);
  await running.output;
  const [saved]=rows(f.root,'SELECT * FROM attempts');
  assert.equal(saved.id,before.id);assert.equal(saved.outcome,null);
  assert.equal(saved.session,before.session);assert.ok(saved.elapsed>=before.elapsed);
  await f.run('resume');
  const attempts=rows(f.root,'SELECT * FROM attempts');assert.equal(attempts.length,1);
  assert.equal(attempts[0].id,saved.id);assert.equal(attempts[0].session,saved.session);
  assert.ok(attempts[0].elapsed>=saved.elapsed);assert.equal(attempts[0].outcome,'passed');
});
