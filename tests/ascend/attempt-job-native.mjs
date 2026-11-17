import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,writeFile,rm,access} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {setup,rows} from './driver_setup.mjs';
import {State} from '../../controller/state.mjs';
import {readAttemptJob,writeAttemptJob} from '../../cli/attempt-job.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';

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

test('a pinned retry resumes and verifies without requiring its missing legacy parent manifest',{timeout:90000},async t=>{
  const f=await setup(t,'phase2',['unrelated']),state=new State(join(f.root,'state.sqlite'));
  let original,retry,before;
  try{
    const queue=new RecoveryQueue(state),owner={pid:2147483647,start:'old',boot:'source'};
    original=queue.claim('unrelated',0,owner).attempt;
    state.checkpoint(original.id,12.75,null);state.finish(original.id,'failed');
    state.event(original.id,'failure-settlement',{closed:true});
    const evidence=join(f.root,'retained-original.log'),text='Retained original failure before retry admission.\n';
    await writeFile(evidence,text);
    queue.recordOriginal(original.id,{passed:false,findings:'Retained failed original',
      evidence:[{path:evidence,sha256:createHash('sha256').update(text).digest('hex')}]},0);
    for(const lesson of queue.lessons())queue.acknowledgeLesson(lesson.id);
    state.event(original.id,'postprocess-completed',{});
    // Seed the already-qualified retry queue; the test exercises its retained
    // admission and native resume, not historical investigation construction.
    state.db.prepare("UPDATE recovery_queue SET kind='retry' WHERE project='unrelated'").run();
    retry=queue.claim('unrelated',1,owner,'local');
    const config=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));
    await writeAttemptJob(state,f.root,retry.attempt.id,{...retry,wave:1,mode:'phase1',options:{},
      skillVersion:config.skills.activeVersion,resource:{id:'local'}});
    before=state.attempt(original.id);
  }finally{state.close();}
  await f.run('resume');
  assert.equal(rows(f.root,`SELECT outcome FROM attempts WHERE id='${retry.attempt.id}'`)[0].outcome,'passed');
  assert.ok(rows(f.root,`SELECT 1 FROM events WHERE attempt='${retry.attempt.id}' AND type='verification' AND json_extract(data,'$.passed')=1`).length);
  assert.ok(rows(f.root,`SELECT 1 FROM events WHERE attempt='${retry.attempt.id}' AND type='postprocess-completed'`).length);
  assert.deepEqual(rows(f.root,`SELECT * FROM attempts WHERE id='${original.id}'`)[0],before);
  assert.equal(rows(f.root,`SELECT count(*) n FROM events WHERE attempt='${original.id}' AND type='attempt-metadata-unavailable'`)[0].n,0);
  await assert.rejects(access(join(f.root,'attempts',original.id,'job.json')));
  f.pass();
});
