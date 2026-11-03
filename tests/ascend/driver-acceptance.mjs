import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {setup,rows,until} from './driver_setup.mjs';

test('CLI original failure -> smaller canonical investigation -> memory -> original retry; unrelated work proceeds',
  {timeout:120000},async t=>{
    const f=await setup(t,'recover',['recover','unrelated']);
    const requested=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8')).projects.find(p=>p.id==='recover');
    await f.run('wave');
    const frozen=JSON.parse(rows(f.root,"SELECT spec FROM projects WHERE id='recover'")[0].spec);
    assert.equal(frozen.goal,requested.goal);assert.equal(frozen.verifier.command,requested.verifier.command);
    let status=await f.run('status');
    assert.deepEqual(Object.fromEntries(status.projects.map(p=>[p.id,p.status])),{recover:'failed',unrelated:'passed'});
    assert.equal(status.attempts.length,2);
    const original=status.attempts.find(a=>a.project==='recover');
    assert.equal(rows(f.root,'SELECT * FROM recovery_queue')[0].eligible_wave,2);
    await f.run('wave');
    const diagnostic=rows(f.root,"SELECT * FROM recovery_attempts WHERE kind='investigation'")[0];
    assert.ok(diagnostic);
    const proposal=JSON.parse(await readFile(join(f.root,'attempts',diagnostic.attempt,'project.json'),'utf8'));
    assert.match(proposal.goal,/SMALL/);
    assert.notEqual(proposal.source,JSON.parse(rows(f.root,"SELECT spec FROM projects WHERE id='recover'")[0].spec).source);
    assert.equal((await f.run('status')).projects.find(p=>p.id==='recover').status,'failed');
    assert.equal(rows(f.root,'SELECT kind,eligible_wave FROM recovery_queue')[0].kind,'retry');
    assert.equal(rows(f.root,'SELECT eligible_wave FROM recovery_queue')[0].eligible_wave,3);
    await f.run('wave');
    status=await f.run('status');
    assert.ok(status.projects.every(p=>p.status==='passed' && !p.frozen));
    assert.equal(status.attempts.filter(a=>a.project==='recover').length,3);
    assert.equal(status.attempts.find(a=>a.id===original.id).outcome,'failed');
    assert.deepEqual(JSON.parse(rows(f.root,"SELECT spec FROM projects WHERE id='recover'")[0].spec),frozen);
    const kinds=rows(f.root,"SELECT DISTINCT kind FROM knowledge_scopes WHERE collection LIKE 'lesson-%'",'knowledge.sqlite').map(r=>r.kind);
    assert.deepEqual(kinds.sort(),['good','negative']);
    const lessons=rows(f.root,'SELECT * FROM recovery_lessons');
    assert.ok(lessons.every(row=>row.recorded===1));
    for(const row of lessons) {
      const lesson=JSON.parse(row.data);
      const evidence=lesson.evidence[0].path;
      const events=rows(f.root,`SELECT data FROM events WHERE attempt='${lesson.attempt_id}' AND type='verification'`);
      const verifierFiles=events.map(event=>JSON.parse(event.data).evidence.correctness);
      const nativeFile=rows(f.root,`SELECT session FROM attempts WHERE id='${lesson.attempt_id}'`)[0].session;
      assert.ok(verifierFiles.includes(evidence) || evidence===nativeFile);
      assert.ok((await readFile(evidence,'utf8')).length>0);
    }
    const delivered=rows(f.root,"SELECT * FROM knowledge_events WHERE operation='record_lesson'",'knowledge.sqlite');
    assert.equal(new Set(delivered.map(e=>e.attempt_id)).size,4);
    for(const attempt of status.attempts.filter(a=>a.outcome==='passed')) {
      const events=rows(f.root,`SELECT type,data FROM events WHERE attempt='${attempt.id}'`);
      assert.equal(events.filter(e=>e.type==='stage-completed').length,8);
      assert.ok(events.some(e=>e.type==='accepted'));
      assert.ok(events.some(e=>e.type==='execution-handle'));
      assert.equal(events.filter(e=>e.type==='knowledge-retrieved').length,4);
    }
    assert.ok(f.server.requests.some(r=>r.messages.some(m=>m.role==='tool')));
    f.pass();
  });

test('CLI freezes at the configured unsuccessful recovery-round limit', {timeout:120000},async t=>{
  const f=await setup(t,'freeze',['freeze']);
  const campaign=await f.run('campaign','--detach');
  assert.equal(campaign.submitted,true);
  await until(async()=>(await f.run('status')).projects.find(p=>p.id==='freeze')?.frozen===1);
  await until(()=>{
    const owner=rows(f.root,"SELECT value FROM control WHERE key='owner'")[0];
    return owner && JSON.parse(owner.value)===null;
  });
  const status=await f.run('status');
  assert.equal(status.projects[0].status,'failed');assert.equal(status.projects[0].frozen,1);
  assert.equal(status.attempts.length,3);
  await f.run('wave');assert.equal((await f.run('status')).attempts.length,3);
  const completions=rows(f.root,"SELECT data FROM events WHERE type='recovery-completed'").map(e=>JSON.parse(e.data));
  assert.deepEqual(completions.map(e=>e.failed_rounds),[1,2]);
  assert.equal(rows(f.root,'SELECT * FROM recovery_lessons').length,3);
  f.pass();
});

test('CLI stop/resume preserves the interrupted attempt, native session and remaining budget', {timeout:120000},async t=>{
  const f=await setup(t,'interrupt',['interrupt']);
  const wave=f.start('wave');
  const before=await until(()=>{
    try {
      const [attempt]=rows(f.root,'SELECT * FROM attempts');
      return attempt?.elapsed>=1 && rows(f.root,"SELECT * FROM events WHERE type='execution-handle'").length ? attempt : null;
    }catch{return null;}
  });
  await f.run('stop');await wave.output;
  const stopped=(await f.run('status')).attempts[0];
  assert.equal(stopped.id,before.id);assert.equal(stopped.outcome,null);assert.ok(stopped.elapsed>=before.elapsed);
  const session=rows(f.root,'SELECT session FROM attempts')[0].session;
  assert.ok(session);
  await f.run('resume');
  const status=await f.run('status');
  assert.equal(status.attempts.length,1);assert.equal(status.attempts[0].id,before.id);
  assert.equal(status.attempts[0].outcome,'passed');assert.ok(status.attempts[0].elapsed>=stopped.elapsed);
  assert.equal(rows(f.root,'SELECT session FROM attempts')[0].session,session);
  assert.equal(rows(f.root,"SELECT * FROM events WHERE type='job-admitted'").length,1);
  f.pass();
});
