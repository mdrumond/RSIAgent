import assert from 'node:assert/strict';
import {readFile,writeFile,readdir,stat} from 'node:fs/promises';
import {join,relative,isAbsolute} from 'node:path';
import {fileURLToPath} from 'node:url';
import {Blocked} from './runner.mjs';
import {schedule} from './fixtures.mjs';

async function prereqs(ctx) {
  const [major,minor]=process.versions.node.split('.').map(Number);
  if(major<22||(major===22&&minor<23))throw new Blocked('Campaign CLI requires Node >=22.23');
  const root=fileURLToPath(new URL('../../../',import.meta.url));
  const rel=relative(root,ctx.directory);
  if(!rel.startsWith('..')&&!isAbsolute(rel))throw new Blocked('Campaign output must be outside source machinery');
  try{
    import.meta.resolve('@earendil-works/pi-coding-agent');
    await stat(join(root,'node_modules'));
  }
  catch{throw new Blocked('Existing native Pi installation unavailable');}
}

export const cases=[{id:'campaign-recovery',tier:'local',async run(ctx){
  await prereqs(ctx);
  const {campaignFixture,digest}=await import('./campaign-fixture.mjs');
  const timing=schedule(ctx),proof=[];
  ctx.onCleanup(async()=>writeFile(join(ctx.directory,'schedule.json'),JSON.stringify(timing.events,null,2)));
  const enter=async name=>{
    await timing.at('campaign.scenario',{name},ctx.signal);
    await ctx.event({type:'campaign-scenario',name,seed:ctx.seed,profile:ctx.profile});
  };
  const events=(f,id)=>f.rows('SELECT type,data FROM events WHERE attempt=?','state.sqlite',[id])
    .map(e=>({type:e.type,data:JSON.parse(e.data)}));
  const project=f=>JSON.parse(f.rows("SELECT spec FROM projects WHERE id='recover'")[0].spec);
  async function hashes(spec) {
    return Promise.all(spec.verifier.files.map(async path=>({path,sha256:digest(await readFile(path))})));
  }
  function canonical(f,attempt) {
    assert.equal(attempt.outcome,'passed');assert.ok(attempt.session);assert.ok(attempt.elapsed>0);
    const trace=events(f,attempt.id);
    assert.equal(trace.filter(e=>e.type==='accepted').length,1);
    assert.equal(trace.filter(e=>e.type==='stage-completed').length,8);
    assert.equal(trace.filter(e=>e.type==='knowledge-retrieved').length,4);
    assert.ok(trace.some(e=>e.type==='verification'&&e.data.passed));
  }
  async function acknowledged(f) {
    const lessons=f.rows('SELECT * FROM recovery_lessons');
    assert.ok(lessons.length>0);assert.ok(lessons.every(l=>l.recorded===1));
    for(const row of lessons) {
      const lesson=JSON.parse(row.data);
      assert.equal(lesson.attempt_id,row.attempt);
      for(const ref of lesson.evidence)if(ref.path)assert.equal(digest(await readFile(ref.path)),ref.sha256);
    }
    return lessons;
  }

  await enter('next-wave-recovery');
  const recovery=await campaignFixture(ctx,'recover',['recover','unrelated']);
  const before=project(recovery),frozenHashes=await hashes(before);
  const firstWave=await recovery.run('wave');
  assert.equal(firstWave.wave,1);
  const original=recovery.rows("SELECT * FROM attempts WHERE project='recover'")[0];
  assert.equal(original.outcome,'failed');assert.ok(original.session);assert.ok(original.elapsed>0);
  const originalSession=digest(await readFile(original.session));
  const unrelated=recovery.rows("SELECT * FROM attempts WHERE project='unrelated'")[0];
  canonical(recovery,unrelated);
  await acknowledged(recovery);
  let queue=recovery.rows('SELECT * FROM recovery_queue')[0];
  assert.equal(queue.kind,'investigation');assert.equal(queue.eligible_wave,2);
  await recovery.run('wave');
  const investigation=recovery.rows("SELECT * FROM recovery_attempts WHERE kind='investigation'")[0];
  assert.equal(investigation.wave,2);
  const child=recovery.rows('SELECT * FROM attempts WHERE id=?','state.sqlite',[investigation.attempt])[0];
  canonical(recovery,child);
  const childSpec=JSON.parse(await readFile(join(recovery.instance,'attempts',child.id,'project.json'),'utf8'));
  assert.notEqual(childSpec.source,before.source);assert.notEqual(childSpec.goal,before.goal);
  const verifierRelative=relative(childSpec.source,childSpec.verifier.files[0]);
  assert.ok(verifierRelative==='..'||verifierRelative.startsWith('../'),
    'Investigation verifier remains outside editable source');
  const completion=events(recovery,child.id).find(e=>e.type==='recovery-completed');
  assert.equal(completion.data.lesson.useful,true);
  await acknowledged(recovery);
  assert.equal(recovery.rows("SELECT status FROM projects WHERE id='recover'")[0].status,'failed');
  queue=recovery.rows('SELECT * FROM recovery_queue')[0];
  assert.equal(queue.kind,'retry');assert.equal(queue.eligible_wave,3);
  await recovery.run('wave');
  const retry=recovery.rows("SELECT * FROM recovery_attempts WHERE kind='retry'")[0];
  assert.equal(retry.wave,3);
  canonical(recovery,recovery.rows('SELECT * FROM attempts WHERE id=?','state.sqlite',[retry.attempt])[0]);
  assert.equal(recovery.rows('SELECT * FROM recovery_queue').length,0);
  assert.ok(recovery.rows('SELECT status,frozen FROM projects').every(p=>p.status==='passed'&&!p.frozen));
  assert.equal(recovery.rows("SELECT * FROM attempts WHERE project='recover'").length,3);
  assert.deepEqual(recovery.rows('SELECT * FROM attempts WHERE id=?','state.sqlite',[original.id])[0],original);
  assert.equal(digest(await readFile(original.session)),originalSession);
  assert.equal(project(recovery).goal,before.goal);
  assert.deepEqual(project(recovery).verifier,before.verifier);
  assert.deepEqual(await hashes(project(recovery)),frozenHashes);
  const recoveryLessons=await acknowledged(recovery);assert.equal(recoveryLessons.length,4);
  proof.push({...await recovery.evidence(),scenario:'recovery',original:original.id,
    investigation:child.id,retry:retry.attempt,frozenHashes});

  await enter('five-failed-rounds');
  const freeze=await campaignFixture(ctx,'freeze',['freeze','unrelated']);
  await freeze.config({curriculum:{phase1:['freeze']}});
  const frozenBefore=JSON.parse(freeze.rows("SELECT spec FROM projects WHERE id='freeze'")[0].spec);
  const limitHashes=await hashes(frozenBefore);
  await freeze.run('campaign');
  const failed=freeze.rows("SELECT * FROM projects WHERE id='freeze'")[0];
  assert.equal(failed.status,'failed');assert.equal(failed.frozen,1);
  const parentAttempts=freeze.rows("SELECT * FROM attempts WHERE project='freeze'");
  assert.equal(parentAttempts.length,6);assert.ok(parentAttempts.every(a=>a.outcome==='failed'));
  const rounds=freeze.rows("SELECT data FROM events WHERE type='recovery-completed' ORDER BY sequence")
    .map(e=>JSON.parse(e.data));
  assert.deepEqual(rounds.map(r=>r.failed_rounds),[1,2,3,4,5]);
  assert.ok(rounds.slice(0,-1).every(r=>!r.frozen));assert.equal(rounds.at(-1).frozen,true);
  assert.equal(freeze.rows('SELECT * FROM recovery_queue').length,0);
  assert.equal(freeze.rows("SELECT status FROM projects WHERE id='unrelated'")[0].status,'queued');
  await acknowledged(freeze);
  await freeze.config({curriculum:{phase1:['freeze','unrelated']}});
  await freeze.run('wave');
  canonical(freeze,freeze.rows("SELECT * FROM attempts WHERE project='unrelated'")[0]);
  assert.deepEqual(freeze.rows("SELECT * FROM attempts WHERE project='freeze'"),parentAttempts);
  assert.deepEqual(JSON.parse(freeze.rows("SELECT spec FROM projects WHERE id='freeze'")[0].spec),frozenBefore);
  assert.deepEqual(await hashes(frozenBefore),limitHashes);
  assert.deepEqual(freeze.rows('SELECT id,frozen FROM projects ORDER BY id').map(p=>[p.id,p.frozen]),
    [['freeze',1],['unrelated',0]]);
  await acknowledged(freeze);
  proof.push({...await freeze.evidence(),scenario:'five-round-freeze',rounds:rounds.map(r=>r.failed_rounds)});

  await enter('delivery-restart');
  const memory=await campaignFixture(ctx,'memory',['unrelated'],{loseDelivery:true});
  await memory.run('wave');
  const settled=memory.rows('SELECT * FROM attempts');assert.equal(settled.length,1);
  canonical(memory,settled[0]);
  const pending=memory.rows('SELECT * FROM recovery_lessons');
  assert.equal(pending.length,1);assert.equal(pending[0].recorded,0);
  const receipt=JSON.parse(await readFile(join(memory.directory,'delivery-lost.json'),'utf8'));
  assert.equal(receipt.attempt_id,settled[0].id);
  const collection=memory.rows('SELECT * FROM knowledge_scopes WHERE collection=?','knowledge.sqlite',[receipt.collection]);
  assert.equal(collection.length,1);
  const sessionHash=digest(await readFile(settled[0].session));
  const storedLessons=await readdir(join(memory.instance,'knowledge.sqlite.lessons'));
  assert.equal(storedLessons.filter(p=>p.endsWith('.json')).length,1);
  // New public CLI process drains the durable outbox, without another actor/summary.
  const requestsBefore=memory.server.requests.length;
  await memory.run('resume');
  assert.deepEqual(memory.rows('SELECT * FROM attempts'),settled);
  assert.equal(digest(await readFile(settled[0].session)),sessionHash);
  assert.equal(memory.server.requests.length,requestsBefore);
  assert.equal(memory.rows('SELECT recorded FROM recovery_lessons')[0].recorded,1);
  assert.deepEqual(memory.rows('SELECT * FROM knowledge_scopes WHERE collection=?','knowledge.sqlite',[receipt.collection]),collection);
  assert.deepEqual(await readdir(join(memory.instance,'knowledge.sqlite.lessons')),storedLessons);
  await acknowledged(memory);
  await memory.run('resume');
  assert.deepEqual(memory.rows('SELECT * FROM attempts'),settled);
  assert.equal(memory.server.requests.length,requestsBefore);
  proof.push({...await memory.evidence(),scenario:'delivery-restart',collection:receipt.collection,
    attempt:settled[0].id,sessionHash,elapsed:settled[0].elapsed});
  await writeFile(join(ctx.directory,'schedule.json'),JSON.stringify(timing.events,null,2));
  await writeFile(join(ctx.directory,'campaign-proof.json'),JSON.stringify(proof,null,2));
  return {proof:'campaign-proof.json',schedule:'schedule.json',scenarios:proof};
}}];
