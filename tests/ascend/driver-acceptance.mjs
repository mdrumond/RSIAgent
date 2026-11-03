import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, mkdir, writeFile, readFile, cp, rm } from 'node:fs/promises';
import { join, resolve, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { setTimeout as delay } from 'node:timers/promises';
import { DatabaseSync } from 'node:sqlite';
import { stages } from '../../controller/workflow.mjs';
import { ownerStopped } from '../../controller/lifecycle.mjs';
import { provider } from './driver_fixture.mjs';
const fixture = resolve('tests/ascend/driver_fixture.py');
const python = process.env.DRIVER_PYTHON ?? 'python3.12';
const quote = text => "'" + text.replaceAll("'", "'\\''") + "'";
const rows = (root,sql,file='state.sqlite') => {
  const db = new DatabaseSync(join(root,file),{readOnly:true});
  try {return db.prepare(sql).all();} finally {db.close();}
};
async function until(check,timeout=15000) {
  const end = Date.now()+timeout;
  while (Date.now()<end) {const value=await check();if(value)return value;await delay(50);}
  throw new Error('Timed out waiting for observable driver transition');
}
async function setup(t,scenario,ids) {
  const root = await mkdtemp(join(process.env.DRIVER_ARTIFACTS ?? tmpdir(),'driver-acceptance-'));
  const instance = join(root,'instance'), server = await provider(scenario,{nativeSession:async()=>
    rows(instance,'SELECT session FROM attempts WHERE session IS NOT NULL ORDER BY rowid DESC LIMIT 1')[0]?.session});
  const env = {...process.env,DRIVER_FIXTURE_KEY:'fixture-only',
    PATH:dirname(python)+':'+process.env.PATH,PYTHONPATH:resolve('.')};
  let sequence=0,okay=false;
  const execute=(executable,args) => {
    const child=spawn(executable,args,{env,stdio:['ignore','pipe','pipe']});
    let stdout='',stderr='';child.stdout.on('data',d=>stdout+=d);child.stderr.on('data',d=>stderr+=d);
    const output=(async()=>{
      const [code]=await once(child,'close');
      const name=join(root,`process-${sequence++}`);
      await writeFile(name+'.stdout',stdout);await writeFile(name+'.stderr',stderr);
      assert.equal(code,0,`${executable} ${args.join(' ')}\n${stderr}\nartifacts=${root}`);
      return stdout.trim() ? JSON.parse(stdout.trim().split('\n').at(-1)) : null;
    })();
    return {child,output};
  };
  let cli=resolve('cli/main.mjs');
  const start=(command,...args)=>execute(process.execPath,[cli,command,'--instance',instance,...args]);
  const run=async(command,...args)=>start(command,...args).output;
  t.after(async()=>{
    try {
      const receipt=JSON.parse(await readFile(join(instance,'knowledge-service.json'),'utf8'));
      if(!await ownerStopped(receipt.owner))process.kill(receipt.owner.pid,'SIGTERM');
    }catch(error){if(error.code!=='ENOENT' && error.code!=='ESRCH')throw error;}
    await server.close();
    assert.deepEqual(server.failures,[],`Fixture protocol failures; artifacts=${root}`);
    if(okay && !process.env.DRIVER_ARTIFACTS)await rm(root,{recursive:true,force:true});
    else t.diagnostic(`Retained actual CLI/native session evidence: ${root}`);
  });
  const baseline=join(root,'baseline.sqlite');
  // Only the embedding backend is a fixture; indexing and serving use production KDB.
  const seed=spawn(python,[fixture,'baseline',baseline],{env,stdio:['ignore','pipe','pipe']});
  let seedError='';seed.stderr.on('data',d=>seedError+=d);
  assert.equal((await once(seed,'close'))[0],0,seedError);
  const skills=join(root,'skills');await mkdir(skills);
  const map=Object.fromEntries([...stages,'global-kdb'].map(stage=>[stage,stage+'.md']));
  for(const path of Object.values(map))await writeFile(join(skills,path),'Read current skill; query both kinds in design/optimisation; use rsi_advance defaults and frozen verifier.');
  await writeFile(join(skills,'skill-map.json'),JSON.stringify(map));
  const projects=[];
  for(const id of ids) {
    const source=join(root,id,'source');await mkdir(source,{recursive:true});
    await writeFile(join(source,'answer.py'),'def answer(): return 0\n');
    projects.push({id,goal:id,source,actor_seconds:120,
      verifier:{files:[fixture],command:`${quote(python)} ${quote(fixture)} verify ${quote(source)}`}});
  }
  const config={baseline,skills,projects,environment:{module:'machinery/integrations/environments/index.mjs',options:{mode:'local'}}};
  const input=join(root,'bootstrap.json');await writeFile(input,JSON.stringify(config));
  const boot=execute(process.execPath,[cli,'bootstrap',instance,'--config',input]);await boot.output;
  cli=join(instance,'machinery','cli','main.mjs');
  // Install only the configurable test embedding module, never replace runtime code.
  await cp(fixture,join(instance,'machinery','driver_fixture.py'));
  const settings=join(root,'settings.json');await writeFile(settings,JSON.stringify({python,
    embedding_factory:'driver_fixture:Embeddings',target:'cpu',investigation_seconds:120,
    failed_recovery_rounds:2,concurrency:1,curriculum_seconds:30,
    actor:{provider:'driver-fixture',model:'scripted',credential_env:'DRIVER_FIXTURE_KEY',base_url:server.url,
      metadata:{id:'scripted',name:'Scripted SSE acceptance',api:'openai-completions',reasoning:false,
        input:['text'],contextWindow:128000,maxTokens:8192,
        cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}}}));
  await run('config','--config',settings);
  return {root:instance,run,start,server,pass:()=>{okay=true;}};
}

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
  await until(()=>rows(f.root,"SELECT frozen FROM projects WHERE id='freeze'")[0]?.frozen===1);
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
