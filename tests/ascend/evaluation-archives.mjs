import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,copyFile,rm,stat,access,symlink} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {State} from '../../controller/state.mjs';
import {CampaignStorage} from '../../controller/storage.mjs';
import {processIdentity} from '../../controller/lifecycle.mjs';
import {run as storageRun} from '../../cli/storage.mjs';
import {backgroundSkillStatus,runBackgroundSkills} from '../../cli/skill-background.mjs';
import {enqueueCompletedEvaluation,evaluationArchiveStatus,drainEvaluationArchives,evaluationArchivesSettled} from '../../cli/evaluation-archives.mjs';
const exec=promisify(execFile),python=process.env.RSI_TEST_PYTHON??'python3';
const save=(path,value)=>writeFile(path,JSON.stringify(value));

async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'rsi-evaluation-archive-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  const storage=new CampaignStorage(state,root);await storage.initialize({id:'parent-campaign'});
  const stopped={...await processIdentity(),start:'previous-process-start'};
  state.add({id:'origin',goal:'fixture',actor_seconds:10});const attempt=state.admit('origin',stopped);
  state.finish(attempt.id,'passed');state.event(attempt.id,'postprocess-completed',{});storage.attempt(attempt.id,{phase:'phase2'});
  const evaluation=join(root,'skills/evaluations/round');await mkdir(evaluation,{recursive:true});
  const baseline=join(root,'baseline.sqlite');
  await exec(python,['-c',`import sqlite3,sys,os
with sqlite3.connect(sys.argv[1]) as c:
 c.execute('CREATE TABLE facts(value TEXT,payload BLOB)')
 c.execute('INSERT INTO facts VALUES (?,?)',('retained',os.urandom(1024*1024)))`,baseline]);
  const directories=[...Array.from({length:8},(_,i)=>join(evaluation,'sample-'+i)),join(evaluation,'inputs')];
  for(const [index,path] of directories.entries()) {
    await mkdir(path);await copyFile(baseline,join(path,'knowledge.sqlite'));
    await save(join(path,'instance.json'),{format:1,mode:'phase2',root:path,skills:{tuningDisabled:true}});
    await save(join(path,'knowledge-service.json'),{owner:stopped});
    await writeFile(join(path,'session.jsonl'),'retained native session');
    const child=new State(join(path,'state.sqlite'));
    if(index<8){child.add({id:'p',goal:'fixture',actor_seconds:10});const a=child.admit('p',stopped);child.finish(a.id,'passed');child.event(a.id,'postprocess-completed',{});}
    child.close();
  }
  const round={status:'completed',samples:directories.slice(0,8).map(instanceRoot=>({instanceRoot,result:{correct:true,score:1}}))};
  await save(join(evaluation,'round.json'),round);await save(join(evaluation,'results.json'),round.samples);
  const evolution={history:[{id:attempt.id,projectId:'origin',kind:'original',phase:'phase2',completed:true}],lastEvaluationRoot:evaluation};
  await save(join(root,'skills/evolution.json'),evolution);
  const config={root,python,storage:{format:1,campaign_id:'parent-campaign'},skills:{archiveEvaluations:true,background:true}};
  return {root,state,storage,attempt,stopped,evaluation,directories,round,evolution,config,
    files:directories.map(path=>join(path,'knowledge.sqlite')),
    enqueue:()=>enqueueCompletedEvaluation(config,evolution),drain:values=>drainEvaluationArchives(config,values)};
}

async function descriptors(f) {return f.state.db.prepare("SELECT * FROM storage_artifacts WHERE kind='evaluation-kdb' ORDER BY path").all();}

// Real SQLite, native parent ledger and shared Restic repository, not a mock archive.
test('one completed comparison deduplicates nine KDBs and restores through parent artifact IDs',async t=>{
  const f=await fixture(t);
  assert.equal(await f.enqueue(),true);assert.equal(await f.enqueue(),false);
  assert.equal((await backgroundSkillStatus(f.config)).hasWork,true);
  await f.drain();assert.equal((await evaluationArchiveStatus(f.config,f.evolution)).complete,1);
  assert.equal((await backgroundSkillStatus(f.config)).hasWork,false);
  const rows=await descriptors(f);assert.equal(rows.length,9);assert.ok(rows.every(r=>r.attempt===f.attempt.id&&r.encoding==='restic'));
  const manifests=await Promise.all(rows.map(async row=>JSON.parse(await readFile(join(f.root,JSON.parse(row.descriptor).manifest),'utf8'))));
  assert.equal(new Set(manifests.map(m=>m.repository)).size,1);assert.equal(new Set(manifests.map(m=>m.snapshot)).size,1);
  for(const file of f.files)await assert.rejects(access(file));
  for(const path of f.directories)assert.equal(await readFile(join(path,'session.jsonl'),'utf8'),'retained native session');
  assert.equal(JSON.parse(await readFile(join(f.evaluation,'results.json'),'utf8')).length,8);
  const stats=JSON.parse((await exec(python,['-c',`from ascend_tools.storage_archive import _run
import sys
print(_run(sys.argv[1],'stats','--mode','raw-data','--json').decode())`,manifests[0].repository])).stdout);
  assert.ok(stats.total_size<rows.reduce((sum,row)=>sum+row.bytes,0)/2);
  await storageRun(f.config,{action:'restore',artifact:rows[0].id,destination:rows[0].path});
  const query=await exec(python,['-c',`import sqlite3,sys
print(sqlite3.connect(sys.argv[1]).execute('SELECT value,length(payload) FROM facts').fetchone())`,join(f.root,rows[0].path)]);
  assert.match(query.stdout,/retained/);assert.match(query.stdout,/1048576/);
  await f.drain();assert.equal((await descriptors(f)).length,9);await access(join(f.root,rows[0].path));
});

test('descriptor publication before deletion is retried without another snapshot',async t=>{
  const f=await fixture(t);await f.enqueue();
  for(const file of f.files){const row=await f.storage.register(file,{attempt:f.attempt.id,kind:'evaluation-kdb',role:'evaluation'});await f.storage.seal(row.id);}
  await storageRun(f.config,{action:'compress',kind:'evaluation-kdb',role:'evaluation',project:'origin',delete_source:false});
  const before=(await descriptors(f)).map(row=>row.descriptor);
  for(const file of f.files)await access(file);
  await f.drain();assert.deepEqual((await descriptors(f)).map(row=>row.descriptor),before);
  for(const file of f.files)await assert.rejects(access(file));
});

for(const problem of ['origin-owner','child-owner','child-kdb','partial','request','changed-source','missing-kdb','aliased-kdb'])test(`archive preserves ${problem} evidence`,async t=>{
  const f=await fixture(t);await f.enqueue();const live=await processIdentity();
  if(problem==='origin-owner')f.state.db.prepare('UPDATE attempts SET owner=? WHERE id=?').run(JSON.stringify(live),f.attempt.id);
  if(problem==='child-owner'){
    const child=new State(join(f.directories[0],'state.sqlite'));
    child.db.exec('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');child.db.prepare('INSERT INTO control VALUES (?,?)').run('owner',JSON.stringify(live));child.close();
  }
  if(problem==='child-kdb')await save(join(f.directories[0],'knowledge-service.json'),{owner:live});
  if(problem==='partial'){f.round.samples[0].result=null;await save(join(f.evaluation,'round.json'),f.round);}
  if(problem==='request'){
    const child=new State(join(f.directories[0],'state.sqlite'));
    child.db.exec("CREATE TABLE requests(status TEXT,owner TEXT); INSERT INTO requests VALUES ('running',NULL)");child.close();
  }
  if(problem==='changed-source'){
    const row=await f.storage.register(f.files[0],{attempt:f.attempt.id,kind:'evaluation-kdb',role:'evaluation'});await f.storage.seal(row.id);
    await writeFile(f.files[0],'changed SQLite source');
  }
  if(problem==='missing-kdb')await rm(f.files[0]);
  if(problem==='aliased-kdb'){await rm(f.files[0]);await symlink(join(f.root,'baseline.sqlite'),f.files[0]);}
  await assert.rejects(f.drain());
  if(problem==='missing-kdb')await assert.rejects(access(f.files[0]));else await access(f.files[0]);
  const status=await backgroundSkillStatus(f.config);assert.equal(status.hasWork,true);assert.equal(status.archives.pending,1);assert.ok(status.archives.error);
  let generated=false;
  await assert.rejects(runBackgroundSkills(f.config,{resource:{id:'reserved'},remainingSeconds:10,generate:async()=>{generated=true;return [];}}));
  assert.equal(generated,false);
});

test('completed round discovery survives crash before enqueue and status never mutates the ledger',async t=>{
  const f=await fixture(t);
  assert.equal((await backgroundSkillStatus(f.config)).archives.untracked,true);
  assert.equal(f.state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='evaluation_archives'").get(),undefined);
  await f.enqueue();const disabled={...f.config,skills:{...f.config.skills,archiveEvaluations:false}};
  assert.equal(await enqueueCompletedEvaluation(disabled,f.evolution),false);
  await assert.rejects(enqueueCompletedEvaluation(f.config,{...f.evolution,lastEvaluationRoot:f.root}),/escaped/);
});

test('cancellation closes archive workers before releasing the background slot and permits retry',async t=>{
  const f=await fixture(t);await f.enqueue();const controller=new AbortController();
  const running=f.drain({signal:controller.signal});
  const end=performance.now()+10000;
  while(performance.now()<end){
    const row=f.state.db.prepare('SELECT phase FROM evaluation_archives').get();
    if(row?.phase==='archiving'){controller.abort();break;}
    await new Promise(resolve=>setTimeout(resolve,5));
  }
  await assert.rejects(running);assert.equal(await evaluationArchivesSettled(f.config),true);
  assert.equal((await evaluationArchiveStatus(f.config,f.evolution)).pending,1);
  await f.drain();assert.equal((await evaluationArchiveStatus(f.config,f.evolution)).complete,1);
});

test('closed hot-WAL contents are checkpointed before the native artifact is sealed',async t=>{
  const f=await fixture(t);await f.enqueue();
  await exec(python,['-c',`import sqlite3,sys,pathlib
p=pathlib.Path(sys.argv[1]);c=sqlite3.connect(p)
c.execute('PRAGMA journal_mode=WAL');c.execute('PRAGMA wal_autocheckpoint=0')
c.execute("UPDATE facts SET value='from-wal'");c.commit()
base=p.read_bytes();wal=pathlib.Path(str(p)+'-wal').read_bytes();c.close()
p.write_bytes(base);pathlib.Path(str(p)+'-wal').write_bytes(wal)`,f.files[0]]);
  await f.drain();
  const row=(await descriptors(f)).find(row=>join(f.root,row.path)===f.files[0]);
  await storageRun(f.config,{action:'restore',artifact:row.id,destination:row.path});
  const query=await exec(python,['-c',`import sqlite3,sys
print(sqlite3.connect(sys.argv[1]).execute('SELECT value FROM facts').fetchone()[0])`,f.files[0]]);
  assert.equal(query.stdout.trim(),'from-wal');
});

test('unknown archive dispatch remains fenced instead of being reset',async t=>{
  const f=await fixture(t);await f.enqueue();
  f.state.db.prepare("UPDATE evaluation_archives SET phase='dispatching'").run();
  assert.equal(await evaluationArchivesSettled(f.config),false);
  await assert.rejects(f.drain(),/unknown/);
  for(const file of f.files)await access(file);
  assert.equal((await evaluationArchiveStatus(f.config,f.evolution)).pending,1);
});
