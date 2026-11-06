import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,access} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {State} from '../../controller/state.mjs';
import {CampaignStorage} from '../../controller/storage.mjs';
import {processIdentity} from '../../controller/lifecycle.mjs';
import {run,sealClosedArtifacts,withPackCapture,assertPackAvailable} from '../../cli/storage.mjs';
const exec=promisify(execFile),python=process.env.RSI_TEST_PYTHON??'python3.13';

async function fixture(t,{live=false,finish=true,postprocess=true,settled=true}={}) {
  const root=await mkdtemp(join(tmpdir(),'rsi-storage-interface-')),state=new State(join(root,'state.sqlite'));
  t.after(async()=>{state.close();await rm(root,{recursive:true,force:true});});
  const storage=new CampaignStorage(state,root);await storage.initialize({id:'campaign-fixture'});
  state.add({id:'p',goal:'fixture',actor_seconds:10});
  const identity=await processIdentity();
  const attempt=state.admit('p',live?identity:{...identity,start:'different-start'});
  storage.activate('skills','skills1');storage.activate('memory','memory1');
  storage.attempt(attempt.id,{phase:'phase1'});
  storage.iteration(attempt.id,{stage:'design'});
  if(finish)state.finish(attempt.id,'failed');
  if(postprocess)state.event(attempt.id,'postprocess-completed',{});
  state.event(attempt.id,'failure-settlement',{closed:settled});
  await writeFile(join(root,'session.jsonl'),'{"type":"session"}\n{"type":"message","message":{"content":"native"}}\n');
  const row=await storage.register('session.jsonl',{attempt:attempt.id,project:'p',kind:'pi-session'});
  const config={root,python};
  const call=async action=>{const file=join(root,'action.json');await writeFile(file,JSON.stringify(action));return run(config,{config:file});};
  return {root,state,storage,row,attempt,call,config};
}

test('pack rejects live writers before sealing for campaign and project exports',async t=>{
  const f=await fixture(t,{live:true});
  for(const project of [undefined,'p'])
    await assert.rejects(f.call({action:'pack',project,destination:f.root+'-package'}),/stopped attempt writers/);
  assert.equal(f.storage.list()[0].status,'open');
  assert.equal(f.state.db.prepare("SELECT value FROM control WHERE key='pack-capture'").get(),undefined);
});

test('pack claim rejects writer admission and competing capture, then releases on failure',async t=>{
  const f=await fixture(t);
  await assert.rejects(withPackCapture(f.config,async()=>{
    // Simulate the shared ownership-claim transaction used by writer startup.
    f.state.db.exec('BEGIN IMMEDIATE');
    try{await assert.rejects(assertPackAvailable(f.state),/writer admission refused/);}
    finally{f.state.db.exec('ROLLBACK');}
    await assert.rejects(withPackCapture(f.config,async()=>assert.fail('competing capture admitted')),/already active/);
    throw new Error('capture failed');
  }),/capture failed/);
  await assertPackAvailable(f.state);
  assert.equal(f.state.db.prepare("SELECT value FROM control WHERE key='pack-capture'").get(),undefined);
  f.state.db.prepare("INSERT INTO control VALUES ('pack-capture',?)").run(JSON.stringify({...await processIdentity(),start:'old'}));
  await withPackCapture(f.config,async()=>{});
  assert.equal(f.state.db.prepare("SELECT value FROM control WHERE key='pack-capture'").get(),undefined);
});

test('CLI capture delegates to Python while retaining parent ownership',async t=>{
  const f=await fixture(t),me=await processIdentity();
  const {withSkillLock}=await import('../../cli/evolution-lock.mjs');
  await withSkillLock(f.config,()=>withPackCapture(f.config,async()=>{
    const code=`from pathlib import Path
from ascend_tools.storage_package import _capture
import sys,sqlite3
with _capture(Path(sys.argv[1])):
 with sqlite3.connect(str(Path(sys.argv[1])/'state.sqlite')) as db:
  print(db.execute("SELECT value FROM control WHERE key='pack-capture'").fetchone()[0])`;
    const {stdout}=await exec(python,['-c',code,f.root]);
    const copyingOwner=JSON.parse(stdout);
    assert.notEqual(copyingOwner.pid,me.pid);assert.equal(copyingOwner.boot,me.boot);
    assert.deepEqual(JSON.parse(f.state.db.prepare("SELECT value FROM control WHERE key='pack-capture'").get().value),me);
  }),{key:'storage-maintenance'});
  assert.equal(f.state.db.prepare("SELECT value FROM control WHERE key='pack-capture'").get(),undefined);
});

test('bounded metadata filters preserve identities and report missing evidence',async t=>{
  const f=await fixture(t);
  await f.storage.register('missing.jsonl',{kind:'profiler'});
  const result=await f.call({action:'inspect',project:'p',phase:'phase1',stage:'design',iteration:1,
    role:'actor',skill_revision:'skills1',memory_revision:'memory1',outcome:'failed',limit:1});
  assert.equal(result.campaign.id,'campaign-fixture');assert.equal(result.artifacts[0].id,f.row.id);
  assert.equal(result.artifacts[0].attempt,f.attempt.id);
  assert.equal((await f.call({action:'inspect',kind:'profiler'})).artifacts[0].missing,true);
  assert.equal((await f.call({action:'inspect',phase:'phase2'})).artifacts.length,0);
  await assert.rejects(f.call({action:'inspect',limit:1001}));
});

for(const [name,options] of [['live',{live:true}],['unsettled',{settled:false}],
  ['postprocessing',{postprocess:false}],['active',{finish:false}]]) {
  test(`compression preserves ${name} actor evidence`,async t=>{
    const f=await fixture(t,options);
    assert.deepEqual(await sealClosedArtifacts(f.config),[]);
    const result=await f.call({action:'compress'});
    assert.equal(result[0].skipped,'not closed local evidence');
    assert.equal(f.storage.list()[0].status,'open');await access(join(f.root,'session.jsonl'));
  });
}

test('closed compression publishes reference, verifies and explicitly cleans redundant source',async t=>{
  const f=await fixture(t);
  const result=await f.call({action:'compress',artifact:f.row.id});
  assert.equal(result[0].source_deleted,false);await access(join(f.root,'session.jsonl'));
  assert.equal(f.storage.list()[0].encoding,'restic');
  assert.equal((await f.call({action:'verify'}))[0].verified,true);
  const inventory=await f.call({action:'prune',project:'p'});
  assert.equal(inventory.entries.length,1);
  const applied=await f.call({action:'prune',apply:true,inventory});
  assert.deepEqual(applied.removed,[f.row.id]);
  await assert.rejects(access(join(f.root,'session.jsonl')));
  assert.equal((await f.call({action:'verify'}))[0].verified,true);
});

test('cleanup refuses changed inventory, source, and newly live ownership',async t=>{
  const f=await fixture(t);await f.call({action:'compress'});
  const inventory=await f.call({action:'prune',artifact:f.row.id});
  f.state.db.prepare("UPDATE storage_artifacts SET role='writer' WHERE id=?").run(f.row.id);
  await assert.rejects(f.call({action:'prune',apply:true,inventory}),/changed/);
  f.state.db.prepare("UPDATE storage_artifacts SET role='actor' WHERE id=?").run(f.row.id);
  f.state.db.prepare('UPDATE attempts SET owner=? WHERE id=?').run(JSON.stringify(await processIdentity()),f.attempt.id);
  await assert.rejects(f.call({action:'prune',apply:true,inventory}),/live/);
  await access(join(f.root,'session.jsonl'));
  f.state.db.prepare('UPDATE attempts SET owner=? WHERE id=?').run(JSON.stringify({...await processIdentity(),start:'old'}),f.attempt.id);
  await writeFile(join(f.root,'session.jsonl'),'changed');
  await assert.rejects(f.call({action:'prune',apply:true,inventory}),/changed/);
  assert.equal(await readFile(join(f.root,'session.jsonl'),'utf8'),'changed');
});

test('walker defaults to native Pi, selects raw provider explicitly and honors cursor',async t=>{
  const f=await fixture(t);
  await writeFile(join(f.root,'provider.jsonl'),'{"type":"provider"}\n');
  await f.storage.register('provider.jsonl',{attempt:f.attempt.id,project:'p',kind:'provider-telemetry'});
  const walk=async action=>{
    const {stdout}=await exec(python,['-m','ascend_tools.storage_cli','--root',f.root,'--action-json',JSON.stringify({action:'walk',...action})]);
    return stdout.trim().split('\n').filter(Boolean).map(JSON.parse);
  };
  let rows=await walk({phase:'phase1'});assert.equal(rows.length,2);assert.equal(rows[0].record.type,'session');
  assert.equal((await walk({cursor:rows[0].metadata.next_cursor})).length,1);
  rows=await walk({raw_provider:true});assert.equal(rows.length,1);assert.equal(rows[0].record.type,'provider');
  await f.call({action:'compress',kind:'pi-session',delete_source:true});
  assert.equal((await walk({})).length,2);
});

test('out-of-root artifact and corrupt archive are never read or deleted',async t=>{
  const f=await fixture(t);await f.call({action:'compress'});
  const inventory=await f.call({action:'prune',artifact:f.row.id});
  const descriptor=JSON.parse(f.storage.list()[0].descriptor),manifest=JSON.parse(await readFile(join(f.root,descriptor.manifest),'utf8'));
  manifest.sha256='0'.repeat(64);
  await writeFile(join(f.root,descriptor.manifest),JSON.stringify(manifest));
  await assert.rejects(f.call({action:'prune',apply:true,inventory}));await access(join(f.root,'session.jsonl'));
  f.state.db.prepare("UPDATE storage_artifacts SET path='../outside' WHERE id=?").run(f.row.id);
  await assert.rejects(f.call({action:'inspect'}),/escapes/);
});

test('package delegation preserves scoped arguments; standalone unpack opens no ledger',async t=>{
  const f=await fixture(t);
  const code=`import json,sys,types
from ascend_tools.storage_cli import execute
module=types.ModuleType('ascend_tools.storage_package')
def pack(root,dest,project=None):
 assert not dest.is_relative_to(root), 'pack forbids outputs inside campaign'
 assert project == 'p'
 return dict(complete=True,scope=project,files=[{}],missing=[])
module.pack=pack
module.unpack=lambda package,dest:dict(package=str(package),dest=str(dest))
sys.modules[module.__name__]=module
print(json.dumps([execute(sys.argv[1],dict(action='pack',project='p',destination=sys.argv[1]+'-package')),
 execute('/no-existing-instance',dict(action='unpack',package=sys.argv[1]+'/package',destination=sys.argv[1]+'/restored'))]))`;
  const {stdout}=await exec(python,['-c',code,f.root]);
  const [packed,unpacked]=JSON.parse(stdout);
  assert.deepEqual(packed,{package:f.root+'-package',complete:true,scope:'p',files:1,missing:[]});
  assert.deepEqual(unpacked,{package:join(f.root,'package'),dest:join(f.root,'restored')});
});


test('compression batches skip missing and directory rows and still archive eligible files',async t=>{
  const f=await fixture(t);
  await f.storage.register('not-created.jsonl',{attempt:f.attempt.id,kind:'pi-session'});
  await mkdir(join(f.root,'directory'));
  await f.storage.register('directory',{attempt:f.attempt.id,kind:'artifact'});
  await writeFile(join(f.root,'removed.jsonl'),'closed then missing');
  const missing=await f.storage.register('removed.jsonl',{attempt:f.attempt.id,kind:'artifact'});
  await f.storage.seal(missing.id);await rm(join(f.root,'removed.jsonl'));
  await writeFile(join(f.root,'second.jsonl'),'second durable evidence');
  const second=await f.storage.register('second.jsonl',{attempt:f.attempt.id,kind:'artifact'});
  const result=await f.call({action:'compress'});
  const archived=await Promise.all([f.row.id,second.id].map(async id=>{const row=f.storage.list().find(r=>r.id===id);return JSON.parse(await readFile(join(f.root,JSON.parse(row.descriptor).manifest),'utf8'));}));
  assert.equal(archived[0].snapshot,archived[1].snapshot);
  assert.equal(result.find(row=>row.id===f.row.id).source_deleted,false);
  assert.equal(result.filter(row=>row.skipped).length,3);
  assert.equal((await f.call({action:'verify',artifact:f.row.id}))[0].verified,true);
});

test('immutable shared cited text seals during a live campaign without sealing actor sessions',async t=>{
  const f=await fixture(t,{live:true,finish:false,postprocess:false});
  await writeFile(join(f.root,'citation.txt'),'immutable retrieved text');
  const cited=await f.storage.register('citation.txt',{kind:'cited-text'});
  const ordinary=await f.storage.register('other.jsonl',{kind:'artifact'});
  assert.deepEqual(await sealClosedArtifacts(f.config,{action:'pack',project:'p'}),[cited.id]);
  const sealed=f.storage.list().find(row=>row.id===cited.id);
  assert.equal(sealed.status,'closed');assert.equal(sealed.bytes,24);assert.equal(sealed.sha256.length,64);
  const result=await f.call({action:'compress'});
  assert.equal(result.find(row=>row.id===cited.id).source_deleted,false);
  assert.equal(f.storage.list().find(row=>row.id===f.row.id).status,'open');
  assert.equal(f.storage.list().find(row=>row.id===ordinary.id).status,'open');
});

test('sealing advances beyond an ineligible page instead of starving later evidence',async t=>{
 const f=await fixture(t);const second=await f.storage.register('later.jsonl',{attempt:f.attempt.id,kind:'provider-telemetry'});
 await writeFile(join(f.root,'later.jsonl'),'{}\n');
 const rows=f.storage.list().sort((a,b)=>a.id.localeCompare(b.id));
 const first=rows[0],last=rows[1];
 f.state.db.prepare('UPDATE storage_artifacts SET external=1 WHERE id=?').run(first.id);
 assert.deepEqual(await sealClosedArtifacts(f.config,{limit:1}),[]);
 assert.deepEqual(await sealClosedArtifacts(f.config,{limit:1}),[last.id]);
 assert.equal(f.storage.list().find(r=>r.id===last.id).status,'closed');
});


test('standalone role sessions retain phase filters and seal only after writer closure',async t=>{
  const f=await fixture(t);
  const session=join(f.root,'curriculum.jsonl'),provider=join(f.root,'curriculum-provider.jsonl');
  await writeFile(session,'{"type":"session","id":"curriculum"}\n');
  await writeFile(provider,'{"type":"provider"}\n');
  const row=await f.storage.session(session,{phase:'phase2',role:'curriculum',provider});
  let inventory=await f.call({action:'inspect',phase:'phase2',role:'curriculum'});
  assert.equal(inventory.artifacts.length,2);
  assert.ok(inventory.artifacts.every(r=>r.phase==='phase2'));
  assert.deepEqual(await sealClosedArtifacts(f.config,{phase:'phase2'}),[]);
  f.storage.closeSession(session);
  const sealed=await sealClosedArtifacts(f.config,{phase:'phase2'});
  assert.ok(sealed.includes(row.id));assert.equal(sealed.length,2);
  assert.equal((await f.call({action:'inspect',phase:'phase1',role:'curriculum'})).artifacts.length,0);
});

test('automatic maintenance submits no empty jobs and reuses a pending request',async t=>{
  const f=await fixture(t,{finish:false});
  const {queueCompression}=await import('../../cli/storage-maintenance.mjs');
  const config={...f.config,storage:{enabled:true}};
  await writeFile(join(f.root,'vanished.log'),'retained then removed');
  const vanished=await f.storage.register('vanished.log',{kind:'cited-text'});
  await f.storage.seal(vanished.id);await rm(join(f.root,'vanished.log'));
  assert.equal(await queueCompression(config),null);
  assert.equal(f.state.db.prepare('SELECT COUNT(*) n FROM requests').get().n,0);
  f.state.db.prepare("INSERT INTO requests(id,kind,status) VALUES ('pending','storage','queued')").run();
  assert.deepEqual(await queueCompression(config),{request:'pending',retained:true});
  f.state.db.prepare("UPDATE requests SET status='passed' WHERE id='pending'").run();
  assert.equal(await queueCompression(config),null);
  assert.equal(f.state.db.prepare('SELECT COUNT(*) n FROM requests').get().n,1);
});

test('project verification includes source Git integrity without treating its directory as a log',async t=>{
  const f=await fixture(t);await f.storage.seal(f.row.id);
  const repository=join(f.root,'source-history','repository');
  await mkdir(repository,{recursive:true});await exec('git',['init','--bare',repository]);
  const history=await f.storage.register(repository,{project:'p',kind:'source-git'});
  const verified=await f.call({action:'verify',project:'p'});
  assert.equal(verified.find(row=>row.id===history.id).verified,true);
  assert.equal(verified.find(row=>row.id===f.row.id).verified,true);
  await rm(join(repository,'HEAD'));
  await assert.rejects(f.call({action:'verify',project:'p'}));
});
