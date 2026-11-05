import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,rm,access} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {State} from '../../controller/state.mjs';
import {CampaignStorage} from '../../controller/storage.mjs';
import {processIdentity} from '../../controller/lifecycle.mjs';
import {run,sealClosedArtifacts} from '../../cli/storage.mjs';
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
  assert.equal(f.storage.list()[0].encoding,'gzip-chunks');
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
  await writeFile(join(manifest.directory,manifest.chunks[0].file),'corrupt');
  await assert.rejects(f.call({action:'prune',apply:true,inventory}));await access(join(f.root,'session.jsonl'));
  f.state.db.prepare("UPDATE storage_artifacts SET path='../outside' WHERE id=?").run(f.row.id);
  await assert.rejects(f.call({action:'inspect'}),/escapes/);
});

test('package delegation preserves scoped arguments; standalone unpack opens no ledger',async t=>{
  const f=await fixture(t);
  const code=`import json,sys,types
from ascend_tools.storage_cli import execute
module=types.ModuleType('ascend_tools.storage_package')
module.pack=lambda root,dest,project=None:dict(root=str(root),dest=str(dest),project=project)
module.unpack=lambda package,dest:dict(package=str(package),dest=str(dest))
sys.modules[module.__name__]=module
print(json.dumps([execute(sys.argv[1],dict(action='pack',project='p',destination='package')),
 execute('/no-existing-instance',dict(action='unpack',package=sys.argv[1]+'/package',destination=sys.argv[1]+'/restored'))]))`;
  const {stdout}=await exec(python,['-c',code,f.root]);
  const [packed,unpacked]=JSON.parse(stdout);
  assert.deepEqual(packed,{root:f.root,dest:join(f.root,'package'),project:'p'});
  assert.deepEqual(unpacked,{package:join(f.root,'package'),dest:join(f.root,'restored')});
});
