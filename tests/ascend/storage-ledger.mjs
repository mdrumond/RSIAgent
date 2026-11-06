import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,mkdir,rm,symlink} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {CampaignStorage} from '../../controller/storage.mjs';

async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'rsi-storage-')),state=new State(join(root,'state.sqlite'));
 t.after(async()=>{state.close();await rm(root,{recursive:true,force:true});});
 state.add({id:'p',goal:'test',actor_seconds:30});
 const ledger=new CampaignStorage(state,root);await ledger.initialize();return {root,state,ledger};
}
test('campaign identity and pinned inputs survive deployment and phase changes',async t=>{
 const {root,state,ledger}=await fixture(t),campaign=await ledger.initialize();
 ledger.activate('memory','memory1','indexed1');ledger.activate('skills','skills1');
 const first=state.admit('p',{});ledger.attempt(first.id,{phase:'phase1',deployment:ledger.deployment('one')});
 state.finish(first.id,'failed');state.db.prepare("UPDATE projects SET status='queued' WHERE id='p'").run();
 ledger.activate('memory','memory2','indexed2');
 const retry=state.admit('p',{});ledger.attempt(retry.id,{phase:'phase2',kind:'retry',parent:first.id,deployment:ledger.deployment('two')});
 const rows=state.db.prepare('SELECT * FROM storage_attempts ORDER BY rowid').all();
 assert.equal(rows[0].memory_revision,'memory1');assert.equal(rows[1].knowledge_generation,'indexed2');assert.equal(rows[1].parent,first.id);
 assert.equal((await CampaignStorage.open(state,root).initialize()).id,campaign.id);
});
test('closed artifacts have stream hashes and bounded inspection; missing remote stays missing',async t=>{
 const {root,state,ledger}=await fixture(t),a=state.admit('p',{});
 await writeFile(join(root,'artifacts','output'),'evidence');
 const artifact=await ledger.register('artifacts/output',{attempt:a.id,project:'p',kind:'profiler'});
 const closed=await ledger.seal(artifact.id);assert.equal(closed.bytes,8);assert.equal(closed.sha256.length,64);
 const remote=await ledger.register('/remote/trace',{attempt:a.id,project:'p',kind:'profiler',external:true});
 assert.equal(remote.status,'not-collected');await assert.rejects(ledger.seal(remote.id),/Collect/);
 assert.equal(ledger.list({attempt:a.id,limit:1}).length,1);
 await assert.rejects(ledger.register('artifacts/output',{project:'p'}),/ownership/);
});
test('source artifacts cannot follow symlinks outside campaign',async t=>{
 const {root,ledger}=await fixture(t);await symlink(tmpdir(),join(root,'escape'));
 await assert.rejects(ledger.register('escape/outside'),/campaign/);
});
test('iteration identity preserves negative revisions without starting another attempt',async t=>{
 const {state,ledger}=await fixture(t),a=state.admit('p',{});
 ledger.iteration(a.id,{stage:'optimisation',reflection:'tiling regressed',sourceHash:'one'});
 ledger.iteration(a.id,{stage:'optimisation',reflection:'retry corrected hypothesis',sourceHash:'two'});
 assert.deepEqual(state.db.prepare('SELECT ordinal,source_hash FROM storage_iterations').all().map(r=>[r.ordinal,r.source_hash]),[[1,'one'],[2,'two']]);
 assert.equal(state.db.prepare('SELECT COUNT(*) n FROM attempts').get().n,1);
});
test('old state has no storage side effects until explicitly initialized',async t=>{
 const {root,state}=await fixture(t);state.db.exec('DROP TABLE storage_campaign');
 assert.equal(CampaignStorage.open(state,root),null);
});

test('collection preserves artifact identity and remote provenance before sealing',async t=>{
 const {root,ledger}=await fixture(t);
 const external=await ledger.register('/target/profile.raw',{kind:'profiler',external:true});
 await writeFile(join(root,'artifacts','profile.raw'),'profile bytes');
 const collected=await ledger.collected(external.id,join(root,'artifacts','profile.raw'));
 assert.equal(collected.id,external.id);assert.equal(collected.external,0);assert.equal(collected.status,'closed');
 assert.equal(collected.bytes,13);assert.equal(JSON.parse(collected.descriptor).collected_from,'/target/profile.raw');
});

test('non-actor role closure is explicit and keeps its provider stream linked',async t=>{
 const {root,state,ledger}=await fixture(t);await mkdir(join(root,'writer'));
 const session=join(root,'writer/session.jsonl'),provider=join(root,'writer/provider.jsonl');
 await ledger.session(session,{phase:'phase2',role:'skill-writer',provider});
 assert.equal(state.db.prepare('SELECT closed FROM storage_sessions').get().closed,0);
 ledger.closeSession(session);assert.equal(state.db.prepare('SELECT closed FROM storage_sessions').get().closed,1);
 assert.equal(ledger.list({kind:'provider-telemetry'})[0].role,'skill-writer');
});
