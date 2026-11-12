import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, rename, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { KnowledgeHistory } from '../../controller/knowledge-history.mjs';

const lesson = (delivery_id, text = 'Use bounded checks') => ({ id: 'health', delivery_id,
  lesson: text, evidence_refs: ['evidence://check/1'], kind: 'good', target: 'a3' });
const callbacks = { indexRevision: async revision => `generation-${revision}`, activate: async () => {} };
async function fixture(t) {
  const directory = await mkdtemp(join(tmpdir(), 'knowledge-test-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return new KnowledgeHistory(directory).init();
}

test('commits before indexing exact snapshot; corrections retain parent and previous contents', async t => {
  const history = await fixture(t);
  const first = await history.commitMemory([lesson('first')], callbacks);
  const calls = [];
  const corrected = await history.commitMemory([lesson('second', 'Corrected check')], {
    indexRevision: async (revision, snapshot) => {
      calls.push('index');
      assert.equal(await history.ref('refs/heads/memory'), revision);
      assert.equal((await history.state('memory')).revision, first.revision);
      assert.equal(snapshot.memory[0].lesson, 'Corrected check');
      assert.deepEqual(snapshot.parents, [first.revision]);
      return 'second-generation';
    }, activate: async state => { calls.push('activate'); assert.equal(state.generation, 'second-generation'); },
  });
  assert.deepEqual(calls, ['index', 'activate']);
  assert.equal((await history.readRevision(first.revision)).memory[0].lesson, 'Use bounded checks');
  assert.deepEqual(await history.state('memory'), corrected);
  assert.equal(await history.git(['remote']), '');
  assert.deepEqual((await history.readRevision(corrected.revision)).event.deliveries, ['second']);
});

test('indexing/activation failure preserves active revision; reopening retries without duplicate commit', async t => {
  const history = await fixture(t);
  const initial = await history.commitMemory([lesson('initial')], callbacks);
  await assert.rejects(history.commitMemory([lesson('next', 'Correction')], {
    ...callbacks, indexRevision: async () => { throw new Error('index unavailable'); },
  }), /index unavailable/);
  const staged = await history.ref('refs/heads/memory');
  assert.deepEqual(await history.state('memory'), initial);
  const reopened = await new KnowledgeHistory(history.directory).init();
  await assert.rejects(reopened.commitMemory([lesson('next', 'Correction')], {
    ...callbacks, activate: async () => { throw new Error('activation unavailable'); },
  }), /activation unavailable/);
  assert.equal(await reopened.ref('refs/heads/memory'), staged);
  assert.deepEqual(await reopened.state('memory'), initial);
  const committed = await reopened.commitMemory([lesson('next', 'Correction')], callbacks);
  assert.equal(committed.revision, staged);
  const before = await reopened.git(['rev-list', '--count', 'refs/heads/memory']);
  assert.deepEqual(await reopened.commitMemory([lesson('next', 'Correction')], {
    indexRevision: () => assert.fail('repeat indexed'), activate: () => assert.fail('repeat activated'),
  }), committed);
  assert.equal(await reopened.git(['rev-list', '--count', 'refs/heads/memory']), before);
  await assert.rejects(reopened.commitMemory([lesson('next', 'Changed payload')], callbacks), /reused/);
  assert.deepEqual(await reopened.commitMemory([lesson('initial')], callbacks), committed);
});

test('skill promotion never overwrites newer memory; decisions and candidate parents are inspectable', async t => {
  const history = await fixture(t);
  const proposed = await history.proposeSkill({ id: 'checks', skills: { health: 'Run checks' }, metadata: { evidence: ['e1'] } });
  const memory = await history.commitMemory([lesson('new-memory')], callbacks);
  const promotion = await history.promoteSkill('checks', { metadata: { reason: 'validated' } });
  assert.deepEqual(await history.state('memory'), memory);
  assert.deepEqual(await history.state('skills'), promotion);
  const snapshot = await history.readRevision(promotion.revision);
  assert.deepEqual(snapshot.parents, [proposed.revision]);
  assert.equal(snapshot.event.type, 'skill-promotion');
  assert.equal(snapshot.event.metadata.reason, 'validated');
  assert.equal(snapshot.skills['health.md'], 'Run checks');
  assert.deepEqual(await history.promoteSkill('checks'), promotion);
  const rejected = await history.proposeSkill({ id: 'bad', skills: { health: 'Skip checks' } });
  const rejection = await history.rejectSkill('bad', { metadata: { reason: 'unsafe' } });
  const decision = await history.readRevision(rejection.revision);
  assert.deepEqual(decision.parents, [rejected.revision]);
  assert.equal(decision.event.metadata.reason, 'unsafe');
  await assert.rejects(history.promoteSkill('bad'), /rejected/);
  assert.deepEqual(await history.state('skills'), promotion);
});

test('stale skill candidates cannot replace newer skills; failed promotion resumes its decision commit', async t => {
  const history = await fixture(t);
  await history.proposeSkill({ id: 'one', skills: { one: 'First' } });
  await history.proposeSkill({ id: 'stale', skills: { stale: 'Old base' } });
  await assert.rejects(history.promoteSkill('one', { activate: async () => { throw new Error('offline'); } }), /offline/);
  const staged = await history.ref('refs/heads/skills/one');
  assert.equal(await history.state('skills'), null);
  assert.equal((await history.promoteSkill('one')).revision, staged);
  await assert.rejects(history.promoteSkill('stale'), /stale/);
  await history.proposeSkill({ id: 'two', skills: { two: 'Second' } });
  const latest = await history.promoteSkill('two');
  await assert.rejects(history.promoteSkill('one'), /superseded/);
  assert.deepEqual(await history.state('skills'), latest);
  assert.equal((await history.readRevision(latest.revision)).skills['one.md'], 'First');
});

test('invalid lessons and conflicting delivery payloads leave history unchanged', async t => {
  const history = await fixture(t);
  const initial = await history.ref('refs/heads/memory');
  await assert.rejects(history.commitMemory([{ ...lesson('bad'), evidence_refs: [] }], callbacks), /Invalid/);
  assert.equal(await history.ref('refs/heads/memory'), initial);
  const a = lesson('a');
  const b = { ...lesson('b'), id: 'other' };
  const result = await history.commitMemory([a, b], callbacks);
  await assert.rejects(history.commitMemory([{...a,lesson:'Changed payload'}], callbacks), /reused/);
  assert.equal(await history.ref('refs/heads/memory'), result.revision);
});

test('overlapping delivery replay preserves receipts and retries only the latest complete snapshot',async t=>{
  const history=await fixture(t),a=lesson('a'),b={...lesson('b'),id:'other'};
  await assert.rejects(history.commitMemory([a],{...callbacks,indexRevision:async()=>{throw Error('offline');}}),/offline/);
  const first=await history.ref('refs/heads/memory');
  await assert.rejects(history.commitMemory([a,b],{...callbacks,activate:async()=>{throw Error('activation offline');}}),/activation offline/);
  const second=await history.ref('refs/heads/memory');assert.notEqual(second,first);
  const count=await history.git(['rev-list','--count','refs/heads/memory']);
  const reopened=await new KnowledgeHistory(history.directory).init();let indexed=0;
  const active=await reopened.commitMemory([b,a],{...callbacks,indexRevision:async(revision,snapshot)=>{
    indexed++;assert.equal(revision,second);assert.deepEqual(snapshot.memory.map(r=>r.delivery_id).sort(),['a','b']);return revision;
  }});
  assert.equal(indexed,1);assert.equal(active.revision,second);
  assert.equal(await reopened.git(['rev-list','--count','refs/heads/memory']),count);
  assert.deepEqual(await reopened.commitMemory([a],{indexRevision:()=>assert.fail('already indexed'),activate:()=>assert.fail('already active')}),active);
  const {createHash}=await import('node:crypto');
  assert.equal(await reopened.ref('refs/knowledge/deliveries/'+createHash('sha256').update('a').digest('hex')),first);
  assert.equal(await reopened.ref('refs/knowledge/deliveries/'+createHash('sha256').update('b').digest('hex')),second);
  await assert.rejects(reopened.commitMemory([{...a,lesson:'conflict'},{...lesson('c'),id:'new'}],callbacks),/reused/);
  assert.equal(await reopened.ref('refs/heads/memory'),second);
});

test('mixed replay does not restore an older delivery over a corrected lesson',async t=>{
 const history=await fixture(t),old=lesson('original');
 await history.commitMemory([old],callbacks);
 const corrected=await history.commitMemory([lesson('correction','Corrected guidance')],callbacks);
 const added={...lesson('new'),id:'another'};
 const result=await history.commitMemory([old,added],callbacks);
 const snapshot=await history.readRevision(result.revision);
 assert.deepEqual(snapshot.parents,[corrected.revision]);
 assert.equal(snapshot.memory.find(r=>r.id==='health').lesson,'Corrected guidance');
 assert.equal(snapshot.memory.find(r=>r.id==='another').delivery_id,'new');
 assert.deepEqual(snapshot.event.deliveries,['new']);
});

test('process exit during indexing leaves a recoverable commit and delivery receipt', async t => {
  const history = await fixture(t);
  const initial = await history.commitMemory([lesson('before-crash')], callbacks);
  const { execFile } = await import('node:child_process');
  const { promisify } = await import('node:util');
  const module = new URL('../../controller/knowledge-history.mjs', import.meta.url).href;
  const records = [lesson('crash', 'Survives process exit')];
  const script = `import { KnowledgeHistory } from ${JSON.stringify(module)};
    const history = await new KnowledgeHistory(${JSON.stringify(history.directory)}).init();
    await history.commitMemory(${JSON.stringify(records)}, {
      indexRevision: async () => process.exit(23), activate: async () => process.exit(24)
    });`;
  await assert.rejects(promisify(execFile)(process.execPath, ['--max-old-space-size=256', '--input-type=module', '-e', script]),
    error => error.code === 23);
  const staged = await history.ref('refs/heads/memory');
  assert.notEqual(staged, initial.revision);
  assert.deepEqual(await history.state('memory'), initial);
  const reopened = await new KnowledgeHistory(history.directory).init();
  // Delivery equality is structural, independent of JSON property insertion order.
  const record = records[0];
  const reordered = { target: record.target, kind: record.kind, evidence_refs: record.evidence_refs,
    lesson: record.lesson, delivery_id: record.delivery_id, id: record.id };
  assert.equal((await reopened.commitMemory([reordered], callbacks)).revision, staged);
  assert.equal(await reopened.git(['rev-list', '--count', 'refs/heads/memory']), '3');
});

test('repository with a remote is rejected without contacting it', async t => {
  const history = await fixture(t);
  await history.git(['remote', 'add', 'origin', '/nonexistent/local-only-test']);
  await assert.rejects(new KnowledgeHistory(history.directory).init(), /remoteless/);
});

test('Git skill blobs preserve Unicode and trailing whitespace exactly',async t=>{
 const history=await fixture(t),text='# Skill\n'+('中文 café 🧪 '.repeat(3000))+'\n\n';
 const proposed=await history.proposeSkill({id:'unicode',skills:{debug:text}});
 assert.ok((await history.readRevision(proposed.revision)).skills['debug.md']===text,'Exact Unicode skill bytes round-trip');
});


async function bareFixture(t) {
  const directory=await mkdtemp(join(tmpdir(),'knowledge-bare-'));
  t.after(()=>rm(directory,{recursive:true,force:true}));
  const history=new KnowledgeHistory(directory);
  await history.git(['init','--bare','--initial-branch=memory']);
  return history.init();
}
const refs=history=>history.git(['for-each-ref','--format=%(refname) %(objectname)']);

test('legacy bare history reopens without nested initialization and retains skill and memory lineage',async t=>{
  const history=await bareFixture(t);
  const memory=await history.commitMemory([lesson('legacy')],callbacks);
  await history.proposeSkill({id:'legacy',skills:{debug:'旧 guidance 🧪\n\n'}});
  const promoted=await history.promoteSkill('legacy'),before=await refs(history);
  const reopened=await new KnowledgeHistory(history.directory).init();
  await assert.rejects(access(join(history.directory,'.git')),{code:'ENOENT'});
  assert.equal(await refs(reopened),before);
  assert.deepEqual(await reopened.state('memory'),memory);
  assert.deepEqual(await reopened.state('skills'),promoted);
  assert.equal((await reopened.readRevision(promoted.revision)).skills['debug.md'],'旧 guidance 🧪\n\n');
  const next=await reopened.commitMemory([lesson('next')],callbacks);
  assert.deepEqual((await reopened.readRevision(next.revision)).parents,[memory.revision]);
});

test('split history imports legacy objects without replacing either ref set and remains reachable after GC',async t=>{
  const legacy=await bareFixture(t);
  await legacy.commitMemory([lesson('legacy')],callbacks);
  await legacy.proposeSkill({id:'baseline',skills:{debug:'旧 baseline 🧪\n\n'}});
  const baseline=await legacy.promoteSkill('baseline'),oldRefs=await refs(legacy);
  const current=await fixture(t),memory=await current.commitMemory([lesson('new-memory')],callbacks);
  const currentRefs=await refs(current);
  await rename(join(current.directory,'.git'),join(legacy.directory,'.git'));
  const split=new KnowledgeHistory(legacy.directory);
  await assert.rejects(split.readRevision(baseline.revision),/git rev-parse/);
  const recovered=await split.init();
  assert.equal(await recovered.git(['--git-dir='+legacy.directory,'for-each-ref','--format=%(refname) %(objectname)']),oldRefs);
  for(const line of currentRefs.split('\n')) {
    const [ref,revision]=line.split(' ');assert.equal(await recovered.ref(ref),revision);
  }
  assert.deepEqual(await recovered.state('memory'),memory);
  assert.equal(await recovered.state('skills'),null);
  assert.equal((await recovered.readRevision(baseline.revision)).skills['debug.md'],'旧 baseline 🧪\n\n');
  for(const line of oldRefs.split('\n')) {
    const [ref,revision]=line.split(' ');
    assert.equal(await recovered.ref('refs/knowledge/legacy-bare/'+ref.slice(5)),revision);
  }
  const allRefs=await refs(recovered);
  await recovered.init();assert.equal(await refs(recovered),allRefs);
  await recovered.git(['gc','--prune=now']);
  assert.equal((await recovered.readRevision(baseline.revision)).event.type,'skill-promotion');
  await recovered.setState('skills',baseline);
  await recovered.proposeSkill({id:'next',skills:{debug:'new guidance'}});
  const promoted=await recovered.promoteSkill('next');
  assert.deepEqual(await recovered.state('memory'),memory);
  assert.equal((await recovered.readRevision(promoted.revision)).skills['debug.md'],'new guidance');
  const next=await recovered.commitMemory([lesson('following')],callbacks);
  assert.deepEqual((await recovered.readRevision(next.revision)).parents,[memory.revision]);
  assert.equal((await recovered.readRevision(memory.revision)).memory[0].delivery_id,'new-memory');
});

test('split history refuses changed legacy aliases and remote repositories without replacing current memory',async t=>{
  const legacy=await bareFixture(t),current=await fixture(t);
  const memory=await current.commitMemory([lesson('current')],callbacks);
  await rename(join(current.directory,'.git'),join(legacy.directory,'.git'));
  const recovered=await new KnowledgeHistory(legacy.directory).init();
  await recovered.git(['update-ref','refs/knowledge/legacy-bare/heads/memory',memory.revision]);
  await assert.rejects(recovered.init(),/Legacy knowledge history reference changed/);
  assert.deepEqual(await recovered.state('memory'),memory);
  await recovered.git(['--git-dir='+legacy.directory,'remote','add','origin','/nonexistent/local-only-test']);
  await assert.rejects(recovered.init(),/remoteless/);
  assert.deepEqual(await recovered.state('memory'),memory);
});
