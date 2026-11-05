import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
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

test('invalid lessons and overlapping delivery batches leave history unchanged', async t => {
  const history = await fixture(t);
  const initial = await history.ref('refs/heads/memory');
  await assert.rejects(history.commitMemory([{ ...lesson('bad'), evidence_refs: [] }], callbacks), /Invalid/);
  assert.equal(await history.ref('refs/heads/memory'), initial);
  const a = lesson('a');
  const b = { ...lesson('b'), id: 'other' };
  const result = await history.commitMemory([a, b], callbacks);
  await assert.rejects(history.commitMemory([a], callbacks), /reused/);
  await assert.rejects(history.commitMemory([a, { ...b, delivery_id: 'c' }], callbacks), /overlaps/);
  assert.equal(await history.ref('refs/heads/memory'), result.revision);
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
 assert.equal((await history.readRevision(proposed.revision)).skills.debug,text);
});
