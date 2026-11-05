import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { tune } from '../../controller/skill-evolution.mjs';
import { evaluateRound, tuningStatus } from '../../controller/evaluation-round.mjs';

const load = async path => JSON.parse(await readFile(path, 'utf8'));
const save = async (path, value) => writeFile(path, JSON.stringify(value));
async function fixture(t, seconds) {
  const root = await mkdtemp(join(tmpdir(), 'evaluation-budget-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'skills', 'versions', 'baseline'), { recursive: true });
  await save(join(root, 'skills', 'versions', 'baseline', 'guidance.json'), []);
  const config = { root, mode: 'phase2', tune_every: 2,
    skills: { activeVersion: 'baseline', tuningProjectIds: ['a', 'b'], ...(seconds ? { evaluationSeconds: seconds } : {}) } };
  await save(join(root, 'instance.json'), config);
  return { root, config, run: values => tune({ instanceRoot: root, config, generate: async () => [], ...values }) };
}
const task = n => ({ id: `t${n}`, projectId: `p${n}`, completed: true, phase: 'phase2', kind: 'original', outcome: 'failed', events: [] });
const score = request => ({ correct: true, score: request.arm === 'candidate' ? 2 : 1 });

test('no allocation batches proposals without evaluations, including manual requests', async t => {
  const f = await fixture(t);
  for (const values of [{ task: task(0) }, { task: task(1) }, { manual: true }]) {
    const result = await f.run({ ...values, evaluate: () => assert.fail('unallocated execution') });
    assert.equal(result.evaluationStatus, 'deferred');
  }
  const state = await load(join(f.root, 'skills', 'evolution.json'));
  assert.equal(state.history.length, 2);
  assert.equal(state.pending, null);
  assert.equal((await load(join(f.root, 'instance.json'))).skills.activeVersion, 'baseline');
  assert.equal((await tuningStatus(f.root)).status, 'deferred');
});

test('deadline retains samples and evidence; unchanged config never refills, explicit grant resumes once', async t => {
  const f = await fixture(t, 1);
  let milliseconds = 0, calls = 0;
  const clock = { now: () => milliseconds, wallNow: () => milliseconds };
  await f.run({ task: task(0) });
  const result = await f.run({ task: task(1), clock, evaluate: async request => {
    calls++; assert.ok(request.remainingSeconds <= 1);
    milliseconds += 1100; return score(request);
  } });
  assert.equal(result.evaluationStatus, 'exhausted'); assert.equal(calls, 1);
  const root = result.evaluationRoot;
  const journal = await load(join(root, 'round.json'));
  assert.equal(journal.samples.filter(sample => sample.result).length, 1);
  const incoming = await f.run({ task: task(2), clock, evaluate: () => assert.fail('silent refill') });
  assert.equal(incoming.evaluationStatus, 'exhausted');
  assert.equal((await load(join(f.root, 'skills', 'evolution.json'))).history.length, 3);
  const resumed = await f.run({ evaluationSeconds: 10, allocationId: 'approved-extension', clock,
    generate: () => assert.fail('grant must not invoke writer'), evaluate: async request => { calls++; return score(request); } });
  assert.equal(resumed.promote, true); assert.equal(calls, 8);
  assert.equal(resumed.evaluationRoot, root);
  assert.equal((await load(join(root, 'round.json'))).allocatedSeconds, 11);
  const config = await load(join(f.root, 'instance.json'));
  await tune({ instanceRoot: f.root, config, evaluationSeconds: 10, allocationId: 'approved-extension',
    generate: () => assert.fail('grant replay writer'), evaluate: () => assert.fail('grant replay evaluator') });
  await assert.rejects(tune({ instanceRoot: f.root, config, evaluationSeconds: 12, allocationId: 'approved-extension' }), /different seconds/);
});

test('interrupted sample retains identity, completed sample is skipped, and paused time is excluded', async t => {
  const f = await fixture(t, 10), controller = new AbortController();
  let milliseconds = 0, firstRoot, interruptedRoot, calls = 0;
  const clock = { now: () => milliseconds, wallNow: () => milliseconds };
  await f.run({ task: task(0) });
  await assert.rejects(f.run({ task: task(1), signal: controller.signal, clock, evaluate: async request => {
    calls++; milliseconds += 100;
    if (calls === 1) { firstRoot = request.instanceRoot; return score(request); }
    interruptedRoot = request.instanceRoot; controller.abort(); request.signal.throwIfAborted();
  } }), { name: 'AbortError' });
  assert.equal((await tuningStatus(f.root)).status, 'interrupted');
  milliseconds += 100000;
  let resumed = false;
  const result = await f.run({ clock, evaluate: async request => {
    assert.notEqual(request.instanceRoot, firstRoot);
    if (!resumed) { assert.equal(request.instanceRoot, interruptedRoot); resumed = true; }
    milliseconds += 100; return score(request);
  } });
  assert.equal(result.promote, true);
  const round = await load(join(result.evaluationRoot, 'round.json'));
  assert.ok(round.elapsedSeconds < 2); assert.equal(round.samples.length, 8);
});

test('crash checkpoint charges unobserved elapsed time and blocks another dispatch', async t => {
  const f = await fixture(t), root = join(f.root, 'round'); await mkdir(root);
  await save(join(root, 'round.json'), { elapsedSeconds: 0.2, runningAt: 1000, samples: [], status: 'running' });
  const result = await evaluateRound({ pending: { evaluationRoot: root, evaluationSeconds: 1 }, config: {},
    wallNow: () => 3000, evaluate: () => assert.fail('crash refunded elapsed') });
  assert.equal(result.status, 'exhausted'); assert.equal(result.elapsedSeconds, 2.2);
});

test('real elapsed deadline cancels nested execution and never promotes partial results', async t => {
  const f = await fixture(t, 0.02); await f.run({ task: task(0) });
  let cancelled = false;
  const result = await f.run({ task: task(1), evaluate: async request => new Promise((resolve, reject) => {
    request.signal.addEventListener('abort', () => { cancelled = true; reject(request.signal.reason); }, { once: true });
  }) });
  assert.equal(cancelled, true); assert.equal(result.evaluationStatus, 'exhausted');
  assert.equal((await load(join(f.root, 'instance.json'))).skills.activeVersion, 'baseline');
});
