import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { cadenceDecision, evaluateDecision, applyProposal, generateCandidate, tune } from '../../controller/skill-evolution.mjs';
import { run } from '../../cli/skills.mjs';

const task = (n, extra = {}) => ({ id: `original-${n}`, projectId: `project-${n}`,
  kind: 'original', phase: 'phase2', completed: true,
  events: [{ type: 'knowledge-retrieved', data: { kind: 'negative', evidence: 'failure.json' } }], ...extra });
const history = n => Array.from({ length: n }, (_, i) => task(i));
const guidance = { id: 'failure', scope: 'debug', text: 'Check the failing case.', critical: false, negative: true };
const config = { phase: 'phase2', skills: { activeVersion: 'baseline', tuningProjectIds: ['reg-a', 'reg-b'] } };
async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'skills-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'skills', 'versions', 'baseline'), { recursive: true });
  await writeFile(join(root, 'config.json'), JSON.stringify(config));
  await writeFile(join(root, 'skills', 'versions', 'baseline', 'guidance.json'), JSON.stringify([guidance]));
  return root;
}
const load = async path => JSON.parse(await readFile(path, 'utf8'));

test('cadence excludes phase1, retries, diagnostics, duplicates and repeated projects', () => {
  for (const extra of [{ phase: 'phase1' }, { kind: 'retry' }, { kind: 'diagnostic' }, { completed: false }])
    assert.equal(cadenceDecision(history(7), task(7, extra)).analyse, false);
  assert.equal(cadenceDecision(history(7), task(0)).analyse, false);
  assert.equal(cadenceDecision(history(7), task(7, { projectId: 'project-0' })).evaluate, false);
  assert.equal(cadenceDecision(history(7), task(7)).evaluate, true);
  assert.equal(cadenceDecision(history(23), task(23)).allowPruning, true);
  assert.equal(cadenceDecision(history(22), task(22)).allowPruning, false);
  assert.equal(cadenceDecision(history(8), task(8), 8).evaluate, false);
});

test('paired decision rejects correctness loss, missing samples, ties and regressions', () => {
  const results = ['reg-a', 'reg-b'].flatMap(projectId => [0, 1].flatMap(run =>
    ['active', 'candidate'].map(arm => ({ projectId, run, arm, correct: true, score: arm === 'candidate' ? 2 : 1 }))));
  assert.equal(evaluateDecision(results, config.skills.tuningProjectIds).promote, true);
  for (const patch of [{ correct: false }, { score: 1 }, { score: 0 }, { correct: undefined }]) {
    const changed = structuredClone(results);
    Object.assign(changed[1], patch);
    assert.equal(evaluateDecision(changed, config.skills.tuningProjectIds).promote, false);
  }
  assert.equal(evaluateDecision(results.slice(1), config.skills.tuningProjectIds).promote, false);
});

test('negative retention and evidence-gated critical pruning are enforced', () => {
  const remove = [{ action: 'remove', id: 'failure' }];
  const options = { allowPruning: true, history: history(24) };
  assert.throws(() => applyProposal([guidance], remove, options));
  const critical = { ...guidance, negative: false, critical: true };
  assert.throws(() => applyProposal([critical], remove, options));
  const contradictory = task(24, { events: [{ type: 'guidance-contradicted',
    data: { guidanceId: 'failure', evidence: 'verifier.json' } }] });
  assert.deepEqual(applyProposal([critical], [{ ...remove[0], contradictoryEvidence: ['verifier.json'] }],
    { ...options, history: [...options.history, contradictory] }), []);
  const ordinary = { ...critical, critical: false };
  assert.throws(() => applyProposal([ordinary], remove, { allowPruning: false, history: [] }));
  assert.deepEqual(applyProposal([ordinary], remove, options), []);
});

test('native SDK injection receives evidence and disposes its session', async t => {
  const root = await fixture(t);
  let listener, prompted, disposed = false, bound = false;
  const native = {
    SessionManager: { create: () => ({}) },
    DefaultResourceLoader: class { async reload() {} },
    createAgentSession: async options => {
      assert.deepEqual(options.tools, []);
      return { session: {
        subscribe(fn) { listener = fn; return () => {}; },
        async bindExtensions() { bound = true; },
        async prompt(text) { prompted = text; listener({ type: 'message_end', message: {
          role: 'assistant', content: [{ type: 'text', text: '{"changes":[]}' }] } }); },
        async waitForIdle() {}, dispose() { disposed = true; }
      } };
    }
  };
  assert.deepEqual(await generateCandidate({ directory: join(root, 'proposal'), input: task(0), native }), []);
  assert.ok(prompted && bound && disposed);
  assert.equal((await load(join(root, 'proposal', 'responses.json'))).length, 1);
});

test('eight originals evaluate in isolated arms, preserve evidence and promote only passing candidate', async t => {
  const root = await fixture(t);
  const seen = [];
  const evaluate = async request => {
    seen.push(request);
    assert.equal(request.config.skills.tuningDisabled, true);
    await mkdir(request.instanceRoot, { recursive: true });
    await writeFile(join(request.instanceRoot, 'sft.json'), '["retained"]');
    return { correct: true, score: request.arm === 'candidate' ? 2 : 1 };
  };
  for (let i = 0; i < 8; i++) {
    const result = await tune({ instanceRoot: root, config, task: task(i), evaluate,
      generate: async ({ input }) => { assert.equal(input.actualEvents.length, 1); return []; } });
    if (i < 7) assert.equal(result.evaluate, false);
    else assert.equal(result.promote, true);
  }
  assert.equal(seen.length, 8);
  assert.equal(new Set(seen.map(x => x.instanceRoot)).size, 8);
  const active = (await load(join(root, 'config.json'))).skills.activeVersion;
  assert.notEqual(active, 'baseline');
  assert.equal((await readdir(join(root, 'skills', 'versions'))).length, 9);
  assert.deepEqual(await load(join(seen[0].instanceRoot, 'sft.json')), ['retained']);
  assert.deepEqual(await load(join(root, 'skills', 'versions', 'baseline', 'guidance.json')), [guidance]);
});

test('inconclusive result keeps instance active; phase1 never invokes proposal or evaluator', async t => {
  const root = await fixture(t);
  await writeFile(join(root, 'skills', 'evolution.json'), JSON.stringify({ history: history(7), lastEvaluated: 0 }));
  const result = await tune({ instanceRoot: root, config, task: task(7), generate: async () => [],
    evaluate: async () => ({ correct: true, score: 1 }) });
  assert.equal(result.promote, false);
  assert.equal((await load(join(root, 'config.json'))).skills.activeVersion, 'baseline');
  await tune({ instanceRoot: root, config, task: task(8, { phase: 'phase1' }),
    generate: async () => assert.fail('phase1 proposal') });
});

test('CLI requires runtime contract and regression project configuration', async () => {
  await assert.rejects(run({ skills: {} }), /tuningProjectIds/);
  await assert.rejects(run(config), /runtimeModule/);
  assert.equal((await run({ phase: 'phase1' })).analyse, false);
});
