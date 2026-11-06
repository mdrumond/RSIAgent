import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, readdir, rm, cp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { cadenceDecision, evaluateDecision, applyProposal, candidateVersion, tune } from '../../controller/skill-evolution.mjs';
import { run } from '../../cli/skills.mjs';
import { nativeWriter } from './skill-proposal-fixture.mjs';

const task = (n, extra = {}) => ({ id: `original-${n}`, projectId: `project-${n}`,
  kind: 'original', phase: 'phase2', completed: true,
  events: [{ type: 'knowledge-retrieved', data: { kind: 'negative', evidence: 'failure.json' } }], ...extra });
const history = n => Array.from({ length: n }, (_, i) => task(i));
const guidance = { id: 'failure', scope: 'debug', text: 'Check the failing case.', critical: false, negative: true };
const config = { root: tmpdir(), mode: 'phase2', skills: { activeVersion: 'baseline', tuningProjectIds: ['reg-a', 'reg-b'] } };
async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'skills-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'skills', 'versions', 'baseline'), { recursive: true });
  await writeFile(join(root, 'instance.json'), JSON.stringify(config));
  await writeFile(join(root, 'skills', 'versions', 'baseline', 'guidance.json'), JSON.stringify([guidance]));
  return root;
}
const load = async path => JSON.parse(await readFile(path, 'utf8'));
const lockedRun = (config, values = {}) => run(config, {
  lock: async (_config, operation) => operation(),
  readConfig: async root => ({ ...await load(join(root, 'instance.json')), root }), ...values });

test('cadence excludes phase1, retries, diagnostics, duplicates and repeated projects', () => {
  for (const extra of [{ phase: 'phase1' }, { completed: false }])
    assert.equal(cadenceDecision(history(7), task(7, extra)).analyse, false);
  for (const kind of ['retry', 'diagnostic', 'investigation']) {
    const decision = cadenceDecision(history(7), task(7, { kind }));
    assert.equal(decision.analyse, true);
    assert.equal(decision.evaluate, false);
    assert.equal(decision.projects, 7);
  }
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

test('candidate repairs active failures before performance, but reverse never promotes', () => {
  const results = ['reg-a', 'reg-b'].flatMap(projectId => [0, 1].flatMap(run =>
    ['active', 'candidate'].map(arm => ({ projectId, run, arm, correct: true,
      score: arm === 'active' ? 2 : 1 }))));
  results[0].correct = false;
  delete results[0].score;
  assert.deepEqual(evaluateDecision(results, config.skills.tuningProjectIds),
    { promote: true, reason: 'reliability-improvement' });
  const reverse = results.map(r => ({ ...r, arm: r.arm === 'active' ? 'candidate' : 'active' }));
  assert.deepEqual(evaluateDecision(reverse, config.skills.tuningProjectIds),
    { promote: false, reason: 'correctness-regression' });
  results[3].correct = false;
  assert.equal(evaluateDecision(results, config.skills.tuningProjectIds).promote, false);
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
  const active = (await load(join(root, 'instance.json'))).skills.activeVersion;
  assert.notEqual(active, 'baseline');
  assert.equal((await load(join(root, 'instance.json'))).revision, 1);
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
  assert.equal((await load(join(root, 'instance.json'))).skills.activeVersion, 'baseline');
  await tune({ instanceRoot: root, config, task: task(8, { phase: 'phase1' }),
    generate: async () => assert.fail('phase1 proposal') });
});

test('Phase1 CLI does not access action files, runtime or actor profile', async () => {
  assert.equal((await run({ mode: 'phase1' }, { config: '/absent' })).analyse, false);
});


test('CLI uses instance.json and invokes bootstrap then canonical wave for each fresh arm', async t => {
  const root = await fixture(t);
  await writeFile(join(root, 'skills', 'evolution.json'), JSON.stringify({
    history: history(8), lastEvaluated: 0, candidate: 'baseline', candidateBase: 'baseline',
    pending: { active: 'baseline', candidate: 'baseline', projects: 8 }
  }));
  // Different immutable candidate prevents interrupted-promotion reconciliation.
  const pending = await load(join(root, 'skills', 'evolution.json'));
  pending.pending.candidate = 'candidate';
  await writeFile(join(root, 'skills', 'evolution.json'), JSON.stringify(pending));
  const calls = [], controller = new AbortController(), resource = { id: 'opaque-lease' };
  const runtime = {
    async bootstrap(request) { assert.equal(request.signal, controller.signal); assert.equal(request.resource, resource);
      assert.deepEqual(request.projectIds, config.skills.tuningProjectIds); calls.push(['bootstrap', request.instanceRoot]); await mkdir(request.instanceRoot); },
    async wave(request) {
      calls.push(['wave', request.instanceRoot]);
      assert.deepEqual(request.projectIds.length, 1);
      assert.equal(request.evaluation, true);
      assert.equal(request.signal, controller.signal);
      assert.equal(request.config.root, request.instanceRoot);
      return { correct: true, score: 1 };
    }
  };
  const result = await lockedRun({ ...config, root }, { runtime, signal: controller.signal, resource });
  assert.equal(result.promote, false);
  assert.equal(calls.length, 16);
  for (let i = 0; i < calls.length; i += 2) {
    assert.equal(calls[i][0], 'bootstrap');
    assert.equal(calls[i + 1][0], 'wave');
    assert.equal(calls[i][1], calls[i + 1][1]);
  }
});

test('CLI consumes action file task and requires an explicit writer profile', async t => {
  const root = await fixture(t);
  const action = join(root, 'action.json');
  await writeFile(action, JSON.stringify({ task: task(0), userinfo: 'Prefer recovery guidance.' }));
  await assert.rejects(lockedRun({ ...config, root }, { config: action,
    runtime: { async bootstrap() {}, async wave() {} } }), /skill_writer_actor or actor/);
  const proposals = await readdir(join(root, 'skills', 'proposals'));
  const evidence = await load(join(root, 'skills', 'proposals', proposals[0], 'evidence.json'));
  assert.equal(evidence.id, task(0).id);
  assert.equal((await run({ mode: 'phase1', root }, { config: '/missing-action.json' })).analyse, false);
});

const knowledgeStub = { async ensureKnowledgeService() { return 'http://127.0.0.1:8765/mcp'; },
  createKdbBridge() { return { extension: () => {} }; } };
test('first CLI update needs no A/B configuration; cadence defaults to two stable originals', async t => {
  const root = await fixture(t);
  const current = { ...config, root, skills: { activeVersion: 'baseline' } };
  await writeFile(join(root, 'instance.json'), JSON.stringify(current));
  const { native } = nativeWriter();
  const first = await lockedRun(current, { task: task(0), native, knowledge: knowledgeStub, modelRuntime: {}, model: {} });
  assert.ok(first.candidate);
  assert.equal(first.evaluate, false);
  const evaluated = [];
  for (let i = 1; i < 8; i++) await tune({ instanceRoot: root, config: current, task: task(i),
    generate: async () => [], evaluate: async request => {
      evaluated.push(request.projectId); return { correct: true, score: 1 };
    } });
  assert.deepEqual([...new Set(evaluated)], ['project-0', 'project-1']);
  assert.equal(evaluated.length, 8);
  assert.equal((await load(join(root, 'skills', 'evolution.json'))).history.length, 8);
});

test('manual CLI uses selected State events without increasing original cadence', async t => {
  const { State } = await import('../../controller/state.mjs');
  const root = await fixture(t);
  const state = new State(join(root, 'state.sqlite'));
  state.add({ id: 'selected', goal: 'numeric', actor_seconds: 20 });
  const attempt = state.admit('selected', { pid: 1 });
  state.event(attempt.id, 'knowledge-retrieved', { kind: 'negative', evidence: 'failed-case' });
  state.finish(attempt.id, 'failed'); state.close();
  const action = join(root, 'manual.json');
  await writeFile(action, JSON.stringify({ userinfo: 'repair', projectId: 'selected' }));
  let input;
  const { native } = nativeWriter(value => { input = value; });
  const result = await lockedRun({ ...config, root }, { config: action, native, knowledge: knowledgeStub, modelRuntime: {}, model: {},
    runtime: { async bootstrap(request) { await mkdir(request.instanceRoot); },
      async wave() { return { correct: true, score: 1 }; } } });
  assert.equal(result.promote, false);
  assert.equal(input.context.projectId, 'selected');
  assert.equal(input.actualEvents[0].data.evidence, 'failed-case');
  const saved = await load(join(root, 'skills', 'evolution.json'));
  assert.equal(saved.history.length, 0);
  assert.equal(saved.lastEvaluated, 0);
  assert.ok(saved.candidate);
  const noIds = { ...config, root, skills: { activeVersion: 'baseline' } };
  await writeFile(join(root, 'instance.json'), JSON.stringify(noIds));
  await writeFile(action, JSON.stringify({ userinfo: 'recover' }));
  assert.equal((await lockedRun(noIds, { config: action, native, knowledge: knowledgeStub, modelRuntime: {}, model: {} })).analyse, true);
  assert.equal((await load(join(root, 'skills', 'evolution.json'))).history.length, 0);
});

test('manual CLI retains bounded project history when its latest attempt has just started', async t => {
  const { State } = await import('../../controller/state.mjs');
  const root = await fixture(t), current = { ...config, root, skills: { activeVersion: 'baseline' } };
  await writeFile(join(root, 'instance.json'), JSON.stringify(current));
  const state = new State(join(root, 'state.sqlite'));
  state.add({ id: 'selected', goal: 'numeric', actor_seconds: 20 });
  const old = state.admit('selected', { pid: 1 });
  state.event(old.id, 'knowledge-retrieved', { kind: 'negative', evidence: 'retained-negative-case' });
  state.event(old.id, 'skill-read', { stage: 'debug', evidence: 'used-guidance' });
  state.recordFailure(old.id, { phase: 'implementation', kind: 'compile', message: 'observed unsupported operation',
    evidence: [{ path: 'retained-compile.log' }] });
  state.finish(old.id, 'failed');
  state.db.prepare("UPDATE projects SET status='queued' WHERE id='selected'").run();
  const latest = state.admit('selected', { pid: 1 });
  const identity = { goal: '1'.repeat(64), startingSource: '2'.repeat(64),
    verifier: '3'.repeat(64), targetRuntime: '4'.repeat(64) };
  state.event(latest.id, 'skill-evidence-context', identity); state.close();
  const action = join(root, 'manual-history.json');
  await writeFile(action, JSON.stringify({ userinfo: 'repair', projectId: 'selected' }));
  let input;
  const { native } = nativeWriter(value => { input = value; });
  const result = await lockedRun(current, { config: action, native, knowledge: knowledgeStub, modelRuntime: {}, model: {} });
  assert.equal(result.evaluate, false);
  assert.equal(input.context.id, latest.id); assert.equal(input.context.outcome, null);
  assert.equal(input.context.completed, false); assert.deepEqual(input.context.comparisonIdentity, identity);
  assert.ok(input.actualEvents.some(event => event.type === 'knowledge-retrieved' && event.data.evidence === 'retained-negative-case'));
  assert.ok(input.actualEvents.some(event => event.type === 'skill-read' && event.data.stage === 'debug'));
  assert.deepEqual(input.context.events.find(event => event.type === 'failure-recorded').data.evidence,
    [{ path: 'retained-compile.log' }]);
  assert.equal(input.context.events.length, 3);
  assert.equal((await load(join(root, 'skills', 'evolution.json'))).history.length, 0);
});

test('Markdown baseline seeds protected entries and preserves original relative assets in candidate', async t => {
  const { stages } = await import('../../controller/workflow.mjs');
  const root = await fixture(t);
  const baseline = join(root, 'skills', 'versions', 'baseline');
  await rm(join(baseline, 'guidance.json'));
  await mkdir(join(baseline, 'fork'));
  const map = {};
  for (const scope of ['global-kdb', ...stages]) {
    map[scope] = `fork/${scope}.md`;
    await writeFile(join(baseline, map[scope]), `# ${scope}\n[asset](reference.bin)`);
  }
  await writeFile(join(baseline, 'fork', 'reference.bin'), Buffer.from([1, 2, 3]));
  await writeFile(join(baseline, 'skill-map.json'), JSON.stringify(map));
  const result = await tune({ instanceRoot: root, config, task: task(0), generate: async ({ input }) => {
    assert.equal(input.entries.length, 10);
    assert.ok(input.entries.every(entry => entry.critical));
    return [];
  } });
  const candidate = join(root, 'skills', 'versions', result.candidate);
  const generatedMap = await load(join(candidate, 'skill-map.json'));
  assert.equal(generatedMap.design, 'fork/evolved-design.md');
  assert.deepEqual(await readFile(join(candidate, 'fork', 'reference.bin')),
    await readFile(join(baseline, 'fork', 'reference.bin')));
  assert.deepEqual(await readFile(join(candidate, map.design)), await readFile(join(baseline, map.design)));
  assert.equal((await load(join(candidate, 'guidance.json'))).length, 10);
  assert.equal((await load(join(root, 'instance.json'))).skills.activeVersion, 'baseline');
});


test('investigation and retry update candidate but only originals advance cadence; state drops Pi telemetry', async t => {
  const root = await fixture(t);
  let analysed = 0;
  for (const [i, kind] of ['original', 'investigation', 'retry'].entries()) {
    const result = await tune({ instanceRoot: root, config, task: task(i, { kind,
      events: [...task(i).events, { type: 'pi-event', data: { duplicate: true } }] }),
      generate: async ({ input, validate }) => {
        analysed++;
        assert.ok(input.history.every(task => task.events.every(event => event.type !== 'pi-event')));
        validate([]); return [];
      } });
    assert.ok(result.candidate);
    assert.equal(result.evaluate, false);
    assert.equal(result.projects, 1);
  }
  assert.equal(analysed, 3);
  const state = await load(join(root, 'skills', 'evolution.json'));
  assert.equal(state.history.length, 3);
  const many = Array.from({ length: 30 }, (_, i) => task(i, { kind: 'investigation' }));
  assert.equal(cadenceDecision(many, task(31)).allowPruning, false);
});

test('CLI refreshes after lock claim and preserves executed mode while using fresh skills and userinfo', async t => {
  const root = await fixture(t);
  await cp(join(root, 'skills', 'versions', 'baseline'), join(root, 'skills', 'versions', 'promoted'), { recursive: true });
  const fresh = { ...config, mode: 'phase1', revision: 1, userinfo: 'fresh', skills: { activeVersion: 'promoted' } };
  const stale = { ...config, root, revision: 0, userinfo: 'stale' };
  let held = false, claims = 0;
  const controller = new AbortController();
  const lock = async (_config, operation, options) => {
    assert.equal(options.signal, controller.signal);
    claims++; held = true;
    if (claims === 1) await writeFile(join(root, 'instance.json'), JSON.stringify(fresh));
    try { return await operation(); } finally { held = false; }
  };
  let observed;
  const { native } = nativeWriter(input => { assert.equal(held, true); observed = input; });
  const dependencies = { lock, signal: controller.signal, readConfig: async path => {
    assert.equal(held, true); return { ...await load(join(path, 'instance.json')), root: path };
  }, native, knowledge: knowledgeStub, modelRuntime: {}, model: {} };
  const result = await run(stale, { ...dependencies, task: task(0) });
  assert.ok(result.candidate);
  assert.equal(observed.userinfo, 'fresh');
  assert.equal(observed.context.phase, 'phase2');
  assert.equal(held, false);
  const latest = await load(join(root, 'instance.json'));
  assert.equal(latest.skills.activeVersion, 'promoted');
  assert.equal(latest.revision, 1);
  latest.mode = 'phase2';
  await writeFile(join(root, 'instance.json'), JSON.stringify(latest));
  assert.ok((await run(stale, { ...dependencies, manual: true })).candidate);
  assert.equal(claims, 2);
  assert.equal(held, false);
});

test('stopped canonical bootstrap/wave retains partial A/B and never promotes or consumes cadence', async t => {
  for (const stopAt of ['bootstrap', 'wave']) {
    const root = await fixture(t), controller = new AbortController();
    await writeFile(join(root, 'skills', 'evolution.json'), JSON.stringify({ history: history(8), lastEvaluated: 0,
      pending: { active: 'baseline', candidate: 'candidate', projects: 8 } }));
    let waves = 0;
    const runtime = {
      async bootstrap(request) {
        assert.equal(request.signal, controller.signal);
        await mkdir(request.instanceRoot); await writeFile(join(request.instanceRoot, 'partial.log'), 'retained');
        if (stopAt === 'bootstrap') controller.abort();
      },
      async wave(request) { waves++; assert.equal(request.signal, controller.signal);
        controller.abort(); return { correct: true, score: 2 }; }
    };
    await assert.rejects(lockedRun({ ...config, root }, { runtime, signal: controller.signal }), { name: 'AbortError' });
    assert.equal(waves, stopAt === 'wave' ? 1 : 0);
    assert.equal((await load(join(root, 'instance.json'))).skills.activeVersion, 'baseline');
    const state = await load(join(root, 'skills', 'evolution.json'));
    assert.equal(state.lastEvaluated, 0); assert.ok(state.pending);
    const evaluations = join(root, 'skills', 'evaluations'), ids = await readdir(evaluations);
    assert.equal((await load(join(evaluations, ids[0], 'results.json'))).length, waves);
  }
});

test('authoritative user edit supersedes pending candidate without deleting evidence or version history', async t => {
  const root = await fixture(t);
  await writeFile(join(root, 'skills', 'evolution.json'), JSON.stringify({ history: history(7), lastEvaluated: 0 }));
  await assert.rejects(tune({ instanceRoot: root, config, task: task(7), generate: async () => [],
    evaluate: async () => { throw new Error('paused'); } }), /paused/);
  const before = await load(join(root, 'skills', 'evolution.json'));
  assert.equal(candidateVersion(before, 'baseline'), before.candidate);
  const edited = { ...config, revision: 1, skills: { ...config.skills, activeVersion: 'user-edit' } };
  const directory = join(root, 'skills', 'versions', 'user-edit');
  await cp(join(root, 'skills', 'versions', 'baseline'), directory, { recursive: true });
  await writeFile(join(directory, 'guidance.json'), JSON.stringify([{ ...guidance, id: 'user-entry' }]));
  await writeFile(join(root, 'instance.json'), JSON.stringify(edited));
  assert.equal(candidateVersion(before, edited.skills.activeVersion), 'user-edit');
  await tune({ instanceRoot: root, config: edited, task: task(7), generate: async () => assert.fail('duplicate update') });
  const invalidated = await load(join(root, 'skills', 'evolution.json'));
  assert.equal(invalidated.candidate, null); assert.equal(invalidated.pending, null);
  assert.equal(invalidated.history.length, 8); assert.equal(invalidated.lastEvaluated, 0);
  assert.equal((await readdir(join(root, 'skills', 'evaluations'))).length, 1);
  await readFile(join(root, 'skills', 'versions', before.candidate, 'guidance.json'));
  const updated = await tune({ instanceRoot: root, config: edited, task: task(8, { kind: 'retry' }),
    generate: async ({ input }) => { assert.equal(input.entries[0].id, 'user-entry'); return []; } });
  const after = await load(join(root, 'skills', 'evolution.json'));
  assert.equal(after.candidateBase, 'user-edit');
  assert.equal(candidateVersion(after, 'user-edit'), updated.candidate);
});

test('configured two-original cadence evaluates earlier than default eight with validated thresholds', async t => {
  assert.equal(cadenceDecision(history(1), task(1)).evaluate, false);
  assert.equal(cadenceDecision(history(1), task(1), 0, 2).evaluate, true);
  assert.equal(cadenceDecision(history(1), task(1), 0, 2, 2).allowPruning, true);
  for (const invalid of [0, -1, 1.5, '2']) assert.throws(() => cadenceDecision([], task(0), 0, invalid));
  const root = await fixture(t), current = { ...config, tune_every: 2,
    skills: { activeVersion: 'baseline', pruneAfterTasks: 4 } };
  await writeFile(join(root, 'instance.json'), JSON.stringify(current));
  let arms = 0;
  for (let i = 0; i < 2; i++) {
    await tune({ instanceRoot: root, config: current, task: task(i),
      generate: async ({ input }) => { assert.equal(input.pruneAfterTasks, 4); return []; },
      evaluate: async () => { arms++; return { correct: true, score: 1 }; } });
    assert.equal(arms, i ? 8 : 0);
  }
  assert.equal((await load(join(root, 'skills', 'evolution.json'))).lastEvaluated, 2);
});

test('finishing or reconciling older A/B consumes a different incoming task exactly once', async t => {
  for (const ending of ['inconclusive', 'promoted', 'reconciled']) {
    const root = await fixture(t), candidate = 'old-candidate';
    await cp(join(root, 'skills', 'versions', 'baseline'), join(root, 'skills', 'versions', candidate), { recursive: true });
    const current = ending === 'reconciled' ? { ...config, skills: { ...config.skills, activeVersion: candidate } } : config;
    await writeFile(join(root, 'instance.json'), JSON.stringify(current));
    await writeFile(join(root, 'skills', 'evolution.json'), JSON.stringify({ history: history(8), lastEvaluated: 0,
      candidate, candidateBase: 'baseline', pending: { active: 'baseline', candidate, projects: 8 } }));
    let analyses = 0, arms = 0;
    const incoming = task(8);
    const generate = async ({ input }) => {
      analyses++; assert.equal(input.context.id, incoming.id); assert.equal(input.history.length, 9); return [];
    };
    const evaluate = async ({ arm }) => { arms++; return { correct: true,
      score: ending === 'promoted' && arm === 'candidate' ? 2 : 1 }; };
    assert.equal((await tune({ instanceRoot: root, config: current, task: incoming, generate, evaluate })).analyse, true);
    const saved = await load(join(root, 'skills', 'evolution.json'));
    assert.equal(saved.history.length, 9); assert.equal(saved.history.filter(x => x.id === incoming.id).length, 1);
    assert.equal(saved.lastEvaluated, 8); assert.equal(saved.pending, null);
    assert.equal(analyses, 1); assert.equal(arms, ending === 'reconciled' ? 0 : 8);
    const latest = await load(join(root, 'instance.json'));
    assert.equal((await tune({ instanceRoot: root, config: latest, task: incoming, generate, evaluate })).analyse, false);
    assert.equal(analyses, 1);
  }
});
