import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { State } from '../../controller/state.mjs';
import { freezeFiles, fingerprint } from '../../controller/evidence.mjs';
import { prepareSourceGit } from '../../controller/source-git.mjs';
import { recordSkillContext, tuningTask, selectSkillEvidence, environmentBinding } from '../../controller/skill-evidence.mjs';
import { tune } from '../../controller/skill-evolution.mjs';
import { nativeWriter } from './skill-proposal-fixture.mjs';

const identity = { goal: '1'.repeat(64), startingSource: '2'.repeat(64),
  verifier: '3'.repeat(64), targetRuntime: '4'.repeat(64) };
const task = (id, outcome = 'failed', extra = {}) => ({ id, projectId: 'reduction', kind: 'original',
  phase: 'phase2', completed: true, outcome, comparisonIdentity: identity,
  events: [{ type: 'verification', data: { passed: outcome === 'passed',
    correctness: { evidence: `${id}.log`, evidence_sha256: '5'.repeat(64) } } }], ...extra });
const load = async path => JSON.parse(await readFile(path, 'utf8'));
async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'skill-evidence-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

test('failure-only and legacy attempts remain usable without historical success', () => {
  for (const comparisonIdentity of [undefined, { ...identity, verifier: null }, identity]) {
    const evidence = selectSkillEvidence(task('current', 'failed', { comparisonIdentity }));
    assert.equal(evidence.current.outcome, 'failed');
    assert.equal(evidence.current.references[0].data.correctness.evidence, 'current.log');
    assert.deepEqual(evidence.comparisons, {});
  }
});

test('latest comparable success/failure are selected without current or duplicate attempts', () => {
  const history = [task('old-success', 'passed'), task('success', 'passed'), task('failure'),
    task('success', 'passed'), task('current', 'passed')];
  const evidence = selectSkillEvidence(task('current'), history);
  assert.equal(evidence.comparisons.success.attemptId, 'success');
  assert.equal(evidence.comparisons.failure.attemptId, 'failure');
  assert.equal(evidence.comparisons.success.references[0].data.correctness.evidence, 'success.log');
  assert.equal(history.length, 5);
});

test('changed goals, starting source, verifiers and target/runtime never supply comparisons', () => {
  const history = Object.keys(identity).map(key => task(key, 'passed', {
    comparisonIdentity: { ...identity, [key]: 'a'.repeat(64) } }));
  history.push(task('legacy', 'passed', { comparisonIdentity: undefined }),
    task('open', 'passed', { completed: false }), task('unknown', 'unknown'));
  assert.deepEqual(selectSkillEvidence(task('current'), history).comparisons, {});
});

test('production context captures initial inputs, survives edits/resume, and reaches completed tuning task', async t => {
  const root = await fixture(t), source = join(root, 'source'), verifier = join(root, 'verify.py');
  await mkdir(source); await writeFile(join(source, 'kernel.py'), 'answer = 0\n');
  await writeFile(verifier, 'assert answer == 42\n');
  const state = new State(join(root, 'state.sqlite')); t.after(() => state.close());
  state.add({ id: 'p', goal: 'Reduction tails', actor_seconds: 20, source,
    verifier: { command: 'python ../verify.py', files: [verifier] } });
  const project = state.project('p').spec, attempt = state.admit('p', {});
  await prepareSourceGit({source,gitDir:join(root,'source-history'),attemptId:attempt.id});
  const startingSource = await fingerprint(source);
  const frozen = await freezeFiles(project.verifier.files, source);
  const input = { state, attemptId: attempt.id, project, verifierHashes: frozen.manifest,
    environmentIdentity: { target: 'cpu', runtime: 'python-fixture' } };
  const initial = await recordSkillContext(input);
  assert.equal(initial.startingSource, startingSource);
  assert.ok(Object.values(initial).every(value => /^[a-f0-9]{64}$/.test(value)));
  await writeFile(join(source, 'kernel.py'), 'answer = 42\n');
  assert.deepEqual(await recordSkillContext(input), initial);
  state.event(attempt.id, 'verification', { passed: false, evidence: 'retained.log' });
  state.finish(attempt.id, 'failed');
  const completed = tuningTask({ state, attemptId: attempt.id, kind: 'original', phase: 'phase2' });
  assert.equal(completed.completed, true); assert.equal(completed.outcome, 'failed');
  assert.deepEqual(completed.comparisonIdentity, initial);
  assert.deepEqual(selectSkillEvidence(completed).current.references,
    [{ type: 'verification', data: { passed: false, evidence: 'retained.log' } }]);
  state.add({ ...project, id: 'legacy' });
  const legacy = state.admit('legacy', {});
  state.checkpoint(legacy.id, 1, 'historical-session');
  assert.equal(await recordSkillContext({ ...input, attemptId: legacy.id }), undefined);
  assert.equal(state.events(legacy.id).length, 0);
});

test('bounded tuning evidence keeps initial identity and failure records without loading telemetry', async t => {
  const root = await fixture(t), state = new State(join(root, 'state.sqlite'));
  t.after(() => state.close());
  state.add({id:'p',goal:'bounded evidence',actor_seconds:20});
  const attempt = state.admit('p', {});
  state.event(attempt.id, 'skill-evidence-context', identity);
  for (let i = 0; i < 1100; i++) {
    state.event(attempt.id, 'pi-event', {payload:'telemetry', ordinal:i});
    state.event(attempt.id, 'stage-completed', {ordinal:i});
  }
  state.recordFailure(attempt.id, {phase:'implementation',kind:'compile',message:'unsupported operation',
    evidence:[{path:'compile.log',sha256:'a'.repeat(64)}]});
  state.finish(attempt.id, 'failed');
  // Evidence selection must use bounded indexed queries, not legacy replay reads.
  state.events = () => assert.fail('unbounded replay read');
  const current = tuningTask({state,attemptId:attempt.id,kind:'original',phase:'phase2'});
  assert.equal(current.events.length, 1000);
  assert.deepEqual(current.comparisonIdentity, identity);
  assert.ok(current.events.every(event => event.type !== 'pi-event' &&
    Object.keys(event).length === 2 && 'data' in event));
  const references = selectSkillEvidence(current).current.references;
  assert.deepEqual(references.at(-1).data.evidence, [{path:'compile.log',sha256:'a'.repeat(64)}]);
  assert.equal(references.at(-1).type, 'failure-recorded');
  const persisted = state.queryEvents(attempt.id,{types:['skill-evidence-context'],limit:1})[0];
  assert.ok(Number.isInteger(persisted.sequence) && persisted.created);
});

test('runtime binding ignores transient health but distinguishes named runtimes and unknown bindings', () => {
  const config = { environment: { module: 'binding.mjs', options: { mode: 'cpl', target: 'target', runtime: 'one' } } };
  const binding = metadata => environmentBinding(config, {}, { metadata });
  assert.deepEqual(binding({ observedAt: 'yesterday', measurements: { idle: false } }),
    binding({ observedAt: 'today', measurements: { idle: true } }));
  assert.notDeepEqual(binding({ runtime: 'one' }), binding({ runtime: 'two' }));
  assert.equal(environmentBinding({ environment: { options: { mode: 'cpl' } } }, {}), undefined);
});

test('writer receives outcome and historical references; proposals retain them without evaluating', async t => {
  const root = await fixture(t), config = { mode: 'phase2', skills: { activeVersion: 'baseline' } };
  await mkdir(join(root, 'skills', 'versions', 'baseline'), { recursive: true });
  await writeFile(join(root, 'instance.json'), JSON.stringify(config));
  await writeFile(join(root, 'skills', 'versions', 'baseline', 'guidance.json'), '[]');
  await writeFile(join(root, 'skills', 'evolution.json'), JSON.stringify({ history: [task('earlier', 'passed')], lastEvaluated: 0 }));
  let observed;
  const { native } = nativeWriter(value => { observed = value; });
  const result = await tune({ instanceRoot: root, config, task: task('now'), native,
    evaluate: async () => assert.fail('Evidence selection must not execute a benchmark') });
  assert.equal(result.evaluate, false); assert.ok(result.candidate);
  assert.equal(observed.evidence.current.outcome, 'failed');
  assert.equal(observed.evidence.comparisons.success.attemptId, 'earlier');
  const state = await load(join(root, 'skills', 'evolution.json'));
  assert.deepEqual(state.history.at(-1).comparisonIdentity, identity);
  const { readdir } = await import('node:fs/promises');
  const [proposal] = await readdir(join(root, 'skills', 'proposals'));
  assert.deepEqual(await load(join(root, 'skills', 'proposals', proposal, 'comparison-evidence.json')), observed.evidence);
  assert.equal((await load(join(root, 'instance.json'))).skills.activeVersion, 'baseline');
});
