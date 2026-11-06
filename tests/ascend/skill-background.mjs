import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { enqueueSkillEvidence, backgroundSkillStatus, runBackgroundSkills, backgroundSkillsSettled } from '../../cli/skill-background.mjs';

const load = async path => JSON.parse(await readFile(path, 'utf8'));
const save = (path, data) => writeFile(path, JSON.stringify(data));
const task = n => ({ id: `t${n}`, projectId: `p${n}`, phase: 'phase2', kind: 'original', completed: true, outcome: 'failed', events: [] });
const resource = { id: 'slot-0' };
async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'skill-background-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'skills/versions/baseline'), { recursive: true });
  await save(join(root, 'skills/versions/baseline/guidance.json'), []);
  const config = { format: 1, root, mode: 'phase1', tune_every: 1,
    skills: { activeVersion: 'baseline', tuningProjectIds: ['fa', 'gdn'], runsPerArm: 2 } };
  await save(join(root, 'instance.json'), config);
  const requests = [];
  const runtime = { bootstrap: async request => {
    requests.push(structuredClone({ root: request.instanceRoot, version: request.skillVersion, config: request.config }));
    await mkdir(request.instanceRoot, { recursive: true });
  }, wave: async request => ({ correct: true, score: request.config.skills.activeVersion === 'baseline' ? 1 : 2 }),
  settled: async () => true };
  runtime.resume = async request => runtime.wave(request);
  return { root, config, requests, runtime,
    run: values => runBackgroundSkills(config, { resource, remainingSeconds: 20, runtime, generate: async () => [], ...values }),
    evolution: () => load(join(root, 'skills/evolution.json')) };
}

test('foreground ingestion is idempotent and does not wait for the model lock', async t => {
  const f = await fixture(t);
  assert.equal(enqueueSkillEvidence(f.config, { ...task(0), phase: 'phase1' }), false);
  assert.equal(enqueueSkillEvidence(f.config, task(0)), true);
  assert.equal(enqueueSkillEvidence(f.config, task(0)), false);
  assert.equal((await backgroundSkillStatus(f.config)).queued, 1);
  await assert.rejects(f.run({ resource: undefined }), /reserved resource/);
  assert.equal((await backgroundSkillStatus(f.config)).hasWork, true);
});

test('one frozen comparison coalesces evidence while new work remains queued', async t => {
  const f = await fixture(t); let release, entered, writers = 0, samples = 0;
  const blocked = new Promise(resolve => { release = resolve; });
  const started = new Promise(resolve => { entered = resolve; });
  enqueueSkillEvidence(f.config, task(0)); enqueueSkillEvidence(f.config, task(1));
  const wave = f.runtime.wave;
  f.runtime.wave = async request => { if (++samples === 1) { entered(); await blocked; } return wave(request); };
  const running = f.run({ generate: async options => { writers++; assert.equal(options.input.history.length, 2); return []; } });
  await started;
  assert.ok((await backgroundSkillStatus(f.config)).owner?.pid);
  assert.equal(enqueueSkillEvidence(f.config, task(2)), true);
  release(); const result = await running;
  assert.equal(result.promote, true); assert.equal(writers, 1); assert.equal(samples, 8);
  const status = await backgroundSkillStatus(f.config);
  assert.equal(status.queued, 1); assert.equal(status.pending, false); assert.equal(status.hasWork, true);
  const evolution = await f.evolution();
  assert.deepEqual(evolution.history.map(item => item.id), ['t0', 't1']);
  assert.equal(new Set(f.requests.map(item => item.root)).size, 8);
  assert.ok(f.requests.every(item => item.config.resource_ids[0] === resource.id));
  assert.equal((await load(join(f.root, 'instance.json'))).skills.activeVersion, result.candidate);
  await f.run({ generate: async options => { writers++; assert.equal(options.input.history.length, 3); return []; } });
  assert.equal(writers, 2);
  assert.equal((await f.run({ generate: () => assert.fail('unchanged evidence writer') })).evaluationStatus, 'idle');
});

test('interruption resumes the same samples during phase1 without refreshing allocation', async t => {
  const f = await fixture(t), abort = new AbortController(); let milliseconds = 0, calls = 0;
  enqueueSkillEvidence(f.config, task(0));
  const wave = f.runtime.wave, roots = [];
  f.runtime.wave = async request => {
    roots.push(request.instanceRoot); milliseconds += 100;
    if (++calls === 2) { abort.abort(); throw abort.signal.reason; }
    return wave(request);
  };
  const clock = { now: () => milliseconds, wallNow: () => milliseconds };
  await assert.rejects(f.run({ signal: abort.signal, clock, remainingSeconds: 10 }), { name: 'AbortError' });
  const pending = (await f.evolution()).pending;
  assert.equal((await backgroundSkillStatus(f.config)).unresolved, false);
  assert.equal(await backgroundSkillsSettled(f.config), true);
  milliseconds += 100000;
  const result = await f.run({ clock, remainingSeconds: 1000, generate: () => assert.fail('resume writer') });
  assert.equal(result.promote, true); assert.equal(calls, 9);
  assert.equal(roots[1], roots[2]);
  const round = await load(join(pending.evaluationRoot, 'round.json'));
  assert.equal(round.allocatedSeconds, 10); assert.ok(round.elapsedSeconds < 2);
  assert.equal(round.samples.length, 8);
});

test('unknown execution keeps positive settlement proof and retained resource mandatory', async t => {
  const f = await fixture(t); enqueueSkillEvidence(f.config, task(0));
  f.runtime.settled = async () => false;
  await assert.rejects(f.run(), /execution remains unresolved/);
  assert.equal((await backgroundSkillStatus(f.config)).unresolved, true);
  assert.equal(await backgroundSkillsSettled(f.config, { runtime: f.runtime }), false);
  const retained = (await f.evolution()).pending.evaluationRoot;
  await assert.rejects(f.run(), /execution remains unresolved/);
  assert.equal((await f.evolution()).pending.evaluationRoot, retained);
  await assert.rejects(f.run({ resource: { id: 'other-slot' } }), /retained resource/);
  f.runtime.settled = async () => true;
  assert.equal(await backgroundSkillsSettled(f.config, { runtime: f.runtime }), true);
  assert.equal((await f.run()).promote, true);
});

test('concurrent promoted-version changes cannot be overwritten by background results', async t => {
  const f = await fixture(t); enqueueSkillEvidence(f.config, task(0)); let calls = 0;
  const wave = f.runtime.wave;
  f.runtime.wave = async request => {
    if (++calls === 8) {
      const config = await load(join(f.root, 'instance.json')); config.skills.activeVersion = 'manual-version';
      await save(join(f.root, 'instance.json'), config);
    }
    return wave(request);
  };
  const result = await f.run();
  assert.equal(result.promote, false); assert.equal(result.reason, 'superseded');
  assert.equal((await load(join(f.root, 'instance.json'))).skills.activeVersion, 'manual-version');
  assert.equal((await load(join(result.evaluationRoot, 'decision.json'))).reason, 'superseded');
});


test('a month-long campaign allocation does not overflow the nested execution deadline', async t => {
  const f = await fixture(t); enqueueSkillEvidence(f.config, task(0));
  const wave = f.runtime.wave;
  f.runtime.wave = async request => {
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(request.signal.aborted, false);
    return wave(request);
  };
  assert.equal((await f.run({ remainingSeconds: 30 * 24 * 60 * 60 })).promote, true);
});
