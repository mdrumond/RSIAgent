import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { State } from '../../controller/state.mjs';
import { stages } from '../../controller/workflow.mjs';
import { RecoveryQueue, modePolicy } from '../../controller/recovery.mjs';

function fixture(t, options = {}, file = ':memory:') {
  const state = new State(file);
  t.after(() => state.close());
  const queue = new RecoveryQueue(state, options);
  function original(id = 'original') {
    state.add({ id, goal: 'immutable goal', verifier: { command: 'verify frozen source' },
      source: 'latest-broken-source', actor_seconds: 240 });
    const attempt = state.admit(id, {});
    state.finish(attempt.id, 'failed');
    return attempt;
  }
  return { state, queue, original };
}
const result = (passed = false, useful = false) => ({ passed, useful,
  findings: 'Observed a reproducible failure in the current source',
  evidence: [{ path: 'failure.json', sha256: 'digest' }] });

test('every mode uses canonical workflow; inference recovery can be disabled', t => {
  for (const mode of ['phase1', 'phase2', 'inference']) {
    const policy = modePolicy(mode);
    assert.equal(policy.stages, stages);
    assert.equal(policy.memory, true);
    assert.equal(policy.skills, mode !== 'phase1');
  }
  const { queue, original, state } = fixture(t, { mode: 'inference', inferenceRecovery: false });
  assert.equal(queue.enqueue(original().id, 0), null);
  assert.deepEqual(queue.ready(1), []);
  assert.equal(state.project('original').status, 'failed');
  assert.throws(() => modePolicy('legacy'));
});

test('investigation and original retry use separate future waves and fresh budgets', t => {
  const { queue, original, state } = fixture(t);
  const attempt = original();
  const spec = state.project('original').spec;
  queue.enqueue(attempt.id, 3);
  assert.equal(queue.entry('original').failed_rounds, 0);
  assert.deepEqual(queue.ready(3), []);
  assert.throws(() => queue.claim('original', 3));
  const investigation = queue.claim('original', 4, { worker: 'diagnostic' });
  assert.equal(investigation.kind, 'investigation');
  assert.equal(investigation.actor_seconds, 9600);
  assert.equal(investigation.attempt.elapsed, 0);
  assert.throws(() => queue.claim('original', 4));
  state.checkpoint(investigation.attempt.id, 91, 'pi-investigation');
  state.finish(investigation.attempt.id, 'passed');
  const completed = queue.complete(investigation.attempt.id, result(true, true), 4);
  assert.equal(completed.failed_rounds, 0);
  assert.equal(state.project('original').status, 'failed');
  assert.deepEqual(queue.ready(4), []);
  const retry = queue.claim('original', 5);
  assert.equal(retry.kind, 'retry');
  assert.equal(retry.actor_seconds, 240);
  assert.equal(retry.attempt.elapsed, 0);
  assert.equal(retry.attempt.session, null);
  assert.notEqual(retry.attempt.id, investigation.attempt.id);
  assert.equal(retry.origin_attempt, attempt.id);
  assert.equal(queue.complete(retry.attempt.id, result(true), 5).accepted, true);
  assert.equal(state.project('original').status, 'passed');
  assert.deepEqual(state.project('original').spec, spec);
  assert.deepEqual(queue.ready(6), []);
  assert.equal(queue.completedOriginals(), 1);
});

test('unusable investigations and failed original retries consume one round each', t => {
  const { queue, original, state } = fixture(t, { maxFailedRounds: 2 });
  queue.enqueue(original().id, 0);
  const first = queue.claim('original', 1);
  assert.equal(queue.complete(first.attempt.id, result(true, false), 1).failed_rounds, 1);
  const second = queue.claim('original', 2);
  queue.complete(second.attempt.id, result(true, true), 2);
  const retry = queue.claim('original', 3);
  const done = queue.complete(retry.attempt.id, result(false), 3);
  assert.equal(done.failed_rounds, 2);
  assert.equal(done.frozen, true);
  assert.equal(state.project('original').frozen, true);
  assert.equal(state.project('original').status, 'failed');
  assert.throws(() => state.admit('original', {}));
  assert.deepEqual(queue.ready(4), []);
  assert.equal(queue.lessons().length, 3);
});

test('default fifth unsuccessful round freezes; initial failure is not a round', t => {
  const { queue, original, state } = fixture(t);
  queue.enqueue(original().id, 0);
  for (let wave = 1; wave <= 5; wave++) {
    const job = queue.claim('original', wave);
    const done = queue.complete(job.attempt.id, result(), wave);
    assert.equal(done.failed_rounds, wave);
    assert.equal(done.frozen, wave === 5);
  }
  assert.equal(state.project('original').frozen, true);
});

test('failed investigation leaves unrelated wave work eligible and persists findings', t => {
  const { queue, original } = fixture(t);
  queue.enqueue(original('a').id, 0);
  queue.enqueue(original('b').id, 0);
  const a = queue.claim('a', 1);
  queue.complete(a.attempt.id, result(false), 1);
  assert.deepEqual(queue.ready(1).map(job => job.project), ['b']);
  assert.deepEqual(queue.ready(2).map(job => job.project), ['b', 'a']);
  const [lesson] = queue.lessons();
  assert.equal(lesson.attempt_id, a.attempt.id);
  assert.deepEqual(lesson.evidence, result().evidence);
  assert.equal(queue.acknowledgeLesson(lesson.id), true);
  assert.equal(queue.acknowledgeLesson(lesson.id), false);
  assert.deepEqual(queue.lessons(), []);
});

test('useful evidence from an unsuccessful investigation still schedules original retry', t => {
  const { queue, original } = fixture(t);
  queue.enqueue(original().id, 0);
  const job = queue.claim('original', 1);
  const done = queue.complete(job.attempt.id, result(false, true), 1);
  assert.equal(done.next.kind, 'retry');
  assert.equal(done.failed_rounds, 0);
});

test('queue and memory outbox survive reopening; interrupted admission is not duplicated', t => {
  const directory = mkdtempSync(join(tmpdir(), 'ascend-recovery-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const file = join(directory, 'state.sqlite');
  const first = fixture(t, {}, file);
  first.queue.enqueue(first.original().id, 0);
  const investigation = first.queue.claim('original', 1);
  const reopened = fixture(t, {}, file);
  assert.deepEqual(reopened.queue.ready(2), []);
  assert.throws(() => reopened.queue.claim('original', 2));
  reopened.queue.complete(investigation.attempt.id, result(), 2);
  assert.equal(first.queue.lessons().length, 1);
  assert.equal(first.queue.entry('original').eligible_wave, 3);
});

test('rejected completion rolls back and duplicate completion cannot consume rounds twice', t => {
  const { queue, original, state } = fixture(t);
  const origin = original();
  assert.throws(() => queue.enqueue(origin.id, -1));
  queue.enqueue(origin.id, 0);
  const job = queue.claim('original', 1);
  assert.throws(() => queue.complete(job.attempt.id, result(), 0));
  assert.throws(() => queue.complete(job.attempt.id,
    { ...result(), evidence: [{ raw_refs: { archive: 'large' } }] }, 1));
  assert.equal(state.project('original').status, 'running');
  assert.deepEqual(queue.lessons(), []);
  queue.complete(job.attempt.id, result(), 1);
  assert.throws(() => queue.complete(job.attempt.id, result(), 1));
  assert.equal(queue.entry('original').failed_rounds, 1);
  assert.equal(queue.lessons().length, 1);
});

test('distinct-original accounting excludes investigations', t => {
  const { queue, original, state } = fixture(t);
  const a = original('a');
  queue.enqueue(a.id, 0);
  const investigation = queue.claim('a', 1);
  queue.complete(investigation.attempt.id, result(), 1);
  original('b');
  assert.equal(queue.completedOriginals(), 2);
  assert.throws(() => new RecoveryQueue(state, { maxFailedRounds: 0 }));
});

test('event failure rolls back queue insertion atomically', t => {
  const { state, original } = fixture(t);
  const attempt = original();
  const queue = new RecoveryQueue({ db: state.db, project: state.project.bind(state),
    event() { throw new Error('event unavailable'); } });
  assert.throws(() => queue.enqueue(attempt.id, 0));
  assert.equal(queue.entry('original'), undefined);
  assert.equal(state.project('original').status, 'failed');
});
