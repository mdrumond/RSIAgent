import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { State } from '../../controller/state.mjs';
import { stages } from '../../controller/workflow.mjs';
import { RecoveryQueue, modePolicy, nextWave, admitJob, recordOutcome, pendingOutcomes, outcomeRecorded } from '../../controller/recovery.mjs';

function projectSpec(t, id, goal) {
  const root = mkdtempSync(join(tmpdir(), 'ascend-recovery-source-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const source = join(root, 'source'), verifier = join(root, 'verify.py');
  mkdirSync(source);
  writeFileSync(join(source, 'answer.py'), 'def answer(): return 42\n');
  writeFileSync(verifier, `import runpy\nassert runpy.run_path(${JSON.stringify(join(source, 'answer.py'))})["answer"]() == 42\n`);
  return { id, goal, source, actor_seconds: 240,
    verifier: { command: `python "${verifier}"`, files: [verifier] } };
}

function fixture(t, options = {}, file = ':memory:') {
  const state = new State(file);
  t.after(() => state.close());
  const queue = new RecoveryQueue(state, options);
  function original(id = 'original') {
    state.add(projectSpec(t, id, 'immutable goal'));
    const attempt = state.admit(id, {});
    state.finish(attempt.id, 'failed');
    return attempt;
  }
  return { state, queue, original };
}
const result = (passed = false, useful = false) => ({ passed, useful,
  findings: 'Observed a reproducible failure in the current source',
  evidence: [{ path: 'failure.json', sha256: 'digest' }] });

function acknowledgeLessons(queue) {
  for (const lesson of queue.lessons()) assert.equal(queue.acknowledgeLesson(lesson.id), true);
}

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
  acknowledgeLessons(queue);
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
  acknowledgeLessons(queue);
  const second = queue.claim('original', 2);
  queue.complete(second.attempt.id, result(true, true), 2);
  acknowledgeLessons(queue);
  const retry = queue.claim('original', 3);
  const done = queue.complete(retry.attempt.id, result(false), 3);
  assert.equal(done.failed_rounds, 2);
  assert.equal(done.frozen, true);
  assert.equal(state.project('original').frozen, true);
  assert.equal(state.project('original').status, 'failed');
  assert.throws(() => state.admit('original', {}));
  assert.deepEqual(queue.ready(4), []);
  assert.equal(queue.lessons().length, 1);
});

test('default fifth unsuccessful round freezes; initial failure is not a round', t => {
  const { queue, original, state } = fixture(t);
  queue.enqueue(original().id, 0);
  for (let wave = 1; wave <= 5; wave++) {
    const job = queue.claim('original', wave);
    const done = queue.complete(job.attempt.id, result(), wave);
    assert.equal(done.failed_rounds, wave);
    assert.equal(done.frozen, wave === 5);
    acknowledgeLessons(queue);
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
  assert.deepEqual(queue.ready(2).map(job => job.project), ['b']);
  const [lesson] = queue.lessons();
  assert.equal(lesson.attempt_id, a.attempt.id);
  assert.deepEqual(lesson.evidence, result().evidence);
  assert.equal(queue.acknowledgeLesson(lesson.id), true);
  assert.equal(queue.acknowledgeLesson(lesson.id), false);
  assert.deepEqual(queue.lessons(), []);
  assert.deepEqual(queue.ready(2).map(job => job.project), ['b', 'a']);
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

test('CLI boundary selects and admits originals and recovery without duplicate bookkeeping', t => {
  const { state, queue } = fixture(t);
  state.add(projectSpec(t, 'new', 'original goal'));
  state.add(projectSpec(t, 'other', 'another goal'));
  const options = { wave: 0, investigationSeconds: 120 };
  const initial = nextWave(state, options);
  assert.deepEqual(initial.map(job => [job.projectId, job.kind]), [['new', 'original'], ['other', 'original']]);
  assert.equal(state.db.prepare('SELECT COUNT(*) AS n FROM attempts').get().n, 0);
  const original = admitJob(state, { ...options, projectId: 'new' });
  assert.equal(original.project.status, 'running');
  assert.equal(original.kind, 'original');
  assert.throws(() => admitJob(state, { ...options, projectId: 'new' }));
  const failed = recordOutcome(state, { ...options, attemptId: original.attempt.id, ...result() });
  assert.equal(failed.failed_rounds, 0);
  assert.equal(failed.next.eligible_wave, 1);
  assert.deepEqual(nextWave(state, options).map(job => job.projectId), ['other']);
  acknowledgeLessons(queue);
  const next = nextWave(state, { ...options, wave: 1 });
  assert.deepEqual(next.map(job => [job.projectId, job.kind]), [['other', 'original'], ['new', 'investigation']]);
  assert.equal(next[1].actor_seconds, 120);
  const diagnostic = admitJob(state, { ...options, wave: 1, projectId: 'new' });
  recordOutcome(state, { ...options, wave: 1, attemptId: diagnostic.attempt.id, ...result(true, true) });
  assert.equal(nextWave(state, { ...options, wave: 1 }).length, 1);
  acknowledgeLessons(queue);
  const retry = admitJob(state, { ...options, wave: 2, projectId: 'new' });
  assert.equal(retry.kind, 'retry');
  state.finish(retry.attempt.id, 'passed');
  recordOutcome(state, { ...options, wave: 2, attemptId: retry.attempt.id, ...result(true) });
  assert.equal(state.project('new').status, 'passed');
  assert.equal(queue.lessons().length, 1);
  assert.equal(queue.completedOriginals(), 1);
});

test('CLI selection never readmits interrupted original; resume retains attempt and remaining allocation', async t => {
  const { state } = fixture(t);
  state.add(projectSpec(t, 'interrupted', 'same source'));
  const job = admitJob(state, { projectId: 'interrupted', wave: 0 });
  state.checkpoint(job.attempt.id, 100, 'same-pi-session');
  assert.deepEqual(nextWave(state, { wave: 10 }), []);
  const resumed = await state.resume(job.attempt.id, async () => true, { resumed: true });
  assert.equal(resumed.id, job.attempt.id);
  assert.equal(resumed.elapsed, 100);
  assert.equal(resumed.session, 'same-pi-session');
  recordOutcome(state, { attemptId: resumed.id, wave: 10, ...result(true) });
  assert.equal(state.project('interrupted').status, 'passed');
  assert.equal(state.db.prepare('SELECT COUNT(*) AS n FROM attempts').get().n, 1);
});

test('CLI outcomes handle inference opt-out and reject duplicate or conflicting originals atomically', t => {
  const { state, queue } = fixture(t);
  state.add(projectSpec(t, 'inference', 'fixed goal'));
  const options = { mode: 'inference', inferenceRecovery: false, wave: 0 };
  const job = admitJob(state, { ...options, projectId: 'inference' });
  assert.throws(() => recordOutcome(state, { ...options, wave: -1, attemptId: job.attempt.id, ...result() }));
  assert.equal(state.attempt(job.attempt.id).outcome, null);
  state.finish(job.attempt.id, 'failed');
  assert.throws(() => recordOutcome(state, { ...options, attemptId: job.attempt.id, ...result(true) }));
  const done = recordOutcome(state, { ...options, attemptId: job.attempt.id, ...result() });
  assert.equal(done.next, null);
  assert.deepEqual(nextWave(state, { ...options, wave: 1 }), []);
  assert.throws(() => recordOutcome(state, { ...options, attemptId: job.attempt.id, ...result() }));
  assert.equal(queue.lessons().length, 1);
});

test('resource collision rolls back original and recovery admission without orphan attempts', t => {
  const { state, queue, original } = fixture(t);
  state.add(projectSpec(t, 'holder', 'occupy opaque resource'));
  state.add(projectSpec(t, 'queued', 'wait without admission'));
  const holder = admitJob(state, { projectId: 'holder', wave: 0, resourceId: 'opaque-slot' });
  queue.enqueue(original('recover').id, 0);
  const attempts = state.db.prepare('SELECT COUNT(*) AS n FROM attempts').get().n;
  for (const projectId of ['queued', 'recover']) {
    assert.throws(() => admitJob(state, { projectId, wave: 1, resourceId: 'opaque-slot' }));
    assert.equal(state.db.prepare('SELECT COUNT(*) AS n FROM attempts').get().n, attempts);
    assert.equal(state.db.prepare("SELECT COUNT(*) AS n FROM events WHERE type='job-admitted'").get().n, 1);
  }
  assert.equal(state.project('queued').status, 'queued');
  assert.equal(state.project('recover').status, 'failed');
  assert.equal(queue.entry('recover').attempt, null);
  assert.equal(state.db.prepare('SELECT COUNT(*) AS n FROM recovery_attempts').get().n, 0);
  assert.equal(state.db.prepare('SELECT attempt FROM resource_leases').get().attempt, holder.attempt.id);
  state.db.prepare('DELETE FROM resource_leases WHERE attempt=?').run(holder.attempt.id);
  const claimed = queue.claim('recover', 1, {}, 'opaque-slot');
  assert.equal(claimed.kind, 'investigation');
  assert.equal(state.db.prepare('SELECT attempt FROM resource_leases').get().attempt, claimed.attempt.id);
});

test('restart finds settled outcomes for postprocessing only, separate from memory acknowledgement', t => {
  const directory = mkdtempSync(join(tmpdir(), 'ascend-outcomes-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const { state } = fixture(t, {}, join(directory, 'state.sqlite'));
  state.add(projectSpec(t, 'failed', 'original remains immutable'));
  state.add(projectSpec(t, 'open', 'interrupted actor'));
  const failed = admitJob(state, { projectId: 'failed', wave: 2 });
  admitJob(state, { projectId: 'open', wave: 2 });
  state.checkpoint(failed.attempt.id, 77, 'original-native-session');
  state.finish(failed.attempt.id, 'failed'); // crash before summary / recordOutcome
  const reopened = fixture(t, {}, join(directory, 'state.sqlite'));
  const [pending] = pendingOutcomes(reopened.state);
  assert.equal(pending.id, failed.attempt.id);
  assert.equal(pending.kind, 'original');assert.equal(pending.wave, 2);
  assert.equal(pending.elapsed, 77);assert.equal(pending.session, 'original-native-session');
  assert.equal(outcomeRecorded(reopened.state, failed.attempt.id), false);
  assert.deepEqual(nextWave(reopened.state, { wave: 3 }), []);
  recordOutcome(reopened.state, { attemptId: pending.id, wave: pending.wave, ...result() });
  assert.equal(reopened.queue.outcomeRecorded(pending.id), true);
  assert.equal(reopened.queue.lessons().length, 1); // KDB delivery still pending
  assert.deepEqual(reopened.queue.pendingOutcomes(), []);
  assert.deepEqual(nextWave(state, { wave: 3 }), []);
  acknowledgeLessons(reopened.queue);
  assert.equal(nextWave(state, { wave: 3 })[0].kind, 'investigation');
  const investigation = admitJob(state, { projectId: 'failed', wave: 3 });
  state.finish(investigation.attempt.id, 'passed'); // second crash after actor closure
  const [diagnostic] = reopened.queue.pendingOutcomes();
  assert.equal(diagnostic.id, investigation.attempt.id);assert.equal(diagnostic.kind, 'investigation');
  assert.equal(diagnostic.wave, 3);
  recordOutcome(reopened.state, { attemptId: diagnostic.id, wave: diagnostic.wave, ...result(true, true) });
  assert.deepEqual(pendingOutcomes(state), []);
  acknowledgeLessons(reopened.queue);
  assert.equal(nextWave(state, { wave: 4 })[0].kind, 'retry');
  assert.equal(state.project('failed').status, 'failed');
  assert.equal(state.db.prepare('SELECT COUNT(*) AS n FROM attempts').get().n, 3);
  assert.equal(state.attempt(failed.attempt.id).session, 'original-native-session');
});

test('unacknowledged findings block only their recovery project until actual memory acknowledgement', t => {
  const { state, queue } = fixture(t);
  for (const id of ['recover', 'unrelated']) state.add(projectSpec(t, id, id));
  const original = admitJob(state, { projectId: 'recover', wave: 0 });
  recordOutcome(state, { attemptId: original.attempt.id, wave: 0, ...result() });
  const [failure] = queue.lessons(); // memory sink failed: no acknowledgement
  assert.equal(queue.acknowledgeLesson(-1), false);
  assert.deepEqual(nextWave(state, { wave: 1 }).map(job => job.projectId), ['unrelated']);
  assert.throws(() => queue.claim('recover', 1));
  assert.equal(queue.acknowledgeLesson(failure.id), true);
  assert.deepEqual(queue.ready(0), []); // acknowledgement does not bypass next-wave boundary
  const investigation = queue.claim('recover', 1);
  queue.complete(investigation.attempt.id, result(true, true), 1);
  assert.deepEqual(queue.ready(2), []);
  assert.throws(() => admitJob(state, { projectId: 'recover', wave: 2 }));
  assert.deepEqual(nextWave(state, { wave: 2 }).map(job => job.projectId), ['unrelated']);
  const [finding] = queue.lessons();
  assert.equal(queue.acknowledgeLesson(finding.id), true);
  assert.deepEqual(queue.ready(1), []);
  assert.equal(queue.ready(2)[0].kind, 'retry');
  assert.equal(admitJob(state, { projectId: 'recover', wave: 2 }).kind, 'retry');
});
