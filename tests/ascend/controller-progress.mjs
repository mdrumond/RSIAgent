import {test} from 'node:test';
import assert from 'node:assert/strict';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {campaignProgress} from '../../controller/progress.mjs';

function fixture(t) {
  const state = new State(':memory:');
  t.after(() => state.close());
  const queue = new RecoveryQueue(state);
  function start(id, phase = 'phase1', mode) {
    state.add({id, goal: 'test', actor_seconds: 10, ...(phase ? {phase} : {})});
    const job = queue.claim(id, 0);
    if (mode !== undefined) state.db.prepare("UPDATE events SET data=json_set(data,'$.mode',?) WHERE attempt=? AND type='job-admitted'")
      .run(mode, job.attempt.id);
    else state.db.prepare("UPDATE events SET data=json_remove(data,'$.mode') WHERE attempt=? AND type='job-admitted'")
      .run(job.attempt.id); // Historical admissions exercise spec/list fallback.
    return job.attempt.id;
  }
  function finish(id, passed = true, ack = true, post = true) {
    queue.recordOriginal(id, {passed, findings: 'observed behavior', evidence: [{collection: 'memory', entry_id: id}]}, 0);
    if (ack) queue.acknowledgeLesson(queue.lessons().find(lesson => lesson.attempt_id === id).id);
    if (post) state.event(id, 'postprocess-completed', {});
  }
  return {state, queue, start, finish};
}

test('completion requires terminal outcome, memory acknowledgement, postprocessing and latest settlement', t => {
  const {state, queue, start, finish} = fixture(t);
  const id = start('one');
  const config = {mode: 'phase1', completion_target: 1, phase_sequence: ['phase1', 'phase2']};
  assert.equal(campaignProgress(state, config).finishedOriginals, 0);
  finish(id, false, false, false);
  assert.equal(campaignProgress(state, config).phaseComplete, false);
  queue.acknowledgeLesson(queue.lessons()[0].id);
  assert.equal(campaignProgress(state, config).phaseComplete, false);
  state.event(id, 'postprocess-completed', {});
  state.event(id, 'failure-settlement', {closed: false});
  assert.equal(campaignProgress(state, config).finishedOriginals, 0);
  state.event(id, 'failure-settlement', {closed: true});
  const result = campaignProgress(state, config);
  assert.equal(result.finishedOriginals, 1);
  assert.equal(result.nextMode, 'phase2');
  assert.equal(result.done, false);
  assert.ok(queue.entry('one'), 'pending recovery does not block advance');
  const investigation = queue.claim('one', 1);
  queue.complete(investigation.attempt.id, {passed: true, useful: true, findings: 'retry supported',
    evidence: [{collection: 'memory', entry_id: 'investigation'}]}, 1);
  queue.acknowledgeLesson(queue.lessons()[0].id);
  state.event(investigation.attempt.id, 'postprocess-completed', {});
  assert.equal(campaignProgress(state, config).finishedOriginals, 1);
  const retry = queue.claim('one', 2);
  queue.complete(retry.attempt.id, {passed: true, findings: 'passed', evidence: [{collection: 'memory', entry_id: 'retry'}]}, 2);
  queue.acknowledgeLesson(queue.lessons()[0].id);
  state.event(retry.attempt.id, 'postprocess-completed', {});
  assert.equal(campaignProgress(state, config).totalOriginals, 1);
});

test('phase scope uses admission then spec then unambiguous configured IDs; unknown stays unknown', t => {
  const {state, start, finish} = fixture(t);
  finish(start('admission', 'phase1', 'phase2'));
  finish(start('spec', 'phase1'));
  finish(start('fallback', null));
  finish(start('ambiguous', null));
  finish(start('unknown', null));
  finish(start('invalid', 'bogus'));
  const result = campaignProgress(state, {mode: 'phase1', curriculum: {
    phase1: ['fallback', 'ambiguous', 'spec'], phase2: ['ambiguous', 'spec']}});
  assert.deepEqual(result.phaseCounts, {
    phase1: {finishedOriginals: 2, totalOriginals: 2},
    phase2: {finishedOriginals: 1, totalOriginals: 1},
    inference: {finishedOriginals: 0, totalOriginals: 0}});
  assert.equal(result.phaseComplete, false);
});

test('explicit targets override list completion; generated unbounded phases keep running', t => {
  const {state, start, finish} = fixture(t);
  finish(start('one'));
  const config = {mode: 'phase1', curriculum: {phase1: ['one', 'one']}};
  assert.equal(campaignProgress(state, config).done, true);
  assert.equal(campaignProgress(state, {...config, completion_target: 2}).phaseComplete, false);
  assert.equal(campaignProgress(state, {...config, completion_target: 2, phase_targets: {phase1: 1}}).done, true);
  const generated = {...config, curriculum: {...config.curriculum, generation: {enabled: true}}};
  assert.equal(campaignProgress(state, generated).phaseComplete, false);
  assert.equal(campaignProgress(state, {...generated, completion_target: 1}).done, true);
  assert.equal(campaignProgress(state, {mode: 'phase1'}).target, null);
  assert.equal(campaignProgress(state, {mode: 'phase1'}).phaseComplete, false);
});

test('malformed settlement and admission cannot create completed originals', t => {
  const {state, start, finish} = fixture(t);
  for (const [index, data] of ['{', '{}', '{"closed":null}', '{"closed":0}'].entries()) {
    const id = start(`bad-${index}`);
    finish(id);
    state.event(id, 'failure-settlement', {});
    state.db.prepare("UPDATE events SET data=? WHERE attempt=? AND type='failure-settlement'").run(data, id);
  }
  const malformed = start('bad-admission'); finish(malformed);
  state.db.prepare("UPDATE events SET data='{' WHERE attempt=? AND type='job-admitted'").run(malformed);
  assert.equal(campaignProgress(state, {mode: 'phase1'}).finishedOriginals, 0);
});

test('duplicate original attempts count once and retain original admission phase', t => {
  const {state, start, finish} = fixture(t);
  finish(start('one'));
  state.db.prepare("UPDATE projects SET status='queued' WHERE id='one'").run();
  const duplicate = state.admit('one', {});
  state.event(duplicate.id, 'job-admitted', {kind: 'original', mode: 'phase2'});
  finish(duplicate.id);
  state.events = () => {throw new Error('unbounded history API');};
  const result = campaignProgress(state, {mode: 'phase1'});
  assert.equal(result.finishedOriginals, 1);
  assert.equal(result.phaseCounts.phase2.totalOriginals, 0);
});

test('validates configuration without imposing a built-in target', t => {
  const {state} = fixture(t);
  for (const config of [
    {phase_sequence: []}, {phase_sequence: ['phase1', 'phase1']}, {phase_sequence: ['bad']},
    {phase_sequence: ['phase2']}, {completion_target: 0}, {completion_target: 1.5},
    {phase_targets: {phase1: -1}}, {phase_targets: {bad: 1}},
  ]) assert.throws(() => campaignProgress(state, {mode: 'phase1', ...config}));
  assert.equal(campaignProgress(state, {mode: 'phase1', completion_target: 128}).target, 128);
});

test('fresh State is readable before RecoveryQueue setup', t => {
  const state = new State(':memory:'); t.after(() => state.close());
  state.add({id: 'one', phase: 'phase1', goal: 'test', actor_seconds: 1});
  const attempt = state.admit('one', {});
  state.finish(attempt.id, 'passed');
  state.event(attempt.id, 'postprocess-completed', {});
  assert.equal(campaignProgress(state, {mode: 'phase1'}).finishedOriginals, 0);
});

test('historical event noise and later settlement reopening preserve completion gates', t => {
  const {state, start, finish} = fixture(t);
  const id = start('one'); finish(id);
  for (let index = 0; index < 250; index++) state.event(id, 'actor-message', {index});
  state.event(id, 'failure-settlement', {closed: false});
  state.events = () => {throw new Error('history must remain in SQL');};
  state.queryEvents = () => {throw new Error('history must remain in SQL');};
  assert.equal(campaignProgress(state, {mode: 'phase1', completion_target: 1}).phaseComplete, false);
  state.event(id, 'failure-settlement', {closed: true});
  assert.equal(campaignProgress(state, {mode: 'phase1', completion_target: 1}).done, true);
});
