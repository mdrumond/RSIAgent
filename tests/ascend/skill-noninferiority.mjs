import test from 'node:test';
import assert from 'node:assert/strict';
import {evaluateNonInferiority} from '../../controller/skill-noninferiority.mjs';

const projectIds = ['layout', 'softmax'];
function observations(ratios = [1, 1, 1, 1]) {
  return projectIds.flatMap((projectId, task) => [0, 1].flatMap(run =>
    ['active', 'candidate'].map(arm => ({projectId, run, arm, correct: true,
      elapsedSeconds: (task ? 1000 : 10) * (arm === 'candidate' ? ratios[task * 2 + run] : 1),
      tokens: 200 * (arm === 'candidate' ? ratios[task * 2 + run] : 1)}))));
}
const decide = (results, options = {}) => evaluateNonInferiority(results, {projectIds, ...options});

test('equal correct results are eligible without winning every pair', () => {
  const result = decide(observations());
  assert.equal(result.promote, true);
  assert.equal(result.reason, 'observed-non-inferiority');
  assert.equal(result.evidence, 'observed');
  assert.equal(result.metrics.tokens.ratio, 1);
});

test('one slower repetition can be offset without allowing aggregate cost regression', () => {
  const result = decide(observations([2, 0.5, 1, 1]), {margin: 0});
  assert.equal(result.promote, true);
  assert.equal(result.metrics.elapsedSeconds.withinMargin, true);
  assert.ok(Math.abs(result.metrics.elapsedSeconds.tasks[0].pairs[0].ratio - 2) < 1e-12);
  assert.ok(Math.abs(result.metrics.elapsedSeconds.tasks[0].pairs[1].ratio - 0.5) < 1e-12);
  const worse = decide(observations([1.2, 1.2, 1, 1]));
  assert.equal(worse.promote, false);
  assert.equal(worse.reason, 'aggregate-cost-regression');
});

test('task weighting uses paired ratios rather than allowing long tasks to dominate', () => {
  const result = decide(observations([2, 2, 0.5, 0.5]), {margin: 0});
  assert.equal(result.promote, true);
  assert.ok(Math.abs(result.metrics.elapsedSeconds.ratio - 1) < 1e-12);
  assert.equal(result.metrics.elapsedSeconds.tasks.length, 2);
});

test('time improvement cannot hide a token regression and metrics are explicitly selectable', () => {
  const results = observations([0.5, 0.5, 0.5, 0.5]);
  for (const row of results) if (row.arm === 'candidate') row.tokens = 400;
  assert.equal(decide(results).promote, false);
  assert.equal(decide(results).metrics.tokens.withinMargin, false);
  assert.equal(decide(results, {metrics: ['elapsedSeconds']}).promote, true);
});

test('margin bounds aggregate cost and supports a strict zero allowance', () => {
  const results = observations([1.04, 1.04, 1.04, 1.04]);
  assert.equal(decide(results).promote, true);
  assert.equal(decide(results, {margin: 0}).promote, false);
  assert.equal(decide(observations([1.05, 1.05, 1.05, 1.05])).promote, true);
});

test('every candidate correctness failure blocks eligibility despite cheaper execution', () => {
  const results = observations([0.1, 0.1, 0.1, 0.1]);
  results.find(row => row.arm === 'candidate').correct = false;
  assert.equal(decide(results).reason, 'correctness-regression');
  for (const row of results) if (row.arm === 'active') row.correct = false;
  assert.equal(decide(results).promote, false);
});

test('a correctness improvement is reported but does not bypass resource accounting', () => {
  const results = observations();
  results[0].correct = false;
  assert.equal(decide(results).correctnessImproved, true);
  assert.equal(decide(results).promote, true);
  results[1].tokens = null;
  assert.equal(decide(results).promote, false);
});

test('incomplete, duplicated, unknown and unbalanced results cannot be accepted', () => {
  const results = observations();
  for (const changed of [results.slice(1), [...results, results[0]],
    [results[0], ...results.slice(0, -1)], results.map((row, i) => i ? row : {...row, projectId: 'unknown'}),
    results.map((row, i) => i ? row : {...row, correct: null}),
    results.map((row, i) => i ? row : {...row, run: 2})]) {
    assert.equal(decide(changed).promote, false);
    assert.equal(decide(changed).reason, 'inconclusive');
  }
});

test('missing, invalid or zero metric telemetry is inconclusive rather than a synthetic win', () => {
  for (const value of [null, undefined, NaN, Infinity, -1, 0]) {
    const results = observations();results[1].tokens = value;
    const result = decide(results);
    assert.equal(result.promote, false);
    assert.equal(result.reason, 'missing-cost-telemetry');
    assert.equal(result.missingMetric, 'tokens');
  }
});

test('two repetitions are required and malformed policy options fail explicitly', () => {
  assert.equal(decide(observations().filter(row => row.run === 0), {runs: 1}).reason, 'insufficient-repetitions');
  for (const options of [{projectIds: []}, {projectIds: ['same', 'same']}, {projectIds: [null]},
    {runs: 0}, {runs: 1.5}, {margin: -0.1}, {margin: Infinity}, {margin: 2},
    {metrics: []}, {metrics: ['score']}, {metrics: ['tokens', 'tokens']}])
    assert.throws(() => decide(observations(), options));
});

test('comparison does not mutate frozen inputs or derive cost from the legacy score', () => {
  const results = observations().map(row => Object.freeze({...row, score: row.arm === 'active' ? -1 : 100}));
  Object.freeze(results);
  assert.equal(decide(results).promote, true);
  const missing = results.map(({elapsedSeconds, tokens, ...row}) => row);
  assert.equal(decide(missing).reason, 'missing-cost-telemetry');
});
