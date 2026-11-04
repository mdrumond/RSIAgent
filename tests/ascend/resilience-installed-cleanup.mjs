import test from 'node:test';
import assert from 'node:assert/strict';
import {setTimeout as delay} from 'node:timers/promises';
import {observeNativeClosure} from './resilience/installed-fixture.mjs';

for (const mode of ['remaining-budget', 'runner-signal']) test(`installed cleanup preserves partial closure on ${mode}`, async () => {
  const abort = new AbortController();
  const timer = mode === 'runner-signal' ? setTimeout(() => abort.abort(new Error('Runner cleanup exhausted')), 40) : null;
  const requested = [], cancelled = [];
  let archived;
  const client = async ([operation, handle], {signal}) => {
    assert.equal(operation, 'observe');
    requested.push(handle);
    if (handle === 'closed') return {state: 'completed', exit: 0};
    try {await delay(1000, null, {signal});}
    catch (error) {cancelled.push(handle); throw error;}
    return {state: 'running'};
  };
  const started = performance.now();
  try {
    await assert.rejects(observeNativeClosure(['slow-a', 'closed', 'slow-b'], client,
      {signal: abort.signal, remainingSeconds: mode === 'remaining-budget' ? 0.04 : 2},
      async observations => {archived = observations;}), error =>
      error instanceof AggregateError && error.errors.length === 2);
    assert.ok(performance.now() - started < 800, 'Cleanup must honor the shared deadline');
    assert.deepEqual(requested.sort(), ['closed', 'slow-a', 'slow-b']);
    assert.deepEqual(cancelled.sort(), ['slow-a', 'slow-b']);
    assert.equal(archived.find(o => o.handle === 'closed').state, 'completed');
    assert.ok(archived.filter(o => o.handle !== 'closed').every(o => o.state === 'unknown' && o.error));
  } finally {clearTimeout(timer);}
});
