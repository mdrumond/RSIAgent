import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createEnvironment, createEnvironmentTools } from '../../integrations/environments/index.mjs';

async function fixture(t, client, mode = 'cpl', extra = {}) {
  const instancePath = await mkdtemp(join(tmpdir(), 'environment-test-'));
  t.after(() => rm(instancePath, { recursive: true, force: true }));
  const events = [], excluded = [];
  const sdk = { createLocalBashOperations: () => ({ exec: async () => ({ exitCode: 0 }) }),
    createBashTool: (cwd, options) => ({ cwd, ...options }) };
  const context = { instancePath, client, sdk, event: async (type, data) => { events.push({ type, ...data }); },
    exclude: async (label, fn) => { excluded.push(label); return fn(); } };
  return { environment: await createEnvironment({ mode, target: 'fake', pollMs: 0, ...extra }, context), events, excluded, sdk, instancePath };
}

test('handle precedes observation; native tool receives output and exit', async t => {
  let f;
  const calls = [];
  f = await fixture(t, async args => {
    calls.push(args);
    if (args[0] === 'run') return { handle: 'remote:fake:job:one' };
    if (args[0] === 'logs') return { content: args.includes('stdout') ? 'answer\n' : '' };
    assert.equal(f.events[0].type, 'execution-handle');
    return { state: 'completed', exit: 7 };
  });
  const [tool] = await createEnvironmentTools(f.environment, '/work', f.sdk);
  let output = '';
  const result = await tool.operations.exec('echo answer', '/work', { timeout: 4, onData: data => { output += data; } });
  assert.equal(result.exitCode, 7);
  assert.equal(output, 'answer\n');
  assert.equal(calls.filter(args => args[0] === 'run').length, 1);
  assert.equal(f.excluded.length, 0);
  assert.ok(Number(calls[0][calls[0].indexOf('--timeout') + 1]) <= 4);
});

test('disconnect preserves handle; observe and stop never resubmit', async t => {
  let connected = false, cancelled = false, runs = 0;
  const f = await fixture(t, async args => {
    if (args[0] === 'run') { runs++; return { handle: 'remote:fake:job:two' }; }
    if (args[0] === 'cancel') { cancelled = true; return {}; }
    if (!connected) throw new Error('offline');
    return { state: cancelled ? 'cancelled' : 'running' };
  });
  const result = await f.environment.execute('work', '/work');
  assert.equal(result.settled, false);
  assert.equal((await f.environment.observe(result.handle)).state, 'unknown');
  connected = true;
  assert.equal((await f.environment.observe(result.handle)).state, 'running');
  assert.equal((await f.environment.stop(result.handle)).state, 'stopped');
  assert.equal(runs, 1);
});

test('native receipt recovers identity after submission disconnect', async t => {
  const f = await fixture(t, async args => {
    if (args[0] === 'run') {
      await writeFile(args[args.indexOf('--receipt-file') + 1], JSON.stringify({ handle: 'remote:fake:job:receipt' }));
      throw new Error('disconnected after receipt');
    }
    return { state: 'completed', exit: 0 };
  });
  assert.equal((await f.environment.execute('true', '/work')).handle, 'remote:fake:job:receipt');
});

test('setup and copies are excluded; workload is charged', async t => {
  let tick = 0, excludedMs = 0;
  const instancePath = await mkdtemp(join(tmpdir(), 'environment-clock-'));
  t.after(() => rm(instancePath, { recursive: true, force: true }));
  const env = await createEnvironment({ mode: 'cpl', target: 'fake', uploads: [{ local: 'input', remote: 'input' }] }, {
    instancePath, sdk: { createLocalBashOperations: () => ({}) }, event: async () => {},
    exclude: async (label, fn) => { const before = tick; const result = await fn(); excludedMs += tick - before; return result; },
    client: async args => { tick += 10; return args[0] === 'run' ? { handle: 'remote:fake:job:clock' } : { state: args[0] === 'capabilities' ? 'available' : 'completed', exit: 0 }; },
  });
  await env.prepare({ project: { id: 'p' }, attempt: { id: 'a' } });
  assert.equal(excludedMs, 30);
  await env.execute('compile', '/work');
  assert.equal(tick - excludedMs, 20);
});

test('local path reuses native operations and exposes resource', async t => {
  const f = await fixture(t, undefined, 'local');
  assert.equal((await f.environment.execute('true', f.instancePath)).exitCode, 0);
  assert.equal(f.events[0].type, 'execution-handle');
  assert.equal((await f.environment.observe(f.events[0].handle)).state, 'stopped');
  assert.equal((await f.environment.resources({}))[0].available, true);
  assert.ok(Array.isArray(await f.environment.browse(f.instancePath)));
});
