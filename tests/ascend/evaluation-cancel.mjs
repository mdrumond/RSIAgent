import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { setup, rows, until } from './driver_setup.mjs';
import { bootstrap, wave, resume, settled } from '../../cli/evaluation.mjs';
import { readConfig } from '../../cli/instance.mjs';
import { ownerStopped } from '../../controller/lifecycle.mjs';
import { operate } from '../../cli/runtime.mjs';

test('internal evaluation cancellation closes its controller and Python worker before same-session resume',
  { timeout: 120000 }, async t => {
    const f = await setup(t, 'interrupt', ['original']);
    const config = await readConfig(f.root), round = join(f.root, 'round'); await mkdir(round);
    const instanceRoot = join(round, 'sample'), credentials = join(round, 'fixture.env');
    await writeFile(credentials, 'DRIVER_FIXTURE_KEY=fixture-only\n');
    config.actor = { ...config.actor, credentials_file: credentials };
    await bootstrap({ sourceInstanceRoot: f.root, instanceRoot, config,
      projectId: 'original', projectIds: ['original'], skillVersion: 'baseline' });
    const sample = await readConfig(instanceRoot);
    assert.equal((await readConfig(join(round, 'inputs'))).python, config.python);
    let cleaned = false;
    const cleanup = async () => {
      if (cleaned) return;
      await operate('stop', sample);
      await until(async () => {
        const control = rows(instanceRoot, "SELECT value FROM control WHERE key='owner'")[0];
        const owner = control && JSON.parse(control.value);
        return !owner || await ownerStopped(owner);
      });
      await until(async () => {
        for (const request of rows(instanceRoot, 'SELECT status,owner FROM requests')) {
          if (['queued', 'running'].includes(request.status)) return false;
          if (request.owner && !await ownerStopped(request.owner)) return false;
        }
        return true;
      });
      const receipt = JSON.parse(await readFile(join(instanceRoot, 'knowledge-service.json'), 'utf8'));
      if (!await ownerStopped(receipt.owner)) process.kill(receipt.owner.pid, 'SIGTERM');
      cleaned = true;
    };
    t.after(cleanup);
    const abort = new AbortController();
    const running = wave({ instanceRoot, config: sample, signal: abort.signal });
    const rejection = assert.rejects(running, { name: 'AbortError' });
    await until(() => rows(instanceRoot, "SELECT data FROM events WHERE type='execution-handle'").length > 0);
    const before = rows(instanceRoot, 'SELECT id,owner,session FROM attempts')[0];
    const controller = JSON.parse(rows(instanceRoot, "SELECT value FROM control WHERE key='owner'")[0].value);
    assert.ok(before.session); abort.abort(); await rejection;
    assert.equal(await ownerStopped(controller), true);
    assert.equal(await ownerStopped(before.owner), true);
    assert.equal(await settled({ instanceRoot }), true);
    assert.equal(JSON.parse(rows(instanceRoot, "SELECT value FROM control WHERE key='stop'")[0].value), true);
    const result = await resume({ instanceRoot });
    assert.equal(result.correct, true); assert.deepEqual(result.attempts, [before.id]);
    assert.equal(rows(instanceRoot, 'SELECT session FROM attempts')[0].session, before.session);
    await cleanup();
    f.pass();
  });
