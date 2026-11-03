import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, cp } from 'node:fs/promises';
import { join, basename } from 'node:path';
import { pathToFileURL } from 'node:url';
import { State } from '../../controller/state.mjs';
import { ownerStopped } from '../../controller/lifecycle.mjs';
import { createEnvironment } from '../../integrations/environments/index.mjs';
import { upgrade } from '../../cli/instance.mjs';
import { setup, rows, until } from './driver_setup.mjs';

test('LOCAL interrupted native attempt upgrades and resumes its same session; live orphan still blocks',
  { timeout: 120000 }, async t => {
    const f = await setup(t, 'interrupt', ['local']);
    // This release retains the fixture's explicitly configured embedding backend.
    const release = join(f.root, 'replacement');
    await cp(join(f.root, 'machinery'), release, { recursive: true,
      filter: path => !['node_modules', '__pycache__'].includes(basename(path)) });
    const wave = f.start('wave');
    await until(() => {
      const [attempt] = rows(f.root, 'SELECT * FROM attempts');
      return attempt?.session && attempt.elapsed >= 1 && rows(f.root,
        "SELECT 1 FROM events WHERE type='execution-handle'").length;
    });
    await f.run('stop');await wave.output;
    const [interrupted] = rows(f.root, 'SELECT * FROM attempts');
    assert.equal(interrupted.outcome, null);
    assert.equal(await ownerStopped(interrupted.owner), true);
    const ledger = new State(join(f.root, 'state.sqlite'));
    t.after(() => ledger.close());
    const sdk = await import(pathToFileURL(join(f.root, 'machinery/node_modules/@earendil-works/pi-coding-agent/dist/index.js')));
    let orphan;
    const environment = await createEnvironment({ mode: 'local' }, {
      instancePath: join(f.root, 'attempts', interrupted.id), sdk,
      event: async (type, data) => {
        ledger.event(interrupted.id, type, data);
        if (type === 'execution-handle') orphan = data.handle;
      }, exclude: async (_, fn) => fn()
    });
    const application = environment.execute('sleep 60', f.root);
    try {
      await until(async () => orphan && (await environment.observe(orphan)).state === 'running');
      await assert.rejects(upgrade(f.root, release), /external executions/);
    } finally {
      if (orphan) await environment.stop(orphan);
      await application;
    }
    const before = rows(f.root, 'SELECT * FROM attempts');
    const session = await readFile(interrupted.session);
    const events = rows(f.root, 'SELECT * FROM events WHERE attempt=\'' + interrupted.id + '\'');
    await upgrade(f.root, release);
    assert.deepEqual(rows(f.root, 'SELECT * FROM attempts'), before);
    assert.deepEqual(await readFile(interrupted.session), session);
    assert.deepEqual(rows(f.root, 'SELECT * FROM events WHERE attempt=\'' + interrupted.id + '\''), events);
    await f.run('resume');
    const [resumed] = rows(f.root, 'SELECT * FROM attempts');
    assert.equal(resumed.id, interrupted.id);
    assert.equal(resumed.session, interrupted.session);
    assert.equal(resumed.outcome, 'passed');
    assert.ok(resumed.elapsed >= before[0].elapsed);
    assert.equal(rows(f.root, "SELECT * FROM events WHERE type='job-admitted'").length, 1);
    f.pass();
  });
