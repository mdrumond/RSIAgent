import test from 'node:test';
import assert from 'node:assert/strict';
import { access, readFile, writeFile, symlink } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { setup, rows, until } from './driver_setup.mjs';
import { bootstrap, readConfig } from '../../cli/instance.mjs';
import { operate } from '../../cli/runtime.mjs';
import { ownerStopped } from '../../controller/lifecycle.mjs';
import { localPythonConfig } from '../../examples/local-python/config.mjs';

// The provider is scripted; the copied machinery and both Python verifiers are real.
test('a reused campaign and Ascend profile admits both fixed local experiment controls',
  { timeout: 120000 }, async t => {
    const f = await setup(t, 'phase2', ['reduction', 'parser']);
    const original = await readConfig(f.root), instance = join(f.root, 'fixed-experiment');
    const credentials = join(f.root, 'fixture.env');
    await writeFile(credentials, 'DRIVER_FIXTURE_KEY=fixture-only\n');
    const profile = { ...original, baseline: join(f.root, 'knowledge.sqlite'),
      skills: join(f.root, 'skills/versions/baseline'), machinery: join(f.root, 'machinery'),
      actor: { ...original.actor, credentials_file: credentials },
      campaign_plan: [{ id: 'old-campaign', phase: 'phase1', new_projects: 20 }],
      phase_sequence: ['phase1', 'phase2'], phase_targets: { phase2: 40 }, completion_target: 40,
      resource_ids: ['ascend-device-3'],
      skillSettings: { evaluationSeconds: 120, tuningDisabled: true, tuningProjectIds: ['old-control'] } };
    const before = structuredClone(profile);
    const input = localPythonConfig(profile, original.projects);
    assert.deepEqual(profile, before);
    let cleaned = false;
    const cleanup = async () => {
      if (cleaned) return;
      try { await access(join(instance, 'state.sqlite')); }
      catch (error) { if (error.code === 'ENOENT') return; throw error; }
      await operate('stop', await readConfig(instance));
      await until(async () => {
        const row = rows(instance, "SELECT value FROM control WHERE key='owner'")[0];
        return !row || !JSON.parse(row.value) || await ownerStopped(JSON.parse(row.value));
      });
      await until(async () => {
        if (!rows(instance, "SELECT 1 FROM sqlite_master WHERE name='requests'").length) return true;
        for (const request of rows(instance, 'SELECT status,owner FROM requests')) {
          if (['queued', 'running'].includes(request.status)) return false;
          if (request.owner && !await ownerStopped(request.owner)) return false;
        }
        return true;
      });
      try {
        const receipt = JSON.parse(await readFile(join(instance, 'knowledge-service.json'), 'utf8'));
        if (!await ownerStopped(receipt.owner)) process.kill(receipt.owner.pid, 'SIGTERM');
      } catch (error) { if (!['ENOENT', 'ESRCH'].includes(error.code)) throw error; }
      cleaned = true;
    };
    t.after(cleanup);
    await bootstrap(instance, { ...input, install: false });
    await symlink(resolve('node_modules'), join(instance, 'machinery/node_modules'), 'dir');
    const config = await readConfig(instance);
    assert.equal(config.python, profile.python);
    assert.deepEqual(config.skills.tuningProjectIds, ['reduction', 'parser']);
    await operate('wave', config);
    assert.deepEqual(rows(instance, 'SELECT project,outcome FROM attempts ORDER BY project')
      .map(({ project, outcome }) => ({ project, outcome })),
      [{ project: 'parser', outcome: 'passed' }, { project: 'reduction', outcome: 'passed' }]);
    assert.equal(rows(instance, "SELECT COUNT(*) n FROM events WHERE type='accepted'")[0].n, 2);
    assert.ok(f.server.requests.length > 0);
    assert.deepEqual(profile, before);
    await cleanup(); f.pass();
  });
