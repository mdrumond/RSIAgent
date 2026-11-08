import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {syncBuiltinESMExports} from 'node:module';
import {mkdtemp, writeFile, rm} from 'node:fs/promises';
import {join} from 'node:path';
import {homedir, tmpdir} from 'node:os';
import {createCplClient} from '../../integrations/environments/index.mjs';

test('failed native transfer keeps transport diagnostics without persisting native routes or secrets', async t => {
  const root = await mkdtemp(join(tmpdir(), 'cpl-failure-'));
  t.after(() => rm(root, {recursive:true, force:true}));
  const read = fs.readFile, skill = join(homedir(), '.agents/skills/remote-access/SKILL.md');
  const mocked = t.mock.method(fs, 'readFile', async (path, ...args) =>
    path === skill ? 'Native client test instructions' : read(path, ...args));
  syncBuiltinESMExports();
  try {
    for (const [name, stderr, state, code, expected] of [
      ['reset', 'token=private-value\nkex_exchange_identification: read: Connection reset by peer private-host', 'failed', 1, ['connection-reset']],
      ['timeout', 'ssh: connect to host private-host port 22: Connection timed out', 'failed', 0, ['connection-timeout']],
      ['closed', 'rsync: connection unexpectedly closed (0 bytes received so far) [sender]\nrsync error: unexplained error (code 255) at io.c(232)', 'failed', 1, ['connection-closed']],
      ['unknown', 'native-token=private-value private-host', 'failed', 1, ['native-error-output']],
      ['success', 'remote transfer attempt 1/3: native rsync resume route', 'completed', 0, undefined],
    ]) {
      await t.test(name, async () => {
        const executable = join(root, name);
        await writeFile(executable, `#!/usr/bin/env node
process.stderr.write(${JSON.stringify(stderr)});
process.stdout.write(JSON.stringify({backend:'ssh',operation:'upload',state:${JSON.stringify(state)},exit:${state === 'failed' ? 255 : 0},attempts:3}));
process.exitCode=${code};
`, {mode:0o700});
        const result = await createCplClient(executable)(['upload','fixture','local','remote']);
        assert.equal(result.state, state);
        assert.equal(result.exit, state === 'failed' ? 255 : 0);
        assert.deepEqual(result.diagnostic?.categories, expected);
        if (name === 'closed') assert.equal(result.diagnostic.rsyncExit, 255);
        assert.equal(JSON.stringify(result).includes('private-value'), false);
        assert.equal(JSON.stringify(result).includes('private-host'), false);
      });
    }
    await t.test('failure before native JSON', async () => {
      const executable = join(root, 'no-json');
      await writeFile(executable, `#!/usr/bin/env node
process.stderr.write('ssh: connect to host private-host port 22: Connection refused token=private-value');
process.exitCode=255;
`, {mode:0o700});
      await assert.rejects(createCplClient(executable)(['preflight','fixture']), error => {
        assert.deepEqual(error.diagnostic.categories, ['connection-refused']);
        assert.equal(error.message.includes('private-value'), false);
        assert.equal(error.message.includes('private-host'), false);
        return true;
      });
    });
  } finally {
    mocked.mock.restore();
    syncBuiltinESMExports();
  }
});
