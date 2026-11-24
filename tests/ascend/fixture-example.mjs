import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {mkdtemp, readFile, readdir, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {test} from 'node:test';
import {fileURLToPath} from 'node:url';
import {rowSumHandoffs} from '../../examples/fixture-design/row-sum/packets.mjs';

test('emitting a worked fixture keeps evaluator programs out of the public packet', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'fixture-example-'));
  t.after(() => rm(dir, {recursive: true, force: true}));
  const emit = fileURLToPath(new URL('../../examples/fixture-design/row-sum/emit.mjs', import.meta.url));
  const result = spawnSync(process.execPath, [emit, dir], {encoding: 'utf8'});
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(await readdir(join(dir, 'public')), ['input.json']);
  const read = async path => JSON.parse(await readFile(join(dir, path), 'utf8'));
  const input = await read('public/input.json');
  const design = await read('evaluator/test.json');
  assert.equal(input.records.find(row => row.reference === 'failing-source').text, design.starter);
  for (const role of ['good', 'bad', 'checker']) {
    assert.ok(!input.records.some(row => row.text === design[role]));
  }
  const handoffs = await rowSumHandoffs();
  assert.deepEqual(handoffs.designer.input, input);
  assert.deepEqual(handoffs.reviewer.input.fixture, design);
  assert.deepEqual(handoffs.feedback.accepted, design);
});
