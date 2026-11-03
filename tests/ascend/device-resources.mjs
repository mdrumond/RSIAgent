import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createEnvironment } from '../../integrations/environments/ascend.mjs';

// Real foundation parser runs against a CPU npu-smi wire-format fixture. No NPU evidence.
async function fixture(t, health = {}) {
  const instancePath = await mkdtemp(join(tmpdir(), 'ascend-resources-'));
  t.after(() => rm(instancePath, { recursive: true, force: true }));
  const bin = join(instancePath, 'bin'), source = join(instancePath, 'project');
  await mkdir(bin); await mkdir(source);
  const instructionFile = join(instancePath, 'user-instructions.md');
  await writeFile(instructionFile, 'User-selected Ascend instructions');
  const telemetry = join(instancePath, 'devices.json');
  const rows = { 0: {}, 1: { status: 'ALARM' }, 2: { busy: true, memory: 49 }, 3: { timeout: true } };
  await writeFile(telemetry, JSON.stringify(rows));
  await writeFile(join(bin, 'npu-smi'), `#!/usr/bin/env python3
import json,sys,os,time
args=sys.argv; device=int(args[args.index('-i')+1]); query=args[args.index('-t')+1]
row=json.load(open(os.environ['ASCEND_TEST_TELEMETRY']))[str(device)]
if row.get('timeout'): time.sleep(2)
outputs={'board':f'NPU ID: {device}\\nChip Name: Ascend950',
'health':'Health Status: '+row.get('status','OK')+'\\nError Code: 0',
'usages':f'HBM Capacity(MB): 1000\\nHBM Usage Rate(%): {row.get("memory",0)}\\nAicore Usage Rate(%): 0\\nAivector Usage Rate(%): 0\\nNPU Utilization(%): 0\\nAicube Usage Rate(%): 0',
'proc-mem':'Process id: 123' if row.get('busy') else 'No process in device.',
'current-fault-event':'Fault Event Count: 0'}
print(outputs[query])
`, { mode: 0o700 });
  const calls = [], events = [], jobs = new Map();
  let enabled = [0,1,2,3], disconnect = false, sequence = 0, clockSkew = 0;
  const client = async args => {
    calls.push(args);
    if (args[0] === 'npu') return { state: 'completed', profiles: [{ target: 'fixture', devices: enabled.map(id => ({ id, enabled: true })) }] };
    if (args[0] === 'capabilities') return { state: 'available' };
    if (args[0] === 'preflight') return { state: 'completed' };
    if (args[0] === 'upload') return { state: 'completed' };
    if (args[0] === 'download') { await writeFile(args[3], ''); return { state: 'completed' }; }
    if (args[0] === 'run') {
      const handle = `remote:fixture:job:${++sequence}`, file = args[args.indexOf('--file')+1];
      let content = '';
      if (file.endsWith('probe.sh')) {
        const out = await promisify(execFile)('bash', [file], { env: { ...process.env, PATH: bin+':'+process.env.PATH, ASCEND_TEST_TELEMETRY: telemetry } });
        const response = JSON.parse(out.stdout); response.observed_at += clockSkew;
        content = JSON.stringify(response);
      }
      jobs.set(handle, content);
      return { handle, state: 'queued' };
    }
    if (args[0] === 'observe') return { state: disconnect ? 'observation-unavailable' : 'completed', exit: 0 };
    if (args[0] === 'logs') return { state: 'completed', content: jobs.get(args[1]) };
    throw new Error('unexpected CPL operation '+args[0]);
  };
  const context = { instancePath, client, sdk: { createLocalBashOperations: () => ({}) },
    event: async (type,data) => events.push({ type,...data }), exclude: async (_,fn) => fn() };
  const options = { mode: 'cpl', target: 'fixture', runtime: 'registered', instructionFile,
    workspacePath: '/registry/workspace', remoteDirectory: 'owned', pollMs: 0, probeSeconds: 3,
    health: { product: 'a5', chip: null, query_timeout_seconds: .1, ...health } };
  const environment = await createEnvironment(options, context);
  return { environment, options, context, source, calls, events, rows, telemetry,
    setEnabled: ids => { enabled = ids; }, disconnect: value => { disconnect = value; }, setClockSkew: seconds => { clockSkew = seconds; },
    update: () => writeFile(telemetry, JSON.stringify(rows)) };
}

test('fresh parser-backed discovery filters permission, busy, Alarm, and unresponsive devices', async t => {
  const f = await fixture(t);
  const first = await f.environment.resources({ target: 'fixture' });
  assert.deepEqual(first.filter(row => row.available).map(row => row.metadata.physicalDevice), [0]);
  assert.ok(first.every(row => row.id.startsWith('ascend-') && row.metadata.healthSha256.length === 64));
  f.rows[0].busy = true; await f.update();
  assert.equal((await f.environment.resources()).some(row => row.available), false);
  f.setEnabled([]);
  assert.deepEqual(await f.environment.resources(), []);
  assert.deepEqual(await f.environment.resources({ target: 'other' }), []);
});

test('ordinary Alarm/busy opt-ins retain policy, parallel availability stays idle, timing is healthy idle', async t => {
  const f = await fixture(t, { allow_alarm: true, allow_busy: true });
  const ordinary = await f.environment.resources();
  assert.deepEqual(ordinary.filter(row => row.available).map(row => row.metadata.physicalDevice), [0,1]);
  assert.equal(ordinary.find(row => row.metadata.physicalDevice === 2).metadata.admitted, true);
  f.rows[2].memory = 50; await f.update();
  const full = await f.environment.resources();
  assert.equal(full.find(row => row.metadata.physicalDevice === 2).metadata.admitted, false);
  const timing = await f.environment.resources({ purpose: 'timing' });
  assert.deepEqual(timing.filter(row => row.available).map(row => row.metadata.physicalDevice), [0]);
});

test('assigned opaque lease rechecks and forwards physical declaration, logical-zero mask and elapsed remainder', async t => {
  const f = await fixture(t);
  const lease = (await f.environment.resources()).find(row => row.available);
  const bound = await createEnvironment(f.options, { ...f.context, resource: lease });
  await bound.prepare({ project: { id:'p', source:f.source }, attempt:{ id:'a' } });
  const result = await bound.execute('kernel-command', f.source, { timeout:3, env:{ ASCEND_RT_VISIBLE_DEVICES:'7' } });
  assert.equal(result.settled, true);
  const run = f.calls.find(args => args[0] === 'run' && args[args.indexOf('--file')+1].endsWith('command.sh'));
  assert.equal(run[run.indexOf('--npu')+1], '0');
  assert.equal(run[run.indexOf('--runtime')+1], 'registered');
  const script = await readFile(run[run.indexOf('--file')+1], 'utf8');
  const config = JSON.parse(script.slice(script.lastIndexOf(" '{")+2,-2));
  assert.equal(config.env.ASCEND_RT_VISIBLE_DEVICES, '0');
  assert.equal(config.env.ASCEND_VISIBLE_DEVICES, '0');
  assert.ok(config.timeout > 0 && config.timeout < 3);
  await assert.rejects(bound.execute('work',f.source,{ resourceId:'0' }), /opaque|Assigned/);
  f.rows[0].busy = true; await f.update();
  await assert.rejects(bound.execute('work',f.source), /admission refused/);
});

test('probe disconnect persists exact handle across restart and never resubmits unknown work', async t => {
  const f = await fixture(t);
  f.disconnect(true);
  await assert.rejects(f.environment.resources(), /unresolved/);
  const handle = f.events.find(row => row.type === 'execution-handle').handle;
  const restarted = await createEnvironment(f.options,f.context);
  await assert.rejects(restarted.resources(), /unresolved/);
  assert.equal(f.calls.filter(args => args[0] === 'run').length, 1);
  assert.ok(f.calls.filter(args => args[0] === 'observe').every(args => args[1] === handle));
  f.disconnect(false);
  assert.equal((await restarted.resources()).filter(row => row.available).length, 1);
  assert.equal(f.calls.filter(args => args[0] === 'run').length, 1);
});

test('actual user instruction file is required rather than fabricated defaults', async t => {
  const f = await fixture(t);
  await assert.rejects(createEnvironment({ ...f.options, instructionFile:undefined },f.context), /instructionFile/);
  await assert.rejects(createEnvironment({ ...f.options, instructionFile:join(f.source,'missing') },f.context), /ENOENT/);
});


test('remote wall-clock skew does not replace local elapsed freshness', async t => {
  const f = await fixture(t); f.setClockSkew(3600);
  const resources = await f.environment.resources();
  assert.equal(resources.filter(row => row.available).length, 1);
  assert.ok(resources[0].metadata.observedAt > resources[0].metadata.checkedAt + 3500);
});
