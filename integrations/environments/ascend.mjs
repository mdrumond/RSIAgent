import { createEnvironment as genericEnvironment, createCplClient } from './index.mjs';
import { readFile, writeFile, mkdir, mkdtemp, rename, unlink } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const quote = value => "'" + String(value).replaceAll("'", "'\\''") + "'";
const digest = text => createHash('sha256').update(text).digest('hex');
const terminal = new Set(['completed', 'failed', 'cancelled']);
const timing = purpose => ['timing', 'profiling', 'performance'].includes(purpose);
const readJson = async path => JSON.parse(await readFile(path, 'utf8'));
async function save(path, value) {
  await writeFile(path + '.tmp', JSON.stringify(value), { mode: 0o600 });
  await rename(path + '.tmp', path);
}

// Health parsing and admission remain entirely in the target foundation helper.
const probe = `import hashlib,json,sys,time
c=json.loads(sys.argv[1]); source=c['source']
if hashlib.sha256(source.encode()).hexdigest()!=c['sha256']: raise ValueError('health helper identity mismatch')
ns={'__name__':'_ascend_health'}; exec(compile(source,'trusted-ascend-health.py','exec'),ns)
deadline=time.monotonic()+c['seconds']; rows=[]
for device in c['devices']:
 rows.append(ns['local_probe'](physical_device=device,configuration=c['configuration'],remaining=lambda:max(0.,deadline-time.monotonic())))
 if time.monotonic()>=deadline: break
print(json.dumps({'observed_at':time.time(),'devices':rows},allow_nan=False))
`;

/** CPL options plus health:{product,chip,allow_alarm?,allow_busy?}, instructionFile,
 * profile? and probeSeconds? (default60). context.resource/options.resource is the
 * leased row or opaque id returned by resources(); no caller-selected raw NPU ID.
 * Idle-only availability keeps the core's parallel leases independent of health policy.
 */
export async function createEnvironment(options, context) {
  if (options.mode !== 'cpl' || !options.target || !options.runtime || !options.instructionFile)
    throw new Error('Ascend binding requires CPL target, registered runtime, and instructionFile');
  if (!['a3', 'a5'].includes(options.health?.product)) throw new Error('Explicit A3/A5 health profile required');
  const seconds = options.probeSeconds ?? 60;
  if (!Number.isFinite(seconds) || seconds <= 0 || seconds > 120) throw new Error('probeSeconds must be in (0,120]');
  const client = context.client ?? createCplClient(options.clientPath);
  const assigned = context.resource ?? options.resource;
  const assignedId = typeof assigned === 'string' ? assigned : assigned?.id;
  const idFor = physical => 'ascend-' + digest(JSON.stringify([options.target, options.runtime, physical]));
  const root = join(context.instancePath, '.ascend-resources', digest(JSON.stringify([options.target, options.runtime, assignedId ?? 'inventory'])));
  await mkdir(root, { recursive: true, mode: 0o700 });
  const pendingPath = join(root, 'pending.json');
  const source = await readFile(new URL('../../ascend_tools/targets/health.py', import.meta.url), 'utf8');
  const healthSha256 = digest(source);
  const bindings = { ...options.env };
  const environment = await genericEnvironment({ ...options, env: bindings }, { ...context, client });
  let queue = Promise.resolve();
  function serial(fn) {
    const result = queue.then(fn);
    queue = result.catch(() => {});
    return result;
  }
  async function enabled() {
    const policy = await client(['npu', 'list', options.target]);
    if (policy.state !== 'completed') throw new Error('NPU permission inventory unavailable');
    const profile = policy.profiles?.find(row => row.target === options.target);
    if (!profile) throw new Error('NPU permission profile missing');
    return profile.devices.filter(row => row.enabled === true && Number.isInteger(row.id) && row.id >= 0).map(row => row.id);
  }
  async function snapshot(devices, purpose, budget = seconds) {
    if (!(budget > 0)) throw new Error('Resource probe elapsed allocation exhausted');
    let pending;
    try { pending = await readJson(pendingPath); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const configuration = { ...options.health, require_idle: timing(purpose) || options.health.require_idle === true };
    if (!pending) {
      const dir = await mkdtemp(join(root, 'probe-'));
      const file = join(dir, 'probe.sh'), receipt = join(dir, 'native-submission.json');
      const request = { source, sha256: healthSha256, seconds: Math.min(seconds, budget), devices, configuration };
      await writeFile(file, `${quote(options.python ?? 'python3')} -c ${quote(probe)} ${quote(JSON.stringify(request))}\n`, { mode: 0o600 });
      pending = { started: Date.now(), startedMonotonic: process.hrtime.bigint().toString(), seconds: request.seconds, devices, configuration, receipt, handle: null };
      // Persist ambiguous submission BEFORE calling transport; never infer no work from no ID.
      await save(pendingPath, pending);
      let submitted;
      try { submitted = await client(['run', options.target, '--runtime', options.runtime, '--file', file, '--receipt-file', receipt]); }
      catch { try { submitted = await readJson(receipt); } catch { /* unknown; retain pending */ } }
      if (submitted?.handle) { pending.handle = submitted.handle; await save(pendingPath, pending); }
    }
    if (!pending.handle) {
      try { pending.handle = (await readJson(pending.receipt)).handle; await save(pendingPath, pending); } catch { /* preserve ambiguity */ }
    }
    if (!pending.handle) throw new Error(`Probe submission identity unknown; do not resubmit (${pendingPath})`);
    await context.event('execution-handle', { handle: pending.handle, purpose: 'resource-health' });
    const elapsed = () => Number(process.hrtime.bigint() - BigInt(pending.startedMonotonic)) / 1e9;
    const deadline = performance.now() + Math.max(0, (pending.seconds - elapsed()) * 1000);
    let observed;
    do {
      try { observed = await client(['observe', pending.handle]); }
      catch { observed = { state: 'observation-unavailable' }; }
      await context.event('execution-state', { handle: pending.handle, purpose: 'resource-health', state: terminal.has(observed.state) ? 'stopped' : observed.state });
      if (terminal.has(observed.state)) break;
      if (!['queued', 'dispatching', 'running'].includes(observed.state) || performance.now() >= deadline)
        throw new Error(`Probe unresolved; observe ${pending.handle}`);
      await delay(options.pollMs ?? 250);
    } while (true);
    if (observed.state !== 'completed' || observed.exit !== 0) {
      await unlink(pendingPath);
      throw new Error(`Health probe failed: ${pending.handle}`);
    }
    // If log retrieval disconnects, retain pending handle and retry the SAME logs next call.
    const logs = await client(['logs', pending.handle, '--stream', 'stdout', '--tail', '1']);
    const result = JSON.parse(logs.content);
    await save(join(root, 'last-observation.json'), { ...result, checked_at: Date.now() / 1000, handle: pending.handle, healthSha256 });
    await unlink(pendingPath);
    // Remote wall clocks may differ; freshness is local elapsed since this exact submission.
    const age = elapsed();
    if (!Number.isFinite(age) || age < 0 || age > seconds || JSON.stringify(configuration) !== JSON.stringify(pending.configuration))
      throw new Error('Resolved probe is stale or belongs to a different policy; request a fresh snapshot');
    return { ...result, checked_at: Date.now() / 1000, handle: pending.handle };
  }
  async function resources({ purpose = 'ordinary', target } = {}) {
    if (target && target !== options.target && target !== options.profile) return [];
    return serial(async () => {
      const devices = await enabled();
      if (!devices.length) return [];
      const result = await snapshot(devices, purpose);
      const permitted = new Set(await enabled());
      return result.devices.filter(row => devices.includes(row.physical_device) && permitted.has(row.physical_device)).map(row => ({
        id: idFor(row.physical_device),
        available: row.admitted === true && row.measurements?.idle === true,
        metadata: { target: options.target, runtime: options.runtime, product: options.health.product,
          physicalDevice: row.physical_device, logicalDevice: 0, observedAt: result.observed_at, checkedAt: result.checked_at,
          admitted: row.admitted === true, measurements: row.measurements, failure: row.failure,
          probeHandle: result.handle, healthSha256 },
      }));
    });
  }
  return {
    ...environment, resources,
    async execute(command, cwd, opts = {}) {
      const started = performance.now();
      const remaining = () => opts.timeout === undefined ? undefined : Math.max(0, opts.timeout - (performance.now() - started) / 1000);
      return serial(async () => {
        if (opts.signal?.aborted || remaining() === 0) throw new Error('Execution cancelled or elapsed budget exhausted');
        const requested = opts.resourceId ?? assignedId;
        if (!requested || assignedId && requested !== assignedId) throw new Error('Assigned opaque Ascend resource required');
        const devices = await enabled();
        const physical = devices.find(device => idFor(device) === requested);
        if (physical === undefined) throw new Error('Assigned resource is not enabled for this target/runtime');
        const result = await snapshot([physical], opts.purpose ?? 'ordinary', remaining() ?? seconds);
        const row = result.devices.find(item => item.physical_device === physical);
        if (row?.admitted !== true || !(await enabled()).includes(physical)) throw new Error('Fresh resource admission refused');
        bindings.ASCEND_RT_VISIBLE_DEVICES = bindings.ASCEND_VISIBLE_DEVICES = String(physical);
        return environment.execute(command, cwd, { ...opts, timeout: remaining(), resourceId: physical,
          env: { ...opts.env, ASCEND_RT_VISIBLE_DEVICES: String(physical), ASCEND_VISIBLE_DEVICES: String(physical) } });
      });
    },
  };
}
