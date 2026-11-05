import {spawn, execFile as nativeExecFile} from 'node:child_process';
import {promisify} from 'node:util';
import {readFile} from 'node:fs/promises';
import {isAbsolute} from 'node:path';
import {randomUUID} from 'node:crypto';
import {processIdentity} from '../controller/lifecycle.mjs';
import {normalizeHostMemory} from '../controller/host-memory.mjs';
export {configureHostMemory} from './host-memory.mjs';

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const integer = value => /^\d+$/.test(String(value)) && Number.isSafeInteger(Number(value)) ? Number(value) : null;
const properties = text => Object.fromEntries(text.trim().split('\n').map(line => {
  const at = line.indexOf('='); return [line.slice(0, at), line.slice(at + 1)];
}));
function context(options = {}) {
  const {execFile = promisify(nativeExecFile), read = readFile, slice = 'rsi.slice'} = options;
  if (!/^[a-zA-Z0-9_-]+\.slice$/.test(slice)) throw new Error('Invalid aggregate slice');
  const {max_bytes} = normalizeHostMemory(options.hostMemory);
  const run = async args => (await execFile('systemctl', ['--user', ...args],
    {timeout:5000, maxBuffer:65536})).stdout;
  const show = async unit => properties(await run(['show', unit, '--no-pager',
    '--property=LoadState,ActiveState,SubState,ControlGroup,MainPID,ExecMainPID,InvocationID,ExecMainCode,ExecMainStatus,Result,MemoryAccounting,MemoryMax,MemorySwapMax,MemoryCurrent,MemoryPeak']));
  const groupPath = group => {
    if (!group?.startsWith('/') || group.split('/').some(part => part === '..' || part === '.'))
      throw new Error('Invalid service control group');
    return '/sys/fs/cgroup' + group;
  };
  async function aggregate() {
    const state = await show(slice), base = groupPath(state.ControlGroup);
    if (state.ActiveState !== 'active' || state.MemoryAccounting !== 'yes' ||
        state.MemoryMax !== String(max_bytes) || state.MemorySwapMax !== '0' ||
        !state.ControlGroup.endsWith('/' + slice) ||
        (await read(base + '/memory.max', 'utf8')).trim() !== String(max_bytes) ||
        (await read(base + '/memory.swap.max', 'utf8')).trim() !== '0')
      throw new Error('Matching aggregate memory enforcement required before admission');
    return state.ControlGroup;
  }
  async function membership(pid, group) {
    if (!Number.isSafeInteger(pid) || pid <= 0) throw new Error('Positive process PID required');
    const identity = options.identify ? await options.identify(pid) : await processIdentity(pid);
    const own = (await read(`/proc/${pid}/cgroup`, 'utf8')).split('\n').find(line => line.startsWith('0::'))?.slice(3);
    if (!own?.startsWith(group + '/')) throw new Error(`Process ${pid} is outside ${slice}; refusing admission`);
    const after = options.identify ? await options.identify(pid) : await processIdentity(pid);
    if (identity.start !== after.start || identity.boot !== after.boot) throw new Error('Process identity changed during admission');
    return {...identity, cgroup:own};
  }
  return {run, show, read, groupPath, aggregate, membership, slice, max_bytes};
}

/** Check an existing KDB before using it, including when no new KDB is spawned. */
export async function verifyProcessMembership(pid, options = {}) {
  const c = context(options);
  return c.membership(pid, await c.aggregate());
}

function descriptorArgs(descriptor) {
  const {argv, cwd, env = {}, stdout, stderr} = descriptor;
  if (!Array.isArray(argv) || !argv.length || argv.some(arg => typeof arg !== 'string' || arg.includes('\0')) ||
      !isAbsolute(argv[0])) throw new Error('Structured argv with absolute executable required');
  if (!isAbsolute(cwd ?? '') || cwd.includes('\0')) throw new Error('Absolute cwd required');
  if (!env || typeof env !== 'object' || Array.isArray(env) || Object.entries(env).some(([key,value]) =>
    !/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) || typeof value !== 'string' || value.includes('\0')))
    throw new Error('String environment entries required');
  for (const path of [stdout, stderr]) if (path !== undefined && (!isAbsolute(path) || /[\0\n\r]/.test(path)))
    throw new Error('Absolute log paths required');
  // systemd-run escapes unit specifiers; pass descriptor strings unchanged.
  return ['--working-directory=' + cwd,
    ...Object.entries(env).map(([key,value]) => '--setenv=' + key + '=' + value),
    ...(stdout ? ['--property=StandardOutput=append:' + stdout] : []),
    ...(stderr ? ['--property=StandardError=append:' + stderr] : []),
    '--', ...argv];
}

/** No implicit provisioning: configureHostMemory must be called externally while stopped.
 * startupFlags runs at submission; supply a durable-state flag that the restarted
 * controller reads on every boot. The caller owns pause interpretation at startup.
 */
export async function startManagedController(descriptor, options = {}) {
  const flags = await options.startupFlags?.() ?? [];
  if (!Array.isArray(flags) || flags.some(flag => typeof flag !== 'string')) throw new Error('String startup flags required');
  return launch({...descriptor, argv:[...descriptor.argv, ...flags]}, {...options, controller:true});
}

/** Returns a native launcher child, but identity.pid is always the service process.
 * wait({timeoutMs}) is bounded; timeout leaves the unit running for explicit stop().
 */
export async function spawnManagedWorker(descriptor, options = {}) {
  return launch(descriptor, {...options, controller:false});
}

async function launch(descriptor, options) {
  const command = descriptorArgs(descriptor), c = context(options);
  const aggregate = await c.aggregate();
  if (!options.controller) await c.membership(process.pid, aggregate);
  for (const pid of options.requiredPids ?? []) await c.membership(pid, aggregate);
  const memory = options.memoryBytes;
  const controllerMemory = options.controllerMemoryBytes ?? Math.min(c.max_bytes, 4 * 1024 ** 3);
  if (!Number.isSafeInteger(controllerMemory) || controllerMemory <= 0 || controllerMemory > c.max_bytes)
    throw new Error('Controller memory cap must fit aggregate');
  if (!options.controller) {
    if (!Number.isSafeInteger(memory) || memory <= 0 || memory > c.max_bytes - controllerMemory)
      throw new Error('Worker memoryBytes must reserve aggregate controller/KDB headroom');
    const current = integer((await c.read(c.groupPath(aggregate) + '/memory.current','utf8')).trim());
    if (current === null || current + Math.ceil(memory * 1.25) > c.max_bytes)
      throw new Error('Insufficient measured aggregate headroom for worker admission');
  }
  const unit = 'rsi-' + (options.controller ? 'controller-' : 'worker-') + randomUUID() + '.service';
  const args = ['--user', '--quiet', '--no-ask-password', '--service-type=exec', '--expand-environment=no',
    '--unit=' + unit, '--slice=' + c.slice, '--property=MemoryAccounting=yes',
    '--property=MemorySwapMax=0', '--property=OOMPolicy=kill', '--property=KillMode=control-group',
    '--property=TimeoutStopSec=5s', '--property=Restart=' + (options.controller ? 'on-failure' : 'no'),
    '--property=MemoryMax=' + (options.controller ? controllerMemory : memory),
    ...(options.controller ? ['--property=RestartSec=1s'] : ['--remain-after-exit']), ...command];
  const child = (options.spawn ?? spawn)('systemd-run', args, {stdio:['ignore','pipe','pipe']});
  // Drain diagnostics with a hard retention bound. Never buffer worker logs here.
  let diagnostic = '';
  for (const stream of [child.stdout,child.stderr]) stream?.on('data', data => {
    diagnostic = (diagnostic + data.toString()).slice(-65536);
  });
  const submitted = new Promise((resolve,reject) => {
    child.once('error',reject);
    child.once('close',code => code === 0 ? resolve() : reject(new Error(`Managed launch failed (${code}): ${diagnostic}`)));
  });
  let timer;
  try {
    await Promise.race([submitted, new Promise((_,reject) => {
      timer = setTimeout(() => {child.kill('SIGKILL'); reject(new Error('Managed launch timed out'));},5000);
    })]);
  } catch (error) {
    await c.run(['stop',unit]).catch(() => {});
    throw error;
  } finally {clearTimeout(timer);}
  let initial, identity;
  try {
    initial = await c.show(unit);
    if (!initial.InvocationID) throw new Error('Missing managed service invocation identity');
    const pid = integer(initial.MainPID) || integer(initial.ExecMainPID);
    identity = {unit, invocation:initial.InvocationID, pid, cgroup:initial.ControlGroup || aggregate + '/' + unit};
    if (integer(initial.MainPID)) Object.assign(identity, await c.membership(pid, aggregate));
  } catch (error) {await c.run(['stop',unit]).catch(() => {}); throw error;}
  let lastMetrics = null, stopped = false;
  const same = (state, afterStop = false) => {
    if (options.controller) return; // Restarts belong to this uniquely named service.
    if (!state.InvocationID && ['inactive','failed'].includes(state.ActiveState) &&
        (afterStop || !integer(state.ExecMainPID) || integer(state.ExecMainPID) === identity.pid)) return;
    if (state.InvocationID !== identity.invocation) throw new Error('Managed service invocation changed');
  };
  async function metrics() {
    if (stopped) return lastMetrics;
    const state = await c.show(unit); same(state);
    const measured = {current_bytes:integer(state.MemoryCurrent), peak_bytes:integer(state.MemoryPeak), events:null};
    const group = state.ControlGroup || identity.cgroup;
    if (group) {
      const base = c.groupPath(group);
      try {
        const values = await Promise.all(['memory.current','memory.peak','memory.events'].map(name => c.read(base + '/' + name,'utf8')));
        measured.current_bytes = integer(values[0].trim());
        measured.peak_bytes = integer(values[1].trim());
        measured.events = Object.fromEntries(values[2].trim().split('\n').map(line => {
          const [key,value] = line.split(/\s+/); return [key,integer(value)];
        }));
      } catch (error) {if (error.code !== 'ENOENT') throw error;}
    }
    lastMetrics = {...measured, peak_bytes:measured.peak_bytes ?? lastMetrics?.peak_bytes ?? null,
      events:measured.events ?? lastMetrics?.events ?? null, observed_at:Date.now()};
    return lastMetrics;
  }
  async function closed() {
    const state = await c.show(unit); same(state, true);
    if (!['inactive','failed'].includes(state.ActiveState) || integer(state.MainPID) !== 0) return false;
    const groups = new Set([identity.cgroup,state.ControlGroup].filter(Boolean));
    for (const group of groups) {
      try {
        const events = properties((await c.read(c.groupPath(group) + '/cgroup.events','utf8')).replaceAll(' ', '='));
        if (events.populated !== '0') return false;
      } catch (error) {if (error.code !== 'ENOENT') throw error;}
    }
    return true;
  }
  async function stop() {
    if (stopped) return {closed:await closed(), identity, metrics:lastMetrics};
    same(await c.show(unit));
    await metrics();
    if (!await closed()) await c.run(['stop',unit]);
    const proof = await closed();
    if (!proof) throw new Error('Managed service descendants are not proven closed');
    stopped = true;
    return {closed:true, identity, metrics:lastMetrics};
  }
  async function wait({timeoutMs = 30000} = {}) {
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0 || timeoutMs > 86400000) throw new Error('Bounded timeoutMs required');
    const deadline = performance.now() + timeoutMs;
    while (performance.now() < deadline) {
      const state = await c.show(unit); same(state);
      await metrics();
      if (['inactive','failed'].includes(state.ActiveState) || state.SubState === 'exited') {
        const result = await stop();
        return {...result, code:integer(state.ExecMainCode) === 1 ? integer(state.ExecMainStatus) : null,
          signal:integer(state.ExecMainCode) > 1 ? integer(state.ExecMainStatus) : null, result:state.Result};
      }
      await sleep(50);
    }
    throw new Error('Managed worker wait timed out; explicit stop required');
  }
  return {child, identity, wait, stop, metrics, closed};
}
