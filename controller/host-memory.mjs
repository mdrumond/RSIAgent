import {execFile as nativeExecFile} from 'node:child_process';
import {promisify} from 'node:util';
import {readFile as nativeReadFile} from 'node:fs/promises';
import {randomUUID} from 'node:crypto';

export const HOST_MEMORY_SLICE = 'rsi.slice';
export function normalizeHostMemory(value = {}) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('host_memory must be an object');
  const max_bytes = value.max_bytes ?? 16 * 1024 ** 3;
  if (!Number.isSafeInteger(max_bytes) || max_bytes <= 0) throw new Error('host_memory.max_bytes must be a positive safe integer');
  return {max_bytes};
}

/** One user-wide aggregate. The operator provisions rsi.slice with MemoryAccounting=yes,
 * MemoryMax=<max_bytes>, MemorySwapMax=0. Never silently relax a shared host limit.
 * All participating controllers and KDB services must descend from this slice.
 * This controls local descendants only, not work submitted to another host.
 */
export function createHostMemorySupervisor(config = {}, {
  execFile = promisify(nativeExecFile), readFile = nativeReadFile,
} = {}) {
  const {max_bytes} = normalizeHostMemory(config);
  let admitted = false;
  const run = async (file, args) => (await execFile(file, args, {timeout:5000, maxBuffer:65536})).stdout.trim();
  const show = async unit => {
    const output = await run('systemctl', ['--user', 'show', unit,
      '--property=ActiveState,ControlGroup,MemoryAccounting,MemoryMax,MemorySwapMax']);
    return Object.fromEntries(output.split('\n').map(line => {
      const at = line.indexOf('='); return [line.slice(0, at), line.slice(at + 1)];
    }));
  };
  const number = (text, name) => {
    const value = Number(text.trim());
    if (!/^\d+$/.test(text.trim()) || !Number.isSafeInteger(value)) throw new Error(`Invalid measured ${name}`);
    return value;
  };
  async function verify() {
    const properties = await show(HOST_MEMORY_SLICE);
    const group = properties.ControlGroup;
    if (properties.ActiveState !== 'active' || properties.MemoryAccounting !== 'yes' ||
        properties.MemoryMax !== String(max_bytes) || properties.MemorySwapMax !== '0' ||
        !group?.startsWith('/') || group.split('/').includes('..') || !group.endsWith('/rsi.slice'))
      throw new Error('Host memory enforcement unavailable: provision matching user rsi.slice');
    const base = '/sys/fs/cgroup' + group;
    const controllers = await readFile('/sys/fs/cgroup/cgroup.controllers', 'utf8');
    if (!controllers.split(/\s+/).includes('memory') ||
        (await readFile(base + '/memory.max', 'utf8')).trim() !== String(max_bytes) ||
        (await readFile(base + '/memory.swap.max', 'utf8')).trim() !== '0')
      throw new Error('Host memory enforcement unavailable: cgroup v2 memory limit mismatch');
    return {group, base};
  }
  async function prepare() {
    admitted = false;
    const {group} = await verify();
    const membership = await readFile('/proc/self/cgroup', 'utf8');
    const own = membership.split('\n').find(line => line.startsWith('0::'))?.slice(3);
    if (!own?.startsWith(group + '/')) throw new Error('Controller must run inside aggregate rsi.slice before worker admission');
    admitted = true;
    return measure();
  }
  async function measure() {
    const {base} = await verify();
    const current_bytes = number(await readFile(base + '/memory.current', 'utf8'), 'memory.current');
    let peak_bytes = null;
    try {peak_bytes = number(await readFile(base + '/memory.peak', 'utf8'), 'memory.peak');}
    catch (error) {if (error.code !== 'ENOENT') throw error;}
    const events = Object.fromEntries((await readFile(base + '/memory.events', 'utf8')).trim().split('\n').map(line => {
      const [key, value] = line.split(/\s+/); return [key, number(value ?? '', 'memory.events')];
    }));
    return {slice:HOST_MEMORY_SLICE, max_bytes, current_bytes, peak_bytes, events, observed_at:Date.now()};
  }
  function commandArgs(command, args) {
    if (typeof command !== 'string' || !command || !Array.isArray(args) || args.some(arg => typeof arg !== 'string'))
      throw new Error('Executable and string argument array required');
    return ['--', command, ...args];
  }
  return {
    prepare, measure,
    // Launch the full controller here; its KDB children inherit aggregate membership.
    async controllerCommand(command, args = []) {
      await verify();
      return {command:'systemd-run', args:['--user', '--scope', '--expand-environment=no', '--slice='+HOST_MEMORY_SLICE,
        ...commandArgs(command, args)]};
    },
    async workerCommand(command, args = [], {cwd, env = {}, memoryBytes} = {}) {
      if (!admitted) throw new Error('Host memory prepare required before worker admission');
      if (!Number.isSafeInteger(memoryBytes) || memoryBytes <= 0 || memoryBytes >= max_bytes)
        throw new Error('Worker memoryBytes must be a positive safe integer below host cap; reserve controller/KDB headroom');
      if (cwd !== undefined && (typeof cwd !== 'string' || !cwd.startsWith('/'))) throw new Error('Absolute worker cwd required');
      if (!env || typeof env !== 'object' || Array.isArray(env) || Object.entries(env).some(([key,value]) => !/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) || typeof value !== 'string' || value.includes('\0'))) throw new Error('Worker env must contain string environment entries');
      await prepare(); // Recheck enforcement and controller membership on every admission.
      return {command:'systemd-run', args:['--user', '--wait', '--pipe', '--collect', '--expand-environment=no',
        '--service-type=exec', '--slice='+HOST_MEMORY_SLICE, '--unit=rsi-worker-'+randomUUID(),
        '--property=MemoryAccounting=yes', '--property=OOMPolicy=kill',
        '--property=MemoryMax='+memoryBytes, '--property=MemorySwapMax=0',
        '--property=KillMode=control-group', ...(cwd === undefined ? [] : ['--working-directory='+cwd]),
        ...Object.entries(env).map(([key,value]) => '--setenv='+key+'='+value), ...commandArgs(command, args)]};
    },
  };
}
