import {spawn, execFile as nativeExecFile} from 'node:child_process';
import {promisify} from 'node:util';
import {readFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {isAbsolute} from 'node:path';
import {randomUUID, createHash} from 'node:crypto';
import {processIdentity} from '../controller/lifecycle.mjs';
import {normalizeHostMemory} from '../controller/host-memory.mjs';
import {configureHostMemory as configureDefaultHostMemory} from './host-memory.mjs';

export const controllerUnit = root => 'rsi-controller-' + createHash('sha256').update(root).digest('hex').slice(0,24) + '.service';
export function managedOptions(config) {
  const controlMemoryBytes=config.managed_runtime?.control_memory_bytes??
    Math.floor(normalizeHostMemory(config.host_memory).max_bytes/2);
  const controllerMemoryBytes=config.managed_runtime?.controller_service_bytes??Math.floor(controlMemoryBytes/4);
  if(!Number.isSafeInteger(controllerMemoryBytes)||controllerMemoryBytes<=0||controllerMemoryBytes>=controlMemoryBytes)
    throw new Error('Controller service must reserve part of control budget for KDB');
  return {hostMemory:config.host_memory,slice:config.managed_runtime?.slice??'rsi.slice',
    controlMemoryBytes,controllerMemoryBytes,knowledgeMemoryBytes:controlMemoryBytes-controllerMemoryBytes};
}
export const knowledgeUnit = root => controllerUnit(root).replace('rsi-controller-','rsi-kdb-');
export async function startManagedKnowledge(descriptor,options={}) {
  return launch(descriptor,{...options,knowledge:true,controller:false,
    memoryBytes:options.knowledgeMemoryBytes});
}
export async function startManagedMaintenance(descriptor,options={}) {
  return launch(descriptor,{...options,maintenance:true,controller:false});
}
export async function managedMaintenanceIdentity(unit,options={}) {
  if(!/^rsi-maintenance-[a-zA-Z0-9-]+\.service$/.test(unit??''))throw new Error('Owned maintenance unit required');
  const c=context(options),state=await c.show(unit);
  return {unit,invocation:state.InvocationID,...await c.membership(integer(state.MainPID),await c.aggregate())};
}
export async function observeManagedMaintenance(identity,options={}) {
  if(!/^rsi-maintenance-[a-zA-Z0-9-]+\.service$/.test(identity?.unit??'')||!identity.invocation||!identity.cgroup)
    throw new Error('Retained maintenance service identity required');
  return managedHandle(context(options),identity.unit,identity,undefined,{controller:false});
}
export async function managedKnowledgeIdentity(root,options={}) {
  const c=context(options),unit=knowledgeUnit(root),state=await c.show(unit);
  if(['inactive','failed'].includes(state.ActiveState))return null;
  if(!integer(state.MainPID))return {pending:true,unit};
  return {unit,invocation:state.InvocationID,...await c.membership(integer(state.MainPID),await c.aggregate())};
}
export async function configureHostMemory(config, options = {}) {
  // Isolated profiles are provisioned externally; only the standard shared slice is installed here.
  if (options.slice && options.slice !== 'rsi.slice') {
    await context({...options,hostMemory:config}).aggregate();
    return;
  }
  return configureDefaultHostMemory(config, options);
}

async function admissionLock(operation, options) {
  if (options.withLock) return options.withLock(operation);
  const file = join(process.env.XDG_RUNTIME_DIR ?? tmpdir(), 'rsi-managed-admission-' + process.getuid() + '.lock');
  // flock releases automatically if the controller dies: stdin closes, cat exits.
  const lock = spawn('flock',['-w','10',file,'cat'],{stdio:['pipe','pipe','ignore']});
  let timer;
  const exited = new Promise(resolve=>lock.once('close',resolve));
  try {
    await new Promise((resolve,reject)=>{
      lock.once('error',reject);
      lock.once('close',()=>reject(new Error('Managed admission lock unavailable')));
      lock.stdout.once('data',resolve);
      lock.stdin.on('error',reject);
      timer=setTimeout(()=>reject(new Error('Managed admission lock timed out')),12000);
      lock.stdin.write('locked\n');
    });
    return await operation();
  } finally {
    clearTimeout(timer);lock.stdin.end();lock.kill('SIGTERM');await exited;
  }
}

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

export async function measureManagedMemory(options = {}) {
  const c=context(options),group=await c.aggregate(),base=c.groupPath(group);
  const current_bytes=integer((await c.read(base+'/memory.current','utf8')).trim());
  const peak_bytes=integer((await c.read(base+'/memory.peak','utf8')).trim());
  if(current_bytes===null||peak_bytes===null)throw new Error('Measured aggregate cgroup memory required');
  return {current_bytes,peak_bytes,max_bytes:c.max_bytes,observed_at:Date.now()};
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
  return admissionLock(() => launchLocked(descriptor,options), options);
}
async function launchLocked(descriptor, options) {
  const c = context(options);
  let aggregate;
  try {aggregate=await c.aggregate();}
  catch(error){error.code='RSI_MEMORY_ENFORCEMENT';throw error;}
  if (!options.controller&&!options.maintenance) await c.membership(process.pid, aggregate);
  for (const pid of options.requiredPids ?? []) await c.membership(pid, aggregate);
  const memory = options.memoryBytes;
  const controllerMemory = options.controllerMemoryBytes ?? Math.floor(c.max_bytes / 2);
  const controlBudget = options.controlMemoryBytes ?? controllerMemory;
  if (!Number.isSafeInteger(controllerMemory) || controllerMemory <= 0 || controllerMemory > c.max_bytes)
    throw new Error('Controller memory cap must fit aggregate');
  if (!options.controller) {
    if (!Number.isSafeInteger(memory) || memory <= 0 || memory > (options.knowledge?controlBudget:c.max_bytes-controlBudget))
      throw new Error('Worker memoryBytes must reserve aggregate controller/KDB headroom');
    const current = integer((await c.read(c.groupPath(aggregate) + '/memory.current','utf8')).trim());
    if (current === null || current + Math.ceil(memory * 1.25) > c.max_bytes)
      throw Object.assign(new Error('Insufficient measured aggregate headroom for worker admission'),{code:'RSI_MEMORY_CAPACITY'});
  }
  const units = await c.run(['list-units','--type=service','--all','--plain','--no-legend','--no-pager',
    'rsi-controller-*.service','rsi-worker-*.service','rsi-kdb-*.service','rsi-maintenance-*.service']);
  let reserved = 0;
  for (const row of units.trim().split('\n').filter(Boolean)) {
    const existing = await c.show(row.trim().split(/\s+/)[0]);
    if (['inactive','failed'].includes(existing.ActiveState) || !existing.ControlGroup?.startsWith(aggregate + '/'))continue;
    const bytes = integer(existing.MemoryMax);
    if (bytes === null)throw new Error('Unbounded existing RSI service prevents admission');
    reserved += bytes;
  }
  if (reserved + (options.controller ? controllerMemory : memory) > c.max_bytes)
    throw Object.assign(new Error('Active managed service reservations exhaust aggregate memory capacity'),{code:'RSI_MEMORY_CAPACITY'});
  const unit = options.unit ?? 'rsi-' + (options.controller ? 'controller-' : options.knowledge?'kdb-':options.maintenance?'maintenance-':'worker-') + randomUUID() + '.service';
  if (!/^rsi-(controller|worker|kdb|maintenance)-[a-zA-Z0-9-]+\.service$/.test(unit))throw new Error('Invalid owned service name');
  // The scheduler may admit/write a job only after enforcement and reservations pass.
  const command=descriptorArgs(typeof descriptor==='function'?await descriptor():descriptor);
  const args = ['--user', '--quiet', '--no-ask-password', '--service-type=exec', '--expand-environment=no',
    '--unit=' + unit, '--slice=' + c.slice, '--property=MemoryAccounting=yes',
    '--property=MemorySwapMax=0', '--property=OOMPolicy=kill', '--property=KillMode=control-group',
    '--property=TimeoutStopSec=5s', '--property=Restart=' + (options.controller||options.knowledge ? 'on-failure' : 'no'),
    '--property=MemoryMax=' + (options.controller ? controllerMemory : memory),
    ...(options.controller||options.knowledge ? ['--property=RestartSec=1s'] : options.maintenance?[]:['--remain-after-exit']), ...command];
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
  return managedHandle(c,unit,identity,child,options);
}

/** Reattach to a retained worker after a controller crash; never resubmit it. */
export async function observeManagedWorker(identity, options = {}) {
  if(!/^rsi-worker-[a-zA-Z0-9-]+\.service$/.test(identity?.unit??'') || !identity.invocation || !identity.cgroup)
    throw new Error('Retained worker service identity required');
  const c=context(options);
  await c.membership(process.pid,await c.aggregate());
  return managedHandle(c,identity.unit,identity,undefined,{...options,controller:false});
}

function managedHandle(c,unit,identity,child,options) {
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
  async function wait({timeoutMs = 30000, signal} = {}) {
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0 || timeoutMs > 86400000) throw new Error('Bounded timeoutMs required');
    const deadline = performance.now() + timeoutMs;
    while (performance.now() < deadline) {
      signal?.throwIfAborted();
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
