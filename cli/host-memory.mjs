import {execFile as nativeExecFile} from 'node:child_process';
import {promisify} from 'node:util';
import * as nativeFs from 'node:fs/promises';
import {homedir} from 'node:os';
import {join, isAbsolute} from 'node:path';
import {randomUUID} from 'node:crypto';
import {HOST_MEMORY_SLICE, normalizeHostMemory, createHostMemorySupervisor} from '../controller/host-memory.mjs';

/** Caller supplies a fresh stopped-instance check. Do not invoke from a running
 * controller. All provisioning clients serialize through the same user lock.
 */
export async function configureHostMemory(config = {}, {
  verifyInstanceStopped, execFile = promisify(nativeExecFile), fs = nativeFs,
  configHome = process.env.XDG_CONFIG_HOME || join(homedir(), '.config'),
} = {}) {
  const policy = normalizeHostMemory(config);
  if (!isAbsolute(configHome)) throw new Error('Absolute user configuration directory required');
  const run = async args => (await execFile('systemctl', ['--user', ...args],
    {timeout:5000,maxBuffer:1024*1024})).stdout.trim();
  async function show(unit) {
    const text = await run(['show',unit,'--property=ActiveState,Slice']);
    return Object.fromEntries(text.split('\n').map(line => {
      const at = line.indexOf('='); return [line.slice(0,at),line.slice(at+1)];
    }));
  }
  const verify = async () => {
    await createHostMemorySupervisor(policy,{execFile,readFile:(...args)=>fs.readFile(...args)})
      .controllerCommand('/bin/true');
    return {slice:HOST_MEMORY_SLICE,...policy};
  };
  async function active() {
    const state = (await show(HOST_MEMORY_SLICE)).ActiveState;
    if (state === 'active') return true;
    if (!['inactive','failed'].includes(state)) throw new Error('Slice state unresolved; refusing configuration');
    return false;
  }
  if (await active()) return verify(); // Matching live limits require no disk changes.
  if (typeof verifyInstanceStopped !== 'function' || await verifyInstanceStopped() !== true)
    throw new Error('Verified stopped instance required before host memory configuration');
  const directory = join(configHome,'systemd','user');
  await fs.mkdir(directory,{recursive:true,mode:0o700});
  const lock = join(directory,'.rsi-memory-config.lock');
  await fs.mkdir(lock,{mode:0o700}); // Existing lock fails closed; never remove another caller's lock.
  const destination = join(directory,HOST_MEMORY_SLICE), temporary = destination+'.'+randomUUID()+'.tmp';
  try {
    if (await active()) return await verify();
    if (await verifyInstanceStopped() !== true) throw new Error('Instance started during configuration');
    const units = await run(['list-units','--all','--plain','--no-legend','--no-pager']);
    for (const row of units.split('\n').filter(Boolean)) {
      const unit = row.trim().split(/\s+/)[0];
      if (unit === HOST_MEMORY_SLICE) continue;
      const properties = await show(unit);
      const slice = unit.endsWith('.slice') ? unit : properties.Slice;
      if ((unit.startsWith('rsi-') || slice === HOST_MEMORY_SLICE ||
          (slice?.startsWith('rsi-') && slice.endsWith('.slice'))) &&
          !['inactive','failed'].includes(properties.ActiveState))
        throw new Error('Other active RSI unit prevents host memory configuration: '+unit);
    }
    if (await active()) return await verify();
    await fs.writeFile(temporary, '[Unit]\nDescription=RSI aggregate host memory\n\n[Slice]\n'+
      'MemoryAccounting=yes\nMemoryMax='+policy.max_bytes+'\nMemorySwapMax=0\n', {mode:0o600,flag:'wx'});
    // Check again after filesystem IO, before replacing the shared configuration.
    if (await active()) return await verify();
    await fs.rename(temporary,destination);
    await run(['daemon-reload']);
    await run(['start',HOST_MEMORY_SLICE]);
    return await verify();
  } finally {
    try {await fs.rm(temporary,{force:true});} finally {await fs.rmdir(lock);}
  }
}
