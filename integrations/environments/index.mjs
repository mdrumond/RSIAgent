import { readFile, readdir, mkdtemp, writeFile, mkdir, rename, realpath, lstat, readlink, rm,copyFile,link } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { homedir } from 'node:os';
import { join, resolve, relative, isAbsolute, posix, dirname } from 'node:path';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import {createLocalExecutionOperations,isPiMaximumTimeoutRejection} from './local-bash.mjs';
import {excludedExecutionProofs} from '../../controller/execution-proof.mjs';
import {localHostIdentity,observeRebootClosure} from '../../controller/local-reboot.mjs';

const quote = value => "'" + String(value).replaceAll("'", "'\\''") + "'";
const terminal = new Set(['completed', 'failed', 'cancelled']);
const normalize = result => ({ state: terminal.has(result.state) ? 'stopped' :
  ['queued', 'dispatching', 'running', 'retrying', 'cancelling'].includes(result.state) ? 'running' : 'unknown',
  ...(terminal.has(result.state) ? {nativeState:result.state} : {}),
  ...(Number.isInteger(result.exit) ? { exitCode: result.exit } : {}),
  ...(result.reason == null ? {} : {reason: result.reason}),
  ...(Number.isFinite(result.executionSeconds) && result.executionSeconds >= 0 ? { executionSeconds: result.executionSeconds } : {}) });

export function closedSetupError(message, {handle, reason, notAdmitted = false} = {}) {
  return Object.assign(new Error(message + (reason == null ? '' : `; reason=${JSON.stringify(reason)}`)), {
    code: notAdmitted ? 'CPL_NOT_ADMITTED' : 'SETUP_CLOSED',
    closure: {closed: true, payloadStarted: false, handles: handle ? [handle] : [], reason: reason ?? null},
  });
}
// Strip SDK-inherited spawn values, retaining explicit deltas and session identity.
const targetEnvironment = env => Object.fromEntries(Object.entries(env ?? {}).filter(([key,value]) =>
  key !== 'PI_SESSION_FILE' && !(key.toUpperCase() === 'PATH' && ('PI_SESSION_ID' in (env ?? {}))) && process.env[key] !== value));
async function sourceSnapshot(path) {
  const files = [], entries = [];
  let cacheable = true;
  async function walk(directory, prefix = '') {
    const mode = (await lstat(directory)).mode & 0o777;
    entries.push([prefix, 'directory', mode]);
    const children = await readdir(directory, { withFileTypes: true });
    children.sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
    for (const entry of children) {
      const name = posix.join(prefix, entry.name), file = join(directory, entry.name);
      if (entry.isDirectory()) { await walk(file, name); continue; }
      files.push(name);
      const stat = await lstat(file);
      if (entry.isFile()) {
        const hash = createHash('sha256');
        for await (const data of createReadStream(file)) hash.update(data);
        entries.push([name, 'file', stat.mode & 0o777, hash.digest('hex')]);
      } else {
        // A retained symlink can resolve to mutable remote data; keep fresh-transfer behavior.
        cacheable = false;
        entries.push([name, 'other', stat.mode & 0o777, entry.isSymbolicLink() ? await readlink(file) : null]);
      }
    }
  }
  await walk(path);
  return { files, entries, cacheable, fingerprint: createHash('sha256').update(JSON.stringify(entries)).digest('hex') };
}
// A proc file opened before reaping can fail its subsequent read with ESRCH.
export async function readProcessStat(pid, read = readFile) {
  try { return await read(`/proc/${pid}/stat`, 'utf8'); }
  catch (error) {
    if (error.code === 'ENOENT' || error.code === 'ESRCH') return null;
    throw error;
  }
}
const json = async path => JSON.parse(await readFile(path, 'utf8'));
async function save(path, value) {
  const temporary=path+'.'+randomUUID()+'.tmp';
  await writeFile(temporary,JSON.stringify(value),{mode:0o600});
  await rename(temporary,path);
}

// Only the installed public CLI owns remote protocols and credentials.
function nativeFailureDiagnostic(stderr) {
  // Native stderr can contain routes, usernames or credentials. Persist only
  // recognised failure categories, never the native text itself.
  const categories = [
    ['connection-reset', /connection reset|broken pipe/i],
    ['connection-refused', /connection refused/i],
    ['connection-timeout', /connection timed out|operation timed out|timeout in data send/i],
    ['connection-closed', /connection closed|connection unexpectedly closed/i],
    ['authentication-failed', /permission denied.*publickey|authentication failed/i],
    ['host-key-failed', /host key verification failed|remote host identification has changed/i],
    ['name-resolution-failed', /could not resolve hostname|name or service not known/i],
    ['remote-tool-missing', /rsync.*(?:command not found|not found)/i],
    ['rsync-protocol-error', /protocol version mismatch|protocol incompatibility/i],
    ['remote-path-missing', /no such file or directory/i],
    ['remote-permission-denied', /permission denied/i],
    ['remote-storage-full', /no space left on device|disk quota exceeded/i],
  ].filter(([, pattern]) => pattern.test(stderr)).map(([category]) => category);
  const rsyncCode = stderr.match(/rsync error:[^\n]*\(code (\d+)\)/i);
  return {categories: categories.length ? categories : ['native-error-output'],
    ...(rsyncCode ? {rsyncExit:Number(rsyncCode[1])} : {})};
}
export function createCplClient(executable = join(homedir(), '.agents/skills/remote-access/scripts/cpl-remote')) {
  return async (args,{signal}={}) => {
    await readFile(join(homedir(), '.agents/skills/remote-access/SKILL.md'), 'utf8');
    return new Promise((resolve, reject) => {
      const child = spawn(executable, ['--json', ...args], { stdio: ['ignore', 'pipe', 'pipe'],signal });
      let output = '', error = '';
      child.stdout.on('data', data => { output += data; });
      child.stderr.on('data', data => { error = (error + data).slice(-16384); });
      child.on('error', reject);
      child.on('close', code => {
        try {
          const result = JSON.parse(output);
          if (error && (code !== 0 || result.state === 'failed'))
            result.diagnostic = nativeFailureDiagnostic(error);
          resolve(result);
        }
        catch {
          const diagnostic = nativeFailureDiagnostic(error);
          reject(Object.assign(new Error(`CPL ${args[0]} failed (${code}): ${diagnostic.categories.join(', ')}`),
            {diagnostic}));
        }
      });
    });
  };
}

// The payload supervisor enforces workload elapsed time, not CLI observation time.
const supervisor = String.raw`import hashlib, json, os, signal, subprocess, sys, time
c=json.loads(sys.argv[1]); os.makedirs(c['spool'],exist_ok=True)
started=time.monotonic()
outcome={'kind':'command'}
def intact():
 try:
  return all(hashlib.sha256(open(path,'rb').read()).hexdigest()==digest for path,digest in c['verifiers'])
 except OSError: return False
with open(c['spool']+'/stdout','wb') as out, open(c['spool']+'/stderr','wb') as err:
 if not intact():
  err.write(b'Frozen verifier changed before execution\n'); code=125
  outcome={'kind':'infrastructure','reason':'frozen-verifier-before'}
 else:
  p=subprocess.Popen(['bash','-c',c['command']],cwd=c['cwd'],env={**os.environ,**c['env']},stdout=out,stderr=err,start_new_session=True)
  def stop(*args):
   try: os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError: pass
  signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
  try: code=p.wait(timeout=c['timeout'])
  except subprocess.TimeoutExpired:
   stop(); p.wait(); code=124; outcome={'kind':'infrastructure','reason':'elapsed-timeout'}
  finally: stop()
  if not intact():
   err.write(b'Frozen verifier changed after execution\n'); code=125
   outcome={'kind':'infrastructure','reason':'frozen-verifier-after'}
with open(c['spool']+'/exit','w') as f: f.write(str(code))
with open(c['spool']+'/timing.json.tmp','w') as f:
 json.dump({'completed':True,'executionSeconds':time.monotonic()-started,'key':c['timingKey'],'outcome':outcome},f); f.flush(); os.fsync(f.fileno())
os.replace(c['spool']+'/timing.json.tmp',c['spool']+'/timing.json')
if c.get('inlineResult'):
 with open(c['spool']+'/timing.json') as f: result=json.load(f)
 result.update(protocol='rsi-command-result-1',exitCode=code if code>=0 else 128-code,inlineOutput=False,admission=None)
 try:
  if os.path.getsize(c['spool']+'/admission.json')<=4096:
   with open(c['spool']+'/admission.json') as f: result['admission']=json.load(f)
 except (OSError,ValueError): pass
 if sum(os.path.getsize(c['spool']+'/'+name) for name in ['stdout','stderr'])<=32768:
  try:
   for name in ['stdout','stderr']:
    with open(c['spool']+'/'+name,encoding='utf8',newline='') as f: result[name]=f.read()
   result['inlineOutput']=len(json.dumps(result,ensure_ascii=False).encode())<=65536
  except UnicodeError: pass
  if not result['inlineOutput']:
   result.pop('stdout',None); result.pop('stderr',None)
 print(json.dumps(result,ensure_ascii=False),flush=True)
sys.exit(code if code>=0 else 128-code)
`;

const synchronize = `import errno, hashlib, json, os, shutil, stat, sys
with open(sys.argv[1],'rb') as manifest_input: manifest_bytes=manifest_input.read()
if hashlib.sha256(manifest_bytes).hexdigest()!=sys.argv[2]: raise ValueError('Source manifest identity mismatch')
c=json.loads(manifest_bytes)
if not os.path.isdir(c['incoming']): raise FileNotFoundError('Retained incoming source snapshot is missing')
if c.get('snapshot') is not None:
 expected={entry[0]:entry[1:] for entry in c['snapshot']}
 root_info=os.lstat(c['incoming'])
 actual={'':['directory' if stat.S_ISDIR(root_info.st_mode) else 'other',root_info.st_mode & 0o777]}
 for directory, directories, files in os.walk(c['incoming'],followlinks=False):
  for name in directories+files:
   path=os.path.join(directory,name); relative=os.path.relpath(path,c['incoming']); info=os.lstat(path)
   if stat.S_ISDIR(info.st_mode): actual[relative]=['directory',info.st_mode & 0o777]
   elif stat.S_ISREG(info.st_mode):
    digest=hashlib.sha256()
    with open(path,'rb') as content:
     for block in iter(lambda:content.read(1024*1024),b''): digest.update(block)
    actual[relative]=['file',info.st_mode & 0o777,digest.hexdigest()]
   else: actual[relative]=['other']
 if actual!=expected: raise ValueError('Retained incoming source snapshot identity mismatch')
os.makedirs(c['project'],exist_ok=True)
manifest=c['project']+'/.environment-source-files.json'
try:
 with open(manifest) as f: previous=json.load(f)
except FileNotFoundError: previous=[]
for name in set(previous)-set(c['files']):
 try: os.unlink(c['project']+'/'+name)
 except FileNotFoundError: pass
parents=set()
for name in previous:
 parent=os.path.dirname(name)
 while parent: parents.add(parent); parent=os.path.dirname(parent)
for name in sorted(parents,key=lambda p:p.count('/'),reverse=True):
 try: os.rmdir(c['project']+'/'+name)
 except OSError as error:
  if error.errno==errno.ENOENT: continue
  if error.errno in (errno.ENOTEMPTY,errno.EEXIST) and name not in c['files']: continue
  raise
# Replace physical entries, never write through an old destination symlink.
for directory, directories, files in os.walk(c['incoming'],followlinks=False):
 for name in directories+files:
  source=os.path.join(directory,name)
  destination=os.path.join(c['project'],os.path.relpath(source,c['incoming']))
  if os.path.islink(destination): os.unlink(destination)
  elif os.path.islink(source) and os.path.lexists(destination):
   if os.path.isdir(destination): os.rmdir(destination)
   else: os.unlink(destination)
def restore_file(source,destination):
 # Retained incoming files were verified above. Preserve unchanged build inputs
 # so compiler caches depending on source timestamps/inodes remain useful.
 if os.path.isfile(destination) and not os.path.islink(destination):
  before=os.stat(source); after=os.stat(destination)
  if before.st_size==after.st_size and stat.S_IMODE(before.st_mode)==stat.S_IMODE(after.st_mode):
   def digest(path):
    result=hashlib.sha256()
    with open(path,'rb') as content:
     for block in iter(lambda:content.read(1024*1024),b''): result.update(block)
    return result.digest()
   if digest(source)==digest(destination): return destination
 return shutil.copy2(source,destination)
shutil.copytree(c['incoming'],c['project'],dirs_exist_ok=True,symlinks=True,copy_function=restore_file)
with open(manifest,'w') as f: json.dump(c['files'],f)
`;

export async function createEnvironment(options, context) {
  const sdk = context.sdk ?? await import('@earendil-works/pi-coding-agent');
  const event = (type, data) => context.event(type, data);
  const exclude = (label, fn) => context.exclude(label, fn);
  const root = join(context.instancePath, '.executions');
  await mkdir(root, { recursive: true, mode: 0o700 });
  const nativeClient = context.client ?? createCplClient(options.clientPath);
  const client = (args, {signal = context.signal} = {}) => {
    signal?.throwIfAborted();
    return nativeClient(args, {signal});
  };
  const local = createLocalExecutionOperations(sdk,options.localExecution,{signal:context.stopSignal});
  const remote = options.mode === 'cpl';
  if (!remote && options.mode !== 'local') throw new Error('Environment mode must be local or cpl');
  let projectRoot, attemptRoot, projectLayout = 'project', frozen = [];
  let capabilities;
  async function discoverCapabilities(signal=context.signal) {
    if(!capabilities) {
      const result=await client(['capabilities',options.target],{signal});
      if(result.state!=='available')throw closedSetupError('CPL target unavailable',{reason:result});
      capabilities=result;
    }
    return capabilities;
  }
  const executionReceipts = new Map(), terminalAttempts = new Set();
  const uploadParents = new Set();
  const instructions = (options.instructionFile ? await exclude('setup', () => readFile(options.instructionFile, 'utf8')) : '') +
    '\nNative bash/read/write/edit operate in the LOCAL project; use them for persistent source edits. rsi_target executes commands in the configured target environment.' +
    (remote ? ' Target commands use a synchronized execution snapshot: local edits sync first, and target-side source edits are overwritten on the next synchronization. Use project-relative paths with rsi_target.' : '');
  const check = handle => {
    if (remote && !handle.startsWith(`remote:${options.target}:`)) throw new Error('Handle belongs to another target');
    if (!remote && !/^local:exec-[A-Za-z0-9]+$/.test(handle)) throw new Error('Invalid local handle');
  };
  async function localState(handle,{signal}={}) {
    check(handle);
    const dir = join(root, handle.slice(6));
    try {
      const receipt = await json(join(dir, 'execution.json'));
      const host=await localHostIdentity();
      if (receipt.boot !== host.boot) return await observeRebootClosure(dir,receipt,host)??{state:'unknown'};
      // Only canonical evaluation setup opts into this legacy prelaunch proof.
      // Preserve the refusal: it proves no process, not a successful command.
      if(context.recoverPiTimeoutRejection===true&&context.sdk===undefined&&
          Object.keys(receipt).every(key=>['boot','cwd','failure','outcome'].includes(key))&&
          typeof receipt.cwd==='string'&&receipt.cwd&&isPiMaximumTimeoutRejection(receipt.failure)&&
          receipt.outcome?.kind==='infrastructure'&&receipt.outcome.reason==='local-execution-interrupted'&&
          (await readdir(dir)).every(name=>name==='execution.json'))
        return {state:'stopped',outcome:receipt.outcome};
      if(receipt.prelaunch?.closed===true&&receipt.prelaunch.payloadStarted===false)
        return {state:'stopped',...(receipt.outcome?{outcome:receipt.outcome}:{})};
      if(receipt.containment) {
        if(!receipt.containment.closed) {
          const state=await local.observe(receipt.containment.unit,{signal});
          if(state.state!=='stopped')return state;
        }
        // Cleanup can finish before the worker persists its closed receipt.
        // Same-boot OS scope closure is also positive descendant-closure proof.
        let exitCode;try{exitCode=Number(await readFile(join(dir,'exit'),'utf8'));}catch{}
        return {state:'stopped',...(Number.isInteger(exitCode)?{exitCode}:{}),...(receipt.outcome?{outcome:receipt.outcome}:{})};
      }
      const original = (await readFile(join(dir, 'identity'), 'utf8')).trim();
      const pid = Number(original.slice(0, original.indexOf(' ')));
      const fields = text => text.slice(text.lastIndexOf(')') + 2).split(' ');
      const current = await readProcessStat(pid);
      if (current && fields(current)[19] !== fields(original)[19]) return { state: 'unknown' };
      let live = false;
      for (const name of await readdir('/proc')) {
        if (!/^\d+$/.test(name)) continue;
        try {
          const processStat = await readProcessStat(name);
          if (processStat === null) continue;
          const stat = fields(processStat);
          if (Number(stat[2]) === pid && !['Z', 'X'].includes(stat[0])) live = true;
        } catch (error) { if (error.code !== 'ENOENT' && error.code !== 'ESRCH') return { state: 'unknown' }; }
      }
      let exitCode;
      try { exitCode = Number(await readFile(join(dir, 'exit'), 'utf8')); } catch { /* interrupted wrapper */ }
      return { state: live ? 'running' : 'stopped', pid, ...(Number.isInteger(exitCode) ? { exitCode } : {}),
        ...(receipt.outcome?{outcome:receipt.outcome}:{}) };
    } catch { return { state: 'unknown' }; }
  }
  async function controllerTiming(receipt, state, legacy) {
    // A persisted monotonic clock survives controller restart on the same host.
    // Wall time is a conservative fallback across boots and for legacy journals.
    Object.assign(receipt,await json(join(receipt.dir,'execution.json')));
    if (!receipt.launch && Number.isFinite(legacy?.wallMs)) receipt.launch={wallMs:legacy.wallMs};
    if (!Number.isFinite(receipt.launch?.wallMs)) return null;
    const terminalPath=join(receipt.dir,'controller-terminal-timing.json');
    let frozen;try {frozen=await json(terminalPath);} catch(error) {if(error.code!=='ENOENT')throw error;}
    const previous=frozen??receipt.controllerTiming;
    const terminalBound=Number.isFinite(legacy?.terminalWallMs)&&legacy.terminalWallMs>=receipt.launch.wallMs;
    let seconds=previous?.executionSeconds??0;
    if (!previous?.terminal) {
      let duration=Math.max(0,(Date.now()-receipt.launch.wallMs)/1000);
      if (terminalBound) duration=Math.max(0,(legacy.terminalWallMs-receipt.launch.wallMs)/1000);
      else if (receipt.launch.monotonicNs && receipt.launch.boot===
          (await readFile('/proc/sys/kernel/random/boot_id','utf8')).trim())
        duration=Math.max(0,Number(process.hrtime.bigint()-BigInt(receipt.launch.monotonicNs))/1e9);
      seconds=Math.max(seconds,duration);
    }
    let timing={executionSeconds:seconds,terminal:previous?.terminal===true||state==='stopped'||terminalBound};
    if(timing.terminal && !frozen) {
      // Publish a complete immutable cutoff: a concurrent late observation must
      // not overwrite the first durable terminal bound.
      const candidate=terminalPath+'.'+randomUUID();
      try {
        await writeFile(candidate,JSON.stringify(timing),{mode:0o600});
        try {await link(candidate,terminalPath);} catch(error) {if(error.code!=='EEXIST')throw error;}
        timing=await json(terminalPath);
      } finally {await rm(candidate,{force:true});}
    }
    receipt.controllerTiming=timing;
    const {dir,...record}=receipt;await save(join(dir,'execution.json'),record);
    return {executionSeconds:timing.executionSeconds,timingSource:'controller-launch',timingEstimated:true};
  }
  // Native terminal observation plus the wrapper's nonce permits replacing
  // several small-file transfers; absent/older results retain file collection.
  const validTerminal = (data,receipt) => data?.protocol==='rsi-command-result-1' &&
    data.completed===true && data.key===receipt.timingKey && Number.isInteger(data.exitCode) &&
    Number.isFinite(data.executionSeconds) && data.executionSeconds>=0 &&
    ['command','infrastructure'].includes(data.outcome?.kind) &&
    typeof data.inlineOutput==='boolean' && Object.hasOwn(data,'admission') &&
    (data.admission===null || typeof data.admission==='object' && !Array.isArray(data.admission)) &&
    (!data.inlineOutput || typeof data.stdout==='string' && typeof data.stderr==='string');
  async function cachedTerminal(receipt) {
    try {const data=await json(join(receipt.dir,'terminal-result.json'));return validTerminal(data,receipt)?data:null;}
    catch {return null;}
  }
  async function terminalResult(handle,receipt,state,signal) {
    if(state!=='stopped')return null;
    const cached=await cachedTerminal(receipt);if(cached)return cached;
    if(terminalAttempts.has(handle))return null;terminalAttempts.add(handle);
    try {
      const response=await client(['logs',handle,'--stream','stdout','--tail','1'],{signal});
      if(typeof response.content!=='string'||Buffer.byteLength(response.content)>65536)return null;
      const data=JSON.parse(response.content.trim());if(!validTerminal(data,receipt))return null;
      await save(join(receipt.dir,'timing.json'),data);
      if(data.admission!==null)await save(join(receipt.dir,'admission.json'),data.admission);
      if(data.inlineOutput)for(const stream of ['stdout','stderr'])await writeFile(join(receipt.dir,stream),data[stream],{mode:0o600});
      await save(join(receipt.dir,'terminal-result.json'),data);
      return data;
    }catch{return null;}
  }
  async function executionTiming(handle, signal, state, legacy) {
    if (handle.includes(':transfer:')) return { executionSeconds: 0, accountingExcluded: true };
    let receipt = executionReceipts.get(handle);
    if (!receipt) for (const name of await readdir(root)) {
      const dir = join(root, name);
      try {
        const execution = await json(join(dir, 'execution.json'));
        if (execution.handle === handle || (!execution.handle && execution.receiptFile && (await json(execution.receiptFile)).handle === handle)) receipt = { ...execution, dir };
      } catch { /* another operation */ }
      if (!receipt) for(const name of ['sync-submission.json','upload-parent-submission.json'])try {
        if ((await json(join(dir, name))).handle === handle) return { executionSeconds: 0, accountingExcluded: true };
      } catch { /* not a setup receipt */ }
      if (receipt) { executionReceipts.set(handle, receipt); break; }
    }
    // Freeze the first terminal observation before potentially slow downloads.
    let fallback;
    try {fallback=receipt?await controllerTiming(receipt,state,legacy):null;} catch { /* accounting failure cannot change native process status */ }
    if (receipt?.timingKey && receipt.spool) {
      await terminalResult(handle,receipt,state,signal);
      if(state==='stopped' && receipt.admissionToken){
        let admission;try {admission=await json(join(receipt.dir,'admission.json'));} catch {}
        if(admission?.token!==receipt.admissionToken)admission=await readAdmission(handle,{signal});
        if(admission?.token===receipt.admissionToken&&admission.decision==='contended')return {executionSeconds:0,accountingExcluded:true};
      }
      const path = join(receipt.dir, 'timing.json');
      const valid=timing=>timing?.completed===true && timing.key===receipt.timingKey &&
        Number.isFinite(timing.executionSeconds) && timing.executionSeconds>=0;
      try {
        let timing;try {timing=await json(path);} catch {}
        if(!valid(timing) && state==='stopped') {
          const dir = await mkdtemp(join(root, 'timing-'));
          await exclude('transfer', () => transfer(['download', options.target, posix.join(receipt.spool, 'timing.json'), path], dir, signal));
          timing = await json(path);
        }
        if(valid(timing))return {executionSeconds:timing.executionSeconds,timingSource:'wrapper',
          ...(['command','infrastructure'].includes(timing.outcome?.kind)?{outcome:timing.outcome}:{})};
      } catch { /* launch accounting remains available if the receipt cannot be recovered */ }
    }
    if(fallback)return fallback;
    await event('accounting-unresolved', { handle, message: 'No completed same-host execution duration or controller launch timestamp' });
    return { accountingUnresolved: true };
  }
  async function observe(handle, {signal = context.signal,accountingExcluded = false,launch} = {}) {
    check(handle);
    let state;
    try {state = remote ? normalize(await client(['observe', handle], {signal})) : await localState(handle,{signal});}
    catch {state={state:'unknown'};}
    if(remote) {
      if(accountingExcluded===true)Object.assign(state,{executionSeconds:0,accountingExcluded:true});
      else if(state.executionSeconds===undefined)Object.assign(state,await executionTiming(handle,signal,state.state,launch));
    }
    await event('execution-state', { handle, ...state });
    return state;
  }
  async function stop(handle, {accountingExcluded=false,signal:stopSignal=context.stopSignal}={}) {
    check(handle);
    // A supplied cleanup signal owns its bounded allowance, independently of
    // the interrupted body. Callers must bound it; absent one, allow five seconds.
    const signal = stopSignal??AbortSignal.timeout(5000);
    if (remote) {
      try { await client(['cancel', handle], {signal}); } catch { /* observe is authoritative */ }
    } else {
      const receipt=await json(join(root,handle.slice(6),'execution.json'));
      // Scope names and process groups belong to their original boot. An old
      // command is observed through custody proof, never stopped on this boot.
      if(receipt.boot!==(await localHostIdentity()).boot)return observe(handle,{signal,accountingExcluded});
      if(receipt.containment) {
        await local.stop(receipt.containment.unit,{signal});
        await save(join(root,handle.slice(6),'execution.json'),{...receipt,containment:{...receipt.containment,closed:true}});
        return observe(handle,{accountingExcluded});
      }
      const state = await localState(handle,{signal});
      if (state.state === 'running') {
        try { process.kill(-state.pid, 'SIGKILL'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
      }
    }
    while (!signal.aborted) {
      const state = await observe(handle, {signal,accountingExcluded});
      if (state.state === 'stopped') return state;
      try { await delay(50,undefined,{signal}); } catch { break; }
    }
    return { state: 'unknown' };
  }
  async function submit(args, receipt, allowCompleted = false, accountingExcluded = false, signal = context.signal, receiptSignal = context.stopSignal) {
    const rejected = reason => {
      const error = closedSetupError(`CPL ${args[0]} was not admitted`, {notAdmitted:true,reason});
      // A refused artifact download says nothing about earlier payload closure.
      if (args[0] === 'download') delete error.closure;
      return error;
    };
    if (signal?.aborted) throw Object.assign(rejected(), {cause: signal.reason});
    const operation = randomUUID();
    await event('execution-submitting', { submission: operation, receiptFile: receipt ?? null, kind: args[0], accountingExcluded });
    let result, cause;
    if(signal?.aborted){await event('execution-not-admitted',{submission:operation,closed:true});throw Object.assign(rejected(),{cause:signal.reason});}
    // Admission follows the actor clock; dispatched run receipts use the caller's
    // independent bounded cleanup allowance before cancellation observes the ID.
    try { result = await client(args, {signal:args[0]==='run'&&receiptSignal?receiptSignal:signal}); }
    catch (error) {
      cause = error;
      if (receipt) try { result = await json(receipt); } catch { /* absence is not closure */ }
    }
    if (result?.handle || (allowCompleted && result?.state === 'completed')) {
      await event('execution-submitted', { submission: operation, handle: result.handle ?? null, closed: !result.handle, accountingExcluded });
      return result;
    }
    if(allowCompleted&&capabilities?.backend==='ssh'&&['upload','download'].includes(args[0])&&
      terminal.has(result?.state)&&Number.isInteger(result.exit)&&!result.handle) {
      await event('execution-submitted',{submission:operation,handle:null,closed:true,accountingExcluded,reason:result});
      const error=closedSetupError(`CPL ${args[0]} failed with exit ${result.exit}`,{reason:result});
      if(args[0]==='download')delete error.closure;
      throw error;
    }
    if (result?.admitted === false || result?.submitted === false) {
      await event('execution-not-admitted', { submission: operation, closed: true });
      throw rejected(result.reason);
    }
    const message = `CPL ${args[0]} submission identity unknown; do not resubmit`;
    await event('execution-unresolved', { operation, submission: operation, handle: null, message });
    throw new Error(message, { cause: cause ?? result });
  }
  async function prepareUploadParent(destination,dir,signal) {
    const path=posix.normalize(destination);
    if(posix.isAbsolute(path)||!attemptRoot||!path.startsWith(attemptRoot+'/'))
      throw closedSetupError('Upload destination must remain within the prepared attempt');
    const parent=posix.dirname(path);
    if(uploadParents.has(parent))return;
    const file=join(dir,'upload-parent.sh'),receipt=join(dir,'upload-parent-submission.json');
    await writeFile(file,`mkdir -p -- ${quote(posix.join(options.workspacePath,parent))}\n`,{mode:0o600});
    const args=['run',options.target,'--file',file,'--receipt-file',receipt,'--cwd',options.workspacePath];
    if(options.runtime)args.push('--runtime',options.runtime);
    const submitted=await submit(args,receipt,false,true,signal);
    await event('execution-handle',{handle:submitted.handle,purpose:'setup'});
    let state;
    try {
      do {
        state=await observe(submitted.handle,{signal,accountingExcluded:true});
        signal?.throwIfAborted();
        if(state.state==='running')await delay(options.pollMs??250,undefined,{signal});
      }while(state.state==='running');
    }catch(error) {
      if(signal?.aborted)state=await stop(submitted.handle,{accountingExcluded:true});
      if(state?.state==='stopped')throw closedSetupError('Upload parent setup interrupted',{handle:submitted.handle,reason:state});
      throw error;
    }
    if(state.state!=='stopped')throw new Error(`Upload parent setup unresolved; observe ${submitted.handle}`);
    if(state.exitCode!==0)throw closedSetupError('Upload parent setup failed',{handle:submitted.handle,reason:state});
    uploadParents.add(parent);
  }
  async function transfer(args, dir, signal = context.signal) {
    const upload = args[0] === 'upload';
    const synchronous=(await discoverCapabilities()).backend==='ssh';
    if(synchronous&&upload) {
      if((await lstat(args[2])).isDirectory()&&!args[2].endsWith('/'))args[2]+='/';
      await prepareUploadParent(args[3],dir,signal);
    }
    if (!synchronous) {
      if (upload) args.push('--receipt-file', join(dir, 'transfer.json'), '--no-wait');
      else args.push('--timeout', '0');
    }
    let result = await submit(args, upload&&!synchronous ? join(dir, 'transfer.json') : undefined, true, true, signal);
    if(synchronous&&(result.state!=='completed'||result.handle))
      throw new Error('Synchronous transfer unresolved; retain any returned handle before retry');
    const handle = result.handle;
    if (handle) {
      await event('execution-handle', { handle, purpose: 'transfer' });
      await event('execution-state', { handle, purpose: 'transfer', ...normalize(result) });
      try {
        // An initial timeout marker still has an observable native identity.
        if (!terminal.has(result.state)) result = await client(['observe', handle], {signal});
        while (!terminal.has(result.state)) {
          signal?.throwIfAborted();
          if (!['queued', 'dispatching', 'running', 'retrying'].includes(result.state)) throw new Error(`Transfer unresolved; observe ${handle}`);
          result = await client(['observe', handle], {signal});
          if (!terminal.has(result.state)) await delay(options.pollMs ?? 250, undefined, {signal});
        }
        if (!upload && result.state === 'completed') await client(['fetch', handle, args[3]], {signal});
      } catch (error) {
        if (signal?.aborted) await stop(handle);
        throw error;
      }
    }
    if (result.state !== 'completed') {
      if (upload && handle && terminal.has(result.state))
        throw closedSetupError(`Transfer did not complete: ${handle}`, {handle, reason: result});
      throw new Error(`Transfer did not complete: ${handle ?? result.state}`);
    }
    return result;
  }
  async function execute(command, cwd, opts = {}) {
    opts = {...opts,stopSignal:opts.stopSignal??context.stopSignal, signal: context.signal && opts.signal
      ? AbortSignal.any([context.signal, opts.signal]) : opts.signal ?? context.signal};
    const signal = opts.signal;
    const started = performance.now();
    let excludedMs = 0;
    const copy = async (label, fn) => {
      const before = performance.now();
      try { return await exclude(label, fn); } finally { excludedMs += performance.now() - before; }
    };
    const remaining = () => opts.timeout === undefined ? undefined : Math.max(0, opts.timeout - (performance.now() - started - excludedMs) / 1000);
    if (opts.signal?.aborted || remaining() === 0) throw new Error('Execution cancelled or elapsed budget exhausted');
    const dir = await mkdtemp(join(root, 'exec-'));
    let handle;
    if (!remote) {
      handle = `local:${dir.slice(dir.lastIndexOf('/') + 1)}`;
      const hostIdentity=await localHostIdentity(),receipt={boot:hostIdentity.boot,hostIdentity,cwd};
      await save(join(dir,'execution.json'),receipt);
      if (opts.leaseReceipt) opts = {...opts, env:{...opts.env, RSI_LEASE_RECEIPT:join(dir,'admission.json')}};
      await event('execution-handle', { handle });
      const wrapper = `cat /proc/$$/stat > ${quote(join(dir, 'identity.tmp'))}\nmv ${quote(join(dir, 'identity.tmp'))} ${quote(join(dir, 'identity'))}\n` +
        `trap ${quote(`code=$?; printf "%s" "$code" > ${quote(join(dir, 'exit.tmp'))}; mv ${quote(join(dir, 'exit.tmp'))} ${quote(join(dir, 'exit'))}`)} EXIT\nbash -c ${quote(command)}`;
      try {
        const result=await local.exec(wrapper,cwd,{...opts,env:opts.env?{...process.env,...opts.env}:undefined,
          timeout:remaining(),onData:opts.onData??(()=>{}),
          onContainment:async containment=>save(join(dir,'execution.json'),{...receipt,containment})});
        await writeFile(join(dir,'exit'),String(result.exitCode));
        await save(join(dir,'execution.json'),{...await json(join(dir,'execution.json')),
          outcome:result.outcome});
      }
      catch (error) {
        const failure={message:error.message,code:error.code??null,syscall:error.syscall??null};
        const prelaunch=error.localPrelaunch;
        const outcome={kind:'infrastructure',reason:prelaunch?.closed===true&&prelaunch.payloadStarted===false
          ?'local-execution-not-started':'local-execution-interrupted'};
        await save(join(dir,'execution.json'),{...await json(join(dir,'execution.json')),failure,outcome,
          ...(prelaunch?{prelaunch}:{})});
        await event('execution-error',{handle,...failure,...(prelaunch?{prelaunch}:{})});
        if(outcome.reason==='local-execution-not-started')return {handle,exitCode:null,settled:true,outcome};
        const state = await stop(handle,{signal:opts.stopSignal});
        if (state.state !== 'stopped') throw Object.assign(new Error(`Local execution unresolved: ${handle}: ${error.code??'execution-error'}: ${error.message}`, { cause: error }),{handle});
        return {handle,exitCode:state.exitCode??null,settled:true,
          outcome:{kind:'infrastructure',reason:'local-execution-interrupted'}};
      }
      const state = await localState(handle);
      if (state.state === 'running') await stop(handle);
      const final = await observe(handle);
      return {handle,exitCode:final.exitCode??null,settled:final.state==='stopped',...(final.outcome?{outcome:final.outcome}:{})};
    }
    if (!projectRoot || !attemptRoot) throw new Error('Call prepare with project.sourceDirectory and attempt.id before remote execute');
    const subdir = relative(projectRoot, resolve(cwd ?? projectRoot));
    if (subdir === '..' || subdir.startsWith('../') || isAbsolute(subdir)) throw new Error('Target execution cwd must be inside local project');
    const runRoot = posix.join(attemptRoot, dir.slice(dir.lastIndexOf('/') + 1));
    const destination = posix.join(attemptRoot, projectLayout);
    const cachePath = join(root, 'source-staging.json');
    await copy('transfer', async () => {
      const snapshot = await sourceSnapshot(projectRoot);
      const key = createHash('sha256').update(JSON.stringify([options.target, options.runtime,
        options.workspacePath, attemptRoot, projectRoot, projectLayout, snapshot.fingerprint])).digest('hex');
      let retained;
      try { retained = await json(cachePath); } catch (error) { if (error.code !== 'ENOENT') throw error; }
      // An interrupted synchronization is observed through its existing native receipt, never replayed.
      if (retained && !retained.confirmed) {
        let receipt;
        try { receipt = await json(retained.receiptFile); } catch { /* no identity is not closure */ }
        if (!receipt?.handle) throw new Error('Source synchronization submission unresolved; retain native receipt');
        const state = await observe(receipt.handle, {signal,accountingExcluded:true});
        if (state.state !== 'stopped') throw new Error(`Source synchronization unresolved; observe ${receipt.handle}`);
        if (state.exitCode !== 0) { await rm(cachePath); throw closedSetupError(`Source synchronization failed: ${receipt.handle}`, {handle: receipt.handle, reason: state}); }
        retained.confirmed = true;
        await save(cachePath, retained);
      }
      const reused = Boolean(snapshot.cacheable && retained?.confirmed && retained.key === key);
      let stage = reused ? retained : { key, cacheable: snapshot.cacheable,
        incoming: posix.join(runRoot, 'incoming'), manifest: posix.join(runRoot, 'sync.json') };
      if (!reused) {
        await transfer(['upload', options.target, projectRoot, stage.incoming], dir, signal);
        const configuration = { incoming: posix.join(options.workspacePath, stage.incoming),
          project: posix.join(options.workspacePath, destination), files: snapshot.files,
          snapshot: snapshot.cacheable ? snapshot.entries : null };
        const manifestDir = await mkdtemp(join(root, 'manifest-'));
        const manifest = join(manifestDir, 'sync.json');
        const bytes = JSON.stringify(configuration);
        stage.manifestSha256 = createHash('sha256').update(bytes).digest('hex');
        await writeFile(manifest, bytes, { mode: 0o600 });
        await transfer(['upload', options.target, manifest, stage.manifest], manifestDir, signal);
      }
      const script = join(dir, 'sync.sh');
      await writeFile(script, `${quote(options.python ?? 'python3')} -c ${quote(synchronize)} ${quote(posix.join(options.workspacePath, stage.manifest))} ${quote(stage.manifestSha256)}\n`, { mode: 0o600 });
      const args = ['run', options.target, '--file', script, '--receipt-file', join(dir, 'sync-submission.json')];
      if (options.runtime) args.push('--runtime', options.runtime);
      stage = { ...stage, confirmed: false, receiptFile: join(dir, 'sync-submission.json') };
      await save(cachePath, stage);
      let submitted;
      try { submitted = await submit(args, stage.receiptFile, false, true, signal, opts.stopSignal); }
      catch (error) { if (error.code === 'CPL_NOT_ADMITTED') await rm(cachePath); throw error; }
      await event('execution-handle', { handle: submitted.handle, purpose: 'setup' });
      let state;
      do {
        state = await observe(submitted.handle, {signal,accountingExcluded:true});
        if (state.state === 'stopped' && state.exitCode !== 0) {
          await rm(cachePath); throw closedSetupError(`Copy failed: ${submitted.handle}`, {handle: submitted.handle, reason: state});
        }
        if (signal?.aborted) { await stop(submitted.handle,{accountingExcluded:true,signal:opts.stopSignal}); signal.throwIfAborted(); }
        if (state.state === 'running') await delay(options.pollMs ?? 250);
      } while (state.state === 'running');
      if (state.state !== 'stopped') throw new Error(`Copy unresolved: ${submitted.handle}`);
      if ((await sourceSnapshot(projectRoot)).fingerprint !== snapshot.fingerprint) {
        await rm(cachePath); throw new Error('Local source changed during staging; wait for edits to settle');
      }
      if (snapshot.cacheable) await save(cachePath, { ...stage, confirmed: true });
      else await rm(cachePath);
      await event('source-staged', { reused, fingerprint: snapshot.fingerprint });
    });
    if (opts.signal?.aborted || remaining() === 0) throw new Error('Execution cancelled or elapsed budget exhausted');
    const spool = posix.join(runRoot, 'output');
    const timingKey = randomUUID();
    const configuration = { command, timingKey, verifiers: ['verification', 'performance'].includes(opts.purpose) ? frozen.map(file => [posix.join(options.workspacePath, file.remote), file.sha256]) : [], cwd: posix.join(options.workspacePath, destination, subdir),
      spool: posix.join(options.workspacePath, spool), inlineResult:true, env: { ...options.env, ...targetEnvironment(opts.env) }, timeout: remaining() ?? null };
    if (opts.leaseReceipt) configuration.env.RSI_LEASE_RECEIPT = posix.join(configuration.spool,'admission.json');
    const script = join(dir, 'command.sh');
    const receipt = join(dir, 'submission.json');
    await writeFile(script, `${quote(options.python ?? 'python3')} -c ${quote(supervisor)} ${quote(JSON.stringify(configuration))}\n`, { mode: 0o600 });
    const args = ['run', options.target, '--file', script, '--receipt-file', receipt];
    if (options.runtime) args.push('--runtime', options.runtime);
    if (opts.resourceId !== undefined) args.push('--npu', String(opts.resourceId));
    const launch={wallMs:Date.now(),monotonicNs:String(process.hrtime.bigint()),boot:(await readFile('/proc/sys/kernel/random/boot_id','utf8')).trim()};
    const receiptData = { handle: null, receiptFile: receipt, spool, timingKey, launch, admissionToken:opts.admissionToken, cwd: configuration.cwd };
    await save(join(dir, 'execution.json'), receiptData);
    const submitted = await submit(args, receipt, false, false, signal, opts.stopSignal);
    handle = submitted.handle;
    check(handle);
    receiptData.handle = handle;
    await save(join(dir, 'execution.json'), receiptData);
    executionReceipts.set(handle, { ...receiptData, dir });
    await event('execution-handle', { handle });
    let state;
    for (;;) {
      const stopping = signal?.aborted || remaining() === 0;
      state = stopping ? await stop(handle,{signal:opts.stopSignal}) : await observe(handle, {signal});
      if (!stopping && signal?.aborted) state = await stop(handle,{signal:opts.stopSignal});
      if (state.state !== 'running') break;
      await delay(Math.min(options.pollMs ?? 250, (remaining() ?? 1) * 1000));
    }
    if (state.state === 'stopped') {
      try {
        for (const stream of ['stdout', 'stderr']) {
          const path = join(dir, stream);
          if(!(await cachedTerminal({...receiptData,dir}))?.inlineOutput)
            await copy('transfer', () => transfer(['download', options.target, posix.join(spool, stream), path], dir, signal));
          if (opts.onData) for await (const chunk of createReadStream(path)) opts.onData(chunk);
        }
        await event('execution-output', { handle, stdout: join(dir, 'stdout'), stderr: join(dir, 'stderr'), partial: false });
      } catch {
        await event('execution-output', { handle, partial: true, archive: { target: options.target, path: spool } });
        if (opts.onData) opts.onData(Buffer.from(`\nOutput unavailable/partial; full remote archive: ${spool}\n`));
      }
    }
    return { handle, exitCode: state.exitCode ?? null, settled: state.state === 'stopped',
      ...(state.outcome?{outcome:state.outcome}:{}),
      ...(state.timingSource?{timingSource:state.timingSource,timingEstimated:state.timingEstimated}:{}),
      ...(state.executionSeconds === undefined ? { accountingUnresolved: true } : { executionSeconds: state.executionSeconds }) };
  }
  // Token-scoped target protocol receipt; payload stdout/exit codes are not admission evidence.
  async function readAdmission(handle, {signal = context.signal} = {}) {
    check(handle);
    let execution=executionReceipts.get(handle),dir=!remote?join(root,handle.slice(6)):execution?.dir;
    if (remote&&!dir) for (const name of await readdir(root)) {
      const candidate=join(root,name);
      try {
        const row=await json(join(candidate,'execution.json'));
        const id=row.handle ?? (row.receiptFile ? (await json(row.receiptFile)).handle : null);
        if(id===handle){dir=candidate;execution=row;break;}
      } catch { /* unrelated execution */ }
    }
    if(!dir) return null;
    const cached=remote?await cachedTerminal({...execution,dir}):null;
    if(cached)return cached.admission;
    const path=join(dir,'admission.json');
    try {
      if(remote) await exclude('transfer',()=>transfer(['download',options.target,posix.join(execution.spool,'admission.json'),path],dir,signal));
      return await json(path);
    } catch { return null; }
  }
  // A lost transport ID remains unknown. The original nonce-bound wrapper and
  // released admission token can nevertheless prove that its payload is closed.
  async function wrapperClosure(data) {
    if(!remote||data.kind!=='run'||data.accountingExcluded||!data.receiptFile)return null;
    const dir=dirname(data.receiptFile),name=relative(root,dir);
    if(!/^exec-[A-Za-z0-9]+$/.test(name)||data.receiptFile!==join(dir,'submission.json'))return null;
    try{
      const receipt=await json(join(dir,'execution.json'));
      if(receipt.receiptFile!==data.receiptFile||receipt.handle||!receipt.timingKey||!receipt.admissionToken||!receipt.spool)return null;
      for(const file of ['timing.json','admission.json','exit']){
        const transferDir=await mkdtemp(join(root,'closure-'));
        await exclude('transfer',()=>transfer(['download',options.target,posix.join(receipt.spool,file),join(dir,file)],transferDir,context.stopSignal??context.signal));
      }
      const timing=await json(join(dir,'timing.json')),admission=await json(join(dir,'admission.json')),exitText=(await readFile(join(dir,'exit'),'utf8')).trim(),exit=Number(exitText);
      if(timing?.completed!==true||timing.key!==receipt.timingKey||!Number.isFinite(timing.executionSeconds)||timing.executionSeconds<0||!/^[-]?\d+$/.test(exitText)||!Number.isInteger(exit)||admission?.token!==receipt.admissionToken||admission.decision!=='admitted'||admission.released!==true)return null;
      const proof={closed:true,receiptFile:data.receiptFile,timingKey:receipt.timingKey,admissionToken:receipt.admissionToken,admission,executionSeconds:timing.executionSeconds,exitCode:exit,nativeIdentityMissing:true,artifactsUnresolved:true};
      await save(join(dir,'wrapper-closure.json'),proof);return proof;
    }catch{return null;}
  }
  async function reconcileOperation({ operation, events }) {
    const pending = events.filter(({ type, data }) => type === 'execution-submitting' && (data.operation === operation || data.submission === operation));
    const handles = new Set(),wrapperClosures=[],unresolvedArtifacts=[];
    if (!pending.length) return { closed: false, accountingUnresolved: true, handles: [] };
    let executionSeconds = 0,timingEstimated=false,closed=true,accountingUnresolved=false,accounted=false;
    const excluded=excludedExecutionProofs(events);
    for (const submission of pending) {
      const {data}=submission;
      const response = events.find(event => ['execution-submitted', 'execution-not-admitted'].includes(event.type) && event.data.submission === data.submission);
      if (response?.type === 'execution-not-admitted' || (response?.data.closed && data.accountingExcluded)) continue;
      let handle = response?.data.handle;
      if (!handle && data.receiptFile) try { handle = (await json(data.receiptFile)).handle; } catch { /* no identity is not closure */ }
      if (!handle) {
        if(data.kind==='download'&&data.accountingExcluded===true){
          unresolvedArtifacts.push({...data});continue;
        }
        const proof=await wrapperClosure(data);
        if(proof){wrapperClosures.push({...proof,submission:data.submission});executionSeconds+=proof.executionSeconds;accounted=true;continue;}
        closed=false;accountingUnresolved=true;
        if(!data.accountingExcluded) {
          let timing;
          if(data.receiptFile)try {
            const dir=dirname(data.receiptFile),name=relative(root,dir);
            if(name&&!name.startsWith('..')&&!isAbsolute(name)) {
              const receipt=await json(join(dir,'execution.json'));
              if(receipt.receiptFile===data.receiptFile)
                timing=await controllerTiming({...receipt,dir},'unknown',{wallMs:Date.parse(submission.created)});
            }
          } catch { /* the original submission remains unidentified and fenced */ }
          if(!timing&&Number.isFinite(Date.parse(submission.created)))
            timing={executionSeconds:Math.max(0,(Date.now()-Date.parse(submission.created))/1000),timingEstimated:true};
          if(timing) {executionSeconds+=timing.executionSeconds;timingEstimated=true;accounted=true;}
        }
        continue;
      }
      check(handle);
      if (handles.has(handle)) continue;
      handles.add(handle);
      const terminalEvent=events.find(e=>e.type==='execution-state'&&e.data.handle===handle&&e.data.state==='stopped'&&Number.isFinite(Date.parse(e.created)));
      const launch={wallMs:Date.parse(submission.created),terminalWallMs:Date.parse(terminalEvent?.created)};
      const state = (data.accountingExcluded === true && excluded.get(handle)) ||
        await observe(handle,{accountingExcluded:data.accountingExcluded === true,launch});
      if(data.accountingExcluded)accounted=true;
      else if(Number.isFinite(state.executionSeconds)) {
        executionSeconds += state.executionSeconds;timingEstimated ||= state.timingEstimated===true;accounted=true;
      }else accountingUnresolved=true;
      if(state.state!=='stopped')closed=false;
    }
    const payloadClosed=closed&&!accountingUnresolved&&(!unresolvedArtifacts.length||
      pending.some(({data})=>!(data.kind==='download'&&data.accountingExcluded===true)));
    return {closed:payloadClosed,payloadClosed,
      ...(unresolvedArtifacts.length?{unresolvedArtifacts}:{}),
      ...((accounted||!accountingUnresolved)?{executionSeconds}:{}),
      ...(timingEstimated?{timingEstimated}:{}),...(accountingUnresolved?{accountingUnresolved}:{}),handles:[...handles],...(wrapperClosures.length?{wrapperClosures,nativeIdentityMissing:true,artifactsUnresolved:true}:{})};
  }
  // Installation eligibility only: never settles work or permits its resumption.
  async function canUpgrade({ attempt, events }) {
    if (!remote || !isAbsolute(options.workspacePath ?? '') || !options.remoteDirectory ||
        isAbsolute(options.remoteDirectory) || options.remoteDirectory.split('/').includes('..')) return false;
    const staging = posix.join(options.remoteDirectory, encodeURIComponent(attempt.id));
    const submitted = events.filter(e => e.type === 'execution-submitting');
    const handles = new Set(), operations = new Set();
    const excluded=excludedExecutionProofs(events);
    try {
      for (const { data } of submitted) {
        if (typeof data.submission !== 'string' || !data.submission) return false;
        if (!['upload', 'download', 'run', 'resource-policy'].includes(data.kind)) return false;
        operations.add(data.operation ?? data.submission);
        const responses = events.filter(e => ['execution-submitted', 'execution-not-admitted'].includes(e.type) &&
          e.data.submission === data.submission);
        if (responses.some(e => e.type === 'execution-not-admitted')) {
          if (responses.some(e => e.data.handle)) return false;
          continue;
        }
        if (['upload', 'download'].includes(data.kind) && data.accountingExcluded === true &&
            responses.some(e => e.type === 'execution-submitted' && e.data.closed === true) &&
            !responses.some(e => e.data.handle)) continue;
        const completed = responses.map(e=>e.data.handle).filter(Boolean);
        if (data.accountingExcluded === true && completed.length === 1 && excluded.get(completed[0])) {
          check(completed[0]);handles.add(completed[0]);continue;
        }
        const receipt = data.receiptFile ? await json(data.receiptFile) : null;
        const ids = new Set([...responses.map(e => e.data.handle), receipt?.handle].filter(Boolean));
        if (ids.size !== 1) return false;
        const [handle] = ids;
        check(handle);
        handles.add(handle);
        let observed;
        try { observed = await client(['observe', handle]); }
        catch { observed = { state: 'unknown' }; }
        if (terminal.has(observed.state)) continue;
        if (data.kind !== 'upload' || data.accountingExcluded !== true ||
            receipt?.operation !== 'upload' || receipt.target !== options.target || receipt.handle !== handle ||
            !handle.startsWith(`remote:${options.target}:transfer:`) ||
            !responses.some(e => e.data.handle === handle)) return false;
        const path = await realpath(data.receiptFile), name = relative(root, path);
        const match = /^(exec-[A-Za-z0-9]+|manifest-[A-Za-z0-9]+)\/transfer\.json$/.exec(name);
        if (!match || path !== resolve(data.receiptFile)) return false;
        if (match[1].startsWith('manifest-')) {
          const manifest = await json(join(dirname(path), 'sync.json'));
          const prefix = posix.join(options.workspacePath, staging) + '/';
          const incoming = manifest.incoming?.slice(prefix.length);
          if (!manifest.incoming?.startsWith(prefix) || !/^exec-[A-Za-z0-9]+\/incoming$/.test(incoming) ||
              !manifest.project?.startsWith(prefix) || posix.normalize(manifest.project) !== manifest.project ||
              !Array.isArray(manifest.files) || manifest.files.some(file => typeof file !== 'string' ||
                posix.isAbsolute(file) || file.split('/').includes('..'))) return false;
        }
        // exec-* uploads have the fixed destination staging/exec-*/incoming.
        // manifest-* uploads carry the existing sync configuration for that staging.
      }
      for (const { type, data } of events) {
        if (type === 'execution-handle' && !handles.has(data.handle)) return false;
        if (['execution-starting', 'execution-unresolved'].includes(type) &&
            !operations.has(data.operation ?? data.submission)) return false;
      }
      return true;
    } catch { return false; }
  }
  return {
    instructions, execute, observe, stop, reconcileOperation, readAdmission,localOperations:local,
    ...(remote ? { canUpgrade } : {}),
    async prepare({ project, attempt, signal=context.signal }) {
      signal?.throwIfAborted();
      return exclude('setup', async () => {
        if (remote) {
          if (!isAbsolute(options.workspacePath ?? '') || !options.remoteDirectory || isAbsolute(options.remoteDirectory) || options.remoteDirectory.split('/').includes('..')) throw new Error('Configure absolute registry workspacePath and relative task-owned remoteDirectory');
          projectRoot = resolve(project.sourceDirectory ?? project.source ?? options.projectDirectory ?? '');
          attemptRoot = posix.join(options.remoteDirectory, encodeURIComponent(String(attempt.id)));
          const verifierFiles = (project.verifier?.files ?? []).map(file => resolve(file));
          let layoutRoot = resolve(project.bundle?.root??projectRoot);
          const insideLayout=file=>{const rel=relative(layoutRoot,file);return rel===''||(!rel.startsWith('../')&&rel!=='..'&&!isAbsolute(rel));};
          if(!insideLayout(projectRoot))throw new Error('Editable source must be inside its bundle root');
          for (const file of verifierFiles) {
            const rel = relative(projectRoot, file);
            if (rel === '' || (!rel.startsWith('../') && !isAbsolute(rel))) throw new Error('Verifier must be outside editable source');
            if(project.bundle&&!insideLayout(file))throw new Error('Frozen verifier must be inside its bundle root');
            while (relative(layoutRoot, file).startsWith('../')) layoutRoot = dirname(layoutRoot);
          }
          projectLayout = verifierFiles.length ? posix.join('tree', relative(layoutRoot, projectRoot)) : 'project';
          frozen = [];
          for (const file of verifierFiles) frozen.push({ local: file,
            remote: posix.join(attemptRoot, 'tree', relative(layoutRoot, file)),
            sha256: createHash('sha256').update(await readFile(file)).digest('hex') });
          await discoverCapabilities(signal);
          const preflight=await client(['preflight',options.target],{signal});
          if(preflight.state!=='completed')throw closedSetupError('CPL preflight failed',{reason:preflight});
          if (frozen.length) await exclude('transfer', async () => {
            // One scoped tree transfer avoids one native transport/setup per
            // trusted file. It contains only frozen files, never actor source.
            const dir = await mkdtemp(join(root, 'verifier-'));
            const tree = join(dir, 'tree');
            await mkdir(tree);
            for (const file of frozen) {
              const destination = join(tree, relative(layoutRoot, file.local));
              await mkdir(dirname(destination), { recursive: true });
              await copyFile(file.local, destination);
            }
            await transfer(['upload', options.target, tree, posix.join(attemptRoot, 'tree')], dir,signal);
          });
          await event('verifier-staged', { files: frozen.map(({ remote, sha256 }) => ({ path: remote, sha256 })) });
        }
        await event('environment-prepared', { project: project.id, attempt: attempt.id });
        return { instructions };
      });
    },
    async resources(request) {
      return context.resources ? context.resources(request) : remote ? [] : [{ id: 'local', available: true, metadata: {} }];
    },
    async browse(path) {
      if (!remote) return readdir(path, { withFileTypes: true }).then(entries => entries.map(entry => ({ name: entry.name, directory: entry.isDirectory() })));
      const dir = await mkdtemp(join(root, 'download-'));
      return exclude('transfer', () => transfer(['download', options.target, path, join(dir, 'content')], dir));
    },
    async collectArtifact(source,destination) {
      return exclude('transfer',async()=>{
        if(!remote)await copyFile(source,destination);
        else {
          const directory=await mkdtemp(join(root,'collect-'));
          await transfer(['download',options.target,source,destination],directory,context.signal);
        }
        return destination;
      });
    },
  };
}

export async function createEnvironmentTools(environment, cwd, sdk) {
  sdk ??= await import('@earendil-works/pi-coding-agent');
  const tool=sdk.createBashTool(cwd, { operations: { exec: async (command, cwd, opts) => {
    // Use the same delta/session filter as the CPL supervisor for local tools.
    const result = await environment.execute(command, cwd, {...opts,env:targetEnvironment(opts.env)});
    if (!result.settled) throw new Error(`Execution remains unresolved: ${result.handle}`);
    return result;
  } } });
  const target={...tool,name:'rsi_target',label:'Run target command',
    description:'Execute a command in the configured target environment. Commands start in the editable source directory. Remote targets use a source execution snapshot without trusted verifier files; run frozen checks through rsi_verify, which stages those files. Use native local bash/read/write/edit for persistent workspace changes. Returns target stdout/stderr and exit status.'};
  // On a device host, override Pi's Bash tool through its supported extension
  // API. Both shell names must pass through the same admission and supervision.
  return environment.shellOnTarget ? [target,{...tool,name:'bash'}] :
    [target,sdk.createBashTool(cwd,{operations:environment.localOperations??sdk.createLocalBashOperations()})];
}

