import { readFile, readdir, mkdtemp, writeFile, mkdir, rename } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { homedir } from 'node:os';
import { join, resolve, relative, isAbsolute, posix, dirname } from 'node:path';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

const quote = value => "'" + String(value).replaceAll("'", "'\\''") + "'";
const terminal = new Set(['completed', 'failed', 'cancelled']);
const normalize = result => ({ state: terminal.has(result.state) ? 'stopped' :
  ['queued', 'dispatching', 'running', 'cancelling'].includes(result.state) ? 'running' : 'unknown',
  ...(Number.isInteger(result.exit) ? { exitCode: result.exit } : {}) });
async function sourceFiles(path, prefix = '') {
  const files = [];
  for (const entry of await readdir(path, { withFileTypes: true })) {
    const name = posix.join(prefix, entry.name);
    if (entry.isDirectory()) files.push(...await sourceFiles(join(path, entry.name), name));
    else files.push(name);
  }
  return files;
}
const json = async path => JSON.parse(await readFile(path, 'utf8'));
async function save(path, value) {
  await writeFile(path + '.tmp', JSON.stringify(value), { mode: 0o600 });
  await rename(path + '.tmp', path);
}

// Only the installed public CLI owns remote protocols and credentials.
export function createCplClient(executable = join(homedir(), '.agents/skills/remote-access/scripts/cpl-remote')) {
  return async args => {
    await readFile(join(homedir(), '.agents/skills/remote-access/SKILL.md'), 'utf8');
    return new Promise((resolve, reject) => {
      const child = spawn(executable, ['--json', ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
      let output = '', error = '';
      child.stdout.on('data', data => { output += data; });
      child.stderr.on('data', data => { error += data; });
      child.on('error', reject);
      child.on('close', code => {
        try { resolve(JSON.parse(output)); }
        catch { reject(new Error(`CPL ${args[0]} failed (${code}): ${error}`)); }
      });
    });
  };
}

// The payload supervisor enforces workload elapsed time, not CLI observation time.
const supervisor = String.raw`import hashlib, json, os, signal, subprocess, sys
c=json.loads(sys.argv[1]); os.makedirs(c['spool'],exist_ok=True)
def intact():
 try:
  return all(hashlib.sha256(open(path,'rb').read()).hexdigest()==digest for path,digest in c['verifiers'])
 except OSError: return False
with open(c['spool']+'/stdout','wb') as out, open(c['spool']+'/stderr','wb') as err:
 if not intact(): err.write(b'Frozen verifier changed before execution\n'); code=125
 else:
  p=subprocess.Popen(['bash','-c',c['command']],cwd=c['cwd'],env={**os.environ,**c['env']},stdout=out,stderr=err,start_new_session=True)
  def stop(*args):
   try: os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError: pass
  signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
  try: code=p.wait(timeout=c['timeout'])
  except subprocess.TimeoutExpired: stop(); p.wait(); code=124
  finally: stop()
  if not intact(): err.write(b'Frozen verifier changed after execution\n'); code=125
with open(c['spool']+'/exit','w') as f: f.write(str(code))
sys.exit(code if code>=0 else 128-code)
`;

const synchronize = `import json, os, shutil, sys
c=json.loads(sys.argv[1]); os.makedirs(c['project'],exist_ok=True)
shutil.copytree(c['incoming'],c['project'],dirs_exist_ok=True)
manifest=c['project']+'/.environment-source-files.json'
try:
 with open(manifest) as f: previous=json.load(f)
except FileNotFoundError: previous=[]
for name in set(previous)-set(c['files']):
 try: os.unlink(c['project']+'/'+name)
 except FileNotFoundError: pass
with open(manifest,'w') as f: json.dump(c['files'],f)
`;

export async function createEnvironment(options, context) {
  const sdk = context.sdk ?? await import('@earendil-works/pi-coding-agent');
  const event = (type, data) => context.event(type, data);
  const exclude = (label, fn) => context.exclude(label, fn);
  const root = join(context.instancePath, '.executions');
  await mkdir(root, { recursive: true, mode: 0o700 });
  const client = context.client ?? createCplClient(options.clientPath);
  const local = sdk.createLocalBashOperations();
  const remote = options.mode === 'cpl';
  if (!remote && options.mode !== 'local') throw new Error('Environment mode must be local or cpl');
  let projectRoot, attemptRoot, projectLayout = 'project', frozen = [];
  const instructions = (options.instructionFile ? await exclude('setup', () => readFile(options.instructionFile, 'utf8')) : '') +
    (remote ? '\nBash runs in a synchronized remote attempt directory. Use project-relative Bash paths. Native read/write/edit tools operate on the LOCAL project; edits sync before each Bash call.' : '');
  const check = handle => {
    if (remote && !handle.startsWith(`remote:${options.target}:`)) throw new Error('Handle belongs to another target');
    if (!remote && !/^local:exec-[A-Za-z0-9]+$/.test(handle)) throw new Error('Invalid local handle');
  };
  async function localState(handle) {
    check(handle);
    const dir = join(root, handle.slice(6));
    try {
      const receipt = await json(join(dir, 'execution.json'));
      if (receipt.boot !== (await readFile('/proc/sys/kernel/random/boot_id', 'utf8')).trim()) return { state: 'unknown' };
      const original = (await readFile(join(dir, 'identity'), 'utf8')).trim();
      const pid = Number(original.slice(0, original.indexOf(' ')));
      const fields = text => text.slice(text.lastIndexOf(')') + 2).split(' ');
      try {
        const current = await readFile(`/proc/${pid}/stat`, 'utf8');
        if (fields(current)[19] !== fields(original)[19]) return { state: 'unknown' };
      } catch (error) { if (error.code !== 'ENOENT') throw error; }
      let live = false;
      for (const name of await readdir('/proc')) {
        if (!/^\d+$/.test(name)) continue;
        try {
          const stat = fields(await readFile(`/proc/${name}/stat`, 'utf8'));
          if (Number(stat[2]) === pid && !['Z', 'X'].includes(stat[0])) live = true;
        } catch (error) { if (error.code !== 'ENOENT' && error.code !== 'ESRCH') return { state: 'unknown' }; }
      }
      let exitCode;
      try { exitCode = Number(await readFile(join(dir, 'exit'), 'utf8')); } catch { /* interrupted wrapper */ }
      return { state: live ? 'running' : 'stopped', pid, ...(Number.isInteger(exitCode) ? { exitCode } : {}) };
    } catch { return { state: 'unknown' }; }
  }
  async function observe(handle) {
    check(handle);
    let state;
    try { state = remote ? normalize(await client(['observe', handle])) : await localState(handle); }
    catch { state = { state: 'unknown' }; }
    await event('execution-state', { handle, ...state });
    return state;
  }
  async function stop(handle) {
    check(handle);
    if (remote) {
      try { await client(['cancel', handle]); } catch { /* observe is authoritative */ }
    } else {
      const state = await localState(handle);
      if (state.state === 'running') {
        try { process.kill(-state.pid, 'SIGKILL'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
      }
    }
    const deadline = performance.now() + 5000;
    while (performance.now() < deadline) {
      const state = await observe(handle);
      if (state.state !== 'running') return state;
      await delay(50);
    }
    return { state: 'unknown' };
  }
  async function submit(args, receipt, allowCompleted = false) {
    const operation = randomUUID();
    let result, cause;
    try { result = await client(args); }
    catch (error) {
      cause = error;
      if (receipt) try { result = await json(receipt); } catch { /* absence is not closure */ }
    }
    if (result?.handle || (allowCompleted && result?.state === 'completed')) return result;
    if (result?.admitted === false || result?.submitted === false) {
      throw new Error(`CPL ${args[0]} was not admitted`);
    }
    const message = `CPL ${args[0]} submission identity unknown; do not resubmit`;
    await event('execution-unresolved', { operation, handle: null, message });
    throw new Error(message, { cause });
  }
  async function transfer(args, dir) {
    const upload = args[0] === 'upload';
    if (upload) args.push('--receipt-file', join(dir, 'transfer.json'), '--no-wait');
    else args.push('--timeout', '0');
    let result = await submit(args, upload ? join(dir, 'transfer.json') : undefined, true);
    const handle = result.handle;
    if (handle) {
      await event('execution-handle', { handle, purpose: 'transfer' });
      await event('execution-state', { handle, purpose: 'transfer', ...normalize(result) });
      if (!terminal.has(result.state)) result = await client(['observe', handle]);
      while (!terminal.has(result.state)) {
        if (!['queued', 'dispatching', 'running'].includes(result.state)) throw new Error(`Transfer unresolved; observe ${handle}`);
        await delay(options.pollMs ?? 250);
        result = await client(['observe', handle]);
      }
      if (!upload && result.state === 'completed') await client(['fetch', handle, args[3]]);
    }
    if (result.state !== 'completed') throw new Error(`Transfer did not complete: ${handle ?? result.state}`);
    return result;
  }
  async function execute(command, cwd, opts = {}) {
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
      await save(join(dir, 'execution.json'), { boot: (await readFile('/proc/sys/kernel/random/boot_id', 'utf8')).trim(), cwd });
      await event('execution-handle', { handle });
      const wrapper = `cat /proc/$$/stat > ${quote(join(dir, 'identity.tmp'))}\nmv ${quote(join(dir, 'identity.tmp'))} ${quote(join(dir, 'identity'))}\n` +
        `trap ${quote(`code=$?; printf "%s" "$code" > ${quote(join(dir, 'exit.tmp'))}; mv ${quote(join(dir, 'exit.tmp'))} ${quote(join(dir, 'exit'))}`)} EXIT\nbash -c ${quote(command)}`;
      try { await local.exec(wrapper, cwd, { ...opts, env: opts.env ? { ...process.env, ...opts.env } : undefined, timeout: remaining(), onData: opts.onData ?? (() => {}) }); }
      catch (error) {
        const state = await stop(handle);
        if (state.state !== 'stopped') throw new Error(`Local execution unresolved: ${handle}`, { cause: error });
        return { handle, exitCode: state.exitCode ?? null, settled: true };
      }
      const state = await localState(handle);
      if (state.state === 'running') await stop(handle);
      const final = await observe(handle);
      return { handle, exitCode: final.exitCode ?? null, settled: final.state === 'stopped' };
    }
    if (!projectRoot || !attemptRoot) throw new Error('Call prepare with project.sourceDirectory and attempt.id before remote execute');
    const subdir = relative(projectRoot, resolve(cwd ?? projectRoot));
    if (subdir === '..' || subdir.startsWith('../') || isAbsolute(subdir)) throw new Error('Bash cwd must be inside local project');
    const runRoot = posix.join(attemptRoot, dir.slice(dir.lastIndexOf('/') + 1));
    const destination = posix.join(attemptRoot, projectLayout);
    const incoming = posix.join(runRoot, 'incoming');
    const files = await sourceFiles(projectRoot);
    await copy('transfer', async () => {
      await transfer(['upload', options.target, projectRoot, incoming], dir);
      const configuration = { incoming: posix.join(options.workspacePath, incoming), project: posix.join(options.workspacePath, destination), files };
      const script = join(dir, 'sync.sh');
      await writeFile(script, `${quote(options.python ?? 'python3')} -c ${quote(synchronize)} ${quote(JSON.stringify(configuration))}\n`, { mode: 0o600 });
      const args = ['run', options.target, '--file', script, '--receipt-file', join(dir, 'sync-submission.json')];
      if (options.runtime) args.push('--runtime', options.runtime);
      const submitted = await submit(args, join(dir, 'sync-submission.json'));
      await event('execution-handle', { handle: submitted.handle, purpose: 'setup' });
      let state;
      do {
        state = await observe(submitted.handle);
        if (state.state === 'running') await delay(options.pollMs ?? 250);
      } while (state.state === 'running');
      if (state.state !== 'stopped' || state.exitCode !== 0) throw new Error(`Copy unresolved or failed: ${submitted.handle}`);
    });
    if (opts.signal?.aborted || remaining() === 0) throw new Error('Execution cancelled or elapsed budget exhausted');
    const spool = posix.join(runRoot, 'output');
    const configuration = { command, verifiers: opts.purpose === 'verification' ? frozen.map(file => [posix.join(options.workspacePath, file.remote), file.sha256]) : [], cwd: posix.join(options.workspacePath, destination, subdir),
      spool: posix.join(options.workspacePath, spool), env: { ...options.env, ...Object.fromEntries(Object.entries(opts.env ?? {}).filter(([key, value]) =>
        key !== 'PI_SESSION_FILE' && !(key.toUpperCase() === 'PATH' && ('PI_SESSION_ID' in (opts.env ?? {}))) && process.env[key] !== value)) }, timeout: remaining() ?? null };
    const script = join(dir, 'command.sh');
    const receipt = join(dir, 'submission.json');
    await writeFile(script, `${quote(options.python ?? 'python3')} -c ${quote(supervisor)} ${quote(JSON.stringify(configuration))}\n`, { mode: 0o600 });
    const args = ['run', options.target, '--file', script, '--receipt-file', receipt];
    if (options.runtime) args.push('--runtime', options.runtime);
    if (opts.resourceId !== undefined) args.push('--npu', String(opts.resourceId));
    const submitted = await submit(args, receipt);
    handle = submitted.handle;
    check(handle);
    await save(join(dir, 'execution.json'), { handle, spool, cwd: configuration.cwd });
    await event('execution-handle', { handle });
    let state;
    for (;;) {
      state = opts.signal?.aborted || remaining() === 0 ? await stop(handle) : await observe(handle);
      if (state.state !== 'running') break;
      await delay(Math.min(options.pollMs ?? 250, (remaining() ?? 1) * 1000));
    }
    if (state.state === 'stopped') {
      try {
        for (const stream of ['stdout', 'stderr']) {
          const path = join(dir, stream);
          await copy('transfer', () => transfer(['download', options.target, posix.join(spool, stream), path], dir));
          if (opts.onData) for await (const chunk of createReadStream(path)) opts.onData(chunk);
        }
        await event('execution-output', { handle, stdout: join(dir, 'stdout'), stderr: join(dir, 'stderr'), partial: false });
      } catch {
        await event('execution-output', { handle, partial: true, archive: { target: options.target, path: spool } });
        if (opts.onData) opts.onData(Buffer.from(`\nOutput unavailable/partial; full remote archive: ${spool}\n`));
      }
    }
    return { handle, exitCode: state.exitCode ?? null, settled: state.state === 'stopped' };
  }
  return {
    instructions, execute, observe, stop,
    async prepare({ project, attempt }) {
      return exclude('setup', async () => {
        if (remote) {
          if (!isAbsolute(options.workspacePath ?? '') || !options.remoteDirectory || isAbsolute(options.remoteDirectory) || options.remoteDirectory.split('/').includes('..')) throw new Error('Configure absolute registry workspacePath and relative task-owned remoteDirectory');
          projectRoot = resolve(project.sourceDirectory ?? project.source ?? options.projectDirectory ?? '');
          attemptRoot = posix.join(options.remoteDirectory, encodeURIComponent(String(attempt.id)));
          const verifierFiles = (project.verifier?.files ?? []).map(file => resolve(file));
          let layoutRoot = projectRoot;
          for (const file of verifierFiles) {
            const rel = relative(projectRoot, file);
            if (rel === '' || (!rel.startsWith('../') && !isAbsolute(rel))) throw new Error('Verifier must be outside editable source');
            while (relative(layoutRoot, file).startsWith('../')) layoutRoot = dirname(layoutRoot);
          }
          projectLayout = verifierFiles.length ? posix.join('tree', relative(layoutRoot, projectRoot)) : 'project';
          frozen = [];
          for (const file of verifierFiles) frozen.push({ local: file,
            remote: posix.join(attemptRoot, 'tree', relative(layoutRoot, file)),
            sha256: createHash('sha256').update(await readFile(file)).digest('hex') });
          const capabilities = await client(['capabilities', options.target]);
          if (capabilities.state !== 'available') throw new Error('CPL target unavailable');
          if ((await client(['preflight', options.target])).state !== 'completed') throw new Error('CPL preflight failed');
          for (const file of frozen) await exclude('transfer', async () => {
            const dir = await mkdtemp(join(root, 'verifier-'));
            await transfer(['upload', options.target, file.local, file.remote], dir);
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
  };
}

export async function createEnvironmentTools(environment, cwd, sdk) {
  sdk ??= await import('@earendil-works/pi-coding-agent');
  return [sdk.createBashTool(cwd, { operations: { exec: async (command, cwd, opts) => {
    const result = await environment.execute(command, cwd, opts);
    if (!result.settled) throw new Error(`Execution remains unresolved: ${result.handle}`);
    return result;
  } } })];
}

export function createExtension(options, context) {
  return async pi => {
    const environment = await createEnvironment(options, context);
    for (const tool of await createEnvironmentTools(environment, options.cwd)) pi.registerTool(tool);
    pi.on('before_agent_start', async event => ({ systemPrompt: `${event.systemPrompt}\n${environment.instructions}` }));
  };
}
