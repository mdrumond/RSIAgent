import { readFile, readdir, mkdtemp, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

const quote = value => "'" + String(value).replaceAll("'", "'\\''") + "'";
const terminal = new Set(['completed', 'failed', 'cancelled']);
const normalize = result => ({
  state: terminal.has(result.state) ? 'stopped' :
    ['queued', 'dispatching', 'running'].includes(result.state) ? 'running' : 'unknown',
  ...(Number.isInteger(result.exit) ? { exitCode: result.exit } : {}),
});

// This is a process adapter for the installed public CLI, not a remote protocol.
export function createCplClient(executable = join(homedir(), '.agents/skills/remote-access/scripts/cpl-remote')) {
  return async args => {
    // Policy is read anew before each invocation, including observations.
    await readFile(join(homedir(), '.agents/skills/remote-access/SKILL.md'), 'utf8');
    return new Promise((resolve, reject) => {
      const child = spawn(executable, ['--json', ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
      let output = '', error = '';
      child.stdout.on('data', data => { output += data; });
      child.stderr.on('data', data => { error += data; });
      child.on('error', reject);
      child.on('close', code => {
        try { resolve(JSON.parse(output)); }
        catch { reject(new Error(`CPL operation ${args[0]} failed (${code}): ${error}`)); }
      });
    });
  };
}

export async function createEnvironment(options, context) {
  const sdk = context.sdk ?? await import('@earendil-works/pi-coding-agent');
  const event = (type, data) => context.event(type, data);
  const exclude = (label, fn) => context.exclude(label, fn);
  const instructions = options.instructionFile
    ? await exclude('access-instructions', () => readFile(options.instructionFile, 'utf8')) : '';
  const client = context.client ?? createCplClient(options.clientPath);
  const local = sdk.createLocalBashOperations();
  const active = new Map();
  let sequence = 0;
  const remote = options.mode === 'cpl';
  if (!remote && options.mode !== 'local') throw new Error('Environment mode must be local or cpl');
  const check = handle => {
    if (remote && !handle.startsWith(`remote:${options.target}:`)) throw new Error('Handle belongs to another target');
  };
  async function observe(handle) {
    check(handle);
    let state;
    if (!remote) state = active.get(handle)?.state ?? { state: 'unknown' };
    else {
      try { state = normalize(await client(['observe', handle])); }
      catch { state = { state: 'unknown' }; }
    }
    await event('execution-state', { handle, ...state });
    return state;
  }
  async function stop(handle) {
    check(handle);
    if (remote) {
      try { await client(['cancel', handle]); } catch { /* observe remains authoritative */ }
    } else active.get(handle)?.controller.abort();
    return observe(handle);
  }
  async function execute(command, cwd, opts = {}) {
    const started = Date.now();
    const remaining = () => opts.timeout === undefined ? undefined : Math.max(0, opts.timeout - (Date.now() - started) / 1000);
    if (opts.signal?.aborted || remaining() === 0) throw new Error('Execution cancelled or elapsed budget exhausted');
    let handle, promise;
    if (remote) {
      // Native CLI receipt is retained locally; there is no remote upload handshake.
      const dir = await mkdtemp(join(context.instancePath, '.cpl-exec-'));
      const script = join(dir, 'command.sh');
      const receipt = join(dir, 'submission.json');
      const exports = Object.entries(opts.env ?? {}).map(([key, value]) => {
        if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) throw new Error('Invalid environment variable');
        return `export ${key}=${quote(value)}`;
      }).join('\n');
      await writeFile(script, `#!/bin/bash\n${exports}\n${command}\n`, { mode: 0o600 });
      const args = ['run', options.target, '--cwd', cwd, '--file', script, '--receipt-file', receipt];
      if (options.runtime) args.push('--runtime', options.runtime);
      if (remaining() !== undefined) args.push('--timeout', String(Math.max(1, Math.ceil(remaining()))));
      if (opts.resourceId !== undefined) args.push('--npu', String(opts.resourceId));
      let submitted;
      try { submitted = await client(args); }
      catch (error) {
        try { submitted = JSON.parse(await readFile(receipt, 'utf8')); }
        catch { throw new Error('Submission identity unknown; do not resubmit', { cause: error }); }
      }
      handle = submitted.handle;
      if (!handle) throw new Error('CPL returned no handle; do not resubmit');
      check(handle);
    } else {
      handle = `local:${process.pid}:${++sequence}`;
      const controller = new AbortController();
      active.set(handle, { controller, state: { state: 'running' } });
      await event('execution-handle', { handle });
      const abort = () => controller.abort();
      opts.signal?.addEventListener('abort', abort, { once: true });
      promise = local.exec(command, cwd, { ...opts, timeout: remaining(), signal: controller.signal,
        onData: opts.onData ?? (() => {}) }).then(async result => {
        const state = { state: 'stopped', exitCode: result.exitCode };
        active.get(handle).state = state;
        await event('execution-state', { handle, ...state });
        return { ...result, handle, settled: true };
      }).catch(async error => {
        active.get(handle).state = { state: 'unknown' };
        await event('execution-state', { handle, state: 'unknown' });
        throw error;
      }).finally(() => opts.signal?.removeEventListener('abort', abort));
      return promise;
    }
    await event('execution-handle', { handle });
    for (;;) {
      if (opts.signal?.aborted || remaining() === 0) {
        const state = await stop(handle);
        return { handle, exitCode: state.exitCode ?? null, settled: state.state === 'stopped' };
      }
      const state = await observe(handle);
      if (state.state === 'stopped' && opts.onData) {
        for (const stream of ['stdout', 'stderr']) {
          const logs = await client(['logs', handle, '--stream', stream, '--tail', String(options.logTail ?? 2000)]);
          opts.onData(Buffer.from(logs.content ?? ''));
        }
      }
      if (state.state !== 'running') {
        return { handle, exitCode: state.exitCode ?? null, settled: state.state === 'stopped' };
      }
      await delay(Math.min(options.pollMs ?? 250, (remaining() ?? 1) * 1000));
    }
  }
  async function transfer(args) {
    let result = await client(args);
    if (result.handle) {
      await event('execution-handle', { handle: result.handle, purpose: 'transfer' });
      while (!terminal.has(result.state)) {
        if (!['queued', 'dispatching', 'running'].includes(result.state)) {
          throw new Error(`Transfer unresolved; observe ${result.handle}`);
        }
        await delay(options.pollMs ?? 250);
        result = await client(['observe', result.handle]);
      }
    }
    if (result.state !== 'completed') throw new Error(`Transfer did not complete: ${result.handle ?? result.state}`);
    return result;
  }
  return {
    instructions, execute, observe, stop,
    async prepare({ project, attempt }) {
      return exclude('environment-prepare', async () => {
        if (remote) {
          const capabilities = await client(['capabilities', options.target]);
          if (capabilities.state !== 'available') throw new Error('CPL target unavailable');
          const preflight = await client(['preflight', options.target]);
          if (preflight.state !== 'completed') throw new Error('CPL preflight failed');
          for (const copy of options.uploads ?? []) await transfer(['upload', options.target, copy.local, copy.remote]);
        }
        await event('environment-prepared', { project: project.id, attempt: attempt.id });
        return { instructions };
      });
    },
    async resources(request) {
      // Target binding owns live device health and opaque resource selection.
      return context.resources ? context.resources(request) : remote ? [] : [{ id: 'local', available: true, metadata: {} }];
    },
    async browse(path) {
      if (!remote) return readdir(path, { withFileTypes: true }).then(entries => entries.map(entry => ({ name: entry.name, directory: entry.isDirectory() })));
      return exclude('environment-download', () => transfer(['download', options.target, path, options.downloadDirectory ?? tmpdir()]));
    },
  };
}

export async function createEnvironmentTools(environment, cwd, sdk) {
  sdk ??= await import('@earendil-works/pi-coding-agent');
  return [sdk.createBashTool(cwd, { operations: { exec: async (...args) => {
    const result = await environment.execute(...args);
    if (!result.settled) throw new Error(`Execution remains unresolved: ${result.handle}`);
    return result;
  } } })];
}

// Use as a native Pi extension by supplying the same ignored instance options.
export function createExtension(options, context) {
  return async pi => {
    const environment = await createEnvironment(options, context);
    for (const tool of await createEnvironmentTools(environment, options.cwd)) pi.registerTool(tool);
    pi.on('before_agent_start', async event => ({ systemPrompt: `${event.systemPrompt}\n${environment.instructions}` }));
  };
}
