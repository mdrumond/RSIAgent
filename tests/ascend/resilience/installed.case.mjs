import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {Blocked} from './runner.mjs';

const exec = promisify(execFile);

export const cases = [{id: 'installed-cli-dashboard', tier: 'local', async run(ctx) {
  const [major, minor] = process.versions.node.split('.').map(Number);
  if (major < 22 || (major === 22 && minor < 23)) throw new Blocked('Installed gate requires Node >=22.23');
  const python = process.env.DRIVER_PYTHON ?? process.env.PYTHON ?? 'python3.12';
  try {
    await exec('npm', ['--version'], {signal: ctx.signal, timeout: 10000});
    await exec(python, ['-c', 'import sys; assert sys.version_info >= (3,11); import mcp, flask'],
      {signal: ctx.signal, timeout: 10000});
  } catch (error) {
    ctx.signal.throwIfAborted();
    throw new Blocked(`Installed gate needs npm and an existing Python >=3.11 profile with MCP/Flask: ${error.code ?? error.message}`);
  }
  await ctx.event({type: 'installed-case-input', seed: ctx.seed, profile: ctx.profile});
  const {runInstalledPackagingCase} = await import('../resilience-installed.mjs');
  return runInstalledPackagingCase({artifactRoot: ctx.directory, python, signal: ctx.signal,
    event: ctx.event, onCleanup: ctx.onCleanup});
}}];
