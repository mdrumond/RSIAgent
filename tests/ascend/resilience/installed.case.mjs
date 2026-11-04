import {Blocked} from './runner.mjs';

export const cases = [
  {id: 'installed-cli-dashboard', tier: 'local'},
  {id: 'installed-cli-dashboard-cpl', tier: 'remote'},
].map(item => ({...item, async run(ctx) {
  const [major, minor] = process.versions.node.split('.').map(Number);
  if (major < 22 || (major === 22 && minor < 23)) throw new Blocked('Installed gate requires Node >=22.23');
  const python = process.env.DRIVER_PYTHON ?? process.env.PYTHON ?? 'python3.12';
  const remoteProfile = item.tier === 'remote' ? process.env.INSTALLED_REMOTE_PROFILE : null;
  if (item.tier === 'remote' && !remoteProfile) throw new Blocked('Set INSTALLED_REMOTE_PROFILE to ignored registered CPL CPU options');
  const {runInstalledPackagingCase, InstalledPrerequisiteError} = await import('../resilience-installed.mjs');
  await ctx.event({type: 'installed-case-input', seed: ctx.seed, profile: ctx.profile});
  try {
    return await runInstalledPackagingCase({artifactRoot: ctx.directory, python, remoteProfile, signal: ctx.signal,
      event: ctx.event, onCleanup: ctx.onCleanup});
  } catch (error) {
    if (error instanceof InstalledPrerequisiteError) throw new Blocked(error.message);
    throw error;
  }
}}));
