#!/usr/bin/env node
// Deliberately outside the automated test suite: this experiment can spend API tokens.
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { bootstrap, readConfig } from '../../cli/instance.mjs';
import { operate } from '../../cli/runtime.mjs';
import { main } from '../../cli/main.mjs';

if (process.env.RSI_LOCAL_PYTHON_LIVE !== '1')
  throw new Error('Set RSI_LOCAL_PYTHON_LIVE=1 to authorize this provider experiment');
const [profileFile, destination] = process.argv.slice(2);
if (!profileFile || !destination) throw new Error('Usage: run.mjs PRIVATE_BOOTSTRAP_JSON NEW_OUTPUT_DIRECTORY');
const profile = JSON.parse(await readFile(resolve(profileFile), 'utf8'));
if (!profile.actor) throw new Error('Private bootstrap JSON must configure actor, baseline and skills');
if (profile.install === false) throw new Error('Live experiment requires installed copied machinery dependencies');
const seconds = profile.skillSettings?.evaluationSeconds;
if (!Number.isFinite(seconds) || seconds <= 0) throw new Error('Set positive skillSettings.evaluationSeconds explicitly');
const python = profile.python ?? 'python3';
const oracle = fileURLToPath(new URL('tasks.py', import.meta.url));
const baseline = execFileSync(python, [oracle, 'source', 'baseline'], { encoding: 'utf8' });
const root = resolve(destination), instance = join(root, 'instance');
await mkdir(root, { recursive: false });
const quote = value => "'" + value.replaceAll("'", "'\\''") + "'";
const projects = [];
for (const kind of ['reduction', 'parser']) {
  const source = join(root, kind); await mkdir(source);
  await writeFile(join(source, 'answer.py'), baseline);
  projects.push({ id: kind, source, actor_seconds: profile.actor_seconds ?? 60,
    goal: kind === 'reduction'
      ? 'Implement reduce_tiles(values, tile=4) in answer.py: sum every integer, including incomplete final tiles and empty input.'
      : 'Implement parse_ints(text) in answer.py: parse comma-separated signed integers, allow whitespace/empty input, and reject malformed nonempty fields.',
    verifier: { files: [oracle], command: `${quote(python)} -B ${quote(oracle)} verify ${kind} ${quote(source)}` } });
}
await bootstrap(instance, { ...profile, projects, mode: 'phase2', tune_every: 2, target: 'cpu',
  concurrency: 1, curriculum: { phase2: projects.map(project => project.id) },
  environment: { module: 'machinery/integrations/environments/index.mjs', options: { mode: 'local' } } });
await operate('wave', await readConfig(instance));
const status = await main(['status', '--instance', instance]);
await writeFile(join(root, 'result.json'), JSON.stringify(status, null, 2) + '\n');
console.log(JSON.stringify({ instance, projects: status.projects.map(({ id, status }) => ({ id, status })),
  tuning: status.tuning, evidence: join(root, 'result.json') }));
