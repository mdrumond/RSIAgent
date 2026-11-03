import { readFile, access } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { tune } from '../controller/skill-evolution.mjs';

// Runtime module exports bootstrap({instanceRoot,config,sourceInstanceRoot,skillVersion})
// and wave({instanceRoot,config,projectIds:[id],evaluation:true}). bootstrap must copy
// the selected immutable skill version from sourceInstanceRoot, project inputs and frozen
// verifiers into a new instance. wave returns {correct:boolean,score:number}; higher wins.
// Both functions retain native provider/session/SFT and verifier evidence inside that instance.
// Programmatic callers may supply values.runtime with these exact functions instead.
export async function run(config, values = {}) {
  const instanceRoot = resolve(values.instanceRoot ?? config.instanceRoot ?? '.');
  if (config.skills?.tuningDisabled || config.phase === 'phase1')
    return { analyse: false, evaluate: false };
  const ids = config.skills?.tuningProjectIds;
  if (!Array.isArray(ids) || ids.length !== 2 || new Set(ids).size !== 2)
    throw new Error('skills.tuningProjectIds requires two distinct canonical regression project IDs');
  const runtimePath = config.skills?.runtimeModule;
  const runtime = values.runtime ?? (runtimePath && await import(pathToFileURL(resolve(instanceRoot, runtimePath)).href));
  if (typeof runtime?.bootstrap !== 'function' || typeof runtime?.wave !== 'function')
    throw new Error('Provide skills.runtimeModule exporting bootstrap and wave, or values.runtime');
  const task = values.task ?? (values.taskFile && JSON.parse(await readFile(resolve(values.taskFile), 'utf8')));
  const evaluate = async request => {
    try {
      await access(request.instanceRoot);
      throw new Error('Evaluation instance already exists');
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    await runtime.bootstrap(request);
    return runtime.wave({ instanceRoot: request.instanceRoot, config: request.config,
      projectIds: [request.projectId], evaluation: true });
  };
  return tune({ instanceRoot, config, configPath: values.configPath ?? join(instanceRoot, 'config.json'),
    task, evaluate, native: values.native, modelRuntime: values.modelRuntime, model: values.model });
}
