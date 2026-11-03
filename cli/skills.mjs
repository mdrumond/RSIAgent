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
  if (config.skills?.tuningDisabled || config.mode === 'phase1')
    return { analyse: false, evaluate: false };
  const instanceRoot = resolve(config.root);
  const action = values.config ? JSON.parse(await readFile(resolve(values.config), 'utf8')) : {};
  const ids = config.skills?.tuningProjectIds;
  if (!Array.isArray(ids) || ids.length !== 2 || new Set(ids).size !== 2)
    throw new Error('skills.tuningProjectIds requires two distinct canonical regression project IDs');
  const runtimePath = config.skills?.runtimeModule;
  const runtime = values.runtime ?? await import(runtimePath
    ? pathToFileURL(resolve(instanceRoot, runtimePath)).href : new URL('./evaluation.mjs', import.meta.url).href);
  if (typeof runtime?.bootstrap !== 'function' || typeof runtime?.wave !== 'function')
    throw new Error('Provide skills.runtimeModule exporting bootstrap and wave, or values.runtime');
  const task = values.task ?? action.task;
  const userinfo = action.userinfo ?? config.userinfo;
  const generate = async options => {
    let { modelRuntime, model } = values;
    if (!modelRuntime || !model) {
      if (modelRuntime || model) throw new Error('Inject both modelRuntime and model');
      const profile = config.skill_writer_actor ?? config.actor;
      if (!profile) throw new Error('Configure skill_writer_actor or actor for native skill generation');
      const { modelFromProfile } = await import('../controller/model.mjs');
      ({ modelRuntime, model } = await modelFromProfile(profile));
    }
    const { generateCandidate } = await import('../controller/skill-evolution.mjs');
    return generateCandidate({ ...options, modelRuntime, model });
  };
  const evaluate = async request => {
    try {
      await access(request.instanceRoot);
      throw new Error('Evaluation instance already exists');
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    await runtime.bootstrap(request);
    return runtime.wave({ instanceRoot: request.instanceRoot, config: request.config,
      projectIds: [request.projectId], evaluation: true });
  };
  return tune({ instanceRoot, config, configPath: values.configPath ?? join(instanceRoot, 'instance.json'),
    task, userinfo, evaluate, generate, native: values.native });
}
