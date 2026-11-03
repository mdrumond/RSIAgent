import { readFile, access, appendFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { randomUUID } from 'node:crypto';
import { tune } from '../controller/skill-evolution.mjs';

// Runtime module exports bootstrap({instanceRoot,config,sourceInstanceRoot,skillVersion})
// and wave({instanceRoot,config,projectIds:[id],evaluation:true}). bootstrap must copy
// the selected immutable skill version from sourceInstanceRoot, project inputs and frozen
// verifiers into a new instance. wave returns {correct:boolean,score:number}; higher wins.
// Both functions retain native provider/session/SFT and verifier evidence inside that instance.
// Programmatic callers may supply values.runtime with these exact functions instead.
export async function run(config, values = {}) {
  if (config.skills?.tuningDisabled || (values.task?.phase ?? config.mode) === 'phase1')
    return { analyse: false, evaluate: false };
  const instanceRoot = resolve(config.root);
  const executedMode = values.task?.phase ?? config.mode;
  const lock = values.lock ?? (await import('./evolution-lock.mjs')).withSkillLock;
  return await lock(config, async () => {
    const readConfig = values.readConfig ?? (await import('./instance.mjs')).readConfig;
    config = { ...await readConfig(instanceRoot), root: instanceRoot };
    if (values.task) config.mode = executedMode;
    const action = values.config ? JSON.parse(await readFile(resolve(values.config), 'utf8')) : {};
    const task = values.task ?? action.task;
    if (task) config.mode = task.phase ?? executedMode;
    if (config.skills?.tuningDisabled || config.mode === 'phase1') return { analyse: false, evaluate: false };
    const userinfo = action.userinfo ?? config.userinfo;
    const manual = !task && (values.manual === true || action.userinfo !== undefined || action.projectId !== undefined);
    const context = { projectId: action.projectId, events: [] };
    if (manual) {
      try {
        await access(join(instanceRoot, 'state.sqlite'));
        const { State } = await import('../controller/state.mjs');
        const state = new State(join(instanceRoot, 'state.sqlite'));
        try {
          if (context.projectId && !state.project(context.projectId)) throw new Error('Unknown tuning project');
          const attempts = context.projectId
            ? state.db.prepare('SELECT id FROM attempts WHERE project=? ORDER BY created').all(context.projectId)
            : state.db.prepare('SELECT id FROM attempts ORDER BY created').all();
          context.events = attempts.flatMap(attempt => state.events(attempt.id));
        } finally { state.close(); }
      } catch (error) { if (error.code !== 'ENOENT') throw error; }
    }
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
      const { ensureKnowledgeService } = values.knowledge ?? await import('./knowledge.mjs');
      const { createKdbBridge } = values.knowledge ?? await import('../integrations/kdb-extension.mjs');
      const url = await ensureKnowledgeService(config);
      const bridge = createKdbBridge({ url, attemptId: task?.id ?? `manual-${randomUUID()}`,
        agentDir: join(options.directory, 'agent'), event: async (type, data) =>
          appendFile(join(options.directory, 'knowledge-events.jsonl'), JSON.stringify({ type, data }) + '\n') });
      return generateCandidate({ ...options, modelRuntime, model, extensions: [bridge.extension] });
    };
    const evaluate = async request => {
      const runtime = values.runtime ?? await import(config.skills?.runtimeModule
        ? pathToFileURL(resolve(instanceRoot, config.skills.runtimeModule)).href
        : new URL('./evaluation.mjs', import.meta.url).href);
      if (typeof runtime?.bootstrap !== 'function' || typeof runtime?.wave !== 'function')
        throw new Error('Provide runtime bootstrap and wave');
      try {
        await access(request.instanceRoot);
        throw new Error('Evaluation instance already exists');
      } catch (error) { if (error.code !== 'ENOENT') throw error; }
      await runtime.bootstrap(request);
      return runtime.wave({ instanceRoot: request.instanceRoot, config: request.config,
        projectIds: [request.projectId], evaluation: true });
    };
    return tune({ instanceRoot, config, configPath: values.configPath ?? join(instanceRoot, 'instance.json'),
      task, userinfo, manual, context, evaluate, generate, native: values.native });
  });
}
