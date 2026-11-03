import { mkdir, readFile, writeFile, rename } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { stages } from './workflow.mjs';

const scopes = ['global-kdb', ...stages];
const json = async path => JSON.parse(await readFile(path, 'utf8'));
const save = async (path, value) => {
  const tmp = `${path}.${randomUUID()}.tmp`;
  await writeFile(tmp, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
  await rename(tmp, path);
};
const original = task => task?.kind === 'original' && task.completed === true &&
  ['phase2', 'inference'].includes(task.phase) && task.id && task.projectId;

// History contains only completed originals; retries/diagnostics cannot advance cadence.
export function cadenceDecision(history, task, lastEvaluated = 0) {
  if (!original(task) || history.some(x => x.id === task.id))
    return { analyse: false, evaluate: false };
  const projects = new Set([...history, task].map(x => x.projectId)).size;
  return { analyse: true, evaluate: projects - lastEvaluated >= 8, projects,
    allowPruning: new Set([...history, task].map(x => x.id)).size >= 24 };
}

// A paired improvement must be repeated, and every candidate correctness result must pass.
export function evaluateDecision(results, projectIds, runs = 2) {
  if (projectIds.length !== 2 || new Set(projectIds).size !== 2 || !Number.isInteger(runs) || runs < 2)
    throw new Error('Require two distinct regression projects and at least two runs');
  const expected = projectIds.flatMap(projectId => Array.from({ length: runs }, (_, run) =>
    ['active', 'candidate'].map(arm => ({ projectId, run, arm })))).flat();
  if (results.length !== expected.length) return { promote: false, reason: 'inconclusive' };
  const paired = [];
  for (const key of expected) {
    const matches = results.filter(r => r.projectId === key.projectId && r.run === key.run && r.arm === key.arm);
    if (matches.length !== 1) return { promote: false, reason: 'inconclusive' };
    const result = matches[0];
    if (result.correct === false && key.arm === 'candidate')
      return { promote: false, reason: 'correctness-regression' };
    if (result.correct !== true || !Number.isFinite(result.score))
      return { promote: false, reason: 'inconclusive' };
    paired.push(result);
  }
  const improvements = paired.filter(r => r.arm === 'candidate').map(r => r.score -
    paired.find(a => a.arm === 'active' && a.projectId === r.projectId && a.run === r.run).score);
  return improvements.every(x => x > 0) ? { promote: true, reason: 'repeated-improvement' }
    : { promote: false, reason: 'inconclusive' };
}

// Structured entries keep negative and critical guidance identifiable across revisions.
export function applyProposal(current, changes, { allowPruning, history }) {
  const entries = new Map(current.map(x => [x.id, x]));
  for (const change of changes) {
    const old = entries.get(change.id);
    if (change.action === 'remove' || old) {
      if (!old) throw new Error('Unknown guidance entry');
      const evidence = change.contradictoryEvidence;
      const contradicted = Array.isArray(evidence) && evidence.length > 0 && evidence.every(ref =>
        history.some(task => task.events.some(event => event.type === 'guidance-contradicted' &&
          event.data?.guidanceId === old.id && event.data?.evidence === ref)));
      if (old.negative || (old.critical && !contradicted)) throw new Error('Protected guidance');
      const used = history.some(task => task.events.some(event => event.type === 'skill-read' &&
        event.data?.stage === old.scope));
      if (!contradicted && (!allowPruning || used)) throw new Error('Pruning requires 24 originals and unused guidance');
    }
    if (change.action === 'remove') entries.delete(change.id);
    else {
      if (change.action !== 'upsert' || !/^[a-z0-9-]+$/.test(change.id) ||
          !scopes.includes(change.scope) || !change.text?.trim() ||
          typeof change.critical !== 'boolean' || typeof change.negative !== 'boolean')
        throw new Error('Invalid guidance proposal');
      entries.set(change.id, { id: change.id, scope: change.scope, text: change.text,
        critical: change.critical, negative: change.negative });
    }
  }
  return [...entries.values()];
}

// One native Pi turn; SDK owns tools, context, provider retry, and native session persistence.
export async function generateCandidate({ directory, input, modelRuntime, model, native }) {
  const sdk = native ?? await import('@earendil-works/pi-coding-agent');
  await mkdir(directory, { recursive: true });
  const agentDir = join(directory, 'agent');
  await mkdir(agentDir, { recursive: true });
  const settingsManager = sdk.SettingsManager.inMemory({ compaction: { enabled: true }, retry: { enabled: true } });
  const manager = sdk.SessionManager.create(directory, directory);
  const loader = new sdk.DefaultResourceLoader({ cwd: directory, agentDir, settingsManager,
    noExtensions: true, noSkills: true, noContextFiles: true,
    extensionFactories: [pi => {
      for (const type of ['before_provider_request', 'provider_stream_event'])
        pi.on(type, async payload => {
          const { appendFile } = await import('node:fs/promises');
          await appendFile(join(directory, 'provider.jsonl'), JSON.stringify(payload) + '\n');
        });
    }] });
  await loader.reload();
  const { session } = await sdk.createAgentSession({ cwd: directory, agentDir, settingsManager, modelRuntime, model,
    tools: [], sessionManager: manager, resourceLoader: loader });
  const responses = [];
  const unsubscribe = session.subscribe(event => {
    if (event.type === 'message_end' && event.message.role === 'assistant') responses.push(event.message);
  });
  try {
    await session.bindExtensions({});
    await session.prompt(`Propose English Ascend workflow guidance changes as JSON {"changes":[]} only.
Prioritize error avoidance, detection and recovery before performance. Retain negative lessons.
Use actual KDB retrieval and skill-read events as evidence, never inferred tool use.
Do not remove critical guidance without contradictory evidence; unused guidance requires 24 distinct originals.
Each change: action upsert/remove, id, scope, text, critical boolean, negative boolean,
optional contradictoryEvidence (references to recorded guidance-contradicted events).
Existing entries and original-task evidence:\n${JSON.stringify(input)}`);
    await session.waitForIdle();
    await save(join(directory, 'responses.json'), responses);
    const last = responses.at(-1);
    if (!last || last.stopReason === 'error') throw new Error('Native candidate generation failed');
    const text = last.content.filter(x => x.type === 'text').map(x => x.text).join('\n');
    return JSON.parse(text).changes;
  } finally {
    unsubscribe?.();
    session.dispose();
  }
}

async function writeVersion(root, entries) {
  const version = randomUUID();
  const directory = join(root, 'skills', 'versions', version);
  await mkdir(directory, { recursive: true });
  await writeFile(join(directory, 'guidance.json'), JSON.stringify(entries, null, 2), { flag: 'wx' });
  const skillMap = {};
  for (const scope of scopes) {
    const path = join(directory, `${scope}.md`);
    const selected = entries.filter(x => x.scope === scope).sort((a, b) => Number(b.critical) - Number(a.critical));
    await writeFile(path, `# ${scope}\n\n${selected.map(x => x.text).join('\n\n')}\n`, { flag: 'wx' });
    skillMap[scope] = `${scope}.md`;
  }
  await writeFile(join(directory, 'skill-map.json'), JSON.stringify(skillMap, null, 2), { flag: 'wx' });
  return version;
}

// Caller serializes tuning per instance. configPath is the authoritative current instance config.
// evaluate({instanceRoot,config,projectId,run,arm,skillVersion}) must bootstrap a fresh isolated
// instance, execute its canonical original wave, retain raw/SFT/evidence there, and return
// {correct:boolean,score:number} (higher is better). No releases are read or adopted.
export async function tune({ instanceRoot, config, configPath = join(instanceRoot, 'instance.json'),
  task, userinfo, evaluate, generate = generateCandidate, native, modelRuntime, model }) {
  const current = await json(configPath);
  if (current.skills?.tuningDisabled || current.mode === 'phase1') return { analyse: false, evaluate: false };
  if (config?.skills?.activeVersion !== current.skills?.activeVersion)
    throw new Error('Stale instance configuration');
  const root = resolve(instanceRoot);
  const directory = join(root, 'skills');
  await mkdir(directory, { recursive: true });
  const statePath = join(directory, 'evolution.json');
  let state;
  try { state = await json(statePath); } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    state = { history: [], lastEvaluated: 0, pending: null };
  }
  if (state.pending && current.skills.activeVersion === state.pending.candidate) {
    state.lastEvaluated = state.pending.projects;
    state.pending = null;
    await save(statePath, state);
    return { promote: true, reason: 'recovered-promotion' };
  }
  const decision = cadenceDecision(state.history, task, state.lastEvaluated);
  if (!decision.analyse && !state.pending) return decision;
  if (task?.phase === 'phase1') return { analyse: false, evaluate: false };
  if (!state.pending) {
    const events = task.events ?? [];
    const entry = { ...task, events };
    const history = [...state.history, entry];
    const active = current.skills?.activeVersion;
    if (!/^[a-zA-Z0-9-]+$/.test(active ?? '')) throw new Error('Instance requires an active skill version');
    const entries = await json(join(directory, 'versions', state.candidate ?? active, 'guidance.json'));
    const proposalRoot = join(directory, 'proposals', randomUUID());
    await mkdir(proposalRoot, { recursive: true });
    await save(join(proposalRoot, 'evidence.json'), entry);
    const changes = await generate({ directory: proposalRoot, input: { entries, history, userinfo,
      actualEvents: events.filter(e => ['skill-read', 'knowledge-retrieved'].includes(e.type)) },
      native, modelRuntime, model });
    if (!Array.isArray(changes)) throw new Error('Candidate must contain changes array');
    const updated = applyProposal(entries, changes, { allowPruning: decision.allowPruning, history });
    const candidate = await writeVersion(root, updated);
    state.history = history;
    state.pending = decision.evaluate ? { active, candidate, projects: decision.projects, proposalRoot } : null;
    state.candidate = candidate;
    await save(statePath, state);
    if (!decision.evaluate) return { ...decision, candidate };
  }
  if (typeof evaluate !== 'function') throw new Error('Canonical evaluator required');
  const ids = current.skills?.tuningProjectIds ?? [];
  const runs = current.skills?.runsPerArm ?? 2;
  evaluateDecision([], ids, runs); // Validate design before executing any wave.
  const pending = state.pending;
  if (current.skills.activeVersion !== pending.active) throw new Error('Active skills changed during evaluation');
  const results = [];
  const evaluationRoot = join(directory, 'evaluations', randomUUID());
  await mkdir(evaluationRoot, { recursive: true });
  for (const projectId of ids) for (let run = 0; run < runs; run++) {
    for (const arm of run % 2 ? ['candidate', 'active'] : ['active', 'candidate']) {
      const isolatedRoot = join(evaluationRoot, randomUUID());
      const skillVersion = pending[arm];
      const isolatedConfig = structuredClone(current);
      isolatedConfig.root = isolatedRoot;
      isolatedConfig.skills = { ...isolatedConfig.skills, activeVersion: skillVersion, tuningDisabled: true };
      const result = await evaluate({ instanceRoot: isolatedRoot, config: isolatedConfig,
        sourceInstanceRoot: root, projectId, run, arm, skillVersion });
      results.push({ ...result, projectId, run, arm, instanceRoot: isolatedRoot });
      await save(join(evaluationRoot, 'results.json'), results);
    }
  }
  const verdict = evaluateDecision(results, ids, runs);
  await save(join(evaluationRoot, 'decision.json'), { ...verdict, ...pending });
  if (verdict.promote) {
    const latest = await json(configPath);
    if (latest.skills.activeVersion !== pending.active) throw new Error('Active skills changed during evaluation');
    latest.skills.activeVersion = pending.candidate;
    await save(configPath, latest);
  }
  state.lastEvaluated = pending.projects;
  state.pending = null;
  await save(statePath, state);
  return { ...verdict, candidate: pending.candidate, evaluationRoot };
}
