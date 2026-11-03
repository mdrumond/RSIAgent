import { mkdir, readFile, writeFile, rename, cp, appendFile } from 'node:fs/promises';
import { join, resolve, relative, dirname, isAbsolute } from 'node:path';
import { randomUUID } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { stages } from './workflow.mjs';
import { Type } from 'typebox';

const scopes = ['global-kdb', ...stages];
const json = async path => JSON.parse(await readFile(path, 'utf8'));
const save = async (path, value) => {
  const tmp = `${path}.${randomUUID()}.tmp`;
  await writeFile(tmp, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
  await rename(tmp, path);
};
const completed = task => task?.completed === true && task.id && task.projectId &&
  ['phase2', 'inference'].includes(task.phase);
const originals = history => history.filter(task => completed(task) && task.kind === 'original');
const usageTypes = new Set(['skill-read', 'knowledge-retrieved', 'guidance-contradicted',
  'verification', 'stage-completed', 'attempt-error', 'rejected', 'accepted']);
const compactTask = task => ({ id: task.id, projectId: task.projectId, kind: task.kind,
  phase: task.phase, completed: task.completed, outcome: task.outcome,
  events: (task.events ?? []).filter(event => usageTypes.has(event.type)) });

export function cadenceDecision(history, task, lastEvaluated = 0) {
  if (!completed(task) || history.some(x => x.id === task.id)) return { analyse: false, evaluate: false };
  const counted = originals([...history, task]);
  const projects = new Set(counted.map(x => x.projectId)).size;
  return { analyse: true, evaluate: task.kind === 'original' && projects - lastEvaluated >= 8, projects,
    allowPruning: new Set(counted.map(x => x.id)).size >= 24 };
}

// Correctness improvements precede timing; every candidate correctness result must pass.
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
    if (result.correct !== true && !(key.arm === 'active' && result.correct === false))
      return { promote: false, reason: 'inconclusive' };
    paired.push(result);
  }
  if (paired.some(r => r.arm === 'active' && r.correct === false))
    return { promote: true, reason: 'reliability-improvement' };
  if (paired.some(r => !Number.isFinite(r.score)))
    return { promote: false, reason: 'inconclusive' };
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

// Native Pi owns tool turns, correction, context, retry and compaction.
export async function generateCandidate({ directory, input, modelRuntime, model, native,
  seconds = 240, extensions = [], validate = () => {} }) {
  if (!Number.isFinite(seconds) || seconds <= 0) throw new Error('Writer elapsed seconds must be positive');
  const started = performance.now();
  const sdk = native ?? await import('@earendil-works/pi-coding-agent');
  await mkdir(directory, { recursive: true });
  await save(join(directory, 'current-guidance.json'), input.entries ?? []);
  await save(join(directory, 'current-usage.json'), { context: input.context, userinfo: input.userinfo,
    actualEvents: input.actualEvents, originals: originals(input.history ?? []).length });
  await writeFile(join(directory, 'usage-history.jsonl'), (input.history ?? []).map(x => JSON.stringify(x)).join('\n'));
  const agentDir = join(directory, 'agent');
  await mkdir(agentDir, { recursive: true });
  const settingsManager = sdk.SettingsManager.inMemory({ compaction: { enabled: true }, retry: { enabled: true } });
  const manager = sdk.SessionManager.create(directory, directory);
  let submitted;
  const id = Type.String({ pattern: '^[a-z0-9-]+$' });
  const contradiction = { contradictoryEvidence: Type.Optional(Type.Array(Type.String())) };
  const parameters = Type.Object({ changes: Type.Array(Type.Union([
    Type.Object({ action: Type.Literal('remove'), id, ...contradiction }),
    Type.Object({ action: Type.Literal('upsert'), id, scope: Type.Union(scopes.map(scope => Type.Literal(scope))),
      text: Type.String({ minLength: 1 }), critical: Type.Boolean(), negative: Type.Boolean(), ...contradiction })
  ])) });
  const extension = pi => {
    pi.registerTool({ name: 'submit_changes', label: 'Submit skill changes',
      description: 'Validate and submit candidate guidance. Correct any rejection using the returned reason.',
      parameters, execute: async (_id, { changes }) => {
        await validate(changes);
        submitted = changes;
        return { content: [{ type: 'text', text: 'Candidate changes accepted.' }], details: {} };
      } });
    for (const type of ['before_provider_request', 'provider_stream_event'])
      pi.on(type, async payload => appendFile(join(directory, 'provider.jsonl'), JSON.stringify(payload) + '\n'));
  };
  const loader = new sdk.DefaultResourceLoader({ cwd: directory, agentDir, settingsManager,
    noExtensions: true, noSkills: true, noContextFiles: true, extensionFactories: [extension, ...extensions] });
  await loader.reload();
  const { session } = await sdk.createAgentSession({ cwd: directory, agentDir, settingsManager, modelRuntime, model,
    tools: ['read', 'grep', 'find', 'ls', 'submit_changes', 'mcp__kdb__search', 'mcp__kdb__read_entry'],
    sessionManager: manager, resourceLoader: loader });
  const responses = [];
  const unsubscribe = session.subscribe(event => {
    if (event.type === 'message_end') responses.push(event.message);
  });
  let timer, expired = false;
  try {
    const remaining = seconds * 1000 - (performance.now() - started);
    if (remaining <= 0) throw new Error('Writer elapsed allocation exhausted');
    timer = setTimeout(() => { expired = true; void session.abort().catch(() => {}); }, remaining);
    await session.bindExtensions({});
    await session.prompt(`Read current-guidance.json and current-usage.json with native read tools.
Use usage-history.jsonl only for relevant prior usage. Resolve cited collection/entry_id with mcp__kdb__read_entry
and search positive/negative evidence with mcp__kdb__search; indexed source files may be absent.
Propose English guidance prioritizing error avoidance, detection and recovery before performance. Retain negatives.
Preserve critical guidance unless recorded contradictory evidence supports replacement; unused pruning needs 24 distinct originals.
Submit via submit_changes; correct any rejection. A final text response does not publish changes.`);
    await session.waitForIdle();
    if (expired) throw new Error('Writer elapsed allocation exhausted');
    if (!submitted) throw new Error('Native writer did not submit validated changes');
    return submitted;
  } finally {
    clearTimeout(timer);
    try {
      await save(join(directory, 'responses.json'), responses);
      const messages = manager.getEntries?.().filter(entry => entry.type === 'message').map(entry => entry.message) ?? responses;
      await writeFile(join(directory, 'sft.native.jsonl'), messages.map(x => JSON.stringify(x)).join('\n'));
      await session.extensionRunner?.emit({ type: 'session_shutdown', reason: 'quit' });
    } finally { unsubscribe?.(); session.dispose(); }
  }
}

async function readEntries(directory) {
  try { return await json(join(directory, 'guidance.json')); } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    const map = await json(join(directory, 'skill-map.json'));
    return Promise.all(scopes.map(async scope => ({ id: `${scope}-baseline`, scope,
      text: await readFile(resolve(directory, map[scope]), 'utf8'), critical: true, negative: false })));
  }
}

async function writeVersion(root, entries, source) {
  const version = randomUUID();
  const directory = join(root, 'skills', 'versions', version);
  await cp(source, directory, { recursive: true, errorOnExist: true, force: false });
  await writeFile(join(directory, 'guidance.json'), JSON.stringify(entries, null, 2));
  let originalMap = {};
  try { originalMap = await json(join(source, 'skill-map.json')); } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  const skillMap = {};
  for (const scope of scopes) {
    const mapped = relative(source, resolve(source, originalMap[scope] ?? `${scope}.md`));
    if (mapped.startsWith('../') || isAbsolute(mapped)) throw new Error('Mapped skills must belong to their version');
    const name = join(dirname(mapped), `evolved-${scope}.md`);
    const selected = entries.filter(x => x.scope === scope).sort((a, b) => Number(b.critical) - Number(a.critical));
    await writeFile(join(directory, name), `# ${scope}\n\n${selected.map(x => x.text).join('\n\n')}\n`);
    skillMap[scope] = name;
  }
  await writeFile(join(directory, 'skill-map.json'), JSON.stringify(skillMap, null, 2));
  return version;
}

// Caller serializes tuning per instance. configPath is the authoritative current instance config.
// evaluate({instanceRoot,config,projectId,run,arm,skillVersion}) must bootstrap a fresh isolated
// instance, execute its canonical original wave, retain raw/SFT/evidence there, and return
// {correct:boolean,score:number} (higher is better). No releases are read or adopted.
export async function tune({ instanceRoot, config, configPath = join(instanceRoot, 'instance.json'),
  task, userinfo, manual = false, context = {}, evaluate, generate = generateCandidate, native, modelRuntime, model }) {
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
    if (!state.pending.manual) state.lastEvaluated = state.pending.projects;
    state.pending = null;
    await save(statePath, state);
    return { promote: true, reason: 'recovered-promotion' };
  }
  const decision = cadenceDecision(state.history, task, state.lastEvaluated);
  if (!decision.analyse && !manual && !state.pending) return decision;
  if (task?.phase === 'phase1') return { analyse: false, evaluate: false };
  if (!state.pending) {
    const events = (manual ? context.events : task?.events) ?? [];
    const rawEntry = manual ? { ...context, userinfo, manual: true, events } : { ...task, events };
    const entry = compactTask(rawEntry);
    const history = (manual ? state.history : [...state.history, entry]).map(compactTask);
    const counted = originals(history);
    const projectIds = current.skills?.tuningProjectIds ?? [...new Set(counted.map(x => x.projectId))].sort().slice(0, 2);
    const shouldEvaluate = Boolean(decision.evaluate || (manual &&
      (current.skills?.tuningProjectIds !== undefined || projectIds.length >= 2)));
    const active = current.skills?.activeVersion;
    if (!/^[a-zA-Z0-9-]+$/.test(active ?? '')) throw new Error('Instance requires an active skill version');
    const source = join(directory, 'versions', state.candidate ?? active);
    const entries = await readEntries(source);
    const proposalRoot = join(directory, 'proposals', randomUUID());
    await mkdir(proposalRoot, { recursive: true });
    await save(join(proposalRoot, 'evidence.json'), rawEntry);
    const validate = changes => applyProposal(entries, changes, {
      allowPruning: new Set(counted.map(x => x.id)).size >= 24,
      history: manual ? [...history, entry] : history });
    const changes = await generate({ directory: proposalRoot, input: { entries, history, userinfo, context: { ...entry, userinfo },
      actualEvents: events.filter(e => ['skill-read', 'knowledge-retrieved'].includes(e.type)) },
      validate, native, modelRuntime, model, seconds: current.skill_writer_seconds ?? 240 });
    if (!Array.isArray(changes)) throw new Error('Candidate must contain changes array');
    const updated = validate(changes);
    const candidate = await writeVersion(root, updated, source);
    state.history = history;
    state.pending = shouldEvaluate ? { active, candidate, projects: new Set(counted.map(x => x.projectId)).size,
      projectIds, proposalRoot, manual } : null;
    state.candidate = candidate;
    await save(statePath, state);
    if (!shouldEvaluate) return { ...decision, analyse: true, evaluate: false, candidate };
  }
  if (typeof evaluate !== 'function') throw new Error('Canonical evaluator required');
  const ids = state.pending.projectIds ?? current.skills?.tuningProjectIds ??
    [...new Set(originals(state.history).map(x => x.projectId))].sort().slice(0, 2);
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
    latest.revision = (latest.revision ?? 0) + 1;
    await save(configPath, latest);
  }
  if (!pending.manual) state.lastEvaluated = pending.projects;
  state.pending = null;
  await save(statePath, state);
  return { ...verdict, candidate: pending.candidate, evaluationRoot };
}
