import { mkdir, readFile, writeFile, rename, cp } from 'node:fs/promises';
import { join, resolve, relative, dirname, isAbsolute } from 'node:path';
import { randomUUID } from 'node:crypto';
import { stages } from './workflow.mjs';
import { generateCandidate } from './skill-proposal.mjs';
export { generateCandidate };

const scopes = ['global-kdb', ...stages];
const json = async path => JSON.parse(await readFile(path, 'utf8'));
const save = async (path, value, signal) => {
  const tmp = `${path}.${randomUUID()}.tmp`;
  await writeFile(tmp, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
  signal?.throwIfAborted();
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
  task, userinfo, manual = false, context = {}, evaluate, generate = generateCandidate, native, modelRuntime, model, signal }) {
  signal?.throwIfAborted();
  const current = await json(configPath);
  const mode = task?.phase ?? config?.mode ?? current.mode;
  if (current.skills?.tuningDisabled || mode === 'phase1') return { analyse: false, evaluate: false };
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
    await save(statePath, state, signal);
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
    signal?.throwIfAborted();
    const changes = await generate({ directory: proposalRoot, input: { entries, history, originalCount: new Set(counted.map(x => x.id)).size, userinfo, context: { ...entry, userinfo },
      actualEvents: events.filter(e => ['skill-read', 'knowledge-retrieved'].includes(e.type)) },
      validate, native, modelRuntime, model, signal, seconds: current.skill_writer_seconds ?? 240 });
    signal?.throwIfAborted();
    if (!Array.isArray(changes)) throw new Error('Candidate must contain changes array');
    const updated = validate(changes);
    const candidate = await writeVersion(root, updated, source);
    signal?.throwIfAborted();
    state.history = history;
    state.pending = shouldEvaluate ? { active, candidate, projects: new Set(counted.map(x => x.projectId)).size,
      projectIds, proposalRoot, manual } : null;
    state.candidate = candidate;
    await save(statePath, state, signal);
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
  signal?.throwIfAborted();
  await mkdir(evaluationRoot, { recursive: true });
  await save(join(evaluationRoot, 'results.json'), results);
  for (const projectId of ids) for (let run = 0; run < runs; run++) {
    for (const arm of run % 2 ? ['candidate', 'active'] : ['active', 'candidate']) {
      signal?.throwIfAborted();
      const isolatedRoot = join(evaluationRoot, randomUUID());
      const skillVersion = pending[arm];
      const isolatedConfig = structuredClone(current);
      isolatedConfig.root = isolatedRoot;
      isolatedConfig.mode = mode;
      isolatedConfig.skills = { ...isolatedConfig.skills, activeVersion: skillVersion, tuningDisabled: true };
      const result = await evaluate({ instanceRoot: isolatedRoot, config: isolatedConfig,
        sourceInstanceRoot: root, projectId, run, arm, skillVersion, signal });
      results.push({ ...result, projectId, run, arm, instanceRoot: isolatedRoot });
      await save(join(evaluationRoot, 'results.json'), results);
    }
  }
  signal?.throwIfAborted();
  const verdict = evaluateDecision(results, ids, runs);
  await save(join(evaluationRoot, 'decision.json'), { ...verdict, ...pending });
  if (verdict.promote) {
    const latest = await json(configPath);
    if (latest.skills.activeVersion !== pending.active) throw new Error('Active skills changed during evaluation');
    latest.skills.activeVersion = pending.candidate;
    latest.revision = (latest.revision ?? 0) + 1;
    await save(configPath, latest, signal);
  }
  signal?.throwIfAborted();
  if (!pending.manual) state.lastEvaluated = pending.projects;
  state.pending = null;
  await save(statePath, state, signal);
  return { ...verdict, candidate: pending.candidate, evaluationRoot };
}
