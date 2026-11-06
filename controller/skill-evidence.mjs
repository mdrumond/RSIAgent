import { createHash } from 'node:crypto';
import { relative } from 'node:path';
import { fingerprint } from './evidence.mjs';

const digest = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const keys = ['goal', 'startingSource', 'verifier', 'targetRuntime'];
const complete = identity => keys.every(key => /^[a-f0-9]{64}$/.test(identity?.[key] ?? ''));
export const skillEventTypes = ['skill-read', 'knowledge-retrieved', 'guidance-contradicted',
  'failure-recorded', 'verification', 'stage-completed', 'attempt-error', 'rejected', 'accepted'];
const contextFor = (state, attemptId) => state.queryEvents(attemptId,
  { types: ['skill-evidence-context'], limit: 1 })[0]?.data;
export const skillEvents = (state, attemptId, limit = 1000) => state.queryEvents(attemptId,
  { types: skillEventTypes, limit, order: 'desc' }).reverse().map(({ type, data }) => ({ type, data }));

export function environmentBinding(config, project, resource) {
  const options = config.environment?.options ?? {}, metadata = resource?.metadata ?? {};
  if (!options.mode || (options.mode !== 'local' && (!options.target || !options.runtime))) return undefined;
  // Health timestamps/utilization change every admission; they are not runtime identity.
  return { module: config.environment.module, mode: options.mode,
    target: metadata.target ?? options.target ?? 'local', runtime: metadata.runtime ?? options.runtime ?? null,
    product: metadata.product ?? null, profile: project.profile ?? null,
    environmentHash: digest(Object.entries(options.env ?? {}).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)),
    localHost: options.mode === 'local' ? [process.platform, process.arch, process.version] : null };
}

// Capture once before actor edits; resumed attempts must retain their original inputs.
export async function recordSkillContext({ state, attemptId, project, verifierHashes, environmentIdentity }) {
  const previous = contextFor(state, attemptId);
  if (previous) return previous;
  // Historical sessions may already have edited source before this feature existed.
  if (state.attempt(attemptId).session) return undefined;
  const identity = {
    goal: digest([project.goal, project.performance_goal ?? null]),
    startingSource: await fingerprint(project.source),
    verifier: digest([project.verifier.command, project.verifier.performance_command ?? null,
      Object.entries(verifierHashes).map(([path, hash]) => [relative(project.source, path), hash])
        .sort(([a], [b]) => a.localeCompare(b))]),
    targetRuntime: environmentIdentity ? digest(environmentIdentity) : null
  };
  state.event(attemptId, 'skill-evidence-context', identity);
  return identity;
}

export function tuningTask({ state, attemptId, kind, phase }) {
  const attempt = state.attempt(attemptId), events = skillEvents(state, attemptId);
  return { id: attemptId, projectId: attempt.project, kind, phase, completed: attempt.outcome != null,
    outcome: attempt.outcome, comparisonIdentity: contextFor(state, attemptId),
    events };
}

const summarize = task => ({ attemptId: task?.id ?? null, projectId: task?.projectId ?? null,
  outcome: task?.outcome ?? null, identity: task?.comparisonIdentity ?? null,
  references: (task?.events ?? []).filter(event =>
    ['verification', 'knowledge-retrieved', 'failure-recorded', 'attempt-error', 'rejected', 'accepted'].includes(event.type)) });

// Missing identities are a reason to omit a comparison, never to acquire another run.
export function selectSkillEvidence(task, history = []) {
  const current = summarize(task), comparisons = {};
  if (task?.id && complete(task.comparisonIdentity)) {
    const seen = new Set([task.id]);
    for (const previous of [...history].reverse()) {
      if (!previous.id || seen.has(previous.id)) continue;
      seen.add(previous.id);
      if (!previous.completed || !['passed', 'failed'].includes(previous.outcome) ||
          !complete(previous.comparisonIdentity) ||
          !keys.every(key => previous.comparisonIdentity[key] === task.comparisonIdentity[key])) continue;
      const role = previous.outcome === 'passed' ? 'success' : 'failure';
      comparisons[role] ??= summarize(previous);
      if (comparisons.success && comparisons.failure) break;
    }
  }
  return { current, comparisons };
}
