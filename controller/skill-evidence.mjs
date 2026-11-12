import { createHash } from 'node:crypto';
import { relative } from 'node:path';
import { fingerprint } from './evidence.mjs';

const digest = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const keys = ['goal', 'startingSource', 'verifier', 'targetRuntime'];
const complete = identity => keys.every(key => /^[a-f0-9]{64}$/.test(identity?.[key] ?? ''));
export const skillEventTypes = ['skill-read', 'knowledge-retrieved', 'guidance-contradicted',
  'failure-recorded', 'verification', 'stage-completed', 'attempt-error', 'rejected', 'accepted',
  'device-lease-blocked', 'execution-unresolved'];
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
  observations: skillObservations(task),
  references: (task?.events ?? []).filter(event =>
    ['verification', 'knowledge-retrieved', 'failure-recorded', 'attempt-error', 'rejected', 'accepted'].includes(event.type)) });

// Attribution is evidence, not an instruction for actors to repair orchestration.
// Unrecognised errors remain unknown; an unsuccessful outcome alone has no cause.
export function skillObservations(task = {}) {
  return (task.events ?? []).map(({type,data = {}}, index) => {
    const message = data.message ?? '';
    const harness = ['device-lease-blocked','execution-unresolved'].includes(type) ||
      /Device reservations require reconciliation|Execution closure unknown|Previous execution is not confirmed stopped|Observed NPU runtime differs|Qualification runtime differs/.test(message);
    const exhausted = type === 'rejected' && data.remaining === 0;
    const failure = harness || ['failure-recorded','attempt-error','rejected'].includes(type);
    return {reference:`${task.id ?? 'current'}:${index}`,type,
      ...(failure ? {responsibility:harness?'harness':exhausted?'actor-workflow':'unknown',
        actorResponse:harness?'report-blocker':exhausted?'reserve-validation-runway':'investigate'} : {}),
      ...(Number.isFinite(data.actor_elapsed)?{actorClockSeconds:data.actor_elapsed}:{}),
      ...(Number.isFinite(data.durationSeconds)&&data.durationSeconds>=0
        ?{durationSeconds:data.durationSeconds}:{}),
      ...(data.stage?{stage:data.stage}:{}),
      ...(message?{message}:{}),
      ...(type==='stage-completed'?{reflection:data.reflection}:{}),
    };
  });
}

// Writer inputs separate cumulative allocation clocks from measured operations.
// The original event journal remains unchanged for audit and actor resume.
export function writerEvidence(value) {
  if(Array.isArray(value))return value.map(writerEvidence);
  if(!value||typeof value!=='object')return value;
  return Object.fromEntries(Object.entries(value).map(([key,item])=>
    [key==='actor_elapsed'?'actorClockSeconds':key,writerEvidence(item)]));
}

export function validateSkillGrounding(changes, observations = []) {
  const byReference = new Map(observations.map(item=>[item.reference,item]));
  const attributed = observations.some(item=>item.responsibility==='harness');
  for(const change of changes){
    if(change.action==='remove')continue;
    if(attributed&&(!change.evidenceRefs?.length||!change.response))
      throw new Error('Attributed failures require evidenceRefs and an actor response');
    const refs=(change.evidenceRefs??[]).map(ref=>{
      const item=byReference.get(ref);if(!item)throw new Error('Unknown skill evidence reference');return item;
    });
    if(refs.some(item=>item.responsibility==='harness')&&change.response!=='report-blocker')
      throw new Error('Harness failures permit reporting a blocker, not actor repair');
    if(change.response==='report-blocker'&&!['global-kdb','preflight'].includes(change.scope))
      throw new Error('Harness blocker guidance belongs before device work');
    if(change.response==='reserve-validation-runway'&&!['global-kdb','preflight'].includes(change.scope))
      throw new Error('Validation runway guidance must be available before the attempt is spent');
    for(const claim of change.measurements??[]){
      const source=byReference.get(claim.reference);
      if(source?.durationSeconds===undefined||claim.durationSeconds!==source.durationSeconds)
        throw new Error('Duration claims require a matching measured operation duration');
    }
  }
  // These responses cross an ownership or elapsed-budget boundary: render the
  // action from the policy, not arbitrary prose attached to a safe-looking label.
  const responses={
    'report-blocker':'When admission, execution closure, transport or frozen runtime identity is blocked, preserve source and receipts and report the infrastructure fault for harness recovery. Do not inspect or modify harness lease/fence state or replay uncertain work.',
    'reserve-validation-runway':'Check remaining elapsed allowance before expensive work. Keep the last correct source recoverable and reserve time to reach the first canonical validation and acceptance before further optimisation. Tokens and request counts are telemetry, not stopping budgets.',
  };
  return changes.map(change=>responses[change.response]?{...change,text:responses[change.response]}:change);
}

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
