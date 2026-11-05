import { mkdir, readFile, writeFile, rename } from 'node:fs/promises';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { performance } from 'node:perf_hooks';

export const positiveSeconds = value => {
  if (!Number.isFinite(value) || value <= 0) throw new Error('Evaluation elapsed allocation must be positive seconds');
  return value;
};
const read = async path => JSON.parse(await readFile(path, 'utf8'));
const save = async (path, value) => {
  const temp = `${path}.${randomUUID()}.tmp`;
  await writeFile(temp, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
  await rename(temp, path);
};

// A running wall checkpoint survives process loss. Normal accounting uses monotonic time;
// after a crash, conservatively charge until the next observer instead of refunding work.
export async function evaluateRound({ pending, config, evaluate, signal,
  now = () => performance.now(), wallNow = Date.now }) {
  const root = pending.evaluationRoot, file = join(root, 'round.json');
  await mkdir(root, { recursive: true });
  let round;
  try { round = await read(file); } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    round = { elapsedSeconds: 0, status: 'deferred', samples: pending.projectIds.flatMap(projectId =>
      Array.from({ length: pending.runs }, (_, run) => (run % 2 ? ['candidate', 'active'] : ['active', 'candidate'])
        .map(arm => ({ projectId, run, arm, instanceRoot: join(root, randomUUID()) }))).flat()) };
  }
  if (round.runningAt !== undefined) {
    round.elapsedSeconds += Math.max(0, wallNow() - round.runningAt) / 1000;
    round.runningAt = wallNow(); // Work may still be live until settlement is proved.
  }
  round.allocatedSeconds = (pending.evaluationSeconds ?? 0) +
    (pending.allocations ?? []).reduce((sum, allocation) => sum + allocation.seconds, 0);
  const remaining = () => Math.max(0, round.allocatedSeconds - round.elapsedSeconds);
  const persist = async () => {
    await save(file, round);
    await save(join(root, 'results.json'), round.samples.filter(sample => sample.result).map(sample =>
      ({ ...sample.result, projectId: sample.projectId, run: sample.run, arm: sample.arm, instanceRoot: sample.instanceRoot })));
  };
  if (!remaining()) {
    round.status = round.allocatedSeconds ? 'exhausted' : 'deferred';
    await persist();
    return { ...round, evaluationRoot: root };
  }
  const controller = new AbortController();
  const abort = () => controller.abort(signal.reason);
  signal?.addEventListener('abort', abort, { once: true });
  let timer, last = now(), expired = false;
  const charge = () => {
    const current = now();
    round.elapsedSeconds += Math.max(0, current - last) / 1000;
    last = current;
  };
  const armTimer = () => {
    timer = setTimeout(() => {
      charge();
      if (remaining()) armTimer();
      else { expired = true; controller.abort(new Error('Evaluation elapsed allocation exhausted')); }
    }, Math.min(remaining() * 1000, 2147483647));
  };
  try {
    signal?.throwIfAborted();
    round.status = 'running';
    armTimer();
    for (const sample of round.samples) {
      if (sample.result) continue;
      charge();
      if (!remaining()) { expired = true; break; }
      controller.signal.throwIfAborted();
      round.runningAt = wallNow();
      await persist(); // Persist sample identity and charged interval before dispatch.
      const isolatedConfig = structuredClone(config);
      isolatedConfig.root = sample.instanceRoot;
      isolatedConfig.skills = { ...isolatedConfig.skills, activeVersion: pending[sample.arm], tuningDisabled: true };
      const result = await evaluate({ ...sample, config: isolatedConfig, sourceInstanceRoot: pending.sourceInstanceRoot,
        projectIds: pending.projectIds, skillVersion: pending[sample.arm], remainingSeconds: remaining(),
        signal: controller.signal });
      sample.result = result;
      charge();
      delete round.runningAt;
      await persist();
    }
    charge();
    signal?.throwIfAborted();
    round.status = expired || !remaining() ? 'exhausted' : 'completed';
  } catch (error) {
    charge();
    round.status = expired || !remaining() ? 'exhausted' : 'interrupted';
    if (error.evaluationStopped === true || round.runningAt === undefined) delete round.runningAt;
    else round.runningAt = wallNow(); // Rebase after charge; retain unknown work across observation gaps.
    await persist();
    if (signal?.aborted || remaining() > 0) throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
  await persist();
  return { ...round, evaluationRoot: root };
}

export async function tuningStatus(root) {
  let state;
  try { state = await read(join(root, 'skills', 'evolution.json')); }
  catch (error) { if (error.code === 'ENOENT') return null; throw error; }
  let round;
  const evaluationRoot = state.pending?.evaluationRoot ??
    (state.evaluationStatus === 'completed' ? state.lastEvaluationRoot : undefined);
  if (evaluationRoot) {
    try { round = await read(join(evaluationRoot, 'round.json')); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  return { candidate: state.candidate, status: round?.status ?? state.evaluationStatus ?? 'deferred',
    evaluationRoot, remainingSeconds: round && Math.max(0, round.allocatedSeconds - round.elapsedSeconds),
    elapsedSeconds: round?.elapsedSeconds, allocatedSeconds: round?.allocatedSeconds,
    completedSamples: round?.samples.filter(sample => sample.result).length };
}
