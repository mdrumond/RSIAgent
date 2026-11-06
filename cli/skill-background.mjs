import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { State } from '../controller/state.mjs';
import { tuningStatus, positiveSeconds } from '../controller/evaluation-round.mjs';
import { withSkillLock } from './evolution-lock.mjs';
import { run } from './skills.mjs';
import {enqueueCompletedEvaluation,evaluationArchiveStatus,evaluationArchivesSettled,drainEvaluationArchives} from './evaluation-archives.mjs';

const read = async path => {
  try { return JSON.parse(await readFile(path, 'utf8')); }
  catch (error) { if (error.code !== 'ENOENT') throw error; return null; }
};
function queue(state, create = false) {
  if (create) state.db.exec(`CREATE TABLE IF NOT EXISTS skill_evidence_queue (
    id TEXT PRIMARY KEY, task TEXT NOT NULL, consumed INTEGER NOT NULL DEFAULT 0)`);
  return Boolean(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='skill_evidence_queue'").get());
}

// No model/file lock is taken here: foreground postprocessing only commits evidence.
export function enqueueSkillEvidence(config, task) {
  if (config.skills?.tuningDisabled || task?.phase !== 'phase2' || task?.kind !== 'original') return false;
  if (task.completed !== true || !task.id || !task.projectId) throw new Error('Require completed skill evidence');
  const state = new State(join(config.root, 'state.sqlite'));
  try {
    queue(state, true);
    return state.db.prepare('INSERT OR IGNORE INTO skill_evidence_queue(id,task) VALUES (?,?)')
      .run(task.id, JSON.stringify(task)).changes > 0;
  } finally { state.close(); }
}

export async function backgroundSkillStatus(config) {
  const evolution = await read(join(config.root, 'skills', 'evolution.json'));
  const archives = await evaluationArchiveStatus(config,evolution);
  const pending = evolution?.pending;
  const round = pending && await read(join(pending.evaluationRoot, 'round.json'));
  const state = new State(join(config.root, 'state.sqlite'));
  try {
    const queued = queue(state) ? state.db.prepare('SELECT count(*) AS n FROM skill_evidence_queue WHERE consumed=0').get().n : 0;
    const control = state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get();
    const owner = control && state.db.prepare("SELECT value FROM control WHERE key='skill-writer'").get();
    return { ...await tuningStatus(config.root), queued, pending: Boolean(pending),
      archives, hasWork: !config.skills?.tuningDisabled && ((pending ? round?.status !== 'exhausted' : queued > 0)||archives.hasWork),
      unresolved: round?.runningAt !== undefined, resource: pending?.resource,
      owner: owner ? JSON.parse(owner.value) : null };
  } finally { state.close(); }
}

// Used by scheduler settlement before returning a reserved slot to foreground work.
export async function backgroundSkillsSettled(config, { runtime } = {}) {
  try {
    if(!await evaluationArchivesSettled(config))return false;
    const evolution = await read(join(config.root, 'skills', 'evolution.json'));
    const pending = evolution?.pending;
    if (!pending) return true;
    const round = await read(join(pending.evaluationRoot, 'round.json'));
    if (round?.runningAt === undefined) return true;
    const sample = round.samples.find(sample => !sample.result);
    if (!sample) return false;
    runtime ??= await import(pending.config.skills?.runtimeModule
      ? pathToFileURL(resolve(config.root, pending.config.skills.runtimeModule)).href
      : new URL('./evaluation.mjs', import.meta.url).href);
    return typeof runtime.settled === 'function' && await runtime.settled({ ...sample,
      config: pending.config, sourceInstanceRoot: config.root, resource: pending.resource });
  } catch { return false; }
}

// The campaign owns scheduling, device capacity and the decreasing remaining allocation.
// One invocation handles one comparison, never recursively drains newer evidence.
export async function runBackgroundSkills(config, values = {}) {
  positiveSeconds(values.remainingSeconds);
  if (!values.resource?.id) throw new Error('Background evaluation requires a reserved resource');
  const deadline = new AbortController(), end = performance.now() + values.remainingSeconds * 1000;
  let timer;
  const arm = () => {
    const remaining = end - performance.now();
    if (remaining <= 0) deadline.abort(new Error('Campaign elapsed allocation exhausted'));
    else {
      timer = setTimeout(arm, Math.min(remaining, 2147483647));
      timer.unref();
    }
  };
  arm();
  const signal = values.signal ? AbortSignal.any([values.signal, deadline.signal]) : deadline.signal;
  try { return await withSkillLock(config, async () => {
    const evolution = await read(join(config.root, 'skills', 'evolution.json'));
    await enqueueCompletedEvaluation(config,evolution);
    await drainEvaluationArchives(config,{signal});
    const pending = evolution?.pending;
    if (pending?.resource && values.resource?.id !== pending.resource.id)
      throw new Error('Resume skill evaluation on its retained resource');
    if (pending && !await backgroundSkillsSettled(config, { runtime: values.runtime }))
      throw new Error('Background evaluation execution remains unresolved');
    const state = new State(join(config.root, 'state.sqlite'));
    let tasks;
    try {
      tasks = !pending && queue(state) ? state.db.prepare('SELECT task FROM skill_evidence_queue WHERE consumed=0 ORDER BY rowid')
        .all().map(row => JSON.parse(row.task)) : [];
    } finally { state.close(); }
    if (!pending && !tasks.length) return { evaluationStatus: 'idle' };
    try {
      return await run(config, { ...values, signal, background: true,
        backgroundSeconds: values.remainingSeconds, backgroundResource: values.resource,
        task: tasks.at(-1), evidenceTasks: tasks,
        lock: async (_config, operation) => operation() });
    } finally {
      // A crash before candidate/history persistence leaves evidence eligible for retry.
      const retained = await read(join(config.root, 'skills', 'evolution.json'));
      await enqueueCompletedEvaluation(config,retained);
      const state = new State(join(config.root, 'state.sqlite'));
      try {
        if (queue(state)) {
          const consume = state.db.prepare('UPDATE skill_evidence_queue SET consumed=1 WHERE id=?');
          for (const task of retained?.history ?? []) consume.run(task.id);
        }
      } finally { state.close(); }
      await drainEvaluationArchives(config,{signal});
    }
  }, { signal }); } finally { clearTimeout(timer); }
}
