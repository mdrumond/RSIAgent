import { randomUUID } from 'node:crypto';
import { stages } from './workflow.mjs';

export function modePolicy(mode, { inferenceRecovery = true } = {}) {
  if (!['phase1', 'phase2', 'inference'].includes(mode)) throw new Error('Unknown mode');
  return Object.freeze({ mode, stages, memory: true, skills: mode !== 'phase1',
    recovery: mode !== 'inference' || inferenceRecovery === true, tuneEvery: 8 });
}

// Owns scheduling and recovery admission; the caller executes the returned attempt
// through the canonical worker. No project spec or source files are rewritten.
export class RecoveryQueue {
  constructor(state, { mode = 'phase1', inferenceRecovery = true,
    maxFailedRounds = 5, investigationSeconds = 9600 } = {}) {
    if (!Number.isSafeInteger(maxFailedRounds) || maxFailedRounds < 1)
      throw new Error('Invalid unsuccessful round limit');
    if (!Number.isFinite(investigationSeconds) || investigationSeconds <= 0)
      throw new Error('Invalid investigation budget');
    this.state = state;
    this.db = state.db;
    this.policy = modePolicy(mode, { inferenceRecovery });
    this.limit = maxFailedRounds;
    this.investigationSeconds = investigationSeconds;
    this.db.exec(`CREATE TABLE IF NOT EXISTS recovery_queue (
      project TEXT PRIMARY KEY REFERENCES projects(id), origin TEXT NOT NULL REFERENCES attempts(id),
      failed_rounds INTEGER NOT NULL DEFAULT 0, kind TEXT NOT NULL,
      eligible_wave INTEGER NOT NULL, attempt TEXT REFERENCES attempts(id));
      CREATE TABLE IF NOT EXISTS recovery_attempts (
      attempt TEXT PRIMARY KEY REFERENCES attempts(id), project TEXT NOT NULL REFERENCES projects(id),
      kind TEXT NOT NULL, wave INTEGER NOT NULL, completed INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE IF NOT EXISTS recovery_lessons (
      id INTEGER PRIMARY KEY, attempt TEXT NOT NULL UNIQUE REFERENCES attempts(id),
      data TEXT NOT NULL, recorded INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE IF NOT EXISTS resource_leases (resource TEXT PRIMARY KEY NOT NULL,
      attempt TEXT NOT NULL UNIQUE REFERENCES attempts(id));`);
  }
  transaction(fn) {
    this.db.exec('BEGIN IMMEDIATE');
    try { const result = fn(); this.db.exec('COMMIT'); return result; }
    catch (error) { this.db.exec('ROLLBACK'); throw error; }
  }
  wave(value) {
    if (!Number.isSafeInteger(value) || value < 0) throw new Error('Invalid wave');
    return value;
  }
  entry(project) {
    return this.db.prepare('SELECT * FROM recovery_queue WHERE project=?').get(project);
  }
  enqueue(attemptId, wave) {
    this.wave(wave);
    return this.transaction(() => {
      const attempt = this.db.prepare('SELECT * FROM attempts WHERE id=?').get(attemptId);
      if (!attempt || attempt.outcome !== 'failed') throw new Error('Original attempt must have failed');
      if (this.db.prepare('SELECT 1 FROM recovery_attempts WHERE attempt=?').get(attemptId))
        throw new Error('Recovery attempts complete through the recovery queue');
      const project = this.state.project(attempt.project);
      if (project.frozen || project.status !== 'failed') throw new Error('Project is not recoverable');
      const existing = this.entry(project.id);
      if (existing) return existing;
      if (!this.policy.recovery) return null;
      this.db.prepare(`INSERT INTO recovery_queue(project,origin,kind,eligible_wave)
        VALUES (?,?,'investigation',?)`).run(project.id, attemptId, wave + 1);
      this.state.event(attemptId, 'recovery-scheduled', { kind: 'investigation', wave: wave + 1 });
      return this.entry(project.id);
    });
  }
  ready(wave) {
    this.wave(wave);
    if (!this.policy.recovery) return [];
    return this.db.prepare(`SELECT q.* FROM recovery_queue q JOIN projects p ON p.id=q.project
      WHERE q.attempt IS NULL AND q.eligible_wave<=? AND p.frozen=0 AND p.status='failed'
      ORDER BY q.eligible_wave,q.project`).all(wave);
  }
  claim(projectId, wave, owner = {}, resourceId) {
    this.wave(wave);
    return this.transaction(() => {
      const queue = this.entry(projectId), project = this.state.project(projectId);
      const original = project?.status === 'queued' && !queue;
      if (!project || project.frozen || (!original && (!this.policy.recovery || !queue ||
          queue.attempt || queue.eligible_wave > wave || project.status !== 'failed')))
        throw new Error('Job is not eligible');
      const kind = original ? 'original' : queue.kind;
      const id = randomUUID();
      this.db.prepare('INSERT INTO attempts(id,project,owner,created) VALUES (?,?,?,?)')
        .run(id, projectId, JSON.stringify(owner), new Date().toISOString());
      if (resourceId !== undefined)
        this.db.prepare('INSERT INTO resource_leases(resource,attempt) VALUES (?,?)').run(resourceId, id);
      if (!original) {
        this.db.prepare('INSERT INTO recovery_attempts(attempt,project,kind,wave) VALUES (?,?,?,?)')
          .run(id, projectId, kind, wave);
        this.db.prepare('UPDATE recovery_queue SET attempt=? WHERE project=?').run(id, projectId);
      }
      this.db.prepare("UPDATE projects SET status='running' WHERE id=?").run(projectId);
      const actorSeconds = kind === 'investigation' ? this.investigationSeconds : project.spec.actor_seconds;
      this.state.event(id, 'job-admitted', { kind, wave, actor_seconds: actorSeconds });
      return { attempt: this.db.prepare('SELECT * FROM attempts WHERE id=?').get(id),
        project: this.state.project(projectId), kind, actor_seconds: actorSeconds,
        origin_attempt: queue?.origin ?? null, failed_rounds: queue?.failed_rounds ?? 0, policy: this.policy };
    });
  }
  jobs(wave) {
    this.wave(wave);
    const originals = this.db.prepare(`SELECT id FROM projects WHERE status='queued' AND frozen=0
      AND NOT EXISTS (SELECT 1 FROM recovery_queue q WHERE q.project=projects.id) ORDER BY id`).all();
    return [
      ...originals.map(row => ({ projectId: row.id, kind: 'original', project: this.state.project(row.id),
        actor_seconds: this.state.project(row.id).spec.actor_seconds,
        origin_attempt: null, failed_rounds: 0, eligible_wave: wave })),
      ...this.ready(wave).map(row => ({ projectId: row.project, kind: row.kind,
        project: this.state.project(row.project), origin_attempt: row.origin,
        actor_seconds: row.kind === 'investigation' ? this.investigationSeconds :
          this.state.project(row.project).spec.actor_seconds,
        failed_rounds: row.failed_rounds, eligible_wave: row.eligible_wave })),
    ];
  }
  // May close an active attempt or consume the same outcome already closed by State.finish.
  // Findings are an outbox: caller records each lesson in KDB, then acknowledges it.
  complete(attemptId, { passed, useful = false, findings, evidence = [] }, wave) {
    this.wave(wave);
    this.validateResult({ passed, findings, evidence });
    return this.transaction(() => {
      const recovery = this.db.prepare('SELECT * FROM recovery_attempts WHERE attempt=?').get(attemptId);
      if (!recovery || recovery.completed) throw new Error('Recovery attempt is not open');
      if (wave < recovery.wave) throw new Error('Completion precedes admission wave');
      const queue = this.entry(recovery.project);
      if (!queue || queue.attempt !== attemptId) throw new Error('Recovery ownership changed');
      const attempt = this.db.prepare('SELECT * FROM attempts WHERE id=?').get(attemptId);
      const outcome = passed ? 'passed' : 'failed';
      if (attempt.outcome && attempt.outcome !== outcome) throw new Error('Conflicting attempt outcome');
      this.db.prepare('UPDATE attempts SET outcome=? WHERE id=?').run(outcome, attemptId);
      const lesson = { attempt_id: attemptId, project: recovery.project, kind: recovery.kind,
        outcome, useful: useful === true, findings, evidence };
      this.db.prepare('INSERT INTO recovery_lessons(attempt,data) VALUES (?,?)')
        .run(attemptId, JSON.stringify(lesson));
      this.db.prepare('UPDATE recovery_attempts SET completed=1 WHERE attempt=?').run(attemptId);
      const accepted = recovery.kind === 'retry' && passed;
      const retry = recovery.kind === 'investigation' && useful === true;
      const failures = queue.failed_rounds + (accepted || retry ? 0 : 1);
      const frozen = !accepted && failures >= this.limit;
      this.db.prepare('UPDATE projects SET status=?,frozen=? WHERE id=?')
        .run(accepted ? 'passed' : 'failed', frozen ? 1 : 0, recovery.project);
      if (accepted || frozen) {
        this.db.prepare('DELETE FROM recovery_queue WHERE project=?').run(recovery.project);
      } else {
        this.db.prepare(`UPDATE recovery_queue SET failed_rounds=?,kind=?,eligible_wave=?,attempt=NULL
          WHERE project=?`).run(failures, retry ? 'retry' : 'investigation', wave + 1, recovery.project);
      }
      const result = { accepted, frozen, failed_rounds: failures,
        next: accepted || frozen ? null : this.entry(recovery.project) };
      this.state.event(attemptId, 'recovery-completed', { ...result, lesson });
      return result;
    });
  }
  validateResult({ passed, findings, evidence }) {
    if (typeof passed !== 'boolean' || typeof findings !== 'string' || !findings.trim())
      throw new Error('Outcome and findings are required');
    if (!Array.isArray(evidence) || evidence.length > 16 || evidence.some(ref =>
      !ref || typeof ref !== 'object' || Array.isArray(ref) ||
      Object.keys(ref).some(key => !['path', 'sha256', 'entry_id'].includes(key)) ||
      !Object.values(ref).every(value => typeof value === 'string' && value.length <= 4096) ||
      !(ref.entry_id || (ref.path && ref.sha256)))) throw new Error('Use bounded direct evidence references');
    if (findings.length > 32000) throw new Error('Findings must be compact');
  }
  recordOriginal(attemptId, { passed, useful = false, findings, evidence = [] }, wave) {
    this.wave(wave);
    this.validateResult({ passed, findings, evidence });
    return this.transaction(() => {
      const attempt = this.db.prepare('SELECT * FROM attempts WHERE id=?').get(attemptId);
      if (!attempt || this.db.prepare('SELECT 1 FROM recovery_attempts WHERE attempt=?').get(attemptId))
        throw new Error('Not an original attempt');
      const latest = this.db.prepare('SELECT id FROM attempts WHERE project=? ORDER BY rowid DESC LIMIT 1').get(attempt.project);
      if (latest.id !== attemptId) throw new Error('Original attempt has been superseded');
      if (this.db.prepare('SELECT 1 FROM recovery_lessons WHERE attempt=?').get(attemptId))
        throw new Error('Original outcome already recorded');
      const admission = this.db.prepare("SELECT data FROM events WHERE attempt=? AND type='job-admitted'").get(attemptId);
      if (admission && wave < JSON.parse(admission.data).wave) throw new Error('Completion precedes admission wave');
      const project = this.state.project(attempt.project);
      if (project.frozen || this.entry(attempt.project)) throw new Error('Original is already in recovery');
      const outcome = passed ? 'passed' : 'failed';
      if (attempt.outcome && attempt.outcome !== outcome) throw new Error('Conflicting attempt outcome');
      this.db.prepare('UPDATE attempts SET outcome=? WHERE id=?').run(outcome, attemptId);
      this.db.prepare('UPDATE projects SET status=? WHERE id=?').run(outcome, attempt.project);
      const lesson = { attempt_id: attemptId, project: attempt.project, kind: 'original',
        outcome, useful: useful === true, findings, evidence };
      this.db.prepare('INSERT INTO recovery_lessons(attempt,data) VALUES (?,?)').run(attemptId, JSON.stringify(lesson));
      if (!passed && this.policy.recovery) {
        this.db.prepare(`INSERT INTO recovery_queue(project,origin,kind,eligible_wave)
          VALUES (?,?,'investigation',?)`).run(attempt.project, attemptId, wave + 1);
      }
      const result = { accepted: passed, frozen: false, failed_rounds: 0,
        next: this.entry(attempt.project) ?? null };
      this.state.event(attemptId, 'original-completed', { ...result, lesson });
      return result;
    });
  }
  // Settled actors awaiting postprocessing must never enter the actor runner again.
  pendingOutcomes() {
    return this.db.prepare(`SELECT a.*, COALESCE(r.kind,'original') AS kind, r.wave
      FROM attempts a LEFT JOIN recovery_attempts r ON r.attempt=a.id
      LEFT JOIN recovery_lessons l ON l.attempt=a.id
      WHERE a.outcome IS NOT NULL AND l.id IS NULL ORDER BY a.created,a.rowid`).all()
      .map(attempt => {
        const admission = this.db.prepare("SELECT data FROM events WHERE attempt=? AND type='job-admitted'").get(attempt.id);
        return { ...attempt, wave: attempt.wave ?? (admission ? JSON.parse(admission.data).wave : null) };
      });
  }
  outcomeRecorded(attemptId) {
    return Boolean(this.db.prepare('SELECT 1 FROM recovery_lessons WHERE attempt=?').get(attemptId));
  }
  lessons() {
    return this.db.prepare('SELECT id,data FROM recovery_lessons WHERE recorded=0 ORDER BY id').all()
      .map(row => ({ id: row.id, ...JSON.parse(row.data) }));
  }
  acknowledgeLesson(id) {
    const result = this.db.prepare('UPDATE recovery_lessons SET recorded=1 WHERE id=? AND recorded=0').run(id);
    return result.changes === 1;
  }
  completedOriginals() {
    return this.db.prepare(`SELECT COUNT(DISTINCT a.project) AS count FROM attempts a
      LEFT JOIN recovery_attempts r ON r.attempt=a.id WHERE a.outcome IS NOT NULL AND r.attempt IS NULL`).get().count;
  }
}

// One selection/admission/outcome boundary for the CLI; OS lifecycle belongs to CLI/core.
export function nextWave(state, { wave, ...options }) {
  return new RecoveryQueue(state, options).jobs(wave);
}
export function admitJob(state, { projectId, wave, owner = {}, resourceId, ...options }) {
  return new RecoveryQueue(state, options).claim(projectId, wave, owner, resourceId);
}
export function recordOutcome(state, { attemptId, wave, passed, useful = false,
  findings, evidence = [], ...options }) {
  const queue = new RecoveryQueue(state, options);
  const result = { passed, useful, findings, evidence };
  const recovery = state.db.prepare('SELECT 1 FROM recovery_attempts WHERE attempt=?').get(attemptId);
  return recovery ? queue.complete(attemptId, result, wave) : queue.recordOriginal(attemptId, result, wave);
}

export function pendingOutcomes(state, options = {}) {
  return new RecoveryQueue(state, options).pendingOutcomes();
}
export function outcomeRecorded(state, attemptId, options = {}) {
  return new RecoveryQueue(state, options).outcomeRecorded(attemptId);
}
