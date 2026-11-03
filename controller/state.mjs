import { DatabaseSync } from 'node:sqlite';
import { randomUUID } from 'node:crypto';

export class State {
  constructor(file) {
    this.db = new DatabaseSync(file);
    this.db.exec('PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;');
    const version = this.db.prepare('PRAGMA user_version').get().user_version;
    if (version !== 0 && version !== 1) throw new Error('Unsupported instance; bootstrap a fresh instance');
    this.db.exec(`CREATE TABLE IF NOT EXISTS projects (
      id TEXT PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued'
      CHECK(status IN ('queued','running','passed','failed')), frozen INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE IF NOT EXISTS attempts (
      id TEXT PRIMARY KEY, project TEXT NOT NULL REFERENCES projects(id), session TEXT,
      elapsed REAL NOT NULL DEFAULT 0, outcome TEXT, owner TEXT, created TEXT NOT NULL);
      CREATE UNIQUE INDEX IF NOT EXISTS active_attempt ON attempts(project) WHERE outcome IS NULL;
      CREATE TABLE IF NOT EXISTS events (
      sequence INTEGER PRIMARY KEY, attempt TEXT REFERENCES attempts(id), type TEXT NOT NULL,
      data TEXT NOT NULL, created TEXT NOT NULL);
      PRAGMA user_version=1;`);
  }
  add(spec) {
    if (!spec.id || !spec.goal || !(spec.actor_seconds > 0)) throw new Error('Invalid project');
    this.db.prepare('INSERT INTO projects(id,spec) VALUES (?,?)').run(spec.id, JSON.stringify(spec));
  }
  project(id) {
    const row = this.db.prepare('SELECT * FROM projects WHERE id=?').get(id);
    return row && {...row, spec: JSON.parse(row.spec), frozen: Boolean(row.frozen)};
  }
  admit(id, owner) {
    this.db.exec('BEGIN IMMEDIATE');
    try {
      const project = this.project(id);
      if (!project || project.frozen || project.status !== 'queued') throw new Error('Project is not eligible');
      const attempt = randomUUID();
      this.db.prepare('INSERT INTO attempts(id,project,owner,created) VALUES (?,?,?,?)')
        .run(attempt,id,JSON.stringify(owner),new Date().toISOString());
      this.db.prepare("UPDATE projects SET status='running' WHERE id=?").run(id);
      this.db.exec('COMMIT');
      return this.attempt(attempt);
    } catch (error) { this.db.exec('ROLLBACK'); throw error; }
  }
  attempt(id) { return this.db.prepare('SELECT * FROM attempts WHERE id=?').get(id); }
  checkpoint(id, elapsed, session = null) {
    const result = this.db.prepare('UPDATE attempts SET elapsed=?, session=COALESCE(?,session) WHERE id=? AND outcome IS NULL AND elapsed<=?')
      .run(elapsed,session,id,elapsed);
    if (!result.changes) throw new Error('Invalid attempt checkpoint');
  }
  event(attempt,type,data) {
    this.db.prepare('INSERT INTO events(attempt,type,data,created) VALUES (?,?,?,?)')
      .run(attempt,type,JSON.stringify(data),new Date().toISOString());
  }
  events(id) {
    return this.db.prepare('SELECT type,data FROM events WHERE attempt=? ORDER BY sequence').all(id)
      .map(row=>({type:row.type,data:JSON.parse(row.data)}));
  }
  finish(id, outcome) {
    if (!['passed','failed'].includes(outcome)) throw new Error('Invalid outcome');
    this.db.exec('BEGIN IMMEDIATE');
    try {
      const attempt = this.attempt(id);
      if (!attempt || attempt.outcome) throw new Error('Attempt already closed');
      this.db.prepare('UPDATE attempts SET outcome=? WHERE id=?').run(outcome,id);
      this.db.prepare('UPDATE projects SET status=? WHERE id=?').run(outcome,attempt.project);
      this.db.exec('COMMIT');
    } catch(error) { this.db.exec('ROLLBACK'); throw error; }
  }
  async resume(id, confirmStopped, owner) {
    const attempt = this.attempt(id);
    if (!attempt || attempt.outcome) throw new Error('Attempt is not resumable');
    if (!await confirmStopped(attempt)) throw new Error('Previous execution is not confirmed stopped');
    this.db.prepare('UPDATE attempts SET owner=? WHERE id=? AND outcome IS NULL').run(JSON.stringify(owner),id);
    return this.attempt(id);
  }
  close() { this.db.close(); }
}
