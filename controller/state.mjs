import { DatabaseSync } from 'node:sqlite';
import { randomUUID, createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

export function validateAcceptance(spec) {
  if (spec.performance_goal != null &&
      (typeof spec.verifier?.performance_command !== 'string' || !spec.verifier.performance_command.trim()))
    throw new Error('performance_goal requires verifier.performance_command');
}

export class State {
  constructor(file) {
    this.db = new DatabaseSync(file);
    this.db.exec('PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000; PRAGMA journal_mode=WAL;');
    const version = this.db.prepare('PRAGMA user_version').get().user_version;
    if (version === 0 && this.db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").all().length)
      throw new Error('Existing unversioned state is unsupported; bootstrap a fresh instance');
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
      CREATE INDEX IF NOT EXISTS events_attempt_sequence ON events(attempt,sequence);
      CREATE INDEX IF NOT EXISTS events_attempt_type_sequence ON events(attempt,type,sequence);
      PRAGMA user_version=1;`);
  }
  add(spec) {
    validateAcceptance(spec);
    if (!spec.id || !spec.goal || !(spec.actor_seconds > 0)) throw new Error('Invalid project');
    if(spec.verifier)spec={...spec,verifier:{...spec.verifier,hashes:Object.fromEntries(spec.verifier.files.map(path=>
      [resolve(path),createHash('sha256').update(readFileSync(path)).digest('hex')]))}};
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
    return this.db.prepare('SELECT type,data FROM events WHERE attempt IS ? ORDER BY sequence').all(id)
      .map(row=>({type:row.type,data:JSON.parse(row.data)}));
  }
  // Sequence cursors are exclusive. Descending queries return newest first;
  // reverse their result when displaying a chronological diagnostic tail.
  // Legacy events() deliberately remains complete for replay and settlement.
  queryEvents(id, {types, excludeTypes, limit=100, order='asc', after, before}={}) {
    if (!Number.isSafeInteger(limit) || limit<1 || limit>1000)
      throw new Error('Event limit must be an integer between 1 and 1000');
    if (!['asc','desc'].includes(order)) throw new Error('Invalid event order');
    const clauses=['attempt IS ?'],parameters=[id];
    for (const [name,values,operator] of [['types',types,'IN'],['excludeTypes',excludeTypes,'NOT IN']]) {
      if (values===undefined) continue;
      if (!Array.isArray(values) || values.some(value=>typeof value!=='string' || !value))
        throw new Error('Invalid event '+name);
      if (!values.length) {
        if (name==='types') clauses.push('0');
        continue;
      }
      clauses.push(`type ${operator} (${values.map(()=>'?').join(',')})`);
      parameters.push(...values);
    }
    for (const [value,operator] of [[after,'>'],[before,'<']]) {
      if (value===undefined) continue;
      if (!Number.isSafeInteger(value) || value<0) throw new Error('Invalid event cursor');
      clauses.push(`sequence ${operator} ?`);parameters.push(value);
    }
    return this.db.prepare(`SELECT sequence,type,data,created FROM events
      WHERE ${clauses.join(' AND ')} ORDER BY sequence ${order} LIMIT ?`).all(...parameters,limit)
      .map(row=>({...row,data:JSON.parse(row.data)}));
  }
  // Factual observations only: recording a failure never settles an attempt,
  // proves execution stopped, changes elapsed accounting, or promotes memory.
  recordFailure(id, {phase,kind,message,evidence=[],operation}={}) {
    for (const [name,value] of Object.entries({phase,kind,message})) {
      if (typeof value!=='string' || !value.trim()) throw new Error('Invalid failure '+name);
    }
    if (!Array.isArray(evidence) || evidence.some(item=>
      !item || typeof item.path!=='string' || !item.path.trim() ||
      (item.sha256!==undefined && !/^[a-f0-9]{64}$/.test(item.sha256))))
      throw new Error('Invalid failure evidence');
    if (operation!==undefined && (typeof operation!=='string' || !operation.trim()))
      throw new Error('Invalid failure operation');
    if (!this.attempt(id)) throw new Error('Unknown failure attempt');
    const data={phase,kind,message,evidence:evidence.map(({path,sha256})=>
      ({path,...(sha256===undefined?{}:{sha256})})),...(operation===undefined?{}:{operation})};
    const result=this.db.prepare('INSERT INTO events(attempt,type,data,created) VALUES (?,?,?,?)')
      .run(id,'failure-recorded',JSON.stringify(data),new Date().toISOString());
    return Number(result.lastInsertRowid);
  }
  failures(id, options={}) {
    return this.queryEvents(id,{...options,types:['failure-recorded']});
  }
  finish(id, outcome, {settleProject=true}={}) {
    if (!['passed','failed'].includes(outcome)) throw new Error('Invalid outcome');
    this.db.exec('BEGIN IMMEDIATE');
    try {
      const attempt = this.attempt(id);
      if (!attempt || attempt.outcome) throw new Error('Attempt already closed');
      this.db.prepare('UPDATE attempts SET outcome=? WHERE id=?').run(outcome,id);
      if(settleProject)this.db.prepare('UPDATE projects SET status=? WHERE id=?').run(outcome,attempt.project);
      this.db.exec('COMMIT');
    } catch(error) { this.db.exec('ROLLBACK'); throw error; }
  }
  async resume(id, confirmStopped, owner) {
    const attempt = this.attempt(id);
    if (!attempt || attempt.outcome) throw new Error('Attempt is not resumable');
    if (!await confirmStopped(attempt)) throw new Error('Previous execution is not confirmed stopped');
    const claimed=this.db.prepare('UPDATE attempts SET owner=? WHERE id=? AND outcome IS NULL AND owner=?')
      .run(JSON.stringify(owner),id,attempt.owner);
    if(!claimed.changes)throw new Error('Attempt ownership changed during resume');
    return this.attempt(id);
  }
  close() { this.db.close(); }
}
