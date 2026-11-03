import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { State } from '../controller/state.mjs';
import { processIdentity,ownerStopped } from '../controller/lifecycle.mjs';

// A short SQLite claim protects files across processes; no transaction spans model work.
export async function withSkillLock(config,operation) {
  const state=new State(join(config.root,'state.sqlite'));
  state.db.exec('CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY,value TEXT NOT NULL)');
  const me=JSON.stringify(await processIdentity()),key='skill-writer';let owned=false;
  const end=performance.now()+(config.skill_queue_seconds??3600)*1000;
  try {
    while(performance.now()<end) {
      const row=state.db.prepare('SELECT value FROM control WHERE key=?').get(key);
      if(row && !await ownerStopped(JSON.parse(row.value))){await delay(100);continue;}
      state.db.exec('BEGIN IMMEDIATE');
      try {
        const current=state.db.prepare('SELECT value FROM control WHERE key=?').get(key);
        if(current?.value!==row?.value){state.db.exec('ROLLBACK');continue;}
        state.db.prepare('INSERT OR REPLACE INTO control(key,value) VALUES (?,?)').run(key,me);
        state.db.exec('COMMIT');owned=true;break;
      }catch(error){state.db.exec('ROLLBACK');throw error;}
    }
    if(!owned)throw new Error('Skill writer queue elapsed allocation exhausted');
    return await operation();
  }finally {
    if(owned)state.db.prepare('DELETE FROM control WHERE key=? AND value=?').run(key,me);
    state.close();
  }
}
