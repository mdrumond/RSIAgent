import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {withSkillLock} from '../../cli/evolution-lock.mjs';
import {State} from '../../controller/state.mjs';

test('stop interrupts a queued writer without releasing the live owner claim',async()=>{
  const root=await mkdtemp(join(tmpdir(),'ascend-writer-lock-'));
  let release,entered;
  const ready=new Promise(resolve=>entered=resolve),hold=new Promise(resolve=>release=resolve);
  const first=withSkillLock({root},async()=>{entered();await hold;});
  try{
    await ready;
    const state=new State(join(root,'state.sqlite'));
    const before=state.db.prepare('SELECT value FROM control').get().value;
    const abort=new AbortController();let called=false;
    const queued=withSkillLock({root},async()=>{called=true;},{signal:abort.signal});
    abort.abort();await assert.rejects(queued);
    assert.equal(called,false);
    assert.equal(state.db.prepare('SELECT value FROM control').get().value,before);
    state.close();
  }finally{release();await first;await rm(root,{recursive:true,force:true});}
});
