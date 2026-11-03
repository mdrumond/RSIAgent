import { readFile } from 'node:fs/promises';

export async function processIdentity(pid=process.pid) {
  const stat=await readFile(`/proc/${pid}/stat`,'utf8');
  const fields=stat.slice(stat.lastIndexOf(')')+2).split(' ');
  return {pid,start:fields[19],boot:(await readFile('/proc/sys/kernel/random/boot_id','utf8')).trim()};
}
export async function ownerStopped(owner) {
  const expected=typeof owner==='string'?JSON.parse(owner):owner;
  if (!expected?.pid || !expected.start || !expected.boot) return false;
  try {
    const actual=await processIdentity(expected.pid);
    return actual.start !== expected.start || actual.boot !== expected.boot;
  } catch(error) { if(error.code==='ENOENT') return true;throw error; }
}
export async function attemptStopped(state,attempt,environment) {
  if (!await ownerStopped(attempt.owner)) return false;
  const events=state.events(attempt.id);
  const reconciled=new Set(events.filter(e=>e.type==='execution-reconciled' && e.data.closed===true).map(e=>e.data.operation));
  const unresolved=events.filter(e=>e.type==='execution-unresolved' && !reconciled.has(e.data.operation));
  if(unresolved.some(e=>!e.data.handle))return false;
  const handles=new Set([...events.filter(e=>e.type==='execution-handle').map(e=>e.data.handle),
    ...unresolved.map(e=>e.data.handle)]);
  for (const handle of handles) {
    if (!handle || (await environment.observe(handle)).state !== 'stopped') return false;
  }
  return true;
}
