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
  } catch(error) { if(error.code==='ENOENT'||error.code==='ESRCH') return true;throw error; }
}
export async function attemptStopped(state,attempt,environment) {
  if (!await ownerStopped(attempt.owner)) return false;
  return operationsStopped(state,attempt,environment);
}
export async function operationsStopped(state,attempt,environment) {
  let events=state.events(attempt.id);
  const completedBash=events.filter(e=>e.type==='native-bash-completed');
  const closedBash=new Set(completedBash.map(e=>e.data.operation));
  // Worker death cannot prove its detached native SDK shell has stopped.
  if(events.some(e=>e.type==='native-bash-started' && !closedBash.has(e.data.operation)))return false;
  const nativeElapsed=Math.max(attempt.elapsed,...completedBash.map(e=>e.data.actor_elapsed).filter(Number.isFinite));
  if(nativeElapsed>attempt.elapsed)state.checkpoint(attempt.id,nativeElapsed);
  const finished=new Set(events.filter(e=>['execution-settled','execution-reconciled'].includes(e.type) &&
    (e.type==='execution-settled'||e.data.closed===true)).map(e=>e.data.operation));
  const pending=new Map();
  for(const {type,data} of events) {
    if(!['execution-starting','execution-unresolved','execution-submitting'].includes(type))continue;
    const operation=data.operation??data.submission;
    if(finished.has(operation))continue;
    const previous=pending.get(operation);
    pending.set(operation,{...previous,...data,startElapsed:previous?.startElapsed??data.startElapsed??data.actor_elapsed});
  }
  let elapsed=nativeElapsed;
  for(const [operation,data] of pending) {
    if(!environment.reconcileOperation)return false;
    const proof=await environment.reconcileOperation({operation,events,startElapsed:data.startElapsed});
    if(!proof.closed || !Number.isFinite(proof.executionSeconds))return false;
    if(Number.isFinite(data.startElapsed)) {
      elapsed=Math.max(elapsed,data.startElapsed+proof.executionSeconds);
      state.checkpoint(attempt.id,elapsed);
    }else {
      const submissions=events.filter(e=>e.type==='execution-submitting' &&
        (e.data.operation??e.data.submission)===operation);
      const actorStarted=events.some(e=>e.type==='execution-starting' &&
        (e.data.operation??e.data.submission)===operation);
      if(actorStarted || proof.executionSeconds!==0 || !submissions.length ||
          submissions.some(e=>e.data.accountingExcluded!==true))return false;
    }
    state.event(attempt.id,'execution-reconciled',{operation,closed:true,executionSeconds:proof.executionSeconds});
    for(const handle of proof.handles??[])state.event(attempt.id,'execution-handle',{operation,handle});
  }
  events=state.events(attempt.id);
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
