import { readFile } from 'node:fs/promises';
import { excludedExecutionProofs } from './execution-proof.mjs';

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
  if(nativeElapsed>attempt.elapsed)checkpointExecution(state,attempt.id,nativeElapsed);
  const pending=new Map();
  for(const {type,data} of events) {
    const operation=data.operation??data.submission;
    // Closure covers only preceding work, not a later submission or unknown
    // marker that happens to carry the same enclosing operation identity.
    if(type==='execution-settled' || (type==='execution-reconciled' && data.closed===true)) {
      if(operation!=null)pending.delete(operation);
      continue;
    }
    if(!['execution-starting','execution-unresolved','execution-submitting'].includes(type))continue;
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
      checkpointExecution(state,attempt.id,elapsed);
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
  const closedThrough=new Map();
  for(const [index,{type,data}] of events.entries()) {
    const operation=data.operation??data.submission;
    if(operation!=null && (type==='execution-settled' ||
        (type==='execution-reconciled' && data.closed===true)))closedThrough.set(operation,index);
  }
  const unresolved=events.filter((e,index)=>e.type==='execution-unresolved' &&
    index>(closedThrough.get(e.data.operation??e.data.submission)??-1));
  if(unresolved.some(e=>!e.data.handle))return false;
  const handles=new Set([...events.filter(e=>e.type==='execution-handle').map(e=>e.data.handle),
    ...events.filter(e=>e.type==='execution-unresolved' && e.data.handle).map(e=>e.data.handle)]);
  const excluded=excludedExecutionProofs(events);
  for (const handle of handles) {
    if (!handle) return false;
    const proof=excluded.get(handle) ?? await environment.observe(handle,{accountingExcluded:excluded.has(handle)});
    if (proof.state !== 'stopped') return false;
  }
  return true;
}

// Outcome is factual; closure is a separate, conservative execution proof.
export function failureSettled(state,id) {
  const event=state.queryEvents(id,{types:['failure-settlement'],order:'desc',limit:1})[0];
  return !event || event.data.closed===true;
}
export async function settleFailure(state,id,environment,{phase='worker',kind='exception',
  message='Worker failed',settleProject=true}={}) {
  state.db.exec('BEGIN IMMEDIATE');
  try {
    state.recordFailure(id,{phase,kind,message});
    if(!state.attempt(id).outcome) {
      state.db.prepare("UPDATE attempts SET outcome='failed' WHERE id=?").run(id);
      if(settleProject)state.db.prepare("UPDATE projects SET status='failed' WHERE id=?")
        .run(state.attempt(id).project);
    }
    // Outcome and fence become visible together, including across worker death.
    state.event(id,'failure-settlement',{closed:false});
    state.db.exec('COMMIT');
  }catch(error){state.db.exec('ROLLBACK');throw error;}
  return reconcileFailure(state,id,environment);
}
export async function reconcileFailure(state,id,environment) {
  let closed=false;
  try {closed=await operationsStopped(state,state.attempt(id),environment??{});}
  catch(error){state.event(id,'settlement-error',{message:error.message});}
  state.event(id,'failure-settlement',{closed});
  return closed;
}

function checkpointExecution(state,id,elapsed) {
  // Factual failure may precede the final detached-operation duration.
  const result=state.db.prepare('UPDATE attempts SET elapsed=? WHERE id=? AND elapsed<=?')
    .run(elapsed,id,elapsed);
  if(!result.changes)throw new Error('Invalid execution checkpoint');
}
