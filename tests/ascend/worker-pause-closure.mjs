import test from 'node:test';
import assert from 'node:assert/strict';
import {State} from '../../controller/state.mjs';
import {operationsStopped} from '../../controller/lifecycle.mjs';
import {stopWorkerExecutions} from '../../cli/runtime.mjs';

function fixture(t) {
  const state=new State(':memory:');t.after(()=>state.close());
  state.add({id:'p',goal:'preserve checkpoint during pause',actor_seconds:600});
  const attempt=state.admit('p',{pid:2147483647,start:'0',boot:'old'});
  state.checkpoint(attempt.id,17.25,'retained-session.jsonl');
  return {state,id:attempt.id,event:(type,data)=>state.event(attempt.id,type,data)};
}
const remote=(n,kind='job')=>`remote:target:${kind}:command-${n}`;

test('pause skips immutable terminal history, deduplicates live work and reconciles lease once',async t=>{
  const {state,id,event}=fixture(t),before=state.attempt(id);
  for(let n=0;n<65;n++) {
    const handle=remote(n,n%2?'job':'transfer');
    event('execution-handle',{handle});event('execution-handle',{handle});
    event('execution-state',{handle,state:'stopped',nativeState:['completed','failed','cancelled'][n%3]});
  }
  const live=remote('live');event('execution-handle',{handle:live});event('execution-handle',{handle:live});
  const calls=[];
  await stopWorkerExecutions(state,id,{stop:async handle=>calls.push(['stop',handle]),reconcileLease:async()=>{calls.push(['lease']);return true;}});
  assert.deepEqual(calls,[['stop',live],['lease']]);
  assert.deepEqual(state.attempt(id),before);
  assert.equal(state.queryEvents(id,{types:['execution-settled','execution-reconciled']}).length,0);
});

test('unknown, local and unproved terminal handles remain stop requests',async t=>{
  const {state,id,event}=fixture(t),handles=['local:exec-retained',remote('unknown'),remote('incomplete')];
  event('execution-handle',{handle:handles[0]});event('execution-state',{handle:handles[0],state:'stopped',nativeState:'completed'});
  event('execution-unresolved',{handle:handles[1]});
  event('execution-submitted',{handle:handles[2]});event('execution-state',{handle:handles[2],state:'stopped'});
  const seen=[];await stopWorkerExecutions(state,id,{stop:async handle=>seen.push(handle)});
  assert.deepEqual(seen,handles);
});

test('excluded health accounting is retained when stopping an unresolved health command',async t=>{
  const {state,id,event}=fixture(t),handle=remote('health');
  event('execution-submitting',{submission:'health',accountingExcluded:true});
  event('execution-submitted',{submission:'health',handle});event('execution-handle',{handle});
  const calls=[];await stopWorkerExecutions(state,id,{stop:async(...args)=>calls.push(args)});
  assert.deepEqual(calls,[[handle,{accountingExcluded:true}]]);
});

test('a stop error does not prevent other stops or turn unavailable lease cleanup into closure',async t=>{
  const {state,id,event}=fixture(t),first=remote('a'),second=remote('b');
  event('execution-handle',{handle:first});event('execution-handle',{handle:second});
  state.db.exec('CREATE TABLE resource_leases(resource TEXT PRIMARY KEY,attempt TEXT)');
  state.db.prepare('INSERT INTO resource_leases VALUES (?,?)').run('slot',id);
  const seen=[];
  await stopWorkerExecutions(state,id,{stop:async handle=>{seen.push(handle);if(handle===first)throw Error('unavailable');},reconcileLease:async()=>{seen.push('lease');return false;}});
  assert.deepEqual(seen,[first,second,'lease']);
  assert.equal(state.queryEvents(id,{types:['stop-observation-error']}).length,1);
  assert.equal(state.queryEvents(id,{types:['stop-lease-unresolved']}).length,1);
  assert.equal(state.db.prepare('SELECT attempt FROM resource_leases').get().attempt,id);
  assert.equal(await operationsStopped(state,state.attempt(id),{observe:async()=>({state:'unknown'})}),false);
});

test('terminal handles do not bypass a live native Bash fence or lease observation',async t=>{
  const {state,id,event}=fixture(t),handle=remote('closed');let leases=0;
  event('execution-handle',{handle});event('execution-state',{handle,state:'stopped',nativeState:'completed'});
  event('native-bash-started',{operation:'detached'});
  await stopWorkerExecutions(state,id,{stop:()=>assert.fail('terminal handle stopped'),reconcileLease:async()=>{leases++;return false;}});
  assert.equal(leases,1);
  assert.equal(await operationsStopped(state,state.attempt(id),{}),false);
  assert.equal(state.queryEvents(id,{types:['native-bash-completed']}).length,0);
});
