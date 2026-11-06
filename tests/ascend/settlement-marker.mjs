import test from 'node:test';
import assert from 'node:assert/strict';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {operationsStopped,reconcileFailure,failureSettled} from '../../controller/lifecycle.mjs';

function fixture(t) {
  const state=new State(':memory:');t.after(()=>state.close());
  state.add({id:'p',goal:'numeric',actor_seconds:600});
  const queue=new RecoveryQueue(state),{attempt}=queue.claim('p',1,{},'cpu');
  const event=(type,data)=>state.event(attempt.id,type,data);
  // Reduced observed lease-control trace: unresolved download at sequence 9323,
  // enclosing operation settled at 9331. The failed outcome is independent.
  event('execution-starting',{operation:'op',startElapsed:380});
  event('execution-submitting',{operation:'op',submission:'download',kind:'download',accountingExcluded:true});
  event('execution-unresolved',{operation:'op',submission:'download',handle:null,
    purpose:'device-lease-control',accountingExcluded:true,actor_elapsed:382.7413525519994});
  state.checkpoint(attempt.id,382.9718549739994,'actor.jsonl');
  event('execution-settled',{operation:'op',handle:'payload',actor_elapsed:382.9718549739994});
  state.finish(attempt.id,'failed');
  return {state,queue,id:attempt.id,event};
}

test('settled enclosing operation clears an earlier null marker without changing elapsed or failure',async t=>{
  const {state,queue,id}=fixture(t),before=state.attempt(id);
  const env={reconcileOperation:()=>assert.fail('settled work must not be replayed')};
  assert.equal(await operationsStopped(state,before,env),true);
  assert.equal(await reconcileFailure(state,id,env),true);
  assert.equal(failureSettled(state,id),true);
  assert.deepEqual(state.attempt(id),before);
  assert.equal(before.outcome,'failed');
  assert.equal(before.elapsed,382.9718549739994);
  assert.equal(before.session,'actor.jsonl');
  assert.equal(queue.pendingOutcomes().length,1);
  const result={passed:false,findings:'Failure retained',evidence:[{path:'failure.json',sha256:'hash'}]};
  queue.recordOriginal(id,result,1);
  assert.equal(await reconcileFailure(state,id,env),true);
  assert.equal(queue.pendingOutcomes().length,0);
  assert.throws(()=>queue.recordOriginal(id,result,1),/already recorded/);
  assert.equal(state.queryEvents(id,{types:['original-completed']}).length,1);
  assert.equal(queue.lessons().length,1);
});

for(const type of ['execution-starting','execution-submitting','execution-unresolved']) {
  for(const operation of ['op','new-op'])test(`later ${type} for ${operation} remains fenced`,async t=>{
    const {state,id,event}=fixture(t);
    event(type,{operation,submission:'new-submission',handle:null,startElapsed:383});
    let calls=0;
    const env={reconcileOperation:async request=>{
      calls++;assert.equal(request.operation,operation);return {closed:false};
    }};
    assert.equal(await operationsStopped(state,state.attempt(id),env),false);
    assert.equal(await reconcileFailure(state,id,env),false);
    assert.equal(calls,2);
    assert.equal(failureSettled(state,id),false);
    assert.equal(state.attempt(id).elapsed,382.9718549739994);
  });
}

test('later work can reconcile and adds elapsed exactly once',async t=>{
  const {state,id,event}=fixture(t);
  event('execution-starting',{operation:'op',startElapsed:383});
  event('execution-submitting',{operation:'op',submission:'new'});
  event('execution-unresolved',{operation:'op',handle:null});
  let calls=0;
  const env={reconcileOperation:async()=>{
    calls++;return {closed:true,executionSeconds:2.25,handles:['new-handle']};
  },observe:async handle=>{assert.equal(handle,'new-handle');return {state:'stopped'};}};
  assert.equal(await reconcileFailure(state,id,env),true);
  assert.equal(await reconcileFailure(state,id,env),true);
  assert.equal(calls,1);
  assert.equal(state.attempt(id).elapsed,385.25);
  assert.equal(state.attempt(id).outcome,'failed');
});

for(const type of ['execution-handle','execution-unresolved'])test(`settlement never hides a live ${type} handle`,async t=>{
  const {state,id,event}=fixture(t);
  event(type,{operation:'op',handle:'live'});
  event('execution-settled',{operation:'op'});
  const observed=[];
  const env={observe:async handle=>{observed.push(handle);return {state:'running'};}};
  assert.equal(await operationsStopped(state,state.attempt(id),env),false);
  assert.deepEqual(observed,['live']);
});

test('settlement does not close detached native Bash',async t=>{
  const {state,id,event}=fixture(t);
  event('native-bash-started',{operation:'bash'});
  assert.equal(await reconcileFailure(state,id,{}),false);
  event('native-bash-completed',{operation:'bash',actor_elapsed:390.125});
  assert.equal(await reconcileFailure(state,id,{}),true);
  assert.equal(state.attempt(id).elapsed,390.125);
  assert.equal(state.attempt(id).outcome,'failed');
});

test('an unidentified unknown cannot borrow another operation closure',async t=>{
  const {state,id,event}=fixture(t);
  event('execution-unresolved',{handle:null});
  event('execution-settled',{});
  assert.equal(await reconcileFailure(state,id,{}),false);
});

for(const closed of [false,true])test(`reconciliation closed=${closed} respects later unknown submissions`,async t=>{
  const {state,id,event}=fixture(t);
  event('execution-unresolved',{operation:'other',submission:'first',handle:null,startElapsed:383});
  event('execution-reconciled',{operation:'other',closed,executionSeconds:0});
  assert.equal(await operationsStopped(state,state.attempt(id),{}),closed);
  event('execution-submitting',{operation:'other',submission:'second',actor_elapsed:384});
  event('execution-unresolved',{operation:'other',submission:'second',handle:null});
  assert.equal(await operationsStopped(state,state.attempt(id),{}),false);
});

test('an earlier unknown operation is not covered by another operation settlement',async t=>{
  const {state,id,event}=fixture(t);
  event('execution-unresolved',{operation:'unknown',handle:null,startElapsed:383});
  event('execution-settled',{operation:'op'});
  assert.equal(await reconcileFailure(state,id,{}),false);
  assert.equal(failureSettled(state,id),false);
});

test('submission-only identities support ordered closure',async t=>{
  const {state,id,event}=fixture(t);
  event('execution-submitting',{submission:'transfer',accountingExcluded:true});
  event('execution-unresolved',{submission:'transfer',handle:null});
  event('execution-reconciled',{operation:'transfer',closed:true,executionSeconds:0});
  assert.equal(await operationsStopped(state,state.attempt(id),{}),true);
  event('execution-unresolved',{submission:'transfer',handle:null});
  assert.equal(await operationsStopped(state,state.attempt(id),{}),false);
});
