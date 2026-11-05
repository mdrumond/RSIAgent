import test from 'node:test';
import assert from 'node:assert/strict';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {runtimeStatus} from '../../controller/runtime-status.mjs';
function fixture(t){const state=new State(':memory:');t.after(()=>state.close());return state;}
function admit(state,id){state.add({id,goal:'numeric',actor_seconds:30});return state.admit(id,{});}

test('fresh status is available before a campaign initializes resource and recovery tables',t=>{
  const state=fixture(t);state.add({id:'queued',goal:'check',actor_seconds:30});
  const status=runtimeStatus(state);
  assert.equal(status.queues.fresh,1);assert.equal(status.queues.recovery,0);
  assert.deepEqual(status.leases,[]);assert.deepEqual(status.hostSlots,[]);
});
test('device projection follows token ownership, release, reuse and uncertain failures',t=>{
  const state=fixture(t),a=admit(state,'a'),b=admit(state,'b');
  state.event(a.id,'device-lease-acquired',{pool:'pool',physicalDevice:2,receipt:'lease.json',token:'first'});
  assert.equal(runtimeStatus(state).leases[0].physicalDevice,2);
  state.event(a.id,'device-lease-released',{token:'different'});
  assert.equal(runtimeStatus(state).leases.length,1);
  state.event(a.id,'device-lease-released',{token:'first'});
  assert.equal(runtimeStatus(state).leases.length,0);
  state.event(b.id,'device-lease-acquired',{pool:'pool',physicalDevice:2,token:'second'});
  state.finish(b.id,'failed');
  assert.equal(runtimeStatus(state).leases[0].state,'unsettled');
});
test('queue waits clear on admission and terminal actors do not remain waiting',t=>{
  const state=fixture(t),a=admit(state,'a');
  state.event(a.id,'device-wait',{pool:'pool',waiting:true,reason:'idle-devices'});
  assert.equal(runtimeStatus(state).actorWaits[0].project,'a');
  state.event(a.id,'device-wait',{waiting:false});assert.equal(runtimeStatus(state).actorWaits.length,0);
  state.event(a.id,'device-wait',{waiting:true});state.finish(a.id,'failed');
  assert.equal(runtimeStatus(state).actorWaits.length,0);
});
test('pending memory and postprocessing remain visible while status ignores event bodies',t=>{
  const state=fixture(t),queue=new RecoveryQueue(state),a=admit(state,'a');
  queue.recordOriginal(a.id,{passed:false,findings:'numeric mismatch',evidence:[{collection:'facts',entry_id:'one'}]},0);
  const before=runtimeStatus(state);assert.equal(before.queues.memory,1);assert.equal(before.queues.postprocessing,1);
  queue.acknowledgeLesson(queue.lessons()[0].id);state.event(a.id,'postprocess-completed',{});
  assert.equal(runtimeStatus(state).queues.recovery,1);
  queue.claim('a',1);
  assert.equal(runtimeStatus(state).queues.recovery,0);
  state.db.prepare('INSERT INTO events(attempt,type,data,created) VALUES (?,?,?,?)').run(a.id,'pi-event','invalid body','now');
  assert.equal(runtimeStatus(state).queues.memory,0);assert.equal(runtimeStatus(state).queues.postprocessing,0);
  for(const limit of [0,1001,Infinity])assert.throws(()=>runtimeStatus(state,{limit}));
});
