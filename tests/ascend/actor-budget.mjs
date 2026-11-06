import test from 'node:test';
import assert from 'node:assert/strict';
import {setTimeout as delay} from 'node:timers/promises';
import {ActorClock} from '../../controller/budget.mjs';
import {State} from '../../controller/state.mjs';

test('20ms State checkpoints stay monotonic through busy admission then charged payload',async()=>{
  const state=new State(':memory:');
  try {
    state.add({id:'clock',goal:'monotonic accounting',actor_seconds:10});
    const attempt=state.admit('clock',{}),clock=new ActorClock(10);
    await delay(30); // Thinking remains charged.
    const samples=[];
    function checkpoint(){const elapsed=clock.elapsed;state.checkpoint(attempt.id,elapsed);samples.push(elapsed);}
    checkpoint();
    const beforeBusy=clock.elapsed;
    await clock.exclude('admission',async()=>{
      const held=clock.elapsed;
      for(let i=0;i<5;i++){await delay(20);checkpoint();assert.equal(clock.elapsed,held);}
      return {queueWait:true};
    });
    checkpoint();
    assert.ok(clock.elapsed-beforeBusy<.03,'100ms busy response is excluded');
    const beforePayload=clock.elapsed;
    await clock.exclude('admission',async()=>{
      const held=clock.elapsed;
      for(let i=0;i<5;i++){await delay(20);checkpoint();assert.equal(clock.elapsed,held);}
      return {queueWait:false};
    });
    checkpoint();
    assert.ok(clock.elapsed-beforePayload>=.09,'payload time is charged at settlement');
    assert.ok(samples[0]>=.02,'thinking time is charged');
    assert.ok(samples.every((value,index)=>!index||value>=samples[index-1]));
    assert.equal(state.attempt(attempt.id).elapsed,samples.at(-1));
  } finally {state.close();}
});

test('admission excludes nested transfers and preserves an outer setup exclusion',async()=>{
  let now=0;const clock=new ActorClock(30,2,()=>now);
  await clock.exclude('admission',async()=>{
    now+=1000;assert.equal(clock.elapsed,2);
    await clock.exclude('transfer',async()=>{now+=5000;assert.equal(clock.elapsed,2);});
    now+=1000;return {queueWait:false};
  });
  assert.equal(clock.elapsed,4);
  await clock.exclude('setup',()=>clock.exclude('admission',async()=>{
    now+=9000;await clock.exclude('transfer',async()=>{now+=5000;});return {queueWait:false};
  }));
  assert.equal(clock.elapsed,4);
  await assert.rejects(clock.exclude('admission',async()=>{
    now+=1000;assert.equal(clock.elapsed,4);
    await clock.exclude('setup',async()=>{now+=7000;});throw new Error('unknown');
  }));
  assert.equal(clock.elapsed,5,'settled unknown charges its billable interval');
  now+=500;assert.equal(clock.elapsed,5.5);
});
