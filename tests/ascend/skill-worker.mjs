import test from 'node:test';
import assert from 'node:assert/strict';
import {skillWorker} from '../../cli/skill-worker.mjs';

function fixture({capacity=8,index=2,work=true,fail=false}={}) {
  const ledger=new Map([['campaign-plan',{index}]]),errors=[];
  let calls=0,closed=true,resolveRun;
  const config={campaign_plan:[{},{}],skills:{background:true}};
  const operations={backgroundSkillStatus:async()=>({hasWork:work}),backgroundSkillsSettled:async()=>closed,
    runBackgroundSkills:async(_config,{signal})=>{
      calls++;if(fail)throw new Error('writer failed');closed=false;
      await new Promise(resolve=>{resolveRun=()=>{closed=true;resolve();};signal.addEventListener('abort',resolve,{once:true});});
    }};
  const worker=skillWorker({control:{get:key=>ledger.get(key),set:(key,value)=>ledger.set(key,value),leased:()=>false},
    config:()=>config,remaining:()=>100,event:(_type,data)=>errors.push(data),operations,now:()=>2000});
  return {worker,ledger,errors,resources:Array.from({length:capacity},(_,i)=>({id:'slot'+i,available:true})),
    get calls(){return calls;},finish:()=>resolveRun?.(),settle:()=>{closed=true;}};
}

test('first iteration uses all capacity; later idle tuning permanently earmarks one slot',async()=>{
  const first=fixture({index:0});await first.worker.poll(first.resources,0,8);
  assert.equal(first.worker.reserved,false);assert.equal(first.calls,0);assert.equal(first.worker.resource,undefined);
  first.ledger.set('campaign-plan',{index:2});await first.worker.poll(first.resources,7,8);
  assert.equal(first.worker.reserved,true);assert.ok(first.worker.resource);assert.equal(first.calls,1);
  first.finish();await first.worker.running;assert.ok(first.worker.resource);
  await first.worker.stop();assert.equal(first.worker.resource,null);assert.equal(first.worker.reserved,true);
  const idle=fixture({work:false});await idle.worker.poll(idle.resources,0,8);
  assert.equal(idle.calls,0);assert.equal(idle.worker.reserved,true);assert.equal(idle.worker.resource.id,'slot0');
});

test('zero, one, two and eight devices obey the reserved-slot capacity contract',async()=>{
  for(const capacity of [0,1,2,8]) {
    const f=fixture({capacity});await f.worker.poll(f.resources,0,8);
    assert.equal(f.worker.capacity,capacity);
    assert.equal(f.worker.reserved,true);
    assert.equal(f.calls,capacity?1:0);
    if(capacity){assert.equal(capacity-1+(f.worker.resource?1:0),capacity);f.finish();await f.worker.running;}
    await f.worker.stop();
  }
  const full=fixture();await full.worker.poll(full.resources,8,8);assert.equal(full.calls,0);
});

test('stop checkpoints active evaluation and retains its slot until actual closure',async()=>{
  const f=fixture({capacity:2});await f.worker.poll(f.resources,0,2);
  const slot=f.worker.resource;
  await f.worker.stop('Campaign paused');
  assert.equal(f.worker.resource,slot);assert.equal(f.worker.running,null);
  assert.equal(f.ledger.get('skill-background-resource'),slot);
  f.settle();await f.worker.stop();assert.equal(f.worker.resource,null);
  assert.deepEqual(f.errors,[]);
});

test('background failures leave foreground reservation and campaign operation independent',async()=>{
  const f=fixture({capacity:2,fail:true});
  const running=f.worker.poll(f.resources,1,2);await running;
  assert.equal(f.calls,1);assert.equal(f.worker.reserved,true);
  await f.worker.running;assert.equal(f.errors[0].message,'writer failed');
  assert.equal(f.worker.resource.id,'slot0');await f.worker.stop();
});


test('marked A/B slot is mandatory and never replaced by an idle foreground slot',async()=>{
  const f=fixture();
  f.resources[0].metadata={hostSlot:0,backgroundSlot:true};
  f.resources[0].available=false;
  f.resources[1].metadata={hostSlot:1,backgroundSlot:false};
  await f.worker.poll(f.resources,0,8);
  assert.equal(f.calls,0);assert.equal(f.worker.resource,undefined);
  const available=fixture();
  available.resources[0].metadata={hostSlot:1,backgroundSlot:false};
  available.resources[1].metadata={hostSlot:0,backgroundSlot:true};
  await available.worker.poll(available.resources,0,8);
  assert.equal(available.calls,1);assert.equal(available.worker.resource.id,'slot1');
  available.finish();await available.worker.running;await available.worker.stop();
});
