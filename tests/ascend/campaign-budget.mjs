import test from 'node:test';
import assert from 'node:assert/strict';
import {State} from '../../controller/state.mjs';
import {campaignBudget} from '../../controller/campaign-budget.mjs';

test('automatic restart retains elapsed time and charges the active gap',t=>{
  const state=new State(':memory:');t.after(()=>state.close());let tick=1000;
  let budget=campaignBudget(state,{seconds:10,cycle:'one',boot:'boot',now:()=>tick});
  tick+=4000;budget.checkpoint();assert.equal(budget.remaining,6);
  tick+=3000;budget=campaignBudget(state,{seconds:10,cycle:'one',boot:'boot',now:()=>tick});
  assert.equal(budget.elapsed,7);assert.equal(budget.remaining,3);
  tick+=4000;assert.equal(budget.remaining,0);assert.equal(budget.stop().elapsed,11);
});
test('pause excludes inactive time and resume never renews the allocation',t=>{
  const state=new State(':memory:');t.after(()=>state.close());let tick=0;
  let budget=campaignBudget(state,{seconds:10,cycle:'one',boot:'boot',now:()=>tick});
  tick=6000;budget.stop();tick=100000;
  assert.equal(budget.elapsed,6);assert.equal(budget.remaining,4);
  budget=campaignBudget(state,{seconds:10,cycle:'one',boot:'boot',now:()=>tick});
  assert.equal(budget.remaining,4);tick+=1000;assert.equal(budget.remaining,3);
});
test('host reboot preserves saved elapsed; only explicit new cycle starts a new allocation',t=>{
  const state=new State(':memory:');t.after(()=>state.close());let tick=10000;
  const old=campaignBudget(state,{seconds:10,cycle:'one',boot:'old',now:()=>tick});
  tick+=5000;old.checkpoint();tick=100;
  const resumed=campaignBudget(state,{seconds:10,cycle:'one',boot:'new',now:()=>tick});
  assert.equal(resumed.remaining,5);
  const fresh=campaignBudget(state,{seconds:10,cycle:'two',boot:'new',now:()=>tick});
  assert.equal(fresh.remaining,10);
  assert.equal(state.db.prepare('SELECT COUNT(*) n FROM attempts').get().n,0);
});
test('changing a resumed limit retains spending and invalid timing refuses to overwrite it',t=>{
  const state=new State(':memory:');t.after(()=>state.close());let tick=0;
  const old=campaignBudget(state,{seconds:10,cycle:'one',boot:'same',now:()=>tick});tick=5000;old.stop();
  assert.equal(campaignBudget(state,{seconds:20,cycle:'one',boot:'same',now:()=>tick}).remaining,15);
  for(const seconds of [0,-1,Infinity,NaN])assert.throws(()=>campaignBudget(state,{seconds,cycle:'one',boot:'same',now:()=>tick}));
  const saved=JSON.parse(state.db.prepare("SELECT value FROM control WHERE key='campaign-budget'").get().value);
  assert.equal(saved.elapsed,5);
});
