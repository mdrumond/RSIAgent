import test from 'node:test';
import assert from 'node:assert/strict';
import {State} from '../../controller/state.mjs';
import {requests,reconcileRequests} from '../../cli/background.mjs';

for(const closed of [false,true])test(`request owner death requires whole-service closure=${closed}`,async t=>{
  const state=new State(':memory:');t.after(()=>state.close());requests(state);
  const owner={pid:2147483647,start:'1',boot:'old',unit:'rsi-maintenance-known.service',invocation:'known',cgroup:'/owned/service'};
  state.db.prepare("INSERT INTO requests(id,kind,status,owner) VALUES ('r','tune','running',?)").run(JSON.stringify(owner));
  let observations=0;
  const manager={observeManagedMaintenance:async actual=>{
    assert.deepEqual(actual,owner);observations++;return {closed:async()=>closed};
  }};
  const [row]=await reconcileRequests('/unused',{state,manager});
  assert.equal(observations,1);
  assert.equal(row.status,closed?'failed':'running');
  assert.equal(row.result,null);
});
test('successful request remains successful after its service owner exits',async t=>{
  const state=new State(':memory:');t.after(()=>state.close());requests(state);
  state.db.prepare("INSERT INTO requests(id,kind,status,result) VALUES ('r','kdb','passed',?)").run(JSON.stringify({indexed:3}));
  const manager={observeManagedMaintenance:async()=>assert.fail('Completed requests must not be replayed or failed')};
  const [row]=await reconcileRequests('/unused',{state,manager});
  assert.equal(row.status,'passed');assert.deepEqual(row.result,{indexed:3});
});


test('stop closes an owned maintenance service even beyond the bounded status page',async t=>{
  const {mkdtemp,rm}=await import('node:fs/promises'),{tmpdir}=await import('node:os'),{join}=await import('node:path');
  const {operate}=await import('../../cli/runtime.mjs'),{processIdentity}=await import('../../controller/lifecycle.mjs');
  const root=await mkdtemp(join(tmpdir(),'maintenance-stop-')),state=new State(join(root,'state.sqlite'));
  t.after(async()=>{state.close();await rm(root,{recursive:true,force:true});});requests(state);
  const owner={...await processIdentity(),unit:'rsi-maintenance-owned.service',invocation:'known',cgroup:'/owned/service'};
  state.db.prepare("INSERT INTO requests(id,kind,status,owner) VALUES ('old','tune','running',?)").run(JSON.stringify(owner));
  for(let index=0;index<101;index++)state.db.prepare("INSERT INTO requests(id,kind,status) VALUES (?,'kdb','passed')").run('recent-'+index);
  assert.equal(requests(state).some(row=>row.id==='old'),false);
  let stopped=false;
  const manager={observeManagedMaintenance:async actual=>{assert.deepEqual(actual,owner);return {stop:async()=>{stopped=true;return {closed:true};}};}};
  await operate('stop',{root,mode:'phase1'}, {},manager);
  assert(stopped);
});
