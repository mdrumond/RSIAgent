import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {settleFailure,reconcileFailure,failureSettled} from '../../controller/lifecycle.mjs';
import {operate} from '../../cli/runtime.mjs';
import {findingsFor} from '../../cli/findings.mjs';

function fixture(t) {
  const state=new State(':memory:'),queue=new RecoveryQueue(state);
  t.after(()=>state.close());
  state.add({id:'p',goal:'numeric',actor_seconds:17,verifier:{files:[],command:'python3 verify.py'}});
  const {attempt}=queue.claim('p',1,{},'cpu-0');
  state.checkpoint(attempt.id,4,'session.jsonl');
  return {state,queue,id:attempt.id};
}
const result={passed:false,useful:false,findings:'Factual failure',evidence:[{path:'failure.json',sha256:'hash'}]};

test('failed outcome with unknown execution fences lease and recovery until closure, without replay',async t=>{
  const {state,queue,id}=fixture(t);
  state.event(id,'execution-starting',{operation:'op',startElapsed:4});
  const env={reconcileOperation:async()=>({closed:false})};
  assert.equal(await settleFailure(state,id,env,{message:'actor died'}),false);
  assert.equal(state.attempt(id).outcome,'failed');
  assert.equal(failureSettled(state,id),false);
  assert.equal(queue.pendingOutcomes().length,0);
  assert.throws(()=>queue.recordOriginal(id,result,1),/settlement/);
  assert.throws(()=>queue.enqueue(id,1),/settlement/);
  await assert.rejects(state.resume(id,async()=>true,{}),/not resumable/);
  assert.equal(state.db.prepare('SELECT COUNT(*) AS n FROM resource_leases').get().n,1);
  env.reconcileOperation=async()=>({closed:true,executionSeconds:3});
  assert.equal(await reconcileFailure(state,id,env),true);
  assert.equal(state.attempt(id).elapsed,7);
  assert.equal(state.attempt(id).session,'session.jsonl');
  assert.equal(queue.pendingOutcomes().length,1);
  queue.recordOriginal(id,result,1);
  assert.equal(queue.ready(2).length,0); // Lesson delivery remains mandatory.
  queue.acknowledgeLesson(queue.lessons()[0].id);
  assert.equal(queue.ready(2).length,1);
  assert.equal(state.project('p').spec.actor_seconds,17);
});

test('observation errors and detached native shells preserve the fence',async t=>{
  const {state,id}=fixture(t);
  state.event(id,'execution-handle',{handle:'h'});
  assert.equal(await settleFailure(state,id,{observe:async()=>{throw new Error('offline');}}),false);
  assert.equal(state.queryEvents(id,{types:['settlement-error']}).length,1);
  state.event(id,'native-bash-started',{operation:'bash'});
  assert.equal(await reconcileFailure(state,id,{observe:async()=>({state:'stopped'})}),false);
  state.event(id,'native-bash-completed',{operation:'bash',actor_elapsed:6});
  assert.equal(await reconcileFailure(state,id,{observe:async()=>({state:'stopped'})}),true);
  assert.equal(state.attempt(id).elapsed,6);
});

test('failed summary leaves deterministic hashed failure evidence after factual persistence',async t=>{
  const {state,id}=fixture(t),path=await mkdtemp(join(tmpdir(),'failure-summary-'));
  t.after(()=>rm(path,{recursive:true,force:true}));
  await settleFailure(state,id,{}, {message:'provider unavailable'});
  const findings=await findingsFor({state,path,job:{attempt:state.attempt(id),kind:'original'},
    passed:false,config:{},bridge:{}});
  assert.equal(findings.useful,false);
  assert.equal(state.queryEvents(id,{types:['summary-error']}).length,1);
  const evidence=JSON.parse(await readFile(findings.evidence[0].path,'utf8'));
  assert.ok(evidence.some(e=>e.type==='failure-recorded'&&e.data.message==='provider unavailable'));
  assert.match(findings.evidence[0].sha256,/^[a-f0-9]{64}$/);
  assert.deepEqual(await findingsFor({state,path,job:{attempt:{id},kind:'original'},passed:false}),findings);
});

for(const code of [0,9,'throw','signal','unknown'])test(`worker exit ${code} with null outcome fails only its project and continues unrelated jobs`,async t=>{
  const root=await mkdtemp(join(tmpdir(),'failure-worker-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  await mkdir(join(root,'skills','versions','v'),{recursive:true});
  // The scheduler stays alive; each real child exits during environment setup.
  await writeFile(join(root,'environment.mjs'),`
    export function createEnvironment(options,{instancePath,event}) {
      if(process.argv[2]==='worker') {
        ${code==='throw'?"throw new Error('worker setup failed');":code==='signal'?"process.kill(process.pid,'SIGKILL');":code==='unknown'?"event('execution-handle',{handle:'unobserved'});process.exit(8);":`process.exit(${code});`}
      }
      return {observe:async()=>({state:'unknown'}),resources:async()=>[{id:'cpu-0',available:true},{id:'cpu-1',available:true}]};
    }`);
  const config={format:1,root,mode:'phase1',python:join(root,'missing-python'),recovery:true,skills:{activeVersion:'v'},
    environment:{module:'environment.mjs'},projects:['a','b'].map(id=>({id,goal:'numeric',actor_seconds:3}))};
  await writeFile(join(root,'instance.json'),JSON.stringify(config));
  const output=await operate('wave',config,{concurrency:1});
  assert.equal(output.results.length,2);
  assert.ok(output.results.every(r=>r.outcome==='failed'));
  const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  assert.ok(state.failures(output.results[0].attempt).length>=1);
  assert.equal(state.db.prepare('SELECT COUNT(*) AS n FROM resource_leases').get().n,code==='unknown'?2:0);
  assert.equal(new RecoveryQueue(state).pendingOutcomes().length,['unknown','throw'].includes(code)?0:2);
  if(code==='unknown') {
    const resumed=await operate('resume',config);
    assert.equal(resumed.results.length,0);
    assert.equal(state.db.prepare('SELECT COUNT(*) AS n FROM attempts').get().n,2);
  }
});

test('failure outcome and unknown fence roll back together when persistence fails',async t=>{
  const {state,id}=fixture(t),event=state.event.bind(state);
  state.event=(id,type,data)=>{if(type==='failure-settlement')throw new Error('disk full');return event(id,type,data);};
  await assert.rejects(settleFailure(state,id,{}),/disk full/);
  assert.equal(state.attempt(id).outcome,null);
  assert.equal(state.project('p').status,'running');
  assert.equal(state.failures(id).length,0);
});

test('unknown settlement prevents promoting a useful investigation to a retry',async t=>{
  const {state,queue,id}=fixture(t);
  state.finish(id,'failed');queue.recordOriginal(id,result,1);
  queue.acknowledgeLesson(queue.lessons()[0].id);
  const child=queue.claim('p',2,{},'cpu-1').attempt;
  state.event(child.id,'execution-handle',{handle:'remote'});
  await settleFailure(state,child.id,{observe:async()=>({state:'unknown'})});
  assert.throws(()=>queue.complete(child.id,{...result,useful:true},2),/settlement/);
  assert.equal(queue.entry('p').attempt,child.id);
  assert.equal(queue.entry('p').kind,'investigation');
  assert.equal(await reconcileFailure(state,child.id,{observe:async()=>({state:'stopped'})}),true);
  queue.complete(child.id,{...result,useful:true},2);
  assert.equal(queue.entry('p').kind,'retry');
});
