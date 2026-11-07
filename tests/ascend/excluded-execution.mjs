import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {operationsStopped} from '../../controller/lifecycle.mjs';
import {createEnvironment} from '../../integrations/environments/index.mjs';

async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'excluded-execution-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const state=new State(':memory:');t.after(()=>state.close());
  state.add({id:'p',goal:'Compute a correct result',actor_seconds:600});
  const {attempt}=new RecoveryQueue(state).claim('p',1,{},'cpu');
  state.checkpoint(attempt.id,37.25,'actor.jsonl');
  state.finish(attempt.id,'failed');
  const event=(type,data)=>state.event(attempt.id,type,data);
  const calls=[],control={observation:{state:'completed',exit:0},timing:null};
  const options={mode:'cpl',target:'fake',workspacePath:root,remoteDirectory:'owned'};
  const environment=await createEnvironment(options,{instancePath:root,
    sdk:{createLocalBashOperations:()=>({})},event,exclude:async(_label,fn)=>fn(),
    client:async args=>{
      calls.push(args);
      if(args[0]==='observe')return control.observation;
      if(args[0]==='capabilities')return {state:'available',backend:'ssh'};
      if(args[0]==='download') {
        assert.equal(args.length,4);
        if(control.timing===null)return {backend:'ssh',operation:'download',state:'failed',exit:23};
        await writeFile(args[3],JSON.stringify(control.timing));return {state:'completed'};
      }
      assert.fail('Reconciliation must not dispatch '+args[0]);
    }});
  function submission(id,{excluded=true,closed=true,kind='run'}={}) {
    const handle=`remote:fake:job:${id}`;
    event('execution-submitting',{operation:'op',submission:id,kind,accountingExcluded:excluded,actor_elapsed:37.25});
    event('execution-submitted',{operation:'op',submission:id,handle,accountingExcluded:excluded});
    event('execution-handle',{operation:'op',handle,purpose:excluded?'resource-health':undefined});
    if(closed)event('execution-state',{operation:'op',handle,state:'stopped',exitCode:0,
      ...(excluded?{accountingExcluded:true,executionSeconds:0}:{})});
    return handle;
  }
  return {root,state,id:attempt.id,event,calls,control,environment,submission};
}

test('resume reuses completed excluded probe proofs without SSH replay or changing elapsed/outcome',async t=>{
  const f=await fixture(t);
  for(let i=0;i<60;i++)f.submission('probe-'+i);
  const before=f.state.attempt(f.id);
  assert.equal(await operationsStopped(f.state,before,f.environment),true);
  assert.deepEqual(f.state.attempt(f.id),before);
  assert.deepEqual(f.calls,[]);
  assert.equal(f.state.queryEvents(f.id,{types:['accounting-unresolved']}).length,0);
  assert.equal(await operationsStopped(f.state,before,f.environment),true);
  assert.deepEqual(f.calls,[]);
});

test('upgrade can use terminal excluded-job proofs even when an old local receipt is unavailable',async t=>{
  const f=await fixture(t);
  f.submission('probe');
  f.event('execution-submitting',{operation:'setup',submission:'parent',kind:'run',accountingExcluded:true,
    receiptFile:join(f.root,'removed-old-directory','submission.json')});
  f.event('execution-submitted',{operation:'setup',submission:'parent',handle:'remote:fake:job:parent',accountingExcluded:true});
  f.event('execution-state',{handle:'remote:fake:job:parent',state:'stopped',accountingExcluded:true,executionSeconds:0});
  assert.equal(await f.environment.canUpgrade({attempt:f.state.attempt(f.id),events:f.state.events(f.id)}),true);
  assert.deepEqual(f.calls,[]);
});

test('reused terminal proofs still reject handles from another configured target',async t=>{
  const f=await fixture(t);f.submission('probe');
  const events=f.state.events(f.id).map(e=>({...e,data:{...e.data,
    ...(e.data.handle?{handle:e.data.handle.replace(':fake:',':other:')}:{})}}));
  await assert.rejects(f.environment.reconcileOperation({operation:'op',events}));
  assert.deepEqual(f.calls,[]);
});

test('unobserved excluded jobs remain fenced until the exact native handle is terminal',async t=>{
  const f=await fixture(t),handle=f.submission('unfinished',{closed:false});
  f.control.observation={state:'observation-unavailable'};
  assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),false);
  f.control.observation={state:'running'};
  assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),false);
  f.control.observation={state:'completed',exit:0};
  assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),true);
  assert.ok(f.calls.every(args=>args[0]==='observe'&&args[1]===handle));
  assert.equal(f.state.attempt(f.id).elapsed,37.25);
  assert.equal(f.state.queryEvents(f.id,{types:['accounting-unresolved']}).length,0);
});

test('a terminal payload is never treated as free work when its same-host timing is missing or invalid',async t=>{
  const f=await fixture(t),handle=f.submission('payload',{excluded:false});
  const dir=join(f.root,'.executions','exec-payload');await mkdir(dir);
  await writeFile(join(dir,'execution.json'),JSON.stringify({handle,spool:'owned/a/output',timingKey:'payload-key'}));
  assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),false);
  f.control.timing={completed:true,executionSeconds:4.25,key:'other-execution'};
  assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),false);
  f.control.timing={completed:true,executionSeconds:4.25,key:'payload-key'};
  // The downloaded mismatched receipt is not proof; a later valid receipt can be recovered.
  await rm(join(dir,'timing.json'));
  assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),true);
  assert.equal(f.state.attempt(f.id).elapsed,41.5);
  assert.equal(f.state.attempt(f.id).outcome,'failed');
  assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),true);
  assert.equal(f.state.attempt(f.id).elapsed,41.5);
  assert.ok(f.calls.some(args=>args[0]==='download'));
});

test('a stopped event without pre-submission exclusion does not bypass payload timing',async t=>{
  const f=await fixture(t),handle=f.submission('payload',{excluded:false});
  f.event('execution-state',{handle,state:'stopped',accountingExcluded:true,executionSeconds:0});
  assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),false);
  assert.equal(f.state.attempt(f.id).elapsed,37.25);
  assert.ok(f.state.queryEvents(f.id,{types:['accounting-unresolved']}).length>0);
});
