import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {operationsStopped} from '../../controller/lifecycle.mjs';
import {resume,settled} from '../../cli/evaluation.mjs';

function fixture(t,path=':memory:') {
  const state=new State(path);let open=true;
  const close=()=>{if(open){open=false;state.close();}};t.after(close);
  state.add({id:'p',goal:'independent numerical check',actor_seconds:600});
  const attempt=state.admit('p',{pid:2147483647,start:'0',boot:'old'});
  state.checkpoint(attempt.id,18.5,'retained-session.jsonl');
  state.finish(attempt.id,'passed');
  return {state,id:attempt.id,event:(type,data)=>state.event(attempt.id,type,data),close};
}
const remote=(n,kind='job')=>`remote:target:${kind}:command-${n}`;

// The real FA history needed 65 serial native observations, longer than the
// settlement window. Already terminal commands must consume no new requests.
test('completed evaluation settles a long remote history without replaying native observations',async t=>{
  const {state,id,event}=fixture(t),before=state.attempt(id);
  for(let n=0;n<65;n++) {
    const handle=remote(n,n%2?'job':'transfer');
    event('execution-handle',{handle});
    event('execution-state',{handle,state:'stopped',nativeState:['completed','failed','cancelled'][n%3],executionSeconds:n+1});
    event('execution-state',{handle,state:'unknown'});
  }
  assert.equal(await operationsStopped(state,before,{observe:()=>assert.fail('historical terminal command queried again')}),true);
  assert.deepEqual(state.attempt(id),before);
});

test('interrupted closure scan persists its terminal prefix across process restart',async t=>{
  const root=await mkdtemp(join(tmpdir(),'settlement-progress-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const path=join(root,'state.sqlite'),{state,id,event,close}=fixture(t,path);
  const handles=Array.from({length:5},(_,n)=>remote(n));
  for(const handle of handles)event('execution-handle',{handle});
  const interrupted=new AbortController(),first=[];
  await assert.rejects(operationsStopped(state,state.attempt(id),{observe:async handle=>{
    interrupted.signal.throwIfAborted();first.push(handle);
    if(first.length===2)interrupted.abort();
    return {state:'stopped',nativeState:'completed',exitCode:0};
  }}),{name:'AbortError'});
  assert.deepEqual(first,handles.slice(0,2));
  close();
  const next=new State(path);t.after(()=>next.close());
  const remaining=[];
  assert.equal(await operationsStopped(next,next.attempt(id),{observe:async handle=>{
    remaining.push(handle);return {state:'stopped',nativeState:'completed'};
  }}),true);
  assert.deepEqual(remaining,handles.slice(2));
  assert.equal(next.attempt(id).elapsed,18.5);
  assert.equal(next.attempt(id).outcome,'passed');
  assert.equal(next.attempt(id).session,'retained-session.jsonl');
});

test('upgraded evaluator recovers a passed sample using its unchanged frozen environment plugin',async t=>{
  const root=await mkdtemp(join(tmpdir(),'evaluation-settlement-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const module=join(root,'frozen-environment.mjs');
  await writeFile(module,`export async function createEnvironment(){return {observe(){throw Error('Native request unavailable');}};}`);
  const original=await readFile(module);
  await writeFile(join(root,'instance.json'),JSON.stringify({format:1,mode:'phase2',environment:{module,options:{}}}));
  const {state,id,event}=fixture(t,join(root,'state.sqlite'));
  await mkdir(join(root,'attempts',id),{recursive:true});
  await writeFile(join(root,'attempts',id,'job.json'),JSON.stringify({resource:{id:'saved-slot'}}));
  event('accepted',{});
  for(let n=0;n<65;n++) {
    const handle=remote(n);event('execution-handle',{handle});
    event('execution-state',{handle,state:'stopped',nativeState:'completed'});
  }
  const before=state.attempt(id);
  assert.equal(await settled({instanceRoot:root}),true);
  const result=await resume({instanceRoot:root});
  assert.equal(result.correct,true);assert.equal(result.score,-18.5);assert.deepEqual(result.attempts,[id]);
  assert.deepEqual(state.attempt(id),before);
  assert.deepEqual(await readFile(module),original);
  assert.equal(state.db.prepare('SELECT COUNT(*) n FROM attempts').get().n,1);
});

for(const [handle,proof] of [
  ['local:exec-retained',{state:'stopped',nativeState:'completed'}],
  ['fixture:remote',{state:'stopped',nativeState:'completed'}],
  [remote(0),{state:'stopped'}],
  [remote(0),{state:'unknown',nativeState:'completed'}],
  [remote(0),{state:'stopped',nativeState:'running'}],
])test(`unproved closure remains observable: ${handle} ${JSON.stringify(proof)}`,async t=>{
  const {state,id,event}=fixture(t);event('execution-handle',{handle});event('execution-state',{handle,...proof});
  const seen=[];
  assert.equal(await operationsStopped(state,state.attempt(id),{observe:async handle=>{
    seen.push(handle);return {state:'unknown'};
  }}),false);
  assert.deepEqual(seen,[handle]);
});

test('cached terminal work cannot hide a later live handle or an unidentified submission',async t=>{
  const {state,id,event}=fixture(t),old=remote(0),live=remote(1);
  event('execution-handle',{handle:old});event('execution-state',{handle:old,state:'stopped',nativeState:'completed'});
  event('execution-handle',{handle:live});const seen=[];
  assert.equal(await operationsStopped(state,state.attempt(id),{observe:async handle=>{seen.push(handle);return {state:'running'};}}),false);
  assert.deepEqual(seen,[live]);
  event('execution-unresolved',{operation:'later',handle:null});
  assert.equal(await operationsStopped(state,state.attempt(id),{reconcileOperation:async()=>({closed:false}),observe:()=>assert.fail('unidentified submission ignored')}),false);
});

test('terminal closure never substitutes for pending operation timing reconciliation',async t=>{
  const {state,id,event}=fixture(t),handle=remote(0);
  event('execution-starting',{operation:'op',startElapsed:19});
  event('execution-submitting',{operation:'op',submission:'s',kind:'run',accountingExcluded:false});
  event('execution-handle',{operation:'op',handle});
  event('execution-state',{handle,state:'stopped',nativeState:'completed',executionSeconds:2.25});
  assert.equal(await operationsStopped(state,state.attempt(id),{reconcileOperation:async()=>({closed:false})}),false);
  assert.equal(state.attempt(id).elapsed,18.5);
  let reconciled=0;
  const environment={reconcileOperation:async()=>{reconciled++;return {closed:true,executionSeconds:2.25,handles:[handle]};},
    observe:()=>assert.fail('terminal command queried again')};
  assert.equal(await operationsStopped(state,state.attempt(id),environment),true);
  assert.equal(await operationsStopped(state,state.attempt(id),environment),true);
  assert.equal(reconciled,1);assert.equal(state.attempt(id).elapsed,21.25);
  assert.equal(state.attempt(id).outcome,'passed');
});
