import test from 'node:test';
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import * as sdk from '@earendil-works/pi-coding-agent';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {attemptStopped,operationsStopped,reconcileFailure,failureSettled,processIdentity} from '../../controller/lifecycle.mjs';
import {createEnvironment} from '../../integrations/environments/index.mjs';

const limits={tasks:16,memory_bytes:128*1024**2};
async function fixture(t,{failed=true,output='LoadState=loaded\nActiveState=failed\nControlGroup=\n',exitCode=0}={}) {
  const root=await mkdtemp(join(tmpdir(),'scope-settlement-'));t.after(()=>rm(root,{recursive:true,force:true}));
  let state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  state.add({id:'p',goal:'Compute a correct result',actor_seconds:600});
  const me=await processIdentity(),old={...me,start:String(BigInt(me.start)-1n)};
  const {attempt}=new RecoveryQueue(state).claim('p',1,old,'cpu');
  state.checkpoint(attempt.id,37.25,'retained-session.jsonl');
  if(failed)state.finish(attempt.id,'failed');
  const unit='rsi-command-'+randomUUID()+'.scope',handle='local:exec-probe',dir=join(root,'.executions',handle.slice(6));
  await mkdir(dir,{recursive:true});
  await writeFile(join(dir,'execution.json'),JSON.stringify({boot:me.boot,cwd:root,
    containment:{unit,closed:false},failure:{message:'timeout:5'},
    outcome:{kind:'infrastructure',reason:'local-execution-interrupted'}}));
  const event=(type,data)=>state.event(attempt.id,type,data);
  event('execution-starting',{operation:'op',startElapsed:37.25});
  event('execution-submitting',{operation:'op',submission:'probe',kind:'run',accountingExcluded:true});
  event('execution-submitted',{operation:'op',submission:'probe',handle,accountingExcluded:true});
  event('execution-handle',{operation:'op',handle,purpose:'resource-health'});
  event('execution-error',{operation:'op',handle,message:'timeout:5',purpose:'resource-health'});
  event('execution-unresolved',{operation:'op',handle:null,startElapsed:37.25,message:'aborted'});
  if(failed)event('failure-settlement',{closed:false});
  const control={output,exitCode,calls:[]};
  const observerSdk={createLocalBashOperations:()=>({exec:async(command,_cwd,options)=>{
    control.calls.push(command);
    assert.ok(command.startsWith('systemctl --user show '),'Reconciliation must only observe the retained scope');
    assert.ok(command.includes(unit));
    if(control.error)throw control.error;
    options.onData(Buffer.from(control.output));return {exitCode:control.exitCode};
  }})};
  const environment=()=>createEnvironment({mode:'local',localExecution:limits},{instancePath:root,sdk:observerSdk,
    event,exclude:async(_label,fn)=>fn()});
  return {root,dir,unit,handle,control,id:attempt.id,event,environment,state:()=>state,
    restart:()=>{state.close();state=new State(join(root,'state.sqlite'));}};
}

for(const status of ['LoadState=not-found\nActiveState=inactive\nControlGroup=\n',
  'LoadState=loaded\nActiveState=failed\nControlGroup=\n',
  'LoadState=loaded\nActiveState=inactive\nControlGroup=\n']) {
  test('restart settles an aborted excluded probe after OS scope closure: '+status.split('\n').slice(0,2).join(' '),async t=>{
    const f=await fixture(t,{output:status});f.restart();const environment=await f.environment();
    assert.equal(await reconcileFailure(f.state(),f.id,environment),true);
    assert.equal(failureSettled(f.state(),f.id),true);
    assert.equal(f.state().attempt(f.id).elapsed,37.25);
    assert.equal(f.state().attempt(f.id).outcome,'failed');
    assert.ok(f.state().queryEvents(f.id,{types:['execution-reconciled']}).some(e=>
      e.data.operation==='op'&&e.data.closed===true&&e.data.executionSeconds===0));
    assert.ok(f.control.calls.length>0);
    const receipt=JSON.parse(await readFile(join(f.dir,'execution.json')));
    assert.equal(receipt.failure.message,'timeout:5');
    const observed=await environment.observe(f.handle,{accountingExcluded:true});
    assert.equal(observed.state,'stopped');assert.equal(observed.outcome.reason,'local-execution-interrupted');
  });
}

test('a paused attempt resumes its original session and elapsed allocation after scope closure',async t=>{
  const f=await fixture(t,{failed:false});f.restart();const environment=await f.environment(),owner=await processIdentity();
  const resumed=await f.state().resume(f.id,a=>attemptStopped(f.state(),a,environment),owner);
  assert.equal(resumed.id,f.id);assert.equal(resumed.session,'retained-session.jsonl');
  assert.equal(resumed.elapsed,37.25);assert.equal(resumed.outcome,null);assert.equal(resumed.owner,JSON.stringify(owner));
});

for(const [label,output,exitCode] of [
  ['running scope','LoadState=loaded\nActiveState=active\nControlGroup=/user.slice/retained.scope\n',0],
  ['failed scope with remaining control group','LoadState=loaded\nActiveState=failed\nControlGroup=/user.slice/retained.scope\n',0],
  ['missing scope with contradictory group','LoadState=not-found\nActiveState=inactive\nControlGroup=/user.slice/retained.scope\n',0],
  ['query error despite terminal-looking output','LoadState=not-found\nActiveState=inactive\nControlGroup=\n',1],
  ['missing load state','ActiveState=inactive\nControlGroup=\n',0],
  ['missing active state','LoadState=loaded\nControlGroup=\n',0],
  ['missing control group','LoadState=loaded\nActiveState=failed\n',0],
  ['empty query output','',0],
])test('settlement keeps its fence for '+label,async t=>{
  const f=await fixture(t,{output,exitCode}),environment=await f.environment();
  assert.notEqual((await environment.localOperations.observe(f.unit)).state,'stopped');
  assert.equal(await reconcileFailure(f.state(),f.id,environment),false);
  assert.equal(failureSettled(f.state(),f.id),false);assert.equal(f.state().attempt(f.id).elapsed,37.25);
});

test('an unavailable systemd observer stays fenced and recovers on a later successful observation',async t=>{
  const f=await fixture(t),environment=await f.environment();f.control.error=Error('timeout:5');
  assert.equal(await operationsStopped(f.state(),f.state().attempt(f.id),environment),false);
  delete f.control.error;
  assert.equal(await operationsStopped(f.state(),f.state().attempt(f.id),environment),true);
});

test('a scope cannot reconcile an aborted operation without its recorded command identity',async t=>{
  const f=await fixture(t),environment=await f.environment();
  f.state().db.prepare("UPDATE events SET data=json_remove(data,'$.handle') WHERE type IN ('execution-submitted','execution-handle','execution-error')").run();
  assert.equal(await operationsStopped(f.state(),f.state().attempt(f.id),environment),false);
  assert.deepEqual(f.control.calls,[]);
});

test('scope status from another boot cannot close the retained execution',async t=>{
  const f=await fixture(t),receipt=JSON.parse(await readFile(join(f.dir,'execution.json')));
  await writeFile(join(f.dir,'execution.json'),JSON.stringify({...receipt,boot:'previous-boot'}));
  assert.equal(await operationsStopped(f.state(),f.state().attempt(f.id),await f.environment()),false);
  assert.deepEqual(f.control.calls,[]);
});

test('actual Pi/systemd execution recovers when cleanup completed before its closed receipt was persisted',async t=>{
  let available='';
  try {
    const result=await sdk.createLocalBashOperations().exec('systemctl --user show --property=ControlGroup --value',process.cwd(),
      {timeout:3,onData:chunk=>available+=chunk});
    if(result.exitCode!==0||!available.startsWith('/'))throw Error('No user manager');
  }catch{t.skip('Native scope test requires a user systemd manager');return;}
  const f=await fixture(t),environment=await createEnvironment({mode:'local',localExecution:limits},
    {instancePath:f.root,sdk,event:async()=>{},exclude:async(_label,fn)=>fn()});
  const result=await environment.execute('printf once >> payload-runs.txt',f.root,{timeout:3});assert.equal(result.exitCode,0);
  const dir=join(f.root,'.executions',result.handle.slice(6)),path=join(dir,'execution.json');
  const receipt=JSON.parse(await readFile(path));assert.equal(receipt.containment.closed,true);
  await writeFile(path,JSON.stringify({...receipt,containment:{...receipt.containment,closed:false}}));
  const observer=await createEnvironment({mode:'local',localExecution:limits},
    {instancePath:f.root,sdk,event:async()=>{},exclude:async(_label,fn)=>fn()});
  assert.equal((await observer.observe(result.handle)).state,'stopped');
  assert.equal(await readFile(join(f.root,'payload-runs.txt'),'utf8'),'once');
});
