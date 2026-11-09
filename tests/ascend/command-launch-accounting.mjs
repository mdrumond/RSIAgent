import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createEnvironment} from '../../integrations/environments/index.mjs';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {operationsStopped,reconcileFailure,failureSettled} from '../../controller/lifecycle.mjs';
import {campaignProgress} from '../../controller/progress.mjs';

async function fixture(t,{launch=true}={}) {
 const root=await mkdtemp(join(tmpdir(),'command-accounting-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const state=new State(':memory:');t.after(()=>state.close());state.add({id:'p',goal:'compute',actor_seconds:600});
 const {attempt}=new RecoveryQueue(state).claim('p',1,{},'cpu');state.checkpoint(attempt.id,10);state.finish(attempt.id,'failed');
 const handle='remote:fake:job:payload',dir=join(root,'.executions/exec-payload');await mkdir(dir,{recursive:true});
 const record={handle,spool:'owned/output',timingKey:'key',...(launch?{launch:{wallMs:Date.now()-4000}}:{})};
 await writeFile(join(dir,'execution.json'),JSON.stringify(record));
 const control={state:'failed',exit:null,receipt:null,downloads:0};
 const options={mode:'cpl',target:'fake'};
 const context={instancePath:root,sdk:{createLocalBashOperations:()=>({})},event:(type,data)=>state.event(attempt.id,type,data),exclude:async(_label,fn)=>fn(),client:async args=>{
  if(args[0]==='observe')return {state:control.state,exit:control.exit,reason:{message:'open failed'}};
  if(args[0]==='capabilities')return {state:'available',backend:'ssh'};
  if(args[0]==='download'){
   control.downloads++;if(!control.receipt)return {backend:'ssh',operation:'download',state:'failed',exit:23};
   await writeFile(args[3],JSON.stringify(control.receipt));return {state:'completed'};
  }
  assert.fail('Unexpected command replay '+args[0]);
 }};
 const event=(type,data)=>state.event(attempt.id,type,data);
 event('execution-starting',{operation:'op',startElapsed:10});
 event('execution-submitting',{operation:'op',submission:'s',kind:'run',accountingExcluded:false});
 event('execution-submitted',{operation:'op',submission:'s',handle});
 event('execution-handle',{operation:'op',handle});
 return {root,dir,record,state,id:attempt.id,control,context,options,event,handle,environment:await createEnvironment(options,context)};
}

test('terminal launch failure without a wrapper receipt settles and preserves its failed outcome',async t=>{
 const f=await fixture(t);assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),true);
 assert.ok(f.state.attempt(f.id).elapsed>=14);assert.equal(f.state.attempt(f.id).outcome,'failed');
 const observed=await f.environment.observe(f.handle);assert.equal(observed.timingSource,'controller-launch');
 assert.equal(observed.state,'stopped');assert.equal(observed.timingEstimated,true);
});

test('first terminal fallback freezes across later observation and observer restart',async t=>{
 const f=await fixture(t);const first=await f.environment.observe(f.handle);
 t.mock.method(Date,'now',()=>2_000_000_000_000);
 const restarted=await createEnvironment(f.options,f.context),second=await restarted.observe(f.handle);
 assert.ok(first.executionSeconds>=4);assert.equal(second.executionSeconds,first.executionSeconds);
 assert.equal(JSON.parse(await readFile(join(f.dir,'execution.json'))).controllerTiming.terminal,true);
});

test('unknown status is accounted but never supplies process closure',async t=>{
 const f=await fixture(t);f.control.state='observation-unavailable';
 const unknown=await f.environment.observe(f.handle);assert.equal(unknown.state,'unknown');assert.ok(unknown.executionSeconds>=4);
 assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),false);
 assert.ok(f.state.attempt(f.id).elapsed>=14);assert.equal(f.control.downloads,0);
 f.control.state='failed';assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),true);
});

test('valid matching wrapper timing is preferred after fallback accounting',async t=>{
 const f=await fixture(t);const first=await f.environment.observe(f.handle);assert.equal(first.timingSource,'controller-launch');
 f.control.receipt={completed:true,key:'key',executionSeconds:1.25};
 const authoritative=await (await createEnvironment(f.options,f.context)).observe(f.handle);
 assert.equal(authoritative.executionSeconds,1.25);assert.equal(authoritative.timingSource,'wrapper');
});

test('legacy settlement uses submission time and first terminal observation, not days of retry delay',async t=>{
 const f=await fixture(t,{launch:false});
 f.event('execution-state',{handle:f.handle,state:'stopped',nativeState:'failed'});
 f.state.db.prepare("UPDATE events SET created='2026-01-01T00:00:00.000Z' WHERE type='execution-submitting'").run();
 f.state.db.prepare("UPDATE events SET created='2026-01-01T00:00:02.500Z' WHERE type='execution-state'").run();
 assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),true);
 assert.equal(f.state.attempt(f.id).elapsed,12.5);
 assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),true);
 assert.equal(f.state.attempt(f.id).elapsed,12.5);
});

test('explicit setup exclusion remains zero while payload launch fallback is charged',async t=>{
 const f=await fixture(t);const result=await f.environment.observe(f.handle,{accountingExcluded:true});
 assert.equal(result.executionSeconds,0);assert.equal(result.accountingExcluded,true);assert.equal(f.control.downloads,0);
});

test('launch timestamp is persisted before dispatch, even if native handle publication is lost',async t=>{
 const root=await mkdtemp(join(tmpdir(),'launch-before-submit-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const project=join(root,'source');await mkdir(project);await writeFile(join(project,'main.py'),'pass');let payloadReceipt;
 const env=await createEnvironment({mode:'cpl',target:'fake',workspacePath:'/registered',remoteDirectory:'owned'},
 {instancePath:root,sdk:{createLocalBashOperations:()=>({})},event:async()=>{},exclude:async(_label,fn)=>fn(),client:async args=>{
  if(args[0]==='capabilities')return {state:'available',backend:'remote-agent'};
  if(args[0]==='upload'||args[0]==='preflight')return {state:'completed'};
  if(args[0]==='run'){
   const file=args[args.indexOf('--file')+1];if(file.endsWith('sync.sh'))return {handle:'remote:fake:job:sync'};
   payloadReceipt=JSON.parse(await readFile(join(file.slice(0,file.lastIndexOf('/')),'execution.json')));
   assert.ok(Number.isFinite(payloadReceipt.launch.wallMs));assert.ok(payloadReceipt.launch.monotonicNs);assert.ok(payloadReceipt.launch.boot);
   await writeFile(args[args.indexOf('--receipt-file')+1],JSON.stringify({handle:'remote:fake:job:lost'}));throw Error('reply lost');
  }
  if(args[0]==='observe')return args[1].endsWith(':sync')?{state:'completed',exit:0}:{state:'observation-unavailable'};
  assert.fail('Unexpected operation '+args[0]);
 }});
 await env.prepare({project:{source:project},attempt:{id:'a'}});
 const result=await env.execute('work',project);assert.equal(result.settled,false);assert.ok(result.executionSeconds>=0);
 assert.equal(result.handle,'remote:fake:job:lost');assert.ok(payloadReceipt);
});


test('same-host monotonic launch time survives restart and ignores wall clock jumps',async t=>{
 const f=await fixture(t);const boot=(await readFile('/proc/sys/kernel/random/boot_id','utf8')).trim();
 f.record.launch={wallMs:Date.now()-500_000,monotonicNs:String(process.hrtime.bigint()-2_000_000_000n),boot};
 await writeFile(join(f.dir,'execution.json'),JSON.stringify(f.record));
 t.mock.method(Date,'now',()=>2_000_000_000_000);
 const observed=await (await createEnvironment(f.options,f.context)).observe(f.handle);
 assert.ok(observed.executionSeconds>=2&&observed.executionSeconds<10);
});

test('missing both launch metadata and ledger timestamps remains unaccounted',async t=>{
 const f=await fixture(t,{launch:false});
 const proof=await f.environment.reconcileOperation({operation:'op',events:f.state.events(f.id)});
 assert.equal(proof.closed,false);assert.equal(proof.executionSeconds,undefined);
 assert.equal(f.state.attempt(f.id).elapsed,10);
});

test('failed launch settlement permits Phase 2 advancement and later timing cannot rewind charged budget',async t=>{
 const f=await fixture(t),queue=new RecoveryQueue(f.state);
 const config={mode:'phase1',phase_sequence:['phase1','phase2'],phase_targets:{phase1:1},curriculum:{phase1:['p'],phase2:[]}};
 f.event('failure-settlement',{closed:false});
 assert.equal(campaignProgress(f.state,config).phaseComplete,false);
 assert.equal(await reconcileFailure(f.state,f.id,f.environment),true);assert.equal(failureSettled(f.state,f.id),true);
 const elapsed=f.state.attempt(f.id).elapsed;assert.ok(elapsed>=14);
 queue.recordOriginal(f.id,{passed:false,findings:'Native launch failed',evidence:[{path:'failure.json',sha256:'hash'}]},1);
 for(const lesson of queue.lessons())queue.acknowledgeLesson(lesson.id);
 f.event('postprocess-completed',{});
 assert.equal(campaignProgress(f.state,config).nextMode,'phase2');
 f.control.receipt={completed:true,key:'key',executionSeconds:0.5};
 assert.equal(await reconcileFailure(f.state,f.id,f.environment),true);
 assert.equal(f.state.attempt(f.id).elapsed,elapsed);
 assert.equal(f.state.queryEvents(f.id,{types:['execution-reconciled']})[0].data.timingEstimated,true);
});


test('legacy terminal cutoff freezes accounting even when the current native observation is unknown',async t=>{
 const f=await fixture(t,{launch:false});f.control.state='observation-unavailable';
 f.event('execution-state',{handle:f.handle,state:'stopped',nativeState:'failed'});
 f.state.db.prepare("UPDATE events SET created='2026-01-01T00:00:00.000Z' WHERE type='execution-submitting'").run();
 f.state.db.prepare("UPDATE events SET created='2026-01-01T00:00:02.500Z' WHERE type='execution-state'").run();
 assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),false);
 assert.equal(f.state.attempt(f.id).elapsed,12.5);
 t.mock.method(Date,'now',()=>2_000_000_000_000);
 const restarted=await createEnvironment(f.options,f.context),observed=await restarted.observe(f.handle);
 assert.equal(observed.state,'unknown');assert.equal(observed.executionSeconds,2.5);
 assert.equal(JSON.parse(await readFile(join(f.dir,'execution.json'))).controllerTiming.terminal,true);
});

test('an unknown excluded probe does not prevent accounting for the subsequent payload',async t=>{
 const f=await fixture(t);f.control.state='observation-unavailable';
 const events=f.state.events(f.id,{timestamps:true});
 events.splice(1,0,{type:'execution-submitting',created:new Date().toISOString(),data:{operation:'op',submission:'probe',accountingExcluded:true}},
  {type:'execution-submitted',data:{operation:'op',submission:'probe',handle:'remote:fake:job:probe'}});
 const proof=await f.environment.reconcileOperation({operation:'op',events});
 assert.equal(proof.closed,false);assert.ok(proof.executionSeconds>=4);assert.equal(proof.timingEstimated,true);
});


test('lost native acknowledgement remains fenced while its persisted launch time advances budget',async t=>{
 const f=await fixture(t),receiptFile=join(f.dir,'submission.json');
 await writeFile(join(f.dir,'execution.json'),JSON.stringify({...f.record,handle:null,receiptFile}));
 f.state.db.prepare("DELETE FROM events WHERE type IN ('execution-submitted','execution-handle')").run();
 const entry=f.state.db.prepare("SELECT rowid AS row_id,data FROM events WHERE type='execution-submitting'").get();
 f.state.db.prepare('UPDATE events SET data=? WHERE rowid=?').run(JSON.stringify({...JSON.parse(entry.data),receiptFile}),entry.row_id);
 assert.equal(await operationsStopped(f.state,f.state.attempt(f.id),f.environment),false);
 assert.ok(f.state.attempt(f.id).elapsed>=14);assert.equal(f.control.downloads,0);
 assert.equal(f.state.queryEvents(f.id,{types:['execution-reconciled']}).length,0);
});
