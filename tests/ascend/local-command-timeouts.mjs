import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm,access} from 'node:fs/promises';
import {join} from 'node:path';
import {constants} from 'node:fs';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {processIdentity} from '../../controller/lifecycle.mjs';
import {createEnvironment} from '../../integrations/environments/index.mjs';
import * as sdk from '@earendil-works/pi-coding-agent';
import {createLocalExecutionOperations} from '../../integrations/environments/local-bash.mjs';
import {managedBootstrap,bootstrapSettled} from '../../cli/evaluation-bootstrap.mjs';
import {evaluateRound} from '../../controller/evaluation-round.mjs';

async function available(t) {
  let output='';
  try {
    const result=await sdk.createLocalBashOperations().exec('systemctl --user show --property=ControlGroup --value',process.cwd(),
      {timeout:3,onData:chunk=>output+=chunk});
    const root='/sys/fs/cgroup'+output.trim();
    if(result.exitCode!==0||!output.startsWith('/'))throw Error('No user manager');
    await access(root,constants.W_OK);
    const controllers=await readFile(root+'/cgroup.controllers','utf8');
    if(!controllers.split(/\s+/).includes('pids')||!controllers.split(/\s+/).includes('memory'))throw Error('No resource delegation');
    return true;
  }catch{t.skip('Native containment requires a delegated user systemd cgroup with pids and memory controllers');return false;}
}

const month=30*24*60*60;
async function directory(t){const root=await mkdtemp(join(tmpdir(),'local-command-timeouts-'));t.after(()=>rm(root,{recursive:true,force:true}));return root;}

test('a month-long comparison bootstraps real Pi commands without shrinking the retained allocation',async t=>{
  const root=await directory(t),pending={evaluationRoot:join(root,'round'),sourceInstanceRoot:root,
    projectIds:['p'],runs:1,active:'baseline',candidate:'candidate',evaluationSeconds:month};
  let milliseconds=0;const remaining=[];
  const result=await evaluateRound({pending,config:{},now:()=>milliseconds,wallNow:()=>milliseconds,
    evaluate:async request=>{
      remaining.push(request.remainingSeconds);const counter=join(request.instanceRoot,'executions');
      await managedBootstrap(request.instanceRoot,{binding:{project:'p'},remainingSeconds:request.remainingSeconds,signal:request.signal},async initialization=>{
        await mkdir(request.instanceRoot,{recursive:true});
        await initialization.setup('fixture',process.execPath,['-e',`require('fs').appendFileSync(${JSON.stringify(counter)},'x')`],root,{});
      });
      assert.equal(await readFile(counter,'utf8'),'x');assert.equal(await bootstrapSettled(request.instanceRoot),true);
      milliseconds+=1000;return {correct:true,score:1};
    }});
  assert.equal(result.status,'completed');assert.equal(result.allocatedSeconds,month);
  assert.equal(result.elapsedSeconds,2);assert.deepEqual(remaining,[month,month-1]);
  assert.equal(result.samples.filter(s=>s.result).length,2);
});

for(const limits of [undefined,{tasks:16,memory_bytes:128*1024**2}])test('real Pi accepts a multi-day command allocation '+(limits?'with OS containment':'without OS containment'),async t=>{
  if(limits && !await available(t))return;
  const root=await directory(t),operations=createLocalExecutionOperations(sdk,limits);let text='';
  const result=await operations.exec('printf accepted',root,{timeout:month,onData:chunk=>text+=chunk});
  assert.equal(result.exitCode,0);assert.equal(text,'accepted');
});

test('SDK request clipping preserves shorter limits, cancellation and caller options',async()=>{
  const calls=[],backend={exec:async(command,cwd,options)=>{calls.push({command,cwd,options});return {exitCode:0};}};
  const operations=createLocalExecutionOperations({createLocalBashOperations:()=>backend});
  const controller=new AbortController(),onData=()=>{},env={FIXTURE:'value'};
  const options={timeout:month,signal:controller.signal,onData,env};
  await operations.exec('fixture','cwd',options);
  assert.equal(options.timeout,month);assert.ok(calls[0].options.timeout<month);
  assert.ok(calls[0].options.timeout>24*24*60*60);
  for(const key of ['signal','onData','env'])assert.equal(calls[0].options[key],options[key]);
  await operations.exec('short','cwd',{timeout:.125,onData});assert.equal(calls[1].options.timeout,.125);
  await operations.exec('unbounded','cwd',{onData});assert.equal(calls[2].options.timeout,undefined);
});

test('invalid command timeout requests retain Pi validation',async t=>{
  const root=await directory(t),operations=createLocalExecutionOperations(sdk);
  for(const timeout of [0,-1,NaN,Infinity])await assert.rejects(operations.exec('printf never',root,{timeout,onData:()=>{}}),/Invalid timeout/);
});

test('a short real Pi timeout still stops a running command', {timeout:5000},async t=>{
  const root=await directory(t),operations=createLocalExecutionOperations(sdk),started=performance.now();
  await assert.rejects(operations.exec('sleep 60',root,{timeout:.05,onData:()=>{}}),/timeout:0.05/);
  assert.ok(performance.now()-started<3000);
});

test('a long allocation still aborts its actual Pi command immediately', {timeout:5000},async t=>{
  const root=await directory(t),operations=createLocalExecutionOperations(sdk),controller=new AbortController();let started=false;
  await assert.rejects(operations.exec('printf ready; sleep 60',root,{timeout:month,signal:controller.signal,
    onData:()=>{started=true;controller.abort();}}),/aborted/i);
  assert.equal(started,true);
});

async function legacyBootstrap(t){
  const parent=await directory(t),root=join(parent,'arm'),binding={project:'frozen'},handle='local:exec-legacy';
  const setup=root+'-setup',dir=join(setup,'.executions',handle.slice(6));await mkdir(dir,{recursive:true});
  let error,ran=false;
  try{await sdk.createLocalBashOperations().exec('printf never',parent,{timeout:month,onData:()=>{ran=true;}});}
  catch(caught){error=caught;}
  assert.ok(error);assert.equal(ran,false);
  const me=await processIdentity(),owner={...me,start:String(BigInt(me.start)-1n)};
  const receipt={boot:me.boot,cwd:parent,failure:{message:error.message,code:error.code??null,syscall:error.syscall??null},
    outcome:{kind:'infrastructure',reason:'local-execution-interrupted'}};
  const path=join(dir,'execution.json');await writeFile(path,JSON.stringify(receipt));
  const checkpoint={format:1,signature:createHash('sha256').update(JSON.stringify(binding)).digest('hex'),
    owner,status:'preparing',handles:[handle],completed:[]};
  await writeFile(root+'.bootstrap.json',JSON.stringify(checkpoint));
  return {parent,root,binding,handle,setup,dir,path,receipt,checkpoint};
}

test('canonical bootstrap resumes an actual persisted Pi timeout rejection without inventing command success',async t=>{
  const f=await legacyBootstrap(t),before=await readFile(f.path,'utf8');
  const ordinary=await createEnvironment({mode:'local'},{instancePath:f.setup,event:async()=>{},exclude:async(_l,fn)=>fn()});
  assert.equal((await ordinary.observe(f.handle)).state,'unknown');
  assert.equal(await bootstrapSettled(f.root),true);
  const recovered=await createEnvironment({mode:'local'},{instancePath:f.setup,recoverPiTimeoutRejection:true,event:async()=>{},exclude:async(_l,fn)=>fn()});
  const proof=await recovered.observe(f.handle);assert.equal(proof.state,'stopped');assert.equal(proof.exitCode,undefined);
  assert.deepEqual(proof.outcome,f.receipt.outcome);assert.equal(await readFile(f.path,'utf8'),before);
  const counter=join(f.parent,'resumed');
  await managedBootstrap(f.root,{binding:f.binding,remainingSeconds:month},async initialization=>{
    assert.equal(initialization.resume,true);
    await initialization.setup('install',process.execPath,['-e',`require('fs').appendFileSync(${JSON.stringify(counter)},'x')`],f.parent,{});
  });
  assert.equal(await readFile(counter,'utf8'),'x');
  const after=JSON.parse(await readFile(f.root+'.bootstrap.json','utf8'));
  assert.equal(after.signature,f.checkpoint.signature);assert.equal(after.status,'ready');assert.equal(after.handles[0],f.handle);
});

test('legacy bootstrap recovery never bypasses live bootstrap ownership',async t=>{
  const f=await legacyBootstrap(t);await writeFile(f.root+'.bootstrap.json',JSON.stringify({...f.checkpoint,owner:await processIdentity()}));
  assert.equal(await bootstrapSettled(f.root),false);
});

for(const change of ['unknown failure','wrong boot','retained PID','scope'])test('legacy timeout recovery fences '+change,async t=>{
  const f=await legacyBootstrap(t),receipt=structuredClone(f.receipt);
  if(change==='unknown failure')receipt.failure={message:'post-launch failure',code:'EIO',syscall:null};
  if(change==='wrong boot')receipt.boot='older-boot';
  if(change==='retained PID')receipt.pid=2147483647;
  if(change==='scope')receipt.containment={unit:'rsi-command-unverified.scope',closed:false};
  await writeFile(f.path,JSON.stringify(receipt));assert.equal(await bootstrapSettled(f.root),false);
});

test('a custom SDK observer cannot assert canonical Pi prelaunch recovery',async t=>{
  const f=await legacyBootstrap(t);let calls=0;
  const custom={createLocalBashOperations:()=>({exec:async()=>{calls++;return {exitCode:0};}})};
  const observer=await createEnvironment({mode:'local'},{instancePath:f.setup,recoverPiTimeoutRejection:true,sdk:custom,event:async()=>{},exclude:async(_l,fn)=>fn()});
  assert.equal((await observer.observe(f.handle)).state,'unknown');assert.equal(calls,0);
});

test('an actual live command identity fences even a retained timeout rejection',{timeout:5000},async t=>{
  const f=await legacyBootstrap(t),abort=new AbortController();t.after(()=>abort.abort());let ready;
  const started=new Promise(resolve=>{ready=resolve;});
  const command=`cat /proc/$$/stat > '${join(f.dir,'identity')}'; printf ready; sleep 60`;
  const completion=sdk.createLocalBashOperations().exec(command,f.parent,{signal:abort.signal,timeout:60,onData:()=>ready()})
    .then(()=>null,error=>error);
  try{await started;assert.equal(await bootstrapSettled(f.root),false);}
  finally{abort.abort();await completion;}
});
