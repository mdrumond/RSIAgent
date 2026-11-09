import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,rm,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import filesystem from 'node:fs/promises';
import {syncBuiltinESMExports} from 'node:module';
import * as sdk from '@earendil-works/pi-coding-agent';
import {createEnvironment} from '../../integrations/environments/index.mjs';

async function fixture(t,{brokenShell=false,event,limits}={}) {
  const root=await mkdtemp(join(tmpdir(),'local-closure-'));t.after(()=>rm(root,{recursive:true,force:true}));
  if(brokenShell)await writeFile(join(root,'missing-bash'),'#!/missing/interpreter\n',{mode:0o700});
  let handle;const context={instancePath:root,exclude:async(_label,fn)=>fn(),
    sdk:brokenShell?{...sdk,createLocalBashOperations:()=>sdk.createLocalBashOperations({shellPath:join(root,'missing-bash')})}:sdk,
    event:async(type,data)=>{if(type==='execution-handle')handle=data.handle;await event?.(type,data);}};
  const environment=await createEnvironment({mode:'local',localExecution:limits},context);
  return {root,environment,context,handle:()=>handle};
}

for(const limits of [undefined,{tasks:16,memory_bytes:128*1024**2}])test('actual Pi spawn failure after retained handle closes only a proven unspawned command '+(limits?'bounded':'legacy'),async t=>{
  const f=await fixture(t,{brokenShell:true,limits});
  const result=await f.environment.execute('printf never',f.root,{timeout:2});
  assert.equal(result.settled,true);assert.equal(result.outcome.reason,'local-execution-not-started');
  const receipt=JSON.parse(await readFile(join(f.root,'.executions',f.handle().slice(6),'execution.json')));
  assert.equal(receipt.failure.code,'ENOENT');assert.equal(receipt.prelaunch.closed,true);
  const observer=await createEnvironment({mode:'local'},f.context);
  assert.equal((await observer.observe(result.handle)).state,'stopped');
});

test('abort at handle publication proves no launch and retains the cancellation cause',async t=>{
  const abort=new AbortController();const f=await fixture(t,{event:type=>{if(type==='execution-handle')abort.abort();}});
  const result=await f.environment.execute('printf never',f.root,{signal:abort.signal,timeout:2});
  assert.equal(result.settled,true);assert.equal(result.outcome.reason,'local-execution-not-started');
  assert.equal((await f.environment.observe(result.handle)).state,'stopped');
});

test('legacy boot-only receipt stays unknown despite absent owner or child identity',async t=>{
  const f=await fixture(t),dir=join(f.root,'.executions/exec-legacy');await mkdir(dir);
  await writeFile(join(dir,'execution.json'),JSON.stringify({boot:(await readFile('/proc/sys/kernel/random/boot_id','utf8')).trim(),cwd:f.root}));
  assert.equal((await f.environment.observe('local:exec-legacy')).state,'unknown');
});

test('an SDK error without positive prelaunch proof stays fenced and retains its cause',async t=>{
  const f=await fixture(t);const original=sdk.createLocalBashOperations();
  const failing={...sdk,createLocalBashOperations:()=>({exec:async(_command,cwd,options)=>{
    await original.exec('printf started',cwd,options);throw Object.assign(new Error('post-launch transport failure'),{code:'EIO'});
  }})};
  const environment=await createEnvironment({mode:'local'},{...f.context,sdk:failing});
  await assert.rejects(environment.execute('printf payload',f.root,{timeout:2}),error=>
    error.cause?.code==='EIO'&&error.handle===f.handle());
  const observer=await createEnvironment({mode:'local'},f.context);
  assert.equal((await observer.observe(f.handle())).state,'unknown');
  const receipt=JSON.parse(await readFile(join(f.root,'.executions',f.handle().slice(6),'execution.json')));
  assert.equal(receipt.failure.code,'EIO');assert.equal(receipt.prelaunch,undefined);
});


test('bounded setup filesystem failure is positively unspawned and preserves its errno',async t=>{
  const f=await fixture(t),original=filesystem.mkdtemp;
  const mock=t.mock.method(filesystem,'mkdtemp',async(prefix,...args)=>{
    if(prefix.includes('rsi-pi-command-'))throw Object.assign(new Error('temporary directory unavailable'),{code:'ENOSPC'});
    return original(prefix,...args);
  });syncBuiltinESMExports();t.after(()=>{mock.mock.restore();syncBuiltinESMExports();});
  const environment=await createEnvironment({mode:'local',localExecution:{tasks:16,memory_bytes:128*1024**2}},f.context);
  const result=await environment.execute('printf never',f.root,{timeout:2});
  assert.equal(result.settled,true);assert.equal(result.outcome.reason,'local-execution-not-started');
  const receipt=JSON.parse(await readFile(join(f.root,'.executions',result.handle.slice(6),'execution.json')));
  assert.equal(receipt.failure.code,'ENOSPC');assert.equal(receipt.prelaunch.stage,'containment-directory');
});
