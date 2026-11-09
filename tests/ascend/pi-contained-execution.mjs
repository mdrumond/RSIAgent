import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,access} from 'node:fs/promises';
import {constants} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import * as sdk from '@earendil-works/pi-coding-agent';
import {createLocalExecutionOperations} from '../../integrations/environments/local-bash.mjs';
import {createEnvironment,createEnvironmentTools} from '../../integrations/environments/index.mjs';

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
async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'pi-contained-test-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const limits={tasks:16,memory_bytes:128*1024**2};
  const operations=createLocalExecutionOperations(sdk,limits);
  let text='';
  assert.equal((await operations.exec('IFS=: read -r a b g < /proc/self/cgroup; cat "/sys/fs/cgroup$g/pids.max" "/sys/fs/cgroup$g/memory.max"',root,
    {timeout:3,onData:chunk=>text+=chunk})).exitCode,0);
  assert.deepEqual(text.trim().split('\n'),['16','134217728']);
  return {root,limits,operations};
}

test('legacy local execution reuses the Pi backend; malformed limits do not dispatch',async()=>{
  let calls=0;const operations={exec:async(command,cwd)=>{assert.equal(command,'printf delegated');assert.equal(cwd,'fixture');calls++;return {exitCode:0};}};
  assert.equal((await createLocalExecutionOperations({createLocalBashOperations:()=>operations}).exec('printf delegated','fixture',{})).exitCode,0);
  assert.equal(calls,1);
  assert.throws(()=>createLocalExecutionOperations({createLocalBashOperations:()=>operations},{tasks:0}),/limits/);
});

test('recursive generated construction command is contained and becomes a normal Pi tool error',{timeout:15000},async t=>{
  if(!await available(t))return;
  const f=await fixture(t),script=join(f.root,'recursive.sh');
  await writeFile(script,'#!/bin/bash\nbash "$0" & wait\n');
  const tool=sdk.createBashTool(f.root,{operations:f.operations});
  const result=await tool.execute('recursive',{command:`bash '${script}'`,timeout:5});
  assert.equal(result.isError,true);assert.equal(result.structuredContent.exit_code,125);
  const repaired=await tool.execute('repair',{command:'printf repaired',timeout:3});
  assert.equal(repaired.isError,undefined);assert.equal(repaired.structuredContent.exit_code,0);
  assert.equal(repaired.structuredContent.output,'repaired');
});

test('normal exit and abort clean detached children through the owned OS scope',{timeout:15000},async t=>{
  if(!await available(t))return;
  const f=await fixture(t),pidfile=join(f.root,'child.pid');
  await f.operations.exec(`setsid sleep 60 </dev/null >/dev/null 2>&1 & echo $! > '${pidfile}'`,f.root,{timeout:3,onData:()=>{}});
  const pid=Number(await readFile(pidfile,'utf8'));
  try{const stat=await readFile('/proc/'+pid+'/stat','utf8');assert.ok(['Z','X'].includes(stat.slice(stat.lastIndexOf(')')+2).split(' ')[0]));}
  catch(error){if(error.code!=='ENOENT'&&error.code!=='ESRCH')throw error;}
  const abort=new AbortController();setTimeout(()=>abort.abort(),100);
  await assert.rejects(f.operations.exec('sleep 60',f.root,{signal:abort.signal,timeout:3,onData:()=>{}}),/aborted/);
});

test('CPU construction execution retains scoped closure across a new observer',{timeout:15000},async t=>{
  if(!await available(t))return;
  const f=await fixture(t),events=[],context={instancePath:f.root,event:async(type,data)=>events.push({type,...data}),exclude:async(_label,fn)=>fn()};
  const env=await createEnvironment({mode:'local',localExecution:f.limits},context);
  const result=await env.execute('printf construction',f.root,{timeout:3});
  assert.equal(result.settled,true);assert.equal(result.exitCode,0);
  const observer=await createEnvironment({mode:'local',localExecution:f.limits},context);
  assert.deepEqual(await observer.observe(result.handle),{state:'stopped',exitCode:0});
  const recursive=join(f.root,'construction.sh');await writeFile(recursive,'#!/bin/bash\nbash "$0" & wait\n');
  const rejected=await env.execute(`bash '${recursive}'`,f.root,{timeout:3});
  assert.equal(rejected.settled,true);assert.equal(rejected.exitCode,125);
  assert.deepEqual(rejected.outcome,{kind:'infrastructure',reason:'process-limit'});
  assert.deepEqual((await observer.observe(rejected.handle)).outcome,rejected.outcome);
  assert.equal((await env.execute('printf repaired',f.root,{timeout:3})).exitCode,0);
  const tools=await createEnvironmentTools({...env,shellOnTarget:false},f.root);
  const local=tools.find(tool=>tool.name==='bash');
  assert.equal((await local.execute('local',{command:'printf local',timeout:3})).structuredContent.exit_code,0);
});

test('missing containment cannot execute an unrestricted fallback',{timeout:15000},async t=>{
  if(!await available(t))return;
  const f=await fixture(t),marker=join(f.root,'escaped');
  const result=await f.operations.exec(`touch '${marker}'`,f.root,{timeout:3,env:{...process.env,PATH:'/nonexistent'},onData:()=>{}});
  assert.notEqual(result.exitCode,0);
  await assert.rejects(access(marker),error=>error.code==='ENOENT');
});

test('a launcher that applies the wrong kernel limit cannot execute the payload',{timeout:15000},async t=>{
  if(!await available(t))return;
  const f=await fixture(t),bin=join(f.root,'bin'),marker=join(f.root,'escaped');await mkdir(bin);
  await writeFile(join(bin,'systemd-run'),'#!/bin/bash\nargs=()\nfor arg; do [[ "$arg" == --property=TasksMax=16 ]] && arg=--property=TasksMax=32; args+=("$arg"); done\nexec /usr/bin/systemd-run "${args[@]}"\n',{mode:0o700});
  const result=await f.operations.exec(`touch '${marker}'`,f.root,{timeout:3,env:{...process.env,PATH:bin+':'+process.env.PATH},onData:()=>{}});
  assert.equal(result.exitCode,126);await assert.rejects(access(marker),error=>error.code==='ENOENT');
});

test('OS memory violation is returned as command failure without stopping the session',{timeout:15000},async t=>{
  if(!await available(t))return;
  const f=await fixture(t),operations=createLocalExecutionOperations(sdk,{tasks:16,memory_bytes:32*1024**2});
  const tool=sdk.createBashTool(f.root,{operations});
  const failed=await tool.execute('memory',{command:"python3 -c 'x=bytearray(128*1024**2)'",timeout:3});
  assert.equal(failed.isError,true);assert.equal(failed.structuredContent.exit_code,125);
  assert.equal((await tool.execute('continue',{command:'printf continued',timeout:3})).structuredContent.output,'continued');
});

test('a restarted observer cancels the retained command scope rather than replaying it',{timeout:15000},async t=>{
  if(!await available(t))return;
  const f=await fixture(t);let handle;
  const context={instancePath:f.root,event:async(type,data)=>{if(type==='execution-handle')handle=data.handle;},exclude:async(_label,fn)=>fn()};
  const env=await createEnvironment({mode:'local',localExecution:f.limits},context);
  const pending=env.execute('sleep 60',f.root,{timeout:5});
  const observer=await createEnvironment({mode:'local',localExecution:f.limits},context);
  const deadline=Date.now()+3000;
  while(!handle||(await observer.observe(handle)).state!=='running') {
    assert.ok(Date.now()<deadline);await new Promise(resolve=>setTimeout(resolve,10));
  }
  assert.equal((await observer.stop(handle)).state,'stopped');
  const result=await pending;assert.equal(result.handle,handle);assert.equal(result.settled,true);assert.notEqual(result.exitCode,0);
});

test('Pi delivers a resource error to its agent and accepts the repaired next tool call',{timeout:15000},async t=>{
  if(!await available(t))return;
  const f=await fixture(t),script=join(f.root,'recursive.sh');await writeFile(script,'#!/bin/bash\nbash "$0" & wait\n');
  const {createServer}=await import('node:http'),{once}=await import('node:events');
  const {InMemoryCredentialStore}=await import('@earendil-works/pi-ai');
  const errors=[],requests=[];
  const provider=createServer(async(req,res)=>{
    let raw='';for await(const chunk of req)raw+=chunk;
    const body=JSON.parse(raw);requests.push(body);const results=body.messages.filter(message=>message.role==='tool');
    const done=results.length>=2;
    const delta=done?{role:'assistant',content:'done'}:{role:'assistant',tool_calls:[{index:0,id:'step-'+results.length,type:'function',
      function:{name:'bash',arguments:JSON.stringify({command:results.length?"printf repaired > repaired.txt":`bash '${script}'`,timeout:3})}}]};
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    for(const [value,finish] of [[delta,null],[{},done?'stop':'tool_calls']])res.write('data: '+JSON.stringify({
      id:'fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:value,finish_reason:finish}]})+'\n\n');
    res.end('data: [DONE]\n\n');
  });
  provider.listen(0,'127.0.0.1');await once(provider,'listening');t.after(()=>{provider.closeAllConnections();provider.close();});
  const runtime=await sdk.ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
  runtime.registerProvider('containment-fixture',{baseUrl:`http://127.0.0.1:${provider.address().port}/v1`,models:[{
    id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,
    cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});
  await runtime.setRuntimeApiKey('containment-fixture','fixture');
  const settingsManager=sdk.SettingsManager.inMemory({compaction:{enabled:false},retry:{enabled:false}});
  const loader=new sdk.DefaultResourceLoader({cwd:f.root,agentDir:f.root,settingsManager,noExtensions:true,noSkills:true,noContextFiles:true,
    extensionFactories:[pi=>pi.registerTool(sdk.createBashToolDefinition(f.root,{operations:f.operations}))]});
  await loader.reload();
  const {session}=await sdk.createAgentSession({cwd:f.root,agentDir:f.root,settingsManager,resourceLoader:loader,
    sessionManager:sdk.SessionManager.inMemory(),modelRuntime:runtime,model:runtime.getModel('containment-fixture','scripted')});
  t.after(()=>session.dispose());session.subscribe(event=>{if(event.type==='tool_execution_end')errors.push(event.isError);});
  await session.bindExtensions({});await session.prompt('Complete the command and repair execution errors.');
  assert.deepEqual(errors,[true,false]);assert.equal(await readFile(join(f.root,'repaired.txt'),'utf8'),'repaired');
  assert.equal(requests.length,3);assert.equal(requests[1].messages.filter(message=>message.role==='tool').length,1);
});
