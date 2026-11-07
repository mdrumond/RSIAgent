import test from 'node:test';
import assert from 'node:assert/strict';
import {createEnvironmentTools} from '../../integrations/environments/index.mjs';

// Exercise Pi's real Bash tool: it retains timeout, streaming and error handling
// while both public shell names delegate admission to the environment.
for(const name of ['bash','rsi_target'])test(`${name} uses device-host execution admission`,async()=>{
  const calls=[],abort=new AbortController();
  const tools=await createEnvironmentTools({shellOnTarget:true,execute:async(command,cwd,options)=>{
    calls.push({command,cwd,options});
    options.onData(Buffer.from('device result\n'));
    return {settled:true,exitCode:0};
  }},process.cwd());
  const tool=tools.find(tool=>tool.name===name);
  const response=await tool.execute('call',{command:'python3 program.py',timeout:5},abort.signal);
  assert.equal(calls.length,1);
  assert.equal(calls[0].command,'python3 program.py');
  assert.equal(calls[0].cwd,process.cwd());
  assert.equal(calls[0].options.timeout,5);
  assert.equal(response.content[0].text.trim(),'device result');
});

test('remote target leaves controller-host Bash available',async()=>{
  const tools=await createEnvironmentTools({execute:()=>assert.fail('not executed')},process.cwd());
  assert.deepEqual(tools.map(tool=>tool.name),['rsi_target','bash']);
});

test('both shell names surface unresolved execution without replay',async()=>{
  let submissions=0;
  const tools=await createEnvironmentTools({shellOnTarget:true,execute:async()=>{
    submissions++;return {settled:false,handle:'remote:fixture:job:unknown'};
  }},process.cwd());
  for(const tool of tools){
    await assert.rejects(tool.execute('call',{command:'python3 program.py'}),/Execution remains unresolved/);
  }
  assert.equal(submissions,2);
});

test('Pi extension overrides built-in Bash and preserves workspace edits',{timeout:15000},async t=>{
  const {mkdtemp,readFile,rm}=await import('node:fs/promises');
  const {tmpdir}=await import('node:os');
  const {join}=await import('node:path');
  const {createServer}=await import('node:http');
  const {once}=await import('node:events');
  const sdk=await import('@earendil-works/pi-coding-agent');
  const {InMemoryCredentialStore}=await import('@earendil-works/pi-ai');
  const {createEnvironment}=await import('../../integrations/environments/index.mjs');
  const root=await mkdtemp(join(tmpdir(),'pi-shell-admission-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  let admissions=0;
  const environment=await createEnvironment({mode:'local'},{instancePath:root,event:async()=>{},exclude:async(_label,fn)=>fn()});
  const tools=await createEnvironmentTools({shellOnTarget:true,execute:async(...args)=>{
    admissions++;return environment.execute(...args);
  }},root);
  const provider=createServer(async(req,res)=>{
    let raw='';for await(const chunk of req)raw+=chunk;
    const body=JSON.parse(raw),completed=body.messages.some(message=>message.role==='tool');
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const delta=completed?{role:'assistant',content:'done'}:{role:'assistant',tool_calls:[{
      index:0,id:'shell',type:'function',function:{name:'bash',arguments:JSON.stringify({command:'echo admitted > result.txt'})}}]};
    for(const [value,finish] of [[delta,null],[{},completed?'stop':'tool_calls']])res.write('data: '+JSON.stringify({
      id:'fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:value,finish_reason:finish}]})+'\n\n');
    res.end('data: [DONE]\n\n');
  });
  provider.listen(0,'127.0.0.1');await once(provider,'listening');
  t.after(()=>{provider.closeAllConnections();provider.close();});
  const runtime=await sdk.ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
  runtime.registerProvider('shell-fixture',{baseUrl:`http://127.0.0.1:${provider.address().port}/v1`,models:[{
    id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,
    cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});
  await runtime.setRuntimeApiKey('shell-fixture','fixture');
  const settingsManager=sdk.SettingsManager.inMemory({compaction:{enabled:false},retry:{enabled:false}});
  const loader=new sdk.DefaultResourceLoader({cwd:root,agentDir:root,settingsManager,noExtensions:true,noSkills:true,noContextFiles:true,
    extensionFactories:[pi=>{for(const tool of tools)pi.registerTool(tool);}]});
  await loader.reload();
  const {session}=await sdk.createAgentSession({cwd:root,agentDir:root,settingsManager,resourceLoader:loader,
    sessionManager:sdk.SessionManager.inMemory(),modelRuntime:runtime,model:runtime.getModel('shell-fixture','scripted')});
  t.after(()=>session.dispose());
  await session.bindExtensions({});
  await session.prompt('Write result.txt using bash.');
  assert.equal(admissions,1);
  assert.equal(await readFile(join(root,'result.txt'),'utf8'),'admitted\n');
});
