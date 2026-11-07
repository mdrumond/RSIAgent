import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {mkdtemp,mkdir,writeFile,readFile,rm,readdir} from 'node:fs/promises';
import {join,resolve,relative} from 'node:path';
import {tmpdir} from 'node:os';
import {ModelRuntime} from '@earendil-works/pi-coding-agent';
import {InMemoryCredentialStore} from '@earendil-works/pi-ai';
import {State} from '../../controller/state.mjs';
import {CampaignStorage} from '../../controller/storage.mjs';
import {runAttempt} from '../../controller/pi.mjs';
import {createEnvironment} from '../../integrations/environments/ascend.mjs';

// Real Pi tools, Ascend event forwarding and campaign artifact registration;
// only provider responses and the native CPL wire are CPU fixtures.
for(const product of ['a3','a5'])test(`${product} lease cleanup output remains campaign-owned through preflight advance`,{timeout:30000},async t=>{
  const root=await mkdtemp(join(tmpdir(),'control-artifacts-')),campaign=join(root,'campaign'),leases=join(root,'global-leases');
  await mkdir(campaign);const source=join(campaign,'source');await mkdir(source);
  await writeFile(join(source,'answer.py'),'answer=42\n');
  const instructionFile=join(root,'instructions.md');await writeFile(instructionFile,'Use target commands.');
  const state=new State(join(campaign,'state.sqlite')),storage=new CampaignStorage(state,campaign);
  t.after(async()=>{state.close();await rm(root,{recursive:true,force:true});});
  await storage.initialize({id:'control-output-'+product});
  state.add({id:'p',goal:'Check target admission before constructing task',source,actor_seconds:30,verifier:{files:[],command:'true'}});
  const attempt=state.admit('p',{}),attemptPath=join(campaign,'attempts',attempt.id);
  const skillMap={};for(const stage of ['global-kdb','preflight']){
    skillMap[stage]=join(root,stage+'.md');await writeFile(skillMap[stage],'Inspect target and advance.');
  }
  const calls=[['rsi_skill',{}],['rsi_target',{command:'true',timeout:10}],['rsi_advance',{}]],responses=[];
  const server=createServer(async(req,res)=>{
    let raw='';for await(const chunk of req)raw+=chunk;const body=JSON.parse(raw);
    responses.push(...body.messages.filter(m=>m.role==='tool'&&m.tool_call_id==='call-2'));
    const index=body.messages.reduce((n,m)=>n+(m.tool_calls?.length??0),0);
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason)=>res.write('data: '+JSON.stringify({id:'control-output',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    if(index<calls.length){const [name,args]=calls[index];frame({role:'assistant',tool_calls:[{index:0,id:'call-'+index,type:'function',function:{name,arguments:JSON.stringify(args)}}]});frame({},'tool_calls');}
    else{frame({role:'assistant',content:'Preflight target check complete.'});frame({},'stop');}
    res.end('data: [DONE]\n\n');
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');
  t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));});
  const runtime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
  runtime.registerProvider('control-fixture',{baseUrl:`http://127.0.0.1:${server.address().port}/v1`,models:[{id:'scripted',name:'Scripted',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});
  await runtime.setRuntimeApiKey('control-fixture','fixture-only');
  const admissions=new Map(),jobs=new Map();let sequence=0;
  const client=async args=>{
    if(args[0]==='npu')return {state:'completed',profiles:[{target:'fixture',pool:root,devices:[{id:0,enabled:true}]}]};
    if(args[0]==='capabilities')return {state:'available'};
    if(['preflight','upload'].includes(args[0]))return {state:'completed'};
    if(args[0]==='download'){
      await writeFile(args[3],admissions.get(args[2])??[...admissions].find(([key])=>key.endsWith('/'+args[2]))?.[1]??(args[2].endsWith('/stdout')?'native stdout\n':'native stderr\n'));
      return {state:'completed'};
    }
    if(args[0]==='run'){
      const handle='remote:fixture:job:'+ ++sequence,file=args[args.indexOf('--file')+1];let content='';
      if(file.endsWith('probe.sh'))content=JSON.stringify({observed_at:Date.now()/1000,devices:[{physical_device:0,admitted:true,measurements:{idle:true}}]});
      if(file.endsWith('command.sh')){
        const script=await readFile(file,'utf8'),outer=JSON.parse(script.slice(script.lastIndexOf(" '{")+2,-2).replaceAll("'\\''","'"));
        const inner=JSON.parse(outer.command.slice(outer.command.lastIndexOf(" '{")+2,-1).replaceAll("'\\''","'"));
        admissions.set(outer.env.RSI_LEASE_RECEIPT,JSON.stringify({token:inner.lease.token,decision:inner.source?'admitted':'reconciled'}));
      }
      jobs.set(handle,{content});
      if(args.includes('--receipt-file'))await writeFile(args[args.indexOf('--receipt-file')+1],JSON.stringify({handle}));
      return {state:'queued',handle};
    }
    if(args[0]==='observe')return {state:'completed',exit:0};
    if(args[0]==='logs')return {state:'completed',content:jobs.get(args[1]).content};
    throw new Error('Unexpected native request '+args[0]);
  };
  const result=await runAttempt({state,attempt,project:state.project('p').spec,attemptPath,skillMap,
    knowledge:{search:async()=>({content:[],evidence:[]})},modelRuntime:runtime,model:runtime.getModel('control-fixture','scripted'),
    environmentFactory:context=>createEnvironment({mode:'cpl',target:'fixture',runtime:'registered',instructionFile,
      workspacePath:'/registry/workspace',remoteDirectory:'owned',pollMs:0,probeSeconds:10,health:{product}},
      {...context,leaseDirectory:leases,client})});
  assert.equal(result.accepted,false,'fixture stops after preflight without completing the project');
  assert.ok(responses.some(m=>JSON.parse(m.content).stage==='task-construction'));
  const events=state.events(attempt.id),outputs=events.filter(e=>e.type==='execution-output'&&e.data.purpose==='device-lease-control');
  assert.equal(outputs.length,1);assert.equal(outputs[0].data.accountingExcluded,true);
  assert.ok(events.some(e=>e.type==='stage-completed'&&e.data.stage==='preflight'));
  assert.equal(events.some(e=>e.type==='attempt-error'),false);
  const rows=state.db.prepare("SELECT * FROM storage_artifacts WHERE attempt=? AND kind IN ('execution-stdout','execution-stderr')").all(attempt.id);
  assert.equal(rows.length,4,'payload and control streams are registered');
  // Closing the global reservation archives its native journal, but campaign
  // copies must remain independently readable and sealable afterward.
  const [closed]=await readdir(join(leases,'.closed'));
  const journal=join(leases,'.closed',closed,'control','.executions');
  const nativeStreams=[];for(const execution of await readdir(journal)){
    try{nativeStreams.push(await readFile(join(journal,execution,'stdout'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
  }
  assert.ok(nativeStreams.includes('native stdout\n'),'global cleanup journal remains durable');
  for(const stream of ['stdout','stderr']){
    const path=outputs[0].data[stream];assert.ok(!relative(attemptPath,path).startsWith('..'));
    const row=rows.find(r=>resolve(campaign,r.path)===path);assert.ok(row);assert.equal(row.external,0);
    const sealed=await storage.seal(row.id);assert.equal(sealed.status,'closed');assert.equal(sealed.bytes,14);
    assert.equal(await readFile(path,'utf8'),'native '+stream+'\n');
  }
});
