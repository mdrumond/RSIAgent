import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {mkdtemp,mkdir,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import {createAgentSession,DefaultResourceLoader,ModelRuntime,SettingsManager,SessionManager} from '@earendil-works/pi-coding-agent';
import {InMemoryCredentialStore} from '@earendil-works/pi-ai';
import {createKdbBridge} from '../../integrations/kdb-extension.mjs';

async function fixture(t,{listDelay=0,missing=false,unavailable=false,extension=true}={}) {
  const root=await mkdtemp(join(tmpdir(),'kdb-readiness-'));const agentDir=join(root,'agent');await mkdir(agentDir);
  const events=[],requests=[],queries=[],timers=new Set();let listStarted;
  const listing=new Promise(resolve=>{listStarted=resolve;});
  const server=createServer(async(req,res)=>{
    let raw='';for await(const chunk of req)raw+=chunk;
    if(req.url==='/provider/v1/chat/completions') {
      const body=JSON.parse(raw);requests.push(body);
      assert.ok(body.tools.some(t=>t.function.name==='mcp__kdb__search'));
      const results=body.messages.filter(m=>m.role==='tool');
      const index=results.length;
      res.writeHead(200,{'Content-Type':'text/event-stream'});
      const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'ready',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
      if(index<2){frame({role:'assistant',tool_calls:[{index:0,id:'q'+index,type:'function',function:{name:'rsi_knowledge',arguments:JSON.stringify({query:'evidence',kind:index?'negative':'good'})}}]});frame({},'tool_calls');}
      else{frame({role:'assistant',content:'Both queries complete.'});frame({},'stop');}
      res.end('data: [DONE]\n\n');return;
    }
    if(req.method!=='POST'){res.writeHead(405);res.end();return;}
    if(unavailable){res.writeHead(400);res.end('Readiness fixture unavailable');return;}
    const message=JSON.parse(raw);
    if(message.id===undefined){res.writeHead(202);res.end();return;}
    let result={};
    if(message.method==='initialize')result={protocolVersion:message.params.protocolVersion,capabilities:{tools:{}},serverInfo:{name:'fixture',version:'1'}};
    if(message.method==='tools/list') {
      listStarted();
      if(listDelay)await new Promise(resolve=>{const timer=setTimeout(()=>{timers.delete(timer);resolve();},listDelay);timers.add(timer);});
      result={tools:missing?[]:[{name:'search',description:'Search facts',inputSchema:{type:'object',properties:{query:{type:'string'},kind:{type:'string'},limit:{type:'number'}}}}]};
    }
    if(message.method==='tools/call') {
      queries.push(message.params.arguments.kind);
      result={content:[{type:'text',text:'indexed fact'}],structuredContent:{result:[{text:'indexed fact',collection:'fixture',chunk_id:'hit',path:'fact.py',start_line:1,end_line:1,source_sha256:'a'.repeat(64)}]}};
    }
    if(!res.destroyed){res.writeHead(200,{'Content-Type':'application/json'});res.end(JSON.stringify({jsonrpc:'2.0',id:message.id,result}));}
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');const base=`http://127.0.0.1:${server.address().port}`;
  const bridge=createKdbBridge({url:base+'/mcp',attemptId:'readiness',agentDir,event:async(type,data)=>events.push({type,...data})});
  const settings=SettingsManager.inMemory({retry:{enabled:false}});
  const loader=new DefaultResourceLoader({cwd:root,agentDir,settingsManager:settings,noExtensions:true,noSkills:true,noContextFiles:true,
    extensionFactories:extension?[bridge.extension,pi=>pi.registerTool(bridge.knowledgeTool())]:[]});await loader.reload();
  const modelRuntime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
  modelRuntime.registerProvider('ready-fixture',{baseUrl:base+'/provider/v1',models:[{id:'scripted',name:'Scripted',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});
  await modelRuntime.setRuntimeApiKey('ready-fixture','fixture-only');
  const {session}=await createAgentSession({cwd:root,agentDir,modelRuntime,model:modelRuntime.getModel('ready-fixture','scripted'),settingsManager:settings,sessionManager:SessionManager.inMemory(root),resourceLoader:loader,noTools:'builtin'});
  t.after(async()=>{await session.extensionRunner.emit({type:'session_shutdown',reason:'quit'});session.dispose();for(const timer of timers)clearTimeout(timer);server.closeAllConnections();await new Promise(ok=>server.close(ok));await rm(root,{recursive:true,force:true});});
  const uiPrompts=[];session.subscribe(event=>{if(event.type==='ui_prompt_start')uiPrompts.push(event);});
  return {session,bridge,events,requests,queries,listing,uiPrompts,root,modelRuntime,base};
}

test('native tools/list beyond the 10s soft wait is ready before first provider; both nested queries succeed',{timeout:25000},async t=>{
  const f=await fixture(t,{listDelay:11000});const started=performance.now();
  await f.bridge.prepareSession(f.session);
  assert.ok(performance.now()-started>=11000);assert.equal(f.requests.length,0);
  await f.session.prompt('Retrieve good and negative evidence.');
  assert.deepEqual(f.queries,['good','negative']);assert.equal(f.requests.length,3);
  assert.equal(f.events.filter(e=>e.type==='knowledge-tool-result'&&!e.failed).length,2);
  assert.ok(f.events.some(e=>e.type==='knowledge-ready'));
  assert.equal(f.uiPrompts.length,0);
});

for(const options of [{missing:true},{unavailable:true},{extension:false}])test('missing required search fails before any provider request '+JSON.stringify(options),{timeout:15000},async t=>{
  const f=await fixture(t,options);
  await assert.rejects(f.bridge.prepareSession(f.session),/Required native/);
  assert.equal(f.requests.length,0);assert.equal(f.queries.length,0);
  if(options.unavailable)assert.ok(f.events.some(e=>e.type==='knowledge-notification'&&e.message.includes('failed')));
  assert.equal(f.events.some(e=>e.type==='knowledge-ready'),false);
});

test('cancel native discovery before provider work and retain notification/error bindings',{timeout:15000},async t=>{
  const f=await fixture(t,{listDelay:11000}),abort=new AbortController();
  const pending=f.bridge.prepareSession(f.session,{signal:abort.signal});await f.listing;abort.abort();
  await assert.rejects(pending,/abort/i);assert.equal(f.requests.length,0);assert.equal(f.queries.length,0);
  f.session.extensionRunner.getUIContext().notify('Retained native notification','warning');
  f.session.extensionRunner.emitError({extensionPath:'fixture',event:'fixture',error:'Retained native error'});
  await delay(0);
  assert.ok(f.events.some(e=>e.type==='knowledge-notification'&&e.message==='Retained native notification'));
  assert.ok(f.events.some(e=>e.type==='knowledge-extension-error'&&e.error==='Retained native error'));
});


test('canonical actor missing required KDB fails setup before provider or execution',{timeout:15000},async t=>{
  const {State}=await import('../../controller/state.mjs');
  const {runAttempt}=await import('../../controller/pi.mjs');
  const {writeFile}=await import('node:fs/promises');
  const f=await fixture(t,{missing:true});
  const source=join(f.root,'source');await mkdir(source);
  const verifier=join(f.root,'verify.py');await writeFile(verifier,'assert 1 == 1\n');
  const state=new State(':memory:');t.after(()=>state.close());
  state.add({id:'p',goal:'tiny numeric',source,actor_seconds:20,verifier:{files:[verifier],command:'python ../verify.py'}});
  const attempt=state.admit('p',{});let executions=0;
  await assert.rejects(runAttempt({state,attempt,project:state.project('p').spec,attemptPath:join(f.root,'attempt'),
    environmentFactory:async()=>({prepare:async()=>{},execute:async()=>{executions++;}}),
    knowledge:f.bridge.knowledge,extensions:[f.bridge.extension],skillMap:{},
    modelRuntime:f.modelRuntime,model:f.modelRuntime.getModel('ready-fixture','scripted')}),/Required native KDB/);
  assert.equal(f.requests.length,0);assert.equal(executions,0);assert.equal(state.attempt(attempt.id).outcome,null);
});


test('curriculum uses the same required native discovery before its first provider request',{timeout:15000},async t=>{
  const {summarize}=await import('../../cli/curriculum.mjs');
  const f=await fixture(t,{missing:true});
  const key='READINESS_CURRICULUM_KEY',previous=process.env[key];process.env[key]='fixture-only';
  t.after(()=>{if(previous===undefined)delete process.env[key];else process.env[key]=previous;});
  const model=f.modelRuntime.getModel('ready-fixture','scripted');
  await assert.rejects(summarize({path:f.root,bridge:f.bridge,passed:false,job:{attempt:{id:'readiness'}},
    state:{events:()=>[],queryEvents:()=>[],attempt:()=>({session:null}),event:()=>{}},
    config:{curriculum_seconds:10,actor:{provider:'ready-fixture',model:'scripted',credential_env:key,base_url:f.base+'/provider/v1',metadata:model}}}),/Required native KDB/);
  assert.equal(f.requests.length,0);assert.equal(f.queries.length,0);
});


test('abort during delayed native session_start binding is handled before the command race exists',{timeout:15000},async()=>{
  const {execFile}=await import('node:child_process');
  const {promisify}=await import('node:util');
  const script=`
    import {mkdtemp,rm} from 'node:fs/promises';
    import {tmpdir} from 'node:os';
    import {join} from 'node:path';
    import {setTimeout as delay} from 'node:timers/promises';
    import {createAgentSession,DefaultResourceLoader,ModelRuntime,SettingsManager,SessionManager} from '@earendil-works/pi-coding-agent';
    import {InMemoryCredentialStore} from '@earendil-works/pi-ai';
    import {createKdbBridge} from ${JSON.stringify(new URL('../../integrations/kdb-extension.mjs',import.meta.url).href)};
    const root=await mkdtemp(join(tmpdir(),'readiness-bind-abort-'));
    const abort=new AbortController();let shutdown=false,providerRequests=0;
    const bridge=createKdbBridge({url:'http://127.0.0.1:9/mcp',attemptId:'binding-abort',agentDir:root});
    const settings=SettingsManager.inMemory({});
    const loader=new DefaultResourceLoader({cwd:root,agentDir:root,settingsManager:settings,noExtensions:true,noSkills:true,noContextFiles:true,
      extensionFactories:[bridge.extension,pi=>{
        pi.on('session_start',async()=>{abort.abort(new Error('cancel during binding'));await delay(100);});
        pi.on('before_provider_request',()=>{providerRequests++;});
        pi.on('session_shutdown',()=>{shutdown=true;});
      }]});
    await loader.reload();
    const modelRuntime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
    const {session}=await createAgentSession({cwd:root,agentDir:root,modelRuntime,settingsManager:settings,sessionManager:SessionManager.inMemory(root),resourceLoader:loader,noTools:'builtin'});
    let cancelled=false;
    try{await bridge.prepareSession(session,{signal:abort.signal});}
    catch(error){if(error.message!=='cancel during binding')throw error;cancelled=true;}
    finally{await session.extensionRunner.emit({type:'session_shutdown',reason:'quit'});session.dispose();await rm(root,{recursive:true,force:true});}
    console.log(JSON.stringify({cancelled,shutdown,providerRequests}));
  `;
  const result=await promisify(execFile)(process.execPath,['--input-type=module','-e',script],{cwd:new URL('../..',import.meta.url).pathname});
  assert.deepEqual(JSON.parse(result.stdout.trim()),{cancelled:true,shutdown:true,providerRequests:0});
});


test('detached KDB starts once and reuses its healthy owned process without a host manager',{timeout:10000},async t=>{
  const {readFile,writeFile}=await import('node:fs/promises');
  const {execFile}=await import('node:child_process');
  const {promisify}=await import('node:util');
  const {ensureKnowledgeService}=await import('../../cli/knowledge.mjs');
  const {ownerStopped}=await import('../../controller/lifecycle.mjs');
  const root=await mkdtemp(join(tmpdir(),'detached-kdb-')),bin=join(root,'bin');await mkdir(bin);
  const {stdout}=await promisify(execFile)(process.env.PYTHON??'python3',['-c','import sys;print(sys.executable)']);
  const launcher=join(root,'python'),launches=join(root,'launches.jsonl');
  await writeFile(launcher,`#!${stdout.trim()}
import json,os,sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
with Path(${JSON.stringify(launches)}).open('a') as output:
 output.write(json.dumps({'pid':os.getpid(),'argv':sys.argv[1:]})+'\\n')
class Health(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(b'{}')
 def log_message(self,*args): pass
HTTPServer(('127.0.0.1',int(sys.argv[sys.argv.index('--port')+1])),Health).serve_forever()
`,{mode:0o700});
  const names=['PATH','RSI_MANAGED_ROOT','RSI_MANAGED_ROLE'],previous=Object.fromEntries(names.map(name=>[name,process.env[name]]));
  let owner;
  t.after(async()=>{
    for(const name of names)if(previous[name]===undefined)delete process.env[name];else process.env[name]=previous[name];
    try {
      if(!owner)try{owner=JSON.parse(await readFile(join(root,'knowledge-service.json'),'utf8')).owner;}catch(error){if(error.code!=='ENOENT')throw error;}
      if(owner&&!await ownerStopped(owner)) {
        process.kill(owner.pid,'SIGTERM');
        const deadline=performance.now()+2000;
        while(!await ownerStopped(owner)&&performance.now()<deadline)await delay(20);
        if(!await ownerStopped(owner))process.kill(owner.pid,'SIGKILL');
      }
    }finally{await rm(root,{recursive:true,force:true});}
  });
  process.env.PATH=bin;
  process.env.RSI_MANAGED_ROOT=join(root,'absent-parent');process.env.RSI_MANAGED_ROLE='worker';
  const config={root,python:launcher};
  const urls=await Promise.all([ensureKnowledgeService(config),ensureKnowledgeService(config)]);
  assert.equal(new Set(urls).size,1);
  const receipt=JSON.parse(await readFile(join(root,'knowledge-service.json'),'utf8'));owner=receipt.owner;
  assert.equal(await ownerStopped(owner),false);
  assert.equal(await ensureKnowledgeService(config),urls[0]);
  const records=(await readFile(launches,'utf8')).trim().split('\n').map(JSON.parse);
  assert.equal(records.length,1);assert.equal(records[0].pid,owner.pid);
  assert.deepEqual(records[0].argv.slice(0,4),['-m','ascend_tools.kdb_service','--db',join(root,'knowledge.sqlite')]);
});


test('healthy legacy KDB receipt reuses process ownership without consulting a service manager',async t=>{
  const {readFile,writeFile}=await import('node:fs/promises');
  const {ensureKnowledgeService}=await import('../../cli/knowledge.mjs');
  const {processIdentity}=await import('../../controller/lifecycle.mjs');
  const root=await mkdtemp(join(tmpdir(),'legacy-kdb-'));
  const server=createServer((_req,res)=>{res.writeHead(200);res.end('{}');});
  server.listen(0,'127.0.0.1');await once(server,'listening');
  t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));await rm(root,{recursive:true,force:true});});
  const receipt={url:`http://127.0.0.1:${server.address().port}/mcp`,owner:await processIdentity(),service:'old-knowledge.service'};
  const path=join(root,'knowledge-service.json');await writeFile(path,JSON.stringify(receipt));
  assert.equal(await ensureKnowledgeService({root,python:'missing-unused-python'}),receipt.url);
  assert.deepEqual(JSON.parse(await readFile(path,'utf8')),receipt);
});
