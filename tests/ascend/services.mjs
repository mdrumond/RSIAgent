import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {spawn,execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {mkdtemp,mkdir,writeFile,readFile,rm,stat,cp} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import {State} from '../../controller/state.mjs';
import {processIdentity,ownerStopped} from '../../controller/lifecycle.mjs';
import {ensureKnowledgeService} from '../../cli/knowledge.mjs';
import {deliverLessons} from '../../cli/memory.mjs';
import {investigate,summarize} from '../../cli/curriculum.mjs';

const read=async path=>JSON.parse(await readFile(path,'utf8'));
function setClaim(config,value) {
  const state=new State(join(config.root,'state.sqlite'));
  try {
    state.db.exec('CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY,value TEXT NOT NULL)');
    state.db.prepare('INSERT OR REPLACE INTO control(key,value) VALUES (?,?)').run('knowledge-startup',value);
  }finally{state.close();}
}
function getClaim(config) {
  const state=new State(join(config.root,'state.sqlite'));
  try{return state.db.prepare('SELECT value FROM control WHERE key=?').get('knowledge-startup')?.value;}
  finally{state.close();}
}
async function until(predicate) {
  const end=performance.now()+5000;
  while(!await predicate()) {if(performance.now()>=end)throw Error('fixture deadline');await delay(20);}
}
async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'ascend-services-'));
  const config={root,python:process.env.PYTHON??'python3.12',target:'a3'};
  t.after(async()=>{try{if(config.ownsService)await closeService(config);}finally{await rm(root,{recursive:true,force:true});}});
  return config;
}
async function closeService(config) {
  const {owner}=await read(join(config.root,'knowledge-service.json'));
  if(await ownerStopped(owner))return;
  process.kill(owner.pid,'SIGTERM');
  const end=performance.now()+1000;
  while(!await ownerStopped(owner)&&performance.now()<end)await delay(20);
  if(!await ownerStopped(owner))process.kill(owner.pid,'SIGKILL');
  await until(()=>ownerStopped(owner));
}

test('alive owner with delayed health is reused by concurrent callers; aborted waiter does not kill it',{timeout:10000},async t=>{
  const config=await fixture(t),began=performance.now();
  const server=createServer((_req,res)=>{const finish=()=>{res.writeHead(200);res.end('{}');};
    if(performance.now()-began<1200)setTimeout(finish,1200);else finish();});
  server.listen(0,'127.0.0.1');await once(server,'listening');
  t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));});
  const url=`http://127.0.0.1:${server.address().port}/mcp`,owner=await processIdentity();
  await writeFile(join(config.root,'knowledge-service.json'),JSON.stringify({url,owner}));
  const abort=new AbortController();setTimeout(()=>abort.abort(),30);
  await assert.rejects(ensureKnowledgeService(config,{signal:abort.signal}),{name:'AbortError'});
  const results=await Promise.all([ensureKnowledgeService(config),ensureKnowledgeService(config),ensureKnowledgeService(config)]);
  assert.deepEqual(results,[url,url,url]);
  assert.deepEqual((await read(join(config.root,'knowledge-service.json'))).owner,owner);
  await assert.rejects(stat(join(config.root,'logs')),{code:'ENOENT'});
  assert.equal(await ownerStopped(owner),false);
});

test('dead SQLite startup claim recovers once and standard MCP delivery ACKs only success',{timeout:20000},async t=>{
  const config=await fixture(t);
  const {stdout}=await promisify(execFile)(config.python,['-c','import sys; print(sys.executable)']);
  const python=stdout.trim();
  const expired=spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{stdio:'ignore'});
  const dead=await processIdentity(expired.pid),exited=once(expired,'exit');expired.kill('SIGKILL');await exited;
  setClaim(config,JSON.stringify(dead));
  await writeFile(join(config.root,'knowledge-service.json'),JSON.stringify({url:'http://127.0.0.1:1/mcp',owner:dead}));
  const serviceRoot=resolve(new URL('../..',import.meta.url).pathname),kdbRoot=resolve(process.env.SOURCE_KDB_ROOT??serviceRoot);
  const wrapper=join(config.root,'python-wrapper');
  const launcher=`#!${python}\nimport sys,runpy\nsys.path[:0]=[${JSON.stringify(config.root)},${JSON.stringify(kdbRoot)}]\nimport ascend_tools\nascend_tools.__path__.append(${JSON.stringify(join(serviceRoot,'ascend_tools'))})\nmodule=sys.argv[2];sys.argv=[module]+sys.argv[3:]\nrunpy.run_module(module,run_name='__main__')\n`;
  await writeFile(wrapper,launcher,{mode:0o700});config.python=wrapper;config.embedding_factory='service_fixture:Embeddings';
  await writeFile(join(config.root,'service_fixture.py'),`import os,time,hashlib\nfrom pathlib import Path\nroot=Path(${JSON.stringify(config.root)})\nclass Embeddings:\n model='deterministic-cpu';revision='1'\n def __init__(self):\n  with (root/'launches').open('a') as f:f.write(str(os.getpid())+'\\n')\n def embed(self,texts):\n  if (root/'block').exists():\n   (root/'entered').write_text('entered');time.sleep(30)\n  return [[x/255 for x in hashlib.sha256(t.encode()).digest()] for t in texts]\n`);
  config.ownsService=true;
  const urls=await Promise.all([ensureKnowledgeService(config),ensureKnowledgeService(config),ensureKnowledgeService(config)]).catch(async error=>{error.message+='\n'+await readFile(join(config.root,'logs','knowledge.log'),'utf8');throw error;});
  assert.equal(new Set(urls).size,1);
  assert.equal((await readFile(join(config.root,'launches'),'utf8')).trim().split('\n').length,1);
  assert.equal(getClaim(config),undefined);
  const evidence=join(config.root,'evidence.log');await writeFile(evidence,'independent numeric evidence');
  const acknowledgements=[];
  let lesson={id:1,attempt_id:'success',outcome:'passed',findings:'numeric evidence',evidence:[{path:evidence}]};
  const queue={lessons:()=>[lesson],acknowledgeLesson:id=>acknowledgements.push(id)};
  await deliverLessons({config,queue,url:urls[0]});assert.deepEqual(acknowledgements,[1]);
  lesson={...lesson,id:2,attempt_id:'failure',evidence:[{path:evidence,sha256:'wrong'}]};
  await assert.rejects(deliverLessons({config,queue,url:urls[0]}));assert.deepEqual(acknowledgements,[1]);
  lesson={...lesson,id:3,attempt_id:'abort',evidence:[{path:evidence}]};
  await writeFile(join(config.root,'block'),'1');
  const abort=new AbortController(),delivery=deliverLessons({config,queue,url:urls[0],signal:abort.signal});
  const rejected=assert.rejects(delivery,{name:'AbortError'});
  await until(async()=>{try{await stat(join(config.root,'entered'));return true;}catch{return false;}});
  abort.abort();await rejected;assert.deepEqual(acknowledgements,[1]);
});

test('live, unknown and empty SQLite startup claims remain preserved on wait cancellation',async t=>{
  for(const value of [JSON.stringify(await processIdentity()),'{}','']) {
    const config=await fixture(t);setClaim(config,value);
    const abort=new AbortController();setTimeout(()=>abort.abort(),40);
    await assert.rejects(ensureKnowledgeService(config,{signal:abort.signal}),{name:'AbortError'});
    assert.equal(getClaim(config),value);
    await assert.rejects(stat(join(config.root,'knowledge-service.json')),{code:'ENOENT'});
  }
});

test('native curriculum returns executable source-relative verifier after relocation',{timeout:15000},async t=>{
  const config=await fixture(t),source=join(config.root,'original');await mkdir(source);await writeFile(join(source,'answer.py'),'answer=42\n');
  const calls=[['write',{path:'verify.py',content:"import runpy\nassert runpy.run_path('answer.py')['answer']==6*7\nprint(42)\n"}],
    ['investigation_knowledge',{query:'numeric control',kind:'good'}],['investigation_knowledge',{query:'numeric mismatch',kind:'negative'}],
    ['submit_investigation',{goal:'independent numeric check',question:'Is the result42?',verifier:'verify.py'}]];
  const server=createServer(async(req,res)=>{
    let raw='';for await(const data of req)raw+=data;const body=JSON.parse(raw);
    assert.ok(body.tools.every(tool=>!['bash','rsi_target'].includes(tool.function.name)));
    const index=body.messages.reduce((n,m)=>n+(m.tool_calls?.length??0),0);res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    if(index<calls.length){const [name,args]=calls[index];frame({role:'assistant',tool_calls:[{index:0,id:'call-'+index,type:'function',function:{name,arguments:JSON.stringify(args)}}]});frame({},'tool_calls');}
    else{frame({role:'assistant',content:'Investigation recorded.'});frame({},'stop');}res.end('data: [DONE]\n\n');
  });server.listen(0,'127.0.0.1');await once(server,'listening');
  t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));});
  const old=process.env.SERVICE_FIXTURE_KEY;process.env.SERVICE_FIXTURE_KEY='fixture-only';
  t.after(()=>{if(old===undefined)delete process.env.SERVICE_FIXTURE_KEY;else process.env.SERVICE_FIXTURE_KEY=old;});
  config.actor={provider:'service-fixture',model:'scripted',base_url:`http://127.0.0.1:${server.address().port}/v1`,credential_env:'SERVICE_FIXTURE_KEY',
    metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
  const path=join(config.root,'attempt');await mkdir(path);
  const state=new State(join(config.root,'curriculum.sqlite'));t.after(()=>state.close());
  state.add({id:'p',source,goal:'numeric',actor_seconds:120,verifier:{files:[],command:'true'}});
  const attempt=state.admit('p',await processIdentity());
  const spec=await investigate({config,path,job:{attempt,origin_attempt:attempt.id,actor_seconds:120},base:{source,goal:'numeric',performance_goal:'parent only'},state,bridge:{extension:()=>{},knowledge:{search:async({kind})=>({content:[{kind}],evidence:[]})}}});
  assert.equal(state.events(attempt.id).find(e=>e.type==='curriculum-session').data.role,'investigation');
  assert.equal(spec.performance_goal,undefined);
  assert.deepEqual(state.queryEvents(attempt.id,{types:['investigation-knowledge']}).map(e=>e.data.kind),['good','negative']);
  const relocated=join(config.root,'relocated');await cp(join(path,'investigation'),relocated,{recursive:true});await rm(path,{recursive:true});
  const result=await promisify(execFile)('bash',['-c',spec.verifier.command],{cwd:join(relocated,'source')});assert.equal(Number(result.stdout.trim()),42);
});

test('native summary and investigation retrieve authoritative diagnostics despite Pi telemetry tail',{timeout:15000},async t=>{
  const config=await fixture(t),path=join(config.root,'attempt'),source=join(config.root,'source');
  await mkdir(path);await mkdir(source);
  const evidence=join(path,'verify.log'),session=join(path,'actor.jsonl');
  await writeFile(evidence,'numeric-check-observed-42');await writeFile(session,'native-actor-diagnostic-17');
  const state=new State(join(config.root,'diagnostics.sqlite'));t.after(()=>state.close());
  state.add({id:'project',goal:'numeric',source,actor_seconds:120,verifier:{files:[],command:'true'}});
  const attempt=state.admit('project',await processIdentity());state.checkpoint(attempt.id,1,session);
  state.event(attempt.id,'verification',{passed:true,evidence});
  state.event(attempt.id,'rejected',{diagnostics:['workflow acceptance incomplete']});
  for(let i=0;i<80;i++)state.event(attempt.id,'pi-event',{type:'message_update',i});
  const retrieved=[];
  const server=createServer(async(req,res)=>{
    let raw='';for await(const data of req)raw+=data;const body=JSON.parse(raw);
    assert.ok(body.tools.every(tool=>!['bash','rsi_target'].includes(tool.function.name)));
    const prompt=body.messages.filter(m=>m.role==='user').map(m=>typeof m.content==='string'?m.content:m.content.map(part=>part.text??'').join('\n')).join('\n');
    const context=JSON.parse(prompt.split('Attempt diagnostics: ')[1].split('\n')[0]);
    const logs=context.diagnostics.find(event=>event.type==='verification')?.data.evidence;
    const tools=body.messages.filter(m=>m.role==='tool');
    const investigation=body.tools.some(tool=>tool.function.name==='submit_investigation');
    let name,args;
    if(tools.length===0){name='read';args={path:context.session};}
    else if(tools.length===1){name='read';args={path:logs};}
    else {
      const results=tools.slice(0,2).map(m=>typeof m.content==='string'?m.content:JSON.stringify(m.content));
      assert.ok(results[0].includes('native-actor-diagnostic-17'));assert.ok(results[1].includes('numeric-check-observed-42'));
      if(investigation&&tools.length===2){name='write';args={path:'verify.py',content:'print(42)\n'};}
      else if(investigation&&[3,4].includes(tools.length)){name='investigation_knowledge';args={query:'numeric acceptance evidence',kind:tools.length===3?'good':'negative'};}
      else if(tools.length===(investigation?5:2)) {
        retrieved.push({session:context.session,evidence:logs});
        name=investigation?'submit_investigation':'record_findings';
        args=investigation?{goal:'inspect numeric',question:'why acceptance incomplete?',verifier:'verify.py'}:
          {findings:'Numeric check observed 42; acceptance incomplete.',useful:true,paths:[logs]};
      }
    }
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason)=>res.write('data: '+JSON.stringify({id:'diagnostics',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    if(name){frame({role:'assistant',tool_calls:[{index:0,id:'call-'+tools.length,type:'function',function:{name,arguments:JSON.stringify(args)}}]},null);frame({},'tool_calls');}
    else{frame({role:'assistant',content:'Recorded.'},null);frame({},'stop');}
    res.end('data: [DONE]\n\n');
  });server.listen(0,'127.0.0.1');await once(server,'listening');
  t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));});
  const old=process.env.SERVICE_DIAGNOSTIC_KEY;process.env.SERVICE_DIAGNOSTIC_KEY='fixture-only';
  t.after(()=>{if(old===undefined)delete process.env.SERVICE_DIAGNOSTIC_KEY;else process.env.SERVICE_DIAGNOSTIC_KEY=old;});
  config.actor={provider:'service-diagnostics',model:'scripted',base_url:`http://127.0.0.1:${server.address().port}/v1`,credential_env:'SERVICE_DIAGNOSTIC_KEY',
    metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
  const bridge={extension:()=>{},knowledge:{search:async({kind})=>({content:[{kind}],evidence:[]})}};
  const findings=await summarize({config,path,state,job:{attempt},bridge,passed:true});
  assert.equal(findings.evidence[0].path,evidence);
  await investigate({config,path,state,job:{attempt,origin_attempt:attempt.id,actor_seconds:120},base:{source,goal:'numeric'},bridge});
  assert.deepEqual(retrieved,[{session,evidence},{session,evidence}]);
  assert.deepEqual(state.queryEvents(attempt.id,{types:['investigation-knowledge']}).map(e=>e.data.kind),['good','negative']);
  assert.equal(state.events(attempt.id).filter(event=>event.type==='pi-event').length,80);
});
