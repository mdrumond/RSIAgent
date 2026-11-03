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
import {processIdentity,ownerStopped} from '../../controller/lifecycle.mjs';
import {ensureKnowledgeService} from '../../cli/knowledge.mjs';
import {deliverLessons} from '../../cli/memory.mjs';
import {investigate} from '../../cli/curriculum.mjs';

const read=async path=>JSON.parse(await readFile(path,'utf8'));
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

test('dead startup lock recovers once and standard MCP delivery ACKs only success',{timeout:20000},async t=>{
  const config=await fixture(t),python=config.python;
  const expired=spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{stdio:'ignore'});
  const dead=await processIdentity(expired.pid),exited=once(expired,'exit');expired.kill('SIGKILL');await exited;
  const lock=join(config.root,'knowledge-service.json.lock');await writeFile(lock,JSON.stringify(dead));
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
  await assert.rejects(stat(lock),{code:'ENOENT'});
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

test('live, unknown and empty legacy startup locks remain preserved on wait cancellation',async t=>{
  for(const value of [JSON.stringify(await processIdentity()),'{}','']) {
    const config=await fixture(t),lock=join(config.root,'knowledge-service.json.lock');await writeFile(lock,value);
    const abort=new AbortController();setTimeout(()=>abort.abort(),40);
    await assert.rejects(ensureKnowledgeService(config,{signal:abort.signal}),{name:'AbortError'});
    assert.equal(await readFile(lock,'utf8'),value);
    await assert.rejects(stat(join(config.root,'knowledge-service.json')),{code:'ENOENT'});
  }
});

test('native curriculum returns executable source-relative verifier after relocation',{timeout:15000},async t=>{
  const config=await fixture(t),source=join(config.root,'original');await mkdir(source);await writeFile(join(source,'answer.py'),'answer=42\n');
  const calls=[['write',{path:'verify.py',content:"import runpy\nassert runpy.run_path('answer.py')['answer']==6*7\nprint(42)\n"}],
    ['submit_investigation',{goal:'independent numeric check',question:'Is the result42?',verifier:'verify.py'}]];
  const server=createServer(async(req,res)=>{
    let raw='';for await(const data of req)raw+=data;const body=JSON.parse(raw);
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
  const spec=await investigate({config,path,job:{origin_attempt:'failed',actor_seconds:120},base:{source,goal:'numeric',performance_goal:'parent only'},state:{events:()=>[]},bridge:{extension:()=>{}}});
  assert.equal(spec.performance_goal,undefined);
  const relocated=join(config.root,'relocated');await cp(join(path,'investigation'),relocated,{recursive:true});await rm(path,{recursive:true});
  const result=await promisify(execFile)('bash',['-c',spec.verifier.command],{cwd:join(relocated,'source')});assert.equal(Number(result.stdout.trim()),42);
});
