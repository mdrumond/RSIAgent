import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {spawn,execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import {ModelRuntime} from '@earendil-works/pi-coding-agent';
import {InMemoryCredentialStore} from '@earendil-works/pi-ai';
import {State} from '../../controller/state.mjs';
import {runAttempt} from '../../controller/pi.mjs';
import {stages} from '../../controller/workflow.mjs';
import {createKdbBridge} from '../../integrations/kdb-extension.mjs';
import {createEnvironment} from '../../integrations/environments/index.mjs';
import {RecoveryQueue,recordOutcome} from '../../controller/recovery.mjs';
import {deliverLessons} from '../../cli/memory.mjs';

// Actual Pi, local Bash, Python MCP service and controller lesson delivery; scripted provider only.
test('actor first provider has read-only KDB and dynamic tools; controller still delivers and acknowledges lessons',{timeout:25000},async()=>{
  const root=await mkdtemp(join(tmpdir(),'controller-lessons-')),python=process.env.PYTHON??'python3.12';
  const source=join(root,'source'),agentDir=join(root,'attempt');await mkdir(source);await mkdir(agentDir);
  await writeFile(join(source,'answer.py'),'answer=42\n');
  const verifier=join(root,'verify.py');await writeFile(verifier,"import runpy\nassert runpy.run_path('answer.py')['answer']==42\nprint('numeric pass')\n");
  const pythonEnv={...process.env,PYTHONPATH:[process.cwd(),resolve('tests/ascend'),process.env.PYTHONPATH??''].join(':')};
  const db=join(root,'knowledge.sqlite'),notes=join(root,'notes');await mkdir(notes);await writeFile(join(notes,'fact.py'),'answer=42\n');
  await promisify(execFile)(python,['-c',"import sys;from ascend_tools.kdb import KDB;from kdb_fixture import Embeddings;k=KDB(sys.argv[1],Embeddings());k.add_repo(sys.argv[2],'baseline','python',target='local-python');k.close()",db,notes],{env:pythonEnv});
  const socket=createServer();socket.listen(0,'127.0.0.1');await once(socket,'listening');const port=socket.address().port;await new Promise(ok=>socket.close(ok));
  const service=spawn(python,['-m','ascend_tools.kdb_service','--db',db,'--port',String(port),'--embedding-factory','kdb_fixture:Embeddings'],{env:pythonEnv,stdio:['ignore','ignore','pipe']});
  let stderr='';service.stderr.on('data',data=>{stderr=(stderr+data).slice(-2000);});
  const state=new State(':memory:'),calls=[],wires=[],events=[];let targetCalls=0,providerRequests=0;
  for(const stage of stages){calls.push(['rsi_skill',{}]);if(stage==='preflight')calls.push(['rsi_future_target',{}]);
    if(['design','optimisation'].includes(stage))for(const kind of ['good','negative']){calls.push(['rsi_knowledge',{query:'answer42',kind}]);if(stage==='design'&&kind==='good')calls.push(['mcp__kdb__read_entry',null]);}
    if(stage==='implementation')calls.push(['bash',{command:'python3 -c "assert 6*7 == 42"'}]);
    if(stage==='validation')calls.push(['rsi_verify',{}]);calls.push(['rsi_advance',{reflection:'Retain the actual independent numeric evidence.'}]);}
  const provider=createServer(async(req,res)=>{
    let raw='';for await(const chunk of req)raw+=chunk;const body=JSON.parse(raw);providerRequests++;
    const names=body.tools.map(tool=>tool.function.name);wires.push(names);
    assert.ok(!names.includes('mcp__kdb__add_repo'));assert.ok(!names.includes('mcp__kdb__record_lesson'));
    for(const name of ['mcp__kdb__search','mcp__kdb__read_entry','rsi_future_target','bash','read','write','edit','rsi_skill','rsi_verify'])assert.ok(names.includes(name),name);
    const index=body.messages.reduce((n,m)=>n+(m.tool_calls?.length??0),0);
    res.writeHead(200,{'Content-Type':'text/event-stream'});const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'lessons',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    if(index<calls.length){let [name,args]=calls[index];if(name==='mcp__kdb__read_entry'){const prior=body.messages.find(m=>m.role==='tool'&&m.tool_call_id==='call-'+(index-1));const [hit]=JSON.parse(prior.content);args={collection:hit.collection,entry_id:hit.chunk_id};}
      frame({role:'assistant',tool_calls:[{index:0,id:'call-'+index,type:'function',function:{name,arguments:JSON.stringify(args)}}]});frame({},'tool_calls');}
    else{frame({role:'assistant',content:'Independent evidence retained; controller owns lesson delivery.'});frame({},'stop');}res.end('data: [DONE]\n\n');
  });
  provider.listen(0,'127.0.0.1');await once(provider,'listening');
  try{
    const deadline=Date.now()+10000;for(;;){if(service.exitCode!==null)throw Error(stderr);try{if((await fetch(`http://127.0.0.1:${port}/health`)).ok)break;}catch{}if(Date.now()>deadline)throw Error('KDB startup timeout');await delay(25);}
    state.add({id:'p',goal:'Return42 with independent verification',source,actor_seconds:30,verifier:{files:[verifier],command:'python3 ../verify.py'}});const attempt=state.admit('p',{});
    const bridge=createKdbBridge({url:`http://127.0.0.1:${port}/mcp`,attemptId:attempt.id,agentDir,target:'local-python',event:(type,data)=>events.push({type,...data})});
    const runtime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
    runtime.registerProvider('lessons-fixture',{baseUrl:`http://127.0.0.1:${provider.address().port}/v1`,models:[{id:'scripted',name:'Scripted',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});await runtime.setRuntimeApiKey('lessons-fixture','fixture-only');
    const skillMap={};for(const stage of [...stages,'global-kdb']){skillMap[stage]=join(root,stage+'.md');await writeFile(skillMap[stage],'Use independent evidence and complete the workflow.');}
    const dynamic=pi=>pi.registerTool({name:'rsi_future_target',label:'Future target tool',description:'Fixture for dynamically retained non-KDB tools',parameters:{type:'object',properties:{}},execute:async(_id,_args,_signal,_update,ctx)=>{
      targetCalls++;const callable=ctx.tools.map(tool=>tool.name);assert.ok(!callable.includes('mcp__kdb__record_lesson'));assert.ok(!callable.includes('mcp__kdb__add_repo'));assert.ok(callable.includes('mcp__kdb__search'));return {content:[{type:'text',text:'retained'}],details:{}};}});
    const result=await runAttempt({state,attempt,project:state.project('p').spec,attemptPath:agentDir,knowledge:bridge.knowledge,extensions:[bridge.extension,dynamic],skillMap,modelRuntime:runtime,model:runtime.getModel('lessons-fixture','scripted'),environmentFactory:context=>createEnvironment({mode:'local'},context)});
    assert.equal(result.accepted,true);assert.ok(providerRequests>0);assert.equal(targetCalls,1);assert.ok(wires.length>1);
    assert.equal(events.filter(e=>e.type==='knowledge-tool-result'&&e.tool==='mcp__kdb__search'&&!e.failed).length,4);
    assert.ok(events.some(e=>e.tool==='mcp__kdb__read_entry'&&!e.failed));assert.equal(events.some(e=>e.tool==='mcp__kdb__record_lesson'),false);
    const verification=state.events(attempt.id).find(e=>e.type==='verification').data.correctness;
    const queue=new RecoveryQueue(state);recordOutcome(state,{attemptId:attempt.id,wave:0,passed:true,findings:'Independent numeric verifier passed.',evidence:[{path:verification.evidence,sha256:verification.evidence_sha256}]});
    assert.equal(queue.lessons().length,1);
    await deliverLessons({config:{python,target:'local-python'},queue,url:`http://127.0.0.1:${port}/mcp`});assert.equal(queue.lessons().length,0);
    assert.equal(state.db.prepare('SELECT recorded FROM recovery_lessons WHERE attempt=?').get(attempt.id).recorded,1);
  }finally{provider.closeAllConnections();await new Promise(ok=>provider.close(ok));if(service.exitCode===null){const exited=once(service,'exit');service.kill('SIGTERM');await exited;}state.close();await rm(root,{recursive:true,force:true});}
});
