import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { mkdtemp,mkdir,writeFile,readFile,rm,stat } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { ModelRuntime } from '@earendil-works/pi-coding-agent';
import { InMemoryCredentialStore } from '@earendil-works/pi-ai';
import { createPiTelemetryJournal } from '../../controller/pi-telemetry.mjs';
import { State } from '../../controller/state.mjs';
import { runAttempt } from '../../controller/pi.mjs';
import { stages } from '../../controller/workflow.mjs';
import { processIdentity,attemptStopped } from '../../controller/lifecycle.mjs';
import { createEnvironment } from '../../integrations/environments/index.mjs';

const records=async path=>(await readFile(path,'utf8')).trim().split('\n').map(line=>JSON.parse(line));
const busy=()=>Object.assign(new Error('database is locked',{cause:Object.assign(new Error('writer held'),{code:'SQLITE_BUSY'})}),
  {code:'ERR_SQLITE_ERROR',errcode:5,errstr:'database is locked'});

test('journal retains failed and later metadata, original exception, and resume latch',async t=>{
  const root=await mkdtemp(join(tmpdir(),'pi-telemetry-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const path=join(root,'events.jsonl'),error=busy(),first={type:'message_update',stage:'optimisation',elapsed:7,actor_elapsed:7.1,operation:'op'};
  let writes=0;
  const sink=createPiTelemetryJournal(path,'attempt',()=>{writes++;throw error;});
  sink(first);sink({...first,type:'message_end',elapsed:8});
  const resumed=createPiTelemetryJournal(path,'attempt',()=>{writes++;assert.fail('Latched SQL must not be retried');});
  resumed({...first,type:'agent_end',elapsed:9});
  const rows=await records(path);
  assert.equal(writes,1);assert.equal(rows.length,3);assert.deepEqual(rows[0].data,first);
  assert.deepEqual(rows.map(r=>r.data.type),['message_update','message_end','agent_end']);
  assert.equal(rows[0].attempt,'attempt');assert.ok(Number.isFinite(Date.parse(rows[0].created)));
  const saved=rows[0].persistence.error;
  for(const key of ['name','message','code','errcode','errstr','stack'])assert.equal(saved[key],error[key]);
  assert.equal(saved.cause.code,error.cause.code);assert.equal(saved.cause.stack,error.cause.stack);
  assert.equal(rows[1].persistence,undefined);assert.equal((await stat(path)).mode&0o777,0o600);
});

test('healthy telemetry keeps SQL only; journal persistence errors escape',async t=>{
  const root=await mkdtemp(join(tmpdir(),'pi-telemetry-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const path=join(root,'events.jsonl'),seen=[],data={type:'message_start'};
  createPiTelemetryJournal(path,'a',value=>seen.push(value))(data);
  assert.deepEqual(seen,[data]);await assert.rejects(stat(path),{code:'ENOENT'});
  await mkdir(path);
  assert.throws(()=>createPiTelemetryJournal(path,'a',()=>assert.fail())(data),{code:'EISDIR'});
  assert.throws(()=>createPiTelemetryJournal(join(root,'missing','events.jsonl'),'a',()=>{throw busy();})(data),
    {code:'ENOENT'});
});

for(const scenario of ['journal','sidecar-failure','bash-closure-failure','checkpoint-failure','acceptance-failure'])
test(`native workflow telemetry: ${scenario}`,{timeout:20000},async()=>{
  const root=await mkdtemp(join(tmpdir(),'pi-telemetry-native-')),source=join(root,'source'),attemptPath=join(root,'attempt');
  await mkdir(source);await mkdir(attemptPath);
  const state=new State(join(root,'state.sqlite')),calls=[],originalEvent=state.event.bind(state),error=busy();
  let telemetryWrites=0,requests=0,closureFailure=false,shutdown=0;
  const python=process.env.DRIVER_PYTHON??'python3';
  for(const stage of stages){
    calls.push(['rsi_skill',{}]);
    if(['design','optimisation'].includes(stage))for(const kind of ['good','negative'])calls.push(['rsi_knowledge',{query:'arithmetic evidence',kind}]);
    if(stage==='implementation')calls.push(['write',{path:'answer.py',content:'answer=6*7\n'}],['bash',{command:`${python} -c "assert 6*7==42"`}]);
    if(stage==='validation')calls.push(['rsi_verify',{}]);
    calls.push(['rsi_advance',{reflection:'Use independent arithmetic evidence.'}]);
  }
  const server=createServer(async(req,res)=>{
    let raw='';for await(const chunk of req)raw+=chunk;
    const input=JSON.parse(raw),index=input.messages.reduce((n,m)=>n+(m.tool_calls?.length??0),0),call=calls[index];requests++;
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>'data: '+JSON.stringify({id:'telemetry',object:'chat.completion.chunk',created:1,
      model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n';
    res.end(frame(call?{role:'assistant',tool_calls:[{index:0,id:'call-'+index,type:'function',function:{name:call[0],arguments:JSON.stringify(call[1])}}]}:
      {role:'assistant',content:'Done.'})+frame({},call?'tool_calls':'stop')+'data: [DONE]\n\n');
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');
  try{
    const verifier=join(root,'verify.py'),verifierSource="import runpy\nassert runpy.run_path('answer.py')['answer']==sum(6 for _ in range(7))\n";
    await writeFile(verifier,verifierSource);await writeFile(join(source,'answer.py'),'answer=0\n');
    const skillMap={};for(const stage of [...stages,'global-kdb']){skillMap[stage]=join(root,stage+'.md');await writeFile(skillMap[stage],'Use independent evidence.');}
    state.add({id:'p',goal:'Compute independent arithmetic',source,actor_seconds:30,verifier:{files:[verifier],command:`${python} ../verify.py`}});
    const attempt=state.admit('p',{...await processIdentity(),start:'previous-owner'});
    state.event=(id,type,data)=>{
      if(type==='pi-event'){telemetryWrites++;if(scenario==='journal')throw error;}
      if(scenario==='bash-closure-failure'&&type==='native-bash-completed'){closureFailure=true;throw error;}
      if(scenario==='acceptance-failure'&&type==='accepted')throw error;
      return originalEvent(id,type,data);
    };
    if(scenario==='checkpoint-failure')state.checkpoint=()=>{throw error;};
    if(scenario==='sidecar-failure')await mkdir(join(attemptPath,'pi-events.jsonl'));
    const runtime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
    runtime.registerProvider('telemetry-fixture',{baseUrl:`http://127.0.0.1:${server.address().port}/v1`,models:[{id:'scripted',name:'Scripted telemetry fixture',
      api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});
    await runtime.setRuntimeApiKey('telemetry-fixture','fixture-only');
    let environment;
    const promise=runAttempt({state,attempt,project:state.project('p').spec,attemptPath,skillMap,knowledge:{search:async()=>({content:[],evidence:[]})},
      modelRuntime:runtime,model:runtime.getModel('telemetry-fixture','scripted'),environmentFactory:async context=>environment=await createEnvironment({mode:'local'},context),
      extensions:[pi=>pi.on('session_shutdown',async()=>{shutdown++;})]});
    if(['sidecar-failure','checkpoint-failure'].includes(scenario)){
      await assert.rejects(promise,scenario==='sidecar-failure'?{code:'EISDIR'}:{message:error.message});
      assert.equal(state.attempt(attempt.id).outcome,null);
    }else{
      const result=await promise;assert.equal(result.accepted,scenario==='journal');
      assert.equal(state.attempt(attempt.id).outcome,scenario==='journal'?'passed':'failed');
    }
    assert.equal(shutdown,1);assert.equal(await readFile(verifier,'utf8'),verifierSource);
    const events=state.events(attempt.id);
    assert.equal(events.some(e=>e.type==='accepted'),scenario==='journal');
    if(scenario==='journal'){
      assert.equal(telemetryWrites,1);assert.ok(requests>10);
      const journal=await records(join(attemptPath,'pi-events.jsonl'));
      assert.ok(journal.some(r=>r.data.type==='agent_end'));assert.equal(journal[0].persistence.error.code,error.code);
      assert.ok(journal.every(r=>r.attempt===attempt.id));
      const session=await readFile(state.attempt(attempt.id).session,'utf8');
      assert.ok(session.split('\n').some(line=>{try{const m=JSON.parse(line).message;return m?.role==='toolResult'&&m.toolName==='bash'&&!m.isError;}catch{return false;}}));
      assert.equal(await attemptStopped(state,state.attempt(attempt.id),environment),true);
    }else if(scenario==='bash-closure-failure'){
      assert.equal(closureFailure,true);assert.equal(await attemptStopped(state,state.attempt(attempt.id),environment),false);
    }
    if(scenario!=='journal'&&scenario!=='sidecar-failure')await assert.rejects(stat(join(attemptPath,'pi-events.jsonl')),{code:'ENOENT'});
  }finally{
    server.closeAllConnections();await new Promise(resolve=>server.close(resolve));state.close();await rm(root,{recursive:true,force:true});
  }
});
