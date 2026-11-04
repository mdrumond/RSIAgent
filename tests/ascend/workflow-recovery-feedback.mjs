import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {ModelRuntime} from '@earendil-works/pi-coding-agent';
import {InMemoryCredentialStore} from '@earendil-works/pi-ai';
import {State} from '../../controller/state.mjs';
import {runAttempt} from '../../controller/pi.mjs';
import {stages} from '../../controller/workflow.mjs';
import {createEnvironment} from '../../integrations/environments/index.mjs';

test('native stale acceptance and wrong-stage verification recover through existing revision',{timeout:15000},async()=>{
  const root=await mkdtemp(join(tmpdir(),'workflow-terminal-')),state=new State(':memory:'),calls=[];
  const source=join(root,'source');await mkdir(source);await writeFile(join(source,'answer.py'),'answer=42\n');
  const verifier=join(root,'verify.py');await writeFile(verifier,"import runpy\nassert runpy.run_path('answer.py')['answer']==42\nprint('numeric passed')\n");
  for(const stage of stages){
    if(stage==='acceptance')break;
    calls.push(['rsi_skill',{}]);
    if(['design','optimisation'].includes(stage))for(const kind of ['good','negative'])calls.push(['rsi_knowledge',{query:'evidence',kind}]);
    if(stage==='validation')calls.push(['rsi_verify',{}]);
    calls.push(['rsi_advance',{reflection:'Retain independent verification.'}]);
  }
  calls.push(['rsi_skill',{}],['write',{path:'answer.py',content:'answer=6*7\n'}],
    ['rsi_advance',{}],['rsi_verify',{}],
    ['rsi_revise',{to:'implementation',reflection:'Source edited after verification; repeat canonical validation.'}]);
  const staleIndex=calls.length-3,wrongStageIndex=calls.length-2,reviseIndex=calls.length-1;
  for(const stage of stages.slice(stages.indexOf('implementation'))){
    calls.push(['rsi_skill',{}]);
    if(stage==='optimisation')for(const kind of ['good','negative'])calls.push(['rsi_knowledge',{query:'evidence',kind}]);
    if(stage==='validation')calls.push(['rsi_verify',{}]);
    calls.push(['rsi_advance',{reflection:'Revalidate the edited source independently.'}]);
  }
  const replies=new Map();
  const server=createServer(async(req,res)=>{
    let raw='';for await(const chunk of req)raw+=chunk;const body=JSON.parse(raw);
    const index=body.messages.reduce((n,m)=>n+(m.tool_calls?.length??0),0);
    for(const m of body.messages)if(m.role==='tool')replies.set(Number(m.tool_call_id.slice(5)),m.content);
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'terminal',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    if(index<calls.length){const [name,args]=calls[index];frame({role:'assistant',tool_calls:[{index:0,id:'call-'+index,type:'function',function:{name,arguments:JSON.stringify(args)}}]});frame({},'tool_calls');}
    else{frame({role:'assistant',content:'Accepted.'});frame({},'stop');}res.end('data: [DONE]\n\n');
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');
  try{
    const skillMap={};for(const stage of [...stages,'global-kdb']){skillMap[stage]=join(root,stage+'.md');await writeFile(skillMap[stage],'Use independent evidence.');}
    state.add({id:'p',goal:'Return42',source,actor_seconds:30,verifier:{files:[verifier],command:'python3 ../verify.py'}});
    const attempt=state.admit('p',{});
    const modelRuntime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
    modelRuntime.registerProvider('terminal-fixture',{baseUrl:`http://127.0.0.1:${server.address().port}/v1`,models:[{id:'scripted',name:'Scripted',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});await modelRuntime.setRuntimeApiKey('terminal-fixture','fixture-only');
    const outcome=await runAttempt({state,attempt,project:state.project('p').spec,attemptPath:join(root,'attempt'),skillMap,
      environmentFactory:context=>createEnvironment({mode:'local'},context),knowledge:{search:async()=>({content:[],evidence:[]})},modelRuntime,model:modelRuntime.getModel('terminal-fixture','scripted')});
    const events=state.events(attempt.id),verifications=events.filter(e=>e.type==='verification');
    assert.equal(outcome.accepted,true);assert.equal(state.attempt(attempt.id).outcome,'passed');
    assert.equal(verifications.length,2);assert.ok(verifications.every(e=>e.data.passed));
    assert.notEqual(verifications[0].data.fingerprint,verifications[1].data.fingerprint);
    for(const index of [staleIndex,wrongStageIndex])assert.throws(()=>JSON.parse(replies.get(index)));
    assert.equal(JSON.parse(replies.get(reviseIndex)).stage,'implementation');
    const transitions=events.filter(e=>e.type==='stage-completed');
    assert.ok(transitions.some(e=>e.data.stage==='acceptance'&&e.data.next==='implementation'));
    assert.equal(events.filter(e=>e.type==='accepted').length,1);
    const final=JSON.parse(replies.get(calls.length-1));assert.equal(final.accepted,true);
  }finally{server.closeAllConnections();await new Promise(ok=>server.close(ok));state.close();await rm(root,{recursive:true,force:true});}
});
