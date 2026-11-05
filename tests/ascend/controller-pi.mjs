import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { mkdtemp,mkdir,writeFile,rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { ModelRuntime } from '@earendil-works/pi-coding-agent';
import { InMemoryCredentialStore } from '@earendil-works/pi-ai';
import { State } from '../../controller/state.mjs';
import { runAttempt } from '../../controller/pi.mjs';
import { stages } from '../../controller/workflow.mjs';
import { processIdentity,attemptStopped } from '../../controller/lifecycle.mjs';

for(const scenario of ['cancel-verifier','unknown-submission','supervised-shell'])test(scenario,{timeout:15000},async()=>{
  const root=await mkdtemp(join(tmpdir(),'ascend-native-lifecycle-'));
  const state=new State(join(root,'state.sqlite')),calls=[];
  for(const stage of stages) {
    calls.push(['rsi_skill',{}]);
    if(['design','optimisation'].includes(stage))for(const kind of ['good','negative'])calls.push(['rsi_knowledge',{query:'prior facts',kind}]);
    if(stage==='validation')calls.push(['rsi_verify',{}]);
    calls.push(['rsi_advance',{reflection:'Previous evidence supports an independent numeric check.'}]);
  }
  if(scenario==='supervised-shell')calls.unshift(['bash',{command:'python program.py'}]);
  let shutdown=0,sawCancellation=false,requests=0;
  const server=createServer(async(req,res)=>{
    let raw='';for await(const data of req)raw+=data;
    const body=JSON.parse(raw),index=body.messages.reduce((n,m)=>n+(m.tool_calls?.length??0),0);
    const [name,args]=calls[index];requests++;
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'fixture',object:'chat.completion.chunk',
      created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    frame({role:'assistant',tool_calls:[{index:0,id:'call-'+index,type:'function',function:{name,arguments:JSON.stringify(args)}}]});
    frame({},'tool_calls');res.end('data: [DONE]\n\n');
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');
  try {
    const source=join(root,'source');await mkdir(source);await writeFile(join(source,'answer.py'),'answer=42\n');
    const verifier=join(root,'verify.py');await writeFile(verifier,'assert 6*7 == 42\n');
    const skillMap={};for(const scope of [...stages,'global-kdb']) {
      skillMap[scope]=join(root,scope+'.md');await writeFile(skillMap[scope],'Use independent evidence and the canonical workflow.');
    }
    state.add({id:'p',goal:'Numeric fixture',actor_seconds:30,source,verifier:{files:[verifier],command:'python ../verify.py'}});
    const attempt=state.admit('p',{...await processIdentity(),start:'previous-process'});
    const runtime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,
      allowModelNetwork:false,refreshOnCreate:false});
    runtime.registerProvider('lifecycle-fixture',{baseUrl:`http://127.0.0.1:${server.address().port}/v1`,models:[{
      id:'scripted',name:'Scripted lifecycle fixture',api:'openai-completions',reasoning:false,input:['text'],
      contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});
    await runtime.setRuntimeApiKey('lifecycle-fixture','fixture-only');
    const abort=new AbortController();
    const environment={instructions:'Local CPU fixture',shellOnTarget:scenario==='supervised-shell',prepare:async()=>{},observe:async()=>({state:'stopped'}),
      execute:async(_command,_cwd,options)=>{
        if(scenario==='unknown-submission')throw new Error('Submission response lost');
        state.event(attempt.id,'execution-handle',{handle:'fixture:verifier'});
        assert.ok(options.signal);
        setTimeout(()=>abort.abort(),10);
        return new Promise(resolve=>options.signal.addEventListener('abort',()=>{
          sawCancellation=true;resolve({handle:'fixture:verifier',settled:true,exitCode:130});
        },{once:true}));
      }};
    await assert.rejects(runAttempt({state,attempt,project:state.project('p').spec,attemptPath:join(root,'attempt'),
      environmentIdentity:{target:'local-test',runtime:'scripted'},
      signal:abort.signal,environmentFactory:async()=>environment,skillMap,
      knowledge:{search:async()=>({content:[],evidence:[]})},modelRuntime:runtime,
      model:runtime.getModel('lifecycle-fixture','scripted'),extensions:[pi=>pi.on('session_shutdown',async()=>{shutdown++;})]}));
    assert.ok(requests>0);assert.equal(shutdown,1);assert.equal(state.attempt(attempt.id).outcome,null);
    const context=state.events(attempt.id).find(event=>event.type==='skill-evidence-context');
    assert.ok(context && Object.values(context.data).every(value=>/^[a-f0-9]{64}$/.test(value)));
    const stopped=await attemptStopped(state,state.attempt(attempt.id),environment);
    if(scenario!=='unknown-submission') {
      assert.equal(sawCancellation,true);assert.equal(stopped,true);
      if(scenario==='supervised-shell') {
        assert.equal(state.events(attempt.id).some(e=>e.type==='native-bash-started'),false);
        assert.ok(state.events(attempt.id).some(e=>e.type==='execution-settled'));
      }
    }
    else {assert.equal(stopped,false);assert.ok(state.events(attempt.id).some(e=>e.type==='execution-unresolved' && e.data.handle===null));}
  }finally {
    server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
    state.close();await rm(root,{recursive:true,force:true});
  }
});
