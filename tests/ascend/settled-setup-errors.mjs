import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {setTimeout as delay} from 'node:timers/promises';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {ModelRuntime} from '@earendil-works/pi-coding-agent';
import {InMemoryCredentialStore} from '@earendil-works/pi-ai';
import {State} from '../../controller/state.mjs';
import {runAttempt} from '../../controller/pi.mjs';
import {stages} from '../../controller/workflow.mjs';
import {processIdentity,attemptStopped} from '../../controller/lifecycle.mjs';
import {createEnvironment,closedSetupError} from '../../integrations/environments/index.mjs';

for(const scenario of ['closed','not-admitted','unknown','unproved-code','unsettled-payload','closed-at-stop','closed-at-budget',
  'closed-ssh-upload','unknown-ssh-upload','failed-ssh-download'])
test(`native Pi setup error: ${scenario}`,{timeout:20000},async()=>{
  const root=await mkdtemp(join(tmpdir(),'settled-setup-')),source=join(root,'source');
  await mkdir(source);
  const state=new State(join(root,'state.sqlite')),calls=[],requests=[];
  const known=scenario.startsWith('closed')||scenario==='not-admitted';
  const mayContinue=['closed','not-admitted','closed-ssh-upload'].includes(scenario),abort=new AbortController();
  const command='python3 -c "import runpy; assert runpy.run_path(\'answer.py\')[\'answer\']() == 42"';
  for(const stage of stages){
    calls.push(['rsi_skill',{}]);
    if(['design','optimisation'].includes(stage))for(const kind of ['good','negative'])
      calls.push(['rsi_knowledge',{query:'arithmetic evidence',kind}]);
    if(stage==='implementation')calls.push(['write',{path:'answer.py',content:'def answer(): return 6 * 7\n'}],
      ['rsi_target',{command}],['rsi_target',{command}]);
    if(stage==='validation')calls.push(['rsi_verify',{}]);
    calls.push(['rsi_advance',{reflection:'Use independent arithmetic evidence.'}]);
  }
  const server=createServer(async(req,res)=>{
    let body='';for await(const chunk of req)body+=chunk;
    const input=JSON.parse(body);requests.push(input);
    const count=input.messages.reduce((n,m)=>n+(m.tool_calls?.length??0),0),call=calls[count];
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>'data: '+JSON.stringify({id:'setup-fixture',object:'chat.completion.chunk',
      created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n';
    const delta=call?{role:'assistant',tool_calls:[{index:0,id:'call-'+count,type:'function',
      function:{name:call[0],arguments:JSON.stringify(call[1])}}]}:{role:'assistant',content:'Done.'};
    res.end(frame(delta)+frame({},call?'tool_calls':'stop')+'data: [DONE]\n\n');
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');
  let backend,executions=0,shutdown=0,failedHandle,sessionAtFailure;
  try{
    await writeFile(join(source,'answer.py'),'def answer(): return 0\n');
    const verifier=join(root,'verify.py');
    const verifierSource="import runpy\nactual=runpy.run_path('answer.py')['answer']()\nassert actual==sum(6 for _ in range(7)),actual\nprint(actual)\n";
    await writeFile(verifier,verifierSource);
    const skillMap={};for(const stage of [...stages,'global-kdb']){
      skillMap[stage]=join(root,stage+'.md');await writeFile(skillMap[stage],'Use independent evidence.');
    }
    state.add({id:'p',goal:'Compute the independent arithmetic value',source,actor_seconds:scenario==='closed-at-budget'?10:30,
      verifier:{files:[verifier],command:'python3 ../verify.py'}});
    const attempt=state.admit('p',{...await processIdentity(),start:'previous-owner'});
    state.checkpoint(attempt.id,2);
    const runtime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,
      allowModelNetwork:false,refreshOnCreate:false});
    runtime.registerProvider('setup-fixture',{baseUrl:`http://127.0.0.1:${server.address().port}/v1`,models:[{
      id:'scripted',name:'Scripted setup fixture',api:'openai-completions',reasoning:false,input:['text'],
      contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});
    await runtime.setRuntimeApiKey('setup-fixture','fixture-only');
    const promise=runAttempt({state,attempt:state.attempt(attempt.id),project:state.project('p').spec,
      attemptPath:join(root,'attempt'),skillMap,signal:abort.signal,modelRuntime:runtime,model:runtime.getModel('setup-fixture','scripted'),
      knowledge:{search:async()=>({content:[],evidence:[]})},
      extensions:[pi=>pi.on('session_shutdown',async()=>{shutdown++;})],
      environmentFactory:async context=>{
        backend=await createEnvironment({mode:'local'},context);
        return {...backend,execute:async(command,cwd,options)=>{
          if(++executions===1){
            sessionAtFailure=state.attempt(attempt.id).session;
            if(scenario==='unsettled-payload')return {handle:'fixture:running',settled:false};
            if(['closed-ssh-upload','unknown-ssh-upload','failed-ssh-download'].includes(scenario))
              throw closedSetupError('SSH transfer setup failed',{reason:{backend:'ssh',
                operation:scenario==='failed-ssh-download'?'download':'upload',
                state:scenario==='unknown-ssh-upload'?'unknown':'failed',exit:23}});
            if(!known)throw Object.assign(new Error('Submission identity unknown'),
              scenario==='unproved-code'?{code:'CPL_NOT_ADMITTED'}:{});
            if(scenario==='not-admitted'){
              await context.exclude('setup',async()=>{
                context.event('execution-submitting',{submission:'denied',kind:'run',accountingExcluded:true});
                context.event('execution-not-admitted',{submission:'denied',closed:true});
              });
              throw closedSetupError('Setup was not admitted',{notAdmitted:true,reason:{state:'rejected'}});
            }
            const failed=await context.exclude('setup',()=>backend.execute('python3 -c "raise SystemExit(1)"',cwd,options));
            assert.equal(failed.settled,true);assert.equal(failed.exitCode,1);failedHandle=failed.handle;
            if(scenario==='closed-at-stop')abort.abort();
            if(scenario==='closed-at-budget')await delay(options.timeout*1000+25);
            throw closedSetupError(`Setup failed: ${failed.handle}`,{handle:failed.handle,reason:failed});
          }
          return backend.execute(command,cwd,options);
        }};
      }});
    if(mayContinue){
      const result=await promise;assert.equal(result.accepted,true);
      assert.equal(executions,3); // Failed setup, model-requested target retry, frozen verifier.
      assert.equal(state.attempt(attempt.id).outcome,'passed');
      assert.equal(state.events(attempt.id).filter(e=>e.type==='accepted').length,1);
      assert.ok(state.events(attempt.id).some(e=>e.type==='verification'&&e.data.passed));
      const settled=state.events(attempt.id).find(e=>e.type==='execution-settled'&&e.data.setupClosure);
      assert.equal(settled.data.setupClosure.payloadStarted,false);
      assert.equal(settled.data.handle,failedHandle??null);
      if(scenario==='closed-ssh-upload') {
        assert.deepEqual(settled.data.setupClosure.handles,[]);
        assert.equal(settled.data.setupClosure.reason.operation,'upload');
        assert.equal(settled.data.setupClosure.reason.exit,23);
      }
      assert.ok(settled.data.actor_elapsed>=settled.data.startElapsed);
      assert.equal(state.events(attempt.id).some(e=>e.type==='execution-unresolved'),false);
      if(failedHandle){
        assert.equal((await backend.observe(failedHandle)).state,'stopped');
        assert.ok(requests.some(r=>r.messages.some(m=>m.role==='tool'&&String(m.content).includes(failedHandle))));
      }
      assert.equal(await attemptStopped(state,state.attempt(attempt.id),backend),true);
    }else if(known){
      if(scenario==='closed-at-stop'){
        await assert.rejects(promise,/Attempt interrupted/);
        assert.equal(state.attempt(attempt.id).outcome,null);
      }else{
        const result=await promise;assert.equal(result.accepted,false);
        assert.equal(state.attempt(attempt.id).outcome,'failed');
      }
      assert.equal(executions,1);
      assert.equal(state.events(attempt.id).some(e=>e.type==='accepted'||e.type==='execution-unresolved'),false);
      assert.ok(state.events(attempt.id).some(e=>e.type==='execution-settled'&&e.data.setupClosure));
      assert.equal(await attemptStopped(state,state.attempt(attempt.id),backend),true);
    }else{
      await assert.rejects(promise,scenario==='unsettled-payload'?/Execution closure unknown/:
        scenario.includes('ssh')?/SSH transfer setup failed/:/Submission identity unknown/);
      assert.equal(executions,1);assert.equal(state.attempt(attempt.id).outcome,null);
      assert.ok(state.events(attempt.id).some(e=>e.type==='execution-unresolved'));
      assert.equal(await attemptStopped(state,state.attempt(attempt.id),backend),false);
    }
    assert.ok(sessionAtFailure);assert.equal(state.attempt(attempt.id).session,sessionAtFailure);
    assert.ok(state.attempt(attempt.id).elapsed>=2);assert.equal(shutdown,1);
    assert.equal(state.db.prepare('SELECT count(*) AS n FROM attempts').get().n,1);
    assert.equal(await readFile(verifier,'utf8'),verifierSource);
  }finally{
    server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
    state.close();await rm(root,{recursive:true,force:true});
  }
});
