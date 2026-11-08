import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {ModelRuntime} from '@earendil-works/pi-coding-agent';
import {InMemoryCredentialStore} from '@earendil-works/pi-ai';
import {State} from '../../controller/state.mjs';
import {runAttempt} from '../../controller/pi.mjs';
import {validateTargetBaseline} from '../../controller/baseline-preflight.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {findingsFor} from '../../cli/findings.mjs';
import {reconcileFailure} from '../../controller/lifecycle.mjs';

const exec=promisify(execFile);
async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'baseline-preflight-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const source=join(root,'source');await mkdir(source);
  await writeFile(join(source,'answer.py'),'answer=42\n');
  const verifier=join(root,'verify.py');await writeFile(verifier,'assert True\n');
  const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  const project={id:'p',goal:'fused reduction',actor_seconds:120,source,performance_goal:'speedup',
    verifier:{files:[verifier],command:'correctness',performance_command:'performance',baseline_command:'baseline'}};
  state.add(project);const queue=new RecoveryQueue(state);
  const {attempt}=queue.claim('p',1,{pid:1,start:'previous',boot:'previous'});
  const attemptPath=join(root,'attempt');await mkdir(attemptPath);
  return {root,state,project,attempt,attemptPath,queue};
}

test('successful baseline opens an actor session and a resumed gate does not repeat execution',async t=>{
  const f=await fixture(t);let requests=0,checks=0;
  const server=createServer(async(req,res)=>{
    for await(const _chunk of req){}requests++;
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    res.end('data: '+JSON.stringify({id:'fixture',object:'chat.completion.chunk',created:1,model:'scripted',
      choices:[{index:0,delta:{role:'assistant',content:'Done'},finish_reason:'stop'}]})+'\n\ndata: [DONE]\n\n');
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');
  t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));});
  const runtime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,
    allowModelNetwork:false,refreshOnCreate:false});
  runtime.registerProvider('baseline-fixture',{baseUrl:`http://127.0.0.1:${server.address().port}/v1`,models:[{
    id:'scripted',name:'Baseline fixture',api:'openai-completions',reasoning:false,input:['text'],
    contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});
  await runtime.setRuntimeApiKey('baseline-fixture','fixture');
  await runAttempt({...f,skillMap:{},knowledge:{search:async()=>({content:[],evidence:[]})},environmentFactory:async()=>({instructions:'fixture',
    prepare:async()=>{},execute:async()=>{checks++;return {settled:true,exitCode:0};}}),
    modelRuntime:runtime,model:runtime.getModel('baseline-fixture','scripted')});
  assert.equal(checks,1);assert.equal(requests,1);assert.ok(f.state.attempt(f.attempt.id).session);
  await validateTargetBaseline({...f,checkVerifier:async()=>{},remaining:()=>60,register:async()=>{},
    execute:async()=>{throw new Error('A resumed successful baseline must not run again');}});
  assert.equal(f.state.events(f.attempt.id).filter(e=>e.type==='baseline-preflight-completed').length,1);
});

test('device-placement defect rejects construction before any model session and has no recovery loop',async t=>{
  const f=await fixture(t);let prepared=false;
  await assert.rejects(runAttempt({...f,skillMap:{},environmentFactory:async()=>({
    prepare:async()=>{prepared=true;},execute:async(command,cwd,options)=>{
      assert.equal(prepared,true);assert.equal(command,'baseline');assert.equal(cwd,f.project.source);
      assert.equal(options.purpose,'performance');options.onData(Buffer.from('baseline CPU/NPU accumulator mismatch'));
      return {settled:true,exitCode:1,handle:'fixture:baseline'};
    }}),modelRuntime:{getModel(){throw new Error('Model must never be requested');}}}),{code:'PROJECT_BASELINE_INVALID'});
  assert.equal(f.state.attempt(f.attempt.id).session,null);
  assert.equal(f.state.events(f.attempt.id).some(e=>e.type==='project-invalid'),true);
  assert.equal(f.state.events(f.attempt.id).some(e=>e.type==='execution-settled'),true);
  f.state.finish(f.attempt.id,'failed');
  assert.equal(await reconcileFailure(f.state,f.attempt.id,{observe:async()=>({state:'stopped'})}),true);
  // A deliberately unusable provider proves postprocessing uses the native
  // verdict/evidence directly instead of spending a summary/investigation call.
  const findings=await findingsFor({state:f.state,job:{attempt:f.attempt,kind:'original'},path:f.attemptPath,
    passed:false,config:{actor:{provider:'must-not-run'}}});
  assert.equal(findings.useful,false);assert.equal(findings.evidence.length,1);
  assert.equal(await readFile(findings.evidence[0].path,'utf8'),'baseline CPU/NPU accumulator mismatch');
  const result=f.queue.recordOriginal(f.attempt.id,{passed:false,...findings},1);
  assert.equal(result.frozen,true);assert.equal(result.next,null);
  assert.equal(f.state.project('p').frozen,true);assert.deepEqual(f.queue.jobs(2),[]);
});

test('transport uncertainty and cancellation never mark a project baseline invalid',async t=>{
  for(const mode of ['transport','unsettled','cancel','timeout','killed','refused','containment','health'])await t.test(mode,async t=>{
    const f=await fixture(t),abort=new AbortController();
    await assert.rejects(validateTargetBaseline({...f,signal:abort.signal,remaining:()=>60,
      register:async()=>{},checkVerifier:async()=>{},execute:async()=>{
        if(mode==='transport')throw new Error('Connection lost');
        if(mode==='cancel')abort.abort();
        return {settled:mode!=='unsettled',exitCode:mode==='timeout'?124:mode==='killed'?137:
          mode==='refused'?125:mode==='containment'?126:1,
          ...(mode==='health'?{outcome:{kind:'infrastructure',reason:'device-health-refused'}}:{})};
      }}));
    assert.equal(f.state.events(f.attempt.id).some(e=>e.type==='project-invalid'),false);
    assert.equal(f.state.project('p').frozen,false);
  });
});

test('real Torch baseline catches omitted device placement while CPU construction passes',async t=>{
  const f=await fixture(t),python=process.env.RSI_TEST_PYTHON??process.env.PYTHON??'python3';
  const baseline=join(f.root,'baseline.py');
  await writeFile(baseline,`import sys, torch
z=torch.ones((2,3),device=sys.argv[1])
total=torch.zeros((2,),dtype=z.dtype,**({'device':z.device} if sys.argv[2]=='fixed' else {}))
for k in range(z.shape[1]): total=total+z[:,k]
print('all baseline cases executed')
`);
  const cpu=await exec(python,[baseline,'cpu','broken']);assert.match(cpu.stdout,/executed/);
  for(const fixed of [false,true]) {
    await validateTargetBaseline({...f,remaining:()=>60,register:async()=>{},checkVerifier:async()=>{},
      execute:async(_command,_cwd,options)=>{
        try {const r=await exec(python,[baseline,'meta',fixed?'fixed':'broken']);options.onData(Buffer.from(r.stdout));
          return {settled:true,exitCode:0};
        }catch(e){options.onData(Buffer.from(e.stderr));return {settled:true,exitCode:e.code};}
      }}).then(()=>assert.equal(fixed,true),error=>{assert.equal(fixed,false);assert.equal(error.code,'PROJECT_BASELINE_INVALID');});
  }
  const checks=f.state.events(f.attempt.id).filter(e=>e.type==='baseline-preflight-completed');
  assert.deepEqual(checks.map(e=>e.data.passed),[false,true]);
});
