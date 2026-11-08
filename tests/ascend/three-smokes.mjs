import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,rm,access} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,dirname,resolve} from 'node:path';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {DatabaseSync} from 'node:sqlite';
import {runSmoke,candidate} from '../../examples/ascend-smoke/run.mjs';
import {provider} from './driver_fixture.mjs';
import {stages} from '../../controller/workflow.mjs';

const python=process.env.RSI_TEST_PYTHON??process.env.DRIVER_PYTHON??'python3';
const quote=text=>"'"+text.replaceAll("'","'\\''")+"'";
const environment={module:'machinery/integrations/environments/index.mjs',
  options:{mode:'local',env:{PATH:dirname(python)+':'+process.env.PATH}}};
async function root(t) {
  const path=await mkdtemp(join(process.env.DRIVER_ARTIFACTS??tmpdir(),'three-smokes-'));
  t.after(()=>{t.diagnostic('Smoke artifacts: '+path);});return path;
}
async function actorProfile(path,url) {
  const credentials_file=join(path,'credentials.env');await writeFile(credentials_file,'SMOKE_KEY=fixture\n');
  return {provider:'three-smokes',model:'scripted',credential_env:'SMOKE_KEY',credentials_file,base_url:url,
    metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],
      contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
}

test('device smoke rejects relaxed containment before creating an instance or contacting a provider',async t=>{
  const path=await root(t),destination=join(path,'must-not-start');
  await assert.rejects(runSmoke({kind:'controller',profile:{target:'a5',
    environment:{...environment,module:'machinery/integrations/environments/ascend.mjs',
      options:{...environment.options,health:{product:'a5'}}},
    local_execution:{tasks:129,memory_bytes:4*1024**3}},destination}),/require containment/);
  await assert.rejects(access(destination),{code:'ENOENT'});
  await rm(path,{recursive:true});
});

test('device smoke rejects a generic backend before creating any instance',async t=>{
  const path=await root(t),destination=join(path,'must-not-start');
  await assert.rejects(runSmoke({kind:'controller',profile:{target:'a5',
    environment:{...environment,options:{...environment.options,health:{product:'a5'}}}},destination}),/copied Ascend adapter/);
  await assert.rejects(access(destination),{code:'ENOENT'});
  await rm(path,{recursive:true});
});

test('device smoke rejects a target/product mismatch before model access or workspace creation',async t=>{
  const path=await root(t);
  for(const [target,product] of [['a3','a5'],['a5','a3']]) {
    const destination=join(path,target);let providerReads=0;
    const profile={target,environment:{module:'machinery/integrations/environments/ascend.mjs',
      options:{mode:target==='a3'?'cpl':'local',health:{product}}},
      get actor(){providerReads++;throw new Error('Provider must not be accessed');}};
    await assert.rejects(runSmoke({kind:'actor',profile,destination,live:true}),/match the Ascend health product/);
    assert.equal(providerReads,0);
    await assert.rejects(access(destination),{code:'ENOENT'});
  }
  await rm(path,{recursive:true});
});

test('fixed controller smoke executes imported commands and full native Pi acceptance without generation',
  {timeout:180000},async t=>{
    const path=await root(t),destination=join(path,'controller');
    const report=await runSmoke({kind:'controller',profile:{python,environment},destination,device:'cpu'});
    assert.equal(report.accepted,true);assert.equal(report.generationCalls,0);assert.equal(report.cleanupClosed,true);
    await assert.rejects(access(join(destination,'planned')),{code:'ENOENT'});
    const config=JSON.parse(await readFile(join(report.instance,'instance.json')));
    assert.equal(config.curriculum.generation.enabled,false);assert.equal(config.skills.background,false);
    assert.equal(config.concurrency,1);assert.ok(config.projects[0].verifier.baseline_command);
    if(!process.env.DRIVER_ARTIFACTS)await rm(path,{recursive:true});
  });

test('actor hello smoke uses actual Pi tools and verifier with no curriculum or performance goal',
  {timeout:180000},async t=>{
    const path=await root(t),server=await provider('hello',{workflowForGoal:()=>{
      const calls=[];for(const stage of stages){calls.push({name:'rsi_skill',args:{}});
        if(['design','optimisation'].includes(stage))for(const kind of ['good','negative'])
          calls.push({name:'rsi_knowledge',args:{query:'tiny hello',kind}});
        if(stage==='implementation')calls.push({name:'write',args:{path:'answer.py',content:candidate}},
          {name:'bash',args:{command:'printf smoke-native-bash',timeout:10}});
        if(stage==='validation')calls.push({name:'rsi_verify',args:{}});
        calls.push({name:'rsi_advance',args:{reflection:'Independent hello oracle.'}});}return calls;
    }});t.after(()=>server.close());
    const report=await runSmoke({kind:'actor',profile:{python,environment,actor:await actorProfile(path,server.url)},
      destination:join(path,'actor'),device:'cpu',live:true});
    assert.equal(report.accepted,true);assert.equal(report.generationCalls,0);assert.equal(report.cleanupClosed,true);
    assert.ok(server.requests.length);assert.deepEqual(server.failures,[]);
    const config=JSON.parse(await readFile(join(report.instance,'instance.json')));
    assert.equal(config.projects[0].performance_goal,undefined);
    assert.equal(config.projects[0].verifier.baseline_command,undefined);
    const state=new DatabaseSync(join(report.instance,'state.sqlite'),{readOnly:true});
    try {assert.equal(JSON.parse(state.prepare("SELECT data FROM events WHERE type='verification'").get().data).performance,null);
      const completed=JSON.parse(state.prepare("SELECT data FROM events WHERE type='native-bash-completed'").get().data);
      const attempt=state.prepare('SELECT elapsed,outcome FROM attempts').get();
      assert.ok(completed.actor_elapsed>0);assert.ok(completed.actor_elapsed<attempt.elapsed);
      assert.equal(attempt.outcome,'passed');}
    finally{state.close();}
    if(!process.env.DRIVER_ARTIFACTS)await rm(path,{recursive:true});
  });

test('curriculum smoke validates and imports exactly one real-tool PyTorch project without actor execution',
  {timeout:180000},async t=>{
    const path=await root(t),calls=[['write',{path:'source/answer.py',content:'def run(x): return x + 1\n'}],
      ['write',{path:'reference/answer.py',content:'def run(x): return x + 1\n'}],
      ['write',{path:'wrong/answer.py',content:'def run(x): return x - 1\n'}],
      ['write',{path:'verify.py',content:"import importlib.util, pathlib, torch\ns=importlib.util.spec_from_file_location('a',pathlib.Path.cwd()/'answer.py')\na=importlib.util.module_from_spec(s);s.loader.exec_module(a)\nx=torch.arange(17,dtype=torch.float32)\nassert torch.equal(a.run(x),torch.tensor(list(range(1,18)),dtype=torch.float32))\n"}],
      ['submit_project',{goal:'Tiny PyTorch x + 1 CPU contract',source:'source',reference:'reference',negative_controls:['wrong'],
        verifier:{files:['verify.py'],command:{entrypoint:'verify.py',args:[]}}}]];
    let requests=0;const server=createServer(async(req,res)=>{
      let text='';for await(const chunk of req)text+=chunk;
      const body=JSON.parse(text),n=body.messages.filter(message=>message.role==='tool').length;requests++;
      const call=calls[n],delta=call?{role:'assistant',tool_calls:[{index:0,id:'call-'+n,type:'function',
        function:{name:call[0],arguments:JSON.stringify(call[1])}}]}:{role:'assistant',content:'Done'};
      res.writeHead(200,{'Content-Type':'text/event-stream'});
      for(const [value,finish_reason] of [[delta,null],[{},call?'tool_calls':'stop']])res.write('data: '+JSON.stringify({
        id:'smoke',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:value,finish_reason}]})+'\n\n');
      res.end('data: [DONE]\n\n');
    });server.listen(0,'127.0.0.1');await once(server,'listening');
    t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));});
    const report=await runSmoke({kind:'curriculum',profile:{python,target:'a3',
      environment:{module:'machinery/integrations/environments/ascend.mjs',options:{mode:'cpl',target:'unused-device-alias',health:{product:'a3'}}},
      actor:await actorProfile(path,`http://127.0.0.1:${server.address().port}/v1`)},
      destination:join(path,'curriculum'),live:true});
    assert.equal(report.passed,true);assert.ok(report.project);assert.ok(requests>=calls.length);
    const imported=JSON.parse(await readFile(join(path,'curriculum/curriculum-project.json')));
    assert.ok(imported.source.startsWith(resolve(report.instance)));assert.ok(imported.bundle);
    assert.equal(imported.target,'cpu');assert.equal(report.target,'cpu');assert.equal(report.deviceEvidence,false);
    const config=JSON.parse(await readFile(join(report.instance,'instance.json')));
    assert.equal(config.curriculum_seconds,180);assert.equal(config.target,'cpu');
    assert.equal(config.environment.module,'machinery/integrations/environments/index.mjs');
    assert.equal(config.environment.options.mode,'local');
    assert.deepEqual(imported.generation.checks.map(check=>check.exitCode),[0,1,0]);
    if(!process.env.DRIVER_ARTIFACTS)await rm(path,{recursive:true});
  });
