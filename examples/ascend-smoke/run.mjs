#!/usr/bin/env node
import assert from 'node:assert/strict';
import {readFile,mkdir,writeFile,cp,rm,symlink} from 'node:fs/promises';
import {resolve,join,dirname} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {setTimeout as delay} from 'node:timers/promises';
import {bootstrap,readConfig} from '../../cli/instance.mjs';
import {projectBundle} from '../../cli/project-bundle.mjs';
import {State} from '../../controller/state.mjs';
import {stages} from '../../controller/workflow.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {CampaignStorage} from '../../controller/storage.mjs';
import {processIdentity,ownerStopped,operationsStopped} from '../../controller/lifecycle.mjs';
import {modelFromProfile} from '../../controller/model.mjs';
import {createKdbBridge} from '../../integrations/kdb-extension.mjs';
import {provider} from '../../tests/ascend/driver_fixture.mjs';

const exec=promisify(execFile),repository=fileURLToPath(new URL('../..',import.meta.url));
const load=async path=>JSON.parse(await readFile(path,'utf8'));
const save=(path,data)=>writeFile(path,JSON.stringify(data,null,2)+'\n',{mode:0o600});
const machinery=(config,path)=>import(pathToFileURL(join(config.root,'machinery',path)));
const quote=text=>"'"+text.replaceAll("'","'\\''")+"'";
export const candidate="def run(x):\n    print('hello', x.device, flush=True)\n    return x + 1\n";
const reference='def run(x):\n    return x + 1\n';

export async function fixedProject(root,{python='python3',device='npu:0',actor=false}={}) {
  const bundle=join(root,'planned'),source=join(bundle,'work/items/source');
  const trusted=join(bundle,'trusted'),verifier=join(trusted,'checks/verify.py');
  const correct=join(trusted,'reference'),wrong=join(trusted,'wrong');
  for(const path of [source,dirname(verifier),correct,wrong])await mkdir(path,{recursive:true});
  await writeFile(join(source,'answer.py'),'def run(x):\n    raise NotImplementedError("Implement device hello world")\n');
  await writeFile(join(correct,'answer.py'),reference);
  await writeFile(join(wrong,'answer.py'),'def run(x):\n    return x\n');
  await cp(fileURLToPath(new URL('verify.py',import.meta.url)),verifier);
  const command=mode=>({entrypoint:'trusted/checks/verify.py',
    args:['--mode',mode,'--device',device,...(actor?['--single']:[])]});
  const packaged=await projectBundle({source,reference:correct,negatives:[wrong],files:[verifier]},
    `${quote(python)} -B ${quote(verifier)} --mode construction --device cpu`,180,root);
  return {id:actor?'actor-hello':'controller-planned',source,actor_seconds:actor?300:600,
    goal:actor?`Device hello world only: implement run(x) in answer.py using exactly this tiny PyTorch operation:\n${candidate}\nThe frozen verifier supplies a 16-element tensor on ${device}, synchronizes and checks every value. Follow the canonical workflow; do not research kernels, optimize, profile, benchmark, or generate a curriculum.`:
      'Fixed controller smoke: implement run(x) in answer.py as x + 1; execute frozen checks and accept. Timings test execution only, with no speedup target.',
    ...(!actor?{performance_goal:{kind:'measurement-only'}}:{}),bundle:packaged.bundle,
    verifier:{files:packaged.files,command:command('correctness'),
      ...(!actor?{baseline_command:command('baseline'),performance_command:command('performance')}:{})}};
}

function workflow() {
  const calls=[];
  for(const stage of stages) {
    calls.push({name:'rsi_skill',args:{}});
    if(['design','optimisation'].includes(stage))for(const kind of ['good','negative'])
      calls.push({name:'rsi_knowledge',args:{query:'tiny arithmetic correctness',kind}});
    if(stage==='implementation')calls.push({name:'write',args:{path:'answer.py',content:candidate}});
    if(stage==='validation')calls.push({name:'rsi_verify',args:{}});
    calls.push({name:'rsi_advance',args:{reflection:'Follow the fixed independent arithmetic oracle.'}});
  }
  return calls;
}

async function scriptedActor(root,instance) {
  const server=await provider('ascend-smoke',{workflowForGoal:workflow,nativeSession:async()=>{
    const state=new State(join(instance,'state.sqlite'));
    try{return state.db.prepare('SELECT session FROM attempts WHERE session IS NOT NULL LIMIT 1').get()?.session;}
    finally{state.close();}
  }});
  const credentials=join(root,'scripted.env');await writeFile(credentials,'SMOKE_FIXTURE_KEY=fixture-only\n',{mode:0o600});
  return {server,actor:{provider:'smoke-fixture',model:'scripted',credential_env:'SMOKE_FIXTURE_KEY',
    credentials_file:credentials,base_url:server.url,metadata:{id:'scripted',name:'Scripted smoke',
      api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,
      cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}}};
}

async function fixtures(root,python,signal) {
  const baseline=join(root,'baseline.sqlite'),skills=join(root,'skills');await mkdir(skills);
  const embeddings=join(repository,'tests/ascend/driver_fixture.py');
  await exec(python,[embeddings,'baseline',baseline],{signal,env:{...process.env,PYTHONPATH:repository}});
  const map=Object.fromEntries([...stages,'global-kdb'].map(stage=>[stage,stage+'.md']));
  for(const path of Object.values(map))await writeFile(join(skills,path),
    'Read this step. In design and optimisation retrieve good and negative evidence, briefly reflect, then advance. Implement only the stated tiny operation; no optimization research. Use frozen verification before acceptance.');
  await save(join(skills,'skill-map.json'),map);return {baseline,skills,embeddings};
}

async function bridgeFor(config,state,id,path,signal) {
  const {ensureKnowledgeService}=await machinery(config,'cli/knowledge.mjs');
  return createKdbBridge({url:await ensureKnowledgeService(config,{signal}),attemptId:id,agentDir:path,
    target:config.target,event:(type,data)=>state.event(null,type,data)});
}

async function actorOnly(config,state,signal) {
  const {runAttempt}=await machinery(config,'controller/pi.mjs');
  const {attempt,project}=new RecoveryQueue(state).claim('actor-hello',1,await processIdentity());
  const path=join(config.root,'attempts',attempt.id);await mkdir(path,{recursive:true});
  CampaignStorage.open(state,config.root).attempt(attempt.id,{phase:'phase1',kind:'original',wave:1});
  const bridge=await bridgeFor(config,state,attempt.id,path,signal);
  const skillsRoot=join(config.root,'skills/versions/baseline'),map=await load(join(skillsRoot,'skill-map.json'));
  for(const key of Object.keys(map))map[key]=join(skillsRoot,map[key]);
  const module=await import(pathToFileURL(resolve(config.root,config.environment.module)));
  let environment;
  const result=await runAttempt({state,attempt,project:project.spec,attemptPath:path,signal,
    environmentIdentity:config.environment,knowledge:bridge.knowledge,extensions:[bridge.extension],skillMap:map,
    ...await modelFromProfile(config.actor),environmentFactory:async context=>environment=await module.createEnvironment(
      {...config.environment.options,localExecution:config.local_execution},context)});
  assert.equal(await operationsStopped(state,attempt,environment),true,'Actor execution must positively close');
  if(environment.reconcileLease)assert.equal(await environment.reconcileLease(),true);
  return result;
}

async function curriculumOnly(config,state,root,signal) {
  const {refillProjects}=await machinery(config,'cli/project-generation.mjs');
  const {importProject}=await machinery(config,'cli/instance.mjs');
  const path=join(config.root,'generation');await mkdir(path);
  const bridge=await bridgeFor(config,state,'smoke-generation',path,signal);
  const generated=await refillProjects({config,state,bridge,concurrency:1,path,signal});
  assert.equal(generated.length,1,'One planned curriculum topic must produce exactly one project');
  const imported=await importProject(config.root,generated[0],{signal,localExecution:config.local_execution});
  assert.ok(generated[0].generation.checks.some(check=>check.kind==='reference'&&check.exitCode===0));
  assert.ok(generated[0].generation.checks.some(check=>check.kind==='negative'&&check.exitCode===1));
  assert.equal(state.db.prepare('SELECT count(*) AS n FROM attempts').get().n,0);
  await save(join(root,'curriculum-project.json'),imported);return {project:imported.id};
}

async function stopKnowledge(instance) {
  let receipt;try{receipt=await load(join(instance,'knowledge-service.json'));}
  catch(error){if(error.code==='ENOENT')return;throw error;}
  if(await ownerStopped(receipt.owner))return;
  try{process.kill(receipt.owner.pid,'SIGTERM');}
  catch(error){if(error.code!=='ESRCH')throw error;}
  const deadline=performance.now()+15000;
  while(!await ownerStopped(receipt.owner)) {
    if(performance.now()>=deadline)throw Error('Smoke knowledge service did not stop');
    await delay(100);
  }
}

export async function runSmoke({kind,profile,destination,live=false,device='npu:0',signal}) {
  assert.ok(['controller','curriculum','actor'].includes(kind));
  if(kind!=='controller'&&!live)throw Error('Real curriculum/actor smoke requires explicit live authorization');
  const cpu=device==='cpu',target=cpu?'cpu':profile.target??profile.environment?.options?.health?.product;
  if(!cpu) {
    assert.ok(['a3','a5'].includes(target),'An explicit A3/A5 target binding is required');
    assert.equal(profile.environment?.options?.mode,target==='a3'?'cpl':'local',
      'A3 uses the host CPL path; A5 must execute directly on its device host');
    assert.equal(profile.environment?.module,'machinery/integrations/environments/ascend.mjs',
      'Device smokes require the copied Ascend adapter for health, masks and admission');
  }
  const localExecution=cpu?profile.local_execution:{tasks:profile.local_execution?.tasks??128,
    memory_bytes:profile.local_execution?.memory_bytes??4*1024**3};
  if(!cpu)assert.ok(Number.isSafeInteger(localExecution.tasks)&&localExecution.tasks>=4&&localExecution.tasks<=128&&
    Number.isSafeInteger(localExecution.memory_bytes)&&localExecution.memory_bytes>=1024**2&&localExecution.memory_bytes<=4*1024**3,
    'Device smokes require containment of at most 128 tasks and 4 GiB; no uncontained fallback');
  const root=resolve(destination),instance=join(root,'instance'),python=profile.python??'python3';
  await mkdir(root,{recursive:false,mode:0o700});
  const outer=AbortSignal.timeout(900000),active=signal?AbortSignal.any([signal,outer]):outer;
  let scripted,result,initialized=false;
  try {
    const fixture=await fixtures(root,python,active),projects=kind==='curriculum'?[]:
      [await fixedProject(root,{python,device,actor:kind==='actor'})];
    if(kind==='controller')scripted=await scriptedActor(root,instance);
    const curriculum={phase1:projects.map(project=>project.id),generation:{enabled:kind==='curriculum',
      topics:['tiny-pytorch'],preprompt:'Construct only run(x)=x+1 for CPU float32 tensors with shapes (1,) and (17,). One independent reference, one wrong control, a verifier, and a straightforward PyTorch starter. No performance goal, device commands, kernel code, optimization, or additional topics.'}};
    const settings={...fixture,python,target,projects,install:false,mode:'phase1',recovery:false,concurrency:1,
      actor:scripted?.actor??profile.actor,curriculum_actor:kind==='curriculum'?(profile.curriculum_actor??profile.actor):undefined,
      actor_seconds:300,curriculum_seconds:180,embedding_factory:'ascend_tools.fixture_embeddings:Embeddings',
      environment:profile.environment,local_execution:localExecution,curriculum,
      skillSettings:{background:false},...(kind==='curriculum'?{campaign_plan:[{id:'curriculum-smoke',phase:'phase1',
        new_projects:1,actor_seconds:300,generation:{topics:['tiny-pytorch'],requirements:{}}}]}:{})};
    delete settings.embeddings;
    await bootstrap(instance,{...settings,signal:active});initialized=true;
    await symlink(join(repository,'node_modules'),join(instance,'machinery/node_modules'),'dir');
    await cp(fixture.embeddings,join(instance,'machinery/ascend_tools/fixture_embeddings.py'));
    if(projects.length)await rm(join(root,'planned'),{recursive:true});
    const config=await readConfig(instance);
    if(kind==='controller') {
      const {main}=await machinery(config,'cli/main.mjs');
      await main(['wave','--instance',instance,'--mode','phase1'],{signal:active});
    }
    const state=new State(join(instance,'state.sqlite'));
    try {
      if(kind==='actor')await actorOnly(config,state,active);
      if(kind==='curriculum')result=await curriculumOnly(config,state,root,active);
      else {
        const attempts=state.db.prepare('SELECT id,outcome,session FROM attempts').all();
        assert.equal(attempts.length,1);assert.equal(attempts[0].outcome,'passed');assert.ok(attempts[0].session);
        const events=state.events(attempts[0].id);assert.ok(events.some(e=>e.type==='accepted'));
        assert.ok(events.some(e=>e.type==='verification'&&e.data.passed));
        const table=state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='generation_drafts'").get();
        assert.equal(table?state.db.prepare('SELECT count(*) AS n FROM generation_drafts').get().n:0,0);
        assert.equal(state.db.prepare("SELECT count(*) AS n FROM events WHERE type='curriculum-session' AND json_extract(data,'$.role')='generation'").get().n,0);
        if(kind==='controller') {
          const baseline=events.find(e=>e.type==='baseline-preflight-completed'&&e.data.passed);
          assert.ok(baseline);
          const outputs=(await readFile(baseline.data.evidence.path,'utf8')).split('\n').flatMap(line=>{
            try{return [JSON.parse(line)];}catch{return [];}
          });
          assert.ok(outputs.some(value=>value.mode==='baseline'&&value.device===device&&value.negativeRejected===3));
        }
        if(kind==='controller') {
          assert.equal(state.db.prepare('SELECT count(*) AS n FROM resource_leases').get().n,0);
          if(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='requests'").get())for(;;) {
            active.throwIfAborted();
            const pending=state.db.prepare("SELECT status,owner FROM requests WHERE kind='storage'").all();
            const stopped=await Promise.all(pending.map(async request=>!['queued','running'].includes(request.status)&&
              (!request.owner||await ownerStopped(request.owner))));
            if(stopped.every(Boolean))break;await delay(100);
          }
        }
        result={attempt:attempts[0].id,accepted:true,generationCalls:0};
      }
    }finally{state.close();}
    assert.deepEqual(scripted?.server.failures??[],[]);
  }finally {
    if(initialized)await stopKnowledge(instance);
    await scripted?.server.close();
  }
  const report={kind,target,device:kind==='curriculum'?null:device,
    deviceEvidence:kind!=='curriculum'&&device!=='cpu',passed:true,instance,...result,cleanupClosed:true};
  await save(join(root,'result.json'),report);return report;
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  const [kind,profileFile,destination]=process.argv.slice(2);
  if(!kind||!profileFile||!destination)throw Error('Usage: run.mjs controller|curriculum|actor PRIVATE_PROFILE_JSON NEW_OUTPUT_DIRECTORY');
  const abort=new AbortController(),stop=()=>abort.abort();process.on('SIGTERM',stop);process.on('SIGINT',stop);
  try{console.log(JSON.stringify(await runSmoke({kind,profile:await load(profileFile),destination,
    live:process.env.RSI_ASCEND_SMOKE_LIVE==='1',signal:abort.signal})));}
  finally{process.off('SIGTERM',stop);process.off('SIGINT',stop);}
}
