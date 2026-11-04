import assert from 'node:assert/strict';
import {mkdir,readFile,writeFile,appendFile,cp,symlink,access} from 'node:fs/promises';
import {join,resolve,dirname} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {spawn,execFile} from 'node:child_process';
import {once} from 'node:events';
import {createServer} from 'node:http';
import {promisify} from 'node:util';
import {setTimeout as delay} from 'node:timers/promises';
import {createHash} from 'node:crypto';
import * as sdk from '@earendil-works/pi-coding-agent';
import {Blocked} from './runner.mjs';
import {schedule,target,serve,connect} from './fixtures.mjs';
import {bootstrap,readConfig} from '../../../cli/instance.mjs';
import {modelFromProfile} from '../../../controller/model.mjs';
import {State} from '../../../controller/state.mjs';
import {processIdentity,ownerStopped,attemptStopped} from '../../../controller/lifecycle.mjs';
import {recordOutcome} from '../../../controller/recovery.mjs';
import {runAttempt} from '../../../controller/pi.mjs';
import {stages} from '../../../controller/workflow.mjs';
import {createKdbBridge} from '../../../integrations/kdb-extension.mjs';
import {createEnvironment as genericEnvironment,createCplClient} from '../../../integrations/environments/index.mjs';

const repository=fileURLToPath(new URL('../../..',import.meta.url));
const json=async path=>JSON.parse(await readFile(path,'utf8'));
export const digest=async path=>createHash('sha256').update(await readFile(path)).digest('hex');
async function ensureKnowledgeService(config,options){
  const module=await import(pathToFileURL(join(config.root,'machinery/cli/knowledge.mjs')));
  return module.ensureKnowledgeService(config,options);
}
async function profileInput(path=process.env.RESILIENCE_LIVE_PROFILE){
  if(!path)throw new Blocked('Set RESILIENCE_LIVE_PROFILE to an ignored native DeepSeek profile');
  let profile;try{profile=await json(path);}catch{throw new Blocked('Live profile is not readable JSON');}
  if(!profile.python||profile.actor?.provider!=='deepseek')throw new Blocked('Live profile requires Python and native provider deepseek');
  for(const value of [profile.actor_seconds??300,profile.curriculum_seconds??180])if(!Number.isFinite(value)||value<=0)throw new Blocked('Live profile requires positive elapsed allocations');
  const actor={...profile.actor,max_tokens:Math.min(profile.actor.max_tokens??8192,8192)};
  if(actor.credentials_file)actor.credentials_file=resolve(dirname(path),actor.credentials_file);
  return {...profile,actor};
}
// Read-only preflight: modelFromProfile disables discovery/network refresh. Never return credentials/profile data.
export async function loadLiveProfile(path){
  const profile=await profileInput(path);let native;
  try{native=await modelFromProfile(profile.actor);}catch{throw new Blocked('Native model or credential unavailable; inspect the private live profile');}
  const m=native.model;
  return {provider:m.provider,model:m.id,api:m.api,contextWindow:m.contextWindow,maxTokens:m.maxTokens,
    actorSeconds:profile.actor_seconds??300,curriculumSeconds:profile.curriculum_seconds??180,credentialsAvailable:true};
}
export async function until(check,{signal,seconds=20}={}){
  const end=performance.now()+seconds*1000;
  while(performance.now()<end){signal?.throwIfAborted();const value=await check();if(value)return value;await delay(20,undefined,{signal});}
  throw Error('Live fixture milestone deadline expired');
}
// Existing retained backend injection; never used as a production transport.
export async function createEnvironment(options,context){
  const {fixtureUrl,...settings}=options;
  const native=fixtureUrl?connect(fixtureUrl):settings.mode==='cpl'?createCplClient(settings.clientPath):null;
  const client=native?((args,options={})=>native(args,{...options,signal:AbortSignal.any([context.signal,options.signal].filter(Boolean))})):null;
  return genericEnvironment(settings,{...context,resources:async()=>[{id:'live-cpu',available:true}],...(client?{client}:{})});
}
export async function runLiveActor({config,state,attempt,signal,extensions=[],environmentFactory,prepareSession,mcpUrl}){
  const path=join(config.root,'attempts',attempt.id);await mkdir(path,{recursive:true});
  const bridge=createKdbBridge({url:mcpUrl??await ensureKnowledgeService(config,{signal}),attemptId:attempt.id,agentDir:path,
    target:config.target,event:(type,data)=>state.event(attempt.id,type,data)});
  const directory=join(config.root,'skills/versions',config.skills.activeVersion),map=await json(join(directory,'skill-map.json'));
  for(const scope in map)map[scope]=resolve(directory,map[scope]);
  const factory=environmentFactory??(async context=>{
    const module=await import(pathToFileURL(resolve(config.root,config.environment.module)));
    return module.createEnvironment(config.environment.options,{...context,instancePath:path});
  });
  return runAttempt({state,attempt,project:state.project(attempt.project).spec,attemptPath:path,signal,skillMap:map,
    ...await modelFromProfile(config.actor),knowledge:{...bridge.knowledge,prepareSession:async(session,options)=>{
      await bridge.prepareSession(session,options);await prepareSession?.(session);
    }},extensions:[bridge.extension,...extensions],environmentFactory:factory});
}
export async function createLiveFixture(ctx,{environment,environmentFactory,actorSeconds,admit=true,goal,mcpDelayMs=0}={}){
  const profile=await profileInput();await loadLiveProfile();
  const seconds=actorSeconds??profile.actor_seconds??300;
  const root=join(ctx.directory,'instance'),source=join(ctx.directory,'input/source'),skills=join(ctx.directory,'skills');
  const observations=[],sessions=[],processes=[];let state,config,backend,proxy;const mcpTimeline=[];
  const save=(name,value)=>writeFile(join(ctx.directory,name),JSON.stringify(value,null,2)+'\n');
  async function command(command,{preload,signal=ctx.signal}={}){
      const args=[join(root,'machinery/cli/main.mjs'),command,'--instance',root],env={...process.env};
      if(preload)env.NODE_OPTIONS=[env.NODE_OPTIONS??'',`--import=${preload}`].join(' ');
      const child=spawn(process.execPath,args,{env,stdio:['ignore','pipe','pipe'],signal});processes.push(child);
      let stdout='',stderr='';child.stdout.on('data',d=>stdout+=d);child.stderr.on('data',d=>stderr+=d);
      const [code]=await once(child,'close');await appendFile(join(ctx.directory,'cli.jsonl'),JSON.stringify({command,code,stdout,stderr})+'\n');assert.equal(code,0);return JSON.parse(stdout.trim().split('\n').at(-1));
  }
  ctx.onCleanup(async cleanup=>{
    const failures=[],closed=[];
    if(config&&state)try{
      await command('stop',{signal:cleanup.signal});
      await until(async()=>{for(const attempt of state.db.prepare('SELECT owner FROM attempts').all()){
        const owner=JSON.parse(attempt.owner);if(owner?.pid!==process.pid&&!await ownerStopped(owner))return false;
      }return true;},{signal:cleanup.signal,seconds:10});
    }catch(error){failures.push(error.message);}
    for(const session of sessions)try{session.abortCompaction();await session.abort();await session.extensionRunner.emit({type:'session_shutdown',reason:'quit'});session.dispose();const file=session.sessionManager.getSessionFile();const rows=(await readFile(file,'utf8')).split('\n').filter(Boolean).map(JSON.parse);await writeFile(join(dirname(file),'sft.native.jsonl'),JSON.stringify({raw_session:file,messages:rows.filter(e=>e.type==='message').map(e=>e.message)})+'\n');}catch(error){failures.push(error.message);}
    if(state)for(const attempt of state.db.prepare('SELECT * FROM attempts').all()){
      const native=state.events(attempt.id),ended=new Set(native.filter(e=>e.type==='native-bash-completed').map(e=>e.data.operation));
      if(native.some(e=>e.type==='native-bash-started'&&!ended.has(e.data.operation)))try{
        const proof=await json(join(ctx.directory,'process-closure.json'));
        if(!proof.workerClosed||!proof.nativeGroupStopped||proof.errors.length)throw Error('Native Bash group closure unproven');
      }catch(error){failures.push('Unmatched native Bash: '+error.message);}
      let env;
      try{env=await factory({instancePath:join(root,'attempts',attempt.id),signal:cleanup.signal,event:()=>{},exclude:async(_,fn)=>fn()});}
      catch(error){failures.push(error.message);continue;}
      for(const handle of new Set(state.events(attempt.id).filter(e=>e.type==='execution-handle').map(e=>e.data.handle))){
        try{cleanup.signal?.throwIfAborted();let observed=await env.observe(handle);if(observed.state==='running')observed=await env.stop(handle);
          closed.push({handle,...observed});if(observed.state!=='stopped')failures.push('Unknown closure '+handle);
        }catch(error){closed.push({handle,state:'unknown',error:error.message});failures.push(error.message);}
      }
      if(attempt.session)try{const rows=(await readFile(attempt.session,'utf8')).split('\n').filter(Boolean).map(JSON.parse);
        await writeFile(join(dirname(attempt.session),'sft.native.jsonl'),JSON.stringify({raw_session:attempt.session,messages:rows.filter(r=>r.type==='message').map(r=>r.message)})+'\n');
      }catch(error){failures.push(error.message);}
    }
    for(const child of processes)if(child.exitCode===null&&child.signalCode===null){child.kill('SIGTERM');try{await until(()=>child.exitCode!==null||child.signalCode!==null,{signal:cleanup.signal,seconds:5});}catch(error){failures.push(error.message);}}
    try{const receipt=await json(join(root,'knowledge-service.json'));if(!await ownerStopped(receipt.owner)){
      process.kill(receipt.owner.pid,'SIGTERM');await until(()=>ownerStopped(receipt.owner),{signal:cleanup.signal,seconds:5});
    }}catch(error){if(error.code!=='ENOENT')failures.push(error.message);}
    try{if(proxy){proxy.closeAllConnections();await new Promise(ok=>proxy.close(ok));}await backend?.close();}catch(error){failures.push(error.message);}
    if(state){await save('state-events.json',state.db.prepare('SELECT * FROM events').all());state.close();}
    await save('closure.json',{closed,failures});assert.deepEqual(failures,[],'Inspect retained closure.json');
  });
  await mkdir(source,{recursive:true});await mkdir(skills);
  await writeFile(join(source,'solution.py'),'def solve(values):\n    return 0\n');
  const verifier=join(ctx.directory,'input/verify.py');
  await writeFile(verifier,"import runpy\nf=runpy.run_path('solution.py')['solve']\nfor xs,expected in [([2,3],5),([-4,2],-2),([],0),([42],42)]:\n assert f(xs)==expected,(xs,f(xs),expected)\nprint('four integer cases passed')\n");
  const map=Object.fromEntries([...stages,'global-kdb'].map(s=>[s,s+'.md']));
  for(const stage of stages)await writeFile(join(skills,map[stage]),
    `Stage ${stage}: read the source and preserve the frozen verifier. Native bash/read/write/edit are local; rsi_target executes a snapshot. Use relative paths on the target. `+
    `Retrieve good and negative evidence in design and optimisation. Run rsi_verify only in validation. Use rsi_revise if a stale or failed check requires repair. `+
    `Complete this stage with rsi_advance and a concise reflection. No reports are required. After passing verification do not edit source before final acceptance.\n`);
  await cp(join(repository,'ascend_tools/templates/kdb/SKILL.md'),join(skills,map['global-kdb']));
  await writeFile(join(skills,'skill-map.json'),JSON.stringify(map));
  for(const [name,text] of [['good','Integer addition gives2+3=5, negative inputs work and empty sum is0.'],['negative','A constant0 implementation fails [2,3]; an empty-only check is insufficient.']]){
    await mkdir(join(ctx.directory,name));await writeFile(join(ctx.directory,name,'fact.md'),text);
  }
  await promisify(execFile)(profile.python,['-c',"import sys;from ascend_tools.kdb import KDB;from kdb_fixture import Embeddings;k=KDB(sys.argv[1]+'/baseline.sqlite',Embeddings());k.add_repo(sys.argv[1]+'/good','positive','python');k.add_repo(sys.argv[1]+'/negative','negative','python',kind='negative');k.close()",ctx.directory],
    {signal:ctx.signal,env:{...process.env,PYTHONPATH:[repository,join(repository,'tests/ascend')].join(':')}});
  if(!environment){
    backend=await serve(await target({root:ctx.directory,scheduler:schedule(ctx)}));
    environment={module:fileURLToPath(import.meta.url),options:{mode:'cpl',target:'fixture',runtime:'declared',workspacePath:join(ctx.directory,'target'),remoteDirectory:'owned',fixtureUrl:backend.url,pollMs:10}};
  }
  environment={module:fileURLToPath(import.meta.url),...environment};
  config={...await bootstrap(root,{baseline:join(ctx.directory,'baseline.sqlite'),skills,machinery:repository,install:false,
    mode:'phase1',target:'live-python',python:profile.python,embedding_factory:'ascend_tools.live_embeddings:Embeddings',
    actor:profile.actor,curriculum_actor:profile.actor,curriculum_seconds:profile.curriculum_seconds??180,investigation_seconds:seconds,
    environment,projects:[{id:'numeric',goal:goal??'Fix solve(values) to return the exact sum of a list of integers. Use all nine stages; keep the independent four-case verifier unchanged. Keep work tiny and reflections concise.',
      source,actor_seconds:seconds,verifier:{files:[verifier],command:'PYTHONDONTWRITEBYTECODE=1 python ../verify.py'}}]}),root};
  await symlink(join(repository,'node_modules'),join(root,'machinery/node_modules'),'dir');
  await cp(join(repository,'tests/ascend/kdb_fixture.py'),join(root,'machinery/ascend_tools/live_embeddings.py'));
  state=new State(join(root,'state.sqlite'));let mcpUrl=await ensureKnowledgeService(config,{signal:ctx.signal});
  if(mcpDelayMs){
    const upstream=mcpUrl;proxy=createServer(async(req,res)=>{
      const abort=new AbortController();res.once('close',()=>abort.abort());const signal=AbortSignal.any([ctx.signal,abort.signal]);
      try{let body='';for await(const part of req)body+=part;
        if(req.method==='POST'&&JSON.parse(body).method==='tools/list'){
          mcpTimeline.push({type:'list-start',at:performance.now()});await delay(mcpDelayMs,undefined,{signal});mcpTimeline.push({type:'list-ready',at:performance.now()});
        }
        const headers={...req.headers};for(const key of ['host','connection','content-length','transfer-encoding'])delete headers[key];
        const response=await fetch(upstream,{method:req.method,headers,body:['GET','HEAD'].includes(req.method)?undefined:body,signal});
        const out=Object.fromEntries(response.headers);for(const key of ['content-length','content-encoding','transfer-encoding'])delete out[key];
        res.writeHead(response.status,out);res.end(Buffer.from(await response.arrayBuffer()));
      }catch(error){if(!res.destroyed){res.writeHead(503);res.end('Fixture discovery interrupted');}}
    });proxy.listen(0,'127.0.0.1');await once(proxy,'listening');mcpUrl=`http://127.0.0.1:${proxy.address().port}/mcp`;
  }
  async function factory(context){
    const base=environmentFactory??(async c=>(await import(pathToFileURL(resolve(root,config.environment.module)))).createEnvironment(config.environment.options,c));
    return base({...context,signal:context.signal??ctx.signal});
  }
  const fixture={ctx,config,state,project:state.project('numeric').spec,observations,sessions,save,factory,mcpTimeline,
    attempt:admit?state.admit('numeric',await processIdentity()):null,
    async run({attempt=fixture.attempt,project,signal=ctx.signal,extensions=[],prepareSession}={}){
      if(project)throw Error('Add a project through State.add before running it');
      return runLiveActor({config,state,attempt,signal,mcpUrl,environmentFactory:context=>factory({...context,instancePath:join(root,'attempts',attempt.id)}),
        prepareSession,extensions:[pi=>pi.on('tool_result',async event=>{
          observations.push({tool:event.toolName,error:event.isError,input:event.input});await appendFile(join(ctx.directory,'tools.jsonl'),JSON.stringify(event)+'\n');
        }),...extensions]});
    },
    verifyAccepted(attempt=fixture.attempt){
      const events=state.events(attempt.id);assert.equal(state.attempt(attempt.id).outcome,'passed');
      assert.ok(events.some(e=>e.type==='accepted'));assert.equal(events.findLast(e=>e.type==='verification').data.passed,true);
      for(const stage of stages)assert.ok(events.some(e=>e.type==='skill-read'&&e.data.stage===stage));
      for(const stage of ['design','optimisation'])for(const kind of ['good','negative'])assert.ok(events.some(e=>e.type==='knowledge-retrieved'&&e.data.stage===stage&&e.data.kind===kind));
      return {id:attempt.id,elapsed:state.attempt(attempt.id).elapsed,session:state.attempt(attempt.id).session};
    },
    async seedFailure(){
      const attempt=fixture.attempt??state.admit('numeric',await processIdentity());
      let text='';const result=await sdk.createLocalBashOperations().exec(config.projects[0].verifier.command,fixture.project.source,{signal:ctx.signal,onData:d=>{text+=d;}});
      assert.notEqual(result.exitCode,0);const path=join(ctx.directory,'baseline-negative.log');await writeFile(path,text);
      state.finish(attempt.id,'failed');recordOutcome(state,{attemptId:attempt.id,wave:0,passed:false,findings:'Independent integer verifier rejects constant0 on [2,3]. Investigate a smaller two-integer case.',evidence:[{path,sha256:await digest(path)}]});
      return state.attempt(attempt.id);
    },
    command,
    async openSession(){
      const path=join(root,'compact');await mkdir(path);const settings=sdk.SettingsManager.inMemory({retry:{enabled:false}});
      const bridge=createKdbBridge({url:mcpUrl,attemptId:'compaction',agentDir:path});
      const loader=new sdk.DefaultResourceLoader({cwd:fixture.project.source,agentDir:path,settingsManager:settings,noExtensions:true,noSkills:true,noContextFiles:true,extensionFactories:[bridge.extension,pi=>{for(const type of ['before_provider_request','provider_stream_event','after_provider_response'])pi.on(type,event=>appendFile(join(path,'provider.jsonl'),JSON.stringify(event)+'\n'));}]});await loader.reload();
      const {session}=await sdk.createAgentSession({cwd:fixture.project.source,agentDir:path,settingsManager:settings,...await modelFromProfile(config.actor),thinkingLevel:'off',resourceLoader:loader,sessionManager:sdk.SessionManager.create(fixture.project.source,path)});
      sessions.push(session);session.subscribe(event=>{void appendFile(join(path,'native-events.jsonl'),JSON.stringify(event)+'\n');});await bridge.prepareSession(session,{signal:ctx.signal});
      session.setActiveToolsByName(session.getActiveToolNames().filter(name=>!['mcp__kdb__record_lesson','mcp__kdb__add_repo'].includes(name)));return session;
    }};
  return fixture;
}
export async function constructionResume(f){
  await f.seedFailure();const marker=join(f.ctx.directory,'constructor-interruption.json'),preload=join(f.ctx.directory,'interrupt.mjs');
  await writeFile(preload,`import {AgentSession} from ${JSON.stringify(import.meta.resolve('@earendil-works/pi-coding-agent'))};\nimport {existsSync,writeFileSync} from 'node:fs';\nconst original=AgentSession.prototype.prompt;\nAgentSession.prototype.prompt=function(...args){if(this.getActiveToolNames().includes('submit_investigation'))this.subscribe(e=>{if(e.type==='tool_execution_end'&&e.toolName==='write'&&!e.isError&&!existsSync(${JSON.stringify(marker)})){writeFileSync(${JSON.stringify(marker)},JSON.stringify({session:this.sessionManager.getSessionFile()}),{flag:'wx'});process.emit('SIGTERM');}});return original.apply(this,args);};\n`);
  await f.command('wave',{preload});const child=f.state.db.prepare('SELECT * FROM attempts WHERE outcome IS NULL').get();assert.ok(child);
  assert.equal(child.elapsed,0);assert.equal(child.session,null);await assert.rejects(access(join(f.config.root,'attempts',child.id,'project.json')),{code:'ENOENT'});
  const before=f.state.events(child.id).find(e=>e.type==='curriculum-interrupted').data;
  assert.equal(await ownerStopped(child.owner),true);await f.command('resume',{preload});
  const sessions=f.state.events(child.id).filter(e=>e.type==='curriculum-session'&&e.data.role==='investigation');
  assert.equal(sessions.length,2);assert.equal(sessions[1].data.session,before.session);assert.equal(sessions[1].data.allocation_seconds,sessions[0].data.allocation_seconds);
  assert.ok(sessions[1].data.elapsed_seconds>=before.elapsed_seconds);return f.verifyAccepted(child);
}
