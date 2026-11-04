import assert from 'node:assert/strict';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {once} from 'node:events';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';
import * as sdk from '@earendil-works/pi-coding-agent';
import {ActorClock} from '../../../controller/budget.mjs';
import {State} from '../../../controller/state.mjs';
import {attemptStopped,processIdentity} from '../../../controller/lifecycle.mjs';
import {runAttempt} from '../../../controller/pi.mjs';
import {stages} from '../../../controller/workflow.mjs';
import {createKdbBridge} from '../../../integrations/kdb-extension.mjs';
import {createEnvironment,readProcessStat} from '../../../integrations/environments/index.mjs';
import {schedule,target,serve,connect} from './fixtures.mjs';
import {listen,provider,runtime,until} from './lifecycle-helpers.mjs';

async function childContext(ctx,name){const directory=join(ctx.directory,name);await mkdir(directory);
  return {...ctx,directory,event:value=>ctx.event({scenario:name,...value})};}
async function closed(backend,environment,ctx,useSignal=()=>{}) {
  ctx.onCleanup(async({signal})=>{
    useSignal(signal);const observations=[],errors=[];
    for(const handle of backend.jobs.keys()) {
      try {signal.throwIfAborted();let state=await environment.observe(handle);
        if(state.state==='running')state=await environment.stop(handle);
        observations.push({handle,...state});if(state.state!=='stopped')errors.push(handle);
      }catch(error){observations.push({handle,state:'unknown',error:error.message});errors.push(handle);}
    }
    await writeFile(join(ctx.directory,'closure.json'),JSON.stringify(observations,null,2));
    assert.deepEqual(errors,[],'Every owned operation must have closure proof');
  });
}
async function accounting(ctx,scheduler) {
  const source=join(ctx.directory,'source');await mkdir(source);await writeFile(join(source,'value.py'),'value=42\n');
  const backend=await target({root:ctx.directory,scheduler});ctx.onCleanup(()=>backend.close());
  const clock=new ActorClock(120,3.5),excluded=[];let clientSignal=ctx.signal;
  const client=(args,settings)=>backend.client(args,{...settings,signal:clientSignal});
  const environment=await createEnvironment({mode:'cpl',target:'fixture',runtime:'declared',
    workspacePath:backend.workspace,remoteDirectory:'owned',pollMs:5},{sdk,client,
    instancePath:ctx.directory,event:(type,data)=>ctx.event({type,data}),exclude:(label,fn)=>clock.exclude(label,async()=>{
      const frozen=clock.elapsed,started=performance.now();
      try{return await fn();}finally{assert.equal(clock.elapsed,frozen);excluded.push({label,seconds:(performance.now()-started)/1000});}
    })});
  await closed(backend,environment,ctx,signal=>{clientSignal=signal;});
  await environment.prepare({project:{id:'accounting',source},attempt:{id:'clock'}});
  const thinkingBefore=clock.elapsed;await scheduler.at('thinking',{},ctx.signal);
  assert.ok(clock.elapsed>thinkingBefore);
  const executions=[];
  for(const [purpose,command] of [
    ['compile',"import py_compile,time; py_compile.compile('value.py',doraise=True); time.sleep(0.04)"],
    ['execute',"import time; assert sum(6 for _ in range(7))==42; time.sleep(0.04)"],
    ['performance',"import cProfile,time; cProfile.run('sum(range(10000))'); time.sleep(0.04)"],
  ]) {
    const before=clock.elapsed;
    const result=await environment.execute(`python3 -c "${command}"`,source,
      {signal:ctx.signal,timeout:clock.remaining,purpose});
    assert.equal(result.exitCode,0);assert.equal(result.settled,true);
    assert.ok(result.executionSeconds>=.04);assert.ok(clock.elapsed-before>=result.executionSeconds-.01);
    executions.push({purpose,handle:result.handle,executionSeconds:result.executionSeconds,charged:clock.elapsed-before});
  }
  assert.ok(excluded.some(e=>e.label==='setup'));assert.ok(excluded.some(e=>e.label==='transfer'&&e.seconds>0));
  assert.ok(clock.elapsed>3.5);assert.ok(clock.remaining<116.5);
  return {thinkingCharged:true,executions,excluded,savedElapsed:3.5,elapsed:clock.elapsed,remaining:clock.remaining};
}
async function receipt(ctx,config,beforeReceipt=false) {
  const scheduler=schedule({...config,faults:[{point:beforeReceipt?'run.before-receipt':'run.after-receipt',occurrence:2,disconnect:true}]});
  const backend=await target({root:ctx.directory,scheduler});let server;
  ctx.onCleanup(()=>server?server.close():backend.close());server=await serve(backend);
  const connection=connect(server.url),events=[];let clientSignal=ctx.signal;
  const client=(args,settings)=>connection(args,{...settings,signal:clientSignal});
  const options={mode:'cpl',target:'fixture',runtime:'declared',workspacePath:backend.workspace,remoteDirectory:'receipt',pollMs:5};
  const context={sdk,client,instancePath:ctx.directory,event:async(type,data)=>{events.push({type,data});await ctx.event({type,data});},exclude:async(_,fn)=>fn()};
  const source=join(ctx.directory,'source');await mkdir(source);await writeFile(join(source,'x.py'),'x=42\n');
  const environment=await createEnvironment(options,context);await closed(backend,environment,ctx,signal=>{clientSignal=signal;});
  await environment.prepare({project:{id:'receipt',source},attempt:{id:'same'}});
  const execute=()=>environment.execute('python3 -c "import x; assert x.x==42; print(x.x)"',source,{signal:ctx.signal,timeout:30});
  if(beforeReceipt){
    await assert.rejects(execute);
    const unresolved=events.find(e=>e.type==='execution-unresolved');assert.ok(unresolved);
    const proof=await environment.reconcileOperation({operation:unresolved.data.operation,events});
    assert.equal(proof.closed,false);
    const commands=backend.calls.filter(c=>c[0]==='run'&&c[c.indexOf('--file')+1].endsWith('command.sh'));
    assert.equal(commands.length,1);
    return {held:true,projectPassed:false,receiptMissing:true,operation:unresolved.data.operation,workloadDispatches:1};
  }
  const result=await execute();assert.equal(result.settled,true);assert.equal(result.exitCode,0);
  const observer=await createEnvironment(options,{...context,client:(args,settings)=>connect(server.url)(args,{...settings,signal:clientSignal})});
  assert.equal((await observer.observe(result.handle)).state,'stopped');
  const commands=backend.calls.filter(c=>c[0]==='run'&&c[c.indexOf('--file')+1].endsWith('command.sh'));
  assert.equal(commands.length,1);assert.equal(scheduler.events.filter(e=>e.fault?.disconnect).length,1);
  return {handle:result.handle,workloadDispatches:commands.length,independentObserver:true,receiptRecovered:true};
}
async function readinessAndRevision(ctx,scheduler,listDelay) {
  let listed=false,lists=0;const queries=[],events=[];
  const endpoint=await listen(ctx,async(req,res,message,signal)=>{
    if(req.method!=='POST'){res.writeHead(405);res.end();return;}
    if(message.id===undefined){res.writeHead(202);res.end();return;}
    let result={};
    if(message.method==='initialize')result={protocolVersion:message.params.protocolVersion,capabilities:{tools:{}},serverInfo:{name:'fixture',version:'1'}};
    if(message.method==='tools/list') {
      lists++;await scheduler.at('kdb.tools-list',{},signal);await delay(listDelay,undefined,{signal});listed=true;
      result={tools:[{name:'search',description:'Search indexed fixture facts',inputSchema:{type:'object',properties:{query:{type:'string'},kind:{type:'string'},limit:{type:'number'}}}}]};
    }
    if(message.method==='tools/call') {
      queries.push(message.params.arguments.kind);
      result={content:[{type:'text',text:'Fixture numeric fact'}],structuredContent:{result:[{text:'Fixture numeric fact',
        collection:'fixture',chunk_id:'numeric',path:'fact.py',start_line:1,end_line:1,source_sha256:'a'.repeat(64)}]}};
    }
    res.writeHead(200,{'Content-Type':'application/json'});res.end(JSON.stringify({jsonrpc:'2.0',id:message.id,result}));
  });
  const calls=[];
  const addSteps=sequence=>{for(const stage of sequence){calls.push(['rsi_skill',{}]);
    if(['design','optimisation'].includes(stage))for(const kind of ['good','negative'])calls.push(['rsi_knowledge',{query:'numeric fact',kind}]);
    if(stage==='validation')calls.push(['rsi_verify',{}]);calls.push(['rsi_advance',{reflection:'Retain independent numeric evidence.'}]);}};
  addSteps(stages);calls.push(['rsi_verify',{}],['rsi_revise',{to:'implementation',reflection:'Source changed after verification.'}]);
  addSteps(stages.slice(stages.indexOf('implementation')));
  const wire=await provider(ctx,scheduler,body=>{
    assert.equal(listed,true);assert.ok(events.some(e=>e.type==='knowledge-ready'));
    assert.ok(body.tools.some(t=>t.function.name==='mcp__kdb__search'));
    const index=body.messages.reduce((n,m)=>n+(m.tool_calls?.length??0),0);
    return index<calls.length?{id:'call-'+index,name:calls[index][0],args:calls[index][1]}:'Completed canonical recovery.';
  });
  const source=join(ctx.directory,'source');await mkdir(source);await writeFile(join(source,'answer.py'),'answer=42\n');
  const verifier=join(ctx.directory,'verify.py');await writeFile(verifier,"import runpy\nassert runpy.run_path('answer.py')['answer']==42\nprint('numeric PASS')\n");
  const state=new State(join(ctx.directory,'state.sqlite'));ctx.onCleanup(()=>state.close());
  state.add({id:'revision',goal:'Return42',source,actor_seconds:120,verifier:{files:[verifier],command:'PYTHONDONTWRITEBYTECODE=1 python3 ../verify.py'}});
  const attempt=state.admit('revision',{}),skillMap={};
  for(const stage of [...stages,'global-kdb']){skillMap[stage]=join(ctx.directory,stage+'.md');await writeFile(skillMap[stage],'Use independent numeric evidence.');}
  const agentDir=join(ctx.directory,'agent');await mkdir(agentDir);
  const bridge=createKdbBridge({url:endpoint+'/mcp',attemptId:attempt.id,agentDir,event:async(type,data)=>{events.push({type,data});await ctx.event({type,data});}});
  let injected=false,staleCall,blocked=false;
  const hook=pi=>{
    pi.on('tool_call',async e=>{
      if(!injected&&e.toolName==='rsi_advance'&&state.events(attempt.id).filter(x=>x.type==='stage-completed').at(-1)?.data.next==='acceptance') {
        injected=true;staleCall=e.toolCallId;await writeFile(join(source,'answer.py'),'answer=42\n# post-verification edit\n');
      }
    });
    pi.on('tool_result',e=>{if(e.toolCallId===staleCall)blocked=e.isError===true;});
  };
  const before=performance.now();
  const result=await runAttempt({state,attempt,project:state.project('revision').spec,attemptPath:join(ctx.directory,'attempt'),skillMap,
    signal:ctx.signal,...await runtime(wire.url),knowledge:bridge.knowledge,extensions:[bridge.extension,hook],
    environmentFactory:context=>createEnvironment({mode:'local'},context)});
  const recorded=state.events(attempt.id),verified=recorded.filter(e=>e.type==='verification');
  const wallSeconds=(performance.now()-before)/1000;
  assert.ok(result.elapsed<wallSeconds);if(listDelay)assert.ok(wallSeconds-result.elapsed>=listDelay/1000-.2);
  assert.equal(result.accepted,true);assert.equal(injected,true);assert.equal(blocked,true);
  assert.equal(verified.length,2);assert.ok(verified.every(e=>e.data.passed));
  assert.notEqual(verified[0].data.fingerprint,verified[1].data.fingerprint);
  assert.ok(recorded.some(e=>e.type==='stage-completed'&&e.data.stage==='acceptance'&&e.data.next==='implementation'));
  assert.equal(recorded.filter(e=>e.type==='accepted').length,1);
  assert.ok(queries.includes('good')&&queries.includes('negative'));assert.ok(lists>=1);
  const starts=recorded.filter(e=>e.type==='native-bash-started'),ends=recorded.filter(e=>e.type==='native-bash-completed');
  assert.equal(starts.length,ends.length);
  await writeFile(join(ctx.directory,'workflow-events.json'),JSON.stringify(recorded,null,2));
  return {nativeMcp:true,lists,queries,providerRequests:wire.requests.length,listDelay,
    setupExcluded:true,wallSeconds,accepted:true,staleBlocked:true,
    session:state.attempt(attempt.id).session,elapsed:result.elapsed,verifications:verified.length};
}

async function workerLoss(ctx,scheduler,graceful) {
  const source=join(ctx.directory,'source');await mkdir(source);
  const state=new State(join(ctx.directory,'state.sqlite'));ctx.onCleanup(()=>state.close());
  state.add({id:'worker',goal:'Observe native child',source,actor_seconds:120,verifier:{files:[],command:'true'}});
  const wire=await provider(ctx,scheduler,async(body,signal)=>{
    if(body.messages.some(m=>m.role==='tool')){await delay(60000,undefined,{signal});return 'Stopped.';}
    return {name:'bash',args:{command:`python3 -c "import os,time; open('child.pid','w').write(str(os.getpid())); time.sleep(60)"`}};
  });
  const worker=spawn(process.execPath,[fileURLToPath(new URL('./lifecycle-worker.mjs',import.meta.url)),ctx.directory,wire.url],{stdio:['ignore','ignore','pipe']});
  const exit=once(worker,'exit');let stderr='',pgid,pid,childStart,id;worker.stderr.on('data',chunk=>{stderr+=chunk;});
  const childStopped=async()=>{const stat=pid&&await readProcessStat(pid);
    if(!stat)return true;const fields=stat.slice(stat.lastIndexOf(')')+2).split(' ');
    return fields[0]==='Z'||fields[19]!==childStart;};
  ctx.onCleanup(async({signal,remainingSeconds})=>{
    if(worker.exitCode===null&&worker.signalCode===null){
      worker.kill('SIGTERM');await Promise.race([exit,delay(Math.min(1000,remainingSeconds*1000),undefined,{ref:false})]);
      if(worker.exitCode===null&&worker.signalCode===null){worker.kill('SIGKILL');await exit;}
    }
    if(!pid)try{pid=Number(await readFile(join(source,'child.pid'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
    if(pid&&!childStart){const stat=await readProcessStat(pid);if(stat){
      const fields=stat.slice(stat.lastIndexOf(')')+2).split(' ');pgid=Number(fields[2]);childStart=fields[19];
    }}
    if(!id)try{id=await readFile(join(ctx.directory,'attempt-id'),'utf8');}catch(error){if(error.code!=='ENOENT')throw error;}
    if(!pid){const recorded=state.events(id??null);
      const completed=new Set(recorded.filter(e=>e.type==='native-bash-completed').map(e=>e.data.operation));
      assert.ok(recorded.filter(e=>e.type==='native-bash-started').every(e=>completed.has(e.data.operation)),'Native Bash closure identity missing');
    }
    if(pgid&&!await childStopped())try{process.kill(-pgid,'SIGKILL');}catch(error){if(error.code!=='ESRCH')throw error;}
    await until(childStopped,signal);
    await writeFile(join(ctx.directory,'process-closure.json'),JSON.stringify({workerClosed:true,childClosed:true,pid,pgid}));
  });
  const safeRead=async file=>{try{return await readFile(file,'utf8');}catch(error){if(worker.exitCode!==null)throw Error(stderr);if(error.code!=='ENOENT')throw error;}};
  id=await until(()=>safeRead(join(ctx.directory,'attempt-id')),ctx.signal);
  pid=Number(await until(()=>safeRead(join(source,'child.pid')),ctx.signal));
  const stat=await readProcessStat(pid),fields=stat.slice(stat.lastIndexOf(')')+2).split(' ');
  pgid=Number(fields[2]);childStart=fields[19];
  await until(()=>state.events(id).some(e=>e.type==='native-bash-started'),ctx.signal);
  const before=state.attempt(id),requests=wire.requests.length;
  worker.kill(graceful?'SIGTERM':'SIGKILL');await exit;
  const environment={observe:async()=>assert.fail('Native Bash has no target handle'),reconcileOperation:async()=>assert.fail('No synthetic reconciliation')};
  if(graceful){
    assert.equal(JSON.parse(await readFile(join(ctx.directory,'worker-closed.json'),'utf8')).interrupted,true);
    await until(childStopped,ctx.signal);
    assert.equal(await attemptStopped(state,state.attempt(id),environment),true);
    const resumed=await state.resume(id,a=>attemptStopped(state,a,environment),await processIdentity());
    assert.equal(resumed.id,id);assert.equal(resumed.session,before.session);assert.equal(resumed.outcome,null);
    assert.ok(resumed.elapsed>=before.elapsed);assert.ok(resumed.elapsed>=2.25);
  }else{
    assert.equal(await childStopped(),false);
    await assert.rejects(state.resume(id,a=>attemptStopped(state,a,environment),await processIdentity()));
    assert.deepEqual(state.attempt(id),before);
    process.kill(-pgid,'SIGKILL');await until(childStopped,ctx.signal);
    assert.equal(await attemptStopped(state,before,environment),false);
  }
  assert.equal(wire.requests.length,requests);assert.equal(state.events(id).filter(e=>e.type==='native-bash-started').length,1);
  await writeFile(join(ctx.directory,'worker-events.json'),JSON.stringify(state.events(id),null,2));
  return {mode:graceful?'graceful-resume':'SIGKILL-held',projectPassed:false,attempt:id,
    session:before.session,elapsedBefore:before.elapsed,elapsedAfter:state.attempt(id).elapsed,noReplay:true};
}

export const cases=[{id:'lifecycle-matrix',tier:'local',async run(ctx){
  const evidence=[];
  for(const config of [{seed:11,profile:'normal'},{seed:29,profile:'slow'},{seed:47,profile:'bursty'}]) {
    ctx.signal.throwIfAborted();const part=await childContext(ctx,config.profile),scheduler=schedule(config);
    const record={...config,schedule:'accounting/schedule.json'};
    record.accounting=await accounting(await childContext(part,'accounting'),scheduler);
    record.receipt=await receipt(await childContext(part,'receipt'),config);
    record.receiptMissing=await receipt(await childContext(part,'receipt-missing'),config,true);
    record.readinessRevision=await readinessAndRevision(await childContext(part,'readiness-revision'),scheduler,config.profile==='normal'?11000:0);
    record.workerLoss=await workerLoss(await childContext(part,'worker-loss'),scheduler,false);
    record.graceful=await workerLoss(await childContext(part,'graceful'),scheduler,true);
    evidence.push(record);await ctx.event({type:'profile-completed',...record});
  }
  return {scope:'Local SDK/MCP/scripted-provider CPU behavioral matrix; no paid model or device evidence',profiles:evidence};
}},{id:'lifecycle-native-compaction',tier:'local',async run(ctx){
  const scheduler=schedule(ctx),wire=await provider(ctx,scheduler,()=> 'Numeric contract42. '+ 'retained fact '.repeat(256));
  const settings=sdk.SettingsManager.inMemory({compaction:{enabled:false,reserveTokens:256,keepRecentTokens:128},retry:{enabled:false}});
  const manager=sdk.SessionManager.create(ctx.directory,join(ctx.directory,'session'));
  const agentDir=join(ctx.directory,'agent');await mkdir(agentDir);
  const loader=new sdk.DefaultResourceLoader({cwd:ctx.directory,agentDir,settingsManager:settings,noExtensions:true,noSkills:true,noContextFiles:true});await loader.reload();
  const {session}=await sdk.createAgentSession({cwd:ctx.directory,agentDir,sessionManager:manager,settingsManager:settings,
    resourceLoader:loader,noTools:'builtin',...await runtime(wire.url)});
  const abort=()=>{session.abortCompaction();void session.abort();};ctx.signal.addEventListener('abort',abort,{once:true});
  ctx.onCleanup(async()=>{ctx.signal.removeEventListener('abort',abort);abort();await session.waitForIdle();session.dispose();
    await writeFile(join(ctx.directory,'schedule.json'),JSON.stringify(scheduler.events,null,2));});
  await session.bindExtensions({});
  await session.prompt('Retain numeric contract42. '+ 'background '.repeat(512));await session.prompt('More context. '+ 'history '.repeat(512));
  const original=manager.getSessionFile(),result=await session.compact();
  assert.ok(result.summary);assert.ok(manager.getEntries().some(e=>e.type==='compaction'));
  assert.equal(manager.getSessionFile(),original);await session.prompt('Continue after native compaction.');
  assert.ok(wire.requests.length>=4);assert.equal(session.sessionManager.getSessionFile(),original);
  return {nativeCompaction:true,session:original,tokensBefore:result.tokensBefore,requests:wire.requests.length};
}}];
