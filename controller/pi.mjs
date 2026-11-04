import * as sdk from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';
import { mkdir, appendFile, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { ActorClock } from './budget.mjs';
import { Workflow } from './workflow.mjs';
import { validateAcceptance } from './state.mjs';
import { fingerprint, freezeFiles } from './evidence.mjs';
import { createEnvironmentTools } from '../integrations/environments/index.mjs';

const result=value=>({content:[{type:'text',text:JSON.stringify(value)}],details:{}});

// Pi owns sessions, context, model retries, standard tools and compaction.
export async function runAttempt({state,attempt,project,attemptPath,environmentFactory,
  knowledge,modelRuntime,model,thinkingLevel='off',skillMap,extensions=[],signal,settleProject=true}) {
  validateAcceptance(project);
  await mkdir(attemptPath,{recursive:true});
  let activeOperation=null;
  const event=(type,data)=>state.event(attempt.id,type,{...data,
    operation:activeOperation??data.operation??data.submission,actor_elapsed:clock.elapsed});
  const clock=new ActorClock(project.actor_seconds,attempt.elapsed);
  const environment=await environmentFactory({instancePath:attemptPath,event,
    exclude:(label,fn)=>clock.exclude(label,fn)});
  const previous=state.events(attempt.id).find(e=>e.type==='verifier-frozen');
  const checkVerifier=await freezeFiles(project.verifier.files,project.source,project.verifier.hashes??previous?.data.hashes);
  if(!previous)event('verifier-frozen',{hashes:checkVerifier.manifest});
  let session,timer,accepted=false,uncertain;
  const abort=()=>{void session?.abort();};
  signal?.addEventListener('abort',abort,{once:true});
  const diagnostics=[];
  const execute=async(command,cwd,options)=>{
    const operation=randomUUID(),startElapsed=clock.elapsed;
    activeOperation=operation;event('execution-starting',{operation,startElapsed});
    let response;
    try {
      response=await environment.execute(command,cwd,{...options,operationId:operation});
      if(!response.settled)throw new Error(`Execution closure unknown: ${response.handle??'submission'}`);
      state.checkpoint(attempt.id,clock.elapsed);
      event('execution-settled',{operation,handle:response.handle});
      return response;
    } catch(error) {
      event('execution-unresolved',{operation,startElapsed,handle:response?.handle??null,message:error.message});
      uncertain=error;void session?.abort();throw error;
    } finally {activeOperation=null;}
  };
  const workflow=new Workflow({clock,event,fingerprint:()=>fingerprint(project.source),
    readSkill:async stage=>{
      const path=skillMap[stage];
      if (!path) throw new Error(`No skill mapped to ${stage}`);
      const content=await readFile(path,'utf8');
      const globalPath=skillMap['global-kdb'];
      if(!globalPath)throw new Error('No global KDB skill mapped');
      const guidance=await readFile(globalPath,'utf8');
      event('skill-read',{stage:'global-kdb',path:globalPath,sha256:createHash('sha256').update(guidance).digest('hex')});
      return {content:guidance+'\n\n'+content,evidence:{path,sha256:createHash('sha256').update(content).digest('hex')}};
    },search:(request,ctx)=>knowledge.search({...request,attempt_id:attempt.id},ctx),
    verify:async(verifySignal)=>{
      const check=async(command,purpose)=>{
        clock.check();
        await checkVerifier();
        let output='';
        const response=await execute(command,project.source,{
          timeout:clock.remaining,signal:verifySignal??signal,purpose,
          onData:data=>{output+=data.toString();}});
        await checkVerifier();
        const evidence=join(attemptPath,`verify-${purpose}-${Date.now()}.log`);
        await appendFile(evidence,output);
        const evidence_sha256=createHash('sha256').update(await readFile(evidence)).digest('hex');
        return {passed:response.exitCode === 0,exitCode:response.exitCode,evidence,evidence_sha256};
      };
      const correctness=await check(project.verifier.command,'verification');
      const performance=project.verifier.performance_command
        ? correctness.passed ? await check(project.verifier.performance_command,'performance')
          : {passed:false,skipped:true} : null;
      return {correctness,performance,passed:correctness.passed && (!performance || performance.passed),
        evidence:{correctness:correctness.evidence,performance:performance?.evidence??null}};
    }}).restore(state.events(attempt.id));
  const extension=async pi=>{
    pi.registerTool({name:'rsi_skill',label:'Read workflow skill',description:'Read the current step skill.',
      parameters:Type.Object({}),execute:async()=>result({stage:workflow.stage,skill:await workflow.skill(),history:workflow.history})});
    pi.registerTool({name:'rsi_knowledge',label:'Retrieve knowledge',description:'Query positive or negative prior evidence.',
      reachesTools:()=>['mcp__kdb__search'],
      parameters:Type.Object({query:Type.String({minLength:1}),kind:Type.Union([Type.Literal('good'),Type.Literal('negative')])}),
      execute:async(_id,args,_signal,_onUpdate,ctx)=>result(await workflow.query(args.query,args.kind,ctx))});
    pi.registerTool({name:'rsi_verify',label:'Verify source',description:'Run the frozen independent verifier. Returns controller-local evidence paths and SHA-256 digests for lesson references.',
      parameters:Type.Object({}),execute:async(_id,_args,nativeSignal)=>result(await workflow.validate(nativeSignal))});
    pi.registerTool({name:'rsi_advance',label:'Advance workflow',description:'Complete the current step and advance. Only accepted:true completes the workflow after the final acceptance gate; entering acceptance is not completion. Design and optimisation require reflection.',
      parameters:Type.Object({reflection:Type.Optional(Type.String())}),
      execute:async(_id,args)=>{
        const response=await workflow.advance({reflection:args.reflection});
        accepted=Boolean(response.accepted);
        return result(response);
      }});
    pi.registerTool({name:'rsi_revise',label:'Revise after findings',description:'After debugging or validation, return to design or implementation with the observed reason.',
      parameters:Type.Object({to:Type.Union([Type.Literal('design'),Type.Literal('implementation')]),reflection:Type.String({minLength:1})}),
      execute:async(_id,args)=>result(await workflow.advance({next:args.to,reflection:args.reflection}))});
    const [target]=await createEnvironmentTools({execute:async(command,cwd,options)=>{
      clock.check();
      return execute(command,cwd,{...options,
        purpose:project.purpose??(project.performance_goal?'performance':'ordinary'),
        timeout:Math.min(options.timeout??Infinity,clock.remaining)});
    }},project.source,sdk);
    pi.registerTool(target);
    for (const type of ['before_provider_request','provider_stream_event']) {
      pi.on(type,async payload=>{
        await appendFile(join(attemptPath,'provider.jsonl'),JSON.stringify(payload)+'\n');
      });
    }
    pi.on('after_provider_response',async payload=>{
      await appendFile(join(attemptPath,'provider.jsonl'),JSON.stringify({type:payload.type,status:payload.status})+'\n');
    });
    pi.on('tool_call',async()=>{
      clock.check();
      if (uncertain) return {block:true,reason:'Execution closure is unknown; resume only after observation'};
      if (accepted) return {block:true,reason:'Project already accepted'};
    });
  };
  try {
    if(signal?.aborted)throw new Error('Attempt interrupted');
    clock.check();
    await environment.prepare({project,attempt});
    const manager=attempt.session ? sdk.SessionManager.open(attempt.session,attemptPath,project.source)
      : sdk.SessionManager.create(project.source,attemptPath);
    await clock.exclude('setup',async()=>{
    const agentDir=join(attemptPath,'agent');
    await mkdir(agentDir,{recursive:true});
    const settingsManager=sdk.SettingsManager.inMemory({compaction:{enabled:true},retry:{enabled:true}});
    const loader=new sdk.DefaultResourceLoader({cwd:project.source,agentDir,settingsManager,noExtensions:true,noSkills:true,noContextFiles:true,
      extensionFactories:[extension,...extensions]});
    await loader.reload();
    ({session}=await sdk.createAgentSession({cwd:project.source,agentDir,settingsManager,modelRuntime,model,thinkingLevel,
      sessionManager:manager,resourceLoader:loader}));
    session.agent.toolExecution='sequential';
    if(knowledge.prepareSession)await knowledge.prepareSession(session,{signal});
    else await session.bindExtensions({});
    });
    state.checkpoint(attempt.id,clock.elapsed,manager.getSessionFile());
    session.subscribe(e=>{
      // Native session files retain model/tool content; this sidecar retains lifecycle telemetry.
      if (e.type === 'message_end' && e.message.role === 'assistant' && e.message.stopReason === 'error')
        diagnostics.push(e.message.errorMessage ?? 'Native provider error');
      event('pi-event',{type:e.type,stage:workflow.stage,elapsed:clock.elapsed});
    });
    timer=setInterval(()=>{
      state.checkpoint(attempt.id,clock.elapsed,manager.getSessionFile());
      if (!clock.remaining) void session.abort();
    },1000);
    if(signal?.aborted)throw new Error('Attempt interrupted');
    await session.prompt(`Goal: ${project.goal}\nPerformance goal: ${JSON.stringify(project.performance_goal??null)}\nFrozen numeric command: ${project.verifier.command}\nFrozen performance command: ${project.verifier.performance_command??'not required'}\nEnvironment instructions: ${environment.instructions}\nFollow the canonical workflow using rsi_skill, rsi_knowledge, rsi_advance and rsi_verify. Current step: ${workflow.stage}. Design and optimisation require positive and negative retrieval and retrospection. Only a passing frozen verifier can accept the project.`);
    await session.waitForIdle();
    if(signal?.aborted)throw new Error('Attempt interrupted');
    if(uncertain)throw uncertain;
    state.checkpoint(attempt.id,clock.elapsed,manager.getSessionFile());
    let reason='actor-ended-without-workflow-acceptance';
    if (accepted && workflow.validation.fingerprint !== await fingerprint(project.source)) {
      accepted=false;reason='source-changed-after-workflow-acceptance';
    }
    if (!accepted) event('rejected',{stage:workflow.stage,reason,diagnostics,remaining:clock.remaining});
    state.finish(attempt.id,accepted?'passed':'failed',{settleProject});
    return {accepted,elapsed:clock.elapsed};
  } catch(error) {
    event('attempt-error',{message:error.message,stage:workflow.stage});
    state.checkpoint(attempt.id,clock.elapsed);
    if(error.code==='ACTOR_ELAPSED_EXHAUSTED' && !uncertain && !signal?.aborted) {
      event('rejected',{stage:workflow.stage,reason:error.code});
      state.finish(attempt.id,'failed',{settleProject});
      return {accepted:false,elapsed:clock.elapsed};
    }
    // Keep the attempt open until the caller confirms all recorded external handles stopped.
    throw error;
  } finally {
    signal?.removeEventListener('abort',abort);
    clearInterval(timer);
    if(session) {
      try {await session.extensionRunner.emit({type:'session_shutdown',reason:'quit'});}
      finally {session.dispose();}
    }
  }
}
