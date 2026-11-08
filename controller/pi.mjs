import * as sdk from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';
import { mkdir, appendFile, readFile } from 'node:fs/promises';
import { join,resolve,relative,basename } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { ActorClock } from './budget.mjs';
import { Workflow } from './workflow.mjs';
import { validateAcceptance } from './state.mjs';
import { fingerprint, freezeFiles } from './evidence.mjs';
import { createEnvironmentTools } from '../integrations/environments/index.mjs';
import { createPiTelemetryJournal } from './pi-telemetry.mjs';
import {CampaignStorage} from './storage.mjs';
import {prepareSourceGit} from './source-git.mjs';
import { recordSkillContext } from './skill-evidence.mjs';
import { validateTargetBaseline } from './baseline-preflight.mjs';

const result=value=>({content:[{type:'text',text:JSON.stringify(value)}],details:{}});
const lifecycleEvents=new Set(['agent_start','agent_end','turn_start','turn_end','message_end',
  'tool_execution_start','tool_execution_end','auto_compaction_start','auto_compaction_end','auto_retry_start','auto_retry_end']);

// Pi owns sessions, context, model retries, standard tools and compaction.
export async function runAttempt({state,attempt,project,attemptPath,environmentFactory,
  knowledge,modelRuntime,model,thinkingLevel='off',skillMap,extensions=[],signal,settleProject=true,
  shouldPause=()=>false,onPause=()=>{},environmentIdentity}) {
  validateAcceptance(project);
  await mkdir(attemptPath,{recursive:true});
  const storage=CampaignStorage.open(state,resolve(attemptPath,'../..'));
  let artifactWrites=Promise.resolve(),artifactError;
  const register=(path,kind,external=false)=>storage?.register(path,{attempt:attempt.id,project:attempt.project,kind,external});
  let activeOperation=null;
  const eventData=data=>({...data,
    operation:activeOperation??data.operation??data.submission,actor_elapsed:clock.elapsed});
  const event=(type,data)=>{
    state.event(attempt.id,type,eventData(data));
    if(storage&&type==='stage-completed'&&data.stage!=='validation') {
      artifactWrites=artifactWrites.then(async()=>{
        const snapshot=sourceGit&&['implementation','optimisation'].includes(data.stage)
          ?await sourceGit.captureTree({signal,timeout:Math.min(30000,clock.remaining*1000)}):null;
        storage.iteration(attempt.id,{stage:data.stage,reflection:data.reflection,
          sourceHash:snapshot?.tree_oid??null,sourceCommit:snapshot?.head??null});
        if(snapshot)state.event(attempt.id,'source-snapshot',{stage:data.stage,...snapshot});
      }).catch(error=>{artifactError??=error;});
      return artifactWrites.then(()=>{if(artifactError)throw artifactError;});
    }
    if(storage&&type==='execution-output')artifactWrites=artifactWrites.then(async()=>{
      for(const stream of ['stdout','stderr'])if(data[stream])await register(data[stream],'execution-'+stream);
      if(data.partial&&data.archive)await register(JSON.stringify(data.archive),'execution-output',true);
    }).catch(error=>{artifactError??=error;});
  };
  const clock=new ActorClock(project.actor_seconds,attempt.elapsed);
  const sourceGit=storage?await clock.exclude('setup',async()=>{
    const key=createHash('sha256').update(attempt.project+'\0'+relative(storage.root,project.source)).digest('hex');
    const gitDir=join(storage.root,'source-history',key);
    const git=await prepareSourceGit({source:project.source,gitDir,attemptId:attempt.id});
    await storage.register(gitDir,{project:attempt.project,kind:'source-git'});return git;
  }):null;
  const telemetry=createPiTelemetryJournal(join(attemptPath,'pi-events.jsonl'),attempt.id,
    data=>state.event(attempt.id,'pi-event',data));
  const environment=await environmentFactory({instancePath:attemptPath,event,signal,
    exclude:(label,fn)=>clock.exclude(label,fn)});
  const previous=state.events(attempt.id).find(e=>e.type==='verifier-frozen');
  const checkVerifier=await freezeFiles(project.verifier.files,project.source,project.verifier.hashes??previous?.data.hashes);
  if(!previous)event('verifier-frozen',{hashes:checkVerifier.manifest});
  await recordSkillContext({state,attemptId:attempt.id,project,verifierHashes:checkVerifier.manifest,environmentIdentity});
  let session,timer,accepted=false,uncertain,telemetryFailure;
  const nativeBash=new Map();
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
      const closure=error.closure;
      const reason=closure?.reason;
      const terminalSshUpload=reason?.backend==='ssh'&&reason.operation==='upload'&&reason.state==='failed'&&
        Number.isInteger(reason.exit)&&reason.exit!==0&&reason.handle==null;
      if(['SETUP_CLOSED','CPL_NOT_ADMITTED'].includes(error.code) && closure?.closed===true &&
          closure.payloadStarted===false && Array.isArray(closure.handles) &&
          (error.code==='CPL_NOT_ADMITTED'||closure.handles.length>0||terminalSshUpload)) {
        state.checkpoint(attempt.id,clock.elapsed);
        event('execution-settled',{operation,startElapsed,handle:closure.handles[0]??null,
          setupClosure:closure,message:error.message});
        throw error; // Native Pi receives a normal tool error and owns its next action.
      }
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
      if(sourceGit) {
        const snapshot=await sourceGit.captureTree({signal:verifySignal??signal,timeout:Math.min(30000,clock.remaining*1000)});
        storage.iteration(attempt.id,{stage:'validation',reflection:JSON.stringify(snapshot),
          sourceHash:snapshot.tree_oid,sourceCommit:snapshot.head});
      }
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
        await register(evidence,'verification');
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
    if(storage)pi.registerTool({name:'rsi_artifact',label:'Register retained evidence',
      description:'Register an evidence file outside editable source. Use external=true for an uncollected remote path; it remains explicitly missing until collected.',
      parameters:Type.Object({path:Type.String(),kind:Type.String(),external:Type.Optional(Type.Boolean()),collect:Type.Optional(Type.Boolean())}),
      execute:async(_id,args)=>{
        const artifact=await register(args.external?args.path:resolve(project.source,args.path),args.kind,args.external);
        if(args.collect) {
          if(!args.external||!environment.collectArtifact)throw new Error('Collection requires an external file and configured environment');
          const directory=join(attemptPath,'artifacts');await mkdir(directory,{recursive:true});
          const destination=join(directory,randomUUID()+'-'+basename(args.path));
          await environment.collectArtifact(args.path,destination);
          return result(await storage.collected(artifact.id,destination));
        }
        return result(artifact);
      }});
    pi.registerTool({name:'rsi_skill',label:'Read workflow skill',description:'Read the current step skill.',
      parameters:Type.Object({}),execute:async()=>result({stage:workflow.stage,skill:await workflow.skill(),history:workflow.history})});
    pi.registerTool({name:'rsi_knowledge',label:'Retrieve knowledge',description:'Query positive or negative prior evidence.',
      reachesTools:()=>['mcp__kdb__search'],
      parameters:Type.Object({query:Type.String({minLength:1}),kind:Type.Union([Type.Literal('good'),Type.Literal('negative')])}),
      execute:async(_id,args,_signal,_onUpdate,ctx)=>result(await workflow.query(args.query,args.kind,ctx))});
    pi.registerTool({name:'rsi_verify',label:'Verify source',description:'Run the frozen independent verifier in validation after completing all project edits. This tool stages trusted verifier files and executes the frozen numeric/performance commands; use it instead of invoking those command metadata strings through rsi_target. All project files, including documentation and receipts, affect the source fingerprint; later edits require revision and fresh verification. Returns retained controller-local evidence paths and SHA-256 digests: read these directly and keep post-validation receipts outside project source.',
      parameters:Type.Object({}),execute:async(_id,_args,nativeSignal)=>result(await workflow.validate(nativeSignal))});
    pi.registerTool({name:'rsi_advance',label:'Advance workflow',description:'Complete the current step and advance. Only accepted:true completes the workflow after the final acceptance gate; entering acceptance is not completion. Design and optimisation require reflection.',
      parameters:Type.Object({reflection:Type.Optional(Type.String())}),
      execute:async(_id,args)=>{
        const response=await workflow.advance({reflection:args.reflection});
        accepted=Boolean(response.accepted);
        return result(response);
      }});
    pi.registerTool({name:'rsi_revise',label:'Revise after findings',description:'From debug, optimisation, validation, or acceptance, return to design or implementation with an observed reason. Then complete the canonical steps and fresh validation before accepting.',
      parameters:Type.Object({to:Type.Union([Type.Literal('design'),Type.Literal('implementation')]),reflection:Type.String({minLength:1})}),
      execute:async(_id,args)=>result(await workflow.advance({next:args.to,reflection:args.reflection}))});
    const shellTools=await createEnvironmentTools({shellOnTarget:environment.shellOnTarget,
      localOperations:environment.localOperations&&{exec:(command,cwd,options)=>{
        clock.check();return environment.localOperations.exec(command,cwd,{...options,timeout:Math.min(options.timeout??Infinity,clock.remaining)});
      }},execute:async(command,cwd,options)=>{
      clock.check();
      return execute(command,cwd,{...options,
        purpose:project.purpose??(project.performance_goal?'performance':'ordinary'),
        timeout:Math.min(options.timeout??Infinity,clock.remaining)});
    }},project.source,sdk);
    for(const tool of shellTools)pi.registerTool(tool);
    for (const type of ['before_provider_request','provider_stream_event']) {
      pi.on(type,async payload=>{
        await appendFile(join(attemptPath,'provider.jsonl'),JSON.stringify(payload)+'\n');
      });
    }
    pi.on('after_provider_response',async payload=>{
      await appendFile(join(attemptPath,'provider.jsonl'),JSON.stringify({type:payload.type,status:payload.status})+'\n');
    });
    pi.on('tool_call',async()=>{
      if(shouldPause()){onPause();return {block:true,reason:'Instance paused at execution boundary'};}
      clock.check();
      if (uncertain) return {block:true,reason:'Execution closure is unknown; resume only after observation'};
      if (accepted) return {block:true,reason:'Project already accepted'};
    });
  };
  try {
    if(signal?.aborted)throw new Error('Attempt interrupted');
    clock.check();
    await environment.prepare({project,attempt});
    if(project.verifier.baseline_command)await validateTargetBaseline({project,state,attempt,attemptPath,
      execute,checkVerifier,signal,remaining:()=>clock.remaining,register});
    const manager=attempt.session ? sdk.SessionManager.open(attempt.session,attemptPath,project.source)
      : sdk.SessionManager.create(project.source,attemptPath);
    if(storage)await storage.session(manager.getSessionFile(),{attempt:attempt.id,
      phase:storage.db.prepare('SELECT phase FROM storage_attempts WHERE attempt=?').get(attempt.id)?.phase??'phase1'});
    await register(join(attemptPath,'provider.jsonl'),'provider-telemetry');
    await clock.exclude('setup',async()=>{
    const agentDir=join(attemptPath,'agent');
    await mkdir(agentDir,{recursive:true});
    const settingsManager=sdk.SettingsManager.inMemory({compaction:{enabled:true},retry:{enabled:true}});
    const loader=new sdk.DefaultResourceLoader({cwd:project.source,agentDir,settingsManager,noExtensions:true,noSkills:true,noContextFiles:true,
      extensionFactories:[extension,...(sourceGit?[sourceGit.extension]:[]),...extensions]});
    await loader.reload();
    ({session}=await sdk.createAgentSession({cwd:project.source,agentDir,settingsManager,modelRuntime,model,thinkingLevel,
      sessionManager:manager,resourceLoader:loader}));
    session.agent.toolExecution='sequential';
    if(knowledge.prepareSession)await knowledge.prepareSession(session,{signal});
    else await session.bindExtensions({});
    session.setActiveToolsByName(session.getActiveToolNames().filter(name=>
      !name.startsWith('mcp__kdb__') || ['mcp__kdb__search','mcp__kdb__read_entry'].includes(name)));
    });
    state.checkpoint(attempt.id,clock.elapsed,manager.getSessionFile());
    session.subscribe(e=>{
      if(e.type==='tool_execution_start' && e.toolName==='bash' && !environment.shellOnTarget) {
        const operation=randomUUID();nativeBash.set(e.toolCallId,operation);
        event('native-bash-started',{operation,tool_call_id:e.toolCallId,session:manager.getSessionFile()});
      }
      if(e.type==='tool_execution_end' && e.toolName==='bash' && nativeBash.has(e.toolCallId)) {
        event('native-bash-completed',{operation:nativeBash.get(e.toolCallId),tool_call_id:e.toolCallId});
        nativeBash.delete(e.toolCallId);
      }
      // Native session files retain model/tool content; this sidecar retains lifecycle telemetry.
      if (e.type === 'message_end' && e.message.role === 'assistant' && e.message.stopReason === 'error')
        diagnostics.push(e.message.errorMessage ?? 'Native provider error');
      // Raw streams remain in provider.jsonl; SQL stores lifecycle labels only.
      if(storage&&!lifecycleEvents.has(e.type))return;
      try { telemetry(eventData({type:e.type,stage:workflow.stage,elapsed:clock.elapsed})); }
      catch(error) { telemetryFailure??=error; throw error; }
    });
    timer=setInterval(()=>{
      state.checkpoint(attempt.id,clock.elapsed,manager.getSessionFile());
      if(shouldPause()&&!activeOperation&&!nativeBash.size)onPause();
      if (!clock.remaining) void session.abort();
    },1000);
    if(signal?.aborted)throw new Error('Attempt interrupted');
    const gitGuidance=sourceGit?await readFile(new URL('../templates/skills/source-git.md',import.meta.url),'utf8'):'';
    await session.prompt(`Goal: ${project.goal}\nPerformance goal: ${JSON.stringify(project.performance_goal??null)}\nFrozen numeric command metadata (executed by rsi_verify): ${project.verifier.command}\nFrozen performance command metadata (executed by rsi_verify): ${project.verifier.performance_command??'not required'}\nUse rsi_verify at validation to stage and run these frozen checks. Do not run their command strings through rsi_target or search the target filesystem for trusted verifier files: ordinary target commands start in editable source and their snapshot excludes trusted verifier files.\nEnvironment instructions: ${environment.instructions}\nFollow the canonical workflow using rsi_skill, rsi_knowledge, rsi_advance and rsi_verify. Current step: ${workflow.stage}. Design and optimisation require positive and negative retrieval and retrospection. Only a passing frozen verifier can accept the project. Persist findings and evidence as local artifacts; the controller records KDB lessons after completion. Actor KDB access is retrieval-only: do not index repositories, submit lessons, or block the workflow on manual KDB writes.\n${gitGuidance}`);
    await session.waitForIdle();
    await artifactWrites;
    if(artifactError)throw artifactError;
    if(telemetryFailure)throw telemetryFailure;
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
    if(!session&&(['SETUP_CLOSED','CPL_NOT_ADMITTED','DEVICE_LEASE_BLOCKED'].includes(error.code)||error.diagnostic))
      event('preactor-infrastructure',{stage:workflow.stage,reason:error.code??'native-client-failed',message:error.message.slice(0,3000)});
    event('attempt-error',{message:error.message,stage:error.code==='PROJECT_BASELINE_INVALID'?'baseline-preflight':workflow.stage});
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
    await artifactWrites;
  }
}
