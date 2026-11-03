import * as sdk from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';
import { mkdir, appendFile, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { ActorClock } from './budget.mjs';
import { Workflow } from './workflow.mjs';
import { fingerprint, freezeFiles } from './evidence.mjs';

const result=value=>({content:[{type:'text',text:JSON.stringify(value)}],details:{}});

// Pi owns sessions, context, model retries, standard tools and compaction.
export async function runAttempt({state,attempt,project,attemptPath,environmentFactory,
  knowledge,modelRuntime,model,thinkingLevel='off',skillMap,extensions=[],signal}) {
  await mkdir(attemptPath,{recursive:true});
  const event=(type,data)=>state.event(attempt.id,type,data);
  const clock=new ActorClock(project.actor_seconds,attempt.elapsed);
  const environment=await environmentFactory({instancePath:attemptPath,event,
    exclude:(label,fn)=>clock.exclude(label,fn)});
  const previous=state.events(attempt.id).find(e=>e.type==='verifier-frozen');
  const checkVerifier=await freezeFiles(project.verifier.files,project.source,project.verifier.hashes??previous?.data.hashes);
  if(!previous)event('verifier-frozen',{hashes:checkVerifier.manifest});
  let session,timer,accepted=false;
  const abort=()=>{void session?.abort();};
  signal?.addEventListener('abort',abort,{once:true});
  const diagnostics=[];
  const workflow=new Workflow({clock,event,fingerprint:()=>fingerprint(project.source),
    readSkill:async stage=>{
      const path=skillMap[stage];
      if (!path) throw new Error(`No skill mapped to ${stage}`);
      const content=await readFile(path,'utf8');
      return {content,evidence:{path,sha256:createHash('sha256').update(content).digest('hex')}};
    },search:(request,ctx)=>knowledge.search({...request,attempt_id:attempt.id},ctx),
    verify:async()=>{
      await checkVerifier();
      let output='';
      const response=await environment.execute(project.verifier.command,project.source,{
        timeout:clock.remaining, purpose:'verification',onData:data=>{output+=data.toString();}});
      await checkVerifier();
      const evidence=join(attemptPath,`verify-${Date.now()}.log`);
      await appendFile(evidence,output);
      return {passed:response.settled === true && response.exitCode === 0,evidence};
    }}).restore(state.events(attempt.id));
  const extension=pi=>{
    pi.registerTool({name:'rsi_skill',label:'Read workflow skill',description:'Read the current step skill.',
      parameters:Type.Object({}),execute:async()=>result(await workflow.skill())});
    pi.registerTool({name:'rsi_knowledge',label:'Retrieve knowledge',description:'Query positive or negative prior evidence.',
      parameters:Type.Object({query:Type.String({minLength:1}),kind:Type.Union([Type.Literal('good'),Type.Literal('negative')])}),
      execute:async(_id,args,_signal,_onUpdate,ctx)=>result(await workflow.query(args.query,args.kind,ctx))});
    pi.registerTool({name:'rsi_verify',label:'Verify source',description:'Run the frozen independent verifier.',
      parameters:Type.Object({}),execute:async()=>result(await workflow.validate())});
    pi.registerTool({name:'rsi_advance',label:'Advance workflow',description:'Complete current step after using its skill and required evidence.',
      parameters:Type.Object({reflection:Type.Optional(Type.String()),next:Type.Optional(Type.String())}),
      execute:async(_id,args)=>{
        const response=await workflow.advance(args);
        accepted=Boolean(response.accepted);
        return result(response);
      }});
    pi.registerTool(sdk.createBashTool(project.source,{operations:{exec:async(command,cwd,options)=>{
      clock.check();
      const response=await environment.execute(command,cwd,{...options,
        timeout:Math.min(options.timeout ?? Infinity,clock.remaining)});
      if (!response.settled) throw new Error('Execution remains observable; resolve its recorded handle before continuing');
      return {exitCode:response.exitCode};
    }}}));
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
      if (accepted) return {block:true,reason:'Project already accepted'};
    });
  };
  try {
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
    await session.bindExtensions({});
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
    await session.prompt(`Goal: ${project.goal}\nEnvironment instructions: ${environment.instructions}\nFollow the canonical workflow using rsi_skill, rsi_knowledge, rsi_advance and rsi_verify. Current step: ${workflow.stage}. Design and optimisation require positive and negative retrieval and retrospection. Only a passing frozen verifier can accept the project.`);
    await session.waitForIdle();
    if(signal?.aborted)throw new Error('Attempt interrupted');
    state.checkpoint(attempt.id,clock.elapsed,manager.getSessionFile());
    if (accepted && workflow.validation.fingerprint !== await fingerprint(project.source)) accepted=false;
    if (!accepted) event('rejected',{stage:workflow.stage,diagnostics,remaining:clock.remaining});
    state.finish(attempt.id,accepted?'passed':'failed');
    return {accepted,elapsed:clock.elapsed};
  } catch(error) {
    event('attempt-error',{message:error.message,stage:workflow.stage});
    state.checkpoint(attempt.id,clock.elapsed);
    // Keep the attempt open until the caller confirms all recorded external handles stopped.
    throw error;
  } finally {
    signal?.removeEventListener('abort',abort);
    clearInterval(timer);
    session?.dispose();
  }
}
