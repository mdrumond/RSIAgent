import * as sdk from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';
import { cp, mkdir, readFile, stat } from 'node:fs/promises';
import { join, resolve, relative, sep } from 'node:path';
import { modelFromProfile } from '../controller/model.mjs';

async function curriculumSession({config,path,bridge,prompt,tool,signal,state,attemptId,role,accepted,resume=false}) {
  const agentDir=join(path,'agent');await mkdir(agentDir,{recursive:true});
  const settingsManager=sdk.SettingsManager.inMemory({compaction:{enabled:true},retry:{enabled:true}});
  const loader=new sdk.DefaultResourceLoader({cwd:path,agentDir,settingsManager,noExtensions:true,noSkills:true,
    extensionFactories:[bridge.extension,pi=>pi.registerTool(tool)]});
  await loader.reload();
  const history=resume?state.events(attemptId).filter(e=>e.data.role===role):[];
  const prior=history.findLast(e=>e.type==='curriculum-session')?.data;
  const seconds=history.find(e=>e.type==='curriculum-session')?.data.allocation_seconds??config.curriculum_seconds??240;
  const elapsed=Math.max(0,...history.filter(e=>['curriculum-checkpoint','curriculum-interrupted'].includes(e.type)).map(e=>e.data.elapsed_seconds));
  if(elapsed>=seconds)throw new Error('Curriculum elapsed allocation exhausted before '+role+' submission');
  const manager=prior?sdk.SessionManager.open(prior.session,join(path,'sessions'),path):sdk.SessionManager.create(path,join(path,'sessions'));
  const {session}=await sdk.createAgentSession({cwd:path,agentDir,settingsManager,...await modelFromProfile(config.curriculum_actor??config.actor),
    thinkingLevel:'off',tools:['read','write','edit','mcp__kdb__*',tool.name],resourceLoader:loader,sessionManager:manager});
  const started=performance.now();
  const spent=()=>elapsed+(performance.now()-started)/1000;
  const checkpoint=()=>state.event(attemptId,'curriculum-checkpoint',{role,session:manager.getSessionFile(),allocation_seconds:seconds,elapsed_seconds:spent()});
  let reason,timer,checkpoints,telemetryError;
  const readiness=new AbortController();
  const interruption=()=>new Error(reason==='elapsed-allocation'
    ? 'Curriculum elapsed allocation exhausted before '+role+' submission'
    : 'Curriculum interrupted by external signal');
  const interrupt=cause=>{reason??=cause;readiness.abort(interruption());void session.abort();};
  const abort=()=>interrupt('external-signal');
  try {
    session.agent.toolExecution='sequential';
    await state.event(attemptId,'curriculum-session',{role,session:manager.getSessionFile(),allocation_seconds:seconds,elapsed_seconds:elapsed});
    signal?.addEventListener('abort',abort,{once:true});
    timer=setTimeout(()=>interrupt('elapsed-allocation'),Math.max(0,seconds-spent())*1000);
    checkpoints=setInterval(()=>{try{checkpoint();}catch(error){telemetryError=error;interrupt('telemetry-failure');}},1000);
    if(signal?.aborted)interrupt('external-signal');
    if(spent()>=seconds)interrupt('elapsed-allocation');
    readiness.signal.throwIfAborted();
    if(bridge.prepareSession)await bridge.prepareSession(session,{signal:readiness.signal});
    else await session.bindExtensions({});
    readiness.signal.throwIfAborted();
    await session.prompt(prompt);await session.waitForIdle();
    if(telemetryError)throw telemetryError;
    if(reason && !accepted())throw interruption();
  }catch(error){
    if(telemetryError)throw telemetryError;
    if(reason && !accepted())throw interruption();
    throw error;
  }finally{
    signal?.removeEventListener('abort',abort);clearTimeout(timer);clearInterval(checkpoints);
    try {
      await checkpoint();
      if(reason)await state.event(attemptId,'curriculum-interrupted',{role,reason,allocation_seconds:seconds,
        elapsed_seconds:spent(),accepted:accepted(),session:session.sessionManager.getSessionFile()});
    }finally{
      try{await session.extensionRunner.emit({type:'session_shutdown',reason:'quit'});}finally{session.dispose();}
    }
  }
}

async function artifactFile(root,name,source) {
  const file=resolve(root,name),rel=relative(root,file);
  if(rel==='..'||rel.startsWith('..'+sep) || (source && (file===source || file.startsWith(source+sep))))
    throw new Error('Artifact must be inside this attempt and outside editable source: '+file);
  let info;
  try {info=await stat(file);}
  catch(error) {
    if(error.code!=='ENOENT')throw error;
    throw new Error('Artifact is missing. Create the local file before submitting: '+file);
  }
  if(!info.isFile())throw new Error('Artifact must be a regular file, not a directory: '+file);
  return file;
}
const reply=()=>({content:[{type:'text',text:'Recorded'}],details:{}});
function diagnosticContext(state,id,limit) {
  const tail=options=>state.queryEvents(id,{...options,order:'desc'}).reverse();
  const events=tail({excludeTypes:['pi-event'],limit}).map(({type,data})=>({type,data}));
  const sessions=tail({types:['curriculum-session'],limit}).map(event=>event.data);
  return {session:state.attempt(id)?.session??sessions.at(-1)?.session??null,curriculum_sessions:sessions,
    diagnostics:tail({types:['verification','rejected'],limit:10}).map(({type,data})=>({type,data})),events};
}
export async function investigate({config,job,base,path,state,bridge,signal}) {
  if(!job.attempt?.id)throw new Error('Investigation requires its current attempt');
  const root=join(path,'investigation');await mkdir(root,{recursive:true});
  const source=join(root,'source');
  const prior=job.resume && state.events(job.attempt.id).some(e=>e.type==='curriculum-session'&&e.data.role==='investigation');
  if(!prior)await cp(base.source,source,{recursive:true,verbatimSymlinks:true});
  let proposal=job.resume?state.queryEvents(job.attempt.id,{types:['curriculum-proposal'],order:'desc',limit:1})[0]?.data.proposal:undefined;
  if(!proposal)await curriculumSession({config,path:root,bridge,signal,state,attemptId:job.attempt.id,role:'investigation',resume:job.resume,accepted:()=>Boolean(proposal),prompt:
    `Read the failed project's diagnostics and existing KDB memory. Original goal: ${base.goal}.\n`+
    `Attempt diagnostics: ${JSON.stringify(diagnosticContext(state,job.origin_attempt,20))}\n`+
    `Read the recorded native session and diagnostic evidence files before proposing an investigation.\n`+
    `Create ONE smaller Python investigation about a specific missing fact, with source in ./source and an independent verifier outside ./source. Read relevant documentation using tools/KDB. Preserve the original project and verifier. Submit only the investigation goal, question and verifier filename. Negative results are useful; do not promise the original target will pass.`,
    tool:{name:'submit_investigation',label:'Submit investigation',description:'Declare the smaller investigation after creating its source and independent verifier.',
      parameters:Type.Object({goal:Type.String({minLength:1}),question:Type.String({minLength:1}),verifier:Type.String({minLength:1})}),
      execute:async(_id,args)=>{
        await artifactFile(root,args.verifier,source);
        await state.event(job.attempt.id,'curriculum-proposal',{proposal:args});
        proposal=args;return reply();
      }}});
  if(!proposal)throw new Error('Curriculum did not submit an investigation');
  const verifier=await artifactFile(root,proposal.verifier,source);
  const quote=s=>"'"+s.replaceAll("'","'\\''")+"'";
  const {performance_goal: _parentPerformance, ...investigationBase}=base;
  return {...investigationBase,goal:`${proposal.goal}\nInvestigate: ${proposal.question}`,source,
    actor_seconds:job.actor_seconds,verifier:{files:[verifier],command:`python ${quote(relative(source,verifier))}`}};
}

export async function summarize({config,job,path,state,bridge,passed,signal}) {
  let findings;
  await curriculumSession({config,path,bridge,signal,state,attemptId:job.attempt.id,role:'summary',accepted:()=>Boolean(findings),prompt:
    `Inspect the native session, verification logs and stage events for this ${job.kind??'original'} attempt. `+
    `Workflow/project accepted: ${passed}. Summarize actual findings and failures as compact reusable memory. `+
    `For an investigation, useful=true ONLY when new observed evidence addresses the original gap, even if the investigation failed. `+
    `Attempt diagnostics: ${JSON.stringify(diagnosticContext(state,job.attempt.id,30))}\n`+
    `Controller diagnostic snapshot: ${join(path,'diagnostics.json')}. Read the recorded native session when present and diagnostic evidence files. Submit the compact findings with direct evidence paths.`,
    tool:{name:'record_findings',label:'Record findings',description:'Record evidence-based findings and usefulness.',
      parameters:Type.Object({findings:Type.String({minLength:1,maxLength:4000}),useful:Type.Boolean(),
        paths:Type.Array(Type.String(),{minItems:1,maxItems:16})}),
      execute:async(_id,args)=>{
        for(const name of args.paths)await artifactFile(path,name);
        findings=args;return reply();
      }}});
  if(!findings)throw new Error('Curriculum did not record attempt findings');
  const {createHash}=await import('node:crypto');
  const evidence=[];
  for(const name of findings.paths) {
    const file=await artifactFile(path,name);
    evidence.push({path:file,sha256:createHash('sha256').update(await readFile(file)).digest('hex')});
  }
  return {...findings,evidence};
}
