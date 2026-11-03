import * as sdk from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';
import { cp, mkdir, readFile, stat } from 'node:fs/promises';
import { join, resolve, relative, sep } from 'node:path';
import { modelFromProfile } from '../controller/model.mjs';

async function curriculumSession({config,path,bridge,prompt,tool,signal}) {
  const agentDir=join(path,'agent');await mkdir(agentDir,{recursive:true});
  const settingsManager=sdk.SettingsManager.inMemory({compaction:{enabled:true},retry:{enabled:true}});
  const loader=new sdk.DefaultResourceLoader({cwd:path,agentDir,settingsManager,noExtensions:true,noSkills:true,
    extensionFactories:[bridge.extension,pi=>pi.registerTool(tool)]});
  await loader.reload();
  const {session}=await sdk.createAgentSession({cwd:path,agentDir,settingsManager,...await modelFromProfile(config.curriculum_actor??config.actor),
    thinkingLevel:'off',tools:['read','write','edit','mcp__kdb__*',tool.name],resourceLoader:loader,sessionManager:sdk.SessionManager.create(path,join(path,'sessions'))});
  session.agent.toolExecution='sequential';await session.bindExtensions({});
  const abort=()=>void session.abort();signal?.addEventListener('abort',abort,{once:true});
  const timer=setTimeout(()=>void session.abort(),(config.curriculum_seconds??240)*1000);
  try {
    if(signal?.aborted)throw new Error('Curriculum interrupted');
    await session.prompt(prompt);await session.waitForIdle();
    if(signal?.aborted)throw new Error('Curriculum interrupted');
  }finally{signal?.removeEventListener('abort',abort);clearTimeout(timer);
    try{await session.extensionRunner.emit({type:'session_shutdown',reason:'quit'});}finally{session.dispose();}}
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
  const events=state.events(id).filter(event=>event.type!=='pi-event');
  return {session:state.attempt(id)?.session??null,
    diagnostics:events.filter(event=>event.type==='verification'||event.type==='rejected').slice(-10),
    events:events.slice(-limit)};
}
export async function investigate({config,job,base,path,state,bridge,signal}) {
  const root=join(path,'investigation');await mkdir(root,{recursive:true});
  const source=join(root,'source');
  await cp(base.source,source,{recursive:true});
  let proposal;
  await curriculumSession({config,path:root,bridge,signal,prompt:
    `Read the failed project's diagnostics and existing KDB memory. Original goal: ${base.goal}.\n`+
    `Attempt diagnostics: ${JSON.stringify(diagnosticContext(state,job.origin_attempt,20))}\n`+
    `Read the recorded native session and diagnostic evidence files before proposing an investigation.\n`+
    `Create ONE smaller Python investigation about a specific missing fact, with source in ./source and an independent verifier outside ./source. Read relevant documentation using tools/KDB. Preserve the original project and verifier. Submit only the investigation goal, question and verifier filename. Negative results are useful; do not promise the original target will pass.`,
    tool:{name:'submit_investigation',label:'Submit investigation',description:'Declare the smaller investigation after creating its source and independent verifier.',
      parameters:Type.Object({goal:Type.String({minLength:1}),question:Type.String({minLength:1}),verifier:Type.String({minLength:1})}),
      execute:async(_id,args)=>{await artifactFile(root,args.verifier,source);proposal=args;return reply();}}});
  if(!proposal)throw new Error('Curriculum did not submit an investigation');
  const verifier=await artifactFile(root,proposal.verifier,source);
  const quote=s=>"'"+s.replaceAll("'","'\\''")+"'";
  const {performance_goal: _parentPerformance, ...investigationBase}=base;
  return {...investigationBase,goal:`${proposal.goal}\nInvestigate: ${proposal.question}`,source,
    actor_seconds:job.actor_seconds,verifier:{files:[verifier],command:`python ${quote(relative(source,verifier))}`}};
}

export async function summarize({config,job,path,state,bridge,passed,signal}) {
  let findings;
  await curriculumSession({config,path,bridge,signal,prompt:
    `Inspect the native session, verification logs and stage events for this ${job.kind??'original'} attempt. `+
    `Frozen verifier passed: ${passed}. Summarize actual findings and failures as compact reusable memory. `+
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
