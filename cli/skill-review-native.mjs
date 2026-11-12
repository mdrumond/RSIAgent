/** Native Pi/KDB and existing curriculum/evaluation adapters for isolated reviews. */
import {mkdir,readFile,writeFile,appendFile,cp,symlink} from 'node:fs/promises';
import {join} from 'node:path';
import {Type} from 'typebox';
import {modelFromProfile} from '../controller/model.mjs';
import {ensureKnowledgeService} from './knowledge.mjs';
import {createKdbBridge} from '../integrations/kdb-extension.mjs';
import {State} from '../controller/state.mjs';
import {importProject} from './instance.mjs';
import {operate} from './runtime.mjs';
import {fileURLToPath} from 'node:url';
import {ownerStopped} from '../controller/lifecycle.mjs';
import {setTimeout as delay} from 'node:timers/promises';
import {refillProjects} from './project-generation.mjs';
import {runMicroeval} from './skill-microeval.mjs';
import {tasks} from './microtasks/tasks.mjs';
import {evaluateNonInferiority} from '../controller/skill-noninferiority.mjs';

export async function reviewSession({directory,config,prompt,validate,signal,seconds,knowledge=false}){
 const sdk=await import('@earendil-works/pi-coding-agent'),agentDir=join(directory,'agent');
 await mkdir(agentDir,{recursive:true});await writeFile(join(directory,'input.json'),JSON.stringify(prompt,null,2));
 const retrievals=[];
 const prior=await readFile(join(directory,'submission.json'),'utf8').catch(e=>{if(e.code!=='ENOENT')throw e;return null;});
 if(prior){const records=await readFile(join(directory,'retrievals.jsonl'),'utf8').catch(()=> '');retrievals.push(...records.split('\n').filter(Boolean).map(line=>JSON.parse(line)));return {value:await validate(JSON.parse(prior),retrievals),retrievals,recovered:true};}
 const bridge=knowledge?createKdbBridge({url:await ensureKnowledgeService(config,{signal}),attemptId:'review-'+directory.split('/').at(-1),agentDir,
  recordRetrieval:async record=>{retrievals.push(record);await appendFile(join(directory,'retrievals.jsonl'),JSON.stringify(record)+'\n');}}):null;
 let submitted;
 const extension=pi=>pi.registerTool({name:'submit_review',label:'Submit review',description:'Submit structured evidence-supported results; fix any validation rejection.',parameters:Type.Object({result:Type.Unknown()}),
  execute:async(_id,{result})=>{submitted=await validate(result,retrievals);await writeFile(join(directory,'submission.json'),JSON.stringify(submitted,null,2));return {content:[{type:'text',text:'Accepted. Finish this session.'}],details:{}};}});
 const sessionFile=join(directory,'native-session.json');let retained;try{retained=JSON.parse(await readFile(sessionFile,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
 const settingsManager=sdk.SettingsManager.inMemory({compaction:{enabled:true},retry:{enabled:true}}),manager=retained?sdk.SessionManager.open(retained.path,directory):sdk.SessionManager.create(directory,directory);
 await writeFile(sessionFile,JSON.stringify({path:manager.getSessionFile()}));
 const loader=new sdk.DefaultResourceLoader({cwd:directory,agentDir,settingsManager,noExtensions:true,noSkills:true,noContextFiles:true,
  extensionFactories:[extension,...(bridge?[bridge.extension]:[])]});await loader.reload();
 const {session}=await sdk.createAgentSession({cwd:directory,agentDir,settingsManager,...await modelFromProfile(config.skill_writer_actor??config.actor),thinkingLevel:'off',
  tools:['read','grep','find','ls','submit_review',...(knowledge?['mcp__kdb__search','mcp__kdb__read_entry']:[])],resourceLoader:loader,sessionManager:manager});
 let expired=false;const abort=()=>{void session.abort();},timer=setTimeout(()=>{expired=true;abort();},seconds*1000);signal?.addEventListener('abort',abort,{once:true});
 try{
  signal?.throwIfAborted();if(bridge)await bridge.knowledge.prepareSession(session,{signal});else await session.bindExtensions({});
  await session.prompt('Read input.json. Perform exactly its task, using only provided observations and retrieved evidence. Submit the requested JSON result using submit_review. Unknown is valid; never invent cause, successful repairs or measurements. Harness faults require user intervention, not actor repairs.');
  await session.waitForIdle();signal?.throwIfAborted();if(expired)throw Error('Review session elapsed allocation exhausted');if(submitted===undefined)throw Error('Review session did not submit');
  return {value:submitted,retrievals,stats:session.getSessionStats(),session:manager.getSessionFile()};
 }finally{
  clearTimeout(timer);signal?.removeEventListener('abort',abort);
  try{await session.exportToJsonl(join(directory,'session.native.jsonl'));}finally{await session.extensionRunner?.emit({type:'session_shutdown',reason:'quit'});await session.dispose();}
 }
}
export async function nativeClassify(input){
 return reviewSession({...input,prompt:{task:'Classify this failure group. Return {cause:actor|harness|specification|provider|unknown,reason,coverage:covered|uncovered|unknown,guidanceIds:[],evidenceRefs:[],gap:missing|insufficient|not-followed|unknown}. Cite occurrence references, use existing guidance IDs only. Availability does not establish the actor read a skill. Inspect copied log files in occurrence.artifacts with native read when the message alone cannot establish a cause; do not access original live artifact paths.',group:{...input.group,occurrenceCount:input.group.occurrences.length,occurrences:input.group.occurrences.slice(0,6)},entries:input.entries}});
}
export async function nativeReview(input){
 const bounded=groups=>groups.map(g=>({...g,occurrenceCount:g.occurrences.length,occurrences:g.occurrences.length>6?[...g.occurrences.slice(0,3),...g.occurrences.slice(-3)]:g.occurrences}));
 const prompt={task:'Review this guidance bucket using KDB search and read_entry. Return {action:update|learn|no-change,reason,groupIds:[],memoryRefs:["collection:entry_id"],changes:[],question,hypothesis,requiredEvidence,learningTarget:cpu|device,learningRequests:[{question,hypothesis,requiredEvidence,learningTarget:cpu|device}]}. Include outstanding learning questions even when proposing a supported update. An update requires retrieved supporting memory, not a failure alone. changes are {action:upsert,id,scope,text,critical:boolean,negative:boolean,evidenceRefs:[occurrence references],response:report-blocker (only for harness guidance)}. Prefer one concise new entry supplementing protected baseline. No device performance claims from CPU results. Detailed kernel cases remain searchable; guidance should explain applicability and retrieval. Questions requiring GPU evidence must stay pending if CPU cannot answer. Harness faults may only produce report/preserve guidance. No new benchmark runs.',groups:bounded(input.groups),entries:input.entries,classification:input.classifications,learning:input.learning??[]};
 return reviewSession({...input,knowledge:true,prompt,validate:(result,retrievals)=>{
  const refs=new Set(retrievals.flatMap(r=>r.entries.map(e=>`${e.collection}:${e.entry_id??e.chunk_id??e.id}`)));
  if(result.action==='update'&&result.memoryRefs?.some(ref=>!refs.has(ref)))throw Error('Memory references must have been retrieved in this session');
  return input.validate(result);
 }});
}
export async function nativeLearn({directory,config,request,signal,seconds}){
 const state=new State(join(directory,'state.sqlite'));
 const begun=performance.now();
 try{
  await cp(join(config.root,'skills'),join(directory,'skills'),{recursive:true});
  await symlink(join(config.root,'knowledge.sqlite'),join(directory,'knowledge.sqlite')).catch(e=>{if(e.code!=='EEXIST')throw e;});
  const cfg={...config,root:directory,target:'cpu',mode:'phase1',actor_seconds:Math.min(300,seconds),curriculum_seconds:Math.min(600,seconds),
   environment:{module:fileURLToPath(new URL('../integrations/environments/index.mjs',import.meta.url)),options:{mode:'local'}},campaign_plan:[{id:'learning',phase:'phase1',new_projects:1}],phase_targets:{phase1:1},curriculum:{generation:{enabled:true,topics:[request.question],preprompt:'Investigate only this CPU-answerable learning question: '+JSON.stringify(request)+'. Keep fixtures tiny, deterministic and independently checked. Record actual observations, not hypothetical device behavior.'}},skills:{...config.skills,review:{enabled:false}}};
  const bridge=createKdbBridge({url:await ensureKnowledgeService(config,{signal}),attemptId:request.id,agentDir:join(directory,'agent')});
  const specs=await refillProjects({config:cfg,state,bridge,concurrency:1,path:join(directory,'drafts'),signal});
  const draft=specs[0];if(!draft){
   const drafts=state.db.prepare('SELECT data FROM generation_drafts').all().map(({data})=>{const d=JSON.parse(data);return {id:d.id,status:d.status,elapsed:d.elapsed,seconds:d.seconds,error:d.error??null};});
   const observations=state.db.prepare("SELECT sequence,draft,type,data FROM generation_events WHERE type IN ('generation-rejected','generation-construction-check','curriculum-interrupted') ORDER BY sequence").all().map(e=>({...e,data:JSON.parse(e.data)}));
   return {status:'inconclusive',reason:'curriculum-no-accepted-project',elapsedSeconds:(performance.now()-begun)/1000,drafts,evidence:observations,claim:'No accepted learning project; constructor observations establish no successful repair'};
  }
  const prior=state.project(draft.id);
  const spec=prior?.spec??await importProject(directory,{...draft,actor_seconds:Math.max(.001,Math.min(300,seconds-(performance.now()-begun)/1000))},{signal});
  if(!prior)state.add(spec);
  cfg.projects=[spec];cfg.curriculum={phase1:[spec.id],generation:{enabled:false}};
  await writeFile(join(directory,'instance.json'),JSON.stringify(cfg,null,2));
  await operate('wave',cfg,{signal,'no-recovery':true});
  const attempts=state.db.prepare('SELECT id,outcome,elapsed FROM attempts WHERE project=?').all(spec.id);
  const passed=attempts.some(a=>a.outcome==='passed');
  return {status:passed?'verified-cpu':'inconclusive',project:spec,evidence:spec.generation.checks,attempts,claim:'Independent CPU verification and actor project evidence only; no GPU claim'};
 }finally{state.close();}
}
export async function nativeEvaluate({directory,active,candidate,config,signal,seconds}){
 let frozen;try{frozen=JSON.parse(await readFile(join(directory,'manifest.json'),'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
 const report=await runMicroeval({directory,skills:{active,candidate},tasks,profile:config.actor,python:config.python??'python3',runs:2,seconds:90,
  totalSeconds:frozen?.totalSeconds??seconds,signal,resume:true,shadow:true});
 return {decision:evaluateNonInferiority(report.results,{projectIds:tasks.map(t=>t.id),runs:2,margin:.05}),report};
}

export async function stopReviewKnowledge(root){
 let receipt;try{receipt=JSON.parse(await readFile(join(root,'knowledge-service.json'),'utf8'));}catch(e){if(e.code==='ENOENT')return;throw e;}
 if(await ownerStopped(receipt.owner))return;process.kill(receipt.owner.pid,'SIGTERM');
 const end=performance.now()+5000;while(!await ownerStopped(receipt.owner)&&performance.now()<end)await delay(50);
 if(!await ownerStopped(receipt.owner))throw Error('Owned review KDB service remains live: '+root);
}
