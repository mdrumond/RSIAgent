/** Native Pi/KDB and existing curriculum/evaluation adapters for isolated reviews. */
import {mkdir,readFile,writeFile,appendFile,cp,symlink,realpath,stat} from 'node:fs/promises';
import {join,resolve} from 'node:path';
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
import {runMicroeval,finalizeKernelPairs} from './skill-microeval.mjs';
import {tasks} from './microtasks/tasks.mjs';
import {stages as workflowStages} from '../controller/workflow.mjs';
import {diagnosticIdentity} from '../controller/skill-review.mjs';
import {evaluateSkillObjective,evaluationSettings} from '../controller/skill-objectives.mjs';

async function reviewEvidence(directory,groups=[]){
 const required=new Set(),previews=[];await mkdir(join(directory,'evidence'),{recursive:true});
 for(const group of groups){
  const representatives=[...new Map(group.occurrences.map(o=>[diagnosticIdentity(o),o])).values()];
  const linked=group.linkedDiagnostics??[],linkedFile=join(directory,'evidence',group.id+'-linked-diagnostics.jsonl');
  await writeFile(linkedFile,linked.map(d=>JSON.stringify(d)).join('\n')+'\n');linked.forEach(d=>required.add(d.reference));
  const all=join(directory,'evidence',group.id+'-occurrences.jsonl'),diagnostics=join(directory,'evidence',group.id+'-diagnostics.jsonl');
  await writeFile(all,group.occurrences.map(o=>JSON.stringify(o)).join('\n')+'\n');
  await writeFile(diagnostics,representatives.map(o=>JSON.stringify(o)).join('\n')+'\n');
  representatives.forEach(o=>required.add(o.reference));
  previews.push({...group,linkedDiagnostics:linked.slice(0,6),linkedDiagnosticCount:linked.length,linkedDiagnosticsFile:linkedFile,occurrences:representatives.slice(0,6),occurrenceCount:group.occurrences.length,
   distinctDiagnostics:representatives.length,fullOccurrences:all,diagnosticsFile:diagnostics});
 }
 return {required,previews};
}
export async function reviewSession({directory,config,prompt,validate,signal,seconds,knowledge=false,evidenceGroups=[],allowedPaths=[]}){
 const sdk=await import('@earendil-works/pi-coding-agent'),agentDir=join(directory,'agent');
 await mkdir(agentDir,{recursive:true});
 // Use the pinned Pi path resolver so policy and native filesystem tools agree.
 const {resolveToCwd}=await import(new URL('./core/tools/path-utils.js',import.meta.resolve('@earendil-works/pi-coding-agent')).href);
 const declarations=[directory,...evidenceGroups.flatMap(g=>[...g.occurrences,...(g.linkedDiagnostics??[])].flatMap(o=>(o.artifacts??[]).map(a=>a.path))),...allowedPaths];
 const allowed=await Promise.all([...new Set(declarations)].map(async path=>{try{const canonical=await realpath(path);return {path:canonical,directory:(await stat(canonical)).isDirectory()};}catch(error){if(error.code!=='ENOENT')throw error;return {path:resolve(path),directory:false};}}));
 const evidence=await reviewEvidence(directory,evidenceGroups);
 if(prompt.group)prompt={...prompt,group:evidence.previews[0]};if(prompt.groups)prompt={...prompt,groups:evidence.previews};
 await writeFile(join(directory,'input.json'),JSON.stringify(prompt,null,2));
 const records=await readFile(join(directory,'retrievals.jsonl'),'utf8').catch(e=>{if(e.code!=='ENOENT')throw e;return '';});
 const retrievals=records.split('\n').filter(Boolean).map(line=>JSON.parse(line));
 const coverageFile=join(directory,'read-evidence.json');
 const seen=new Set(JSON.parse(await readFile(coverageFile,'utf8').catch(e=>{if(e.code!=='ENOENT')throw e;return '[]';})));
 const check=async result=>{
  if([...evidence.required].some(ref=>!seen.has(ref)))throw Error('Read every distinct diagnostic and linked diagnostic using native read with offset/limit on diagnosticsFile/linkedDiagnosticsFile before submitting; duplicate occurrences need no repeated reads');
  return validate(result,retrievals);
 };
 const prior=await readFile(join(directory,'submission.json'),'utf8').catch(e=>{if(e.code!=='ENOENT')throw e;return null;});
 if(prior)return {value:await check(JSON.parse(prior)),retrievals,recovered:true};
 const bridge=knowledge?createKdbBridge({url:await ensureKnowledgeService(config,{signal}),attemptId:'review-'+directory.split('/').at(-1),agentDir,
  recordRetrieval:async record=>{retrievals.push(record);await appendFile(join(directory,'retrievals.jsonl'),JSON.stringify(record)+'\n');}}):null;
 let submitted;
 const extension=pi=>{
  pi.on('tool_call',async event=>{if(!['read','grep','find','ls'].includes(event.toolName))return;
   const requested=resolveToCwd(event.input.path??'.',directory);
   const path=await realpath(requested).catch(error=>{if(error.code!=='ENOENT')throw error;return requested;});
   if(allowed.some(a=>path===a.path||a.directory&&path.startsWith(a.path+'/')))return;
   await appendFile(join(directory,'tool-policy.jsonl'),JSON.stringify({tool:event.toolName,path,blocked:true,toolCallId:event.toolCallId})+'\n');
   return {block:true,reason:'Filesystem inspection is limited to this review directory and explicitly copied evidence/verified learning paths. Read input.json and correct the submitted fields; use KDB MCP search/read_entry for indexed knowledge, not workspace searches.'};
  });
  pi.on('tool_result',async event=>{if(event.toolName!=='read'||event.isError)return;
   const path=resolve(directory,event.input.path??'');if(path!==join(directory,'input.json')&&!path.startsWith(join(directory,'evidence')+'/'))return;
   const text=(event.content??[]).filter(c=>c.type==='text').map(c=>c.text).join('\n');
   for(const ref of evidence.required){const quoted=JSON.stringify(ref);if(text.includes('\"reference\":'+quoted)||text.includes('\"reference\": '+quoted))seen.add(ref);}
   await writeFile(coverageFile,JSON.stringify([...seen]));
  });
  pi.registerTool({name:'submit_review',label:'Submit review',description:'Submit structured evidence-supported results; fix any validation rejection.',parameters:Type.Object({result:Type.Unknown()}),
  execute:async(_id,{result})=>{submitted=await check(result);await writeFile(join(directory,'submission.json'),JSON.stringify(submitted,null,2));return {content:[{type:'text',text:'Accepted. Finish this session.'}],details:{}};}});};
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
  await session.prompt('Read input.json, then read all distinct diagnostics listed by diagnosticsFile and linkedDiagnosticsFile using native read offset/limit when the bounded preview is incomplete. Full occurrences remain available in fullOccurrences; duplicate diagnostics need not be reread. Filesystem tools may only inspect this review directory and declared copied evidence/verified learning paths; indexed knowledge is available through KDB tools. Correct validation errors directly rather than searching implementation source. Perform exactly its task, using only provided observations and retrieved evidence. Submit the requested JSON result using submit_review. Unknown is valid; never invent cause, successful repairs or measurements. Harness faults require user intervention, not actor repairs.');
  await session.waitForIdle();signal?.throwIfAborted();if(expired)throw Error('Review session elapsed allocation exhausted');if(submitted===undefined)throw Error('Review session did not submit');
  return {value:submitted,retrievals,stats:session.getSessionStats(),session:manager.getSessionFile()};
 }finally{
  clearTimeout(timer);signal?.removeEventListener('abort',abort);
  try{await session.exportToJsonl(join(directory,'session.native.jsonl'));}finally{await session.extensionRunner?.emit({type:'session_shutdown',reason:'quit'});await session.dispose();}
 }
}
export async function nativeClassify(input){
 return reviewSession({...input,evidenceGroups:[input.group],prompt:{task:`Classify this failure group. ${input.bucketPolicy?'Bucket is orthogonal to fault ownership: harness needs user intervention, correctness concerns functional behavior, optimization concerns kernel latency.':''} Return {cause:actor|harness|specification|provider|unknown,${input.bucketPolicy?'bucket:harness|correctness|optimization,':''}reason,coverage:covered|uncovered|unknown,guidanceIds:[],evidenceRefs:[],gap:missing|insufficient|not-followed|unknown}. Cite occurrence references, use existing guidance IDs only. Availability does not establish the actor read a skill. A pre-actor stage or wrapper label does NOT prove a harness root cause: read the linked same-attempt negative verdicts and copied logs, distinguish invalid specifications/baseline APIs from harness faults, and retain unknown if the actual cause is not established. Inspect copied log files in occurrence.artifacts with native read when the message alone cannot establish a cause; do not access original live artifact paths.`,group:input.group,entries:input.entries}});
}
export async function nativeClassifyBatch(input){
 return reviewSession({...input,evidenceGroups:input.groups,prompt:{task:`Classify EVERY failure group using the shared guidance. ${input.bucketPolicy?'Bucket is orthogonal to fault ownership: harness needs user intervention, correctness concerns functional behavior, optimization concerns kernel latency.':''} Return {classifications:[{groupId,cause:actor|harness|specification|provider|unknown,${input.bucketPolicy?'bucket:harness|correctness|optimization,':''}reason,coverage:covered|uncovered|unknown,guidanceIds:[],evidenceRefs:[],gap:missing|insufficient|not-followed|unknown}]}. Include each supplied groupId exactly once; keep each reason at most two short sentences. Cite actual occurrence/linked-diagnostic references. covered requires existing guidance IDs; unknown/uncovered requires guidanceIds=[]. Read actual copied diagnostics before attributing ownership; a pre-actor stage alone is not proof of a harness fault. Preserve unknown when evidence is insufficient.`,groups:input.groups,entries:input.entries}});
}
export async function nativeReview(input){
 const prompt={guidanceScopes:['global-kdb',...workflowStages],task:'Review this guidance bucket using KDB search and read_entry. Return {action:update|learn|no-change,reason,groupIds:[],memoryRefs:["collection:entry_id"],changes:[],question,hypothesis,requiredEvidence,learningTarget:cpu|device,learningRequests:[{groupIds:[originating failure group IDs],question,hypothesis,requiredEvidence,learningTarget:cpu|device}]}. Include outstanding learning questions even when proposing a supported update. Each learning request must identify its own originating groupIds, a nonempty subset of this bucket; do not attach unrelated harness faults to an actor/specification question. An update requires retrieved supporting memory, not a failure alone. For KDB search results, chunk_id is the entry_id argument to read_entry; do not substitute a lesson hash, Git hash or filename. memoryRefs uses exact collection:chunk_id from actual retrieval. evidenceRefs uses exact failure occurrence/linked-diagnostic IDs, not KDB IDs. Use only the enumerated guidanceScopes for changes.scope; correctness/optimization/harness are evaluation buckets, not guidance scopes. changes are {action:upsert,id,scope,text,critical:boolean,negative:boolean,evidenceRefs:[occurrence references],response:actor-change|report-blocker|reserve-validation-runway|retrieve-scoped-evidence}. report-blocker and reserve-validation-runway require scope=global-kdb or scope=preflight; fix that scope field rather than reordering changes. actor-change is the valid response label for actor-owned guidance. Prefer one concise new entry supplementing protected baseline. No device performance claims from CPU results or unsupported duration/speed claims. Cumulative actor clocks are not operation timings. Keep references and guidance compact. Detailed kernel cases remain searchable; guidance should explain applicability and retrieval. Questions requiring GPU evidence must stay pending if CPU cannot answer. Harness faults may only produce report/preserve guidance. No new benchmark runs.',groups:input.groups,entries:input.entries,classification:input.classifications,learning:input.learning??[]};
 const allowedPaths=(input.learning??[]).flatMap(([,record])=>record.result?.status==='verified-cpu'?[record.result.project?.source,...(record.result.project?.verifier?.files??[]),...(record.result.evidence??[]).flatMap(e=>e.log?[e.log]:[])].filter(Boolean):[]);
 return reviewSession({...input,allowedPaths,evidenceGroups:input.groups,knowledge:true,prompt,validate:(result,retrievals)=>{
  const refs=new Set(retrievals.flatMap(r=>r.entries.map(e=>`${e.collection}:${e.entry_id??e.chunk_id??e.id}`)));
  if(result.action==='update'&&result.memoryRefs?.some(ref=>!refs.has(ref)))throw Error('Memory references must have been retrieved in this session');
  return input.validate(result);
 }});
}
// QA deliberately reuses retrieved evidence rather than reopening KDB or executing kernels.
async function qaPrompt(input){
 let memory=input.memoryEvidence;
 if(!memory){
  const records=await readFile(join(input.proposalDirectory??input.directory,'retrievals.jsonl'),'utf8').catch(error=>{if(error.code!=='ENOENT')throw error;return '';});
  memory=records.split('\n').filter(Boolean).flatMap(line=>JSON.parse(line).entries??[]);
 }
 const cited=new Set(input.proposal.memoryRefs??[]);
 memory=memory.filter(item=>cited.has(item.ref??item.citationId??`${item.collection}:${item.entry_id??item.chunk_id??item.id}`));
 const refs=new Set((input.proposal.changes??[]).flatMap(change=>change.evidenceRefs??[]));
 const groups=input.groups.filter(group=>[...group.occurrences,...(group.linkedDiagnostics??[])].some(item=>refs.has(item.reference))).map(group=>({...group,occurrences:group.occurrences.filter(item=>refs.has(item.reference)),linkedDiagnostics:group.linkedDiagnostics??[]}));
 return {proposal:input.proposal,baseline:input.entries,objective:input.objective,target:input.target,groups,memory,
  learning:input.learning??[],questions:{action:'What is the smallest concrete actor edit, and what must remain unchanged?',mechanism:'How does that edit prevent the observed failure?',example:'Give a worked example and a counterexample or applicability boundary.',evidence:'Separate observation, mathematical deduction, and untested claims; state target applicability.',objective:'Is this correctness or optimization? Does the instruction demand unnecessary optimization?'},
  rules:'Only supplied evidence is available. Do not run benchmarks or search live sources. Unsupported facts must remain explicit questions. Performance hypotheses need not already prove a speedup. The actor sees the skill text, not these answers.'};
}
export async function nativeQaAnswer(input,dependencies={}){
 const prompt=await qaPrompt(input);
 const allowedPaths=(input.learning??[]).flatMap(([,record])=>record.result?.status==='verified-cpu'?[record.result.project?.source,...(record.result.project?.verifier?.files??[]),...(record.result.evidence??[]).flatMap(e=>e.log?[e.log]:[])].filter(Boolean):[]);
 return (dependencies.session??reviewSession)({...input,allowedPaths,knowledge:false,evidenceGroups:prompt.groups,prompt:{...prompt,previousReview:input.previousReview,
  task:'Answer all five questions. Return {answers:{action:{text,refs:[]},mechanism:{text,refs:[]},example:{text,refs:[]},evidence:{text,refs:[]},objective:{text,refs:[]}},changes?:[],memoryRefs?:[]}. refs must identify supplied failure references, baseline IDs or already cited memory. Revise changes when necessary so essential actionable advice and boundaries appear in the actual skill text. Preserve the existing change schema. Do not merely explain a missing instruction in private answers.'}});
}
export async function nativeQaReview(input,dependencies={}){
 const prompt=await qaPrompt(input);
 const allowedPaths=(input.learning??[]).flatMap(([,record])=>record.result?.status==='verified-cpu'?[record.result.project?.source,...(record.result.project?.verifier?.files??[]),...(record.result.evidence??[]).flatMap(e=>e.log?[e.log]:[])].filter(Boolean):[]);
 return (dependencies.session??reviewSession)({...input,allowedPaths,knowledge:false,evidenceGroups:prompt.groups,prompt:{...prompt,answers:input.answers,
  task:'Independently review the draft and five answers against supplied evidence. Return {verdict:approved|revise|learn|inconclusive,reason,issues?:[],learningRequests?:[{groupIds:[],question,hypothesis,requiredEvidence,learningTarget:cpu|device}]}. Approve only if the skill text itself communicates the concrete edit, mechanism and material boundaries; private answers cannot repair absent guidance. Gate material contradictions, unsupported target claims and unnecessary optimization in correctness guidance. Do not demand prior performance proof of an explicitly untested optimization hypothesis. Use revise for fixable writing issues, learn for missing facts requiring CPU/device learning, inconclusive for insufficient review evidence. Harness problems remain user handled. Cite concrete issues; approval admits A/B only.'}});
}
export async function nativeLearn({directory,config,request,signal,seconds},dependencies={}){
 const ops={construct:refillProjects,importProject,operate,ensureKnowledgeService,createKdbBridge,...dependencies};
 const state=new State(join(directory,'state.sqlite')),begun=performance.now();
 try{
  const existing=state.db.prepare('SELECT id FROM projects ORDER BY rowid LIMIT 2').all();
  if(existing.length>1)throw Error('Learning instance must contain only its original project');
  let cfg;try{cfg=JSON.parse(await readFile(join(directory,'instance.json'),'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
  cfg??={...config,root:directory,target:'cpu',mode:'phase1',actor_seconds:Math.min(300,seconds),curriculum_seconds:Math.min(600,seconds),
   environment:{module:fileURLToPath(new URL('../integrations/environments/index.mjs',import.meta.url)),options:{mode:'local'}},campaign_plan:[{id:'learning',phase:'phase1',new_projects:1}],phase_targets:{phase1:1},curriculum:{generation:{enabled:true,topics:[request.question],preprompt:'Investigate only this CPU-answerable learning question: '+JSON.stringify(request)+'. Keep fixtures tiny, deterministic and independently checked. Record actual observations, not hypothetical device behavior.'}},skills:{...config.skills,review:{enabled:false}}};
  let spec=existing.length?state.project(existing[0].id).spec:null;
  if(!spec){
   await cp(join(config.root,'skills'),join(directory,'skills'),{recursive:true});
   await symlink(join(config.root,'knowledge.sqlite'),join(directory,'knowledge.sqlite')).catch(e=>{if(e.code!=='EEXIST')throw e;});
   const bridge=ops.createKdbBridge({url:await ops.ensureKnowledgeService(config,{signal}),attemptId:request.id,agentDir:join(directory,'agent')});
   const drafts=await ops.construct({config:cfg,state,bridge,concurrency:1,path:join(directory,'drafts'),signal});
   if(!drafts[0]){
    const rows=state.db.prepare('SELECT data FROM generation_drafts').all().map(({data})=>{const d=JSON.parse(data);return {id:d.id,status:d.status,elapsed:d.elapsed,seconds:d.seconds,error:d.error??null};});
    const observations=state.db.prepare("SELECT sequence,draft,type,data FROM generation_events WHERE type IN ('generation-rejected','generation-construction-check','curriculum-interrupted') ORDER BY sequence").all().map(e=>({...e,data:JSON.parse(e.data)}));
    return {status:'inconclusive',reason:'curriculum-no-accepted-project',elapsedSeconds:(performance.now()-begun)/1000,drafts:rows,evidence:observations,claim:'No accepted learning project; constructor observations establish no successful repair'};
   }
   spec=await ops.importProject(directory,{...drafts[0],actor_seconds:Math.max(.001,Math.min(300,seconds-(performance.now()-begun)/1000))},{signal});
   state.add(spec);
  }
  cfg.actor_seconds=spec.actor_seconds;cfg.projects=[spec];cfg.curriculum={phase1:[spec.id],generation:{enabled:false}};
  await writeFile(join(directory,'instance.json'),JSON.stringify(cfg,null,2));
  const attempts=()=>state.db.prepare(`SELECT a.id,a.outcome,a.elapsed,a.session,
   EXISTS(SELECT 1 FROM events e WHERE e.attempt=a.id AND e.type='postprocess-completed') AS postprocessed FROM attempts a WHERE project=?`).all(spec.id);
  const previous=attempts();
  if(!previous.length||previous.some(a=>a.outcome===null||!a.postprocessed))await ops.operate(previous.length?'resume':'wave',cfg,{signal,'no-recovery':true});
  signal?.throwIfAborted();const completed=attempts(),passed=completed.some(a=>a.outcome==='passed'&&a.postprocessed),unfinished=completed.some(a=>!a.postprocessed);
  return {status:passed?'verified-cpu':unfinished?'postprocessing-pending':'inconclusive',project:spec,evidence:spec.generation?.checks??[],attempts:completed,claim:'Independent CPU verification and actor project evidence only; no GPU claim'};
 }finally{state.close();}
}
export async function nativeEvaluate({directory,active,candidate,config,signal,seconds,objective},dependencies={}){
 const ops={runMicroeval,finalizeKernelPairs,...dependencies},settings=objective&&objective!=='legacy'?evaluationSettings(config.skill_review_evaluations,objective):null;
 const selected=settings?.tasks??config.skill_review_tasks??tasks;
 let frozen;try{frozen=JSON.parse(await readFile(join(directory,'manifest.json'),'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
 if(settings&&frozen&&frozen.totalSeconds!==settings.developmentSeconds)throw Error('Frozen development allocation changed');
 let report=await ops.runMicroeval({directory,objective,binding:settings?.binding,knowledge:settings?.knowledge,skills:{active,candidate},tasks:selected,profile:config.actor,python:config.python??'python3',runs:2,seconds:settings?180:90,verificationSeconds:settings?.verificationSeconds??0,
  totalSeconds:frozen?.totalSeconds??settings?.developmentSeconds??seconds,signal,resume:true,shadow:true});
 if(objective==='optimization'&&report.status==='complete')report=await ops.finalizeKernelPairs({directory,python:config.python??'python3',seconds:settings.pairedSeconds,signal});
 return {decision:evaluateSkillObjective(report.results,{objective,projectIds:selected.map(t=>t.id),runs:2,margin:.05,pairedTimings:report.pairedTimings}),report};
}

export async function stopReviewKnowledge(root){
 let receipt;try{receipt=JSON.parse(await readFile(join(root,'knowledge-service.json'),'utf8'));}catch(e){if(e.code==='ENOENT')return;throw e;}
 if(await ownerStopped(receipt.owner))return;process.kill(receipt.owner.pid,'SIGTERM');
 const end=performance.now()+5000;while(!await ownerStopped(receipt.owner)&&performance.now()<end)await delay(50);
 if(!await ownerStopped(receipt.owner))throw Error('Owned review KDB service remains live: '+root);
}

export async function evaluationSettled({directory}){
 let report;try{report=JSON.parse(await readFile(join(directory,'results.json'),'utf8'));}catch(e){if(e.code==='ENOENT')return true;throw e;}
 return !report.runningSample&&!report.pairedReplay?.runningPair&&(!report.owner||await ownerStopped(report.owner));
}
