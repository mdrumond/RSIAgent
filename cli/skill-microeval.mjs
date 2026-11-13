#!/usr/bin/env node
/** Short native-Pi skill comparisons. Shadow artifacts never update a campaign. */
import {mkdir,readFile,writeFile,rename,realpath,appendFile} from 'node:fs/promises';
import {resolve,join,relative,dirname,isAbsolute} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {spawn} from 'node:child_process';
import {parseArgs} from 'node:util';
import {processIdentity,ownerStopped} from '../controller/lifecycle.mjs';
import {createEnvironmentTools} from '../integrations/environments/index.mjs';
import {createLocalExecutionOperations} from '../integrations/environments/local-bash.mjs';
import {createKdbBridge} from '../integrations/kdb-extension.mjs';
import {evaluateSkillObjective,retrievalIdentity} from '../controller/skill-objectives.mjs';
import {modelFromProfile} from '../controller/model.mjs';

const hash=value=>createHash('sha256').update(value).digest('hex');
function unknownExecution(error){
 if(error.code!=='RSI_EXECUTION_UNRESOLVED'&&!/unresolved|(?:identity|closure) unknown|do not resubmit/i.test(error.message??''))return null;
 return {handle:error.handle??error.message?.match(/remote:[^\s:]+:(?:job|transfer|command|session):[\w-]+|local:[\w-]+/)?.[0]??null};
}
async function save(path,value){const tmp=path+'.'+randomUUID();await writeFile(tmp,JSON.stringify(value,null,2)+'\n');await rename(tmp,path);}
const inside=(root,name)=>{const path=resolve(root,name),rel=relative(root,path);if(isAbsolute(name)||!rel||rel==='..'||rel.startsWith('../'))throw Error('Task files must be relative descendants');return path;};
async function skillInput(directory){
 const root=resolve(directory),map=JSON.parse(await readFile(join(root,'skill-map.json'),'utf8')),files={};
 for(const [scope,name] of Object.entries(map))files[scope]=await readFile(inside(root,name),'utf8');
 return {files,sha256:hash(JSON.stringify(files)),text:Object.entries(files).map(([scope,text])=>`## ${scope}\n${text}`).join('\n\n')};
}
async function check(python,checker,cwd,seconds,signal,args=[cwd]){
 signal?.throwIfAborted();
 const child=spawn(python,[checker,...args],{cwd,stdio:['ignore','pipe','pipe'],env:{...process.env,PYTHONDONTWRITEBYTECODE:'1'}});
 let output='',error='';child.stdout.on('data',d=>{output=(output+d).slice(-8192);});child.stderr.on('data',d=>{error=(error+d).slice(-8192);});
 const abort=()=>child.kill('SIGKILL');signal?.addEventListener('abort',abort,{once:true});
 const timer=setTimeout(abort,Math.max(1,seconds*1000));
 try{return await new Promise((done,reject)=>{child.once('error',reject);child.once('close',(code,signal)=>done({correct:code===0,exitCode:code,signal,output,error}));});}
 finally{clearTimeout(timer);signal?.removeEventListener('abort',abort);}
}
const canonical=value=>Array.isArray(value)?value.map(canonical):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(k=>[k,canonical(value[k])])):value;
export function protectedFingerprint(text,mode='bytes'){
 if(!['bytes','json'].includes(mode))throw Error('Protected fingerprint mode must be bytes or json');
 return hash(mode==='json'?JSON.stringify(canonical(JSON.parse(text))):text);
}
function lastReceipt(output){return JSON.parse(output.trim().split('\n').at(-1));}
export async function validateKernelReceipt(verification,cwd){
 let receipt;try{receipt=lastReceipt(verification.output);}catch(error){if(verification.exitCode===125)return verification;throw error;}
 if(verification.exitCode===125){
  if(receipt.correct===null&&receipt.bucket==='harness'&&receipt.kernelSha256===hash(await readFile(join(cwd,'kernel.py')))&&receipt.evidence&&typeof receipt.evidence==='object'&&!Array.isArray(receipt.evidence)&&typeof receipt.evidence.reason==='string'&&receipt.evidence.reason.trim()&&(receipt.latencySeconds===undefined||receipt.latencySeconds===null))return {...verification,correct:null,receipt};
  return verification;
 }
 if(receipt.correct!==true||!Number.isFinite(receipt.latencySeconds)||receipt.latencySeconds<=0||
  receipt.kernelSha256!==hash(await readFile(join(cwd,'kernel.py')))||!receipt.evidence||typeof receipt.evidence!=='object')throw Error('Invalid trusted kernel verification receipt');
 return {...verification,receipt};
}
export async function nativeSample({cwd,guidance,prompt,profile,seconds,signal,sessionDirectory,resumeSession,kind,knowledge,targetEnvironment,verificationReserved=0}){
 const begun=performance.now();
 const sdk=await import('@earendil-works/pi-coding-agent');
 const agentDir=join(sessionDirectory,'agent');await mkdir(agentDir,{recursive:true});
 const settingsManager=sdk.SettingsManager.inMemory({compaction:{enabled:true},retry:{enabled:true}});
 let unresolvedExecution;
 const sampleAbort=new AbortController(),sampleSignal=signal?AbortSignal.any([signal,sampleAbort.signal]):sampleAbort.signal,remaining=()=>Math.max(.001,seconds-(performance.now()-begun)/1000),pending=new Set();
 const local=createLocalExecutionOperations(sdk),{resolveToCwd}=await import(new URL('./core/tools/path-utils.js',import.meta.resolve('@earendil-works/pi-coding-agent')).href),taskRoot=await realpath(cwd);
 const bridge=knowledge?createKdbBridge({url:knowledge.url,attemptId:hash(sessionDirectory),agentDir}):null;
 const extension=async pi=>{
  pi.on('tool_call',async event=>{
   if(event.toolName.startsWith('mcp__kdb__')&&!['mcp__kdb__search','mcp__kdb__read_entry'].includes(event.toolName))return {block:true,reason:'Evaluation knowledge is read-only: use search or read_entry.'};
   let outside=false;
   if(['read','write','edit','grep','find','ls'].includes(event.toolName)){
    const path=resolveToCwd(event.input.path??'.',cwd),canonical=await realpath(path).catch(error=>{if(error.code!=='ENOENT')throw error;return path;});
    outside=canonical!==taskRoot&&!canonical.startsWith(taskRoot+'/');
   }
   // Shell directory changes/searches are unnecessary: native tools already use the task cwd.
   const shellEscape=event.toolName==='bash'&&/(?:^|[;&|()\n])\s*(?:cd|grep|rg|find)\b/.test(event.input.command);
   if(!outside&&!shellEscape)return;
   await appendFile(join(sessionDirectory,'tool-policy.jsonl'),JSON.stringify({tool:event.toolName,blocked:true,toolCallId:event.toolCallId})+'\n');
   return {block:true,reason:'Keep filesystem work inside the current task. Bash directory changes and shell searches are blocked; use native read/grep/find/ls scoped to this task. Run the declared local oracle command directly without changing directory.'};
  });
  if(targetEnvironment){
   const [target]=await createEnvironmentTools({execute:async(command,path,options)=>{
    const merged=options.signal?AbortSignal.any([sampleSignal,options.signal]):sampleSignal;merged.throwIfAborted();
    const operation=targetEnvironment.execute(command,path,{...options,signal:merged,timeout:Math.min(options.timeout??Infinity,remaining())});pending.add(operation);
    try{const result=await operation;if(!result.settled){unresolvedExecution={handle:result.handle??null};interrupt(Error('Target execution remains unresolved: '+result.handle));}return result;}
    catch(error){const unknown=unknownExecution(error);if(unknown){unresolvedExecution=unknown;interrupt(error);}throw error;}finally{pending.delete(operation);}
   }},cwd,sdk);pi.registerTool(target);
  }
  pi.registerTool(sdk.createBashToolDefinition(cwd,{operations:{exec:async(command,path,options)=>{
   const merged=options.signal?AbortSignal.any([sampleSignal,options.signal]):sampleSignal;merged.throwIfAborted();
   const operation=local.exec(command,path,{...options,signal:merged,timeout:Math.min(options.timeout??Infinity,remaining())});pending.add(operation);
   try{return await operation;}finally{pending.delete(operation);}
  }}}));
 };
 const loader=new sdk.DefaultResourceLoader({cwd,agentDir,settingsManager,noExtensions:true,noSkills:true,noContextFiles:true,extensionFactories:[extension,...(bridge?[bridge.extension]:[])]});await loader.reload();
 const manager=resumeSession?sdk.SessionManager.open(resumeSession,sessionDirectory):sdk.SessionManager.create(cwd,sessionDirectory);
 const {session}=await sdk.createAgentSession({cwd,agentDir,settingsManager,...await modelFromProfile(profile),thinkingLevel:'off',
  tools:['read','write','edit','grep','find','ls','bash',...(targetEnvironment?['rsi_target']:[]),...(bridge?['mcp__kdb__search','mcp__kdb__read_entry']:[])],resourceLoader:loader,sessionManager:manager});
 let timedOut=false,failure,lastAssistant;
 const unsubscribe=session.subscribe(event=>{if(event.type==='message_end'&&event.message.role==='assistant')lastAssistant=event.message;});
 let aborting;const interrupt=reason=>{sampleAbort.abort(reason);aborting??=session.abort();void aborting.catch(()=>{});};
 const timer=setTimeout(()=>{timedOut=true;interrupt(Error('Sample elapsed allocation exhausted'));},Math.max(1,seconds*1000-(performance.now()-begun)));
 const abort=()=>interrupt(signal.reason??Error('Sample interrupted'));signal?.addEventListener('abort',abort,{once:true});
 try{
  signal?.throwIfAborted();if(performance.now()-begun>=seconds*1000){timedOut=true;throw Error('Sample elapsed allocation exhausted');}
  if(bridge)await bridge.knowledge.prepareSession(session,{signal:sampleSignal});else await session.bindExtensions({});session.agent.toolExecution='sequential';
  const preparedContract=targetEnvironment&&kind==='a3-kernel'?`\n\nPrepared microtask contract: the controller owns runtime setup and device health/admission; these are complete. WORKLOAD.md is an archived mathematical contract: historical paths, GOAL.md, verifier commands and full case catalogues in it are context, not files to locate or setup work to repeat. case.json defines this task. Use local native tools to inspect and edit kernel.py, and rsi_target to run the declared python check.py . device check. Do not inspect external verifier/reference trees, provenance or runtime metadata. Use this contract when reusable guidance describes a larger project workflow. ${verificationReserved?'The controller has already reserved independent verification time within the overall sample limit; use this actor allocation for the implementation and declared check.':`Finish with approximately ${Math.min(45,Math.floor(seconds/3))} seconds left for the controller's independent final verification; that verification shares this elapsed allowance.`}`:'';
  await session.prompt(`${kind==='a3-kernel'?'A3 kernel-engineering evaluation on this registered runtime. Compilation and checking count toward elapsed time. KDB retrieval is read-only; never write lessons.':'Host-only kernel-engineering microtask.'} Repair kernel.py in the current directory. Use native tools to inspect, edit and test it. This is not the full RSI workflow; unavailable RSI/KDB tools are not required. Do not inspect or modify files outside this task directory. Finish with the implementation, not just advice. ${targetEnvironment?'The model, files and bash are local. Use rsi_target for compilation and device checks on the prepared remote target; keep persistent edits local.':''}\n\nTask:\n${prompt}\n\nFrozen skill guidance:\n${guidance}${preparedContract}`);
  await session.waitForIdle();signal?.throwIfAborted();
  if(lastAssistant?.stopReason==='error')failure=lastAssistant.errorMessage??'Native provider error';
 }catch(error){failure=error.message;}
 finally{
  clearTimeout(timer);signal?.removeEventListener('abort',abort);unsubscribe();
  sampleAbort.abort(Error('Sample finished'));await Promise.allSettled([...pending]);if(aborting)await aborting;
  try{await session.exportToJsonl(join(sessionDirectory,'session.native.jsonl'));}
  finally{await session.extensionRunner?.emit({type:'session_shutdown',reason:'quit'});await session.dispose();}
 }
 const stats=session.getSessionStats();
 return {timedOut,error:failure??null,...(unresolvedExecution?{unresolvedExecution}:{}),tokens:stats.tokens.total>0?stats.tokens.total:null,usage:stats.tokens,
  cost:stats.assistantMessages>0?stats.cost:null,toolCalls:stats.toolCalls,session:manager.getSessionFile()};
}

export async function runMicroeval({directory,skills,tasks,profile,python='python3',runs=2,seconds=120,verificationSeconds=0,totalSeconds,signal,resume=false,shadow=true,objective,knowledge,binding},dependencies={}){
 const begun=performance.now();signal?.throwIfAborted();
 if(totalSeconds!==undefined&&(!Number.isFinite(totalSeconds)||totalSeconds<=0))throw Error('Positive total elapsed allocation required');
 if(!Number.isSafeInteger(runs)||runs<1||!Number.isFinite(seconds)||seconds<=0||!Array.isArray(tasks)||!tasks.length)throw Error('Positive elapsed allocation, repetitions and tasks required');
 if(!Number.isFinite(verificationSeconds)||verificationSeconds<0||verificationSeconds>=seconds)throw Error('Verification reserve must fit inside the sample elapsed allocation');
 if(objective==='harness'||tasks.some(t=>t.bucket==='harness'))throw Error('Harness faults require user intervention, not skill A/B');
 if(tasks.some(t=>t.bucket)&&(!objective||objective==='legacy'))throw Error('Bucketed tasks require an explicit matching objective');
 if(objective&&!['legacy','correctness','optimization'].includes(objective))throw Error('Unknown skill objective');
 const ids=tasks.map(t=>t.id);if(new Set(ids).size!==ids.length||ids.some(id=>!/^[a-z0-9-]+$/.test(id)))throw Error('Unique task identifiers required');
 const inputs={active:await skillInput(skills.active),candidate:await skillInput(skills.candidate)};
 const frozen=[];for(const t of tasks){
  if(!t.prompt||!t.files||!t.files['kernel.py']||!t.checker)throw Error('Runnable task files and independent Python checker required');
  const scopes=t.scopes??Object.keys(inputs.active.files);
  if(!Array.isArray(scopes)||scopes.some(scope=>!inputs.active.files[scope]||!inputs.candidate.files[scope]))throw Error('Unknown skill scope');
  if(t.bucket!==undefined&&!['correctness','optimization'].includes(t.bucket))throw Error('Invalid task bucket');
  if(objective&&objective!=='legacy'&&t.bucket&&t.bucket!==objective)throw Error('Task bucket differs from objective');
  frozen.push({...((t.bucket)?{bucket:t.bucket}:{}),...(t.protectedFiles?{protectedFiles:t.protectedFiles}:{}),...(t.pairedChecker?{pairedChecker:await readFile(t.pairedChecker,'utf8')}:{}),id:t.id,kind:t.kind??'unspecified',prompt:t.prompt,scopes,files:t.files,checker:await readFile(t.checker,'utf8')});
 }
 directory=resolve(directory);await mkdir(dirname(directory),{recursive:true});
 let retained;try{if(resume)retained=JSON.parse(await readFile(join(directory,'results.json'),'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
 if(retained?.owner&&!await ownerStopped(retained.owner)||retained?.runningSample)throw Error('Micro evaluation execution remains unresolved');
 if(!retained)await mkdir(directory);
 const manifest={format:1,shadow,...(verificationSeconds?{verificationSeconds}:{}),...(objective?{objective}:{}),...(binding?{binding}:{}),...(knowledge?{knowledge:retrievalIdentity(knowledge)}:{}),claim:frozen.some(t=>t.kind==='a3-kernel')?'isolated-a3-kernel-evaluation; trusted verification and paired timing':'host-contract-microtasks; not device performance',runs,seconds,totalSeconds:totalSeconds??null,createdAt:new Date().toISOString(),
  skills:Object.fromEntries(Object.entries(inputs).map(([arm,x])=>[arm,{sha256:x.sha256}])),
  model:{provider:profile.provider,model:profile.model,maxTokens:profile.max_tokens??null},
  tasks:frozen.map(t=>({id:t.id,kind:t.kind,scopes:t.scopes,sha256:hash(JSON.stringify(t))}))};
 if(retained){const {createdAt:_old,...old}=JSON.parse(await readFile(join(directory,'manifest.json'),'utf8'));const {createdAt:_new,...next}=manifest;if(JSON.stringify(old)!==JSON.stringify(next))throw Error('Frozen micro evaluation changed');}
 else{await save(join(directory,'manifest.json'),manifest);await mkdir(join(directory,'frozen'));await save(join(directory,'frozen','tasks.json'),frozen);}
 for(const [arm,input]of Object.entries(inputs))await writeFile(join(directory,'frozen',arm+'.md'),input.text);
 const report=retained??{...manifest,status:'running',results:[],partials:{},elapsedSeconds:0};
 report.status='running';report.owner=await processIdentity();const previousElapsed=report.elapsedSeconds??0;
 const persist=async()=>{report.elapsedSeconds=previousElapsed+(performance.now()-begun)/1000;await save(join(directory,'results.json'),report);await dependencies.checkpoint?.(report);};
 await persist();
 const sample=dependencies.sample??nativeSample;
 try{
  for(const task of frozen){
   const checker=join(directory,'frozen',task.id+'.py');await writeFile(checker,task.checker);
   for(let run=0;run<runs;run++)for(const arm of run%2?['candidate','active']:['active','candidate']){
    signal?.throwIfAborted();const key=task.id+'/'+run+'/'+arm;
    if(report.results.some(r=>r.taskId===task.id&&r.run===run&&r.arm===arm))continue;
    const partial=report.partials?.[key],spent=partial?.elapsedSeconds??0;const left=(totalSeconds??Infinity)-previousElapsed-(performance.now()-begun)/1000;
    if(left<=0){report.status='elapsed';return report;}
    if(spent>=seconds){report.results.push({...partial,correct:false,timedOut:true});delete report.partials[key];await persist();continue;}
    const allocation=Math.max(.001,Math.min(seconds-spent,left));const path=join(directory,task.id,`${run}-${arm}`),cwd=join(path,'work');await mkdir(cwd,{recursive:true});
    if(!partial)for(const [name,text]of Object.entries(task.files)){const file=inside(cwd,name);await mkdir(dirname(file),{recursive:true});await writeFile(file,text);}
    report.runningSample={key,owner:report.owner};await persist();
    const started=performance.now(),deadline=AbortSignal.timeout(Math.max(1,Math.ceil(allocation*1000))),sampleSignal=signal?AbortSignal.any([signal,deadline]):deadline;let observed,verification;
    const actorAllocation=allocation-Math.min(verificationSeconds,allocation/2),actorDeadline=verificationSeconds?AbortSignal.timeout(Math.max(1,Math.ceil(actorAllocation*1000))):null;
    const actorSignal=actorDeadline?AbortSignal.any([sampleSignal,actorDeadline]):sampleSignal;
    try{
     observed=await sample({cwd,guidance:task.scopes.map(scope=>`## ${scope}\n${inputs[arm].files[scope]}`).join('\n\n'),prompt:task.prompt,profile,kind:task.kind,knowledge,seconds:actorAllocation,verificationReserved:allocation-actorAllocation,signal:actorSignal,sampleSignal,sampleSeconds:allocation,sessionDirectory:path,resumeSession:partial?.session});
     if(observed.unresolvedExecution)throw Object.assign(Error('Micro evaluation execution remains unresolved'),{code:'RSI_EXECUTION_UNRESOLVED',handle:observed.unresolvedExecution.handle});
     if(actorDeadline?.aborted&&!sampleSignal.aborted)observed={...observed,actorTimedOut:true,actorError:observed.error??null,timedOut:false,error:null};
     const remaining=allocation-(performance.now()-started)/1000;
     if(hash(await readFile(checker,'utf8'))!==hash(task.checker))throw Error('Frozen checker changed');
     for(const [name,mode]of Object.entries(task.protectedFiles??{}))if(protectedFingerprint(task.files[name],mode)!==protectedFingerprint(await readFile(inside(cwd,name),'utf8'),mode))throw Error('Protected input changed: '+name);
     verification=remaining>0&&!sampleSignal.aborted&&!observed.timedOut&&!observed.error?await (dependencies.verify?dependencies.verify({task,python,checker,cwd,seconds:remaining,signal:sampleSignal}):(task.kind==='a3-kernel'?kernelCheck:check)(python,checker,cwd,task.kind==='a3-kernel'?remaining:Math.min(20,remaining),sampleSignal)):{correct:observed.error&&!observed.timedOut?null:false,error:'Sample elapsed allocation or native session failed'};
     if(task.kind==='a3-kernel'&&(verification.correct||verification.exitCode===125))verification=await validateKernelReceipt(verification,cwd);
     for(const [name,mode]of Object.entries(task.protectedFiles??{}))if(protectedFingerprint(task.files[name],mode)!==protectedFingerprint(await readFile(inside(cwd,name),'utf8'),mode))throw Error('Protected input changed during verification: '+name);
    }catch(error){const unknown=unknownExecution(error);if(unknown){report.runningSample.unresolvedExecution=unknown;throw Object.assign(error,{code:'RSI_EXECUTION_UNRESOLVED',handle:unknown.handle});}if(resume&&!observed)throw error;observed??={tokens:null,cost:null};verification={correct:null,error:error.message};}
    if(deadline.aborted){observed.timedOut=true;verification={correct:false,error:'Sample elapsed allocation exhausted'};}
    const result={projectId:task.id,taskId:task.id,kind:task.kind,scopes:task.scopes,run,arm,allocatedSeconds:allocation,...observed,elapsedSeconds:spent+(performance.now()-started)/1000,
     correct:verification.correct,...(verification.receipt?.bucket?{bucket:verification.receipt.bucket}:{}),...(verification.receipt?{kernelSha256:verification.receipt.kernelSha256,latencySeconds:verification.receipt.latencySeconds}:{}),verification,skillsSha256:inputs[arm].sha256};
    delete report.runningSample;
    if(resume&&signal?.aborted){report.partials??={};report.partials[key]=result;}else{report.results.push(result);if(report.partials)delete report.partials[key];}
    await persist();signal?.throwIfAborted();
   }
  }
  report.status='complete';if(objective)report.decision=evaluateSkillObjective(report.results,{objective,projectIds:ids,runs,pairedTimings:report.pairedTimings});return report;
 }catch(error){report.status='interrupted';report.error=error.message;throw error;}
 finally{delete report.owner;await persist();}
}
/** Final alternating-order replay runs trusted frozen paired checkers, never agent claims. */
export async function finalizeKernelPairs({directory,python='python3',seconds,signal},dependencies={}){
 if(!Number.isFinite(seconds)||seconds<=0)throw Error('Positive paired replay elapsed allocation required');
 const report=JSON.parse(await readFile(join(directory,'results.json'),'utf8'));
 if(report.status!=='complete'||report.runningSample||report.owner&&!await ownerStopped(report.owner))throw Error('Complete settled samples required');
 if(report.objective!=='optimization')throw Error('Paired replay requires optimization objective');
 if(report.pairedReplay?.runningPair)throw Error('Paired replay execution remains unresolved');
 const tasks=JSON.parse(await readFile(join(directory,'frozen','tasks.json'),'utf8'));
 for(const task of tasks)if(report.tasks.find(t=>t.id===task.id)?.sha256!==hash(JSON.stringify(task)))throw Error('Frozen paired task changed');
 report.pairedReplay??={startedAt:Date.now(),seconds};
 if(report.pairedReplay.seconds!==seconds)throw Error('Paired replay allocation changed');
 report.pairedTimings??=[];await save(join(directory,'results.json'),report);
 for(const task of tasks)for(let run=0;run<report.runs;run++){
  signal?.throwIfAborted();
  const left=seconds-(Date.now()-report.pairedReplay.startedAt)/1000;
  const pair=Object.fromEntries(['active','candidate'].map(arm=>[arm,report.results.find(r=>r.taskId===task.id&&r.run===run&&r.arm===arm)]));
  if(!pair.active?.correct||!pair.candidate?.correct)continue;
  if(!task.pairedChecker)throw Error('Frozen paired checker required');
  const paths=Object.fromEntries(['active','candidate'].map(arm=>[arm,join(directory,task.id,`${run}-${arm}`,'work')]));
  for(const arm of ['active','candidate'])if(hash(await readFile(join(paths[arm],'kernel.py')))!==pair[arm].kernelSha256)throw Error('Sample kernel changed before paired replay');
  const checker=join(directory,'frozen',task.id+'-paired.py');await writeFile(checker,task.pairedChecker);
  for(const arm of ['active','candidate'])for(const [name,mode]of Object.entries(task.protectedFiles??{}))if(protectedFingerprint(task.files[name],mode)!==protectedFingerprint(await readFile(inside(paths[arm],name),'utf8'),mode))throw Error('Protected paired input changed: '+name);
  if(report.pairedTimings.some(r=>r.projectId===task.id&&r.run===run)||left<=0)continue;
  report.pairedReplay.runningPair={projectId:task.id,run,owner:await processIdentity()};await save(join(directory,'results.json'),report);
  const measured=dependencies.measure?await dependencies.measure({task,run,...paths,seconds:left,signal}):await kernelCheck(python,checker,directory,left,signal,[paths.active,paths.candidate]);
  delete report.pairedReplay.runningPair;await save(join(directory,'results.json'),report);
  const receipt=measured.output!==undefined?lastReceipt(measured.output):measured;
  if(measured.exitCode!==undefined&&measured.exitCode!==0)throw Error('Paired checker failed');
  if(receipt.correct!==true||!receipt.evidence||typeof receipt.evidence!=='object')throw Error('Invalid paired verification receipt');
  for(const arm of ['active','candidate'])if(receipt[arm+'KernelSha256']!==pair[arm].kernelSha256||!Number.isFinite(receipt[arm+'LatencySeconds'])||receipt[arm+'LatencySeconds']<=0)throw Error('Invalid paired timing identity or latency');
  for(const arm of ['active','candidate']){
   if(hash(await readFile(join(paths[arm],'kernel.py')))!==pair[arm].kernelSha256)throw Error('Sample kernel changed during paired replay');
   for(const [name,mode]of Object.entries(task.protectedFiles??{}))if(protectedFingerprint(task.files[name],mode)!==protectedFingerprint(await readFile(inside(paths[arm],name),'utf8'),mode))throw Error('Protected paired input changed during replay: '+name);
  }
  report.pairedTimings.push({...receipt,projectId:task.id,run});await save(join(directory,'results.json'),report);
 }
 report.decision=evaluateSkillObjective(report.results,{objective:report.objective,projectIds:tasks.map(t=>t.id),runs:report.runs,pairedTimings:report.pairedTimings});
 await save(join(directory,'results.json'),report);return report;
}
async function kernelCheck(python,checker,cwd,seconds,signal,args=[cwd]){
 const sdk=await import('@earendil-works/pi-coding-agent'),ops=createLocalExecutionOperations(sdk);let output='';
 const quote=s=>"'"+String(s).replaceAll("'", "'\\''")+"'";
 const result=await ops.exec([python,checker,...args].map(quote).join(' '),cwd,{timeout:seconds,signal,env:{...process.env,PYTHONDONTWRITEBYTECODE:'1'},onData:d=>{output=(output+d).slice(-8192);}});
 return {correct:result.exitCode===0,exitCode:result.exitCode,output};
}
if(import.meta.main){
 const {values}=parseArgs({options:{config:{type:'string'}}});
 const abort=new AbortController();process.on('SIGINT',()=>abort.abort());process.on('SIGTERM',()=>abort.abort());
 try{const input=JSON.parse(await readFile(values.config,'utf8'));if(input.taskModule){const module=await import(new URL(input.taskModule,'file://'+resolve(values.config)).href);input.tasks=module.tasks;}
  const report=await runMicroeval({...input,signal:abort.signal});console.log(JSON.stringify({status:report.status,results:report.results}));}
 catch(error){console.error(JSON.stringify({error:error.message}));process.exitCode=1;}
}
