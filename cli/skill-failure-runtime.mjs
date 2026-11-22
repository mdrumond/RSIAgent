/** Native Pi/CPL execution for a frozen failure-first study. Callers own device admission. */
import assert from 'node:assert/strict';
import {mkdir,readFile,writeFile,rename,readdir,stat} from 'node:fs/promises';
import {join,dirname,resolve,relative,isAbsolute,basename} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {hypothesisTarget,hypothesisEvaluationSettled} from './skill-hypothesis-evaluation.mjs';
import {skillProfile} from '../controller/skill-context.mjs';
import {ownerStopped} from '../controller/lifecycle.mjs';
import {DatabaseSync} from 'node:sqlite';
import {homedir} from 'node:os';
import {failureStageElapsed,failureTiming} from './skill-failure-protocol.mjs';
export const failureProtocol=Object.freeze({actorSeconds:failureTiming.actorSeconds,verificationSeconds:failureTiming.verificationSeconds,checkerSeconds:180,cleanupSeconds:90});
const hash=x=>createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const read=async p=>{try{return JSON.parse(await readFile(p,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;return null;}};
const save=async(p,x)=>{await mkdir(dirname(p),{recursive:true});const tmp=p+'.'+randomUUID();await writeFile(tmp,JSON.stringify(x,null,2),{mode:0o600});await rename(tmp,p);};
const inside=(root,name)=>{const path=resolve(root,name),rel=relative(root,path);assert(!isAbsolute(name)&&rel&&rel!=='..'&&!rel.startsWith('../'),'Relative task file required');return path;};
export function kdbContamination(jsonl){
 for(const line of jsonl.split('\n').filter(Boolean)){const m=JSON.parse(line).message;
  if(m?.role==='toolResult'&&m.toolName?.startsWith('mcp__kdb__')&&m.isError===true&&/timed?\s*out|timeout|transport|connection|ECONN|socket|disconnected/i.test(JSON.stringify(m.content)))return {infrastructure:true,correct:null,reason:'KDB retrieval infrastructure failure',tool:m.toolName,toolCallId:m.toolCallId};
 }return null;
}
export function failureReceipt(row,task){
 if(!row?.sourceFilesMatch)return {correct:null,reason:'Source identity unavailable'};
 let value;try{value=JSON.parse(row.output.trim().split('\n').at(-1));}catch{return {correct:null,reason:'Trusted receipt missing'};}
 if(value.kernelSha256!==row.sourceFilesSha256?.['kernel.py']||typeof value.evidence!=='object'||!value.evidence)return {correct:null,reason:'Trusted source identity or evidence mismatch'};
 if(typeof value.correct==='boolean'&&value.evidence.processGuard?.allChildrenReaped!==true)return {correct:null,reason:'Trusted process closure missing'};
 if(task&&typeof value.correct==='boolean'){const cases=value.evidence.caseResults,ids=task.caseIds;if(!Array.isArray(cases)||cases.length!==ids.length||new Set(cases.map(c=>c?.id)).size!==ids.length||cases.some(c=>!ids.includes(c?.id)||typeof c.correct!=='boolean')||value.correct!==cases.every(c=>c.correct))return {correct:null,reason:'Explicit case outcomes missing or inconsistent',evidence:value.evidence};}
 return value;
}
function submissionDiagnostic(task,variant,value,declaredMechanism){
 const evidence=value.evidence??{};
 const observedMechanism={phase:evidence.phase??null,type:evidence.type??null};
 const missingFields=value.correct===false?Object.keys(observedMechanism)
  .filter(key=>typeof observedMechanism[key]!=='string'||!observedMechanism[key].trim())
  .map(key=>'evidence.'+key):[];
 if(!Array.isArray(evidence.caseResults))missingFields.push('evidence.caseResults');
 if(typeof value.correct==='boolean'&&!Array.isArray(evidence.checkedCases))missingFields.push('evidence.checkedCases');
 const checkedCases=Array.isArray(evidence.checkedCases)?evidence.checkedCases:[];
 const unverifiedCases=task.caseIds.filter(id=>!checkedCases.includes(id));
 const failedCases=(Array.isArray(evidence.caseResults)?evidence.caseResults:[])
  .filter(row=>row?.correct===false&&checkedCases.includes(row.id)).map(row=>row.id);
 const expectedCorrect=variant!=='starter'&&!variant.startsWith('negative-');
 const declaredFailureObserved=value.correct===false&&sameMechanism(value,{mechanism:declaredMechanism});
 const mechanismMatch=declaredFailureObserved&&(variant!=='starter'||failedCases.length>0);
 const passed=expectedCorrect
  ?value.correct===true&&unverifiedCases.length===0
  :mechanismMatch;
 let problem=null,detail;
 if(!passed){
  if(value.correct!==true&&value.correct!==false)problem='Checker did not return a valid correctness verdict';
  else if(missingFields.length)problem='Checker must emit the missing evidence fields';
  else if(expectedCorrect&&value.correct===false)problem='Passing control failed';
  else if(value.correct===true&&!expectedCorrect)problem='Failing control passed';
  else if(value.correct===true)problem='Passing verdict omitted checked cases';
  else if(!declaredFailureObserved)problem='Checker emitted a different mechanism from the manifest';
  else problem='Starter verdict did not identify an executed failing case';
  if(evidence.detail!==undefined){
   const encoded=JSON.stringify(evidence.detail),limit=2048;
   detail=encoded.length<=limit?evidence.detail:{preview:encoded.slice(0,limit),truncated:true};
  }
 }
 return {passed,diagnostic:{expectedCorrect,declaredMechanism,observedMechanism,missingFields,failedCases,unverifiedCases,problem,detail,
  mechanismMatch,checkedCases,phase:observedMechanism.phase,type:observedMechanism.type,observedCorrect:value.correct??null,
  correct:value.correct===false?false:value.correct===true&&unverifiedCases.length===0?true:null}};
}
export async function failureChecker(task,runtime=null,guard={}){
 // On A5 the NPU runtime can take more than five seconds to exit after a
 // valid receipt. Keep the same elapsed verifier limit and require a clean exit.
 const payload={checker:await readFile(task.checker,'utf8'),helpers:task.checkerFiles??{},runtime,seconds:failureProtocol.checkerSeconds,...(runtime?.target==='a5'?{cleanupSeconds:15}:{}),...guard};
 assert(!Object.hasOwn(payload.helpers,'oracle.py')&&!Object.hasOwn(payload.helpers,'runtime_identity.py'),'Reserved oracle helper filename');
 if(runtime)payload.runtimeHelper=await readFile(new URL('./skill-suites/runtime-identity.py',import.meta.url),'utf8');
 return (await readFile(new URL('./skill-failure-check.py',import.meta.url),'utf8')).replace('__PAYLOAD__','json.loads('+JSON.stringify(JSON.stringify(payload))+')');
}
// Only this retained runtime revision used the verified local sandbox/Pi export
// and did not mark a completed pre-dispatch trial as closed.
const legacyPreDispatchSource='863159a786452c9365e8b35de1962996b0b92e29';
const legacyBash=/^(?:ls -la(?:; echo (?:"---"|---); cat kernel\.py(?:; echo (?:"---"|---); cat public_verify\.py 2>\/dev\/null \| head -100)?)?)$/;
const jsonLines=async path=>(await readFile(path,'utf8')).trimEnd().split('\n').map(line=>JSON.parse(line));
const absent=async path=>{try{await stat(path);return false;}catch(e){if(e.code!=='ENOENT')throw e;return true;}};

async function legacySessionEnded(trial,cwd) {
 const sandbox=await read(join(trial,'sandbox-preflight.json'));
 if(sandbox?.kind!=='local-filesystem-sandbox'||sandbox.available!==true||sandbox.exitCode!==0)return false;
 const [header,...records]=await jsonLines(join(trial,'session.native.jsonl'));
 if(header?.type!=='session'||header.cwd!==cwd||!(/^[a-f0-9-]{36}$/).test(header.id))return false;
 const originals=(await readdir(trial)).filter(name=>name.endsWith('_'+header.id+'.jsonl'));
 if(originals.length!==1)return false;
 const [original,...events]=await jsonLines(join(trial,originals[0]));
 if(original?.type!=='session'||original.id!==header.id||original.cwd!==cwd||hash(events)!==hash(records))return false;
 const messages=records.filter(record=>record.type==='message').map(record=>record.message);
 if(!messages.some(m=>m.role==='user'&&JSON.stringify(m.content).includes('The model, files and bash are local. Use rsi_target')))return false;
 const last=messages.at(-1);
 if(last?.role!=='assistant'||!['error','aborted'].includes(last.stopReason)||
    !messages.some(m=>m.role==='assistant'&&m.stopReason==='error'&&m.errorMessage?.includes('Request timed out')))return false;
 const policy=join(trial,'tool-policy.jsonl'),blocked=await absent(policy)?[]:await jsonLines(policy);
 const pending=new Map();
 for(const message of messages){
  if(message.role==='assistant')for(const call of message.content.filter(item=>item.type==='toolCall')){
   if(!call.id||pending.has(call.id)||!['read','edit','bash'].includes(call.name))return false;
   const shellBlocked=call.name==='bash'&&!legacyBash.test(call.arguments?.command);
   if(shellBlocked&&!blocked.some(row=>row.tool==='bash'&&row.blocked===true&&row.toolCallId===call.id))return false;
   pending.set(call.id,{name:call.name,shellBlocked});
  }
  if(message.role==='toolResult'){
   const call=pending.get(message.toolCallId);
   if(call?.name!==message.toolName||call.shellBlocked&&message.isError!==true)return false;
   pending.delete(message.toolCallId);
  }
 }
 return pending.size===0;
}

async function legacyLeaseAbsent(instancePath) {
 const leases=join(homedir(),'.local/state/rsi/device-leases');
 if(await absent(leases))return true;
 for(const name of (await readdir(leases)).filter(name=>/^[a-f0-9]{64}$/.test(name))){
  if((await read(join(leases,name,'lease.json')))?.instancePath===instancePath)return false;
 }
 return true;
}

/** Read-only proof for archived pre-dispatch A5 actors with completed Pi exports. */
export async function legacyFailureRuntimePreDispatchClosures({directory,config,signal}) {
 const closed=new Set(),run=dirname(directory),id=basename(run),pilotDirectory=join(run,'pilot');
 if(config.target!=='a5'||!config.root||directory!==join(resolve(config.root),'skills','priority','failure-first',id,'runtime'))return closed;
 try{
  const manifest=await read(join(pilotDirectory,'manifest.json'));
  if(manifest?.mode!==undefined||manifest?.identities?.source!==legacyPreDispatchSource||
     hash(manifest)!==hash(await read(join(run,'experiment.json'))))return closed;
  const pilot=await read(join(pilotDirectory,'pilot.json'));
  if(pilot?.status!=='inconclusive'||pilot.reason!=='Actor custody remains unresolved'||pilot.unresolvedOperation||
     !Object.values(pilot.operations).every(op=>op.status==='complete'))return closed;
  let owner;
  const db=new DatabaseSync(join(config.root,'state.sqlite'),{readOnly:true});
  try{
   const row=db.prepare('SELECT status,owner,checkpoint FROM skill_backlog_runs WHERE id=?').get(id);
   if(row?.status!=='running')return closed;
   const checkpoint=JSON.parse(row.checkpoint);
   if(checkpoint?.workflow!=='failure-first-v1'||checkpoint.phase!=='evaluate'||checkpoint.directory!==run||checkpoint.runtimeDirectory!==directory)return closed;
   owner=JSON.parse(row.owner);
  }finally{db.close();}
  if(!await ownerStopped(owner))return closed;
  const rows=await read(join(directory,'runtime-contexts.json'));
  for(const [key,operation] of Object.entries(pilot.operations)){
   signal?.throwIfAborted();
   const op=join(pilotDirectory,'operations',key),result=operation.result;
   if(!/^screen--[\w-]+--development--baseline--\d+$/.test(key)||operation.directory!==op||operation.kind!=='trial'||operation.attempts?.length||
      result?.infrastructure!==true||result.correct!==null||result.handle||
      !['Actor custody remains unresolved','Operation elapsed allocation exhausted before result admission'].includes(result.reason))continue;
   try{
    const adapter=await read(join(op,'adapter.json'));
    if(!adapter||dirname(adapter.publicRoot)!==join(run,'public')||adapter.privateRoot!==join(op,'verification')||
       !await absent(adapter.privateRoot)||!await absent(join(op,'operation-result.json')))continue;
    const trial=join(adapter.publicRoot,'micro','development','0-trial'),cwd=join(trial,'work'),target=join(trial,'target');
    const row=rows.find(row=>row.root===adapter.publicRoot),env=row?.binding?.environment;
    if(adapter.cwd!==cwd||row?.binding?.target!=='a5'||env?.mode!=='local'||env.target!=='bz-a5-2'||
       env.deviceAffinity?.length!==1||!Number.isInteger(env.deviceAffinity[0])||
       !await absent(join(adapter.publicRoot,'micro','results.json')))continue;
    const children=await readdir(target,{withFileTypes:true,recursive:true});
    if(children.filter(entry=>entry.isFile()&&entry.name==='environment-events.jsonl'&&entry.parentPath===target).length!==1||
       children.some(entry=>!(entry.isFile()&&entry.name==='environment-events.jsonl'&&entry.parentPath===target)&&
        !(entry.isDirectory()&&((entry.parentPath===target&&['.ascend-resources','.executions'].includes(entry.name))||
          (entry.parentPath===join(target,'.ascend-resources')&&/^[a-f0-9]{64}$/.test(entry.name))))))continue;
    const journal=await jsonLines(join(target,'environment-events.jsonl')),prepared=journal[1]?.data;
    const attempt='micro-'+hash(target).slice(0,24);
    if(journal.length!==2||journal[0]?.type!=='resource-policy'||journal[1]?.type!=='environment-prepared'||
       prepared?.project!==attempt||prepared.attempt!==attempt||!await legacyLeaseAbsent(target)||
       !await legacySessionEnded(trial,cwd))continue;
    closed.add(adapter.publicRoot);
   }catch(e){if(signal?.aborted)throw e;}
  }
 }catch(e){if(signal?.aborted)throw e;closed.clear();}
 return closed;
}

function closedWithoutDispatch({actorClosure,...row}) {
 return actorClosure?.kind==='native-no-target'&&actorClosure.contextSha256===hash(row);
}

/** Does not launch KDB, probes or model work; suitable for crash reconciliation. */
export async function settleFailureRuntime({directory,config,signal},dependencies={}){
 const rows=await read(join(directory,'runtime-contexts.json'))??[],result=[];
 let recovered;
 for(const row of rows){
  let closed=closedWithoutDispatch(row)||await (dependencies.settled??hypothesisEvaluationSettled)({directory:row.root,config,binding:row.binding,resource:row.resource,signal});
  if(!closed){
   recovered??=await legacyFailureRuntimePreDispatchClosures({directory,config,signal});
   closed=recovered.has(row.root);
  }
  result.push({root:row.root,closed});
 }
 return {closed:result.every(r=>r.closed),contexts:result};
}
export const failureRuntimeSettled=settleFailureRuntime;
export async function createFailureRuntime({config,directory,publicDirectory,privateDirectory,hiddenDirectories=[],slots,knowledge,signal,stopSignal},dependencies={}){
 assert(slots?.length&&slots.length<=4&&new Set(slots.map(s=>s.resource.id)).size===slots.length,'Distinct admitted slots required');
 if(config.target!=='cpu')assert(slots.every(s=>Number.isInteger(s.physical))&&new Set(slots.map(s=>s.physical)).size===slots.length,'Distinct admitted physical devices required');
 directory=resolve(directory);privateDirectory=resolve(privateDirectory??dirname(directory));publicDirectory=resolve(publicDirectory??join(dirname(directory),'public-'+hash(directory).slice(0,12)));
 await mkdir(directory,{recursive:true});await mkdir(publicDirectory,{recursive:true});
 const free=[...slots],registry=join(directory,'runtime-contexts.json'),contexts=new Map((await read(registry)??[]).map(r=>[r.root,r])),mechanisms=await read(join(directory,'mechanisms.json'))??{};let writes=Promise.resolve(),runtime=await read(join(directory,'runtime-identity.json'));
 const remember=async row=>{contexts.set(row.root,row);writes=writes.then(()=>save(registry,[...contexts.values()]));await writes;};
 const closed=async root=>{const row=contexts.get(root);return !row||closedWithoutDispatch(row)||await (dependencies.settled??hypothesisEvaluationSettled)({directory:root,config,binding:row.binding,resource:row.resource,signal:stopSignal});};
 const target=async(root,slot,kind,fingerprint,work)=>{
  const environment={...config.environment.options,...slot.binding?.environment,deviceAffinity:[slot.physical],remoteDirectory:'rsi-failure-'+hash([fingerprint,kind]).slice(0,20)};delete environment.backgroundDevice;
  const binding={target:config.target,python:environment.python??'python3',...slot.binding,environment};
  const row={root,binding,resource:slot.resource},old=contexts.get(root);if(old)assert.deepEqual(row,old,'Retained runtime binding changed');else await remember(row);
  return (dependencies.target??hypothesisTarget)({root,binding,resource:slot.resource,signal:work,stopSignal,localExecution:config.local_execution},dependencies);
 };
 const freeze=async(task,familyId)=>{
  const bytes={files:task.files,checker:await readFile(task.checker,'utf8'),publicChecker:await readFile(task.publicChecker,'utf8'),helpers:task.checkerFiles??{},cases:task.caseIds,controls:task.controls};
  const file=join(directory,'tasks',hash([familyId,task.id])+'.json'),frozen={sha256:hash(bytes)},old=await read(file);if(old)assert.deepEqual(old,frozen,'Frozen task changed');else await save(file,frozen);return {...frozen,...bytes};
 };
 const verify=async(root,slot,task,entries,seconds,work,familyId,stage='verification')=>{
  assert(Number.isFinite(seconds)&&seconds>0,'Positive verification allowance required');
  const started=performance.now(),timeout=AbortSignal.timeout(Math.max(1,Math.ceil(seconds*1000))),bounded=work?AbortSignal.any([work,timeout]):timeout;
  try{
   bounded.throwIfAborted();
   const frozen=await freeze(task,familyId),host=await target(root,slot,'private',hash([frozen.sha256,runtime]),bounded),checker=await failureChecker(task,runtime);
   bounded.throwIfAborted();
   const left=seconds-(performance.now()-started)/1000;
   if(left<=0)throw failureStageElapsed(stage);
   const rows=await host.qualifyBatch({directory:root,entries:entries.map((e,i)=>({...e,cwd:'candidate-'+i,checker,args:['.'],timeoutSeconds:195})),seconds:left,signal:bounded,sampleStopSignal:stopSignal});
   assert(await closed(root),'Verification custody remains unresolved');bounded.throwIfAborted();return rows;
  }catch(error){
   if(!work?.aborted&&(timeout.aborted||performance.now()-started>=seconds*1000)&&await closed(root))throw failureStageElapsed(stage);
   throw error;
  }
 };
 const identify=async({seconds=120,signal:work=signal}={})=>{
  work?.throwIfAborted();
  if(config.target==='cpu')return {target:'cpu'};
  const root=join(directory,'identity'),host=await target(root,slots[0],'identity',hash(config.environment.options),work),actual=await host.runtimeIdentity({directory:root,seconds,signal:work,sampleStopSignal:stopSignal});
  assert(await closed(root),'Runtime probe custody remains unresolved');work?.throwIfAborted();if(runtime)assert.deepEqual(actual,runtime,'Qualified runtime changed');else{runtime=actual;await save(join(directory,'runtime-identity.json'),runtime);}return runtime;
 };
 const checkReferences=async input=>{
  input.signal?.throwIfAborted();signal?.throwIfAborted();assert(Number.isFinite(input.seconds)&&input.seconds>0,'Positive reference allowance required');assert(config.target==='cpu'||runtime,'Identify target before references');
  const {development,transfer}=input.family,tasks=[development,...transfer];
  const starterIndex=development.controls.negatives.findIndex(n=>hash(n.files['kernel.py'])===hash(development.files['kernel.py']));
  assert(starterIndex>=0,'Development negative must replay the isolated starter');
  const checks=[[development,'starter',development.controls.negatives[starterIndex].files],
   ...tasks.map(task=>[task,'reference',task.controls.reference.files])];
  // New A5 submissions carry focused controls. Check them while the author can still revise.
  if(config.target==='a5'&&tasks.every(task=>task.controls.calibration?.files)){
   for(const task of tasks){
    for(const [index,control] of task.controls.negatives.entries())
     if(task!==development||index!==starterIndex)checks.push([task,'negative-'+index,control.files]);
    checks.push([task,'calibration',task.controls.calibration.files]);
   }
  }
  const slot=free.pop();assert(slot,'No reference slot');const roots=[],started=performance.now(),diagnostics=[];
  try{for(const [task,variant,files] of checks){
   input.signal?.throwIfAborted();signal?.throwIfAborted();
   // Key the execution by its inputs. Compare its verdict to the current declaration below.
   const fingerprint=hash({submission:input.cacheScope??input.directory,family:input.family.id,task:task.id,caseIds:task.caseIds,checkerFiles:task.checkerFiles,
    checker:await readFile(task.checker,'utf8'),publicChecker:await readFile(task.publicChecker,'utf8'),variant,files,runtime,slot});
   const cache=join(directory,'author-checks',fingerprint+'.json'),retained=await read(cache);
   const root=retained?.root??join(input.directory,'references',fingerprint);
   roots.push(root);await save(join(input.directory,'adapter.json'),{qualificationRoots:roots});
   const began=performance.now(),left=input.seconds-(began-started)/1000;
   if(left<=0)throw failureStageElapsed('isolate');
   const stage=variant==='starter'?'baseline-replay':'verification';
   const allocatedSeconds=Math.min(variant==='starter'?failureTiming.baselineReplaySeconds:failureTiming.verificationSeconds,left);
   const timing=()=>({stage,allocatedSeconds,elapsedSeconds:(performance.now()-began)/1000});let row,measured;
   if(retained){
    assert(contexts.has(root)&&await closed(root),'Retained checker custody remains unresolved');
    ({row,timing:measured}=retained);
   }else{
    let rows;
    try{rows=await verify(root,slot,{...task,files,controls:undefined},[{key:variant,files}],allocatedSeconds,input.signal,'reference-'+fingerprint,stage);}
    catch(error){await save(join(root,'reference.json'),{task:task.id,timing:timing(),error:{message:error.message,code:error.code??null}});throw error;}
    measured=timing();await save(join(root,'reference.json'),{task:task.id,timing:measured,rows});
    assert.equal(rows.length,1,'Trusted reference row missing');[row]=rows;
   }
   const value=failureReceipt(row,task),guard=value.evidence?.processGuard;
   const settledVerdict=row.sourceFilesMatch===true&&guard?.allChildrenReaped===true&&guard.reason==null&&!input.signal?.aborted&&!signal?.aborted;
   if(!retained&&settledVerdict)await save(cache,{root,row,timing:measured});
   const declaredMechanism=variant==='starter'?input.family.mechanism:task.mechanism??input.family.mechanism;
   const {passed,diagnostic:feedback}=submissionDiagnostic(task,variant,value,declaredMechanism);
   const diagnostic={task:task.id,variant,...feedback,timing:measured,reused:Boolean(retained),sourceSha256:value.kernelSha256,
    reason:value.reason??value.evidence?.reason??value.evidence?.message??null,error:[value.error,row.error].filter(Boolean).join('\n').slice(-2048),path:join(root,'reference.json')};
   diagnostics.push(diagnostic);await save(join(input.directory,'reference-results.json'),diagnostics);
   if(!passed){
    const error=Error((variant==='starter'?'Starter must reproduce the declared failure before references run. ':'')+task.id+' '+variant+' rejected: '+JSON.stringify(diagnostic));
    if(settledVerdict){
     error.code='FAILURE_REFERENCE_REJECTED';error.diagnostics=diagnostics;
    }
    throw error;
   }
  }return {checked:true,references:diagnostics.filter(row=>row.variant==='reference'),diagnostics};
  }finally{if((await Promise.all(roots.map(closed))).every(Boolean))free.push(slot);}
 };
 const qualify=async input=>{
  input.signal?.throwIfAborted();signal?.throwIfAborted();
  assert(config.target==='cpu'||runtime,'Identify target before qualification');const slot=free.pop();assert(slot,'No qualification slot');const roots=[];let referencePassed=true,negativeFailed=true,calibrationPassed=true;
  if(mechanisms[input.family.id])assert.deepEqual(mechanisms[input.family.id],input.family.mechanism,'Failure mechanism changed');else{mechanisms[input.family.id]=input.family.mechanism;await save(join(directory,'mechanisms.json'),mechanisms);}
  const start=performance.now(),controls=[];try{for(const task of input.task?[input.task]:[input.family.development,...input.family.transfer]){
   const root=join(input.directory,'controls',task.id);roots.push(root);await save(join(input.directory,'adapter.json'),{qualificationRoots:roots});const entries=[{key:'reference',files:task.controls.reference.files},...task.controls.negatives.map((n,i)=>({key:'negative-'+i,files:n.files}))];
   if(input.task){assert(task.controls.calibration?.files,'Focused task calibration is required');entries.push({key:'calibration',files:task.controls.calibration.files});}
   const left=input.remainingSeconds-(performance.now()-start)/1000;assert(left>0,'Qualification elapsed allocation exhausted');
   const rows=await verify(root,slot,task,entries,Math.min(left,failureTiming.verificationSeconds),input.signal,input.family.id);assert.equal(rows.length,entries.length,'Trusted control rows missing');
   for(const row of rows){
    const value=failureReceipt(row,task);if(typeof value.correct!=='boolean')throw Error(value.evidence?.reason??value.reason??'Control infrastructure failure');
    if(value.correct)assert(task.caseIds.every(id=>value.evidence.checkedCases?.includes(id)),'Reference receipt omits frozen cases');
    if(row.key==='reference')referencePassed&&=value.correct;
    else if(row.key==='calibration')calibrationPassed&&=value.correct;
    else negativeFailed&&=!value.correct&&(!input.task&&task.id!==input.family.development.id||sameMechanism(value,{mechanism:task.mechanism??input.family.mechanism}));
   }await save(join(root,'qualification.json'),{referencePassed,negativeFailed,rows});controls.push({task:task.id,path:join(root,'qualification.json')});
  }const result={qualified:referencePassed&&negativeFailed&&calibrationPassed,referencePassed,negativeFailed,...(input.task?{calibrationPassed}:{}),controls};await save(join(input.directory,'operation-result.json'),result);return result;
  }finally{if((await Promise.all(roots.map(closed))).every(Boolean))free.push(slot);}
 };
 const trial=async input=>{
  const {actorSeconds=failureTiming.actorSeconds,verificationSeconds=failureTiming.verificationSeconds}=input;
  input.signal?.throwIfAborted();signal?.throwIfAborted();assert(Number.isFinite(actorSeconds)&&actorSeconds>0&&actorSeconds<=failureTiming.actorSeconds&&Number.isFinite(verificationSeconds)&&verificationSeconds>0&&verificationSeconds<=failureTiming.verificationSeconds,'Trial grants exceed the frozen protocol');
  assert(config.target==='cpu'||runtime,'Identify target before trials');const slot=free.pop();assert(slot,'No admitted trial slot');
  const publicRoot=join(publicDirectory,randomUUID()),privateRoot=join(input.directory,'verification'),cwd=join(publicRoot,'micro',input.task.id,'0-trial','work'),sessionDirectory=dirname(cwd);let host;
  try{
   const frozen=await freeze(input.task,input.familyId);await mkdir(cwd,{recursive:true});for(const [name,text]of Object.entries({...input.task.files,'public_verify.py':frozen.publicChecker})){const file=inside(cwd,name);await mkdir(dirname(file),{recursive:true});await writeFile(file,text);}
   await mkdir(join(publicRoot,'micro','frozen'),{recursive:true});await writeFile(join(publicRoot,'micro','frozen',input.task.id+'.py'),frozen.publicChecker);
   await save(join(input.directory,'adapter.json'),{publicRoot,privateRoot,cwd});host=await target(publicRoot,slot,'public',frozen.sha256,input.signal);
   const profile=skillProfile(config.actor),hiddenPaths=[privateDirectory,publicDirectory,...hiddenDirectories,...(profile.credentials_file?[dirname(profile.credentials_file)]:[])];
   const actor=await host.sample({cwd,sessionDirectory,profile,guidance:input.guidance,prompt:input.task.prompt+'\nUse native local file/Bash tools for edits and rsi_target for device commands. Public check: python3 public_verify.py .',actorBash:'local',seconds:actorSeconds,signal:input.signal,sampleSignal:input.signal,sampleStopSignal:stopSignal,knowledge,localFilesystemSandbox:{hiddenPaths}});
   assert(await host.sampleSettled({cwd}),'Actor custody remains unresolved');
   if(actor.targetDispatched===false&&!actor.unresolvedExecution){
    const row=contexts.get(publicRoot);
    await remember({...row,actorClosure:{kind:'native-no-target',contextSha256:hash(row)}});
   }
   assert(await closed(publicRoot),'Actor custody remains unresolved');
   if(actor.session){const contamination=kdbContamination(await readFile(actor.session,'utf8'));if(contamination)return contamination;}
   if(actor.providerTimeout||actor.error&&!actor.timedOut)
    return {infrastructure:true,correct:null,reason:'Native session failed: '+(actor.providerTimeout??actor.error),
     ...(actor.providerTimeout?{errorCode:'FAILURE_TRIAL_PROVIDER_TIMEOUT'}:actor.errorCode?{errorCode:actor.errorCode}:{}),...(actor.bucket?{bucket:actor.bucket}:{})};
   const files={};async function gather(path,prefix=''){for(const entry of await readdir(path,{withFileTypes:true})){if(entry.name.startsWith('.')||['__pycache__','node_modules'].includes(entry.name))continue;const name=prefix+entry.name;if(entry.isDirectory())await gather(join(path,entry.name),name+'/');else if(entry.isFile()&&entry.name!=='public_verify.py'&&(Object.hasOwn(input.task.files,name)||/\.(py|c|cc|cpp|cxx|cce|h|hpp|inc|cmake|sh|json|md)$/.test(name)||['Makefile','CMakeLists.txt'].includes(name))){assert((await stat(join(path,entry.name))).size<=16*1048576,'Source artifact too large');files[name]=await readFile(join(path,entry.name),'utf8');}}}await gather(cwd);assert(files['kernel.py'],'Candidate kernel missing');
   const protectedNames=[...new Set([...Object.keys(input.task.protectedFiles??{}),...['CONTRACT.md','WORKLOAD.md','case.json','cases.json'].filter(n=>Object.hasOwn(input.task.files,n))])];const unchanged=protectedNames.every(name=>files[name]===input.task.files[name])&&(await readFile(join(cwd,'public_verify.py'),'utf8'))===frozen.publicChecker;
   const rows=await verify(privateRoot,slot,input.task,[{key:'verification',files}],verificationSeconds,input.signal,input.familyId);assert.equal(rows.length,1,'Trusted verification row missing');const row=rows[0],value=failureReceipt(row,input.task);
   await save(join(input.directory,'verification.json'),row);if(typeof value.correct!=='boolean')return {infrastructure:true,correct:null,reason:value.evidence?.reason??value.reason??'Verification unavailable'};
   const correct=value.correct&&unchanged,result={correct,mechanismMatch:!correct&&unchanged&&sameMechanism(value,{mechanism:input.task.mechanism??mechanisms[input.familyId]}),caseResults:value.evidence.caseResults.map(c=>({...c,correct:unchanged&&c.correct})),sourceSha256:hash(row.sourceFilesSha256),artifacts:[{path:join(input.directory,'verification.json')},...(actor.session?[{path:actor.session}]:[])],handles:[row.handle],actorTimedOut:actor.timedOut,failure:correct?null:unchanged?value.evidence:{phase:'protected-input',reason:'Protected actor input changed'}};
   if(value.correct)assert(input.task.caseIds.every(id=>value.evidence.checkedCases?.includes(id)),'Successful receipt omits frozen cases');
   if(input.mode==='focused-correctness'&&!unchanged)Object.assign(result,{correct:null,reason:'Protected actor input changed'});
   await save(join(input.directory,'source-manifest.json'),{files:row.sourceFilesSha256,cwd});await save(join(input.directory,'operation-result.json'),result);return result;
  }catch(error){return {infrastructure:true,correct:null,reason:error.message,handle:error.handle??null,...(error.failureStage?{stage:error.failureStage}:{})};}
  finally{if(await closed(publicRoot)&&await closed(privateRoot))free.push(slot);}
 };
 const settle=async op=>{const adapter=await read(join(op.directory,'adapter.json'));if(!adapter)return {closed:false};if(!(await Promise.all((adapter.qualificationRoots??[adapter.publicRoot,adapter.privateRoot]).map(closed))).every(Boolean))return {closed:false};const result=await read(join(op.directory,'operation-result.json')),manifest=await read(join(op.directory,'source-manifest.json'));if(result&&manifest)for(const [name,sha]of Object.entries(manifest.files))assert.equal(hash(await readFile(inside(manifest.cwd,name),'utf8')),sha,'Recovered source changed');return {closed:true,...(result?{result}:{})};};
 return {identify,checkReferences,qualify,trial,settle,close:async()=>{await writes;return (await settleFailureRuntime({directory,config,signal:stopSignal},dependencies)).closed;}};
}
function sameMechanism(value,family){return value.evidence?.phase===family.mechanism?.phase&&value.evidence?.type===family.mechanism?.type&&!!family.mechanism;}
