/** Native Pi/CPL execution for a frozen failure-first study. Callers own device admission. */
import assert from 'node:assert/strict';
import {mkdir,readFile,writeFile,rename,readdir,stat} from 'node:fs/promises';
import {join,dirname,resolve,relative,isAbsolute} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {hypothesisTarget,hypothesisEvaluationSettled} from './skill-hypothesis-evaluation.mjs';
import {skillProfile} from '../controller/skill-context.mjs';
export const failureProtocol=Object.freeze({actorSeconds:180,verificationSeconds:300,checkerSeconds:180,cleanupSeconds:90});
const hash=x=>createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const read=async p=>{try{return JSON.parse(await readFile(p,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;return null;}};
const save=async(p,x)=>{await mkdir(dirname(p),{recursive:true});const tmp=p+'.'+randomUUID();await writeFile(tmp,JSON.stringify(x,null,2),{mode:0o600});await rename(tmp,p);};
const inside=(root,name)=>{const path=resolve(root,name),rel=relative(root,path);assert(!isAbsolute(name)&&rel&&rel!=='..'&&!rel.startsWith('../'),'Relative task file required');return path;};
export function kdbContamination(jsonl){
 for(const line of jsonl.split('\n').filter(Boolean)){const m=JSON.parse(line).message;
  if(m?.role==='toolResult'&&m.toolName?.startsWith('mcp__kdb__')&&m.isError===true&&/timed?\s*out|timeout|transport|connection|ECONN|socket|disconnected/i.test(JSON.stringify(m.content)))return {infrastructure:true,correct:null,reason:'KDB retrieval infrastructure failure',tool:m.toolName,toolCallId:m.toolCallId};
 }return null;
}
export function failureReceipt(row){
 if(!row?.sourceFilesMatch)return {correct:null,reason:'Source identity unavailable'};
 let value;try{value=JSON.parse(row.output.trim().split('\n').at(-1));}catch{return {correct:null,reason:'Trusted receipt missing'};}
 if(value.kernelSha256!==row.sourceFilesSha256?.['kernel.py']||typeof value.evidence!=='object'||!value.evidence)return {correct:null,reason:'Trusted source identity or evidence mismatch'};
 if(typeof value.correct==='boolean'&&value.evidence.processGuard?.allChildrenReaped!==true)return {correct:null,reason:'Trusted process closure missing'};
 return value;
}
export async function failureChecker(task,runtime=null,guard={}){
 const payload={checker:await readFile(task.checker,'utf8'),helpers:task.checkerFiles??{},runtime,seconds:failureProtocol.checkerSeconds,...guard};
 assert(!Object.hasOwn(payload.helpers,'oracle.py')&&!Object.hasOwn(payload.helpers,'runtime_identity.py'),'Reserved oracle helper filename');
 if(runtime)payload.runtimeHelper=await readFile(new URL('./skill-suites/runtime-identity.py',import.meta.url),'utf8');
 return (await readFile(new URL('./skill-failure-check.py',import.meta.url),'utf8')).replace('__PAYLOAD__','json.loads('+JSON.stringify(JSON.stringify(payload))+')');
}
/** Does not launch KDB, probes or model work; suitable for crash reconciliation. */
export async function settleFailureRuntime({directory,config,signal},dependencies={}){
 const rows=await read(join(directory,'runtime-contexts.json'))??[],result=[];
 for(const row of rows)result.push({root:row.root,closed:await (dependencies.settled??hypothesisEvaluationSettled)({directory:row.root,config,binding:row.binding,resource:row.resource,signal})});
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
 const closed=async root=>{const row=contexts.get(root);return !row||await (dependencies.settled??hypothesisEvaluationSettled)({directory:root,config,binding:row.binding,resource:row.resource,signal:stopSignal});};
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
 const verify=async(root,slot,task,entries,seconds,work,familyId)=>{
  assert(Number.isFinite(seconds)&&seconds>0,'Positive verification allowance required');const timeout=AbortSignal.timeout(Math.max(1,Math.ceil(seconds*1000))),bounded=work?AbortSignal.any([work,timeout]):timeout;bounded.throwIfAborted();
  const frozen=await freeze(task,familyId),host=await target(root,slot,'private',hash([frozen.sha256,runtime]),bounded),checker=await failureChecker(task,runtime);
  const rows=await host.qualifyBatch({directory:root,entries:entries.map((e,i)=>({...e,cwd:'candidate-'+i,checker,args:['.'],timeoutSeconds:195})),seconds,signal:bounded,sampleStopSignal:stopSignal});
  assert(await closed(root),'Verification custody remains unresolved');bounded.throwIfAborted();return rows;
 };
 const identify=async({seconds=120,signal:work=signal}={})=>{
  work?.throwIfAborted();
  if(config.target==='cpu')return {target:'cpu'};
  const root=join(directory,'identity'),host=await target(root,slots[0],'identity',hash(config.environment.options),work),actual=await host.runtimeIdentity({directory:root,seconds,signal:work,sampleStopSignal:stopSignal});
  assert(await closed(root),'Runtime probe custody remains unresolved');work?.throwIfAborted();if(runtime)assert.deepEqual(actual,runtime,'Qualified runtime changed');else{runtime=actual;await save(join(directory,'runtime-identity.json'),runtime);}return runtime;
 };
 const qualify=async input=>{
  input.signal?.throwIfAborted();signal?.throwIfAborted();
  assert(config.target==='cpu'||runtime,'Identify target before qualification');const slot=free.pop();assert(slot,'No qualification slot');const roots=[];let referencePassed=true,negativeFailed=true;
  if(mechanisms[input.family.id])assert.deepEqual(mechanisms[input.family.id],input.family.mechanism,'Failure mechanism changed');else{mechanisms[input.family.id]=input.family.mechanism;await save(join(directory,'mechanisms.json'),mechanisms);}
  const start=performance.now(),controls=[];try{for(const task of [input.family.development,...input.family.transfer]){
   const root=join(input.directory,'controls',task.id);roots.push(root);await save(join(input.directory,'adapter.json'),{qualificationRoots:roots});const entries=[{key:'reference',files:task.controls.reference.files},...task.controls.negatives.map((n,i)=>({key:'negative-'+i,files:n.files}))];
   const left=input.remainingSeconds-(performance.now()-start)/1000;assert(left>0,'Qualification elapsed allocation exhausted');
   const rows=await verify(root,slot,task,entries,Math.min(left,entries.length*300),input.signal,input.family.id);assert.equal(rows.length,entries.length,'Trusted control rows missing');
   for(const row of rows){
    const value=failureReceipt(row);if(typeof value.correct!=='boolean')throw Error(value.evidence?.reason??value.reason??'Control infrastructure failure');
    if(value.correct)assert(task.caseIds.every(id=>value.evidence.checkedCases?.includes(id)),'Reference receipt omits frozen cases');
    if(row.key==='reference')referencePassed&&=value.correct;else negativeFailed&&=!value.correct&&(task.id!==input.family.development.id||sameMechanism(value,input.family));
   }await save(join(root,'qualification.json'),{referencePassed,negativeFailed,rows});controls.push({task:task.id,path:join(root,'qualification.json')});
  }const result={qualified:referencePassed&&negativeFailed,referencePassed,negativeFailed,controls};await save(join(input.directory,'operation-result.json'),result);return result;
  }finally{if((await Promise.all(roots.map(closed))).every(Boolean))free.push(slot);}
 };
 const trial=async input=>{
  input.signal?.throwIfAborted();signal?.throwIfAborted();assert(Number.isFinite(input.actorSeconds??180)&&(input.actorSeconds??180)>0&&(input.actorSeconds??180)<=180&&Number.isFinite(input.verificationSeconds??300)&&(input.verificationSeconds??300)>0&&(input.verificationSeconds??300)<=300,'Trial grants exceed the frozen protocol');
  assert(config.target==='cpu'||runtime,'Identify target before trials');const slot=free.pop();assert(slot,'No admitted trial slot');
  const publicRoot=join(publicDirectory,randomUUID()),privateRoot=join(input.directory,'verification'),cwd=join(publicRoot,'micro',input.task.id,'0-trial','work'),sessionDirectory=dirname(cwd);let host;
  try{
   const frozen=await freeze(input.task,input.familyId);await mkdir(cwd,{recursive:true});for(const [name,text]of Object.entries({...input.task.files,'public_verify.py':frozen.publicChecker})){const file=inside(cwd,name);await mkdir(dirname(file),{recursive:true});await writeFile(file,text);}
   await mkdir(join(publicRoot,'micro','frozen'),{recursive:true});await writeFile(join(publicRoot,'micro','frozen',input.task.id+'.py'),frozen.publicChecker);
   await save(join(input.directory,'adapter.json'),{publicRoot,privateRoot,cwd});host=await target(publicRoot,slot,'public',frozen.sha256,input.signal);
   const profile=skillProfile(config.actor),hiddenPaths=[privateDirectory,publicDirectory,...hiddenDirectories,...(profile.credentials_file?[dirname(profile.credentials_file)]:[])];
   const actor=await host.sample({cwd,sessionDirectory,profile,guidance:input.guidance,prompt:input.task.prompt+'\nUse native local file/Bash tools for edits and rsi_target for device commands. Public check: python3 public_verify.py .',seconds:input.actorSeconds??180,signal:input.signal,sampleSignal:input.signal,sampleStopSignal:stopSignal,knowledge,localFilesystemSandbox:{hiddenPaths}});
   assert(await host.sampleSettled({cwd})&&await closed(publicRoot),'Actor custody remains unresolved');
   if(actor.session){const contamination=kdbContamination(await readFile(actor.session,'utf8'));if(contamination)return contamination;}
   if(actor.error&&!actor.timedOut)return {infrastructure:true,correct:null,reason:'Native session failed: '+actor.error};
   const files={};async function gather(path,prefix=''){for(const entry of await readdir(path,{withFileTypes:true})){if(entry.name.startsWith('.')||['__pycache__','node_modules'].includes(entry.name))continue;const name=prefix+entry.name;if(entry.isDirectory())await gather(join(path,entry.name),name+'/');else if(entry.isFile()&&entry.name!=='public_verify.py'&&(Object.hasOwn(input.task.files,name)||/\.(py|c|cc|cpp|cxx|cce|h|hpp|inc|cmake|sh|json|md)$/.test(name)||['Makefile','CMakeLists.txt'].includes(name))){assert((await stat(join(path,entry.name))).size<=16*1048576,'Source artifact too large');files[name]=await readFile(join(path,entry.name),'utf8');}}}await gather(cwd);assert(files['kernel.py'],'Candidate kernel missing');
   const protectedNames=[...new Set([...Object.keys(input.task.protectedFiles??{}),...['CONTRACT.md','WORKLOAD.md','case.json','cases.json'].filter(n=>Object.hasOwn(input.task.files,n))])];const unchanged=protectedNames.every(name=>files[name]===input.task.files[name])&&(await readFile(join(cwd,'public_verify.py'),'utf8'))===frozen.publicChecker;
   const rows=await verify(privateRoot,slot,input.task,[{key:'verification',files}],input.verificationSeconds??300,input.signal,input.familyId);assert.equal(rows.length,1,'Trusted verification row missing');const row=rows[0],value=failureReceipt(row);
   await save(join(input.directory,'verification.json'),row);if(typeof value.correct!=='boolean')return {infrastructure:true,correct:null,reason:value.evidence?.reason??value.reason??'Verification unavailable'};
   const correct=value.correct&&unchanged,result={correct,mechanismMatch:!correct&&unchanged&&sameMechanism(value,{mechanism:mechanisms[input.familyId]}),caseResults:input.task.caseIds.map(id=>({id,correct:correct||unchanged&&value.evidence.checkedCases?.includes(id)===true})),sourceSha256:hash(row.sourceFilesSha256),artifacts:[{path:join(input.directory,'verification.json')},...(actor.session?[{path:actor.session}]:[])],handles:[row.handle],actorTimedOut:actor.timedOut,failure:correct?null:unchanged?value.evidence:{phase:'protected-input',reason:'Protected actor input changed'}};
   if(value.correct)assert(input.task.caseIds.every(id=>value.evidence.checkedCases?.includes(id)),'Successful receipt omits frozen cases');
   await save(join(input.directory,'source-manifest.json'),{files:row.sourceFilesSha256,cwd});await save(join(input.directory,'operation-result.json'),result);return result;
  }catch(error){return {infrastructure:true,correct:null,reason:error.message,handle:error.handle??null};}
  finally{if(await closed(publicRoot)&&await closed(privateRoot))free.push(slot);}
 };
 const settle=async op=>{const adapter=await read(join(op.directory,'adapter.json'));if(!adapter)return {closed:false};if(!(await Promise.all((adapter.qualificationRoots??[adapter.publicRoot,adapter.privateRoot]).map(closed))).every(Boolean))return {closed:false};const result=await read(join(op.directory,'operation-result.json')),manifest=await read(join(op.directory,'source-manifest.json'));if(result&&manifest)for(const [name,sha]of Object.entries(manifest.files))assert.equal(hash(await readFile(inside(manifest.cwd,name),'utf8')),sha,'Recovered source changed');return {closed:true,...(result?{result}:{})};};
 return {identify,qualify,trial,settle,close:async()=>{await writes;return (await settleFailureRuntime({directory,config,signal:stopSignal},dependencies)).closed;}};
}
function sameMechanism(value,family){return value.evidence?.phase===family.mechanism?.phase&&value.evidence?.type===family.mechanism?.type&&!!family.mechanism;}
