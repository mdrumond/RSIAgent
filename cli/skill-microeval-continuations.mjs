/** Longer diagnostic continuations; original study artifacts and decisions stay immutable. */
import {mkdir,readFile,writeFile,rename,cp,readdir,lstat,readlink} from 'node:fs/promises';
import {resolve,join,dirname,relative,isAbsolute} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {processIdentity,ownerStopped} from '../controller/lifecycle.mjs';
import {retrievalIdentity} from '../controller/skill-objectives.mjs';
import {nativeSample,skillInput,check,kernelCheck,unknownExecution,protectedFingerprint,validateKernelReceipt} from './skill-microeval.mjs';
const hash=x=>createHash('sha256').update(x).digest('hex');
const load=async p=>JSON.parse(await readFile(p,'utf8'));
async function optional(p){try{return await load(p);}catch(e){if(e.code!=='ENOENT')throw e;return null;}}
async function save(p,x){const tmp=p+'.'+randomUUID();await writeFile(tmp,JSON.stringify(x,null,2)+'\n');await rename(tmp,p);}
const childPath=(root,name)=>{const p=resolve(root,name),r=relative(root,p);if(!r||isAbsolute(r)||r==='..'||r.startsWith('../'))throw Error('Invalid frozen task path');return p;};
async function treeIdentity(root){
 const rows=[];async function walk(dir,prefix=''){
  for(const name of (await readdir(dir)).sort()){
   const p=join(dir,name),key=prefix+name,s=await lstat(p);
   if(s.isDirectory())await walk(p,key+'/');
   else if(s.isSymbolicLink())rows.push([key,'link',await readlink(p)]);
   else if(s.isFile())rows.push([key,hash(await readFile(p))]);
   else throw Error('Unsupported sample file: '+key);
  }
 }await walk(root);return hash(JSON.stringify(rows));
}
async function forkSession({source,cwd,sessionDirectory}){
 const sdk=await import('@earendil-works/pi-coding-agent');
 return sdk.SessionManager.forkFrom(source,cwd,sessionDirectory).getSessionFile();
}
/** Dependencies match runMicroeval's native sample/verification hooks. One key per lane is supported. */
export async function continueMicroevalSamples({originDirectory,directory,sampleKeys,skills,profile,python='python3',seconds=600,verificationSeconds=180,knowledge,signal,resume=true},dependencies={}){
 if(!Number.isFinite(seconds)||seconds<=0||!Number.isFinite(verificationSeconds)||verificationSeconds<0||verificationSeconds>=seconds)throw Error('Positive continuation allowance and smaller verification reserve required');
 signal?.throwIfAborted();originDirectory=resolve(originDirectory);directory=resolve(directory);
 if(directory===originDirectory||directory.startsWith(originDirectory+'/'))throw Error('Continuation directory must be separate from the original study');
 const originalManifest=await readFile(join(originDirectory,'manifest.json'),'utf8'),origin=await load(join(originDirectory,'results.json'));
 const tasks=await load(join(originDirectory,'frozen/tasks.json')),inputs={active:await skillInput(skills.active),candidate:await skillInput(skills.candidate)};
 const model={provider:profile.provider,model:profile.model,maxTokens:profile.max_tokens??null};
 const manifest=JSON.parse(originalManifest);
 if(JSON.stringify(model)!==JSON.stringify(manifest.model))throw Error('Continuation model differs from frozen model');
 if(JSON.stringify(knowledge?retrievalIdentity(knowledge):null)!==JSON.stringify(manifest.knowledge??null))throw Error('Continuation retrieval identity differs');
 for(const arm of ['active','candidate'])if(inputs[arm].sha256!==manifest.skills[arm]?.sha256)throw Error('Continuation guidance changed');
 await mkdir(dirname(directory),{recursive:true});
 let retained=resume?await optional(join(directory,'results.json')):null;
 if(retained?.owner&&!await ownerStopped(retained.owner)||retained?.runningSample)throw Error('Continuation execution remains unresolved');
 const retainedManifest=resume?await optional(join(directory,'manifest.json')):null;
 const keyOf=x=>`${x.taskId}/${x.run}/${x.arm}`;
 sampleKeys??=retainedManifest?.samples.map(x=>x.key)??origin.results.filter(x=>x.timedOut&&x.bucket!=='harness').map(keyOf);
 if(!Array.isArray(sampleKeys)||new Set(sampleKeys).size!==sampleKeys.length)throw Error('Unique continuation sample keys required');
 const selected=[];
 for(const key of sampleKeys){
  const row=origin.results.find(x=>keyOf(x)===key),task=tasks.find(t=>t.id===row?.taskId);
  if(!row?.timedOut||row.bucket==='harness'||row.unresolvedExecution||row.actorObservation?.unresolvedExecution||!task)throw Error('Only settled non-harness timed-out samples are eligible');
  if(origin.runningSample?.key===key)throw Error('Original sample remains unresolved');
  if(!Number.isFinite(row.elapsedSeconds)||row.elapsedSeconds<0)throw Error('Original elapsed evidence required');
  if(hash(JSON.stringify(task))!==manifest.tasks.find(t=>t.id===task.id)?.sha256)throw Error('Original frozen task changed');
  const home=join(originDirectory,task.id,`${row.run}-${row.arm}`),cwd=join(home,'work');
  const sourceSession=row.session??row.actorObservation?.session;
  selected.push({key,row,task,cwd,sourceSession,identity:{key,resultSha256:hash(JSON.stringify(row)),taskSha256:hash(JSON.stringify(task)),sourceTreeSha256:await treeIdentity(cwd),sessionSha256:sourceSession?hash(await readFile(sourceSession)):null,originalElapsedSeconds:row.elapsedSeconds}});
 }
 const identity={format:1,claim:'diagnostic continuations; no primary rescoring or skill promotion',originDirectory,originManifestSha256:hash(originalManifest),seconds,verificationSeconds,model,skills:manifest.skills,knowledge:manifest.knowledge??null,samples:selected.map(x=>x.identity)};
 if(retainedManifest&&JSON.stringify(retainedManifest)!==JSON.stringify(identity))throw Error('Frozen continuation changed');
 if(!retainedManifest){await mkdir(directory);await save(join(directory,'manifest.json'),identity);await mkdir(join(directory,'frozen'));}
 const report=retained??{format:1,claim:identity.claim,status:'running',samples:{}};
 report.owner=await processIdentity();report.status='running';
 const persist=async()=>{await save(join(directory,'results.json'),report);await dependencies.checkpoint?.(report);};
 await persist();
 try{
  for(const item of selected){
   signal?.throwIfAborted();const {key,row,task}=item;
   let state=report.samples[key];if(['complete','elapsed','inconclusive','harness-invalid'].includes(state?.status))continue;
   const home=join(directory,task.id,`${row.run}-${row.arm}`),cwd=join(home,'work'),checker=join(directory,'frozen',task.id+'.py');
   if(!state){
    await mkdir(home,{recursive:true});await cp(item.cwd,cwd,{recursive:true});
    if(await treeIdentity(cwd)!==item.identity.sourceTreeSha256||await treeIdentity(item.cwd)!==item.identity.sourceTreeSha256)throw Error('Original work changed while copying continuation');
    state=report.samples[key]={key,taskId:task.id,run:row.run,arm:row.arm,originalElapsedSeconds:row.elapsedSeconds,extensionElapsedSeconds:0,status:'pending',phase:'initial-verification'};
   }
   await writeFile(checker,task.checker);
   const started=performance.now(),previous=state.extensionElapsedSeconds;
   const left=()=>Math.max(0,seconds-state.originalElapsedSeconds-previous-(performance.now()-started)/1000);
   if(left()<=0){state.status='elapsed';state.timedOut=true;state.elapsedSeconds=state.originalElapsedSeconds+previous;await persist();continue;}
   const deadline=AbortSignal.timeout(Math.max(1,Math.ceil(left()*1000))),wholeSignal=signal?AbortSignal.any([signal,deadline]):deadline;
   report.runningSample={key,phase:state.phase,owner:report.owner};await persist();
   const protectedInputs=async()=>{
    if(hash(await readFile(checker,'utf8'))!==hash(task.checker))throw Error('Frozen checker changed');
    for(const [name,mode]of Object.entries(task.protectedFiles??{}))if(protectedFingerprint(task.files[name],mode)!==protectedFingerprint(await readFile(childPath(cwd,name),'utf8'),mode))throw Error('Protected input changed: '+name);
   };
   const verify=async cap=>{
    await protectedInputs();const allocation=Math.min(cap,left());
    if(allocation<=0)throw Error('Continuation elapsed allocation exhausted');
    const verifySignal=AbortSignal.any([wholeSignal,AbortSignal.timeout(Math.max(1,Math.ceil(allocation*1000)))]);
    let value=await (dependencies.verify?dependencies.verify({task,python,checker,cwd,seconds:allocation,signal:verifySignal}):(task.kind==='a3-kernel'?kernelCheck:check)(python,checker,cwd,allocation,verifySignal));
    if(task.kind==='a3-kernel'&&(value.correct||value.exitCode===125))value=await validateKernelReceipt(value,cwd);
    await protectedInputs();return verifySignal.aborted?{...value,correct:null,timedOut:true,error:'Independent verification elapsed allocation exhausted'}:value;
   };
   try{
    if(!state.initialVerification){
     const checked=await verify(Math.min(120,left()));
     if(!wholeSignal.aborted)state.initialVerification=checked;
     signal?.throwIfAborted();
     if(checked.correct===true){state.verification=checked;state.verificationOnly=true;state.status='complete';state.correct=true;continue;}
     if(checked.timedOut){state.verificationOnly=true;state.actorCompleted=true;}
     else if(checked.correct!==false||!Number.isInteger(checked.exitCode)||checked.exitCode===0||checked.exitCode===125){state.verification=checked;state.correct=checked.correct;state.bucket=checked.receipt?.bucket;state.status='inconclusive';continue;}
    }
    if(!state.actorCompleted&&left()>verificationSeconds){
     if(!state.session){
      if(item.sourceSession){state.session=await (dependencies.forkSession??forkSession)({source:item.sourceSession,cwd,sessionDirectory:home});state.sessionMode='forked';}
      else if(row.actorStarted===false)state.sessionMode='created';
      else throw Error('Original Pi session or positive unstarted-actor evidence required');
      await persist();
     }
     state.phase='actor';report.runningSample.phase=state.phase;await persist();
     const actorSeconds=left()-verificationSeconds,actorDeadline=AbortSignal.timeout(Math.max(1,Math.ceil(actorSeconds*1000))),actorSignal=AbortSignal.any([wholeSignal,actorDeadline]);
     const observed=await (dependencies.sample??nativeSample)({cwd,guidance:task.scopes.map(scope=>`## ${scope}\n${inputs[row.arm].files[scope]}`).join('\n\n'),prompt:task.prompt,profile,kind:task.kind,knowledge,seconds:actorSeconds,verificationReserved:verificationSeconds,signal:actorSignal,sampleSignal:wholeSignal,sampleSeconds:left(),sessionDirectory:home,resumeSession:state.session});
     state.actorObservation=observed;state.session=observed.session??state.session;state.actorSessionTotalsIncludeOriginal=true;
     if(observed.unresolvedExecution)throw Object.assign(Error('Continuation target execution remains unresolved'),{code:'RSI_EXECUTION_UNRESOLVED',...observed.unresolvedExecution});
     state.actorCompleted=!signal?.aborted;state.actorTimedOut=Boolean(observed.timedOut||actorDeadline.aborted);
     signal?.throwIfAborted();
    }
    state.phase='final-verification';report.runningSample.phase=state.phase;await persist();
    state.verification=left()>0?await verify(left()):{correct:null,error:'Continuation elapsed allocation exhausted'};
    state.correct=state.verification.correct;state.status=typeof state.correct==='boolean'?'complete':'inconclusive';
   }catch(error){
    const unknown=unknownExecution(error);
    if(unknown){report.runningSample.unresolvedExecution=unknown;state.status='blocked';throw Object.assign(error,{code:'RSI_EXECUTION_UNRESOLVED',...unknown});}
    if(signal?.aborted){state.status='interrupted';throw error;}
    state.correct=null;state.error=error.message;state.status=/Protected input|Frozen checker/.test(error.message)?'harness-invalid':'inconclusive';
   }finally{
    state.extensionElapsedSeconds=previous+(performance.now()-started)/1000;state.elapsedSeconds=state.originalElapsedSeconds+state.extensionElapsedSeconds;
    if(deadline.aborted){state.timedOut=true;state.correct=null;state.status='elapsed';}
    if(!report.runningSample?.unresolvedExecution)delete report.runningSample;
    await persist();
   }
  }
  report.status='complete';return report;
 }catch(error){report.status='interrupted';report.error=error.message;throw error;}
 finally{delete report.owner;await persist();}
}
