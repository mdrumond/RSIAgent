/** One target-bound hypothesis, frozen inputs, and the caller's remaining elapsed allocation. */
import {mkdir,readFile,writeFile,rename,appendFile,cp} from 'node:fs/promises';
import {join,dirname,basename,relative,resolve,posix} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {evaluationSettings,evaluateSkillObjective,retrievalIdentity} from '../controller/skill-objectives.mjs';
import {runMicroeval,finalizeKernelPairs,nativeSample,skillInput,check,unknownExecution,validateKernelReceipt} from './skill-microeval.mjs';
import {ownerStopped,processIdentity} from '../controller/lifecycle.mjs';
import {createEnvironment} from '../integrations/environments/ascend.mjs';
const hash=value=>createHash('sha256').update(typeof value==='string'?value:JSON.stringify(value)).digest('hex');
const quote=s=>"'"+String(s).replaceAll("'","'\\''")+"'";
const save=async(path,value)=>{const tmp=path+'.'+randomUUID();await writeFile(tmp,JSON.stringify(value,null,2)+'\n');await rename(tmp,path);};
const optional=async path=>{try{return JSON.parse(await readFile(path,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}};
const product=binding=>binding?.product??binding?.target;
export function prepareHypothesisEvaluation({objective,target,tasks,binding}){
 if(objective==='harness')return {status:'unsupported',reason:'user-intervention-required'};
 if(!tasks?.length)return {status:'needs-learning',reason:'No validated hypothesis-relevant evaluation cases'};
 if(product(binding)!==target)return {status:'unsupported',reason:'Evaluation binding differs from hypothesis target'};
 try{
  if(tasks.some(t=>!/^[a-z0-9-]+$/.test(t.id)))throw Error('Unique task identifiers must be lowercase slugs');
  evaluationSettings({[objective]:{tasks,binding}},objective);
  if(target!=='cpu'&&!binding.environment)throw Error('Campaign registered environment binding required');
  return {status:'ready',objective,target,tasks,binding};
 }catch(error){return {status:'unsupported',reason:error.message};}
}

// Existing Ascend environment owns health, masks, leases, cancellation and CPL transport.
export function hypothesisTarget({root,binding,resource,signal,stopSignal,localExecution,event:externalEvent},dependencies={}){
 const contexts=new Map(),factory=dependencies.createEnvironment??createEnvironment;
 const environmentOptions={...binding.environment,...(localExecution?{localExecution}:{})};
 const cleanup=child=>child&&stopSignal?AbortSignal.any([child,stopSignal]):child??stopSignal;
 const make=async(cwd,instancePath,checker,workSignal,cleanupSignal,prepareSignal=workSignal)=>{
  await mkdir(instancePath,{recursive:true});const outputs=new Map(),journal=join(instancePath,'environment-events.jsonl');
  const event=async(type,data)=>{if(type==='execution-output'&&data.handle)outputs.set(data.handle,data);await appendFile(journal,JSON.stringify({type,data})+'\n');await externalEvent?.(type,data);};
  const env=await factory(environmentOptions,{instancePath,campaignRoot:root,event,resource,exclude:async(_name,fn)=>fn(),signal:workSignal,stopSignal:cleanup(cleanupSignal)});
  const id='micro-'+hash(instancePath).slice(0,24);
  await env.prepare({project:{id,source:cwd,bundle:{root},verifier:{files:[checker]}},attempt:{id},signal:prepareSignal});
  const rel=relative(root,checker);if(rel.startsWith('..'))throw Error('Checker must remain in the frozen hypothesis directory');
  const preparedChecker=environmentOptions.mode==='local'?checker:posix.join(environmentOptions.workspacePath,environmentOptions.remoteDirectory,encodeURIComponent(id),'tree',rel.split('\\').join('/'));
  return {env,outputs,checker:preparedChecker,instructions:env.instructions??''};
 };
 const command=async(context,text,cwd,seconds,workSignal,cleanupSignal,purpose)=>{
  let output='';const result=await context.env.execute(text,cwd,{timeout:seconds,signal:workSignal,stopSignal:cleanup(cleanupSignal),purpose,onData:d=>{output=(output+d).slice(-65536);}});
  if(!result.settled)throw Object.assign(Error('Hypothesis execution remains unresolved'),{code:'RSI_EXECUTION_UNRESOLVED',handle:result.handle});
  const files=context.outputs.get(result.handle);if(files?.stdout)output=await readFile(files.stdout,'utf8');
  if(environmentOptions.mode!=='local'&&!files?.stdout)throw Error('Trusted target output unavailable: '+result.handle);
  return {correct:result.exitCode===0,exitCode:result.exitCode,output,handle:result.handle,outputEvidence:files??null};
 };
 return {
  sample:async args=>{
   const began=performance.now(),evaluation=dirname(dirname(dirname(args.cwd))),taskId=basename(dirname(dirname(args.cwd)));
   const context=await make(args.cwd,join(args.sessionDirectory,'target'),join(evaluation,'frozen',taskId+'.py'),args.sampleSignal??args.signal,args.sampleStopSignal,args.signal);contexts.set(args.cwd,context);
   const left=args.seconds-(performance.now()-began)/1000;if(left<=0)throw Error('Sample elapsed allocation exhausted during preparation');
   return (dependencies.nativeSample??nativeSample)({...args,targetBash:environmentOptions.mode==='local',sampleStopSignal:cleanup(args.sampleStopSignal),seconds:left,prompt:args.prompt+'\n\nPrepared target:\n'+context.instructions,targetEnvironment:context.env});
  },
  verify:async args=>{
   let context=contexts.get(args.cwd);if(!context){context=await make(args.cwd,join(dirname(args.cwd),'target'),args.checker,args.signal,args.sampleStopSignal??stopSignal);contexts.set(args.cwd,context);}
   return command(context,[binding.python??'python3',context.checker,'.'].map(quote).join(' '),args.cwd,args.seconds,args.signal,args.sampleStopSignal??stopSignal,'verification');
  },
  measure:async args=>{
   const evaluation=dirname(dirname(dirname(args.active))),home=join(evaluation,'paired-target',args.task.id+'-'+args.run),cwd=join(home,'source');await mkdir(cwd,{recursive:true});
   for(const arm of ['active','candidate']){await mkdir(join(cwd,arm),{recursive:true});for(const file of Object.keys(args.task.files)){await mkdir(dirname(join(cwd,arm,file)),{recursive:true});await cp(join(args[arm],file),join(cwd,arm,file),{recursive:true});}}
   const work=signal?AbortSignal.any([signal,AbortSignal.timeout(Math.max(1,Math.ceil(args.seconds*1000)))]):AbortSignal.timeout(Math.max(1,Math.ceil(args.seconds*1000)));
   const began=performance.now(),context=await make(cwd,join(home,'target'),join(evaluation,'frozen',args.task.id+'-paired.py'),work,stopSignal),left=args.seconds-(performance.now()-began)/1000;
   if(left<=0)throw Error('Paired replay elapsed allocation exhausted during preparation');
   return command(context,[binding.python??'python3',context.checker,'active','candidate'].map(quote).join(' '),cwd,left,work,stopSignal,'performance');
  }
 };
}

export async function evaluateHypothesis({directory,config,objective,target,tasks,binding,skills,remainingSeconds,signal,resource,event},dependencies={}){
 const started=performance.now(),prepared=prepareHypothesisEvaluation({objective,target,tasks,binding});
 if(prepared.status!=='ready')return {...prepared,decision:{objective,promote:false,reason:prepared.reason}};
 if(!Number.isFinite(remainingSeconds)||remainingSeconds<=0||remainingSeconds>1800)throw Error('Remaining hypothesis elapsed allocation must be in (0,1800]');
 await mkdir(directory,{recursive:true});
 const frozen=await Promise.all(tasks.map(async task=>({...task,checker:await readFile(task.checker,'utf8'),...(task.pairedChecker?{pairedChecker:await readFile(task.pairedChecker,'utf8')}:{})})));
 const skillHashes=Object.fromEntries(await Promise.all(Object.entries(skills).map(async([arm,path])=>[arm,(await skillInput(path)).sha256])));
 const {knowledge,...runtimeBinding}=binding;
 const identity=hash({objective,target,binding:runtimeBinding,...(knowledge?{knowledge:retrievalIdentity(knowledge)}:{}),tasks:frozen,skills:skillHashes});
 const file=join(directory,'hypothesis.json'),old=await optional(file);
 if(old&&old.identity!==identity)throw Error('Frozen hypothesis inputs, skill baseline or target changed');
 const interrupted=await optional(join(directory,'micro','results.json'));
 if(interrupted?.settledInterruption)return {status:'inconclusive',identity,directory,report:interrupted,decision:{objective,promote:false,reason:'execution-interrupted'}};
 const available=()=>Math.max(0,remainingSeconds-(performance.now()-started)/1000),cleanupSeconds=30;
 const work=available()-cleanupSeconds;
 if(work<=0)return {status:'elapsed',identity,decision:{objective,promote:false,reason:'inconclusive'}};
 const state=old??{identity,objective,target,binding:runtimeBinding,allocationSeconds:remainingSeconds,developmentSeconds:work-(objective==='optimization'?Math.min(300,work/3):0),pairedSeconds:objective==='optimization'?Math.min(300,work/3):0};
 if(!old)await save(file,state);
 if(available()<=cleanupSeconds)return {status:'elapsed',identity,decision:{objective,promote:false,reason:'inconclusive'}};
 const cleanup=AbortSignal.timeout(Math.max(1,Math.ceil(available()*1000))),deadline=AbortSignal.timeout(Math.max(1,Math.ceil((available()-cleanupSeconds)*1000))),workSignal=signal?AbortSignal.any([signal,deadline]):deadline;
 const targetOps=target==='cpu'?{}:hypothesisTarget({root:resolve(directory),binding,resource,signal:workSignal,stopSignal:cleanup,localExecution:config.local_execution,event},dependencies);
 let report;try{report=await (dependencies.runMicroeval??runMicroeval)({directory:join(directory,'micro'),skills,tasks,profile:config.actor,python:config.python??'python3',runs:2,seconds:180,verificationSeconds:binding.verificationSeconds??30,totalSeconds:state.developmentSeconds,objective,binding:runtimeBinding,knowledge,signal:workSignal,resume:true,shadow:true},
  {...targetOps,...dependencies,sample:dependencies.sample??targetOps.sample,verify:dependencies.verify??targetOps.verify});
 }catch(error){if(error.code==='RSI_EXECUTION_UNRESOLVED'||signal?.aborted||!deadline.aborted)throw error;report=await optional(join(directory,'micro','results.json'))??{results:[]};report.status='elapsed';}
 if(objective==='optimization'&&report.status==='complete'&&available()>cleanupSeconds){
  report=await (dependencies.finalizeKernelPairs??finalizeKernelPairs)({directory:join(directory,'micro'),python:config.python??'python3',seconds:state.pairedSeconds,signal:workSignal},{...targetOps,...dependencies});
 }
 const decision=report.status==='complete'&&!workSignal.aborted?evaluateSkillObjective(report.results,{objective,projectIds:tasks.map(t=>t.id),runs:2,pairedTimings:report.pairedTimings}):{objective,promote:false,reason:'inconclusive'};
 return {status:report.status,identity,decision,report,directory};
}

export async function hypothesisEvaluationSettled({directory,config={},binding,resource,signal},dependencies={}){
 try{
  const file=join(directory,'micro','results.json'),bytes=await readFile(file,'utf8').catch(e=>{if(e.code!=='ENOENT')throw e;});
  if(!bytes)return true;const report=JSON.parse(bytes);
  if(report.owner&&!await ownerStopped(report.owner))return false;
  const marker=report.runningSample??report.pairedReplay?.runningPair;
  if(!marker)return true;
  const unknown=marker.unresolvedExecution;if(!unknown)return false;
  binding??=(await optional(join(directory,'hypothesis.json')))?.binding;
  if(!binding?.environment)return false;
  const parts=report.runningSample?marker.key.split('/'):null;
  if(parts&&(parts.length!==3||parts.some(p=>!p||p==='..')))return false;
  const instancePath=parts?join(directory,'micro',parts[0],parts[1]+'-'+parts[2],'target'):
   join(directory,'micro','paired-target',marker.projectId+'-'+marker.run,'target');
  const observationSignal=signal?AbortSignal.any([signal,AbortSignal.timeout(30000)]):AbortSignal.timeout(30000);
  const env=await (dependencies.createEnvironment??createEnvironment)({...binding.environment,probeObservationSeconds:Math.min(30,binding.environment.probeObservationSeconds??30),...(config.local_execution?{localExecution:config.local_execution}:{})},
   {instancePath,campaignRoot:directory,resource,signal:observationSignal,stopSignal:observationSignal,exclude:async(_name,fn)=>fn(),event:async()=>{}});
  const proofs=[];
  if(unknown.handle){const state=await env.observe(unknown.handle,{signal:observationSignal});if(state.state!=='stopped')return false;proofs.push({handle:unknown.handle,state});}
  for(const reservation of unknown.reservations??[]){
   if(!Number.isInteger(reservation.physicalDevice)||!reservation.receipt||!reservation.token)return false;
   const proof=await env.reconcileReservation(reservation.physicalDevice,{receipt:reservation.receipt,token:reservation.token});if(proof.closed!==true)return false;proofs.push(proof);
  }
  if(!proofs.length||env.reconcileLease&&!await env.reconcileLease())return false;
  if(await readFile(file,'utf8')!==bytes)return false; // Never overwrite a concurrent evaluator's checkpoint.
  const closure=join(directory,'settled-'+randomUUID()+'.json');
  await writeFile(closure,JSON.stringify({at:new Date().toISOString(),marker,proofs,elapsedSeconds:report.elapsedSeconds},null,2)+'\n',{flag:'wx'});
  report.settledInterruption={closure,marker};report.status='inconclusive';
  delete report.runningSample;if(report.pairedReplay)delete report.pairedReplay.runningPair;
  await save(file,report);return true;
 }catch{return false;}
}

/** Qualify a NEW learned test contract before spending model work on either arm. */
export async function qualifyHypothesisTasks({directory,config,objective,target,tasks,binding,remainingSeconds,signal,resource},dependencies={}){
 objective??=tasks?.[0]?.bucket;const prepared=prepareHypothesisEvaluation({objective,target,tasks,binding});if(prepared.status!=='ready')return {status:'needs-learning',reason:prepared.reason};
 if(tasks.some(t=>typeof t.controls?.reference?.files?.['kernel.py']!=='string'||!Array.isArray(t.controls.negatives)||!t.controls.negatives.length||t.controls.negatives.some(c=>typeof c.files?.['kernel.py']!=='string')))return {status:'needs-learning',reason:'Reference and independent negative controls required'};
 if(!Number.isFinite(remainingSeconds)||remainingSeconds<=0||remainingSeconds>1800)throw Error('Remaining hypothesis elapsed allocation must be in (0,1800]');
 const begun=performance.now(),available=()=>Math.max(0,remainingSeconds-(performance.now()-begun)/1000);
 const frozen=await Promise.all(tasks.map(async task=>({...task,checker:await readFile(task.checker,'utf8'),...(task.pairedChecker?{pairedChecker:await readFile(task.pairedChecker,'utf8')}:{})})));
 const {knowledge,...runtimeBinding}=binding,identity=hash({target,tasks:frozen,binding:runtimeBinding});
 const micro=join(directory,'micro'),file=join(micro,'results.json');await mkdir(join(micro,'frozen'),{recursive:true});
 let report=await optional(file);if(report&&report.identity!==identity)throw Error('Frozen qualification contract changed');
 if(report?.runningSample)throw Object.assign(Error('Qualification execution remains unresolved'),{code:'RSI_EXECUTION_UNRESOLVED',handle:report.runningSample.unresolvedExecution?.handle});
 if(report?.settledInterruption||report?.status==='blocked')return {status:'blocked',reason:report.reason??'Qualification execution interrupted',identity,receipts:report.results};
 if(report?.status==='qualified')return {status:'qualified',identity,receipts:report.results,elapsedSeconds:report.elapsedSeconds};
 report??={kind:'qualification',identity,status:'running',results:[],elapsedSeconds:0};const previous=report.elapsedSeconds;
 const persist=async()=>{report.elapsedSeconds=previous+(performance.now()-begun)/1000;await save(file,report);};
 await save(join(directory,'hypothesis.json'),{identity,target,binding:runtimeBinding});report.owner=await processIdentity();await persist();
 const cleanup=AbortSignal.timeout(Math.max(1,Math.ceil(available()*1000))),ops=target==='cpu'?{}:hypothesisTarget({root:resolve(directory),binding,resource,signal,stopSignal:cleanup,localExecution:config.local_execution},dependencies);
 try{
  for(const task of frozen){const checker=join(micro,'frozen',task.id+'.py');await writeFile(checker,task.checker);
   for(const [arm,control,expectedExit]of [['reference',task.controls.reference,0],...task.controls.negatives.map((c,n)=>['negative-'+n,c,1])]){
    const key=task.id+'/0/'+arm;if(report.results.some(r=>r.key===key))continue;signal?.throwIfAborted();
    const seconds=Math.min(180,available()-30);if(seconds<=0){report.status='blocked';report.reason='Qualification elapsed allocation exhausted';return {status:'blocked',reason:report.reason,identity,receipts:report.results};}
    const cwd=join(micro,task.id,'0-'+arm,'work'),files={...task.files,...control.files};await mkdir(cwd,{recursive:true});
    for(const [name,text]of Object.entries(files)){const path=resolve(cwd,name),rel=relative(cwd,path);if(!rel||rel==='..'||rel.startsWith('../'))throw Error('Control files must remain inside their source');await mkdir(dirname(path),{recursive:true});await writeFile(path,text);}
    report.runningSample={key,owner:report.owner};await persist();const start=performance.now(),deadline=AbortSignal.timeout(Math.max(1,Math.ceil(seconds*1000))),work=signal?AbortSignal.any([signal,deadline]):deadline;let verification;
    try{
     verification=await (dependencies.verify??ops.verify??(args=>check(config.python??'python3',args.checker,args.cwd,args.seconds,args.signal)))({task,checker,cwd,seconds,signal:work,sampleSignal:work,sampleStopSignal:cleanup});
     if(hash(await readFile(checker,'utf8'))!==hash(task.checker))throw Error('Frozen qualification checker changed');
     for(const [name,text]of Object.entries(files))if(await readFile(join(cwd,name),'utf8')!==text)throw Error('Qualification changed a control source');
     if(target!=='cpu'&&expectedExit===0&&verification.correct)verification=await validateKernelReceipt(verification,cwd,{requireLatency:false});
    }catch(error){const unknown=unknownExecution(error);if(unknown){report.runningSample.unresolvedExecution=unknown;throw Object.assign(error,{code:'RSI_EXECUTION_UNRESOLVED'});}if(signal?.aborted){delete report.runningSample;throw error;}verification={correct:null,error:error.message};}
    delete report.runningSample;if(signal?.aborted)signal.throwIfAborted();
    const passed=!deadline.aborted&&verification.exitCode===expectedExit&&verification.correct===(expectedExit===0);
    report.results.push({key,expectedExit,passed,elapsedSeconds:(performance.now()-start)/1000,sourceSha256:hash(files),verification});await persist();
    if(!passed){report.status='blocked';report.reason=expectedExit===0?'Reference control failed':'Negative control was not independently rejected';return {status:'blocked',reason:report.reason,identity,receipts:report.results};}
   }
  }
  report.status='qualified';return {status:'qualified',identity,receipts:report.results,elapsedSeconds:report.elapsedSeconds};
 }finally{delete report.owner;await persist();}
}
