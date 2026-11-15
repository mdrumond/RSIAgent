/** One target-bound hypothesis, frozen inputs, and the caller's remaining elapsed allocation. */
import {mkdir,readFile,writeFile,rename,appendFile,cp,readdir} from 'node:fs/promises';
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

// One admitted process runs trusted controls sequentially; it never invokes a model or transport.
async function trustedBatch({cwd,checker,entries,seconds},execute){
 await mkdir(cwd,{recursive:true});const plan=[];
 for(const entry of entries){const source=join(cwd,entry.cwd),expected={};await mkdir(source,{recursive:true});
  for(const [name,text]of Object.entries(entry.files)){const path=resolve(source,name),rel=relative(source,path);if(!rel||rel==='..'||rel.startsWith('../'))throw Error('Batch source escapes its control directory');await mkdir(dirname(path),{recursive:true});await writeFile(path,text);expected[name]=hash(text);}
  plan.push({...entry,files:undefined,expected,checkerSha256:hash(entry.checker)});
 }
 const script=`import sys,json,base64,pathlib,subprocess,time,tempfile,hashlib
plan=json.loads(base64.b64decode('${Buffer.from(JSON.stringify(plan)).toString('base64')}')); root=pathlib.Path(sys.argv[1]).resolve(); start=time.monotonic()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='rsi-check-') as temporary:
 for i,c in enumerate(plan):
  source=root/c['cwd']; checker=pathlib.Path(temporary)/(str(i)+'.py'); checker.write_text(c['checker']); before={n:sha(source/n) for n in c['expected']}; left=${seconds}-(time.monotonic()-start)
  if left<=0: break
  command=[sys.executable,str(checker),*c['args']]; begun=time.monotonic()
  try:
   p=subprocess.run(command,cwd=source,capture_output=True,text=True,errors='replace',timeout=min(180,left)); code=p.returncode; output=p.stdout; error=p.stderr
  except subprocess.TimeoutExpired:
   code=None; output=''; error='Control elapsed allocation exhausted'
  after={n:sha(source/n) for n in c['expected']}
  print(json.dumps({'key':c['key'],'exitCode':code,'output':output[-8192:],'error':error[-2048:],'elapsedSeconds':time.monotonic()-begun,'command':command,'checkerSha256':sha(checker),'sourceFilesSha256':after,'sourceFilesMatch':before==c['expected'] and after==c['expected']}),flush=True)
  if code is None: break
`;
 await mkdir(dirname(checker),{recursive:true});await writeFile(checker,script);const result=await execute();
 if(result.exitCode!==0)throw Object.assign(Error('Trusted batch command failed'),{code:'TRUSTED_BATCH_CLOSED',closure:{closed:true,handles:result.handle?[result.handle]:[]},verification:result});
 const rows=result.output.trim().split('\n').filter(Boolean).map(line=>JSON.parse(line));const keys=new Set();
 for(const row of rows){const entry=plan.find(e=>e.key===row.key);if(!entry||keys.has(row.key)||!row.sourceFilesMatch||row.checkerSha256!==entry.checkerSha256)throw Error('Batch control source, checker or identity changed');keys.add(row.key);row.correct=row.exitCode===0;row.handle=result.handle;row.outputEvidence=result.outputEvidence;}
 return rows;
}

// Existing Ascend environment owns health, masks, leases, cancellation and CPL transport.
export function hypothesisTarget({root,binding,resource,signal,stopSignal,localExecution,event:externalEvent},dependencies={}){
 const contexts=new Map(),factory=dependencies.createEnvironment??createEnvironment;
 const environmentOptions={...binding.environment,...(localExecution?{localExecution}:{})};
 const cleanup=child=>child&&stopSignal?AbortSignal.any([child,stopSignal]):child??stopSignal;
 const make=async(cwd,instancePath,checker,workSignal,cleanupSignal,prepareSignal=workSignal)=>{
  await mkdir(instancePath,{recursive:true});const outputs=new Map(),journal=join(instancePath,'environment-events.jsonl');
  const event=async(type,data)=>{if(type==='execution-output'&&data.handle)outputs.set(data.handle,data);await appendFile(journal,JSON.stringify({type,data,created:new Date().toISOString()})+'\n');await externalEvent?.(type,data);};
  const env=await factory(environmentOptions,{instancePath,campaignRoot:root,event,resource,exclude:async(_name,fn)=>fn(),signal:workSignal,stopSignal:cleanup(cleanupSignal)});
  const id='micro-'+hash(instancePath).slice(0,24);
  await env.prepare({project:{id,source:cwd,bundle:{root},verifier:{files:[checker]}},attempt:{id},signal:prepareSignal});
  const rel=relative(root,checker);if(rel.startsWith('..'))throw Error('Checker must remain in the frozen hypothesis directory');
  const preparedChecker=environmentOptions.mode==='local'?checker:posix.join(environmentOptions.workspacePath,environmentOptions.remoteDirectory,encodeURIComponent(id),'tree',rel.split('\\').join('/'));
  return {env,outputs,checker:preparedChecker,instructions:env.instructions??''};
 };
 const command=async(context,text,cwd,seconds,workSignal,cleanupSignal,purpose)=>{
  let output='';const result=await context.env.execute(text,cwd,{timeout:seconds,signal:workSignal,stopSignal:cleanup(cleanupSignal),purpose,onData:d=>{output=(output+d).slice(-1048576);}});
  if(!result.settled)throw Object.assign(Error('Hypothesis execution remains unresolved'),{code:'RSI_EXECUTION_UNRESOLVED',handle:result.handle});
  const files=context.outputs.get(result.handle);if(files?.stdout)output=await readFile(files.stdout,'utf8');
  if(environmentOptions.mode!=='local'&&!files?.stdout)throw Error('Trusted target output unavailable: '+result.handle);
  return {correct:result.exitCode===0,exitCode:result.exitCode,output,handle:result.handle,outputEvidence:files??null};
 };
 return {
  runtimeIdentity:async args=>{
   const micro=join(args.directory,'micro'),cwd=join(micro,'runtime','0-probe','work'),checker=join(micro,'frozen','runtime.py'),file=join(micro,'results.json');
   await mkdir(cwd,{recursive:true});await mkdir(dirname(checker),{recursive:true});
   const source=await readFile(new URL('./skill-suites/runtime-identity.py',import.meta.url),'utf8');
   await writeFile(checker,source+'\nprint(json.dumps(identity(),sort_keys=True))\n');
   await save(join(args.directory,'hypothesis.json'),{binding});
   const report={owner:await processIdentity(),runningSample:{key:'runtime/0/probe'},results:[]};await save(file,report);
   try{
    const began=performance.now(),context=await make(cwd,join(dirname(cwd),'target'),checker,args.signal,args.sampleStopSignal);
    const left=args.seconds-(performance.now()-began)/1000;if(left<=0)throw Error('Runtime identity allocation exhausted');
    const result=await command(context,[binding.python??'python3',context.checker].map(quote).join(' '),cwd,left,args.signal,args.sampleStopSignal,'verification');
    if(result.exitCode!==0)throw Error('Fresh target runtime identity unavailable');
    return JSON.parse(result.output);
   }catch(error){const unknown=unknownExecution(error);if(unknown){report.runningSample.unresolvedExecution=unknown;throw Object.assign(error,{code:'RSI_EXECUTION_UNRESOLVED'});}delete report.runningSample;throw error;}
   finally{if(!report.runningSample?.unresolvedExecution)delete report.runningSample;delete report.owner;await save(file,report);}
  },
  sample:async args=>{
   const began=performance.now(),evaluation=dirname(dirname(dirname(args.cwd))),taskId=basename(dirname(dirname(args.cwd)));
   const context=await make(args.cwd,join(args.sessionDirectory,'target'),join(evaluation,'frozen',taskId+'.py'),args.sampleSignal??args.signal,args.sampleStopSignal,args.signal);contexts.set(args.cwd,context);
   const left=args.seconds-(performance.now()-began)/1000;if(left<=0)throw Error('Sample elapsed allocation exhausted during preparation');
   return (dependencies.nativeSample??nativeSample)({...args,targetBash:environmentOptions.mode==='local',sampleStopSignal:cleanup(args.sampleStopSignal),seconds:left,prompt:args.prompt+'\n\nPrepared target:\n'+context.instructions,targetEnvironment:context.env});
  },
  sampleSettled:async({cwd})=>{const context=contexts.get(cwd);return Boolean(context?.env.reconcileLease&&await context.env.reconcileLease());},
  verify:async args=>{
   let context=contexts.get(args.cwd);if(!context){context=await make(args.cwd,join(dirname(args.cwd),'target'),args.checker,args.signal,args.sampleStopSignal??stopSignal);contexts.set(args.cwd,context);}
   return command(context,[binding.python??'python3',context.checker,'.'].map(quote).join(' '),args.cwd,args.seconds,args.signal,args.sampleStopSignal??stopSignal,'verification');
  },
  qualifyBatch:async args=>{
   const cwd=join(args.directory,'micro','qualification','0-batch','work'),checker=join(args.directory,'micro','frozen','qualification-batch.py');
   return trustedBatch({cwd,checker,entries:args.entries,seconds:args.seconds},async()=>{
    const begun=performance.now(),context=await make(cwd,join(dirname(cwd),'target'),checker,args.signal,args.sampleStopSignal),left=args.seconds-(performance.now()-begun)/1000;
    if(left<=0)throw Error('Batch qualification elapsed allocation exhausted during preparation');
    return command(context,[binding.python??'python3',context.checker,'.'].map(quote).join(' '),cwd,left,args.signal,args.sampleStopSignal,'verification');
   });
  },
  measureBatch:async args=>{
   const home=join(args.directory,'paired-target','batch-0'),cwd=join(home,'source'),checker=join(args.directory,'frozen','paired-batch.py'),entries=[];
   for(const pair of args.pairs){const files={};for(const arm of ['active','candidate'])for(const name of Object.keys(pair.task.files))files[arm+'/'+name]=await readFile(join(pair[arm],name),'utf8');entries.push({key:pair.task.id+'/'+pair.run,cwd:pair.task.id+'-'+pair.run,files,checker:pair.task.pairedChecker,args:['active','candidate']});}
   const rows=await trustedBatch({cwd,checker,entries,seconds:args.seconds},async()=>{
    const begun=performance.now(),context=await make(cwd,join(home,'target'),checker,args.signal,stopSignal),left=args.seconds-(performance.now()-begun)/1000;
    if(left<=0)throw Error('Batch paired replay elapsed allocation exhausted during preparation');
    return command(context,[binding.python??'python3',context.checker,'.'].map(quote).join(' '),cwd,left,args.signal,stopSignal,'performance');
   });
   return rows.map(row=>{const pair=args.pairs.find(p=>p.task.id+'/'+p.run===row.key);return {...row,projectId:pair.task.id,run:pair.run};});
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

/** Freeze timing before proposal review; historical roots retain their original wire identity. */
export async function freezeHypothesisProtocol({directory,target,binding}){
 const file=join(directory,'evaluation-protocol.json'),saved=await optional(file),state=await optional(join(directory,'hypothesis.json')),manifest=await optional(join(directory,'micro','manifest.json'))??await optional(join(directory,'micro','results.json'));
 const explicit=binding.verificationSeconds;
 if(explicit!==undefined&&(!Number.isFinite(explicit)||explicit<0||explicit>=180))throw Error('Verification reserve must fit inside the sample elapsed allocation');
 if(saved){
  if(saved.target!==target||saved.explicitVerificationSeconds!==(explicit??null))throw Error('Frozen hypothesis sample timing changed');
  return {protocol:saved.protocol,verificationSeconds:saved.protocol.verificationSeconds};
 }
 if(state||manifest){
  const verificationSeconds=state?.protocol?.verificationSeconds??manifest?.verificationSeconds??state?.binding?.verificationSeconds??30;
  if(explicit!==undefined&&explicit!==verificationSeconds)throw Error('Frozen hypothesis sample timing changed');
  return {protocol:state?.protocol??null,verificationSeconds};
 }
 const verificationSeconds=explicit??(target==='cpu'?30:90),protocol={version:1,sampleSeconds:180,actorSeconds:180-verificationSeconds,verificationSeconds,policy:target==='cpu'?'cpu-reserve-v1':'target-transport-reserve-v1'};
 await mkdir(directory,{recursive:true});const record={target,explicitVerificationSeconds:explicit??null,protocol},bytes=JSON.stringify(record,null,2)+'\n';
 try{await writeFile(file,bytes,{flag:'wx'});}catch(error){if(error.code!=='EEXIST')throw error;if(await readFile(file,'utf8')!==bytes)throw Error('Frozen hypothesis sample timing changed');}
 return {protocol,verificationSeconds};
}

export async function evaluateHypothesis({directory,config,objective,target,tasks,binding,skills,remainingSeconds,signal,resource,event},dependencies={}){
 const started=performance.now(),prepared=prepareHypothesisEvaluation({objective,target,tasks,binding});
 if(prepared.status!=='ready')return {...prepared,decision:{objective,promote:false,reason:prepared.reason}};
 if(!Number.isFinite(remainingSeconds)||remainingSeconds<=0||remainingSeconds>1800)throw Error('Remaining hypothesis elapsed allocation must be in (0,1800]');
 await mkdir(directory,{recursive:true});
 const frozen=await Promise.all(tasks.map(async task=>({...task,checker:await readFile(task.checker,'utf8'),...(task.pairedChecker?{pairedChecker:await readFile(task.pairedChecker,'utf8')}:{})})));
 const skillHashes=Object.fromEntries(await Promise.all(Object.entries(skills).map(async([arm,path])=>[arm,(await skillInput(path)).sha256])));
 const {knowledge,...runtimeBinding}=binding;
 const {protocol,verificationSeconds}=await freezeHypothesisProtocol({directory,target,binding});
 const identity=hash({objective,target,...(protocol?{protocol}:{}),binding:runtimeBinding,...(knowledge?{knowledge:retrievalIdentity(knowledge)}:{}),tasks:frozen,skills:skillHashes});
 const file=join(directory,'hypothesis.json'),old=await optional(file);
 if(old&&old.identity!==identity)throw Error('Frozen hypothesis inputs, skill baseline or target changed');
 const interrupted=await optional(join(directory,'micro','results.json'));
 if(interrupted?.settledInterruption)return {status:'inconclusive',identity,directory,report:interrupted,decision:{objective,promote:false,reason:'execution-interrupted'}};
 const available=()=>Math.max(0,remainingSeconds-(performance.now()-started)/1000),cleanupSeconds=30;
 const work=available()-cleanupSeconds;
 if(work<=0)return {status:'elapsed',identity,decision:{objective,promote:false,reason:'inconclusive'}};
 const state=old??{identity,objective,target,...(protocol?{protocol}:{}),binding:runtimeBinding,allocationSeconds:remainingSeconds,developmentSeconds:work-(objective==='optimization'?Math.min(300,work/3):0),pairedSeconds:objective==='optimization'?Math.min(300,work/3):0};
 if(!old)await save(file,state);
 if(available()<=cleanupSeconds)return {status:'elapsed',identity,decision:{objective,promote:false,reason:'inconclusive'}};
 const cleanup=AbortSignal.timeout(Math.max(1,Math.ceil(available()*1000))),deadline=AbortSignal.timeout(Math.max(1,Math.ceil((available()-cleanupSeconds)*1000))),workSignal=signal?AbortSignal.any([signal,deadline]):deadline;
 const targetOps=target==='cpu'?{}:hypothesisTarget({root:resolve(directory),binding,resource,signal:workSignal,stopSignal:cleanup,localExecution:config.local_execution,event},dependencies);
 let report;try{report=await (dependencies.runMicroeval??runMicroeval)({directory:join(directory,'micro'),skills,tasks,profile:config.actor,python:config.python??'python3',runs:2,seconds:180,verificationSeconds,...(protocol?{sampleProtocol:protocol}:{}),totalSeconds:state.developmentSeconds,objective,binding:runtimeBinding,knowledge,signal:workSignal,resume:true,shadow:true,requireDevice:target!=='cpu'},
  {...targetOps,...dependencies,sample:dependencies.sample??targetOps.sample,verify:dependencies.verify??targetOps.verify});
 if(objective==='optimization'&&report.status==='complete'&&available()>cleanupSeconds){
  report=await (dependencies.finalizeKernelPairs??finalizeKernelPairs)({directory:join(directory,'micro'),python:config.python??'python3',seconds:state.pairedSeconds,signal:workSignal,requireDevice:target!=='cpu'},{...targetOps,...dependencies});
 }
 }catch(error){if(error.code==='RSI_EXECUTION_UNRESOLVED'||signal?.aborted||!deadline.aborted)throw error;report=await optional(join(directory,'micro','results.json'))??{results:[]};report.status='elapsed';await save(join(directory,'micro','results.json'),report);}
 const decision=report.status==='complete'&&!workSignal.aborted?evaluateSkillObjective(report.results,{objective,projectIds:tasks.map(t=>t.id),runs:2,pairedTimings:report.pairedTimings}):{objective,promote:false,reason:'inconclusive'};
 return {status:report.status,identity,decision,report,directory};
}

async function retainedTargets(directory){
 const targets=[];const dirs=async path=>{try{return (await readdir(path,{withFileTypes:true})).filter(entry=>entry.isDirectory());}catch(error){if(error.code==='ENOENT')return [];throw error;}};
 const micro=join(directory,'micro');
 for(const parent of await dirs(micro))if(parent.name!=='frozen')for(const child of await dirs(join(micro,parent.name))){
  const target=join(micro,parent.name,child.name,'target');
  if(await optional(join(target,'.ascend-command-lease.json'))||await readFile(join(target,'environment-events.jsonl'),'utf8').catch(error=>{if(error.code!=='ENOENT')throw error;return '';}))targets.push(target);
 }
 return targets;
}
async function targetCustody({instancePath,directory,binding,config,resource,signal,unknown},dependencies){
 const journal=join(instancePath,'environment-events.jsonl'),text=await readFile(journal,'utf8').catch(error=>{if(error.code!=='ENOENT')throw error;return '';});
 const lines=text.split('\n');if(!text.endsWith('\n'))lines.pop();const events=lines.filter(Boolean).map(line=>JSON.parse(line)),pointer=await optional(join(instancePath,'.ascend-command-lease.json'));
 const proofs=[],covered=new Set(),terminal=new Map(events.filter(event=>event.type==='execution-state'&&event.data.state==='stopped').map(event=>[event.data.handle,event.data]));
 let env;const environment=async()=>env??=await (dependencies.createEnvironment??createEnvironment)({...binding.environment,probeObservationSeconds:Math.min(30,binding.environment.probeObservationSeconds??30),...(config.local_execution?{localExecution:config.local_execution}:{})},
  {instancePath,campaignRoot:directory,resource,signal,stopSignal:signal,exclude:async(_name,fn)=>fn(),event:async()=>{}});
 const observe=async handle=>{if(terminal.has(handle)){proofs.push({handle,state:terminal.get(handle),journalClosed:true});covered.add(handle);return true;}const state=await (await environment()).observe(handle,{signal});if(state.state!=='stopped')return false;proofs.push({handle,state});covered.add(handle);return true;};
 const submitted=events.filter(event=>event.type==='execution-submitting');
 const operations=new Set(events.filter(event=>['execution-starting','execution-submitting'].includes(event.type)).map(event=>event.data.operation??event.data.submission));
 for(const operation of operations){
  if(!operation)return null;const group=submitted.filter(event=>(event.data.operation??event.data.submission)===operation);
  const responses=group.map(row=>events.find(event=>['execution-submitted','execution-not-admitted'].includes(event.type)&&event.data.submission===row.data.submission));
  if(group.length&&responses.every(response=>response?.type==='execution-not-admitted'&&response.data.closed===true||response?.data.handle&&terminal.has(response.data.handle))){
   for(const response of responses)if(response.data.handle){covered.add(response.data.handle);proofs.push({handle:response.data.handle,state:terminal.get(response.data.handle),journalClosed:true});}else proofs.push({submission:response.data.submission,notAdmitted:true,closed:true});continue;
  }
  const context=await environment();if(!context.reconcileOperation)return null;const proof=await context.reconcileOperation({operation,events});
  if(proof.closed!==true){
   if(!group.length)return null;
   for(const response of responses){if(response?.type==='execution-not-admitted'&&response.data.closed===true)continue;if(!response?.data.handle||!await observe(response.data.handle))return null;}
  }
  for(const handle of [...(proof.handles??[]),...(proof.deferredHandles??[])])covered.add(handle);proofs.push({operation,...proof,...(proof.closed!==true?{nativeClosed:true}:{})});
 }
 for(const handle of new Set(events.filter(event=>event.type==='execution-handle').map(event=>event.data.handle).filter(Boolean)))if(!covered.has(handle)&&!await observe(handle))return null;
 if(unknown?.handle&&!covered.has(unknown.handle)&&!await observe(unknown.handle))return null;
 for(const reservation of unknown?.reservations??[]){if(!Number.isInteger(reservation.physicalDevice)||!reservation.receipt||!reservation.token)return null;const proof=await (await environment()).reconcileReservation(reservation.physicalDevice,{receipt:reservation.receipt,token:reservation.token});if(proof.closed!==true)return null;proofs.push(proof);}
 if(pointer){
  if(!pointer.receipt||!pointer.token||!Number.isInteger(pointer.physicalDevice))return null;
  const context=await environment();if(!context.reconcileLease||!await context.reconcileLease())return null;proofs.push({lease:pointer,closed:true});
 }else for(const {data}of events.filter(event=>event.type==='device-lease-acquired')){
  if(events.some(event=>event.type==='device-lease-released'&&event.data.token===data.token&&event.data.receipt===data.receipt||event.type==='device-cleanup-queued'&&event.data.token===data.token)){proofs.push({lease:data,journalClosed:true});continue;}
  const context=await environment();if(!context.reconcileReservation)return null;const proof=await context.reconcileReservation(data.physicalDevice,{receipt:data.receipt,token:data.token});if(proof.closed!==true)return null;proofs.push(proof);
 }
 if(!proofs.length)return null;
 if(env?.reconcileLease&&!await env.reconcileLease())return null;
 if(await readFile(journal,'utf8').catch(error=>{if(error.code!=='ENOENT')throw error;return '';})!==text)return null;
 return {instancePath,journalSha256:hash(text),proofs};
}
export async function hypothesisEvaluationSettled({directory,config={},binding,resource,signal,retirePrePayload=false},dependencies={}){
 try{
  const continuation=await optional(join(directory,'qualification-continuation.json'));
  if(continuation)return hypothesisEvaluationSettled({directory:continuation.directory,config,binding,resource,signal,retirePrePayload},dependencies);
  const file=join(directory,'micro','results.json'),bytes=await readFile(file,'utf8').catch(error=>{if(error.code!=='ENOENT')throw error;});const report=bytes?JSON.parse(bytes):null;
  if(report?.owner&&!await ownerStopped(report.owner))return false;
  const marker=report?.runningSample??report?.pairedReplay?.runningPair,targets=new Set(await retainedTargets(directory));
  let markerTarget;
  if(marker){const parts=report.runningSample?marker.key.split('/'):null;if(parts&&(parts.length!==3||parts.some(part=>!part||part==='..')))return false;
   markerTarget=parts?join(directory,'micro',parts[0],parts[1]+'-'+parts[2],'target'):join(directory,'micro','paired-target',marker.projectId+'-'+marker.run,'target');targets.add(markerTarget);}
  if(!targets.size)return true;
  binding??=(await optional(join(directory,'hypothesis.json')))?.binding;if(!binding?.environment)return false;
  const observationSignal=signal?AbortSignal.any([signal,AbortSignal.timeout(30000)]):AbortSignal.timeout(30000),ledgerFile=join(directory,'target-settlements.json'),ledger=await optional(ledgerFile)??{};
  const closures=[];let markerClosure,reportBytes=bytes;
  for(const instancePath of targets){
   observationSignal.throwIfAborted();const key=relative(directory,instancePath),journal=await readFile(join(instancePath,'environment-events.jsonl'),'utf8').catch(error=>{if(error.code!=='ENOENT')throw error;return '';}),pointer=await optional(join(instancePath,'.ascend-command-lease.json'));
   if(instancePath!==markerTarget&&!pointer&&ledger[key]?.journalSha256===hash(journal)&&await optional(ledger[key].closure)){closures.push(ledger[key].closure);continue;}
   let proof=await targetCustody({instancePath,directory,binding,config,resource,signal:observationSignal,unknown:instancePath===markerTarget?marker?.unresolvedExecution:null},dependencies);
   const retirementPointer=pointer??(report?.settledInterruption?.retirementPending?await optional(report.settledInterruption.fence):null);
   if(!proof&&retirePrePayload&&marker&&instancePath===markerTarget&&retirementPointer){
    // Fence the scientific comparison durably BEFORE releasing its device. Even
    // a crash or failed retirement cannot replay/rescore this partial target.
    if(reportBytes&&await readFile(file,'utf8')!==reportBytes)return false;
    const fence=join(directory,'pre-payload-retirement-'+hash(instancePath)+'.json');
    const guard={permanent:true,instancePath,receipt:retirementPointer.receipt,token:retirementPointer.token,marker};
    const previous=await optional(fence);if(previous&&JSON.stringify(previous)!==JSON.stringify(guard))return false;if(!previous)await save(fence,guard);
    report.settledInterruption??={retirementPending:true,fence,marker};await save(file,report);reportBytes=await readFile(file,'utf8');
    const env=await (dependencies.createEnvironment??createEnvironment)({...binding.environment,...(config.local_execution?{localExecution:config.local_execution}:{})},{instancePath,campaignRoot:directory,resource,signal:observationSignal,stopSignal:observationSignal,exclude:async(_name,fn)=>fn(),event:async()=>{}});
    if(!env.retirePrePayloadCustody)return false;
    const retired=await env.retirePrePayloadCustody({...retirementPointer,fence});if(retired.closed!==true)return false;
    proof={instancePath,journalSha256:hash(journal),proofs:[retired]};
   }
   if(!proof)return false;
   const closure=join(directory,'settled-'+randomUUID()+'.json');await writeFile(closure,JSON.stringify({at:new Date().toISOString(),marker:instancePath===markerTarget?marker:null,...proof},null,2)+'\n',{flag:'wx'});
   ledger[key]={journalSha256:proof.journalSha256,closure};await save(ledgerFile,ledger);closures.push(closure);if(instancePath===markerTarget)markerClosure=closure;
  }
  if(reportBytes&&await readFile(file,'utf8')!==reportBytes)return false;
  if(marker){report.settledInterruption={closure:markerClosure,closures,marker};report.status='inconclusive';delete report.runningSample;if(report.pairedReplay)delete report.pairedReplay.runningPair;await save(file,report);}
  return true;
 }catch{return false;}
}

const qualificationFailure=results=>{
 const failed=results.filter(result=>!result.passed);
 return failed.length&&failed.every(result=>result.verification.exitCode===125&&result.verification.receipt?.bucket==='harness')?'infrastructure':'controls';
};
/** A closed infrastructure failure may continue under its original elapsed allocation. */
export async function qualifyHypothesisTasks(args,dependencies={}){
 const {directory,remainingSeconds}=args;
 if(!Number.isFinite(remainingSeconds)||remainingSeconds<=0||remainingSeconds>1800)throw Error('Remaining hypothesis elapsed allocation must be in (0,1800]');
 const begun=performance.now(),redirect=await optional(join(directory,'qualification-continuation.json'));
 if(redirect)return qualifyHypothesisTasks({...args,directory:redirect.directory,remainingSeconds:Math.max(.001,remainingSeconds-(performance.now()-begun)/1000)},dependencies);
 const file=join(directory,'micro','results.json'),report=await optional(file);
 if(report?.status==='blocked'&&report.failureKind==='infrastructure'&&Number.isFinite(report.allocationSeconds)){
  const ledgerFile=join(directory,'qualification-recovery.json'),ledger=await optional(ledgerFile)??{elapsedSeconds:0},previous=ledger.elapsedSeconds;
  const spent=()=>report.elapsedSeconds+previous+(performance.now()-begun)/1000;
  if(Math.min(remainingSeconds,report.allocationSeconds-spent())<=30)return {status:'blocked',reason:'Qualification elapsed allocation exhausted',identity:report.identity,receipts:report.results,elapsedSeconds:spent(),directory};
  const persist=()=>save(ledgerFile,{elapsedSeconds:previous+(performance.now()-begun)/1000});
  let pending=Promise.resolve();const timer=setInterval(()=>{pending=pending.then(persist);},1000);let closed;
  try{closed=await (dependencies.settled??hypothesisEvaluationSettled)({...args},dependencies);}
  finally{clearInterval(timer);await pending;await persist();}
  if(!closed)return {status:'blocked',reason:'Qualification infrastructure execution remains unresolved',identity:report.identity,receipts:report.results,elapsedSeconds:spent(),directory};
  const elapsedSeconds=spent(),left=Math.min(remainingSeconds-(performance.now()-begun)/1000,report.allocationSeconds-elapsedSeconds);
  if(left<=30)return {status:'blocked',reason:'Qualification elapsed allocation exhausted',identity:report.identity,receipts:report.results,elapsedSeconds,directory};
  const next=join(directory,'continuations',randomUUID());await mkdir(join(next,'micro'),{recursive:true});
  const continuation={from:directory,at:new Date().toISOString(),closed:true,elapsedSeconds,originalAllocationSeconds:report.allocationSeconds};
  const carried={...report,status:'running',elapsedSeconds,results:report.results.filter(result=>result.passed),continuation};
  delete carried.reason;delete carried.failureKind;delete carried.owner;delete carried.runningSample;
  await save(join(next,'micro','results.json'),carried);
  await save(join(directory,'qualification-continuation.json'),{directory:next,...continuation}); // Custody is indexed before any child command.
  return qualifyHypothesisTasks({...args,directory:next,remainingSeconds:left},dependencies);
 }
 return {...await qualifyAttempt(args,dependencies),directory};
}

/** Qualify controls once; bad references and elapsed failures never renew. */
async function qualifyAttempt({directory,config,objective,target,tasks,binding,remainingSeconds,signal,resource},dependencies={}){
 objective??=tasks?.[0]?.bucket;const prepared=prepareHypothesisEvaluation({objective,target,tasks,binding});if(prepared.status!=='ready')return {status:'needs-learning',reason:prepared.reason};
 if(tasks.some(t=>typeof t.controls?.reference?.files?.['kernel.py']!=='string'||!Array.isArray(t.controls.negatives)||!t.controls.negatives.length||t.controls.negatives.some(c=>typeof c.files?.['kernel.py']!=='string')))return {status:'needs-learning',reason:'Reference and independent negative controls required'};
 if(!Number.isFinite(remainingSeconds)||remainingSeconds<=0||remainingSeconds>1800)throw Error('Remaining hypothesis elapsed allocation must be in (0,1800]');
 const begun=performance.now();let allocationSeconds=remainingSeconds,priorElapsed=0;
 const available=()=>Math.max(0,Math.min(remainingSeconds,allocationSeconds-priorElapsed)-(performance.now()-begun)/1000);
 const frozen=await Promise.all(tasks.map(async task=>({...task,checker:await readFile(task.checker,'utf8'),...(task.pairedChecker?{pairedChecker:await readFile(task.pairedChecker,'utf8')}:{})})));
 const {knowledge,...runtimeBinding}=binding,identity=hash({target,tasks:frozen,binding:runtimeBinding});
 const micro=join(directory,'micro'),file=join(micro,'results.json');await mkdir(join(micro,'frozen'),{recursive:true});
 let report=await optional(file);if(report&&report.identity!==identity)throw Error('Frozen qualification contract changed');
 if(report?.runningSample)throw Object.assign(Error('Qualification execution remains unresolved'),{code:'RSI_EXECUTION_UNRESOLVED',handle:report.runningSample.unresolvedExecution?.handle});
 if(report?.owner&&!await ownerStopped(report.owner))throw Object.assign(Error('Qualification owner remains live'),{code:'RSI_EXECUTION_UNRESOLVED'});
 if(report?.status==='running'&&report.results.some(result=>!result.passed)){report.status='blocked';report.failureKind=qualificationFailure(report.results);report.reason='Interrupted failed control remains unqualified';await save(file,report);return {status:'blocked',reason:report.reason,identity,receipts:report.results};}
 if(report?.settledInterruption||report?.status==='blocked')return {status:'blocked',reason:report.reason??'Qualification execution interrupted',identity,receipts:report.results};
 if(report?.status==='qualified')return {status:'qualified',identity,receipts:report.results,elapsedSeconds:report.elapsedSeconds};
 priorElapsed=report?.elapsedSeconds??0;allocationSeconds=report?.allocationSeconds??priorElapsed+remainingSeconds;
 report??={kind:'qualification',identity,status:'running',results:[],elapsedSeconds:0,allocationSeconds};report.allocationSeconds??=allocationSeconds;const previous=report.elapsedSeconds;
 const persist=async()=>{report.elapsedSeconds=previous+(performance.now()-begun)/1000;await save(file,report);};
 await save(join(directory,'hypothesis.json'),{identity,target,binding:runtimeBinding});report.owner=await processIdentity();await persist();
 const cleanup=AbortSignal.timeout(Math.max(1,Math.ceil(available()*1000))),ops=target==='cpu'?{}:hypothesisTarget({root:resolve(directory),binding,resource,signal,stopSignal:cleanup,localExecution:config.local_execution},dependencies);
 try{
  if(target!=='cpu'&&!dependencies.verify){
   const entries=[];for(const task of frozen)for(const [arm,control,expectedExit]of [['reference',task.controls.reference,0],...task.controls.negatives.map((c,n)=>['negative-'+n,c,1])]){const key=task.id+'/0/'+arm;if(!report.results.some(r=>r.key===key))entries.push({key,cwd:task.id+'-'+arm,files:{...task.files,...control.files},checker:task.checker,args:['.'],expectedExit,requireDevice:true});}
   if(entries.length){
    const seconds=available()-30;if(seconds<=0){report.status='blocked';report.failureKind='elapsed';report.reason='Qualification elapsed allocation exhausted';return {status:'blocked',reason:report.reason,identity,receipts:report.results};}
    report.runningSample={key:'qualification/0/batch',owner:report.owner};await persist();const deadline=AbortSignal.timeout(Math.max(1,Math.ceil(seconds*1000))),work=signal?AbortSignal.any([signal,deadline]):deadline;let rows;
    try{rows=await ops.qualifyBatch({directory,entries,seconds,signal:work,sampleStopSignal:cleanup});}
    catch(error){const unknown=unknownExecution(error);if(unknown){report.runningSample.unresolvedExecution=unknown;throw Object.assign(error,{code:'RSI_EXECUTION_UNRESOLVED'});}delete report.runningSample;if(signal?.aborted)throw error;report.status='blocked';report.failureKind=error.closure?.closed===true?'infrastructure':'controls';report.reason=error.message;report.closure=error.closure;report.verification=error.verification;return {status:'blocked',reason:report.reason,identity,receipts:report.results};}
    delete report.runningSample;signal?.throwIfAborted();
    for(const row of rows){const entry=entries.find(e=>e.key===row.key),cwd=join(micro,'qualification','0-batch','work',entry.cwd);let verification=row;
     if(row.exitCode===125||entry.expectedExit===0&&row.correct)try{verification=await validateKernelReceipt(row,cwd,{requireLatency:false,requireDevice:entry.requireDevice});}catch(error){verification={...row,correct:null,error:error.message};}
     const passed=!deadline.aborted&&row.exitCode===entry.expectedExit&&verification.correct===(entry.expectedExit===0);report.results.push({key:row.key,expectedExit:entry.expectedExit,passed,sourceSha256:hash(entry.files),elapsedSeconds:row.elapsedSeconds,verification});await persist();
    }
    if(rows.length!==entries.length||report.results.some(r=>!r.passed)){report.status='blocked';report.failureKind=deadline.aborted||rows.length!==entries.length?'elapsed':qualificationFailure(report.results);report.reason='Reference or negative control did not qualify';return {status:'blocked',reason:report.reason,identity,receipts:report.results};}
   }
   report.status='qualified';return {status:'qualified',identity,receipts:report.results,elapsedSeconds:report.elapsedSeconds};
  }
  for(const task of frozen){const checker=join(micro,'frozen',task.id+'.py');await writeFile(checker,task.checker);
   for(const [arm,control,expectedExit]of [['reference',task.controls.reference,0],...task.controls.negatives.map((c,n)=>['negative-'+n,c,1])]){
    const key=task.id+'/0/'+arm;if(report.results.some(r=>r.key===key))continue;signal?.throwIfAborted();
    const seconds=Math.min(180,available()-30);if(seconds<=0){report.status='blocked';report.failureKind='elapsed';report.reason='Qualification elapsed allocation exhausted';return {status:'blocked',reason:report.reason,identity,receipts:report.results};}
    const cwd=join(micro,task.id,'0-'+arm,'work'),files={...task.files,...control.files};await mkdir(cwd,{recursive:true});
    for(const [name,text]of Object.entries(files)){const path=resolve(cwd,name),rel=relative(cwd,path);if(!rel||rel==='..'||rel.startsWith('../'))throw Error('Control files must remain inside their source');await mkdir(dirname(path),{recursive:true});await writeFile(path,text);}
    report.runningSample={key,owner:report.owner};await persist();const start=performance.now(),deadline=AbortSignal.timeout(Math.max(1,Math.ceil(seconds*1000))),work=signal?AbortSignal.any([signal,deadline]):deadline;let verification;
    try{
     verification=await (dependencies.verify??ops.verify??(args=>check(config.python??'python3',args.checker,args.cwd,args.seconds,args.signal)))({task,checker,cwd,seconds,signal:work,sampleSignal:work,sampleStopSignal:cleanup});
     if(hash(await readFile(checker,'utf8'))!==hash(task.checker))throw Error('Frozen qualification checker changed');
     for(const [name,text]of Object.entries(files))if(await readFile(join(cwd,name),'utf8')!==text)throw Error('Qualification changed a control source');
     if(target!=='cpu'&&expectedExit===0&&verification.correct)verification=await validateKernelReceipt(verification,cwd,{requireLatency:false,requireDevice:true});
    }catch(error){const unknown=unknownExecution(error);if(unknown){report.runningSample.unresolvedExecution=unknown;throw Object.assign(error,{code:'RSI_EXECUTION_UNRESOLVED'});}if(signal?.aborted){delete report.runningSample;throw error;}verification={correct:null,error:error.message};}
    delete report.runningSample;if(signal?.aborted)signal.throwIfAborted();
    const passed=!deadline.aborted&&verification.exitCode===expectedExit&&verification.correct===(expectedExit===0);
    report.results.push({key,expectedExit,passed,elapsedSeconds:(performance.now()-start)/1000,sourceSha256:hash(files),verification});await persist();
    if(!passed){report.status='blocked';report.failureKind=deadline.aborted||verification.exitCode===null?'elapsed':verification.exitCode===125&&verification.receipt?.bucket==='harness'?'infrastructure':'controls';report.reason=expectedExit===0?'Reference control failed':'Negative control was not independently rejected';return {status:'blocked',reason:report.reason,identity,receipts:report.results};}
   }
  }
  report.status='qualified';return {status:'qualified',identity,receipts:report.results,elapsedSeconds:report.elapsedSeconds};
 }finally{delete report.owner;await persist();}
}
