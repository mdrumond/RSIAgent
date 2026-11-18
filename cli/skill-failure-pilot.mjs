/** Isolated failure-first pilot. Adapters own Pi sessions, oracle execution and device custody. */
import {mkdir,readFile,writeFile,rename,open,unlink} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {processIdentity,ownerStopped} from '../controller/lifecycle.mjs';

const digest=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const load=async path=>{try{return JSON.parse(await readFile(path,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;return null;}};
const save=async(path,value)=>{const temporary=path+'.'+randomUUID()+'.tmp';await writeFile(temporary,JSON.stringify(value,null,2)+'\n');await rename(temporary,path);};
const slug=value=>typeof value==='string'&&/^[a-zA-Z0-9_-]+$/.test(value);
const nonempty=value=>typeof value==='string'&&value.trim();
const positive=value=>Number.isFinite(value)&&value>0;
const clone=value=>structuredClone(value);
const publicFamily=family=>({id:family.id,development:clone(family.development),evidence:clone(family.evidence??[])});
export function validatePilotManifest(manifest){
 if(!nonempty(manifest?.baseline)||!manifest.identities||!['model','runtime','memory'].every(k=>manifest.identities[k]!==undefined)||!Array.isArray(manifest.families)||!manifest.families.length||manifest.families.length>3)throw Error('Frozen baseline, model/runtime/memory identities and one to three families required');
 const ids=new Set();
 for(const family of manifest.families){
  if(!slug(family.id)||ids.has(family.id)||!slug(family.development?.id)||family.transfer?.length!==2||family.transfer.some(task=>!slug(task.id))||new Set([family.development.id,...family.transfer.map(t=>t.id)]).size!==3)throw Error('Unique family and development/two transfer task IDs required');
  for(const task of [family.development,...family.transfer])if(!Array.isArray(task.caseIds)||!task.caseIds.length||task.caseIds.some(id=>!nonempty(id))||new Set(task.caseIds).size!==task.caseIds.length||task.controlCaseIds?.some(id=>!task.caseIds.includes(id)))throw Error('Frozen task case IDs and optional control IDs required');
  ids.add(family.id);
 }
 return manifest;
}
export function validatePilotTrial(result,task){
 if(result?.infrastructure===true){if(result.correct!==null||!nonempty(result.reason))throw Error('Infrastructure results require correct:null and a reason');return result;}
 if(typeof result?.correct!=='boolean'||typeof result.mechanismMatch!=='boolean'||!Array.isArray(result.caseResults)||result.caseResults.some(c=>!nonempty(c.id)||typeof c.correct!=='boolean')||new Set(result.caseResults.map(c=>c.id)).size!==result.caseResults.length||! /^[a-f0-9]{64}$/.test(result.sourceSha256??'')||!Array.isArray(result.artifacts)||!Array.isArray(result.handles))throw Error('Trial requires trusted correctness, mechanism match, case outcomes, source hash, artifacts and handles');
 if(result.correct&&(!result.caseResults.length||result.caseResults.some(c=>!c.correct)))throw Error('Successful trial must pass every reported case');
 if(task){if(result.caseResults.some(c=>!task.caseIds.includes(c.id))||result.correct&&result.caseResults.length!==task.caseIds.length)throw Error('Trial case coverage differs from the frozen task');result={...result,caseResults:result.caseResults.map(c=>({...c,control:(task.controlCaseIds??task.caseIds).includes(c.id)}))};}
 return result;
}
function candidates(value){
 const rows=Array.isArray(value)?value:value?.candidates;
 if(rows?.length!==3||rows.some(c=>!slug(c.id)||c.id==='baseline'||!nonempty(c.text)||!Array.isArray(c.evidence)||!c.evidence.length||c.evidence.some(p=>!nonempty(p.reference)))||new Set(rows.map(c=>c.id)).size!==3||new Set(rows.map(c=>c.text)).size!==3)throw Error('Three distinct candidate IDs/texts with evidence selectors required');
 return rows;
}
const passes=rows=>rows.filter(r=>r.correct===true).length;
function controlRegression(baseline,candidate){
 const ids=new Set(baseline.flatMap(r=>r.caseResults??[]).filter(c=>c.control).map(c=>c.id));
 return [...ids].some(id=>baseline.every(r=>r.caseResults.some(c=>c.id===id&&c.correct))&&!candidate.every(r=>r.caseResults.some(c=>c.id===id&&c.correct)));
}
export function pilotWinner(results,rows){
 const baseline=results.baseline;
 return rows.filter(c=>passes(results[c.id])>=passes(baseline)+2&&!controlRegression(baseline,results[c.id])).sort((a,b)=>passes(results[b.id])-passes(results[a.id])||a.text.length-b.text.length||a.id.localeCompare(b.id))[0]??null;
}
/** Five repetitions are a frozen workload, never a renewed elapsed allowance. */
export async function runFailurePilot({directory,manifest,seconds=7200,trialSeconds=180,concurrency=4,signal},dependencies){
 manifest=clone(validatePilotManifest(manifest));
 if(!positive(seconds)||!positive(trialSeconds)||!Number.isInteger(concurrency)||concurrency<1||concurrency>4||!['qualify','trial','design'].every(k=>typeof dependencies?.[k]==='function'))throw Error('Positive elapsed allocations, concurrency1..4 and trusted adapters required');
 await mkdir(directory,{recursive:true});
 const owner=await processIdentity(),lock=join(directory,'pilot.lock');let handle;
 try{handle=await open(lock,'wx');}catch(error){
  if(error.code!=='EEXIST')throw error;
  let previous;try{previous=await load(lock);}catch(error){if(!(error instanceof SyntaxError))throw error;return {status:'needs-settlement',reason:'Pilot owner record is incomplete'};}
  if(!previous||!await ownerStopped(previous))return {status:'needs-settlement',reason:'Pilot owner remains live or unknown'};
  await unlink(lock);handle=await open(lock,'wx');
 }
 await handle.writeFile(JSON.stringify(owner));await handle.close();
 const now=dependencies.now??(()=>performance.now()),wall=dependencies.wallNow??Date.now,file=join(directory,'pilot.json');let state,timer,readOnly=false,finalised=false,writes=Promise.resolve(),last=now();
 const identity=digest({manifest,seconds,trialSeconds,concurrency}),end=new AbortController(),active=signal?AbortSignal.any([signal,end.signal]):end.signal;
 const working=new Map();
 const charge=()=>{const current=now();state.elapsedSeconds+=Math.max(0,current-last)/1000;last=current;state.runningAt=wall();for(const [id,start]of working){const op=state.operations[id];op.elapsedSeconds=start.prior+Math.max(0,current-start.at)/1000;op.runningAt=wall();}};
 const persist=()=>{charge();const snapshot=clone(state);writes=writes.then(()=>save(file,snapshot));return writes;};
 const remaining=()=>Math.max(0,seconds-state.elapsedSeconds-Math.max(0,now()-last)/1000);
 const expired=()=>{active.throwIfAborted();if(remaining()<=0)throw Error('Pilot elapsed allocation exhausted');};
 const result=async(status,reason,extra={})=>{state.status=status;state.reason=reason;Object.assign(state,extra);await persist();if(!Object.values(state.operations).some(op=>op.status==='running'))delete state.runningAt;await writes;await save(file,state);finalised=true;return clone(state);};
 try{
  state=await load(file);
  if(state&&state.identity!==identity){readOnly=true;throw Error('Frozen pilot manifest or allocations changed');}
  if(state?.status==='complete'){readOnly=true;return clone(state);}
  if(!state){state={format:1,identity,status:'running',elapsedSeconds:0,operations:{},qualifications:{},screens:{},candidates:null,winner:null,development:{},confirmation:{}};await save(join(directory,'manifest.json'),manifest);}
  else if(state.runningAt!==undefined)state.elapsedSeconds+=Math.max(0,wall()-state.runningAt)/1000;
  last=now();
  for(const op of Object.values(state.operations).filter(op=>op.status==='running')){
   const settled=dependencies.settle&&await dependencies.settle(clone(op));
   if(!settled?.closed)return await result('needs-settlement','Interrupted operation custody remains unresolved',{unresolvedOperation:op.id});
   op.elapsedSeconds+=Math.max(0,wall()-op.runningAt)/1000;
   if(settled.result===undefined)return await result('inconclusive','Settled operation has no recoverable result',{unresolvedOperation:op.id});
   if(op.elapsedSeconds>op.allocatedSeconds)return await result('inconclusive','Interrupted operation exceeded its retained elapsed allowance',{unresolvedOperation:op.id});
   op.result=clone(settled.result);op.status='complete';delete op.runningAt;
  }
  delete state.unresolvedOperation;state.status='running';await persist();
  timer=setInterval(()=>{void persist().catch(error=>end.abort(error));if(remaining()<=0)end.abort(Error('Pilot elapsed allocation exhausted'));},1000);
  const invoke=async(id,kind,input,fn,cap=remaining())=>{
   const prior=state.operations[id];if(prior?.status==='complete')return clone(prior.result);
   expired();const available=Math.min(cap,remaining()),op={id,kind,status:'running',directory:join(directory,'operations',id),allocatedSeconds:available,elapsedSeconds:0,runningAt:wall()};
   state.operations[id]=op;working.set(id,{at:now(),prior:0});await mkdir(op.directory,{recursive:true});await persist();
   const timeout=AbortSignal.timeout(Math.max(1,Math.ceil(available*1000))),bounded=AbortSignal.any([active,timeout]);
   try{const answer=await fn({...clone(input),directory:op.directory,remainingSeconds:available,signal:bounded});
    if(bounded.aborted||(now()-working.get(id).at)/1000>available||remaining()<=0){op.result={infrastructure:true,correct:null,reason:'Operation elapsed allocation exhausted before result admission'};op.status='complete';throw Error('Operation elapsed allocation exhausted');}
    op.result=clone(answer);op.status='complete';return clone(answer);}
   finally{const start=working.get(id);op.elapsedSeconds=start.prior+Math.max(0,now()-start.at)/1000;working.delete(id);if(op.status==='complete')delete op.runningAt;await persist();}
  };
  const trials=async(phase,family,task,arms)=>{
   const cells=Array.from({length:5},(_,repetition)=>arms.map((_,offset)=>arms[(offset+repetition)%arms.length]).map(arm=>({arm,repetition}))).flat();
   const outputs=Object.fromEntries(arms.map(a=>[a.id,[]]));let cursor=0,failure;
   const workers=Array.from({length:Math.min(concurrency,cells.length)},async()=>{
    while(!failure&&cursor<cells.length){const {arm,repetition}=cells[cursor++],id=[phase,family.id,task.id,arm.id,repetition].join('--');
     try{const answer=validatePilotTrial(await invoke(id,'trial',{phase,familyId:family.id,task,arm:arm.id,guidance:arm.text,repetition,identities:manifest.identities},dependencies.trial,trialSeconds),task);outputs[arm.id][repetition]=answer;}
     catch(error){failure=error;end.abort(error);}
    }
   });await Promise.all(workers);if(failure)throw failure;
   return outputs;
  };
  const baseline={id:'baseline',text:manifest.baseline};let selected;
  for(const family of manifest.families){
   const qualified=await invoke('qualify--'+family.id,'qualify',{family},dependencies.qualify);state.qualifications[family.id]=qualified;await persist();
   if(qualified.qualified!==true||qualified.referencePassed!==true||qualified.negativeFailed!==true)continue;
   const screen=await trials('screen',family,family.development,[baseline]);state.screens[family.id]=screen.baseline;await persist();
   if(screen.baseline.some(r=>r.infrastructure))return await result('inconclusive','Baseline infrastructure outcome does not establish struggle');
   if(screen.baseline.filter(r=>!r.correct&&r.mechanismMatch).length>=3){selected=family;break;}
  }
  if(!selected)return await result('complete','No qualified family demonstrated baseline struggle');
  state.selectedFamily=selected.id;
  state.candidates=candidates(await invoke('design--'+selected.id,'design',{family:publicFamily(selected),baselineResults:state.screens[selected.id]},dependencies.design));await persist();
  const arms=[baseline,...state.candidates];state.development=await trials('development',selected,selected.development,arms);await persist();
  if(Object.values(state.development).flat().some(r=>r.infrastructure))return await result('inconclusive','Development contains infrastructure outcomes');
  const winner=pilotWinner(state.development,state.candidates);if(!winner)return await result('complete','No development candidate improved success without control regression');
  if(state.winner&&digest(state.winner)!==digest(winner))throw Error('Frozen winner changed');state.winner=clone(winner);await save(join(directory,'winner.json'),winner);await persist();
  for(const task of selected.transfer){state.confirmation[task.id]=await trials('confirmation',selected,task,[baseline,winner]);await persist();}
  if(Object.values(state.confirmation).flatMap(x=>Object.values(x).flat()).some(r=>r.infrastructure))return await result('inconclusive','Confirmation contains infrastructure outcomes');
  const promising=Object.values(state.confirmation).every(rows=>passes(rows[winner.id])>=passes(rows.baseline)+2&&!controlRegression(rows.baseline,rows[winner.id]));
  return await result('complete',promising?'Transfer improvement demonstrated':'No demonstrated transfer improvement',{promising});
 }catch(error){
  if(!state||readOnly)throw error;
  return await result(Object.values(state.operations).some(op=>op.status==='running')?'needs-settlement':'inconclusive',error.message);
 }finally{
  clearInterval(timer);
  if(state&&!readOnly&&!finalised){charge();if(!Object.values(state.operations).some(op=>op.status==='running'))delete state.runningAt;await writes;await save(file,state);}
  const previous=await load(lock);if(previous?.pid===owner.pid&&previous.start===owner.start&&previous.boot===owner.boot)await unlink(lock);
 }
}
