/** Isolated failure-first pilot. Adapters own Pi sessions, oracle execution and device custody. */
import assert from 'node:assert/strict';
import {mkdir,readFile,writeFile,rename,open,unlink,link,realpath} from 'node:fs/promises';
import {isAbsolute,join,relative,resolve} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {processIdentity,ownerStopped} from '../controller/lifecycle.mjs';
import {replayFailureCohort} from './skill-failure-replay.mjs';

const sourceHash=value=>createHash('sha256').update(value).digest('hex');
const digest=value=>sourceHash(JSON.stringify(value));
const isHash=value=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
const load=async path=>{try{return JSON.parse(await readFile(path,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;return null;}};
const save=async(path,value)=>{const temporary=path+'.'+randomUUID()+'.tmp';await writeFile(temporary,JSON.stringify(value,null,2)+'\n');await rename(temporary,path);};
const slug=value=>typeof value==='string'&&/^[a-zA-Z0-9_-]+$/.test(value)&&!value.includes('--')&&!value.startsWith('-')&&!value.endsWith('-');
const nonempty=value=>typeof value==='string'&&value.trim();
const positive=value=>Number.isFinite(value)&&value>0;
const clone=value=>structuredClone(value);
const publicFamily=family=>({id:family.id,development:clone(family.development),evidence:clone(family.evidence??[])});
export function validatePilotManifest(manifest){
 if(!nonempty(manifest?.baseline)||!manifest.identities||!['model','runtime','memory'].every(k=>manifest.identities[k]!==undefined)||!Array.isArray(manifest.families)||!manifest.families.length||manifest.families.length>3)throw Error('Frozen baseline, model/runtime/memory identities and one to three families required');
 if(manifest.mode!==undefined&&manifest.mode!=='focused-correctness')throw Error('Unknown pilot mode');
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
 if(result?.correct===null&&nonempty(result.reason))return {...result,mechanismMatch:false};
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
export function focusedBaselineDecision(rows){
 if(rows.length!==5||rows.some(row=>row.infrastructure))return 'incomplete';
 if(passes(rows)>3)return 'baseline-ceiling';
 if(!rows.some(row=>row.correct===false&&row.mechanismMatch))return 'no-exact-recurrence';
 return 'admitted';
}
function controlRegression(baseline,candidate){
 const ids=new Set(baseline.flatMap(r=>r.caseResults??[]).filter(c=>c.control).map(c=>c.id));
 return [...ids].some(id=>baseline.every(r=>r.caseResults?.some(c=>c.id===id&&c.correct))&&!candidate.every(r=>r.caseResults?.some(c=>c.id===id&&c.correct)));
}
export function pilotWinner(results,rows){
 const baseline=results.baseline;
 return rows.filter(c=>passes(results[c.id])>=passes(baseline)+2&&!controlRegression(baseline,results[c.id])).sort((a,b)=>passes(results[b.id])-passes(results[a.id])||a.text.length-b.text.length||a.id.localeCompare(b.id))[0]??null;
}
/** Five repetitions are a frozen workload, never a renewed elapsed allowance.
 * Trial adapters enforce actorSeconds separately; remainingSeconds covers actor plus private verification. */
export async function runFailurePilot({directory,manifest,seconds=7200,trialSeconds=180,verificationSeconds=0,concurrency=4,initialElapsedSeconds=0,signal},dependencies){
 manifest=clone(validatePilotManifest(manifest));
 if(!positive(seconds)||!positive(trialSeconds)||!Number.isFinite(verificationSeconds)||verificationSeconds<0||!Number.isFinite(initialElapsedSeconds)||initialElapsedSeconds<0||initialElapsedSeconds>seconds||!Number.isInteger(concurrency)||concurrency<1||concurrency>4||!['qualify','trial','design'].every(k=>typeof dependencies?.[k]==='function'))throw Error('Positive elapsed allocations, concurrency1..4 and trusted adapters required');
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
 const manifestSha256=digest(manifest);
 const focused=manifest.mode==='focused-correctness';
 const identity=digest({manifest,seconds,trialSeconds,verificationSeconds,concurrency,initialElapsedSeconds}),end=new AbortController(),active=signal?AbortSignal.any([signal,end.signal]):end.signal;
 const working=new Map();
 const charge=()=>{const current=now();state.elapsedSeconds+=Math.max(0,current-last)/1000;last=current;state.runningAt=wall();for(const [id,start]of working){const op=state.operations[id];op.elapsedSeconds=start.prior+Math.max(0,current-start.at)/1000;op.runningAt=wall();}};
 const persist=()=>{charge();const snapshot=clone(state);writes=writes.then(()=>save(file,snapshot));return writes;};
 const remaining=()=>Math.max(0,seconds-state.elapsedSeconds-Math.max(0,now()-last)/1000);
 const expired=()=>{active.throwIfAborted();if(remaining()<=0)throw Error('Pilot elapsed allocation exhausted');};
 const result=async(status,reason,extra={})=>{state.status=status;state.reason=reason;Object.assign(state,extra);await persist();if(!Object.values(state.operations).some(op=>op.status==='running'))delete state.runningAt;await writes;await save(file,state);finalised=true;return clone(state);};
 try{
  state=await load(file);
  if(state&&state.identity!==identity){readOnly=true;throw Error('Frozen pilot manifest or allocations changed');}
  if(focused&&state?.manifestSha256!==undefined&&state.manifestSha256!==manifestSha256){
   readOnly=true;throw Error('Frozen focused pilot manifest changed');
  }
  if(state?.status==='complete'){
   readOnly=true;
   if(focused&&state.manifestSha256===undefined){
    state.manifestSha256=manifestSha256;await save(file,state);
   }
   return clone(state);
  }
  if(!state){state={format:1,identity,status:'running',promising:false,elapsedSeconds:initialElapsedSeconds,operations:{},qualifications:{},screens:{},candidates:null,winner:null,development:{},confirmation:{}};await save(join(directory,'manifest.json'),manifest);}
  else if(state.runningAt!==undefined)state.elapsedSeconds+=Math.max(0,wall()-state.runningAt)/1000;
  if(focused)state.manifestSha256=manifestSha256;
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
   expired();if(kind==='trial'&&remaining()<cap)throw Object.assign(Error('Insufficient pilot elapsed time for a full actor and verification allocation'),{code:'PILOT_ADMISSION_BUDGET'});
   const available=kind==='trial'?cap:Math.min(cap,remaining()),op={id,kind,status:'running',directory:join(directory,'operations',id),allocatedSeconds:available,...(kind==='trial'?{actorSeconds:input.actorSeconds,verificationSeconds:input.verificationSeconds}:{}),elapsedSeconds:0,runningAt:wall()};
   state.operations[id]=op;working.set(id,{at:now(),prior:0});await mkdir(op.directory,{recursive:true});await persist();
   const timeout=AbortSignal.timeout(Math.max(1,Math.ceil(available*1000))),bounded=AbortSignal.any([active,timeout]);
   try{const answer=await fn({...clone(input),directory:op.directory,remainingSeconds:available,signal:bounded});
    if(bounded.aborted||(now()-working.get(id).at)/1000>available||remaining()<=0){op.result={infrastructure:true,correct:null,reason:'Operation elapsed allocation exhausted before result admission'};op.status='complete';throw Error('Operation elapsed allocation exhausted');}
    op.result=clone(answer);op.status='complete';return clone(answer);}
   finally{const start=working.get(id);op.elapsedSeconds=start.prior+Math.max(0,now()-start.at)/1000;working.delete(id);if(op.status==='complete')delete op.runningAt;else op.runningAt=wall();await persist();}
  };
  const trials=async(phase,family,task,arms)=>{
   const cells=Array.from({length:5},(_,repetition)=>arms.map((_,offset)=>arms[(offset+repetition)%arms.length]).map(arm=>({arm,repetition}))).flat();
   const outputs=Object.fromEntries(arms.map(a=>[a.id,[]]));let cursor=0,failure;
   const workers=Array.from({length:Math.min(concurrency,cells.length)},async()=>{
    while(!failure&&cursor<cells.length){const {arm,repetition}=cells[cursor++],id=[phase,family.id,task.id,arm.id,repetition].join('--');
     try{const answer=validatePilotTrial(await invoke(id,'trial',{phase,familyId:family.id,task,arm:arm.id,guidance:arm.text,repetition,identities:manifest.identities,...(manifest.mode?{mode:manifest.mode}:{}),actorSeconds:trialSeconds,verificationSeconds},dependencies.trial,trialSeconds+verificationSeconds),task);outputs[arm.id][repetition]=answer;
      if(answer.infrastructure)throw Object.assign(Error(answer.reason),{code:'PILOT_INFRASTRUCTURE'});}
     catch(error){failure??=error;if(error.code!=='PILOT_ADMISSION_BUDGET')end.abort(failure);}
    }
   });await Promise.all(workers);if(failure)throw failure;
   return outputs;
  };
  const qualify=async(family,task)=>{
   const key=focused?family.id+'--'+task.id:family.id;
   if(focused&&nonempty(task.checker)){
    const checkerSha256=sourceHash(await readFile(task.checker));
    state.checkerHashes??={};
    assert.equal(state.checkerHashes[key]??checkerSha256,checkerSha256,'Frozen checker changed');
    if(!state.checkerHashes[key]){state.checkerHashes[key]=checkerSha256;await persist();}
   }
   const verdict=await invoke('qualify--'+key,'qualify',focused?{family,task}:{family},dependencies.qualify);
   state.qualifications[key]=verdict;await persist();
   return verdict.qualified===true&&verdict.referencePassed===true&&verdict.negativeFailed===true&&(!focused||verdict.calibrationPassed===true);
  };
  const qualifyTransfers=async family=>{
   const qualified=[];
   let next=0,failure;
   // Qualification gets an exclusive batch of the same device pool. The
   // trusted adapter owns slots and returns each only after positive closure.
   await Promise.all(Array.from({length:Math.min(concurrency,family.transfer.length)},async()=>{
    while(!failure&&next<family.transfer.length){
     const index=next++;
     try{qualified[index]=await qualify(family,family.transfer[index]);}
     catch(error){failure??=error;end.abort(failure);}
    }
   }));
   if(failure)throw failure;
   return qualified.every(Boolean);
  };
  if(focused)state.screenDecisions??={};
  const baseline={id:'baseline',text:manifest.baseline};let selected;
  for(const family of manifest.families){
   if(!await qualify(family,family.development))continue;
   const screen=await trials('screen',family,family.development,[baseline]);state.screens[family.id]=screen.baseline;await persist();
   if(screen.baseline.some(r=>r.infrastructure))return await result('inconclusive','Baseline infrastructure outcome does not establish struggle');
   if(focused){
    state.screenDecisions[family.id]=focusedBaselineDecision(screen.baseline);await persist();
    if(state.screenDecisions[family.id]==='admitted'){selected=family;break;}
    continue;
   }
   if(screen.baseline.filter(r=>!r.correct&&r.mechanismMatch).length>=3){selected=family;break;}
  }
  if(!selected)return await result('complete','No qualified family demonstrated baseline struggle');
  state.selectedFamily=selected.id;
  const designInput={family:publicFamily(selected),baselineResults:state.screens[selected.id],baseline:manifest.baseline,identities:manifest.identities,
   ...(focused?{mode:manifest.mode,qualification:state.qualifications[selected.id+'--'+selected.development.id]}:{})};
  let designed=await invoke('design--'+selected.id,'design',designInput,dependencies.design);
  if(focused&&designed?.status==='rejected'&&nonempty(designed.revisionFeedback)){
   designed=await invoke('redesign--'+selected.id,'design',{...designInput,revisionFeedback:designed.revisionFeedback},dependencies.design);
  }
  if(designed?.status==='rejected'&&nonempty(designed.reason))return await result('complete',designed.reason,{designRejected:true});
  state.candidates=candidates(designed);await persist();
  const arms=focused?state.candidates:[baseline,...state.candidates];
  state.development=await trials('development',selected,selected.development,arms);
  if(focused)state.development={baseline:state.screens[selected.id],...state.development};
  await persist();
  if(Object.values(state.development).flat().some(r=>r.infrastructure))return await result('inconclusive','Development contains infrastructure outcomes');
  const winner=pilotWinner(state.development,state.candidates);if(!winner)return await result('complete','No development candidate improved success without control regression');
  if(state.winner&&digest(state.winner)!==digest(winner))throw Error('Frozen winner changed');state.winner=clone(winner);await save(join(directory,'winner.json'),winner);await persist();
  if(focused&&!await qualifyTransfers(selected))return await result('complete','Transfer controls did not qualify');
  for(const task of selected.transfer){
   if(focused){
    const rows=await trials('transfer-screen',selected,task,[baseline]);
    state.confirmation[task.id]=rows;await persist();
    const decision=focusedBaselineDecision(rows.baseline);
    if(decision!=='admitted')return await result('complete','Transfer baseline: '+decision);
    Object.assign(rows,await trials('confirmation',selected,task,[winner]));
   }else state.confirmation[task.id]=await trials('confirmation',selected,task,[baseline,winner]);
   await persist();
  }
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

/** Audit exact retained sources with a replacement checker. A new promotion
 * decision still requires fresh confirmation with the replacement frozen. */
export async function replayRetainedFailureTask({directory,taskId,candidateId,checkerSha256},{score}){
 assert(slug(taskId)&&isHash(checkerSha256)&&typeof score==='function','Valid task, checker hash and trusted replay scorer required');
 directory=resolve(directory);
 const [pilot,manifest]=await Promise.all([load(join(directory,'pilot.json')),load(join(directory,'manifest.json'))]);
 assert(pilot?.status==='complete'&&manifest?.mode==='focused-correctness'&&
  Object.values(pilot.operations).every(op=>op.status==='complete'),'Completed focused pilot with settled execution required');
 assert.equal(pilot.manifestSha256,digest(manifest),'Frozen manifest identity changed');
 const family=manifest.families.find(row=>row.id===pilot.selectedFamily);
 assert(family,'Selected failure family unavailable');
 const development=family.development.id===taskId;
 const task=development?family.development:family.transfer.find(row=>row.id===taskId);
 assert(task,'Unknown replay task');
 const candidate=candidateId??pilot.winner?.id;
 assert(development?pilot.candidates?.some(row=>row.id===candidate):candidate&&candidate===pilot.winner?.id,
  'Choose a retained candidate from the completed comparison');
 const key=[family.id,task.id].join('--');
 const qualified=pilot.qualifications[key],oldChecker=pilot.checkerHashes?.[key];
 assert(qualified?.qualified===true&&qualified.referencePassed===true&&qualified.negativeFailed===true&&
  qualified.calibrationPassed===true&&isHash(oldChecker)&&oldChecker!==checkerSha256,'Frozen qualified controls and a distinct checker required');
 const result=development?pilot.development:pilot.confirmation[task.id];
 const sources=new Map(),records=[];
 const control=(arm,id,files,outcome)=>{
  sources.set(id,{files});records.push({id,arm,files:Object.fromEntries(Object.entries(files).map(([name,text])=>[name,sourceHash(text)])),verdict:{outcome}});
 };
 control('positive','positive',task.controls.reference.files,'pass');
 task.controls.negatives.forEach(({files},index)=>control('negative','negative-'+index,files,'exact-failure'));
 control('calibration','calibration',task.controls.calibration.files,'pass');
 for(const [arm,actor,phase]of [['baseline','baseline',development?'screen':'transfer-screen'],
  ['candidate',candidate,development?'development':'confirmation']]){
  const rows=result?.[actor];
  assert.equal(rows?.length,5,'Complete saved arms required for symmetric replay');
  for(const [repetition,row]of rows.entries()){
   const id=[phase,family.id,taskId,actor,repetition].join('--');
   const operation=pilot.operations[id];
   const retained=await load(join(directory,'operations',id,'source-manifest.json'));
   assert(operation?.result&&!row.infrastructure&&isHash(row.sourceSha256)&&retained?.cwd&&retained.files,
    'Every replay row requires retained trusted actor bytes');
   assert.deepEqual(validatePilotTrial(operation.result,task),row,'Historical verdict changed');
   assert.equal(digest(retained.files),row.sourceSha256,'Historical actor file identity changed');
   sources.set(id,{cwd:retained.cwd});
   const outcome=row.correct===true?'pass':row.correct===null?'unscored':row.mechanismMatch?'exact-failure':'other-failure';
   records.push({id,arm,files:retained.files,verdict:{outcome,result:row}});
  }
 }
 const identity={fixture:digest({family:family.id,task}),oracle:digest({cases:task.caseIds,controls:task.controlCaseIds,helpers:task.checkerFiles}),
  runtime:manifest.identities.runtime,skills:manifest.identities.skills??digest(manifest.baseline),checker:oldChecker};
 const audit=await replayFailureCohort({original:{identity,records},correctedIdentity:{...identity,checker:checkerSha256}},{score,
  readSource:async ({id},name)=>{
   const source=sources.get(id);
   if(!source.cwd)return Buffer.from(source.files[name]);
   const cwd=await realpath(source.cwd),requested=resolve(cwd,name);
   const within=file=>{const rel=relative(cwd,file);return rel&&rel!=='..'&&!rel.startsWith('../')&&!isAbsolute(rel);};
   assert(!isAbsolute(name)&&within(requested),'Retained replay path escaped actor directory');
   const file=await realpath(requested);
   assert(within(file),'Retained replay link escaped actor directory');
   return readFile(file);
  }});
 const audits=join(directory,'checker-replays');
 await mkdir(audits,{recursive:true});
 const path=join(audits,taskId+'--'+candidate+'--'+checkerSha256+'.json'),temporary=path+'.'+randomUUID()+'.tmp';
 await writeFile(temporary,JSON.stringify(audit,null,2)+'\n',{mode:0o600});
 try{await link(temporary,path);}
 catch(error){if(error.code!=='EEXIST')throw error;assert.deepEqual(await load(path),audit,'Existing checker replay verdicts changed');}
 finally{await unlink(temporary);}
 return {...audit,path};
}
