import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir,rename,readdir,realpath} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash,randomUUID} from 'node:crypto';
import {spawn,spawnSync} from 'node:child_process';

const hash=x=>createHash('sha256').update(x).digest('hex');
const text=p=>readFile(p,'utf8');
const get=async p=>{try{return JSON.parse(await text(p));}catch(e){if(e.code!=='ENOENT')throw e;return null;}};
const save=async(p,data)=>{const tmp=p+'.'+randomUUID();await writeFile(tmp,JSON.stringify(data,null,2));await rename(tmp,p);};
const schema=names=>({type:'object',properties:Object.fromEntries(names.map(k=>[k,{type:'string'}])),required:names,additionalProperties:false});
const designFields=['prompt','starter','good','bad','checker','diagnosticRef','checkId'];
const choice=(decision,names)=>{const spec=schema(['decision','reason',...names]);spec.properties.decision={const:decision,type:'string'};return spec;};
const designSchema={anyOf:[choice('test',designFields),choice('probe',['code']),choice('withhold',['missing','basis'])]};
const shape=fileURLToPath(new URL('./fixture-native-shape.py',import.meta.url));
const route=fileURLToPath(new URL('./fixture-native-route.py',import.meta.url));
const pinnedDesigner=new URL('./skills/fixture-designer/SKILL.md',import.meta.url);
function validReason(reason){return typeof reason==='string'&&reason.trim()&&!/^(placeholder|todo|n\/a|unknown)[.!]?$/i.test(reason.trim());}
async function sdkSourceIdentity(root){
 const entries=[];
 async function visit(dir){
  const files=await readdir(dir,{withFileTypes:true});
  for(const file of files.sort((a,b)=>a.name.localeCompare(b.name))){
   const path=join(dir,file.name);
   if(file.isDirectory()&&file.name!=='__pycache__')await visit(path);
   else if(file.isFile()&&/\.pyi?$/.test(file.name))entries.push([path.slice(root.length+1),hash(await readFile(path))]);
  }
 }
 await visit(root);assert(entries.length,'SDK contains no Python sources');
 return hash(JSON.stringify([await realpath(root),entries]));
}
function diagnosticLine(line){
 const value=line.trim();
 return value&&!/^(Traceback|File ["']|During handling|The above exception|raise |assert )/.test(value)&&
  !/^(?:[\w.]+(?:Error|Exception)|Error|Exception|error|reason|FAILED)[:\s]*$/i.test(value);
}
function causeLines(output){return output.split(/\r?\n/).filter(line=>diagnosticLine(line)&&/^\s*(?:[\w.]+(?:Error|Exception)|error|reason|fatal|AssertionError|failed)\s*:/i.test(line));}
const fields=['public-task','failing-source','observed-output','inherited-guidance'];

export function publicPacket(input){
 const visible={};
 for(const name of fields){
  const matches=input.records.filter(r=>r.reference===name);
  assert(matches.length===1&&typeof matches[0].text==='string','expected one public '+name+' record');visible[name]=matches[0].text;
 }
 const operation=input.operation;
 assert(typeof operation==='string'&&operation.trim(),'bounded operation is required');
 const originalContract=input.originalContract??input.publicTask;
 assert(typeof originalContract==='string'&&originalContract.trim(),'original project contract is required');
 assert.equal(originalContract,visible['public-task'],'original contract differs from archived public-task');
 if(input.publicTask!==undefined)assert.equal(input.publicTask,originalContract,'publicTask and originalContract disagree');
 const diagnosticOptions=visible['observed-output'].split(/\r?\n/).flatMap((line,i)=>causeLines(line).length?[{id:'observed-output:L'+(i+1),text:line.trim()}]:[]);
 return {...visible,operation,originalContract,diagnosticOptions};
}
const roles=['starter','good','bad'];
const sourceKey=v=>v.replaceAll('\r\n','\n').trim();
const material=d=>JSON.stringify(designFields.map(k=>sourceKey(d[k])));
function expectedCause(d,packet){
 const ref=packet.diagnosticOptions.find(r=>r.id===d.diagnosticRef);
 assert(ref,'select an observed diagnostic reference: '+packet.diagnosticOptions.map(r=>r.id+' '+r.text).join(' | '));
 return ref.text;
}
export function validateDesign(d,packet,feedback){
 assert(['test','probe','withhold'].includes(d.decision),'submit test, probe or withhold');
 assert(validReason(d.reason),'give the concrete reason for your decision');
 if(d.decision==='withhold'){
  assert(d.missing?.trim()&&d.missing!==d.reason,'name the missing fact or capability');
  const bases=['public-task','failing-source','observed-output',...packet.diagnosticOptions.map(r=>r.id),...(feedback?.measuredControls?roles:[]),...(feedback?.probe?['probe']:[])];
  assert(bases.includes(d.basis),'choose an available evidence basis: '+bases.join(', '));return d;
 }
 if(d.decision==='probe'){assert(d.code?.trim(),'supply an executable development probe');const syntax=spawnSync('python3',['-c','import ast,sys; ast.parse(sys.stdin.read())'],{input:d.code,encoding:'utf8'});if(syntax.error)throw syntax.error;assert(syntax.status===0,syntax.stderr);assert(d.code!==feedback?.probe?.code,'the same probe already returned; use its measurement or change the experiment');return d;}
 for(const key of designFields)assert(typeof d[key]==='string'&&d[key].trim(),'missing '+key);
 const diagnostic=expectedCause(d,packet);
 assert(/^[A-Za-z][A-Za-z0-9_-]*$/.test(d.checkId),'select one stable numerical check ID, for example output-values');
 assert(new Set(roles.map(role=>sourceKey(d[role]))).size===roles.length,'starter, good and bad must differ; the same implementation cannot demonstrate the required contrasts');
 if(feedback?.previousFixture)assert(material(d)!==material(feedback.previousFixture),'unchanged fixture; revise the affected artifact or withhold with the missing evidence');
 const p=spawnSync('python3',[shape],{input:JSON.stringify([{...d,diagnostic:diagnostic.replace(/^.*?:\s*/, '')},[packet['failing-source']]]),encoding:'utf8'});
 if(p.error)throw p.error;if(p.status!==0)throw Error(p.stderr.trim());
 return d;
}
const verified=/^VERIFIED_NATIVE_ROUTE [1-9]\d* [1-9]\d*\s*$/m;
export function assessControls(design,runs,packet){
 const diagnostic=expectedCause(design,packet);
 return Object.fromEntries(roles.map(role=>{
  const {exit,output}=runs[role],routeVerified=verified.test(output),causes=causeLines(output);
  const report=output.split(/\r?\n/).findLast(l=>l.startsWith('VERIFIED_CHECK_FAILURE '));
  let numerical=null;
  if(report){
   try{const hit=JSON.parse(report.slice('VERIFIED_CHECK_FAILURE '.length));if(hit.id===design.checkId&&typeof hit.details==='string'&&hit.details.trim())numerical=hit;}catch{/* Raw diagnostics remain available to the author. */}
  }
  const expected=role==='starter'?{exit:'nonzero',diagnostic}:role==='good'?{exit:0,routeVerified:true}:{exit:'nonzero',routeVerified:true,checkId:design.checkId};
  const matched=role==='starter'?exit!==0&&causes.some(l=>l.trim()===diagnostic):role==='good'?exit===0&&routeVerified:exit!==0&&routeVerified&&Boolean(numerical);
  return [role,{matched,expected,observed:{exit,routeVerified,diagnostics:causes.slice(-4),numerical},output:output.slice(-3000)}];
 }));
}
export function checkControls(design,runs,packet){
 return Object.entries(assessControls(design,runs,packet)).filter(([,v])=>!v.matched).map(([role,v])=>({role,...v}));
}
export function validateFixtureReview(review,design){
 assert(['approve','revise','withhold'].includes(review.verdict)&&validReason(review.reason),'submit a verdict with a specific reason');
 assert([...roles,'checker',design.diagnosticRef].includes(review.reference),'cite a scored role, checker or selected diagnostic');return review;
}
export async function scoreControl({dir,role,design,probe,device,remainingSeconds,signal}){
 await mkdir(dir,{recursive:true});
 if(role==='probe')await writeFile(join(dir,'probe.py'),design.code);
 else await Promise.all([['kernel.py',design[role]],['check.py',design.checker],['route.py',await text(route)],['probe.py',"import runpy,sys\nsys.argv=['route.py','check.py','kernel.py'];runpy.run_path('route.py',run_name='__main__')\n"]].map(([name,bytes])=>writeFile(join(dir,name),bytes)));
 const clock=AbortSignal.timeout(Math.max(1,Math.ceil(remainingSeconds*1000)));
 const stop=signal?AbortSignal.any([signal,clock]):clock;stop.throwIfAborted();
 const child=spawn('/bin/bash',[probe,dir,String(device)],{detached:true,stdio:['ignore','pipe','pipe']});
 const chunks=[];let bytes=0,overflow=false;
 const terminate=()=>{if(child.pid)try{process.kill(-child.pid,'SIGKILL');}catch(e){if(e.code!=='ESRCH')throw e;}};
 const collect=chunk=>{bytes+=chunk.length;if(bytes<=8*1024*1024)chunks.push(chunk);else if(!overflow){overflow=true;terminate();}};
 child.stdout.on('data',collect);child.stderr.on('data',collect);
 stop.addEventListener('abort',terminate,{once:true});
 let exit;
 try{
  exit=await new Promise((res,rej)=>{child.once('error',rej);child.once('close',res);});
 }finally{stop.removeEventListener('abort',terminate);}
 stop.throwIfAborted();assert(!overflow,'control output exceeded 8 MiB');
 const output=Buffer.concat(chunks).toString();
 assert(Number.isInteger(exit)&&output.trim()&&!/sandbox policy unavailable|aclInit, error code is 507899/i.test(output),'control infrastructure error: '+output.slice(-1200));
 return {exit,output};
}

export async function runFixtureDesign({config,input,arm,dir,probe,device,sdk,runtime,signal,seconds=3600},{session,score=scoreControl,now=Date.now,designerSkill=pinnedDesigner,modelIdentity}={}){
 assert(['control','skill','designer'].includes(arm)&&Number.isFinite(seconds)&&seconds>0);
 const packet=publicPacket(input);
 const examples=await (await import('../examples/fixture-design/row-sum/packets.mjs')).rowSumHandoffs();
 const profile=config.skill_writer_actor??config.actor;
 const model=await (modelIdentity??(await import('../controller/skill-context.mjs')).skillModelIdentity)(profile);
 const sdkHash=await sdkSourceIdentity(sdk);
 const identity=hash(JSON.stringify([arm,String(device),hash(JSON.stringify(examples)),sdkHash,model,profile?.base_url,packet,runtime,seconds,hash(await text(probe)),hash(await text(route)),hash(await text(shape)),hash(await text(new URL(import.meta.url))),arm==='control'?null:hash(await text(designerSkill))]));
 await mkdir(dir,{recursive:true});
 const binding=join(dir,'binding.json'),previous=await get(binding);
 if(previous)assert.equal(previous.identity,identity,'replay inputs changed');
 else await writeFile(binding,JSON.stringify({identity,deadline:now()+seconds*1000}),{flag:'wx'});
 const {deadline}=await get(binding);
 const outcome=join(dir,'outcome.json');
 if(await get(outcome))return get(outcome);
 const finish=async result=>{await save(outcome,result);return result;};
 session??=(await import('./skill-review-native.mjs')).reviewSession;
 let feedback=null;const scored=new Map();
 try{
 for(let round=1;now()<deadline;round++){
  signal?.throwIfAborted();
  const path=join(dir,'round-'+round);await mkdir(path,{recursive:true});
  const prior=await get(join(path,'feedback.json'));if(prior){
   for(const role of roles){const key=prior.controlKeys?.[role];if(key)scored.set(key,join(path,role,'score.json'));}
   feedback=prior;continue;
  }
  const author=join(path,'fixture.json');let d=await get(author);
  if(!d){
   const {value}=await session({directory:join(path,'designer'),config,fixtureDesigner:arm!=='control',signal,seconds:(deadline-now())/1000,allowedPaths:[sdk],resultSchema:designSchema,
    prompt:{workedExample:feedback?examples.feedback:examples.designer,task:'Design a bounded device fixture for the observed leaf operation. Use the parent contract for context. Submit test, probe (a small standalone Python question run in the bound runtime), or withhold with a specific missing fact and evidence basis. Supply callable implementations and an independent checker accepting the implementation filename as argv[1]. Vary values, include boundaries and preserved output sentinels where relevant. Select diagnosticRef from records.diagnosticOptions. Bad must launch in the same Python process and trigger AssertionError with the stable checkId followed by colon, space and actual numerical details. Keep the oracle in the checker. The runner executes each control and returns observations.',operation:packet.operation,originalContract:packet.originalContract,records:packet,sdk,feedback},validate:v=>validateDesign(v,packet,feedback)});
   d=validateDesign(value,packet,feedback);await save(author,d);
  }
  if(d.decision==='withhold'){
   if(now()>=deadline)return finish({status:'elapsed',round});
   const {value}=await session({directory:join(path,'withhold-review'),config,signal,seconds:(deadline-now())/1000,resultSchema:schema(['verdict','reason','reference']),
    prompt:{workedExample:examples.reviewer,task:'Check whether the claimed missing evidence is already available in these public records or development measurements. Return withhold if the gap is genuine. Otherwise return revise identifying the existing evidence or a specific development probe.',operation:packet.operation,records:packet,proposedDisposition:d,measurements:feedback?.assessment,probe:feedback?.probe},
    validate:v=>{assert(['withhold','revise'].includes(v.verdict)&&validReason(v.reason));assert(['public-task','failing-source','observed-output',...packet.diagnosticOptions.map(r=>r.id),...(feedback?.assessment?roles:[]),...(feedback?.probe?['probe']:[])].includes(v.reference),'cite an available record');return v;}});
   if(value.verdict==='withhold')return finish({status:'withheld',round,reason:d.reason,missing:d.missing,basis:d.basis,review:value});
   feedback={...feedback,previousDisposition:d,review:value};delete feedback.controlKeys;await save(join(path,'feedback.json'),feedback);continue;
  }
  if(d.decision==='probe'){
   if(now()>=deadline)return finish({status:'elapsed',round});
   const probeFile=join(path,'probe-result.json');
   const result=await get(probeFile)??await score({dir:join(path,'probe'),role:'probe',design:d,probe,device,signal,remainingSeconds:(deadline-now())/1000});
   await save(probeFile,result);feedback={...feedback,probe:{code:d.code,reason:d.reason,exit:result.exit,output:result.output.slice(-3000)}};
   delete feedback.controlKeys;await save(join(path,'feedback.json'),feedback);continue;
  }
  const runs={},controlKeys={};
  for(const role of ['starter','good','bad']){
   const where=join(path,role),stored=join(where,'score.json');await mkdir(where,{recursive:true});
   const key=hash(JSON.stringify([identity,role,d[role],d.checker]));controlKeys[role]=key;
   runs[role]=await get(stored);
   if(!runs[role]&&scored.has(key)){runs[role]=await get(scored.get(key));assert(runs[role]);await save(stored,{...runs[role],reusedFrom:scored.get(key)});}
   if(!runs[role]){
    if(now()>=deadline)return finish({status:'elapsed',round});
    runs[role]=await score({dir:where,role,design:d,probe,device,signal,remainingSeconds:(deadline-now())/1000});await save(stored,runs[role]);
   }
  }
  const failures=checkControls(d,runs,packet);
  for(const role of roles)scored.set(controlKeys[role],join(path,role,'score.json'));
  feedback={previousFixture:d,controlKeys,assessment:assessControls(d,runs,packet),measuredControls:Object.fromEntries(Object.entries(runs).map(([role,r])=>[role,{exit:r.exit,output:r.output.slice(-3000)}])),failures};
  if(!failures.length){
   if(now()>=deadline)return finish({status:'elapsed',round});
   const {value}=await session({directory:join(path,'independent-review'),config,signal,seconds:(deadline-now())/1000,resultSchema:schema(['verdict','reason','reference']),
    prompt:{workedExample:examples.reviewer,task:'Approve only if this bounds the witnessed operation, exercises native behavior and independently checks actual values, tails and untouched output as applicable. Inspect code and measured controls. Cite the role (starter/good/bad), checker or archived diagnostic reference supporting your decision. Give a specific correction or withhold if unsupported.',operation:packet.operation,records:packet,fixture:d,measurements:feedback.assessment},validate:v=>validateFixtureReview(v,d)});
   if(value.verdict!=='revise')return finish({status:value.verdict==='approve'?'qualified':'withheld',round,reason:value.reason});
   feedback.review=value;
  }
  await save(join(path,'feedback.json'),feedback);
 }
 }catch(error){
  signal?.throwIfAborted();
  if(error.name!=='TimeoutError'||(now()<deadline&&error.reviewSession?.kind!=='elapsed'))throw error;
 }
 return finish({status:'elapsed'});
}

export async function fixtureReplayRequest(config,request,signal){
 const {withSkillLock}=await import('./evolution-lock.mjs');
 const input=JSON.parse(await text(request.packet));
 publicPacket(input);
 assert(config.target==='a5'&&/^\d+$/.test(String(request.device))&&request.sdk&&request.runtime);
 const dir=resolve(request.output);
 return withSkillLock(config,()=>runFixtureDesign({...request,input,dir,config,signal}),{signal,key:'fixture-replay:'+hash(dir)});
}
