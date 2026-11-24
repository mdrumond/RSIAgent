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
const designSchema=schema(['decision','reason','prompt','starter','good','bad','checker','diagnostic','mathFailure']);
const shape=fileURLToPath(new URL('./fixture-native-shape.py',import.meta.url));
const route=fileURLToPath(new URL('./fixture-native-route.py',import.meta.url));
const pinnedDesigner=new URL('./skills/fixture-designer/SKILL.md',import.meta.url);
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
 return {...visible,operation,originalContract};
}
export function validateDesign(d,packet){
 assert(d.decision==='test'||d.decision==='withhold','submit test or withhold');
 assert(d.reason?.trim(),'supply a reason');
 if(d.decision==='withhold')return d;
 for(const key of designSchema.required)assert(typeof d[key]==='string'&&d[key].trim(),'missing '+key);
 const cause=d.diagnostic.trim();
 assert(!/[\r\n]/.test(cause)&&diagnosticLine(cause)&&/[a-zA-Z]{2}/.test(cause)&&cause.split(/\W+/).filter(Boolean).length>=2&&causeLines(packet['observed-output']).some(line=>line.includes(cause)),'copy a concrete cause from an archived diagnostic line, including its distinguishing detail');
 const p=spawnSync('python3',[shape],{input:JSON.stringify([d,[packet['failing-source']]]),encoding:'utf8'});
 if(p.error)throw p.error;if(p.status!==0)throw Error(p.stderr.trim());
 return d;
}
const verified=/^VERIFIED_NATIVE_ROUTE [1-9]\d* [1-9]\d*\s*$/m;
export function checkControls(design,runs){
 return ['starter','good','bad'].flatMap(role=>{
  const {exit,output}=runs[role];
  const ok=role==='starter'?exit!==0&&causeLines(output).some(line=>line.includes(design.diagnostic.trim())):role==='good'?exit===0&&verified.test(output):exit!==0&&verified.test(output)&&output.split(/\r?\n/).includes('AssertionError: '+design.mathFailure);
  return ok?[]:[{role,exit,output:output.slice(-3000)}];
 });
}
export async function scoreControl({dir,role,design,probe,device,remainingSeconds,signal}){
 await mkdir(dir,{recursive:true});
 await Promise.all([['kernel.py',design[role]],['check.py',design.checker],['route.py',await text(route)],['probe.py',"import runpy,sys\nsys.argv=['route.py','check.py','kernel.py'];runpy.run_path('route.py',run_name='__main__')\n"]].map(([name,bytes])=>writeFile(join(dir,name),bytes)));
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
 const profile=config.skill_writer_actor??config.actor;
 const model=await (modelIdentity??(await import('../controller/skill-context.mjs')).skillModelIdentity)(profile);
 const sdkHash=await sdkSourceIdentity(sdk);
 const identity=hash(JSON.stringify([arm,String(device),sdkHash,model,profile?.base_url,packet,runtime,seconds,hash(await text(probe)),hash(await text(route)),hash(await text(shape)),hash(await text(new URL(import.meta.url))),arm==='control'?null:hash(await text(designerSkill))]));
 await mkdir(dir,{recursive:true});
 const binding=join(dir,'binding.json'),previous=await get(binding);
 if(previous)assert.equal(previous.identity,identity,'replay inputs changed');
 else await writeFile(binding,JSON.stringify({identity,deadline:now()+seconds*1000}),{flag:'wx'});
 const {deadline}=await get(binding);
 const outcome=join(dir,'outcome.json');
 if(await get(outcome))return get(outcome);
 const finish=async result=>{await save(outcome,result);return result;};
 session??=(await import('./skill-review-native.mjs')).reviewSession;
 let feedback=null;
 for(let round=1;now()<deadline;round++){
  signal?.throwIfAborted();
  const path=join(dir,'round-'+round);await mkdir(path,{recursive:true});
  const prior=await get(join(path,'feedback.json'));if(prior){feedback=prior;continue;}
  const author=join(path,'fixture.json');let d=await get(author);
  if(!d){
   const {value}=await session({directory:join(path,'designer'),config,fixtureDesigner:arm!=='control',signal,seconds:(deadline-now())/1000,allowedPaths:[sdk],resultSchema:designSchema,
    prompt:{task:'Design a bounded device fixture for the observed leaf operation. Use the parent contract for context. Submit test or withhold. Supply callable implementations and an independent checker accepting the implementation filename as argv[1]. Vary values, include boundaries and preserved output sentinels where relevant. Bad must launch and trigger AssertionError with mathFailure. The runner executes each control and returns observations.',operation:packet.operation,originalContract:packet.originalContract,records:packet,sdk,feedback},validate:v=>validateDesign(v,packet)});
   d=validateDesign(value,packet);await save(author,d);
  }
  if(d.decision==='withhold')return finish({status:'withheld',round,reason:d.reason});
  const runs={};
  for(const role of ['starter','good','bad']){
   const where=join(path,role),stored=join(where,'score.json');await mkdir(where,{recursive:true});
   runs[role]=await get(stored);
   if(!runs[role]){
    if(now()>=deadline)return finish({status:'elapsed',round});
    runs[role]=await score({dir:where,role,design:d,probe,device,signal,remainingSeconds:(deadline-now())/1000});await save(stored,runs[role]);
   }
  }
  const failures=checkControls(d,runs);
  feedback={previousFixture:d,measuredControls:Object.fromEntries(Object.entries(runs).map(([role,r])=>[role,{exit:r.exit,output:r.output.slice(-3000)}])),failures};
  if(!failures.length){
   if(now()>=deadline)return finish({status:'elapsed',round});
   const {value}=await session({directory:join(path,'independent-review'),config,signal,seconds:(deadline-now())/1000,resultSchema:schema(['verdict','reason']),
    prompt:{task:'Approve only if this bounds the witnessed operation, exercises native behavior and independently checks actual values, tails and untouched output as applicable. Inspect code and measured controls. Give a specific correction or withhold if unsupported.',operation:packet.operation,records:packet,fixture:d,measurements:runs},validate:v=>{assert(['approve','revise','withhold'].includes(v.verdict)&&v.reason?.trim());return v;}});
   if(value.verdict!=='revise')return finish({status:value.verdict==='approve'?'qualified':'withheld',round,reason:value.reason});
   feedback.review=value;
  }
  await save(join(path,'feedback.json'),feedback);
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
