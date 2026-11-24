import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir,rename} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash,randomUUID} from 'node:crypto';
import {spawnSync} from 'node:child_process';

const hash=x=>createHash('sha256').update(x).digest('hex');
const text=p=>readFile(p,'utf8');
const get=async p=>{try{return JSON.parse(await text(p));}catch(e){if(e.code!=='ENOENT')throw e;return null;}};
const save=async(p,data)=>{const tmp=p+'.'+randomUUID();await writeFile(tmp,JSON.stringify(data,null,2));await rename(tmp,p);};
const schema=names=>({type:'object',properties:Object.fromEntries(names.map(k=>[k,{type:'string'}])),required:names,additionalProperties:false});
const designSchema=schema(['decision','reason','prompt','starter','good','bad','checker','diagnostic','mathFailure']);
const shape=fileURLToPath(new URL('./fixture-native-shape.py',import.meta.url));
const route=fileURLToPath(new URL('./fixture-native-route.py',import.meta.url));
const fields=['public-task','failing-source','observed-output','inherited-guidance'];

export function publicPacket(input){
 const visible={};
 for(const name of fields){
  const matches=input.records.filter(r=>r.reference===name);
  assert(matches.length===1&&typeof matches[0].text==='string','expected one public '+name+' record');visible[name]=matches[0].text;
 }
 return visible;
}
export function validateDesign(d,packet){
 assert(d.decision==='test'||d.decision==='withhold','submit test or withhold');
 assert(d.reason?.trim(),'supply a reason');
 if(d.decision==='withhold')return d;
 for(const key of designSchema.required)assert(typeof d[key]==='string'&&d[key].trim(),'missing '+key);
 assert(!/[\r\n]/.test(d.diagnostic)&&packet['observed-output'].includes(d.diagnostic),'use a single diagnostic anchor copied from observed output');
 const p=spawnSync('python3',[shape],{input:JSON.stringify([d,[packet['failing-source']]]),encoding:'utf8'});
 if(p.error)throw p.error;if(p.status!==0)throw Error(p.stderr.trim());
 return d;
}
const verified=/^VERIFIED_NATIVE_ROUTE [1-9]\d* [1-9]\d*\s*$/m;
export function checkControls(design,runs){
 return ['starter','good','bad'].flatMap(role=>{
  const {exit,output}=runs[role];
  const ok=role==='starter'?exit!==0&&output.includes(design.diagnostic):role==='good'?exit===0&&verified.test(output):exit!==0&&verified.test(output)&&output.split(/\r?\n/).includes('AssertionError: '+design.mathFailure);
  return ok?[]:[{role,exit,output:output.slice(-3000)}];
 });
}
export async function scoreControl({dir,role,design,probe,device,remainingSeconds}){
 await mkdir(dir,{recursive:true});
 await Promise.all([['kernel.py',design[role]],['check.py',design.checker],['route.py',await text(route)],['probe.py',"import runpy,sys\nsys.argv=['route.py','check.py','kernel.py'];runpy.run_path('route.py',run_name='__main__')\n"]].map(([name,bytes])=>writeFile(join(dir,name),bytes)));
 const p=spawnSync('/bin/bash',[probe,dir,String(device)],{encoding:'utf8',maxBuffer:8*1024*1024,timeout:Math.max(1,Math.ceil(remainingSeconds*1000))});
 if(p.error)throw p.error;
 const output=p.stdout+p.stderr;
 assert(Number.isInteger(p.status)&&output.trim()&&!/sandbox policy unavailable|aclInit, error code is 507899/i.test(output),'control infrastructure error: '+output.slice(-1200));
 return {exit:p.status,output};
}

export async function runFixtureDesign({config,input,arm,dir,probe,device,sdk,runtime,signal,seconds=3600},{session,score=scoreControl,now=Date.now}={}){
 assert(['control','skill','designer'].includes(arm)&&Number.isFinite(seconds)&&seconds>0);
 const packet=publicPacket(input);
 const identity=hash(JSON.stringify([arm,packet,input.operation,input.originalContract,runtime,seconds,hash(await text(probe)),hash(await text(route)),hash(await text(shape))]));
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
    prompt:{task:'Design a bounded device fixture for the observed leaf operation. Use the parent contract for context. Submit test or withhold. Supply callable implementations and an independent checker accepting the implementation filename as argv[1]. Vary values, include boundaries and preserved output sentinels where relevant. Bad must launch and trigger AssertionError with mathFailure. The runner executes each control and returns observations.',operation:input.operation,originalContract:input.originalContract,records:packet,sdk,feedback},validate:v=>validateDesign(v,packet)});
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
  feedback=checkControls(d,runs);
  if(!feedback.length){
   if(now()>=deadline)return finish({status:'elapsed',round});
   const {value}=await session({directory:join(path,'independent-review'),config,signal,seconds:(deadline-now())/1000,resultSchema:schema(['verdict','reason']),
    prompt:{task:'Approve only if this bounds the witnessed operation, exercises native behavior and independently checks actual values, tails and untouched output as applicable. Inspect code and measured controls. Give a specific correction or withhold if unsupported.',operation:input.operation,records:packet,fixture:d,measurements:runs},validate:v=>{assert(['approve','revise','withhold'].includes(v.verdict)&&v.reason?.trim());return v;}});
   if(value.verdict!=='revise')return finish({status:value.verdict==='approve'?'qualified':'withheld',round,reason:value.reason});
   feedback=value;
  }
  await save(join(path,'feedback.json'),feedback);
 }
 return finish({status:'elapsed'});
}

export async function fixtureReplayRequest(config,request,signal){
 const {withSkillLock}=await import('./evolution-lock.mjs');
 const input=JSON.parse(await text(request.packet));
 assert(config.target==='a5'&&/^\d+$/.test(String(request.device))&&request.sdk&&request.runtime);
 const dir=resolve(request.output);
 return withSkillLock(config,()=>runFixtureDesign({...request,input,dir,config,signal}),{signal,key:'fixture-replay:'+hash(dir)});
}
