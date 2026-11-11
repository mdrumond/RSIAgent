#!/usr/bin/env node
/** Short native-Pi skill comparisons. Shadow artifacts never update a campaign. */
import {mkdir,readFile,writeFile,rename} from 'node:fs/promises';
import {resolve,join,relative,dirname,isAbsolute} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {spawn} from 'node:child_process';
import {parseArgs} from 'node:util';
import {modelFromProfile} from '../controller/model.mjs';

const hash=value=>createHash('sha256').update(value).digest('hex');
async function save(path,value){const tmp=path+'.'+randomUUID();await writeFile(tmp,JSON.stringify(value,null,2)+'\n');await rename(tmp,path);}
const inside=(root,name)=>{const path=resolve(root,name),rel=relative(root,path);if(isAbsolute(name)||!rel||rel==='..'||rel.startsWith('../'))throw Error('Task files must be relative descendants');return path;};
async function skillInput(directory){
 const root=resolve(directory),map=JSON.parse(await readFile(join(root,'skill-map.json'),'utf8')),files={};
 for(const [scope,name] of Object.entries(map))files[scope]=await readFile(inside(root,name),'utf8');
 return {files,sha256:hash(JSON.stringify(files)),text:Object.entries(files).map(([scope,text])=>`## ${scope}\n${text}`).join('\n\n')};
}
async function check(python,checker,cwd,seconds,signal){
 signal?.throwIfAborted();
 const child=spawn(python,[checker,cwd],{cwd,stdio:['ignore','pipe','pipe'],env:{...process.env,PYTHONDONTWRITEBYTECODE:'1'}});
 let output='',error='';child.stdout.on('data',d=>{output=(output+d).slice(-8192);});child.stderr.on('data',d=>{error=(error+d).slice(-8192);});
 const abort=()=>child.kill('SIGKILL');signal?.addEventListener('abort',abort,{once:true});
 const timer=setTimeout(abort,Math.max(1,seconds*1000));
 try{return await new Promise((done,reject)=>{child.once('error',reject);child.once('close',(code,signal)=>done({correct:code===0,exitCode:code,signal,output,error}));});}
 finally{clearTimeout(timer);signal?.removeEventListener('abort',abort);}
}
async function nativeSample({cwd,guidance,prompt,profile,seconds,signal,sessionDirectory}){
 const sdk=await import('@earendil-works/pi-coding-agent');
 const agentDir=join(sessionDirectory,'agent');await mkdir(agentDir,{recursive:true});
 const settingsManager=sdk.SettingsManager.inMemory({compaction:{enabled:true},retry:{enabled:true}});
 const loader=new sdk.DefaultResourceLoader({cwd,agentDir,settingsManager,noExtensions:true,noSkills:true,noContextFiles:true});await loader.reload();
 const manager=sdk.SessionManager.create(cwd,sessionDirectory);
 const {session}=await sdk.createAgentSession({cwd,agentDir,settingsManager,...await modelFromProfile(profile),thinkingLevel:'off',
  tools:['read','write','edit','bash'],resourceLoader:loader,sessionManager:manager});
 let timedOut=false,failure;const timer=setTimeout(()=>{timedOut=true;void session.abort();},Math.max(1,seconds*1000));
 const abort=()=>{void session.abort();};signal?.addEventListener('abort',abort,{once:true});
 try{
  signal?.throwIfAborted();await session.bindExtensions({});session.agent.toolExecution='sequential';
  await session.prompt(`Host-only kernel-engineering microtask. Repair kernel.py in the current directory. Use native tools to inspect, edit and test it. This is not the full RSI workflow; unavailable RSI/KDB tools are not required. Do not inspect or modify files outside this task directory. Finish with the implementation, not just advice.\n\nTask:\n${prompt}\n\nFrozen skill guidance:\n${guidance}`);
  await session.waitForIdle();signal?.throwIfAborted();
 }catch(error){failure=error.message;}
 finally{
  clearTimeout(timer);signal?.removeEventListener('abort',abort);
  try{await session.exportToJsonl(join(sessionDirectory,'session.native.jsonl'));}
  finally{await session.extensionRunner?.emit({type:'session_shutdown',reason:'quit'});await session.dispose();}
 }
 const stats=session.getSessionStats();
 return {timedOut,error:failure??null,tokens:stats.tokens.total>0?stats.tokens.total:null,usage:stats.tokens,
  cost:stats.assistantMessages>0?stats.cost:null,toolCalls:stats.toolCalls,session:manager.getSessionFile()};
}

export async function runMicroeval({directory,skills,tasks,profile,python='python3',runs=2,seconds=120,totalSeconds,signal},dependencies={}){
 const begun=performance.now();signal?.throwIfAborted();
 if(totalSeconds!==undefined&&(!Number.isFinite(totalSeconds)||totalSeconds<=0))throw Error('Positive total elapsed allocation required');
 if(!Number.isSafeInteger(runs)||runs<1||!Number.isFinite(seconds)||seconds<=0||!Array.isArray(tasks)||!tasks.length)throw Error('Positive elapsed allocation, repetitions and tasks required');
 const ids=tasks.map(t=>t.id);if(new Set(ids).size!==ids.length||ids.some(id=>!/^[a-z0-9-]+$/.test(id)))throw Error('Unique task identifiers required');
 const inputs={active:await skillInput(skills.active),candidate:await skillInput(skills.candidate)};
 const frozen=[];for(const t of tasks){
  if(!t.prompt||!t.files||!t.files['kernel.py']||!t.checker)throw Error('Runnable task files and independent Python checker required');
  if(t.scopes&&(!Array.isArray(t.scopes)||t.scopes.some(scope=>!inputs.active.files[scope]||!inputs.candidate.files[scope])))throw Error('Unknown skill scope');
  frozen.push({id:t.id,kind:t.kind??'unspecified',prompt:t.prompt,scopes:t.scopes??Object.keys(inputs.active.files),files:t.files,checker:await readFile(t.checker,'utf8')});
 }
 directory=resolve(directory);await mkdir(dirname(directory),{recursive:true});await mkdir(directory); // Refuse replay/refilled elapsed allocations.
 const manifest={format:1,shadow:true,claim:'host-contract-microtasks; not device performance',runs,seconds,totalSeconds:totalSeconds??null,createdAt:new Date().toISOString(),
  skills:Object.fromEntries(Object.entries(inputs).map(([arm,x])=>[arm,{sha256:x.sha256}])),
  model:{provider:profile.provider,model:profile.model,maxTokens:profile.max_tokens??null},
  tasks:frozen.map(t=>({id:t.id,kind:t.kind,scopes:t.scopes,sha256:hash(JSON.stringify(t))}))};
 await save(join(directory,'manifest.json'),manifest);await mkdir(join(directory,'frozen'));await save(join(directory,'frozen','tasks.json'),frozen);
 for(const [arm,input]of Object.entries(inputs))await writeFile(join(directory,'frozen',arm+'.md'),input.text);
 const report={...manifest,status:'running',results:[]};await save(join(directory,'results.json'),report);
 const sample=dependencies.sample??nativeSample;
 try{
  for(const task of frozen){
   const checker=join(directory,'frozen',task.id+'.py');await writeFile(checker,task.checker);
   for(let run=0;run<runs;run++)for(const arm of run%2?['candidate','active']:['active','candidate']){
    signal?.throwIfAborted();const left=(totalSeconds??Infinity)-(performance.now()-begun)/1000;
    if(left<=0){report.status='elapsed';return report;}
    const allocation=Math.min(seconds,left);const path=join(directory,task.id,`${run}-${arm}`),cwd=join(path,'work');await mkdir(cwd,{recursive:true});
    for(const [name,text]of Object.entries(task.files)){const file=inside(cwd,name);await mkdir(dirname(file),{recursive:true});await writeFile(file,text);}
    const started=performance.now();let observed,verification;
    try{
     observed=await sample({cwd,guidance:task.scopes.map(scope=>`## ${scope}\n${inputs[arm].files[scope]}`).join('\n\n'),prompt:task.prompt,profile,seconds:allocation,signal,sessionDirectory:path});
     const remaining=allocation-(performance.now()-started)/1000;
     if(hash(await readFile(checker,'utf8'))!==hash(task.checker))throw Error('Frozen checker changed');
     verification=remaining>0&&!observed.timedOut&&!observed.error?await check(python,checker,cwd,Math.min(20,remaining),signal):{correct:observed.error&&!observed.timedOut?null:false,error:'Sample elapsed allocation or native session failed'};
    }catch(error){observed??={tokens:null,cost:null};verification={correct:null,error:error.message};}
    const result={projectId:task.id,taskId:task.id,kind:task.kind,scopes:task.scopes,run,arm,allocatedSeconds:allocation,...observed,elapsedSeconds:(performance.now()-started)/1000,
     correct:verification.correct,verification,skillsSha256:inputs[arm].sha256};
    report.results.push(result);await save(join(directory,'results.json'),report);signal?.throwIfAborted();
   }
  }
  report.status='complete';return report;
 }catch(error){report.status='interrupted';report.error=error.message;throw error;}
 finally{await save(join(directory,'results.json'),report);}
}
if(import.meta.main){
 const {values}=parseArgs({options:{config:{type:'string'}}});
 const abort=new AbortController();process.on('SIGINT',()=>abort.abort());process.on('SIGTERM',()=>abort.abort());
 try{const input=JSON.parse(await readFile(values.config,'utf8'));if(input.taskModule){const module=await import(new URL(input.taskModule,'file://'+resolve(values.config)).href);input.tasks=module.tasks;}
  const report=await runMicroeval({...input,signal:abort.signal});console.log(JSON.stringify({status:report.status,results:report.results}));}
 catch(error){console.error(JSON.stringify({error:error.message}));process.exitCode=1;}
}
