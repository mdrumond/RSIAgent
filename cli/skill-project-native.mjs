/** Proposal-only native Pi file sessions. Device execution remains outside the model. */
import {mkdir,readFile,writeFile,realpath,appendFile} from 'node:fs/promises';
import {join,resolve,relative,isAbsolute,dirname} from 'node:path';
import {skillModelFromProfile} from '../controller/skill-context.mjs';
const inside=(root,path)=>{const r=relative(root,path);return r===''||!isAbsolute(r)&&r!=='..'&&!r.startsWith('../');};
export function projectDeadline(seconds,signal,began=performance.now()){
 if(!Number.isFinite(seconds)||seconds<=0)throw Error('Positive elapsed allocation required');
 const controller=new AbortController();let timer,expired=false;
 const arm=()=>{const left=seconds*1000-(performance.now()-began);if(left<=0){expired=true;controller.abort(Error('Elapsed allocation exhausted'));}else{timer=setTimeout(arm,Math.min(left,2147483647));timer.unref();}};arm();
 return {signal:signal?AbortSignal.any([signal,controller.signal]):controller.signal,expired:()=>expired,dispose:()=>clearTimeout(timer)};
}
export async function nativeProjectRole({directory,files,prompt,profile,seconds,signal,editable=[],submission,validate}){
 const began=performance.now();
 const sdk=await import('@earendil-works/pi-coding-agent'),cwd=join(directory,'work'),agentDir=join(directory,'agent');
 await mkdir(cwd,{recursive:true});await mkdir(agentDir,{recursive:true});
 for(const[name,text]of Object.entries(files)){const file=join(cwd,name);await mkdir(dirname(file),{recursive:true});await writeFile(file,text,{flag:'wx'});}
 const canonical=await realpath(cwd),writable=new Set([...editable,...(submission?[submission]:[])]);
 const extension=pi=>pi.on('tool_call',async e=>{
  const name=resolve(cwd,e.input.path??'.'),rel=relative(cwd,name),actual=await realpath(name).catch(error=>{if(error.code!=='ENOENT')throw error;return name;});
  if(!inside(canonical,actual)||!['read','write','edit'].includes(e.toolName)||!(e.toolName==='read'||writable.has(rel))){
   await appendFile(join(directory,'policy.jsonl'),JSON.stringify({tool:e.toolName,path:e.input.path})+'\n');return {block:true,reason:'Read the named task files and write only the declared output files.'};
  }
  if(validate&&rel===submission&&e.toolName!=='read'){
   try{if(e.toolName!=='write')throw Error('Use write with the complete JSON submission so it can be validated before saving');await validate(JSON.parse(e.input.content));}
   catch(error){await appendFile(join(directory,'submission-rejections.jsonl'),JSON.stringify({tool:e.toolName,error:error.message,input:e.input})+'\n');return {block:true,reason:'Submission rejected: '+error.message};}
  }
 });
 const settings=sdk.SettingsManager.inMemory({retry:{enabled:true},compaction:{enabled:true}}),loader=new sdk.DefaultResourceLoader({cwd,agentDir,settingsManager:settings,noExtensions:true,noSkills:true,noContextFiles:true,extensionFactories:[extension]});await loader.reload();
 const {session}=await sdk.createAgentSession({cwd,agentDir,settingsManager:settings,...await skillModelFromProfile(profile,{directory:directory}),thinkingLevel:'off',tools:['read','write','edit'],resourceLoader:loader,sessionManager:sdk.SessionManager.create(cwd,join(directory,'session'))});
 let last,expired=false,error=null;const unsubscribe=session.subscribe(e=>{if(e.type==='message_end'&&e.message.role==='assistant')last=e.message;}),abort=()=>{void session.abort();};
 const deadline=projectDeadline(seconds,signal,began);signal=deadline.signal;signal.addEventListener('abort',abort,{once:true});
 try{signal?.throwIfAborted();await session.bindExtensions({});session.agent.toolExecution='sequential';await session.prompt(prompt);await session.waitForIdle();signal?.throwIfAborted();if(['error','length','aborted'].includes(last?.stopReason))error=last.errorMessage??'provider-'+last.stopReason;}
 catch(e){error=e.message;}
 finally{expired=deadline.expired();deadline.dispose();signal?.removeEventListener('abort',abort);unsubscribe();try{await session.exportToJsonl(join(directory,'session.native.jsonl'));}finally{await session.dispose();}}
 let value;if(submission&&!error&&!expired)try{value=JSON.parse(await readFile(join(cwd,submission),'utf8'));await validate?.(value);}catch(e){error='Invalid '+submission+': '+e.message;}
 const result={value,expired,error,stopReason:last?.stopReason??null,elapsedSeconds:(performance.now()-began)/1000,stats:session.getSessionStats()};await writeFile(join(directory,'role-result.json'),JSON.stringify(result,null,2)+'\n');return {...result,cwd};
}
