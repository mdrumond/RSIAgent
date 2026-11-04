import assert from 'node:assert/strict';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {homedir} from 'node:os';
import {randomUUID} from 'node:crypto';
import {Blocked} from './runner.mjs';
import {createEnvironment,createCplClient} from '../../../integrations/environments/index.mjs';

export function caseClient({native,signal,readInstructions,event=async()=>{}}) {
  const handles=new Set();let active=signal;
  const client=async(args,settings={})=>{
    const requestSignal=AbortSignal.any([active,settings.signal].filter(Boolean));
    requestSignal.throwIfAborted();await readInstructions();
    const before=performance.now();
    try {
      const result=await native(args,{...settings,signal:requestSignal});
      if(result.handle)handles.add(result.handle);return result;
    }finally {
      if(args.includes('--receipt-file'))try {
        const receipt=JSON.parse(await readFile(args[args.indexOf('--receipt-file')+1],'utf8'));
        if(receipt.handle)handles.add(receipt.handle);
      }catch{/* Retain the native receipt/error; absence is never closure. */}
      await event({type:'transport-operation',operation:args[0],elapsedSeconds:(performance.now()-before)/1000});
    }
  };
  return {client,handles,async close(environment,directory,{milliseconds=60000,signal}={}){
    active=AbortSignal.any([AbortSignal.timeout(Math.max(1,Math.floor(milliseconds))),signal].filter(Boolean));const observations=[];
    for(const handle of handles) {
      let state;
      try {
        active.throwIfAborted();state=await environment.observe(handle);
        if(state.state==='running')state=await environment.stop(handle);
      }catch(error){state={state:'unknown',error:error.message};}
      observations.push({handle,...state});
    }
    await writeFile(join(directory,'closure.json'),JSON.stringify(observations,null,2));
    assert.ok(observations.every(o=>o.state==='stopped'),'Unresolved owned operations; inspect closure.json');
  }};
}

export const cases=[{id:'registered-cpu-transport',tier:'remote',async run(ctx){
  const profile=process.env.RESILIENCE_REMOTE_PROFILE;
  if(!profile)throw new Blocked('Set RESILIENCE_REMOTE_PROFILE to ignored CPL environment options');
  const config=JSON.parse(await readFile(profile,'utf8'));
  const options=config.environment?.options??config;
  if(!options.target||!options.runtime||!options.workspacePath)throw new Blocked('Registered target/runtime/workspace required');
  const sdk=await import('@earendil-works/pi-coding-agent'),native=createCplClient(options.clientPath);
  const skill=process.env.CPL_REMOTE_SKILL??join(homedir(),'.agents','skills','remote-access','SKILL.md');
  const scoped=caseClient({native,signal:ctx.signal,event:ctx.event,readInstructions:()=>readFile(skill,'utf8')});
  const source=join(ctx.directory,'source');await mkdir(source);
  await writeFile(join(source,'answer.py'),'def answer(): return 6 * 7\n');
  const environment=await createEnvironment({...options,mode:'cpl',remoteDirectory:`resilience/${randomUUID()}`},
    {sdk,client:scoped.client,instancePath:ctx.directory,event:async(type,data)=>{
      if(data.handle)scoped.handles.add(data.handle);await ctx.event({type,data});
    },exclude:async(_label,fn)=>fn()});
  ctx.onCleanup(cleanup=>scoped.close(environment,ctx.directory,{signal:cleanup.signal}));
  await environment.prepare({project:{id:'cpu',source},attempt:{id:'proof'}});
  let stdout='';const result=await environment.execute(
    'python3 -c "import answer; assert answer.answer() == sum(6 for _ in range(7)); print(42)"',source,
    {signal:ctx.signal,timeout:30,onData:data=>{stdout+=data;}});
  assert.equal(result.settled,true);assert.equal(result.exitCode,0);assert.match(stdout,/42/);
  await writeFile(join(ctx.directory,'stdout.txt'),stdout);
  return {scope:'Python CPU only; no NPU or performance evidence',handle:result.handle,closure:'closure.json'};
}}];
