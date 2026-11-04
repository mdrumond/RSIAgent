import assert from 'node:assert/strict';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {homedir} from 'node:os';
import {randomUUID} from 'node:crypto';
import {Blocked} from './runner.mjs';
import {createEnvironment,createCplClient} from '../../../integrations/environments/index.mjs';

export const cases=[{id:'registered-cpu-transport',tier:'remote',async run(ctx){
  const profile=process.env.RESILIENCE_REMOTE_PROFILE;
  if(!profile)throw new Blocked('Set RESILIENCE_REMOTE_PROFILE to ignored CPL environment options');
  const config=JSON.parse(await readFile(profile,'utf8'));
  const options=config.environment?.options??config;
  if(!options.target||!options.runtime||!options.workspacePath)throw new Blocked('Registered target/runtime/workspace required');
  const sdk=await import('@earendil-works/pi-coding-agent'),native=createCplClient(options.clientPath);
  const handles=new Set(),observations=[];
  const skill=process.env.CPL_REMOTE_SKILL??join(homedir(),'.agents','skills','remote-access','SKILL.md');
  const client=async(args,settings)=>{
    await readFile(skill,'utf8');const result=await native(args,settings);
    if(result.handle)handles.add(result.handle);return result;
  };
  const source=join(ctx.directory,'source');await mkdir(source);
  await writeFile(join(source,'answer.py'),'def answer(): return 6 * 7\n');
  const environment=await createEnvironment({...options,mode:'cpl',remoteDirectory:`resilience/${randomUUID()}`},
    {sdk,client,instancePath:ctx.directory,event:async(type,data)=>{
      if(data.handle)handles.add(data.handle);await ctx.event({type,data});
    },exclude:async(_label,fn)=>fn()});
  ctx.onCleanup(async()=>{
    for(const handle of handles) {
      let state=await environment.observe(handle);
      if(state.state==='running')state=await environment.stop(handle);
      observations.push({handle,...state});assert.equal(state.state,'stopped',`Unclosed owned operation ${handle}`);
    }
    await writeFile(join(ctx.directory,'closure.json'),JSON.stringify(observations,null,2));
  });
  await environment.prepare({project:{id:'cpu',source},attempt:{id:'proof'}});
  let stdout='';const result=await environment.execute(
    'python3 -c "import answer; assert answer.answer() == sum(6 for _ in range(7)); print(42)"',source,
    {signal:ctx.signal,timeout:30,onData:data=>{stdout+=data;}});
  assert.equal(result.settled,true);assert.equal(result.exitCode,0);assert.match(stdout,/42/);
  await writeFile(join(ctx.directory,'stdout.txt'),stdout);
  return {scope:'Python CPU only; no NPU or performance evidence',handle:result.handle,closure:'closure.json'};
}}];
