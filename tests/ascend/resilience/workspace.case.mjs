import assert from 'node:assert/strict';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {schedule,target} from './fixtures.mjs';
import {createEnvironment} from '../../../integrations/environments/index.mjs';

export const cases=[{id:'workspace-snapshots',tier:'local',async run(ctx){
  const sdk=await import('@earendil-works/pi-coding-agent');
  const source=join(ctx.directory,'source');await mkdir(source);
  await writeFile(join(source,'answer.py'),'value = 41\n');
  const backend=await target({root:ctx.directory,scheduler:schedule(ctx)});ctx.onCleanup(()=>backend.close());
  const events=[];
  const environment=await createEnvironment({mode:'cpl',target:'fixture',runtime:'declared',
    workspacePath:backend.workspace,remoteDirectory:'owned',pollMs:5},{sdk,client:backend.client,
    instancePath:ctx.directory,event:async(type,data)=>{events.push({type,data});await ctx.event({type,data});},
    exclude:async(_label,fn)=>fn()});
  await environment.prepare({project:{id:'tiny',source},attempt:{id:'attempt'}});
  const bash=sdk.createBashTool(source);await bash.execute('edit',{command:"python3 -c \"from pathlib import Path;Path('answer.py').write_text('value = 42\\n')\""},ctx.signal);
  assert.equal(await readFile(join(source,'answer.py'),'utf8'),'value = 42\n');
  let output='';const first=await environment.execute('python3 -c "import answer; assert answer.value == sum(6 for _ in range(7)); print(answer.value)"',source,
    {signal:ctx.signal,onData:data=>{output+=data;}});
  assert.equal(first.exitCode,0);assert.equal(first.settled,true);assert.match(output,/42/);
  const observer=await createEnvironment({mode:'cpl',target:'fixture',runtime:'declared',workspacePath:backend.workspace,
    remoteDirectory:'owned',pollMs:5},{sdk,client:backend.client,instancePath:ctx.directory,event:async()=>{},exclude:async(_,fn)=>fn()});
  assert.equal((await observer.observe(first.handle)).state,'stopped');
  assert.equal(backend.calls.filter(c=>c[0]==='run'&&c[c.indexOf('--file')+1].endsWith('command.sh')).length,1);
  return {nativeBashEdit:true,independentOracle:true,handle:first.handle,dispatches:1,
    schedule:'schedule.json',events:events.length};
}}];
