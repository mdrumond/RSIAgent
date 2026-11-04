import assert from 'node:assert/strict';
import {mkdir,readFile,writeFile,symlink,unlink} from 'node:fs/promises';
import {join} from 'node:path';
import {homedir} from 'node:os';
import {randomUUID} from 'node:crypto';
import {Blocked} from './runner.mjs';
import {caseClient} from './transport.case.mjs';
import {createEnvironment,createCplClient} from '../../../integrations/environments/index.mjs';

export async function exerciseSymlinks(environment,source,signal){
  await mkdir(source,{recursive:true});await writeFile(join(source,'value.txt'),'first');
  await writeFile(join(source,'../external.txt'),'LOCAL external unchanged');
  await environment.prepare({project:{id:'symlinks',source},attempt:{id:'physical'}});
  const run=async command=>{
    let output='';const result=await environment.execute(command,source,{signal,timeout:60,onData:b=>{output+=b;}});
    assert.equal(result.settled,true);assert.equal(result.exitCode,0,output);return output;
  };
  await run("ln -s . source; printf 'TARGET external unchanged' > ../external.txt; printf retained > target-only.txt");
  await symlink('.',join(source,'source'));await symlink('../external.txt',join(source,'external-link'));
  const check="python3 -c \"import os; from pathlib import Path; assert os.readlink('source')=='.'; assert os.readlink('external-link')=='../external.txt'; assert Path('external-link').read_text()=='TARGET external unchanged'; assert Path('target-only.txt').read_text()=='retained'; ";
  await run(check+"assert Path('source/value.txt').read_text()=='first'\"");
  await writeFile(join(source,'value.txt'),'second');
  await run(check+"assert Path('source/value.txt').read_text()=='second'\"");
  await unlink(join(source,'source'));await unlink(join(source,'external-link'));
  await run("python3 -c \"import os; from pathlib import Path; assert not os.path.lexists('source'); assert not os.path.lexists('external-link'); assert Path('value.txt').read_text()=='second'; assert Path('target-only.txt').read_text()=='retained'; assert Path('../external.txt').read_text()=='TARGET external unchanged'\"");
  assert.equal(await readFile(join(source,'../external.txt'),'utf8'),'LOCAL external unchanged');
  return {physicalLinks:true,preexistingAlias:true,repeatedEdit:true,removedLinks:true,targetArtifactsRetained:true};
}

export const cases=[{id:'registered-cpu-source-symlinks',tier:'remote',async run(ctx){
  const path=process.env.RESILIENCE_REMOTE_PROFILE;
  if(!path)throw new Blocked('Set RESILIENCE_REMOTE_PROFILE to ignored registered CPL options');
  const config=JSON.parse(await readFile(path,'utf8')),base=config.environment?.options??config;
  if(!base.target||!base.runtime||!base.workspacePath)throw new Blocked('Registered target/runtime/workspace required');
  const options={...base,mode:'cpl',remoteDirectory:`symlinks/${randomUUID()}`};
  const skill=process.env.CPL_REMOTE_SKILL??join(homedir(),'.agents/skills/remote-access/SKILL.md');
  const scoped=caseClient({native:createCplClient(base.clientPath),signal:ctx.signal,event:ctx.event,readInstructions:()=>readFile(skill,'utf8')});
  const context={instancePath:ctx.directory,client:scoped.client,event:async(type,data)=>{
    if(data.handle)scoped.handles.add(data.handle);await ctx.event({type,data});
  },exclude:async(_,fn)=>fn()};
  const environment=await createEnvironment(options,{...context,signal:ctx.signal});
  ctx.onCleanup(async cleanup=>{
    const observer=await createEnvironment(options,{...context,signal:cleanup.signal});
    await scoped.close(observer,ctx.directory,{signal:cleanup.signal,milliseconds:Math.max(1,cleanup.remainingSeconds*1000)});
  });
  return {...await exerciseSymlinks(environment,join(ctx.directory,'source'),ctx.signal),scope:'CPU physical source layout; no sandbox or model claim',closure:'closure.json'};
}}];
