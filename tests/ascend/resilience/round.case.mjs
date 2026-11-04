import assert from 'node:assert/strict';
import {mkdir,readdir,writeFile} from 'node:fs/promises';
import {join,dirname,delimiter} from 'node:path';
import {fileURLToPath} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';
import {runCases,Blocked} from './runner.mjs';
import {cases as workspace} from './workspace.case.mjs';
import {closeRegressionServices} from './regression-cleanup.mjs';

const root=fileURLToPath(new URL('../../..',import.meta.url));
async function program(ctx,name,command,args,env=process.env) {
  const {createLocalBashOperations}=await import('@earendil-works/pi-coding-agent');
  const quote=value=>"'"+String(value).replaceAll("'","'\\''")+"'";
  let output='';
  try {
    const result=await createLocalBashOperations().exec([command,...args].map(quote).join(' '),root,
      {env,signal:ctx.signal,onData:data=>{output+=data.toString();}});
    assert.equal(result.exitCode,0,`${name} failed; retained ${name}.log`);
  }
  finally {
    await writeFile(join(ctx.directory,name+'.log'),output);
  }
  return output;
}
const sample={id:'seeded-workspace',tier:'local',async run(ctx){
  return workspace[0].run(ctx);
}};
export const cases=[{id:'existing-regressions',tier:'local',async run(ctx){
  const python=process.env.RESILIENCE_PYTHON??process.env.DRIVER_PYTHON??'python3';
  try {await program(ctx,'prerequisites',python,['-c','import pytest, flask; from mcp.server.fastmcp import FastMCP']);}
  catch{throw new Blocked('Select existing Python with pytest, Flask and FastMCP using RESILIENCE_PYTHON');}
  const files=(await readdir(join(root,'tests','ascend'))).filter(f=>f.endsWith('.mjs')&&
    !/(?:_fixture|_setup|-worker|-fixture)\.mjs$/.test(f)).map(f=>'tests/ascend/'+f).sort();
  const artifacts=join(ctx.directory,'native-tests'),temporary=join(ctx.directory,'temporary');
  await mkdir(artifacts);await mkdir(temporary);
  ctx.onCleanup(options=>closeRegressionServices(ctx.directory,options));
  const env={...process.env,PYTHON:python,DRIVER_PYTHON:python,RSI_KDB_TEST_PYTHON:python,RECOVERY_PYTHON:python,DRIVER_ARTIFACTS:artifacts,
    TMPDIR:temporary,TEMP:temporary,TMP:temporary,
    PATH:dirname(process.execPath)+delimiter+process.env.PATH};
  await program(ctx,'node',process.execPath,['--test','--test-concurrency=1',...files],env);
  await program(ctx,'python',python,['-m','pytest','-q','tests/ascend'],env);
  return {nodeEntryPoints:files,python:'tests/ascend',logs:['node.log','python.log']};
}},sample,{id:'seeded-soak',tier:'local',async run(ctx){
  const until=performance.now()+600000,completed=[];let seed=ctx.seed;
  while(performance.now()<until) {
    ctx.signal.throwIfAborted();const remaining=(until-performance.now())/1000;
    if(remaining<30){await delay(remaining*1000,undefined,{signal:ctx.signal});break;}
    seed=(Math.imul(seed,1664525)+1013904223)>>>0;
    const profile=['normal','slow','bursty'][seed%3],output=join(ctx.directory,`sample-${completed.length}`);
    const bounded={...sample,run:inner=>sample.run({...inner,signal:AbortSignal.any([inner.signal,ctx.signal])})};
    const report=await runCases([bounded],{output,tier:'local',seed,profile,
      deadlineSeconds:Math.min(remaining,60),cleanupSeconds:5});
    assert.equal(report.status,'passed',`Soak sample failed; retained ${report.output}`);
    completed.push({seed,profile,report:join(output,'report.json')});
    await ctx.event({type:'soak-sample',...completed.at(-1)});
  }
  assert.ok(completed.length>0);return {durationTargetSeconds:600,samples:completed};
}}];
