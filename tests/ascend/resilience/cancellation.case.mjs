import assert from 'node:assert/strict';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {homedir} from 'node:os';
import {randomUUID} from 'node:crypto';
import {Blocked} from './runner.mjs';
import {caseClient} from './transport.case.mjs';
import {createEnvironment,createCplClient} from '../../../integrations/environments/index.mjs';

export function cancellationEvidence(result,handle,bodySignal) {
  assert.ok(handle);assert.equal(result.handle,handle);assert.equal(bodySignal.aborted,true);
  assert.ok(result.settled===true||result.settled===false);
  if(!result.settled)assert.equal(result.accountingUnresolved,true,'Unsettled cancellation must remain held');
  return {scope:'Registered CPU cancellation only; no model/NPU',handle,noReplay:true,
    projectPassed:false,initiallyHeld:!result.settled||result.accountingUnresolved===true,
    held:!result.settled||result.accountingUnresolved===true,result,
    closure:'closure.json',nativeCalls:'native-calls.json'};
}

export async function closeCancellation({scoped,observer,directory,cleanup,handle,events,calls,evidence}) {
  const submitted=events.find(e=>e.type==='execution-submitted'&&e.data.handle===handle);
  assert.ok(submitted,'Cancellation requires its retained submission identity');
  let reconciliation;
  const recorded=async(h,state)=>{
    if(h===handle&&state.state==='stopped'&&!reconciliation)
      reconciliation=await observer.reconcileOperation({operation:submitted.data.submission,events});
    return state;
  };
  await scoped.close({observe:async h=>recorded(h,await observer.observe(h)),
    stop:async h=>recorded(h,await observer.stop(h))},directory,
    {signal:cleanup.signal,milliseconds:Math.max(1,Math.min(60000,cleanup.remainingSeconds*1000))});
  assert.ok(reconciliation,'Same-handle terminal closure must precede reconciliation');
  if(reconciliation.closed)assert.ok(Number.isFinite(reconciliation.executionSeconds)&&reconciliation.executionSeconds>=0);
  else assert.equal(reconciliation.accountingUnresolved,true,'Missing timing must remain held');
  assert.ok(calls.some(c=>c.operation==='cancel'&&c.handle===handle&&!c.signalAborted),
    'Native cancellation must run with a fresh signal');
  assert.ok(calls.some(c=>c.operation==='observe'&&c.handle===handle&&!c.signalAborted&&
    ['completed','failed','cancelled'].includes(c.result?.state)));
  assert.equal(calls.filter(c=>c.operation==='run'&&c.file?.endsWith('command.sh')).length,1);
  Object.assign(evidence,{held:!reconciliation.closed,accountingUnresolved:!reconciliation.closed,reconciliation});
  await writeFile(join(directory,'cancellation-proof.json'),JSON.stringify(evidence,null,2));
}

export const cases=[{id:'registered-cpu-cancellation',tier:'remote',async run(ctx){
  const profile=process.env.RESILIENCE_REMOTE_PROFILE;
  if(!profile)throw new Blocked('Set RESILIENCE_REMOTE_PROFILE to ignored registered CPL options');
  const config=JSON.parse(await readFile(profile,'utf8')),base=config.environment?.options??config;
  if(!base.target||!base.runtime||!base.workspacePath)throw new Blocked('Registered target/runtime/workspace required');
  const options={...base,mode:'cpl',remoteDirectory:`resilience-cancellation/${randomUUID()}`};
  const skill=process.env.CPL_REMOTE_SKILL??join(homedir(),'.agents/skills/remote-access/SKILL.md');
  const native=createCplClient(base.clientPath),calls=[],events=[],body=new AbortController();
  let evidence;
  const signal=AbortSignal.any([ctx.signal,body.signal]);let handle;
  const scoped=caseClient({signal:ctx.signal,readInstructions:()=>readFile(skill,'utf8'),event:ctx.event,
    native:async(args,settings)=>{
      const call={operation:args[0],handle:args[1],file:args.includes('--file')?args[args.indexOf('--file')+1]:undefined,
        signalAborted:settings.signal.aborted};calls.push(call);
      try{const result=await native(args,settings);call.result=result;return result;}
      catch(error){call.error=error.message;throw error;}
      finally{await writeFile(join(ctx.directory,'native-calls.json'),JSON.stringify(calls,null,2));}
    }});
  const context={instancePath:ctx.directory,client:scoped.client,signal,exclude:async(_,fn)=>fn(),
    event:async(type,data)=>{
      events.push({type,data});
      if(data.handle)scoped.handles.add(data.handle);
      if(type==='execution-handle'&&!data.purpose){handle=data.handle;body.abort(new Error('Known-receipt cancellation'));}
      await ctx.event({type,data});
      await writeFile(join(ctx.directory,'cancellation-events.json'),JSON.stringify(events,null,2));
    }};
  ctx.onCleanup(async cleanup=>{
    // The body is deliberately aborted; closure uses only the runner's cleanup scope.
    const observer=await createEnvironment(options,{...context,signal:cleanup.signal});
    if(evidence)await closeCancellation({scoped,observer,directory:ctx.directory,cleanup,handle,events,calls,evidence});
    else await scoped.close(observer,ctx.directory,{signal:cleanup.signal});
  });
  const environment=await createEnvironment(options,context),source=join(ctx.directory,'source');
  await mkdir(source);await writeFile(join(source,'wait.py'),'import time\ntime.sleep(90)\n');
  await environment.prepare({project:{id:'cancel-cpu',source},attempt:{id:'proof'}});
  const result=await environment.execute('python3 wait.py',source,{signal,timeout:120});
  evidence=cancellationEvidence(result,handle,body.signal);
  return evidence;
}}];
