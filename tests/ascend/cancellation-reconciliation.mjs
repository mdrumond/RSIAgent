import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createEnvironment} from '../../integrations/environments/index.mjs';
import {caseClient} from './resilience/transport.case.mjs';
import {cancellationEvidence,closeCancellation} from './resilience/cancellation.case.mjs';

async function fixture(t,{settled=false,unknown=false,timing}={}) {
  const directory=await mkdtemp(join(tmpdir(),'cancel-reconcile-'));t.after(()=>rm(directory,{recursive:true,force:true}));
  const body=new AbortController();body.abort();
  const cleanup={signal:AbortSignal.timeout(2000),remainingSeconds:2},handle='remote:fixture:job:retained',calls=[],events=[];
  let state=unknown?'unknown':settled?'cancelled':'running';
  const scoped=caseClient({signal:body.signal,readInstructions:async()=>{},native:async(args,{signal})=>{
    assert.equal(signal.aborted,false);assert.notEqual(signal,body.signal);
    const call={operation:args[0],handle:args[1],signalAborted:signal.aborted};calls.push(call);
    if(args[0]==='cancel')state=unknown?'unknown':'cancelled';
    const result=args[0]==='observe'?{state,...(timing===undefined?{}:{executionSeconds:timing})}:{state:'completed'};
    call.result=result;return result;
  }});
  scoped.handles.add(handle);
  calls.push({operation:'run',file:'/retained/command.sh',result:{handle}});
  // The production stop already requested cancellation; its bounded wait may have expired.
  calls.push({operation:'cancel',handle,signalAborted:false,result:{state:'cancelling'}});
  events.push({type:'execution-submitting',data:{submission:'work',kind:'run',accountingExcluded:false}},
    {type:'execution-submitted',data:{submission:'work',handle}});
  const observer=await createEnvironment({mode:'cpl',target:'fixture',workspacePath:'/registered',remoteDirectory:'test'},
    {instancePath:directory,client:scoped.client,signal:cleanup.signal,event:async(type,data)=>events.push({type,data}),exclude:async(_,fn)=>fn()});
  const evidence=cancellationEvidence({handle,settled,accountingUnresolved:timing===undefined},handle,body.signal);
  return {directory,cleanup,handle,calls,events,observer,scoped,evidence};
}
for(const settled of [true,false])test(`cancelled work ${settled?'immediately stopped':'still running after stop budget'} closes without inventing timing`,async t=>{
  const f=await fixture(t,{settled});await closeCancellation(f);
  assert.equal(f.evidence.projectPassed,false);assert.equal(f.evidence.held,true);
  assert.equal(f.evidence.accountingUnresolved,true);assert.equal(f.evidence.reconciliation.closed,false);
  const proof=JSON.parse(await readFile(join(f.directory,'cancellation-proof.json'),'utf8'));
  assert.equal(proof.result.settled,settled);assert.equal(proof.held,true);
  const closure=JSON.parse(await readFile(join(f.directory,'closure.json'),'utf8'));
  assert.equal(closure.find(c=>c.handle===f.handle).state,'stopped');
  assert.equal(f.calls.filter(c=>c.operation==='run').length,1);
});
test('terminal work with actual elapsed evidence reconciles accounting but never claims project acceptance',async t=>{
  const f=await fixture(t,{settled:false,timing:1.25});await closeCancellation(f);
  assert.equal(f.evidence.initiallyHeld,true);assert.equal(f.evidence.held,false);
  assert.equal(f.evidence.reconciliation.closed,true);assert.equal(f.evidence.reconciliation.executionSeconds,1.25);
  assert.equal(f.evidence.projectPassed,false);
});
test('unknown terminal identity fails and preserves unknown closure instead of passing held work',async t=>{
  const f=await fixture(t,{unknown:true});await assert.rejects(closeCancellation(f),/Unresolved owned operations/);
  const closure=JSON.parse(await readFile(join(f.directory,'closure.json'),'utf8'));
  assert.equal(closure[0].state,'unknown');assert.equal(f.evidence.held,true);
  assert.equal(f.calls.filter(c=>c.operation==='run').length,1);
});
test('unsettled result without held accounting evidence is rejected',()=>{
  const body=new AbortController();body.abort();
  assert.throws(()=>cancellationEvidence({handle:'known',settled:false},'known',body.signal));
});
