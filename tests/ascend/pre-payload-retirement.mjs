import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {processIdentity} from '../../controller/lifecycle.mjs';
import {reserveDevice,retirePrePayloadLease,closedDeviceLease,reconcileDeviceLease} from '../../integrations/environments/device-leases.mjs';
import {createEnvironment} from '../../integrations/environments/ascend.mjs';
import {hypothesisEvaluationSettled,evaluateHypothesis} from '../../cli/skill-hypothesis-evaluation.mjs';
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'retire-upload-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const directory=join(root,'evaluation'),instancePath=join(directory,'micro','one','0-active','target');await mkdir(join(instancePath,'.executions','exec-Ab12'),{recursive:true});
 const child=spawn(process.execPath,['-e','setInterval(()=>{},1000)']);await once(child,'spawn');const owner=await processIdentity(child.pid);const exited=once(child,'exit');child.kill();await exited;
 const lease=await reserveDevice('fixture',0,join(root,'leases'),{instancePath});await lease.update({owner});
 const row={submission:'source-upload',kind:'upload',accountingExcluded:true,receiptFile:join(instancePath,'.executions','exec-Ab12','transfer.json')};await lease.event('execution-submitting',row);
 const events=[{type:'device-lease-acquired',data:{receipt:lease.receipt,token:lease.token}},{type:'execution-submitting',data:row}];
 const journal=join(instancePath,'environment-events.jsonl');await writeFile(journal,events.map(JSON.stringify).join('\n')+'\n');
 const pointer={receipt:lease.receipt,token:lease.token,physicalDevice:0};await writeFile(join(instancePath,'.ascend-command-lease.json'),JSON.stringify(pointer));
 const fence=join(root,'fence.json');await writeFile(fence,JSON.stringify({permanent:true,instancePath,...pointer}));
 const args={receipt:lease.receipt,token:lease.token,instancePath,fence};return {root,directory,instancePath,lease,row,events,journal,pointer,fence,args};
}
test('explicit upload retirement preserves unknown transfer evidence while releasing only device custody',async t=>{
 const f=await fixture(t),original=await readFile(f.lease.receipt,'utf8');
 assert.equal((await reconcileDeviceLease(f.lease.receipt,{observe:()=>assert.fail('Unknown identity must not be invented')})).reason,'submission-unknown');
 const proof=await retirePrePayloadLease(f.args);assert.equal(proof.closed,true);assert.equal(proof.payloadStarted,false);assert.equal(proof.retiredPrePayload,true);
 assert.deepEqual(proof.unresolvedArtifacts,[{...f.row,fenced:true}]);assert.equal(await readFile(join(proof.path,'lease.json'),'utf8'),original);
 assert.equal((await closedDeviceLease(f.lease.receipt,f.lease.token)).unresolvedArtifacts[0].handle,undefined);
 const replacement=await reserveDevice('fixture',0,join(f.root,'leases'),{instancePath:join(f.root,'new-target')});assert(replacement);assert.notEqual(replacement.token,f.lease.token);
 assert.equal((await retirePrePayloadLease(f.args)).token,f.lease.token);
});
for(const mode of ['active-owner','missing-owner','wrong-token','unknown-run','mixed-upload-run','noncanonical-upload','payload-intent','missing-fence','unfinished-journal','unclassified-start'])test('pre-payload retirement refuses '+mode,async t=>{
 const f=await fixture(t);
 if(mode==='active-owner')await f.lease.update({owner:await processIdentity()});
 if(mode==='missing-owner')await f.lease.update({owner:null});
 if(mode==='wrong-token')f.args.token='foreign';
 if(mode==='unknown-run')await f.lease.update({submissions:[{...f.row,kind:'run'}]});
 if(mode==='mixed-upload-run')await f.lease.event('execution-submitting',{submission:'run',kind:'run'});
 if(mode==='noncanonical-upload')await f.lease.update({submissions:[{...f.row,receiptFile:join(f.root,'transfer.json')}]});
 if(mode==='payload-intent')await f.lease.update({controlStarted:true});
 if(mode==='missing-fence')await writeFile(f.fence,JSON.stringify({permanent:false}));
 if(mode==='unclassified-start'){f.events.unshift({type:'execution-starting',data:{operation:'unclassified'}});await writeFile(f.journal,f.events.map(JSON.stringify).join('\n')+'\n');}
 if(mode==='unfinished-journal')await writeFile(f.journal,JSON.stringify(f.events[0]));
 assert.equal((await retirePrePayloadLease(f.args)).closed,false);assert(await readFile(f.lease.receipt));assert.equal(await closedDeviceLease(f.lease.receipt,f.lease.token),null);
});
test('a known transfer remains an artifact if observation is unavailable; concurrent lease changes prevent retirement',async t=>{
 const f=await fixture(t),handle='remote:fixture:transfer:known';await f.lease.event('execution-submitted',{submission:f.row.submission,handle});await f.lease.retain(handle);
 const rejected=await retirePrePayloadLease(f.args,{observe:async()=>{await f.lease.update({controlStarted:true});return {state:'unknown'};}});assert.equal(rejected.reason,'ownership-changed');assert(await readFile(f.lease.receipt));
 await f.lease.update({controlStarted:false});const proof=await retirePrePayloadLease(f.args,{observe:async()=>({state:'observation-unavailable'})});assert.equal(proof.closed,true);assert.equal(proof.unresolvedArtifacts[0].closed,undefined);assert.equal(proof.observations[0].state.state,'observation-unavailable');
});
test('evaluation fences before SDK retirement, preserves samples/time, and never silently retires by default',async t=>{
 const f=await fixture(t),file=join(f.directory,'micro','results.json'),original={status:'interrupted',elapsedSeconds:81,results:[{projectId:'completed',correct:true}],runningSample:{key:'one/0/active',unresolvedExecution:{handle:null}}};await writeFile(file,JSON.stringify(original));
 const instructions=join(f.root,'instructions.md');await writeFile(instructions,'fixture');
 const options={mode:'cpl',target:'fixture',runtime:'fixture',instructionFile:instructions,workspacePath:f.root,remoteDirectory:'owned',health:{product:'a3'}},binding={target:'a3',environment:options};
 let retireCalls=0;const deps={createEnvironment:async(options,context)=>{
  const env=await createEnvironment(options,{...context,sdk:{createLocalBashOperations:()=>({})},client:()=>assert.fail('Unknown upload must not be replayed')});
  return {...env,reconcileOperation:async()=>({closed:false}),retirePrePayloadCustody:async args=>{retireCalls++;const report=JSON.parse(await readFile(file,'utf8'));assert.equal(report.settledInterruption.retirementPending,true);return env.retirePrePayloadCustody(args);}};
 }};
 assert.equal(await hypothesisEvaluationSettled({directory:f.directory,binding},deps),false);assert.equal(retireCalls,0);assert.deepEqual(JSON.parse(await readFile(file,'utf8')),original);
 assert.equal(await hypothesisEvaluationSettled({directory:f.directory,binding,retirePrePayload:true},deps),true);assert.equal(retireCalls,1);
 const report=JSON.parse(await readFile(file,'utf8'));assert.equal(report.status,'inconclusive');assert.equal(report.elapsedSeconds,81);assert.deepEqual(report.results,original.results);assert.equal(report.runningSample,undefined);assert(report.settledInterruption.closure);
 assert.equal((await closedDeviceLease(f.lease.receipt,f.lease.token)).unresolvedArtifacts[0].closed,undefined);
 assert.equal(await hypothesisEvaluationSettled({directory:f.directory,binding},deps),true);assert.equal(retireCalls,1);
 const checker=join(f.root,'checker.py');await writeFile(checker,'raise SystemExit(0)');
 const tasks=['one','two','three','four','five'].map(id=>({id,kind:'target-kernel',target:'a3',bucket:'correctness',checker,prompt:'fixture',files:{'kernel.py':'pass'}}));
 const skills={};for(const arm of ['active','candidate']){skills[arm]=join(f.root,arm);await mkdir(skills[arm]);await writeFile(join(skills[arm],'skill-map.json'),JSON.stringify({implementation:'skill.md'}));await writeFile(join(skills[arm],'skill.md'),arm);}
 const outcome=await evaluateHypothesis({directory:f.directory,binding,config:{actor:{provider:'fixture',model:'fixture'}},objective:'correctness',target:'a3',tasks,skills,remainingSeconds:300},{runMicroeval:()=>assert.fail('A permanently retired comparison cannot be rescored')});
 assert.equal(outcome.status,'inconclusive');assert.equal(outcome.decision.reason,'execution-interrupted');assert.deepEqual(outcome.report.results,original.results);assert.equal(outcome.report.elapsedSeconds,81);

});
test('retirement interrupted after archival resumes only settlement through the durable fence',async t=>{
 const f=await fixture(t),file=join(f.directory,'micro','results.json');await writeFile(file,JSON.stringify({status:'interrupted',elapsedSeconds:9,results:[],runningSample:{key:'one/0/active'}}));
 const instructions=join(f.root,'instructions.md');await writeFile(instructions,'fixture');const options={mode:'cpl',target:'fixture',runtime:'fixture',instructionFile:instructions,workspacePath:f.root,remoteDirectory:'owned',health:{product:'a3'}},binding={target:'a3',environment:options};
 let interrupted=false;const deps={createEnvironment:async(options,context)=>{
  const env=await createEnvironment(options,{...context,sdk:{createLocalBashOperations:()=>({})},client:()=>assert.fail('Retired transfer must not be replayed')});
  return {...env,reconcileOperation:async()=>({closed:false}),retirePrePayloadCustody:async args=>{const proof=await env.retirePrePayloadCustody(args);if(!interrupted){interrupted=true;throw Error('Injected interruption after atomic archival');}return proof;}};
 }};
 assert.equal(await hypothesisEvaluationSettled({directory:f.directory,binding,retirePrePayload:true},deps),false);
 assert.equal(JSON.parse(await readFile(file,'utf8')).settledInterruption.retirementPending,true);assert((await closedDeviceLease(f.lease.receipt,f.lease.token)).retiredPrePayload);
 await assert.rejects(readFile(join(f.instancePath,'.ascend-command-lease.json')),{code:'ENOENT'});
 assert.equal(await hypothesisEvaluationSettled({directory:f.directory,binding,retirePrePayload:true},deps),true);
 const report=JSON.parse(await readFile(file,'utf8'));assert.equal(report.elapsedSeconds,9);assert.equal(report.runningSample,undefined);assert(report.settledInterruption.closure);
});
