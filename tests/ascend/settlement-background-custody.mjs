import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createEnvironment} from '../../integrations/environments/ascend.mjs';
import {reserveDevice,closeDeviceLease,closedDeviceLease} from '../../integrations/environments/device-leases.mjs';
import {cleanupStatus} from '../../integrations/environments/device-cleanup.mjs';
import {processIdentity} from '../../controller/lifecycle.mjs';
async function fixture(t,observe){
 const root=await mkdtemp(join(tmpdir(),'settlement-custody-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const instructionFile=join(root,'instruction.md');await writeFile(instructionFile,'fixture');
 const events=[],leases=join(root,'leases');let calls=0;
 const environment=await createEnvironment({mode:'cpl',target:'fixture',runtime:'cpu',instructionFile,
  workspacePath:'/fixture',remoteDirectory:'owned',health:{product:'a3'}},{instancePath:root,leaseDirectory:leases,
  event:async(type,data)=>events.push({type,data}),exclude:async(_label,fn)=>fn(),
  cleanupLaunch:async()=>processIdentity(),client:async args=>{calls++;if(args[0]==='observe'&&observe)return observe(args[1]);throw Error('No new native command permitted');}});
 return {root,leases,events,environment,calls:()=>calls};
}
const start={type:'execution-starting',data:{operation:'blocked',startElapsed:651}};
const failure={type:'execution-unresolved',data:{operation:'blocked',handle:null,
 message:'Device lease unresolved; observe retained handles before retry'}};
test('legacy pre-admission failure requires positive preceding token closure and matching observations',async t=>{
 const f=await fixture(t),lease=await reserveDevice('pool',6,f.leases);await lease.retain('remote:fixture:job:prior');
 const observed={type:'execution-state',data:{operation:'blocked',handle:'remote:fixture:job:prior',state:'stopped'}};
 const released={type:'device-lease-released',data:{receipt:lease.receipt,token:lease.token}};
 let events=[{type:'device-lease-acquired',data:{receipt:lease.receipt,token:lease.token,operation:'previous'}},start,observed,failure,released];
 assert.equal((await f.environment.reconcileOperation({operation:'blocked',events})).closed,false);
 await closeDeviceLease(lease);
 const proof=await f.environment.reconcileOperation({operation:'blocked',events});
 assert.equal(proof.closed,true);assert.equal(proof.payloadStarted,false);assert.equal(proof.executionSeconds,0);
 for(const extra of [{type:'execution-submitting',data:{operation:'blocked',submission:'own',kind:'run'}},
  {type:'execution-state',data:{operation:'blocked',handle:'remote:fixture:job:other'}}]){
  assert.equal((await f.environment.reconcileOperation({operation:'blocked',events:[...events,extra]})).closed,false);
 }
 assert.equal(f.calls(),0);
});
test('old-token cleanup custody preserves the newer live physical reservation and unidentified artifact',async t=>{
 const f=await fixture(t,()=>({state:'completed',exit:0,executionSeconds:2})),old=await reserveDevice('pool',5,f.leases);
 const pointer=join(f.root,'.ascend-command-lease.json');
 await writeFile(pointer,JSON.stringify({receipt:old.receipt,token:old.token,pool:'pool',physicalDevice:5}));
 await old.release();const newer=await reserveDevice('pool',5,f.leases);const before=await readFile(newer.receipt,'utf8');
 const events=[{type:'device-lease-acquired',data:{operation:'legacy',token:old.token}},
  {type:'execution-submitting',data:{operation:'legacy',submission:'run',kind:'run',accountingExcluded:false}},
  {type:'execution-submitted',data:{submission:'run',handle:'remote:fixture:job:payload'}},
  {type:'execution-state',data:{handle:'remote:fixture:job:payload',state:'stopped',nativeState:'completed',executionSeconds:2}},
  {type:'execution-submitting',data:{operation:'legacy',submission:'lost-download',kind:'download',accountingExcluded:true,receiptFile:null}}];
 const unknown=[...events,{type:'execution-submitting',data:{operation:'legacy',submission:'unknown-payload',kind:'run',accountingExcluded:false}}];
 assert.equal((await f.environment.reconcileOperation({operation:'legacy',events:unknown})).closed,false);
 assert.equal((await cleanupStatus(join(f.leases,'.cleanup'))).length,0);
 assert.equal(await readFile(newer.receipt,'utf8'),before);
 const proof=await f.environment.reconcileOperation({operation:'legacy',events});
 assert.equal(proof.closed,true);assert.equal(proof.unresolvedArtifacts[0].submission,'lost-download');
 assert.equal(proof.cleanupPending.id,old.token);assert.equal(await readFile(newer.receipt,'utf8'),before);
 assert.equal(await closedDeviceLease(old.receipt,old.token),null);
 const [job]=await cleanupStatus(join(f.leases,'.cleanup'));assert.equal(job.request.token,old.token);
 assert.notEqual(job.status,'closed');assert.equal(f.events.some(e=>e.type==='device-lease-released'),false);
 assert.ok(f.calls()>0);
});

for(const identified of [false,true])test(`failed attempt settles payload and separate metadata download/${identified} before admitting the next batch`,async t=>{
 const {State}=await import('../../controller/state.mjs');
 const {RecoveryQueue}=await import('../../controller/recovery.mjs');
 const {syncCampaignPlan,advanceCampaignEntry}=await import('../../controller/progress.mjs');
 const {reconcileFailure}=await import('../../controller/lifecycle.mjs');
 const {createEnvironment:generic}=await import('../../integrations/environments/index.mjs');
 const f=await fixture(t),state=new State(':memory:');t.after(()=>state.close());const queue=new RecoveryQueue(state);
 const config={campaign_plan:[{id:'one',phase:'phase1',new_projects:1},{id:'two',phase:'phase2',new_projects:1}],projects:[]};
 for(const entry of config.campaign_plan){const p={id:entry.id,goal:'test',actor_seconds:10,phase:entry.phase,campaign_entry:entry.id};config.projects.push(p);state.add(p);}
 syncCampaignPlan(state,config);const job=queue.claim('one',0),id=job.attempt.id;
 queue.recordOriginal(id,{passed:false,findings:'failed',evidence:[{collection:'memory',entry_id:'failed'}]},0);
 queue.acknowledgeLesson(queue.lessons()[0].id);state.event(id,'postprocess-completed',{});
 const handle='remote:fixture:job:payload';
 state.event(id,'execution-starting',{operation:'payload',startElapsed:0});
 state.event(id,'execution-submitting',{operation:'payload',submission:'native',kind:'run',accountingExcluded:false});
 state.event(id,'execution-submitted',{submission:'native',handle});
 // Actual legacy second-operation shape has no enclosing operation and no handle.
 state.event(id,'execution-submitting',{submission:'metadata',kind:'download',accountingExcluded:true,receiptFile:null});
 state.event(id,'execution-submitted',{submission:'metadata',closed:false,handle:identified?'remote:fixture:transfer:metadata':null});
 if(identified)state.event(id,'execution-handle',{handle:'remote:fixture:transfer:metadata'});
 state.event(id,'failure-settlement',{closed:false});let terminal=false;
 const environment=await generic({mode:'cpl',target:'fixture'},{instancePath:f.root,
  event:async(type,data)=>state.event(id,type,data),exclude:async(_label,fn)=>fn(),
  client:async args=>{assert.equal(args[0],'observe');assert.equal(args[1],handle);return {state:terminal?'completed':'running',exit:0,executionSeconds:2};}});
 assert.equal(await reconcileFailure(state,id,environment),false);assert.equal(advanceCampaignEntry(state,config),false);
 terminal=true;assert.equal(await reconcileFailure(state,id,environment),true);
 assert.equal(state.attempt(id).outcome,'failed');assert.equal(state.attempt(id).elapsed,2);
 const custody=state.events(id).find(e=>e.type==='execution-cleanup-custody').data;
 assert.equal(custody.unresolvedArtifacts[0].submission,'metadata');
 assert.equal(state.events(id).some(e=>e.type==='execution-state'&&e.data.handle==null),false);
 assert.equal(advanceCampaignEntry(state,config),true);assert.equal(queue.claim('two',1).project.id,'two');
});

for(const [kind,known]of [['run',true],['run',false],['upload',false],['download',false]])test(`standalone lease-control ${kind}/${known} is reconciled independently`,async t=>{
 const f=await fixture(t,()=>({state:known?'completed':'running',exit:0}));
 const events=[{type:'execution-submitting',data:{submission:'standalone-control',kind,
  accountingExcluded:true,purpose:'device-lease-control',receiptFile:null}},
  {type:'execution-submitted',data:{submission:'standalone-control',handle:known?'remote:fixture:job:control':null,closed:false,purpose:'device-lease-control'}}];
 const proof=await f.environment.reconcileOperation({operation:'standalone-control',events});
 assert.equal(proof.closed,known||kind==='download');
 if(proof.closed)assert.equal(proof.executionSeconds,0);
 if(kind==='download')assert.equal(proof.unresolvedArtifacts[0].submission,'standalone-control');
 assert.equal(f.events.some(e=>e.type==='device-lease-released'),false);
});

test('historical control custody requires its own positive archive and never covers payload handles',async t=>{
 const f=await fixture(t),lease=await reserveDevice('pool',4,f.leases),control='remote:fixture:transfer:control',payload='remote:fixture:job:payload';
 await lease.retain(control);await lease.retain(payload);
 const events=[{type:'device-lease-acquired',data:{receipt:lease.receipt,token:lease.token}},
  {type:'execution-handle',data:{handle:control,purpose:'device-lease-control'}},
  {type:'execution-handle',data:{handle:payload}}];
 assert.deepEqual(await f.environment.cleanupCustody({events}),[]);await closeDeviceLease(lease);
 const [proof]=await f.environment.cleanupCustody({events});assert.deepEqual(proof.handles,[control]);
 assert.equal(proof.cleanup.closedLease.token,lease.token);assert.equal(f.calls(),0);
});
