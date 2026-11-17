import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {DatabaseSync} from 'node:sqlite';
import {admitLearning,learningAdmissionSettled,refreshLearningRoute} from '../../cli/learning-admission.mjs';
import {reserveDevice} from '../../integrations/environments/device-leases.mjs';
import {cleanupDependencies,cleanupCandidates} from '../../integrations/environments/cleanup-dependencies.mjs';
async function fixture(t){const root=await mkdtemp(join(tmpdir(),'learning-admission-'));t.after(()=>rm(root,{recursive:true,force:true}));return root;}
test('CPU learning skips all target work; device checks reject existing lease without releasing it',async t=>{
 const root=await fixture(t),directory=join(root,'admission'),leaseDirectory=join(root,'leases'),config={root,target:'a3',environment:{options:{target:'registered'}}},request={learningTarget:'device'},resource={id:'slot',metadata:{pool:'pool'}};
 let probes=0;const ops={leaseDirectory,createEnvironment:async()=>({inspectDevices:async()=>{probes++;return [4,5].map(physicalDevice=>({available:true,metadata:{physicalDevice,admitted:true}}));}})};
 assert.equal((await admitLearning({config,request:{learningTarget:'cpu'}},ops)).kind,'cpu');assert.equal(probes,0);
 const lease=await reserveDevice('pool',4,leaseDirectory,{instancePath:join(root,'deleted-pilot')});await lease.retain('remote:registered:job:unresolved');
 const bytes=await readFile(lease.receipt,'utf8'),input={config,request,resource,directory};
 const admitted=await admitLearning(input,ops);assert.equal(admitted.physicalDevice,5);assert.equal(await readFile(lease.receipt,'utf8'),bytes);
 const second=await reserveDevice('pool',5,leaseDirectory,{instancePath:join(root,'other-pilot')});
 const blocked=await admitLearning(input,ops);assert.equal(blocked.admitted,false);assert.equal(blocked.reason,'device-reservation-unresolved');assert.equal(blocked.reservations.length,2);
 await second.release();
});
test('unknown probe journal is reconciled by the existing environment before declaring closure',async t=>{
 const root=await fixture(t),directory=join(root,'probe');await mkdir(directory);
 const events=[{type:'execution-submitting',data:{submission:'one',kind:'run',accountingExcluded:true}},{type:'execution-submitted',data:{submission:'one',handle:'remote:target:job:probe'}}];
 await writeFile(join(directory,'environment-events.jsonl'),events.map(x=>JSON.stringify(x)).join('\n')+'\n');let closed=false,calls=0;
 const input={directory,config:{root,target:'a3',environment:{options:{}}},resource:{id:'slot'}};
 const ops={createEnvironment:async()=>({reconcileOperation:async({operation,events:retained})=>{calls++;assert.equal(operation,'one');assert.deepEqual(retained,events);return {closed};}})};
 assert.equal(await learningAdmissionSettled(input,ops),false);closed=true;assert.equal(await learningAdmissionSettled(input,ops),true);assert.equal(calls,2);
 await writeFile(join(directory,'environment-events.jsonl'),events.concat({type:'execution-state',data:{handle:'remote:target:job:probe',state:'stopped'}}).map(x=>JSON.stringify(x)).join('\n')+'\n');
 assert.equal(await learningAdmissionSettled(input,{createEnvironment:async()=>{throw Error('Closed probe must not redispatch');}}),true);
});
test('route refresh cannot silently change runtime identity',()=>{
 const child={target:'a3',environment:{options:{target:'registered',runtime:'old',deviceAffinity:[4,5],backgroundDevice:4}}};
 assert.equal(refreshLearningRoute(child,{target:'a3',environment:{options:{...child.environment.options,runtime:'new'}}}),null);
 assert.equal(refreshLearningRoute(child,{target:'a3',environment:{module:'different-binding',options:child.environment.options}}),null);
 assert.equal(refreshLearningRoute(child,{target:'a3',environment:{options:{...child.environment.options,backgroundDevice:5}}}).environment.options.backgroundDevice,5);
});
test('cleanup inventory protects missing leased paths, pending cleanup and active checkpoints',async t=>{
 const root=await fixture(t),leaseDirectory=join(root,'leases'),pilot=join(root,'deleted-pilot'),journal=join(root,'missing-control'),receiptFile=join(pilot,'receipt.json');
 const lease=await reserveDevice('pool',4,leaseDirectory,{instancePath:pilot});await lease.update({controlDirectory:journal,submissions:[{receiptFile}]});
 const queue=join(leaseDirectory,'.cleanup');await mkdir(queue);const db=new DatabaseSync(join(queue,'queue.sqlite'));db.exec('CREATE TABLE jobs(status TEXT,request TEXT)');db.prepare('INSERT INTO jobs VALUES (?,?)').run('queued',JSON.stringify({instancePath:join(root,'pending-cleanup'),journal:join(root,'pending.log')}));db.close();
 const deps=await cleanupDependencies({leaseDirectory,checkpoints:[{directory:join(root,'active-campaign'),paths:[join(root,'pending-transfer')]}]});
 const rows=cleanupCandidates([pilot,journal,join(root,'pending-cleanup'),join(root,'active-campaign'),join(root,'pending-transfer'),join(root,'unrelated'),root],deps);
 assert(rows.slice(0,5).every(x=>x.protectedBy.length));assert.equal(rows[5].protectedBy.length,0);assert(rows[6].protectedBy.length>=5);
 assert.equal(JSON.parse(await readFile(lease.receipt,'utf8')).controlDirectory,journal);
 await writeFile(lease.receipt,'{');await assert.rejects(cleanupDependencies({leaseDirectory}),error=>error.code==='CLEANUP_DEPENDENCY_UNKNOWN');
});
