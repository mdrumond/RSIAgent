import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {DatabaseSync} from 'node:sqlite';
import {setTimeout as delay} from 'node:timers/promises';
import {processIdentity} from '../../controller/lifecycle.mjs';
import {reserveDevice} from '../../integrations/environments/device-leases.mjs';
import {queueDeviceCleanup,resumeDeviceCleanup,drainDeviceCleanup,cleanupStatus,campaignCleanupPending} from '../../integrations/environments/device-cleanup.mjs';
async function fixture(t){const directory=await mkdtemp(join(tmpdir(),'device-cleanup-'));t.after(()=>rm(directory,{recursive:true,force:true}));return directory;}
const request=(root,id,extra={})=>({id,token:id,receipt:join(root,id,'lease.json'),pointer:join(root,id+'.pointer'),instancePath:root,createdAt:Date.now(),deadline:Date.now()+2000,...extra});
async function until(predicate){for(let n=0;n<100;n++){if(await predicate())return;await delay(5);}throw Error('Expected cleanup progress');}
test('durable queue bounds concurrency and does not spawn a worker for every poll or job',async t=>{
 const root=await fixture(t);let launches=0,ticket;
 const launch=async(_directory,key)=>{launches++;ticket=key;return processIdentity();};
 for(const id of ['one','two','three'])await queueDeviceCleanup(root,request(root,id),{launch});
 await resumeDeviceCleanup(root,{launch});assert.equal(launches,1);
 let active=0,maximum=0;const gates=[];
 const draining=drainDeviceCleanup(root,ticket,{run:async row=>{active++;maximum=Math.max(active,maximum);await new Promise(resolve=>gates.push({id:row.id,resolve}));active--;return true;}});
 await until(()=>gates.length===2);assert.equal(active,2);gates[0].resolve();await until(()=>gates.length===3);assert.equal(active,2);gates[1].resolve();gates[2].resolve();await draining;
 assert.equal(maximum,2);assert.ok((await cleanupStatus(root)).every(row=>row.status==='closed'));
});
test('dead worker claim resumes exact cleanup identity with its original deadline',async t=>{
 const root=await fixture(t);let ticket;const launch=async(_dir,key)=>{ticket=key;return processIdentity();};
 const original=request(root,'same-token');await queueDeviceCleanup(root,original,{launch});
 const stopped={pid:2147483647,start:'gone',boot:'gone'},db=new DatabaseSync(join(root,'queue.sqlite'));
 db.prepare('UPDATE worker SET owner=?').run(JSON.stringify(stopped));db.prepare("UPDATE jobs SET status='running',owner=?").run(JSON.stringify(stopped));db.close();
 await resumeDeviceCleanup(root,{launch});let seen;
 await drainDeviceCleanup(root,ticket,{run:async row=>{seen=row;return true;}});
 assert.equal(seen.id,original.id);assert.equal(seen.deadline,original.deadline);assert.equal((await cleanupStatus(root))[0].status,'closed');
});
test('expired and failed cleanup never grant time or claim release',async t=>{
 const root=await fixture(t);let ticket;const launch=async(_dir,key)=>{ticket=key;return processIdentity();};
 const original=request(root,'expired',{deadline:Date.now()-1});await queueDeviceCleanup(root,original,{launch});
 await queueDeviceCleanup(root,{...original,deadline:Date.now()+10000},{launch});let calls=0;
 await drainDeviceCleanup(root,ticket,{run:async()=>{calls++;return true;}});assert.equal(calls,0);assert.equal((await cleanupStatus(root))[0].status,'expired');assert.equal((await cleanupStatus(root))[0].request.deadline,original.deadline);
 const failed=request(root,'failed',{deadline:Date.now()+30});await queueDeviceCleanup(root,failed,{launch});
 await drainDeviceCleanup(root,ticket,{run:async(_row,signal)=>{calls++;await new Promise(resolve=>signal.addEventListener('abort',resolve,{once:true}));return false;}});
 assert.equal(calls,1);assert.equal((await cleanupStatus(root)).find(row=>row.id==='failed').status,'expired');
});
test('campaign settlement is scoped to its own jobs despite a shared cleanup coordinator',async t=>{
 const root=await fixture(t),a=join(root,'campaign-a'),b=join(root,'campaign-b');await mkdir(a);await mkdir(b);let ticket;
 const launch=async(_dir,key)=>{ticket=key;return processIdentity();};
 await queueDeviceCleanup(root,request(root,'a',{campaignRoot:a}),{launch});await queueDeviceCleanup(root,request(root,'b',{campaignRoot:b}),{launch});
 assert.equal(await campaignCleanupPending(a),true);assert.equal(await campaignCleanupPending(b),true);
 const db=new DatabaseSync(join(root,'queue.sqlite'));db.prepare("UPDATE jobs SET status='closed' WHERE id='a'").run();db.close();
 assert.equal(await campaignCleanupPending(a),false);assert.equal(await campaignCleanupPending(b),true);
 await drainDeviceCleanup(root,ticket,{run:async()=>true});
});
test('dead handoff resumes while retaining a newer command pointer',async t=>{
 const root=await fixture(t),lease=await reserveDevice('pool',0,join(root,'leases')),activePointer=join(root,'active.json'),pointer=join(root,'old-pointer.json');
 await writeFile(pointer,JSON.stringify({receipt:lease.receipt,token:lease.token,pool:'pool',physicalDevice:0}));
 const newer={receipt:'new-receipt',token:'new-token'};await writeFile(activePointer,JSON.stringify(newer));let ticket;
 const original=request(root,lease.token,{receipt:lease.receipt,pointer,activePointer,handoffOwner:{pid:2147483647,start:'gone',boot:'gone'}});
 await queueDeviceCleanup(root,original,{launch:async()=>{throw Error('Must wait for handoff');}});
 await resumeDeviceCleanup(root,{launch:async(_dir,key)=>{ticket=key;return processIdentity();}});
 assert.deepEqual(JSON.parse(await readFile(activePointer,'utf8')),newer);let called=0;
 await drainDeviceCleanup(root,ticket,{run:async row=>{called++;assert.equal(row.pointer,pointer);return true;}});assert.equal(called,1);assert.deepEqual(JSON.parse(await readFile(activePointer,'utf8')),newer);
});
test('expired cleanup remains fenced without pretending that a stopped writer is alive',async t=>{
 const root=await fixture(t),campaign=join(root,'campaign');await mkdir(campaign);let ticket;
 await queueDeviceCleanup(root,request(root,'fenced',{campaignRoot:campaign,deadline:Date.now()-1}),{launch:async(_dir,key)=>{ticket=key;return processIdentity();}});
 await drainDeviceCleanup(root,ticket,{run:async()=>{throw Error('No elapsed allocation');}});
 assert.equal((await cleanupStatus(root))[0].status,'expired');assert.equal(await campaignCleanupPending(campaign),false);
});
test('owned cleanup coordinator survives its submitting process and closes through the existing CPL environment',{timeout:60000},async t=>{
 const {spawn}=await import('node:child_process'),{once}=await import('node:events'),{fileURLToPath}=await import('node:url');
 const root=await fixture(t),leases=join(root,'leases'),lease=await reserveDevice('fixture',0,leases),pointer=join(root,'cleanup-pointer.json'),instructionFile=join(root,'instructions.md');
 await writeFile(pointer,JSON.stringify({receipt:lease.receipt,token:lease.token,pool:'fixture',physicalDevice:0}));await writeFile(instructionFile,'Fixture only');
 const wire=join(root,'wire.json'),clientPath=join(root,'cpl-fixture'),observeGate=join(root,'observe-released');
 await writeFile(wire,'{}');await writeFile(clientPath,`#!/usr/bin/env python3
import json,sys,pathlib,shlex,uuid,time
args=sys.argv[1:];args.remove('--json');wire=pathlib.Path(${JSON.stringify(wire)})
kind=args[0];out={'state':'completed'}
if kind=='capabilities':out={'state':'available','backend':'remote-agent'}
if kind=='run':
 file=pathlib.Path(args[args.index('--file')+1]);handle='remote:fixture:job:'+str(uuid.uuid4())
 if file.name=='command.sh':
  c=json.loads(shlex.split(file.read_text())[3]);inner=json.loads(shlex.split(c['command'])[-1]);data=json.loads(wire.read_text())
  data['_controls']=data.get('_controls',0)+1;data['_control_handle']=handle
  data[c['spool']+'/admission.json']=json.dumps({'token':inner['lease']['token'],'decision':'reconciled'})
  data[c['spool']+'/timing.json']=json.dumps({'completed':True,'key':c['timingKey'],'executionSeconds':.01,'outcome':{'kind':'command'}})
  data[c['spool']+'/stdout']='';data[c['spool']+'/stderr']='';wire.write_text(json.dumps(data));time.sleep(.25)
 if '--receipt-file' in args:pathlib.Path(args[args.index('--receipt-file')+1]).write_text(json.dumps({'handle':handle}))
 out={'state':'queued','handle':handle}
if kind=='observe':
 if json.loads(wire.read_text()).get('_control_handle')==args[1]:
  while not pathlib.Path(${JSON.stringify(observeGate)}).exists():time.sleep(.02)
 out={'state':'completed','exit':0}
if kind=='logs':out={'state':'completed','content':''}
if kind=='download':
 data=json.loads(wire.read_text());value=data.get(args[2],next((v for k,v in data.items() if k.endswith('/'+args[2])),''));pathlib.Path(args[3]).write_text(value)
print(json.dumps(out))
`,{mode:0o700});
 await lease.update({cleanup:{id:lease.token}});
 const directory=join(leases,'.cleanup'),job=request(root,lease.token,{receipt:lease.receipt,pointer,leaseDirectory:leases,journal:join(root,'events.jsonl'),campaignRoot:root,deadline:Date.now()+45000,options:{mode:'cpl',target:'fixture',runtime:'cpu',instructionFile,clientPath,workspacePath:'/fixture',remoteDirectory:'owned',probeSeconds:1,health:{product:'a3'}}});
 const producer=join(root,'producer.mjs'),module=fileURLToPath(new URL('../../integrations/environments/device-cleanup.mjs',import.meta.url));
 await writeFile(producer,`import {queueDeviceCleanup} from ${JSON.stringify(module)};console.log(JSON.stringify(await queueDeviceCleanup(${JSON.stringify(directory)},${JSON.stringify(job)})));`);
 const producerChild=spawn(globalThis.process.execPath,[producer],{stdio:['ignore','pipe','pipe']});let stdout='',stderr='';producerChild.stdout.on('data',chunk=>stdout+=chunk);producerChild.stderr.on('data',chunk=>stderr+=chunk);const [code]=await once(producerChild,'exit');assert.equal(code,0,stderr);
 const submitted=JSON.parse(stdout.trim());assert.ok(submitted.owner?.pid);assert.notEqual(submitted.owner.pid,producerChild.pid);
 const {ownerStopped}=await import('../../controller/lifecycle.mjs');let cleanupOwner=submitted.owner;t.after(async()=>{if(!await ownerStopped(cleanupOwner))globalThis.process.kill(-cleanupOwner.pid,'SIGKILL');});
 assert.equal(await ownerStopped(submitted.owner),false);
 const until=async(check,phase)=>{
  while(Date.now()<job.deadline){if(await check())return;await delay(25);}
  const log=await readFile(join(directory,'worker.log'),'utf8').catch(error=>error.message);
  assert.fail(`${phase}: original fixture allocation exhausted\n${log}\n${JSON.stringify(await cleanupStatus(directory))}`);
 };
 // Block observation explicitly: slow platform setup cannot race natural closure
 // before the test has killed the first coordinator with its native ID retained.
 await until(async()=>Boolean(JSON.parse(await readFile(lease.receipt,'utf8')).controlHandle),'native handle persisted');
 globalThis.process.kill(-cleanupOwner.pid,'SIGKILL');await until(()=>ownerStopped(cleanupOwner),'first coordinator stopped');
 await writeFile(observeGate,'released');
 const resumed=await resumeDeviceCleanup(directory);cleanupOwner=resumed.owner;assert.notEqual(cleanupOwner.pid,submitted.owner.pid);
 assert.equal((await cleanupStatus(directory))[0].request.deadline,job.deadline);
 await until(async()=>(await cleanupStatus(directory))[0].status==='closed','resumed exact-handle cleanup closed');
 assert.equal(await campaignCleanupPending(root),false);
 const {closedDeviceLease}=await import('../../integrations/environments/device-leases.mjs');assert.ok(await closedDeviceLease(lease.receipt,lease.token));assert.equal(JSON.parse(await readFile(wire,'utf8'))._controls,1);
});
test('cleanup custody cannot race a live public lease reconciliation claim',async t=>{
 const {claimCleanupLease}=await import('../../integrations/environments/device-leases.mjs'),root=await fixture(t),lease=await reserveDevice('pool',0,join(root,'leases')),owner=await processIdentity();
 await lease.update({cleanup:{id:lease.token},reconciliation:{owner,id:'existing-recovery'}});
 assert.equal(await claimCleanupLease(lease.receipt,lease.token,owner),false);
 await lease.update({reconciliation:null});assert.equal(await claimCleanupLease(lease.receipt,lease.token,owner),true);
 const record=JSON.parse(await readFile(lease.receipt,'utf8'));assert.deepEqual(record.payloadOwner,owner);assert.deepEqual(record.owner,owner);
});
test('crash before pointer detachment is recovered only after the recorded handoff owner stops',async t=>{
 const root=await fixture(t),lease=await reserveDevice('pool',0,join(root,'leases')),activePointer=join(root,'active.json'),pointer=join(root,'cleanup.json'),identity=await processIdentity();
 const old={receipt:lease.receipt,token:lease.token,pool:'pool',physicalDevice:0};await writeFile(activePointer,JSON.stringify(old));await writeFile(pointer,JSON.stringify(old));
 const row=request(root,lease.token,{receipt:lease.receipt,pointer,activePointer,handoffOwner:identity});
 await queueDeviceCleanup(root,row);assert.equal((await cleanupStatus(root))[0].status,'handoff');
 assert.equal(await resumeDeviceCleanup(root),null);assert.deepEqual(JSON.parse(await readFile(activePointer,'utf8')),old);
 const db=new DatabaseSync(join(root,'queue.sqlite'));row.handoffOwner={pid:2147483647,start:'gone',boot:'gone'};db.prepare('UPDATE jobs SET request=?').run(JSON.stringify(row));db.close();
 let ticket;await resumeDeviceCleanup(root,{launch:async(_dir,key)=>{ticket=key;return processIdentity();}});await assert.rejects(readFile(activePointer),{code:'ENOENT'});
 assert.equal(JSON.parse(await readFile(lease.receipt,'utf8')).cleanup.id,lease.token);
 await drainDeviceCleanup(root,ticket,{run:async()=>true});
});
