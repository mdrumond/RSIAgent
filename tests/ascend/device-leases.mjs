import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,rm,mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawn} from 'node:child_process';
import {setTimeout as delay} from 'node:timers/promises';
import {reserveDevice,reconcileDeviceLease,hostLease,hostReconcile,closeDeviceLease,closedDeviceLease} from '../../integrations/environments/device-leases.mjs';
import {ActorClock} from '../../controller/budget.mjs';

async function root(t){const path=await mkdtemp(join(tmpdir(),'command-leases-'));t.after(()=>rm(path,{recursive:true,force:true}));return path;}
test('atomic pool/physical reservation survives owner loss and preserves handles',async t=>{
  const directory=await root(t);
  const results=await Promise.all(Array.from({length:8},()=>reserveDevice('aliases',2,directory)));
  const leases=results.filter(Boolean);assert.equal(leases.length,1);
  const lease=leases[0];await lease.retain('remote:alias:job:one');
  const receipt=JSON.parse(await readFile(lease.receipt,'utf8'));
  assert.deepEqual(receipt.handles,['remote:alias:job:one']);
  assert.equal(await reserveDevice('aliases',2,directory),null);
  const other=await reserveDevice('other-pool',2,directory);assert.ok(other);await other.release();
  await lease.release();assert.ok(await reserveDevice('aliases',2,directory));
});

test('only explicit queue-wait pauses elapsed time including nested exclusions',async()=>{
  let now=0;const clock=new ActorClock(10,2,()=>now);
  now=1000;assert.equal(clock.elapsed,3);
  await clock.exclude('queue-wait',async()=>{
    now+=20000;await clock.exclude('setup',async()=>{now+=5000;});
    assert.equal(clock.elapsed,3);
  });
  now+=1000;assert.equal(clock.elapsed,4);assert.equal(clock.remaining,6);
  await assert.rejects(clock.exclude('payload',async()=>{}));
});

// Separate processes model controller hosts with no shared controller state.
// Only the target HOME and OS lock are shared. No CPL/network/NPU involved.
test('target lock excludes a second controller; killed owner leaves durable fence',{timeout:10000},async t=>{
  const directory=await root(t),marker=join(directory,'started'),release=join(directory,'release');
  const preamble=`import os,sys,json,hashlib,time\nc=json.loads(sys.argv[1]); deadline=time.monotonic()+10\nns={'local_probe':lambda **kw:{'admitted':True,'measurements':{'idle':True}}}\n`;
  const code=preamble+hostLease,children=[];
  t.after(()=>{for(const child of children)child.kill('SIGKILL');});
  function run(command,action=hostLease,token='owner'){
    const child=spawn('python3',['-c',preamble+action,JSON.stringify({devices:[0],configuration:{},lease:{pool:'aliases',token},command})],
      {env:{...process.env,HOME:directory,RSI_LEASE_RECEIPT:join(directory,'receipt-'+children.length)},stdio:['ignore','pipe','pipe']});
    children.push(child);let error='';child.stderr.on('data',d=>error+=d);
    return {child,done:new Promise(resolve=>child.on('close',(status,signal)=>resolve({status,signal,error})))};
  }
  const first=run(`touch '${marker}'; while [ ! -e '${release}' ]; do sleep .01; done`);
  for(let n=0;;n++){try{await readFile(marker);break;}catch{}assert.ok(n<200);await delay(10);}
  assert.equal(JSON.parse(await readFile(join(directory,'receipt-0'),'utf8')).released,undefined);
  assert.equal((await run('exit 0').done).status,126);
  await writeFile(release,'go');assert.equal((await first.done).status,0);
  assert.equal(JSON.parse(await readFile(join(directory,'receipt-0'),'utf8')).released,true);
  assert.equal((await run('exit 0').done).status,0);
  // No payload descendants remain when killing this wrapper after completion of health.
  const killed=run('exec sleep 0.2');await delay(80);killed.child.kill('SIGKILL');
  await killed.done;await delay(220);
  assert.equal((await run('exit 0').done).status,126,'PID disappearance cannot reclaim the fence');
  assert.equal((await run('',hostReconcile,'wrong-owner').done).status,125);
  assert.equal((await run('',hostReconcile).done).status,0);
  assert.equal((await run('exit 126').done).status,126);
  assert.deepEqual(JSON.parse(await readFile(join(directory,'receipt-'+(children.length-1)),'utf8')),{token:'owner',decision:'admitted',released:true});
});

test('admission holds unpublished time then charges only nonqueue work',async()=>{
  let now=0;const clock=new ActorClock(30,0,()=>now);
  await clock.exclude('admission',async()=>{now+=1000;return {queueWait:false};});
  assert.equal(clock.elapsed,1);
  await clock.exclude('admission',async()=>{
    now+=2000;await clock.exclude('transfer',async()=>{now+=4000;});return {queueWait:true};
  });
  assert.equal(clock.elapsed,1);
  await assert.rejects(clock.exclude('admission',async()=>{now+=1000;throw new Error('unknown');}));
  assert.equal(clock.elapsed,2);
});

test('a stopped owner is reclaimed only after every native identity and host fence close',async t=>{
  const directory=await root(t),lease=await reserveDevice('pool',0,directory,{instancePath:'/old/deleted/attempt'});
  await lease.update({owner:{...lease.record.owner,start:'old-process'},handles:['remote:target:job:payload'],
    submissions:[{submission:'copy',handle:'remote:target:transfer:copy'}]});
  const calls=[];
  const proof=await reconcileDeviceLease(lease.receipt,{
    observe:async h=>{calls.push(h);return {state:'completed'};},
    reconcileHost:async held=>{assert.equal(held.token,lease.token);calls.push('fence');return true;}
  });
  assert.equal(proof.closed,true);assert.deepEqual(calls,['remote:target:job:payload','remote:target:transfer:copy','fence']);
  await assert.rejects(readFile(lease.receipt),{code:'ENOENT'});
  assert.ok(await reserveDevice('pool',0,directory));
});

test('an unidentified excluded output download remains recorded after positive device closure',async t=>{
  const directory=await root(t),lease=await reserveDevice('pool',0,directory);
  const artifact={submission:'lost-output-download',kind:'download',accountingExcluded:true,receiptFile:null};
  await lease.update({owner:{...lease.record.owner,start:'old-process'},handles:['remote:target:job:payload']});
  await lease.event('execution-submitting',artifact);
  const observed=[];
  const result=await reconcileDeviceLease(lease.receipt,{
    observe:async handle=>{observed.push(handle);return {state:'completed'};},reconcileHost:async()=>true
  });
  assert.equal(result.closed,true);assert.deepEqual(observed,['remote:target:job:payload']);
  assert.deepEqual(result.unresolvedArtifacts,[artifact]);
  const proof=await closedDeviceLease(lease.receipt,lease.token);
  assert.deepEqual(proof.unresolvedArtifacts,[artifact]);
  const archived=JSON.parse(await readFile(join(proof.path,'lease.json'),'utf8'));
  assert.deepEqual(archived.submissions,[artifact]);
  assert.equal(archived.submissions[0].closed,undefined,'the transfer was never declared closed');
  assert.ok(await reserveDevice('pool',0,directory));
});

for(const row of [{kind:'run',accountingExcluded:true},{kind:'upload',accountingExcluded:true},{kind:'download',accountingExcluded:false}])test(`unknown ${row.kind}/${row.accountingExcluded} cannot be treated as an output-only artifact`,async t=>{
  const directory=await root(t),lease=await reserveDevice('pool',0,directory);
  await lease.update({owner:{...lease.record.owner,start:'old-process'}});
  await lease.event('execution-submitting',{submission:'unknown',...row});
  let fenced=0;
  const result=await reconcileDeviceLease(lease.receipt,{observe:async()=>({state:'completed'}),reconcileHost:async()=>{fenced++;return true;}});
  assert.equal(result.closed,false);assert.equal(result.reason,'submission-unknown');assert.equal(fenced,0);
  assert.equal(JSON.parse(await readFile(lease.receipt,'utf8')).token,lease.token);
});

for(const scenario of ['live-owner','legacy-owner','live-job','unknown-submission','unknown-fence'])test(`reservation reconciliation preserves ${scenario}`,async t=>{
  const directory=await root(t),lease=await reserveDevice('pool',0,directory);
  if(scenario!=='live-owner')await lease.update({owner:scenario==='legacy-owner'?null:{...lease.record.owner,start:'old-process'}});
  if(scenario==='live-job')await lease.retain('remote:target:job:live');
  if(scenario==='unknown-submission')await lease.event('execution-submitting',{submission:'lost',receiptFile:join(directory,'deleted-receipt')});
  let fences=0;
  const proof=await reconcileDeviceLease(lease.receipt,{
    observe:async()=>({state:'running'}),reconcileHost:async()=>{fences++;return false;}
  });
  assert.equal(proof.closed,false);await readFile(lease.receipt);
  assert.equal(fences,scenario==='unknown-fence'?1:0);
  assert.equal(await reserveDevice('pool',0,directory),null);
});

test('legacy owner closure requires explicit independent proof, then normal remote proofs still apply',async t=>{
  const directory=await root(t),lease=await reserveDevice('pool',0,directory);
  await lease.update({owner:null,handles:['remote:target:job:old']});
  let closed=false;
  const callbacks={stopped:async(_owner,record)=>record.token===lease.token,
    observe:async()=>({state:closed?'completed':'observation-unavailable'}),reconcileHost:async()=>true};
  assert.equal((await reconcileDeviceLease(lease.receipt,callbacks)).reason,'execution-unresolved');
  closed=true;assert.equal((await reconcileDeviceLease(lease.receipt,callbacks)).closed,true);
});

test('concurrent reconcilers cannot release or replay another live reconciliation',async t=>{
  const directory=await root(t),lease=await reserveDevice('pool',0,directory);
  await lease.update({owner:{...lease.record.owner,start:'old-process'},handles:['remote:target:job:old']});
  let started,finish;const entered=new Promise(ok=>started=ok),gate=new Promise(ok=>finish=ok);let fences=0;
  const callbacks={observe:async()=>{started();await gate;return {state:'completed'};},reconcileHost:async()=>{fences++;return true;}};
  const first=reconcileDeviceLease(lease.receipt,callbacks);await entered;
  assert.equal((await reconcileDeviceLease(lease.receipt,callbacks)).reason,'reconciliation-active');
  finish();assert.equal((await first).closed,true);assert.equal(fences,1);
});

for(const legacy of [false,true])test(`peer closure during host reconciliation completes the same ${legacy?'legacy':'stable'} token`,async t=>{
  const directory=await root(t),lease=await reserveDevice('pool',0,directory);
  const record=JSON.parse(await readFile(lease.receipt,'utf8'));
  if(legacy){delete record.controlDirectory;await writeFile(lease.receipt,JSON.stringify(record));}
  const control=legacy?join(lease.receipt,'..','control'):record.controlDirectory;
  await mkdir(join(control,'.executions','exec-old'),{recursive:true});
  await writeFile(join(control,'.executions','exec-old','stdout'),'retained output');
  const result=await reconcileDeviceLease(lease.receipt,{
    stopped:async()=>true,observe:async()=>({state:'completed'}),
    reconcileHost:async held=>{await closeDeviceLease(held);return false;}
  });
  assert.equal(result.closed,true);
  const proof=await closedDeviceLease(lease.receipt,lease.token);
  const journal=legacy?join(proof.path,'control'):proof.controlDirectory;
  assert.equal(await readFile(join(journal,'.executions','exec-old','stdout'),'utf8'),'retained output');
  assert.ok(await reserveDevice('pool',0,directory));
});

test('peer closure during native observation is positive proof rather than a missing receipt error',async t=>{
  const directory=await root(t),lease=await reserveDevice('pool',0,directory);
  await lease.retain('remote:target:job:payload');
  const result=await reconcileDeviceLease(lease.receipt,{
    stopped:async()=>true,
    observe:async()=>{await closeDeviceLease(lease);return {state:'completed'};},
    reconcileHost:async held=>{await held.update({controlStarted:true});return true;}
  });
  assert.equal(result.closed,true);assert.ok(await reserveDevice('pool',0,directory));
});
