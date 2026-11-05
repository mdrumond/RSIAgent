import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawn} from 'node:child_process';
import {setTimeout as delay} from 'node:timers/promises';
import {reserveDevice,hostLease} from '../../integrations/environments/device-leases.mjs';
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
  function run(command){
    const child=spawn('python3',['-c',code,JSON.stringify({devices:[0],configuration:{},lease:{pool:'aliases'},command})],
      {env:{...process.env,HOME:directory},stdio:['ignore','pipe','pipe']});
    children.push(child);let error='';child.stderr.on('data',d=>error+=d);
    return {child,done:new Promise(resolve=>child.on('close',(status,signal)=>resolve({status,signal,error})))};
  }
  const first=run(`touch '${marker}'; while [ ! -e '${release}' ]; do sleep .01; done`);
  for(let n=0;;n++){try{await readFile(marker);break;}catch{}assert.ok(n<200);await delay(10);}
  assert.equal((await run('exit 0').done).status,126);
  await writeFile(release,'go');assert.equal((await first.done).status,0);
  assert.equal((await run('exit 0').done).status,0);
  // No payload descendants remain when killing this wrapper after completion of health.
  const killed=run('exec sleep 0.2');await delay(80);killed.child.kill('SIGKILL');
  await killed.done;await delay(220);
  assert.equal((await run('exit 0').done).status,126,'PID disappearance cannot reclaim the fence');
});
