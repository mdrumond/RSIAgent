import test from 'node:test';
import * as sdk from '@earendil-works/pi-coding-agent';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir,homedir} from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import {createEnvironment} from '../../integrations/environments/ascend.mjs';
import {createEnvironmentTools} from '../../integrations/environments/index.mjs';
import {State} from '../../controller/state.mjs';
import {ActorClock} from '../../controller/budget.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {setup,rows,until as driverUntil} from './driver_setup.mjs';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {reserveDevice} from '../../integrations/environments/device-leases.mjs';
import {attemptStopped,processIdentity,ownerStopped} from '../../controller/lifecycle.mjs';

const quote=s=>"'"+s.replaceAll("'","'\\''")+"'";
async function until(check){const end=performance.now()+10000;
  for(;;){if(await check())return;assert.ok(performance.now()<end,'Native milestone missing');await delay(10);}}
async function exists(path){try{await readFile(path);return true;}catch(error){if(error.code!=='ENOENT')throw error;return false;}}
async function fixture(t){
  const root=await mkdtemp(join(tmpdir(),'local-devices-')),bin=join(root,'bin'),source=join(root,'source');
  await mkdir(bin);await mkdir(source);const telemetry=join(root,'telemetry.json'),instructionFile=join(root,'instructions.md');
  const rows={0:{},1:{}};let enabled=[0,1],sequence=0;const instances=[],calls=[],closers=[];
  await writeFile(instructionFile,'Use leased execution.');
  const update=()=>writeFile(telemetry,JSON.stringify(rows));await update();
  await writeFile(join(bin,'npu-smi'),`#!/usr/bin/env python3
import json,sys,os,time
args=sys.argv; i=int(args[args.index('-i')+1]); q=args[args.index('-t')+1]
r=json.load(open(os.environ['LOCAL_HEALTH']))[str(i)]
if r.get('hang'): time.sleep(2)
u=r.get('memory',0)
text={'board':f'NPU ID: {i}\\nChip Name: Ascend950','health':'Health Status: '+r.get('status','OK')+'\\nError Code: 0',
'usages':f'HBM Capacity(MB): 1000\\nHBM Usage Rate(%): {u}\\nAicore Usage Rate(%): 0\\nAivector Usage Rate(%): 0\\nNPU Utilization(%): 0\\nAicube Usage Rate(%): 0',
'proc-mem':'Process id: 123' if r.get('busy') else 'No process in device.','current-fault-event':f'Fault Event Count: {r.get("faults",0)}'}
print(text[q])
`,{mode:0o700});
  const options={mode:'local',target:'local-fixture',runtime:'inherited-cpu',instructionFile,
    probeSeconds:10,probeObservationSeconds:10,env:{PATH:bin+':'+process.env.PATH,LOCAL_HEALTH:telemetry},
    health:{product:'a5',chip:null,query_timeout_seconds:.5}};
  async function bind(resource,extra={},path=join(root,'instance-'+sequence++)){
    const events=[],handles=new Set();let depth=0;
    const context={instancePath:path,resource,signal:extra.signal,...(extra.sdk?{sdk:extra.sdk}:{}),leaseDirectory:join(root,'leases'),
      client:async args=>{calls.push(args);assert.deepEqual(args,['npu','list','local-fixture']);
        return {state:'completed',profiles:[{target:'local-fixture',pool:root,devices:enabled.map(id=>({id,enabled:true}))}]};},
      exclude:async(label,fn)=>{if(label==='admission')return fn();depth++;try{return await fn();}finally{depth--;}},
      event:async(type,data)=>{
        events.push({type,data});await extra.onEvent?.(type,data);
        if(type==='execution-handle'){
          handles.add(data.handle);assert.match(data.handle,/^local:exec-/);
          assert.equal(depth>0,['resource-health','device-lease-control'].includes(data.purpose));
        }
      }};
    const environment=await createEnvironment({...options,...extra,health:{...options.health,...extra.health}},context);
    const f={environment,events,path,handles,context};instances.push(f);return f;
  }
  t.after(async()=>{
    try{for(const f of instances)for(const handle of f.handles){const state=await f.environment.observe(handle);
      if(state.state==='running')assert.equal((await f.environment.stop(handle)).state,'stopped');}}
    finally{try{for(const close of closers)await close();}finally{await rm(root,{recursive:true,force:true});}}
  });
  return {root,source,rows,options,calls,update,bind,onClose:fn=>closers.push(fn),setEnabled:ids=>{enabled=ids;}};
}

async function dead(pid){
  await until(async()=>{try{const stat=await readFile('/proc/'+pid+'/stat','utf8');return ['Z','X'].includes(stat.slice(stat.lastIndexOf(')')+2).split(' ')[0]);}catch(error){if(error.code==='ENOENT')return true;throw error;}});
}

test('two native local devices overlap with distinct masks and a third waits for the existing lease',{timeout:30000},async t=>{
  const f=await fixture(t),inventory=await f.bind(),resources=(await inventory.environment.resources()).filter(r=>r.available);
  assert.equal(resources.length,2);assert.notEqual(resources[0].id,resources[1].id);
  const state=new State(join(f.root,'state.sqlite'));f.onClose(()=>state.close());const queue=new RecoveryQueue(state);
  for(const id of ['a','b','c'])state.add({id,goal:'masks',source:f.source,actor_seconds:30,verifier:{files:[],command:'true'}});
  const owner=await processIdentity(),a=queue.claim('a',1,owner,resources[0].id),b=queue.claim('b',1,owner,resources[1].id);
  assert.throws(()=>queue.claim('c',1,owner,resources[0].id));
  assert.equal(state.project('c').status,'queued');assert.equal(state.db.prepare('SELECT count(*) AS n FROM resource_leases').get().n,2);
  const first=await f.bind(resources[0]),second=await f.bind(resources[1]);
  await first.environment.prepare({project:{source:f.source},attempt:a.attempt});
  await second.environment.prepare({project:{source:f.source},attempt:b.attempt});
  const code=`import os,time,json; from pathlib import Path; p=os.environ['ASCEND_PHYSICAL_DEVICE']; assert os.environ['ASCEND_RT_VISIBLE_DEVICES']==p==os.environ['ASCEND_VISIBLE_DEVICES']; Path(p+'.started').touch();\nwhile not Path(p+'.release').exists(): time.sleep(.01)\nprint(json.dumps({'physical':p,'logical':0}))`;
  let outA='',outB='',doneB=false;
  const runA=first.environment.execute('python3 -c '+quote(code),f.source,{timeout:20,onData:d=>outA+=d});
  const runB=second.environment.execute('python3 -c '+quote(code),f.source,{timeout:20,onData:d=>outB+=d}).then(r=>{doneB=true;return r;});
  void runA.catch(()=>{});void runB.catch(()=>{});
  await until(async()=>await exists(join(f.source,'0.started'))&&await exists(join(f.source,'1.started')));
  assert.equal(doneB,false);assert.equal(state.db.prepare('SELECT count(*) AS n FROM attempts').get().n,2);
  const selectedA=first.events.find(e=>e.type==='device-lease-acquired').data.physicalDevice;
  const selectedB=1-selectedA;
  await writeFile(join(f.source,selectedA+'.release'),'release');const resultA=await runA;
  assert.equal(resultA.exitCode,0);assert.equal((await first.environment.observe(resultA.handle)).state,'stopped');
  state.finish(a.attempt.id,'passed');state.db.prepare('DELETE FROM resource_leases WHERE attempt=?').run(a.attempt.id);
  const third=queue.claim('c',1,owner,resources[0].id),thirdEnv=await f.bind(resources[0]);
  await thirdEnv.environment.prepare({project:{source:f.source},attempt:third.attempt});
  let outC='';assert.equal((await thirdEnv.environment.execute('printf "%s" "$ASCEND_PHYSICAL_DEVICE"',f.source,{timeout:5,onData:d=>outC+=d})).exitCode,0);
  assert.equal(outC.trim().split('\n').at(-1),String(selectedA));assert.equal(doneB,false);
  await writeFile(join(f.source,selectedB+'.release'),'release');assert.equal((await runB).exitCode,0);
  assert.equal(JSON.parse(outA.trim().split('\n').at(-1)).physical,String(selectedA));assert.equal(JSON.parse(outB.trim().split('\n').at(-1)).physical,String(selectedB));
  assert.ok(f.calls.every(args=>args[0]==='npu'));
});

test('local health uses the real parser and bounded per-device queries; policy and timing stay strict',{timeout:60000},async t=>{
  const f=await fixture(t);Object.assign(f.rows,{1:{status:'ALARM'},2:{busy:true,memory:49},3:{faults:1},4:{hang:true},5:{}});
  f.setEnabled([0,1,2,3,4]);await f.update();
  const ordinary=await f.bind(),rows=await ordinary.environment.inspectDevices();
  assert.deepEqual(rows.filter(r=>r.available).map(r=>r.metadata.physicalDevice),[0]);
  assert.ok(rows.find(r=>r.metadata.physicalDevice===4).metadata.failure);
  const alarm=await f.bind(undefined,{health:{allow_alarm:true}});
  assert.deepEqual((await alarm.environment.inspectDevices()).filter(r=>r.available).map(r=>r.metadata.physicalDevice),[0,1]);
  assert.deepEqual((await alarm.environment.inspectDevices({purpose:'profiling'})).filter(r=>r.available).map(r=>r.metadata.physicalDevice),[0]);
  const resource=rows[0],bound=await f.bind(resource);f.setEnabled([]);
  await assert.rejects(bound.environment.execute('must-not-run',f.source),/No enabled/);
  assert.equal(bound.events.some(e=>e.type==='execution-handle'&&!e.data.purpose),false);
  assert.ok(f.calls.every(args=>args[0]==='npu'));
});

test('local command selects a new idle device after earlier attempt assignment',async t=>{
  const f=await fixture(t),inventory=await f.bind(),slot=(await inventory.environment.resources())[0];
  assert.equal(slot.env,undefined);assert.equal(slot.metadata.kind,'host-slot');
  const bound=await f.bind(slot);
  for(const physical of [0,1]){
    if(physical){f.rows[0].busy=true;await f.update();}
    let output='';
    const result=await bound.environment.execute('printf "%s" "$ASCEND_PHYSICAL_DEVICE"',f.source,{timeout:5,onData:d=>output+=d});
    assert.equal(result.exitCode,0);assert.equal(output,String(physical));
  }
  await assert.rejects(createEnvironment({...f.options,runtime:undefined},{}),/registered runtime/);
});

 test('no idle device queues without payload, cancellation retains positive probe closure',async t=>{
  const f=await fixture(t);f.rows[0].busy=f.rows[1].busy=true;await f.update();
  const abort=new AbortController(),bound=await f.bind(undefined,{signal:abort.signal});
  const pending=bound.environment.execute('touch forbidden-payload',f.source,{signal:abort.signal,timeout:5});
  void pending.catch(()=>{});
  await until(()=>bound.events.some(e=>e.type==='resource-probe-completed'));
  abort.abort();await assert.rejects(pending);
  assert.equal(await exists(join(f.source,'forbidden-payload')),false);
  assert.equal(bound.events.some(e=>e.type==='execution-handle'&&!e.data.purpose),false);
  for(const handle of bound.handles)assert.equal((await bound.environment.observe(handle)).state,'stopped');
});

test('STOP closes native local work; fresh binding observes the same handle without replay',{timeout:20000},async t=>{
  const f=await fixture(t),inventory=await f.bind(),row=(await inventory.environment.resources())[0],abort=new AbortController();
  const bound=await f.bind(row,{signal:abort.signal});let handle;
  const run=bound.environment.execute('python3 -c '+quote("from pathlib import Path; import time; Path('started').write_text(str(__import__('os').getpid())); time.sleep(30)"),f.source,{signal:abort.signal,timeout:10});
  void run.catch(()=>{});await until(()=>exists(join(f.source,'started')));
  handle=bound.events.find(e=>e.type==='execution-handle'&&!e.data.purpose).data.handle;abort.abort();
  assert.equal((await run).settled,true);
  const pid=Number(await readFile(join(f.source,'started'),'utf8'));await dead(pid);
  const reused=await f.bind(row,{},bound.path);
  assert.equal((await reused.environment.execute('true',f.source,{timeout:5})).settled,true);
  const resumed=await f.bind(row,{},bound.path);assert.equal((await resumed.environment.observe(handle)).state,'stopped');
  assert.equal(resumed.events.some(e=>e.type==='execution-handle'&&!e.data.purpose),false);
  assert.ok(bound.events.some(e=>e.type==='resource-probe-completed'&&e.data.accountingExcluded&&e.data.executionSeconds===0));
});

test('native last-moment health refusal returns infrastructure125 without executing the candidate',async t=>{
  const f=await fixture(t),bound=await f.bind(),event=bound.context.event;
  bound.context.event=async(type,data)=>{
    await event(type,data);
    if(type==='execution-handle'&&!data.purpose){f.rows[0].busy=true;await f.update();}
  };
  const result=await bound.environment.execute('touch forbidden-payload',f.source,{timeout:10});
  assert.equal(result.settled,true);assert.equal(result.exitCode,125);
  assert.equal(result.admission.decision,'health-refused');
  assert.deepEqual(result.outcome,{kind:'infrastructure',reason:'device-health-refused'});
  assert.equal(await exists(join(f.source,'forbidden-payload')),false);
});

test('local pending probe without native identity stays held on restart, never redispatched',async t=>{
  const f=await fixture(t),bound=await f.bind();
  const {readdir}=await import('node:fs/promises');const root=join(bound.path,'.ascend-resources');
  const [name]=await readdir(root),pending=join(root,name,'pending.json');
  await writeFile(pending,JSON.stringify({submission:'unknown',handle:null}));
  const restarted=await f.bind(undefined,{},bound.path);
  await assert.rejects(restarted.environment.inspectDevices(),/identity unknown/);
  assert.equal(restarted.events.some(e=>e.type==='execution-handle'),false);
  const state=new State(join(f.root,'held.sqlite'));f.onClose(()=>state.close());state.add({id:'held',goal:'proof',source:f.source,actor_seconds:30});
  const a=state.admit('held',await processIdentity());state.event(a.id,'execution-starting',{operation:'unknown'});
  for(const e of restarted.events)state.event(a.id,e.type,{...e.data,operation:'unknown'});
  assert.equal(await attemptStopped(state,{...a,owner:{pid:2147483647,start:'gone',boot:'gone'}},restarted.environment),false);
});

test('public CLI auto scheduler overlaps two masks and admits third only after finalizer exit',{timeout:180000},async t=>{
  let driver,gate,wave;
  t.after(async()=>{
    if(!driver)return;
    if(gate)await writeFile(join(gate,'release-all'),'release');
    await driver.run('stop');if(wave)await wave.output.catch(()=>{});
    await driverUntil(async()=>{for(const a of rows(driver.root,'SELECT owner FROM attempts'))if(!await ownerStopped(a.owner))return false;return true;},15000);
  });
  const health=await fixture(t);driver=await setup(t,'interrupt',['first','second','third'],{install:false});
  gate=join(driver.root,'barriers');await mkdir(gate);
  const module=join(driver.root,'machinery','local-devices-fixture.mjs');
  const payload=`import os,time,json; from pathlib import Path; root=Path(${JSON.stringify(gate)}); name=os.environ['FIXTURE_ATTEMPT']; p=os.environ['ASCEND_PHYSICAL_DEVICE']; assert p==os.environ['ASCEND_RT_VISIBLE_DEVICES']==os.environ['ASCEND_VISIBLE_DEVICES']; (root/(name+'.started')).write_text(json.dumps({'physical':p,'logical':0}));\nwhile not (root/(name+'.release')).exists() and not (root/'release-all').exists(): time.sleep(.01)`;
  await writeFile(module,`import {createEnvironment as base} from './integrations/environments/ascend.mjs';
import {createBashTool} from '@earendil-works/pi-coding-agent';
export async function createEnvironment(options,context){
 const environment=await base(options,{...context,leaseDirectory:${JSON.stringify(join(gate,'leases'))},client:async args=>{
  if(args[0]!=='npu')throw Error('Unexpected local transport operation');
  return {state:'completed',profiles:[{target:options.target,pool:${JSON.stringify(gate)},devices:[0,1].map(id=>({id,enabled:true}))}]};
 }});
 return {...environment,execute:(command,cwd,opts)=>environment.execute(command.includes('holding execution')?
 'python3 -c '+${JSON.stringify(quote(payload))}:command,cwd,{...opts,env:{...opts.env,FIXTURE_ATTEMPT:context.instancePath.split('/').at(-1)}})};
}
`);
  const config=JSON.parse(await readFile(join(driver.root,'instance.json'),'utf8'));
  const {stdout}=await promisify(execFile)(config.python,['-c','import sys;print(sys.executable)']);
  const python=stdout.trim(),wrapper=join(driver.root,'hold-finalizer');
  await writeFile(wrapper,`#!${python}
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(${JSON.stringify(gate)})
# Storage memory publication passes the actual MCP request on stdin.
if sys.argv[1:2]==['-c'] and 'from ascend_tools.kdb_client import invoke' in sys.argv[2]:
 data=sys.stdin.buffer.read()
 request=json.loads(data)
 if request.get('tool')=='stage_memory_revision' and (root/'hold-attempt').exists() and request.get('attempt')==(root/'hold-attempt').read_text():
  (root/'finalizer-entered').touch()
  while not (root/'finalizer-release').exists() and not (root/'release-all').exists(): time.sleep(.01)
 sys.exit(subprocess.run([${JSON.stringify(python)},*sys.argv[1:]],input=data).returncode)
os.execv(${JSON.stringify(python)},[${JSON.stringify(python)},*sys.argv[1:]])
`,{mode:0o700});
  const patch=join(driver.root,'local-config.json');
  await writeFile(patch,JSON.stringify({python:wrapper,concurrency:'auto',environment:{module:'machinery/local-devices-fixture.mjs',options:{...health.options,profile:config.target,
    env:{...health.options.env,ASCEND_PHYSICAL_DEVICE:'7',ASCEND_RT_VISIBLE_DEVICES:'7',ASCEND_VISIBLE_DEVICES:'7'}}}}));
  await driver.run('config','--config',patch);wave=driver.start('wave');void wave.output.catch(()=>{});
  await driverUntil(async()=>{
    const attempts=rows(driver.root,'SELECT * FROM attempts');
    return attempts.length===2&&(await Promise.all(attempts.map(a=>exists(join(gate,a.id+'.started'))))).every(Boolean);
  },30000);
  const pair=rows(driver.root,'SELECT * FROM attempts ORDER BY rowid');
  const masks=await Promise.all(pair.map(async a=>JSON.parse(await readFile(join(gate,a.id+'.started'),'utf8'))));
  assert.deepEqual(new Set(masks.map(m=>m.physical)),new Set(['0','1']));
  for(const a of pair){
    const job=JSON.parse(await readFile(join(driver.root,'attempts',a.id,'job.json'),'utf8'));
    assert.equal(job.resource.env,undefined);assert.equal(job.resource.metadata.kind,'host-slot');
  }
  assert.equal(rows(driver.root,'SELECT * FROM resource_leases').length,2);
  await writeFile(join(gate,'hold-attempt'),pair[0].id);await writeFile(join(gate,pair[0].id+'.release'),'release');
  await driverUntil(()=>exists(join(gate,'finalizer-entered')),30000);
  assert.equal(rows(driver.root,'SELECT * FROM attempts').length,2);
  assert.equal(await ownerStopped(rows(driver.root,`SELECT owner FROM attempts WHERE id='${pair[0].id}'`)[0].owner),false);
  assert.equal(rows(driver.root,'SELECT * FROM resource_leases').length,2);
  await writeFile(join(gate,'finalizer-release'),'release');
  await driverUntil(async()=>{const attempts=rows(driver.root,'SELECT * FROM attempts');
    return attempts.length===3&&await exists(join(gate,attempts.find(a=>a.project==='third').id+'.started'));},30000);
  const first=rows(driver.root,`SELECT * FROM attempts WHERE id='${pair[0].id}'`)[0],session=await readFile(first.session);
  assert.equal(await ownerStopped(first.owner),true);
  const third=rows(driver.root,"SELECT * FROM attempts WHERE project='third'")[0];
  assert.equal(JSON.parse(await readFile(join(gate,third.id+'.started'),'utf8')).physical,masks[0].physical);
  await writeFile(join(gate,'release-all'),'release');await wave.output;
  const finished=rows(driver.root,'SELECT * FROM attempts');assert.equal(finished.length,3);
  assert.ok(finished.every(a=>a.outcome==='passed'));assert.equal(rows(driver.root,'SELECT * FROM resource_leases').length,0);
  assert.deepEqual(await readFile(first.session),session);driver.pass();
});

test('native target tool filters inherited env but retains session identity and explicit overrides',{timeout:20000},async t=>{
  const f=await fixture(t),inventory=await f.bind(),row=(await inventory.environment.resources())[0];
  const runtime={...f.options.env,CANN_HOME:'configured-cann',LD_LIBRARY_PATH:'configured-libraries'};
  const bound=await f.bind(row,{env:runtime});let reached=false;
  const [tool]=await createEnvironmentTools({execute:(command,cwd,opts)=>{
    assert.equal(Object.hasOwn(opts.env,'PATH'),false);assert.equal(Object.hasOwn(opts.env,'PI_SESSION_FILE'),false);
    assert.equal(opts.env.PI_SESSION_ID,'native-session');reached=true;
    return bound.environment.execute(command,cwd,{...opts,env:{...opts.env,UNIQUE_CALL:'preserved'}});
  }},f.source);
  const code="import os,json; print(json.dumps({k:os.environ[k] for k in ['PATH','CANN_HOME','LD_LIBRARY_PATH','UNIQUE_CALL','ASCEND_PHYSICAL_DEVICE','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','PI_SESSION_ID']}))";
  const result=await tool.execute('native-env',{command:'python3 -c '+quote(code),timeout:10},undefined,undefined,
    {cwd:f.source,sessionManager:{getSessionId:()=> 'native-session',getSessionFile:()=> '/local/session.jsonl'}});
  assert.equal(reached,true);
  const value=JSON.parse(result.content.filter(c=>c.type==='text').map(c=>c.text).join('\n').trim().split('\n').at(-1));
  assert.equal(value.PATH,runtime.PATH);assert.equal(value.CANN_HOME,runtime.CANN_HOME);
  assert.equal(value.LD_LIBRARY_PATH,runtime.LD_LIBRARY_PATH);assert.equal(value.UNIQUE_CALL,'preserved');assert.equal(value.PI_SESSION_ID,'native-session');
  for(const key of ['ASCEND_PHYSICAL_DEVICE','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'])assert.equal(value[key],'0');
  let output='';const explicit=await bound.environment.execute('python3 -c '+quote(code),f.source,{timeout:10,
    env:{CANN_HOME:'explicit-cann',UNIQUE_CALL:'explicit',PI_SESSION_ID:'direct-session',ASCEND_PHYSICAL_DEVICE:'7'},onData:b=>output+=b});
  assert.equal(explicit.exitCode,0);const direct=JSON.parse(output.trim().split('\n').at(-1));
  assert.equal(direct.CANN_HOME,'explicit-cann');assert.equal(direct.UNIQUE_CALL,'explicit');assert.equal(direct.PI_SESSION_ID,'direct-session');
  assert.equal(direct.PATH,runtime.PATH);assert.equal(direct.ASCEND_PHYSICAL_DEVICE,'0');
});

for(const mode of ['timeout','ordinary-background'])test(`native ${mode} proves actual payload PID death before device reuse`,{timeout:20000},async t=>{
  const f=await fixture(t);f.setEnabled([0]);const bound=await f.bind();
  const command=mode==='timeout'?'python3 -c '+quote("import os,time; from pathlib import Path; Path('pid').write_text(str(os.getpid())); time.sleep(30)"):'sleep 30 & echo $! > pid';
  const result=await bound.environment.execute(command,f.source,{timeout:mode==='timeout'?1:5});
  assert.equal(result.settled,true);await dead(Number(await readFile(join(f.source,'pid'),'utf8')));
  assert.equal((await bound.environment.execute('true',f.source,{timeout:5})).settled,true);
});

test('target contention retries only structured nonadmission; payload exit 126 executes once',{timeout:20000},async t=>{
  const f=await fixture(t);f.setEnabled([0]);const first=await f.bind(),second=await f.bind();
  let now=0;const clock=new ActorClock(30,0,()=>now);
  second.environment=await createEnvironment(f.options,{...second.context,leaseDirectory:join(f.root,'different-controller'),
    exclude:(label,fn)=>clock.exclude(label,()=>second.context.exclude(label,async()=>{if(label==='admission')now+=1000;return fn();}))});
  const holding=first.environment.execute('touch started; while [ ! -f release ]; do sleep .01; done',f.source,{timeout:10});
  void holding.catch(()=>{});await until(()=>exists(join(f.source,'started')));
  const next=second.environment.execute('echo ran >> invocations; exit 126',f.source,{timeout:5});void next.catch(()=>{});
  await until(()=>second.events.filter(e=>e.type==='device-lease-acquired').length>=2);
  assert.equal(await exists(join(f.source,'invocations')),false);
  await writeFile(join(f.source,'release'),'go');const held=await holding;
  // A competing controller may briefly hold the target cleanup lock.
  // Reconcile the same completed handle before releasing its controller lease.
  await until(()=>first.environment.reconcileLease());
  assert.equal((await first.environment.observe(held.handle)).state,'stopped');
  const result=await next;assert.equal(result.exitCode,126);assert.equal(result.admission.decision,'admitted');
  assert.equal(await readFile(join(f.source,'invocations'),'utf8'),'ran\n');
  assert.equal(clock.elapsed,1,'confirmed contention refunded; genuine payload 126 charged');
});


test('closed cleanup contention retries after the other controller releases its fence',{timeout:30000},async t=>{
  const f=await fixture(t);f.setEnabled([0]);const first=await f.bind(),second=await f.bind();
  const prior=await reserveDevice(f.root,0,first.context.leaseDirectory,{instancePath:first.path});
  await writeFile(join(first.path,'.ascend-command-lease.json'),JSON.stringify({receipt:prior.receipt,token:prior.token,pool:f.root,physicalDevice:0}));
  second.environment=await createEnvironment(f.options,{...second.context,leaseDirectory:join(f.root,'other-controller')});
  const holding=second.environment.execute('touch newer-started; while [ ! -f newer-release ]; do sleep .01; done',f.source,{timeout:20});
  void holding.catch(()=>{});await until(()=>exists(join(f.source,'newer-started')));
  assert.equal(await first.environment.reconcileLease(),false);
  const controls=first.events.filter(e=>e.type==='execution-handle'&&e.data.purpose==='device-lease-control');
  assert.equal(controls.length,1);
  const receipt=JSON.parse(await readFile(join(prior.receipt,'..','control','.executions',controls[0].data.handle.slice(6),'admission.json'),'utf8'));
  assert.equal(receipt.token,prior.token);assert.equal(receipt.decision,'contended');
  assert.equal(JSON.parse(await readFile(prior.receipt,'utf8')).controlStarted,false);
  // A second refusal proves the first cleanup left the newer fence intact.
  assert.equal(await first.environment.reconcileLease(),false);
  await writeFile(join(f.source,'newer-release'),'go');assert.equal((await holding).exitCode,0);
  await until(()=>second.environment.reconcileLease());
  assert.equal(await first.environment.reconcileLease(),true);
  assert.equal(first.events.filter(e=>e.type==='execution-handle'&&e.data.purpose==='device-lease-control').length,3);
});


test('closed cleanup ownership mismatch preserves an orphan fence and retries after its closure',{timeout:20000},async t=>{
  const f=await fixture(t);const first=await f.bind();
  const prior=await reserveDevice(f.root,0,first.context.leaseDirectory,{instancePath:first.path});
  await writeFile(join(first.path,'.ascend-command-lease.json'),JSON.stringify({receipt:prior.receipt,token:prior.token,pool:f.root,physicalDevice:0}));
  const key=createHash('sha256').update(JSON.stringify([f.root,0])).digest('hex');
  const fence=join(homedir(),'.local/state/rsi/device-leases-host',key+'.held');await mkdir(fence,{recursive:true});
  t.after(async()=>{await rm(fence,{recursive:true,force:true});await rm(fence.replace(/\.held$/,'.lock'),{force:true});});
  const newer=JSON.stringify({pool:f.root,token:'newer-owner'});await writeFile(join(fence,'lease.json'),newer);
  assert.equal(await first.environment.reconcileLease(),false);
  assert.equal(await readFile(join(fence,'lease.json'),'utf8'),newer);
  const handle=first.events.find(e=>e.type==='execution-handle'&&e.data.purpose==='device-lease-control').data.handle;
  const receipt=JSON.parse(await readFile(join(prior.receipt,'..','control','.executions',handle.slice(6),'admission.json'),'utf8'));
  assert.equal(receipt.token,prior.token);assert.equal(receipt.decision,'ownership-mismatch');
  await rm(fence,{recursive:true});
  assert.equal(await first.environment.reconcileLease(),true);
});


test('local health cancellation after journal publication has positive closure and no payload',async t=>{
  const f=await fixture(t),abort=new AbortController();
  const bound=await f.bind(undefined,{signal:abort.signal,onEvent:(type,data)=>{
    if(type==='execution-handle'&&data.purpose==='resource-health')abort.abort();
  }});
  await bound.environment.prepare({project:{source:f.source},attempt:{id:'cancel-health'}});
  await assert.rejects(bound.environment.execute('touch payload-started',f.source,{timeout:5}),error=>error.closure?.closed===true);
  assert.equal(await exists(join(f.source,'payload-started')),false);
  const handle=bound.events.find(event=>event.type==='execution-handle'&&event.data.purpose==='resource-health').data.handle;
  assert.equal((await bound.environment.observe(handle)).state,'stopped');
});

test('proven unspawned local lease cleanup retries without replaying the device payload',async t=>{
  const f=await fixture(t),shell=join(f.root,'missing-shell-interpreter');
  await writeFile(shell,'#!/missing/interpreter\n',{mode:0o700});
  let failures=0;const native=sdk.createLocalBashOperations(),broken=sdk.createLocalBashOperations({shellPath:shell});
  const bridge={...sdk,createLocalBashOperations:()=>({exec:(command,cwd,options)=>{
    if(command.includes('/control/.executions/')&&failures++===0)return broken.exec(command,cwd,options);
    return native.exec(command,cwd,options);
  }})};
  const bound=await f.bind(undefined,{sdk:bridge});
  await bound.environment.prepare({project:{source:f.source},attempt:{id:'cleanup-prelaunch'}});
  const result=await bound.environment.execute("printf x >> payload-count",f.source,{timeout:10});
  assert.equal(result.settled,true);assert.equal(result.exitCode,0);
  await until(()=>bound.environment.reconcileLease());
  assert.ok(failures>=2);assert.equal(await readFile(join(f.source,'payload-count'),'utf8'),'x');
  assert.ok(bound.events.some(event=>event.type==='execution-error'&&event.data.code==='ENOENT'));
  assert.equal(bound.events.filter(event=>event.type==='device-lease-released').length,1);
});
