import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import {createEnvironment} from '../../integrations/environments/ascend.mjs';
import {createEnvironmentTools} from '../../integrations/environments/index.mjs';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {setup,rows,until as driverUntil} from './driver_setup.mjs';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
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
    const context={instancePath:path,resource,signal:extra.signal,
      client:async args=>{calls.push(args);assert.deepEqual(args,['npu','list','local-fixture']);
        return {state:'completed',profiles:[{target:'local-fixture',devices:enabled.map(id=>({id,enabled:true}))}]};},
      exclude:async(_label,fn)=>{depth++;try{return await fn();}finally{depth--;}},
      event:async(type,data)=>{
        events.push({type,data});
        if(type==='execution-handle'){
          handles.add(data.handle);assert.match(data.handle,/^local:exec-/);
          assert.equal(depth>0,data.purpose==='resource-health');
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
  await writeFile(join(f.source,'0.release'),'release');const resultA=await runA;
  assert.equal(resultA.exitCode,0);assert.equal((await first.environment.observe(resultA.handle)).state,'stopped');
  state.finish(a.attempt.id,'passed');state.db.prepare('DELETE FROM resource_leases WHERE attempt=?').run(a.attempt.id);
  const third=queue.claim('c',1,owner,resources[0].id),thirdEnv=await f.bind(resources[0]);
  await thirdEnv.environment.prepare({project:{source:f.source},attempt:third.attempt});
  let outC='';assert.equal((await thirdEnv.environment.execute('printf "%s" "$ASCEND_PHYSICAL_DEVICE"',f.source,{timeout:5,onData:d=>outC+=d})).exitCode,0);
  assert.equal(outC.trim().split('\n').at(-1),'0');assert.equal(doneB,false);
  await writeFile(join(f.source,'1.release'),'release');assert.equal((await runB).exitCode,0);
  assert.equal(JSON.parse(outA.trim().split('\n').at(-1)).physical,'0');assert.equal(JSON.parse(outB.trim().split('\n').at(-1)).physical,'1');
  assert.ok(f.calls.every(args=>args[0]==='npu'));
});

test('local health uses the real parser and bounded per-device queries; policy and timing stay strict',{timeout:60000},async t=>{
  const f=await fixture(t);Object.assign(f.rows,{1:{status:'ALARM'},2:{busy:true,memory:49},3:{faults:1},4:{hang:true},5:{}});
  f.setEnabled([0,1,2,3,4]);await f.update();
  const ordinary=await f.bind(),rows=await ordinary.environment.resources();
  assert.deepEqual(rows.filter(r=>r.available).map(r=>r.metadata.physicalDevice),[0]);
  assert.ok(rows.find(r=>r.metadata.physicalDevice===4).metadata.failure);
  const alarm=await f.bind(undefined,{health:{allow_alarm:true}});
  assert.deepEqual((await alarm.environment.resources()).filter(r=>r.available).map(r=>r.metadata.physicalDevice),[0,1]);
  assert.deepEqual((await alarm.environment.resources({purpose:'profiling'})).filter(r=>r.available).map(r=>r.metadata.physicalDevice),[0]);
  const resource=rows[0],bound=await f.bind(resource);f.setEnabled([1]);
  await assert.rejects(bound.environment.execute('must-not-run',f.source),/not enabled/);
  assert.equal(bound.events.some(e=>e.type==='execution-handle'&&!e.data.purpose),false);
  assert.ok(f.calls.every(args=>args[0]==='npu'));
});

test('local pre-execution admission rechecks health and rejects raw physical IDs',{timeout:20000},async t=>{
  const f=await fixture(t),inventory=await f.bind(),row=(await inventory.environment.resources())[0],bound=await f.bind(row);
  f.rows[0].busy=true;await f.update();await assert.rejects(bound.environment.execute('must-not-run',f.source),/admission refused/);
  await assert.rejects(createEnvironment({...f.options,runtime:undefined},{}),/registered runtime/);
  const raw=await f.bind('0');await assert.rejects(raw.environment.execute('must-not-run',f.source),/not enabled/);
  assert.equal(bound.events.some(e=>e.type==='execution-handle'&&!e.data.purpose),false);
});

test('STOP closes native local work; fresh binding observes the same handle without replay',{timeout:20000},async t=>{
  const f=await fixture(t),inventory=await f.bind(),row=(await inventory.environment.resources())[0],abort=new AbortController();
  const bound=await f.bind(row,{signal:abort.signal});let handle;
  const run=bound.environment.execute('python3 -c '+quote("from pathlib import Path; import time; Path('started').touch(); time.sleep(30)"),f.source,{signal:abort.signal,timeout:10});
  void run.catch(()=>{});await until(()=>exists(join(f.source,'started')));
  handle=bound.events.find(e=>e.type==='execution-handle'&&!e.data.purpose).data.handle;abort.abort();
  assert.equal((await run).settled,true);
  const resumed=await f.bind(row,{},bound.path);assert.equal((await resumed.environment.observe(handle)).state,'stopped');
  assert.equal(resumed.events.some(e=>e.type==='execution-handle'&&!e.data.purpose),false);
  assert.ok(bound.events.some(e=>e.type==='resource-probe-completed'&&e.data.accountingExcluded&&e.data.executionSeconds===0));
});

test('local pending probe without native identity stays held on restart, never redispatched',async t=>{
  const f=await fixture(t),bound=await f.bind();
  const {readdir}=await import('node:fs/promises');const root=join(bound.path,'.ascend-resources');
  const [name]=await readdir(root),pending=join(root,name,'pending.json');
  await writeFile(pending,JSON.stringify({submission:'unknown',handle:null}));
  const restarted=await f.bind(undefined,{},bound.path);
  await assert.rejects(restarted.environment.resources(),/identity unknown/);
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
export async function createEnvironment(options,context){
 const environment=await base(options,{...context,client:async args=>{
  if(args[0]!=='npu')throw Error('Unexpected local transport operation');
  return {state:'completed',profiles:[{target:options.target,devices:[0,1].map(id=>({id,enabled:true}))}]};
 }});
 return {...environment,execute:(command,cwd,opts)=>environment.execute(command.includes('holding execution')?
 'python3 -c '+${JSON.stringify(quote(payload))}:command,cwd,{...opts,env:{...opts.env,FIXTURE_ATTEMPT:context.instancePath.split('/').at(-1)}})};
}
`);
  const config=JSON.parse(await readFile(join(driver.root,'instance.json'),'utf8'));
  const {stdout}=await promisify(execFile)(config.python,['-c','import sys;print(sys.executable)']);
  const python=stdout.trim(),wrapper=join(driver.root,'hold-finalizer');
  await writeFile(wrapper,`#!${python}
import os,sys,time
from pathlib import Path
root=Path(${JSON.stringify(gate)})
if sys.argv[1:3]==['-m','ascend_tools.kdb_client'] and '--attempt' in sys.argv and (root/'hold-attempt').exists() and sys.argv[sys.argv.index('--attempt')+1]==(root/'hold-attempt').read_text():
 (root/'finalizer-entered').touch()
 while not (root/'finalizer-release').exists() and not (root/'release-all').exists(): time.sleep(.01)
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

test('native target tool omits implicit spawn env while direct API retains explicit overrides',{timeout:20000},async t=>{
  const f=await fixture(t),inventory=await f.bind(),row=(await inventory.environment.resources())[0];
  const runtime={...f.options.env,CANN_HOME:'configured-cann',LD_LIBRARY_PATH:'configured-libraries'};
  const bound=await f.bind(row,{env:runtime});let reached=false;
  const [tool]=await createEnvironmentTools({execute:(command,cwd,opts)=>{
    assert.equal(Object.hasOwn(opts,'env'),false);reached=true;
    return bound.environment.execute(command,cwd,{...opts,env:{UNIQUE_CALL:'preserved'}});
  }},f.source);
  const code="import os,json; print(json.dumps({k:os.environ[k] for k in ['PATH','CANN_HOME','LD_LIBRARY_PATH','UNIQUE_CALL','ASCEND_PHYSICAL_DEVICE','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES']}))";
  const result=await tool.execute('native-env',{command:'python3 -c '+quote(code),timeout:10});
  assert.equal(reached,true);
  const value=JSON.parse(result.content.filter(c=>c.type==='text').map(c=>c.text).join('\n').trim().split('\n').at(-1));
  assert.equal(value.PATH,runtime.PATH);assert.equal(value.CANN_HOME,runtime.CANN_HOME);
  assert.equal(value.LD_LIBRARY_PATH,runtime.LD_LIBRARY_PATH);assert.equal(value.UNIQUE_CALL,'preserved');
  for(const key of ['ASCEND_PHYSICAL_DEVICE','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'])assert.equal(value[key],'0');
  let output='';const explicit=await bound.environment.execute('python3 -c '+quote(code),f.source,{timeout:10,
    env:{CANN_HOME:'explicit-cann',UNIQUE_CALL:'explicit',ASCEND_PHYSICAL_DEVICE:'7'},onData:b=>output+=b});
  assert.equal(explicit.exitCode,0);const direct=JSON.parse(output.trim().split('\n').at(-1));
  assert.equal(direct.CANN_HOME,'explicit-cann');assert.equal(direct.UNIQUE_CALL,'explicit');
  assert.equal(direct.PATH,runtime.PATH);assert.equal(direct.ASCEND_PHYSICAL_DEVICE,'0');
});
