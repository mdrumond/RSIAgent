import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm,access} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,relative} from 'node:path';
import {localHostIdentity,fileSha256,attestLocalReboot} from '../../controller/local-reboot.mjs';
import {createEnvironment} from '../../integrations/environments/index.mjs';
import {main} from '../../cli/main.mjs';
import {State} from '../../controller/state.mjs';
import {processIdentity,operationsStopped,attemptStopped} from '../../controller/lifecycle.mjs';

async function fixture(t,{native=false,exit}={}) {
 const root=await mkdtemp(join(tmpdir(),'local-reboot-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const host=await localHostIdentity(),events=[];
 const context={instancePath:root,event:async(type,data)=>events.push({type,data}),exclude:async(_label,fn)=>fn()};
 const env=await createEnvironment({mode:'local'},context);
 let handle='local:exec-legacy';
 if(native){const result=await env.execute('printf original',root,{timeout:3});handle=result.handle;assert.equal(result.exitCode,0);}
 const directory=join(root,'.executions',handle.slice(6));await mkdir(directory,{recursive:true});
 const path=join(directory,'execution.json');
 if(native) {
  const receipt=JSON.parse(await readFile(path));assert.equal(receipt.hostIdentity.fingerprint,host.fingerprint);
  await writeFile(path,JSON.stringify({...receipt,boot:'previous-boot',hostIdentity:{...host,boot:'previous-boot'}}));
 }else {
  await writeFile(path,JSON.stringify({boot:'previous-boot',cwd:root}));
  if(exit!==undefined)await writeFile(join(directory,'exit'),String(exit));
 }
 const evidence=join(root,'custody.json');await writeFile(evidence,JSON.stringify({previousTarget:'fixture',boot:'previous-boot'}));
 const input={format:1,kind:'operator-attested-same-host-reboot',previousBoot:'previous-boot',currentHost:host,
  operator:{sameHost:true,migrated:false,target:'fixture',previousTarget:'fixture',previousArchitecture:host.architecture,
   reason:'Operator verified unchanged host custody across the reboot'},
  evidence:[{path:evidence,sha256:await fileSha256(evidence)}],
  receipts:[{path:relative(root,path),sha256:await fileSha256(path)}]};
 return {root,directory,path,handle,host,input,env,events,context};
}

test('actual Pi completion needs explicit custody proof after boot change; receipts and result stay unchanged',async t=>{
 const f=await fixture(t,{native:true}),before=await readFile(f.path);
 assert.equal((await f.env.observe(f.handle)).state,'unknown');
 const preview=await attestLocalReboot(f.root,f.input);assert.equal(preview.applied,false);
 await assert.rejects(access(join(f.directory,'reboot-closure.json')),{code:'ENOENT'});
 assert.equal((await f.env.observe(f.handle)).state,'unknown');
 await attestLocalReboot(f.root,{...f.input,apply:true});
 const state=await f.env.observe(f.handle);
 assert.equal(state.state,'stopped');assert.equal(state.exitCode,0);assert.equal(state.executionSeconds,undefined);
 assert.equal(state.closure.kind,'operator-attested-same-host-reboot');
 assert.deepEqual(await readFile(f.path),before);
 const proof=await readFile(join(f.directory,'reboot-closure.json'));
 assert.equal((await attestLocalReboot(f.root,{...f.input,apply:true})).closures[0].reused,true);
 assert.deepEqual(await readFile(join(f.directory,'reboot-closure.json')),proof);
});

test('missing or empty exit never turns a reboot interruption into success or synthetic duration',async t=>{
 for(const exit of [undefined,'']) {
  const f=await fixture(t,{exit});await attestLocalReboot(f.root,{...f.input,apply:true});
  const state=await f.env.observe(f.handle);assert.equal(state.state,'stopped');assert.equal(state.exitCode,undefined);
  assert.equal(state.executionSeconds,undefined);assert.equal(state.outcome.reason,'host-reboot');
 }
});

for(const attested of [false,true])test(`old-boot scope stop only observes ${attested?'attested closure':'unknown custody'} and preserves the receipt`,async t=>{
 const f=await fixture(t),receipt=JSON.parse(await readFile(f.path));
 receipt.containment={unit:'rsi-command-11111111-1111-1111-1111-111111111111.scope',closed:false};
 await writeFile(f.path,JSON.stringify(receipt));f.input.receipts[0].sha256=await fileSha256(f.path);
 const before=await readFile(f.path);
 if(attested)await attestLocalReboot(f.root,{...f.input,apply:true});
 const sdk={createLocalBashOperations:()=>({exec:async()=>assert.fail('Old boot scope must not be controlled on this boot')})};
 const observer=await createEnvironment({mode:'local',localExecution:{tasks:16,memory_bytes:128*1024**2}},
  {...f.context,sdk});
 assert.equal((await observer.stop(f.handle)).state,attested?'stopped':'unknown');
 assert.deepEqual(await readFile(f.path),before);
 if(attested)assert.equal((await observer.observe(f.handle)).state,'stopped');
});

for(const scenario of ['migration','foreign boot','foreign host','foreign architecture','changed receipt','changed evidence','not a reboot'])
test('reconciliation refuses '+scenario+' without publishing closure',async t=>{
 const f=await fixture(t),input=structuredClone(f.input);input.apply=true;
 if(scenario==='migration')input.operator.previousTarget='source-machine';
 if(scenario==='foreign boot')input.previousBoot='source-boot';
 if(scenario==='foreign host') {
  await writeFile(f.path,JSON.stringify({boot:'previous-boot',cwd:f.root,hostIdentity:{...f.host,boot:'previous-boot',fingerprint:'f'.repeat(64)}}));
  input.receipts[0].sha256=await fileSha256(f.path);
 }
 if(scenario==='foreign architecture')input.operator.previousArchitecture='another-architecture';
 if(scenario==='changed receipt')await writeFile(f.path,JSON.stringify({boot:'previous-boot',cwd:'changed'}));
 if(scenario==='changed evidence')input.evidence[0].sha256='0'.repeat(64);
 if(scenario==='not a reboot')input.previousBoot=f.host.boot;
 await assert.rejects(attestLocalReboot(f.root,input));
 await assert.rejects(access(join(f.directory,'reboot-closure.json')),{code:'ENOENT'});
 assert.equal((await f.env.observe(f.handle)).state,'unknown');
});

test('a receipt edit or a proof copied from another host stays fenced',async t=>{
 const f=await fixture(t);await attestLocalReboot(f.root,{...f.input,apply:true});
 await writeFile(f.path,JSON.stringify({boot:'previous-boot',cwd:'modified'}));
 assert.equal((await f.env.observe(f.handle)).state,'unknown');
 const g=await fixture(t);await attestLocalReboot(g.root,{...g.input,apply:true});
 const path=join(g.directory,'reboot-closure.json'),proof=JSON.parse(await readFile(path));
 await writeFile(path,JSON.stringify({...proof,observedHost:{...proof.observedHost,fingerprint:'e'.repeat(64)}}));
 assert.equal((await g.env.observe(g.handle)).state,'unknown');
});

test('operator CLI retains attempt identity and elapsed while settling only excluded rebooted setup',async t=>{
 const f=await fixture(t),state=new State(join(f.root,'state.sqlite'));t.after(()=>state.close());
 state.add({id:'p',goal:'retained project',actor_seconds:60});
 const owner=await processIdentity(),attempt=state.admit('p',{...owner,start:'previous-process'});
 state.checkpoint(attempt.id,12.5,'original-session.jsonl');
 const emit=(type,data)=>state.event(attempt.id,type,data);
 emit('execution-starting',{operation:'probe',startElapsed:12.5});
 emit('execution-submitting',{operation:'probe',submission:'probe-submit',kind:'run',accountingExcluded:true});
 emit('execution-submitted',{operation:'probe',submission:'probe-submit',handle:f.handle,accountingExcluded:true});
 emit('execution-handle',{operation:'probe',handle:f.handle});
 const environment=await createEnvironment({mode:'local'},{...f.context,event:emit});
 assert.equal(await operationsStopped(state,state.attempt(attempt.id),environment),false);
 await writeFile(join(f.root,'instance.json'),JSON.stringify({format:1,environment:{options:{mode:'local',target:'fixture'}}}));
 const action=join(f.root,'reboot.json');await writeFile(action,JSON.stringify(f.input));
 const before=state.attempt(attempt.id);
 assert.equal((await main(['reconcile-local','--instance',f.root])).currentHost.fingerprint,f.host.fingerprint);
 const result=await main(['reconcile-local','--instance',f.root,'--config',action,'--apply']);
 assert.equal(result.applied,true);assert.deepEqual(state.attempt(attempt.id),before);
 assert.equal(await attemptStopped(state,state.attempt(attempt.id),environment),true);
 assert.equal(state.attempt(attempt.id).elapsed,12.5);assert.equal(state.attempt(attempt.id).outcome,null);
 assert.ok(state.events(null).some(e=>e.type==='local-reboot-reconciled'));
 const payloadEvents=state.events(attempt.id).filter(e=>e.type!=='execution-reconciled').map(e=>
  e.type==='execution-submitting'||e.type==='execution-submitted'?{...e,data:{...e.data,accountingExcluded:false}}:e);
 const unmeasured=await environment.reconcileOperation({operation:'probe',events:payloadEvents});
 assert.equal(unmeasured.closed,false);assert.equal(unmeasured.executionSeconds,undefined);
});

test('operator CLI refuses live writers and foreign target without changing receipts',async t=>{
 const f=await fixture(t),state=new State(join(f.root,'state.sqlite'));t.after(()=>state.close());
 state.add({id:'p',goal:'live',actor_seconds:30});state.admit('p',await processIdentity());
 await writeFile(join(f.root,'instance.json'),JSON.stringify({format:1,environment:{options:{mode:'local',target:'fixture'}}}));
 const action=join(f.root,'reboot.json');await writeFile(action,JSON.stringify(f.input));
 await assert.rejects(main(['reconcile-local','--instance',f.root,'--config',action,'--apply']),/stopped attempt writers/);
 await writeFile(action,JSON.stringify({...f.input,operator:{...f.input.operator,target:'foreign',previousTarget:'foreign'}}));
 await assert.rejects(main(['reconcile-local','--instance',f.root,'--config',action,'--apply']),/target differs/);
 await assert.rejects(access(join(f.directory,'reboot-closure.json')),{code:'ENOENT'});
});
