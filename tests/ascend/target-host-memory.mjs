import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {hostMemorySource, hostMemoryConfiguration, requireHostMemory} from '../../integrations/environments/host-memory.mjs';

const python = `
from types import SimpleNamespace
files={'/proc/self/cgroup':'0::/user.slice/user-1000.slice/user@1000.service/rsi.slice/job-a',
 '/proc/self/mountinfo':'1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw',
 '/proc/sys/kernel/random/boot_id':'boot',
 '/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/rsi.slice/memory.max':'4096',
 '/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/rsi.slice/memory.current':'123'}
def read(p):
 if p not in files: raise FileNotFoundError(p)
 return files[p]
def proof(config={'max_bytes':4096}):
 return host_memory_proof(config,read,lambda p:SimpleNamespace(st_dev=1,st_ino=2))
a=proof(); assert a['available'] and a['current_bytes']==123
files['/proc/self/cgroup']='0::/user.slice/user-1000.slice/user@1000.service/rsi.slice/job-b'
assert proof()['fingerprint']==a['fingerprint']
assert not proof({'max_bytes':4096,'fingerprint':'wrong'})['available']
for maximum in ['max','8192','0','garbage']:
 files['/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/rsi.slice/memory.max']=maximum
 assert not proof()['available']
files['/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/rsi.slice/memory.max']='2048'
assert proof()['available']
files['/proc/self/cgroup']='0::/job-2G/rsi.slice'
assert not proof()['available']
files['/proc/self/cgroup']='0::/job-2G'
files['/sys/fs/cgroup/job-2G/memory.max']='2048'
assert not proof()['available']
custom='/user.slice/user-1000.slice/user@1000.service/rsiclitest123.slice'
files['/proc/self/cgroup']='0::'+custom+'/actor.scope'
files['/sys/fs/cgroup'+custom+'/memory.max']=str(32*1024**3)
files['/sys/fs/cgroup'+custom+'/memory.current']='123'
assert proof({'max_bytes':32*1024**3,'slice':'rsiclitest123.slice'})['available']
assert not proof({'max_bytes':16*1024**3,'slice':'rsiclitest123.slice'})['available']
files['/proc/self/cgroup']='0::/jobs/rsiclitest123.slice/actor.scope'
files['/sys/fs/cgroup/jobs/rsiclitest123.slice/memory.max']=str(32*1024**3)
files['/sys/fs/cgroup/jobs/rsiclitest123.slice/memory.current']='123'
assert not proof({'max_bytes':32*1024**3,'slice':'rsiclitest123.slice'})['available']
files['/proc/self/cgroup']='0::/containers/shared/job'
files['/proc/self/mountinfo']='1 0 0:1 /containers/shared /sys/fs/cgroup rw - cgroup2 cgroup rw'
files['/sys/fs/cgroup/memory.max']='4096'; files['/sys/fs/cgroup/memory.current']='44'
assert proof({'max_bytes':4096,'container_root':'/containers/shared'})['available']
assert not proof({'max_bytes':4096,'container_root':'/containers/other'})['available']
files['/sys/fs/cgroup/memory.max']='max'
assert not proof({'max_bytes':4096,'container_root':'/containers/shared'})['available']
print(json.dumps(a))
`;
test('portable kernel reader verifies shared aggregates and rejects unlimited/job-only/context mismatch', () => {
  const proof=JSON.parse(execFileSync('python3',['-c',hostMemorySource+python],{timeout:3000,encoding:'utf8'}));
  assert.equal(requireHostMemory(proof,hostMemoryConfiguration({max_bytes:4096})),proof);
  for (const slice of ['../fake.slice','/rsi.slice','job/owned.slice','rsi.service',''])
    assert.throws(()=>hostMemoryConfiguration({slice}));
  for (const bad of [undefined,{...proof,max_bytes:8192},{...proof,fingerprint:'bad'},{...proof,current_bytes:-1}])
    assert.throws(()=>requireHostMemory(bad,{max_bytes:4096}),{code:'MEMORY_ENFORCEMENT_UNAVAILABLE'});
});
test('real local cgroup inspection is bounded and never claims enforcement without a shared scope', () => {
  const proof=JSON.parse(execFileSync('python3',['-c',hostMemorySource+'\nprint(json.dumps(host_memory_proof({"max_bytes":17179869184})))'],{timeout:3000,encoding:'utf8'}));
  if(proof.available) assert.ok(proof.kind==='rsi.slice' && proof.max_bytes<=17179869184);
  else assert.equal(proof.code,'MEMORY_ENFORCEMENT_UNAVAILABLE');
});

test('resources use existing CPL snapshot receipts and fail closed before admission', async t => {
  const {mkdtemp,writeFile,rm}=await import('node:fs/promises');
  const {tmpdir}=await import('node:os');
  const {join}=await import('node:path');
  const {createEnvironment}=await import('../../integrations/environments/ascend.mjs');
  const root=await mkdtemp(join(tmpdir(),'memory-snapshot-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const instructionFile=join(root,'instructions');await writeFile(instructionFile,'CPU fixture');
  let proof={available:true,kind:'rsi.slice',fingerprint:'a'.repeat(64),max_bytes:4096,current_bytes:42};
  const calls=[];
  const environment=await createEnvironment({mode:'cpl',target:'fake',runtime:'registered',instructionFile,
    health:{product:'a3'},host_memory:{max_bytes:4096}}, {instancePath:root,
    sdk:{createLocalBashOperations:()=>({})},exclude:async(_,fn)=>fn(),event:async()=>{},client:async args=>{
      calls.push(args);
      if(args[0]==='npu') return {state:'completed',profiles:[{target:'fake',pool:'shared',devices:[{id:0,enabled:true}]}]};
      if(args[0]==='run') return {handle:'remote:fake:job:1'};
      if(args[0]==='observe') return {state:'completed',exit:0};
      if(args[0]==='logs') return {content:JSON.stringify({devices:[],host_memory:proof})};
      assert.fail('Unexpected operation '+args[0]);
    }});
  assert.equal((await environment.resources())[0].metadata.host_memory.current_bytes,42);
  for (const next of [undefined,{...proof,max_bytes:8192},{...proof,fingerprint:'b'.repeat(64)}]) {
    proof=next;
    await assert.rejects(environment.resources(),{code:'MEMORY_ENFORCEMENT_UNAVAILABLE'});
  }
  assert.equal(calls.filter(args=>args[0]==='run').length,4);
});

test('only local mode inherits managed host policy; explicit target policy takes precedence', async t => {
  const {mkdtemp,writeFile,rm}=await import('node:fs/promises');
  const {tmpdir}=await import('node:os');
  const {join}=await import('node:path');
  const {createEnvironment}=await import('../../integrations/environments/ascend.mjs');
  const root=await mkdtemp(join(tmpdir(),'memory-policy-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const instructionFile=join(root,'instructions');await writeFile(instructionFile,'CPU fixture');
  const injectedSource=`def host_memory_proof(configuration):
 return {'available':True,'kind':'rsi.slice','scope':'/'+configuration['slice'],'fingerprint':'${'a'.repeat(64)}','max_bytes':configuration['max_bytes'],'current_bytes':123}
`;
  for (const [mode,policy,expectedCap,expectedSlice] of [
    ['local',undefined,32*1024**3,'rsiclitest123.slice'],
    ['local',{max_bytes:4096,slice:'explicit.slice'},4096,'explicit.slice'],
    ['cpl',undefined,16*1024**3,'rsi.slice'],
  ]) {
    let output;
    const environment=await createEnvironment({mode,target:'fake',runtime:'registered',instructionFile,
      health:{product:'a3'},host_memory:policy}, {instancePath:join(root,mode+expectedCap),
      hostMemory:{max_bytes:32*1024**3,slice:'rsiclitest123.slice'},hostMemorySource:injectedSource,
      exclude:async(_,fn)=>fn(),event:async()=>{},client:async args=>{
        if(args[0]==='npu') return {state:'completed',profiles:[{target:'fake',pool:'shared',devices:[{id:0,enabled:true}]}]};
        if(args[0]==='run') {
          output=execFileSync('bash',[args[args.indexOf('--file')+1]],{timeout:3000,encoding:'utf8'});
          return {handle:'remote:fake:job:1'};
        }
        if(args[0]==='observe') return {state:'completed',exit:0};
        if(args[0]==='logs') return {content:output};
        assert.fail('Unexpected operation '+args[0]);
      }});
    const proof=(await environment.resources())[0].metadata.host_memory;
    assert.equal(proof.max_bytes,expectedCap);
    assert.equal(proof.scope,'/'+expectedSlice);
  }
});
