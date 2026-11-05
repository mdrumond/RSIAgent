import test from 'node:test';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import assert from 'node:assert/strict';
import {mkdtemp, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHostMemorySupervisor, normalizeHostMemory} from '../../controller/host-memory.mjs';
import {readConfig} from '../../cli/instance.mjs';

function fixture(config) {
  const cap = config?.max_bytes ?? 16 * 1024 ** 3;
  const group = '/user.slice/user-1000.slice/user@1000.service/rsi.slice';
  const files = new Map(Object.entries({
    '/sys/fs/cgroup/cgroup.controllers':'cpu memory pids',
    '/proc/self/cgroup':`0::${group}/controller.scope\n`,
    [`/sys/fs/cgroup${group}/memory.max`]:String(cap),
    [`/sys/fs/cgroup${group}/memory.swap.max`]:'0',
    [`/sys/fs/cgroup${group}/memory.current`]:'1234',
    [`/sys/fs/cgroup${group}/memory.peak`]:'5678',
    [`/sys/fs/cgroup${group}/memory.events`]:'low 0\nhigh 1\nmax 2\noom 3\noom_kill 1\n',
  }));
  const calls = [];
  let properties = `ActiveState=active\nControlGroup=${group}\nMemoryAccounting=yes\nMemoryMax=${cap}\nMemorySwapMax=0`;
  const supervisor = createHostMemorySupervisor(config, {
    execFile:async (file,args,options) => {calls.push({file,args,options}); return {stdout:properties};},
    readFile:async path => {if (!files.has(path)) throw Object.assign(new Error(path),{code:'ENOENT'}); return files.get(path);},
  });
  return {supervisor,files,group,calls,setProperties:value=>properties=value};
}

test('configuration defaults and persisted override are normalized on read', async t => {
  const root = await mkdtemp(join(tmpdir(),'host-memory-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  await writeFile(join(root,'instance.json'),JSON.stringify({format:1}));
  assert.equal((await readConfig(root)).host_memory.max_bytes,17179869184);
  await writeFile(join(root,'instance.json'),JSON.stringify({format:1,host_memory:{max_bytes:4096}}));
  assert.equal((await readConfig(root)).host_memory.max_bytes,4096);
  for (const max_bytes of [0,-1,1.5,'4096',Infinity]) assert.throws(()=>normalizeHostMemory({max_bytes}));
});

test('admission requires verified controller membership and measures aggregate cgroup', async () => {
  const f = fixture();
  await assert.rejects(f.supervisor.workerCommand('/bin/true'));
  const report = await f.supervisor.prepare();
  assert.equal(report.current_bytes,1234);
  assert.equal(report.peak_bytes,5678);
  assert.deepEqual(report.events,{low:0,high:1,max:2,oom:3,oom_kill:1});
  const worker = await f.supervisor.workerCommand('/bin/echo',['hello world'],{cwd:'/tmp/work space',memoryBytes:8192,env:{DEVICE:'0',CUSTOM:'a b=c'}});
  assert.equal(worker.command,'systemd-run');
  assert.ok(worker.args.includes('--working-directory=/tmp/work space'));
  assert.ok(worker.args.includes('--setenv=CUSTOM=a b=c'));
  await assert.rejects(f.supervisor.workerCommand('/bin/true',[],{memoryBytes:8192,env:{BAD:42}}));
  assert.ok(worker.args.includes('--slice=rsi.slice'));
  assert.ok(worker.args.includes('--property=KillMode=control-group'));
  assert.ok(worker.args.includes('--property=OOMPolicy=kill'));
  assert.ok(worker.args.includes('--property=MemoryMax=8192'));
  assert.ok(worker.args.includes('--property=MemorySwapMax=0'));
  assert.deepEqual(worker.args.slice(-3),['--','/bin/echo','hello world']);
  const next = await f.supervisor.workerCommand('/bin/true',[],{memoryBytes:4096});
  assert.notEqual(worker.args.find(a=>a.startsWith('--unit=')),next.args.find(a=>a.startsWith('--unit=')));
  assert.ok(f.calls.every(c=>c.options.timeout===5000));
});

test('multiple instances use same aggregate, controller wrapper allows entry but admission fails outside', async () => {
  const a = fixture(), b = fixture();
  a.files.set('/proc/self/cgroup','0::/outside/controller.scope');
  const launch = await a.supervisor.controllerCommand('/usr/bin/node',['controller.mjs']);
  assert.ok(launch.args.includes('--scope'));
  assert.ok(launch.args.includes('--slice=rsi.slice'));
  await assert.rejects(a.supervisor.prepare(),/Controller must run inside/);
  await b.supervisor.prepare();
  assert.equal((await b.supervisor.measure()).slice,'rsi.slice');
});

test('unsupported accounting, mismatched caps, swap and lost enforcement all fail closed', async () => {
  for (const failure of ['accounting','kernel','swap','controllers','missing','membership']) {
    const f = fixture({max_bytes:4096});
    await f.supervisor.prepare();
    if (failure==='accounting') f.setProperties('ActiveState=active\nMemoryAccounting=no');
    if (failure==='kernel') f.files.set('/sys/fs/cgroup'+f.group+'/memory.max','max');
    if (failure==='swap') f.files.set('/sys/fs/cgroup'+f.group+'/memory.swap.max','max');
    if (failure==='controllers') f.files.set('/sys/fs/cgroup/cgroup.controllers','cpu');
    if (failure==='missing') f.files.delete('/sys/fs/cgroup'+f.group+'/memory.max');
    if (failure==='membership') f.files.set('/proc/self/cgroup','0::/outside');
    await assert.rejects(f.supervisor.workerCommand('/bin/true',[],{memoryBytes:2048}),undefined,failure);
    await assert.rejects(f.supervisor.workerCommand('/bin/true'),/prepare required/);
  }
  const mismatch = fixture();
  mismatch.setProperties('ActiveState=active\nMemoryAccounting=yes\nMemoryMax=4096');
  await assert.rejects(mismatch.supervisor.controllerCommand('/bin/true'));
});

test('measurement never substitutes estimates; optional peak may be absent', async () => {
  const f = fixture();
  f.files.delete('/sys/fs/cgroup'+f.group+'/memory.peak');
  assert.equal((await f.supervisor.prepare()).peak_bytes,null);
  f.files.set('/sys/fs/cgroup'+f.group+'/memory.current','unknown');
  await assert.rejects(f.supervisor.measure(),/Invalid measured/);
});


test('workers require an explicit cap with controller headroom and preserve caller budget', async () => {
  const f = fixture({max_bytes:16384});
  await f.supervisor.prepare();
  for (const memoryBytes of [undefined,0,-1,1.5,'8192',Infinity,16384,32768])
    await assert.rejects(f.supervisor.workerCommand('/bin/true',[],{memoryBytes}),/memoryBytes/);
  const worker = await f.supervisor.workerCommand('/bin/true',[],{memoryBytes:4096});
  assert.ok(worker.args.includes('--property=MemoryMax=4096'));
  assert.equal((await f.supervisor.measure()).max_bytes,16384);
});


test('real systemd worker preserves dollar arguments without environment expansion', {
  skip:process.env.RSI_HOST_MEMORY_SYSTEMD_TEST !== '1', timeout:15000,
}, async () => {
  const f = fixture();
  await f.supervisor.prepare();
  const expected = ['$RSI_ARGV_VALUE', '${RSI_ARGV_VALUE}', 'prefix ${RSI_ARGV_VALUE} suffix',
    '$RSI_ARGV_MISSING', '$$', 'a b', '', 'single\'quote', 'double"quote'];
  const worker = await f.supervisor.workerCommand('/bin/python3',
    ['-c','import json,sys; print(json.dumps(sys.argv[1:]))',...expected],
    {memoryBytes:64*1024**2, cwd:'/tmp', env:{RSI_ARGV_VALUE:'expanded words'}});
  // Keep the production service command and limits; omit only the shared slice
  // binding so this probe neither requires provisioning nor changes RSI limits.
  const args = worker.args.filter(arg => arg !== '--slice=rsi.slice');
  const {stdout} = await promisify(execFile)(worker.command,args,{timeout:10000,maxBuffer:65536});
  assert.deepEqual(JSON.parse(stdout.trim()),expected);
});
