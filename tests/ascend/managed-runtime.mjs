import test from 'node:test';
import assert from 'node:assert/strict';
import {EventEmitter} from 'node:events';
import {PassThrough} from 'node:stream';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {spawnManagedWorker,startManagedController,verifyProcessMembership,observeManagedWorker} from '../../cli/managed-runtime.mjs';

const cap = 64 * 1024 ** 2;
function fixture() {
  let phase = 'running', populated = '0', outside = false, reservation = 0;
  const commands = [], launches = [];
  const aggregate = '/user.slice/rsi.slice', group = aggregate + '/worker.service';
  const options = {
    withLock:operation=>operation(),
    hostMemory:{max_bytes:cap}, memoryBytes:24 * 1024 ** 2, controllerMemoryBytes:32 * 1024 ** 2,
    identify:async pid => ({pid,start:'17',boot:'boot'}),
    execFile:async (file,args,limits) => {
      assert.equal(file,'systemctl'); assert.equal(limits.timeout,5000); assert.equal(limits.maxBuffer,65536);
      commands.push(args);
      if (args.includes('list-units'))return {stdout:reservation?'rsi-worker-existing.service loaded active running':''};
      if (args.includes('stop')) {phase = 'inactive'; return {stdout:''};}
      const slice = args.includes('rsi.slice');
      return {stdout:Object.entries(slice ? {
        ActiveState:'active',MemoryAccounting:'yes',MemoryMax:cap,MemorySwapMax:0,ControlGroup:aggregate,
      } : {ActiveState:phase === 'inactive' ? 'inactive' : 'active',SubState:phase === 'exited' ? 'exited' : phase,
        MainPID:phase === 'running' ? 123 : 0,ExecMainPID:123,InvocationID:'invocation',ControlGroup:group,
        MemoryMax:reservation,MemoryCurrent:4096,MemoryPeak:8192,ExecMainCode:1,ExecMainStatus:0,Result:'success',
      }).map(([key,value]) => key + '=' + value).join('\n')};
    },
    read:async path => {
      if (path.endsWith('/memory.max')) return String(cap);
      if (path.endsWith('/memory.swap.max')) return '0';
      if (path.startsWith('/proc/')) return '0::' + (outside ? '/outside' : group) + '\n';
      if (path.endsWith('/memory.current')) return '4096\n';
      if (path.endsWith('/memory.peak')) return '8192\n';
      if (path.endsWith('/memory.events')) return 'oom 0\noom_kill 0\n';
      if (path.endsWith('/cgroup.events')) return 'populated ' + populated + '\nfrozen 0\n';
      throw new Error(path);
    },
    spawn:(file,args,settings) => {
      launches.push({file,args,settings});
      const child = new EventEmitter(); child.pid = 999; child.stdout = new PassThrough(); child.stderr = new PassThrough();
      queueMicrotask(() => child.emit('close',0)); return child;
    },
  };
  return {options,commands,launches,setPhase:value => phase=value,setOutside:value => outside=value,
    setPopulated:value => populated=value,setReservation:value=>reservation=value};
}
const descriptor = {argv:['/bin/echo','$VALUE','space and "quote"','%h'],cwd:'/tmp',env:{VALUE:'literal $x %h'}};
test('worker admission, real PID identity, measured memory, and whole-service closure',async () => {
  const f = fixture(), worker = await spawnManagedWorker(descriptor,f.options);
  assert.equal(worker.identity.pid,123); assert.equal(worker.child.pid,999);
  assert.deepEqual(await worker.metrics().then(({observed_at,...rest}) => rest),
    {current_bytes:4096,peak_bytes:8192,events:{oom:0,oom_kill:0}});
  f.setPhase('exited');
  const result = await worker.wait({timeoutMs:500});
  assert.equal(result.closed,true); assert.equal(result.code,0);
  assert.equal((await worker.metrics()).peak_bytes,8192);
  assert(f.commands.some(args => args.includes('stop') && args.includes(worker.identity.unit)));
  const launch = f.launches[0];
  assert.deepEqual(launch.args.slice(launch.args.indexOf('--')+1),descriptor.argv);
  assert(launch.args.includes('--expand-environment=no'));
  assert(launch.args.includes('--property=OOMPolicy=kill'));
  assert(launch.args.includes('--property=KillMode=control-group'));
});
test('outside existing KDB refuses admission before spawn',async () => {
  const f = fixture(); f.setOutside(true);
  await assert.rejects(verifyProcessMembership(44,f.options),/outside/);
  await assert.rejects(startManagedController(descriptor,{...f.options,requiredPids:[44]}),/outside/);
  assert.equal(f.launches.length,0);
});
test('retained worker is observed after controller restart without spawning again',async () => {
  const f=fixture(),first=await spawnManagedWorker(descriptor,f.options);
  const retained=await observeManagedWorker(first.identity,f.options);
  f.setPhase('exited');
  assert.equal((await retained.wait({timeoutMs:500})).closed,true);
  assert.equal(f.launches.length,1);
});
test('nonempty descendant group never produces a closure claim',async () => {
  const f = fixture(), worker = await spawnManagedWorker(descriptor,f.options);
  f.setPopulated('1');
  await assert.rejects(worker.stop(),/not proven closed/);
});
test('bounded wait leaves explicit stop available and rejects invalid caps',async () => {
  const f = fixture();
  await assert.rejects(spawnManagedWorker(descriptor,{...f.options,memoryBytes:cap}),/headroom/);
  const worker = await spawnManagedWorker(descriptor,f.options);
  await assert.rejects(worker.wait({timeoutMs:1}),/timed out/);
  assert.equal((await worker.stop()).closed,true);
});
test('measured aggregate pressure refuses workers before launch',async () => {
  const f = fixture(), read = f.options.read;
  f.options.read = path => path === '/sys/fs/cgroup/user.slice/rsi.slice/memory.current' ? Promise.resolve(String(cap)) : read(path);
  await assert.rejects(spawnManagedWorker(descriptor,f.options),/measured aggregate headroom/);
  assert.equal(f.launches.length,0);
});
test('existing service caps prevent aggregate overcommit before spawn',async () => {
  const f=fixture();f.setReservation(cap);
  await assert.rejects(spawnManagedWorker(descriptor,f.options),/reservations exhaust/);
  assert.equal(f.launches.length,0);
});
test('failed final enforcement never invokes scheduler admission callback',async () => {
  const f=fixture();let admitted=false;
  const read=f.options.read;f.options.read=path=>path.endsWith('/memory.max')?Promise.resolve('1'):read(path);
  await assert.rejects(spawnManagedWorker(async()=>{admitted=true;return descriptor;},f.options),{code:'RSI_MEMORY_ENFORCEMENT'});
  assert.equal(admitted,false);assert.equal(f.launches.length,0);
});
test('controller receives durable startup flags with restart policy',async () => {
  const f = fixture(); let called = false;
  await startManagedController(descriptor,{...f.options,startupFlags:async () => {called=true;return ['--paused-state=/tmp/state'];}});
  assert(called); assert(f.launches[0].args.includes('--property=Restart=on-failure'));
  assert.equal(f.launches[0].args.at(-1),'--paused-state=/tmp/state');
});

// Explicit opt-in: caller holds the shared validation flock. Never touches rsi.slice.
test('isolated real user services preserve descriptors, metrics, identity and descendant cleanup',
  {skip:process.env.RSI_MANAGED_SYSTEMD_TEST !== '1',timeout:45000},async () => {
    const run = promisify(execFile), dir = await mkdtemp(join(tmpdir(),'rsi-managed-'));
    const id = 'rsitest' + process.pid + Date.now(), slice = id + '.slice';
    const unitDir = join(process.env.XDG_RUNTIME_DIR,'systemd/user');
    const unitFile = join(unitDir,slice), log = join(dir,'worker%log'), report = join(dir,'report.json');
    const systemctl = args => run('systemctl',['--user',...args],{timeout:5000,maxBuffer:65536});
    let controller;
    try {
      await mkdir(unitDir,{recursive:true});
      await writeFile(unitFile,'[Slice]\nMemoryAccounting=yes\nMemoryMax='+cap+'\nMemorySwapMax=0\n');
      await systemctl(['daemon-reload']); await systemctl(['start',slice]);
      const module = new URL('../../cli/managed-runtime.mjs',import.meta.url).href;
      const harness = join(dir,'harness.mjs');
      await writeFile(harness,`
import {spawnManagedWorker} from ${JSON.stringify(module)};
import {writeFile,readFile} from 'node:fs/promises';
const options = ${JSON.stringify({slice,hostMemory:{max_bytes:cap},memoryBytes:16*1024**2,controllerMemoryBytes:48*1024**2})};
const worker = await spawnManagedWorker({argv:['/usr/bin/python3','-c',
  'import os,sys,time,json; print(json.dumps([sys.argv[1:],os.getenv("LITERAL"),os.getcwd(),os.getpid()]),flush=True); time.sleep(0.3)',
  '$LITERAL','space " quote','%h'],cwd:${JSON.stringify(dir)},env:{LITERAL:'$HOME %h'},stdout:${JSON.stringify(log)},stderr:${JSON.stringify(log)}},options);
const metrics = await worker.metrics();
const done = await worker.wait({timeoutMs:5000});
const output = JSON.parse((await readFile(${JSON.stringify(log)},'utf8')).trim());
// A detached child must also die when its own service is stopped.
const tree = await spawnManagedWorker({argv:['/usr/bin/python3','-c',
  'import os,time; p=os.fork(); open("descendant","w").write(str(p)) if p else None; time.sleep(30)'],cwd:${JSON.stringify(dir)}},options);
let descendant;
for (let i=0;i<100;i++) {
  try {descendant=Number(await readFile(${JSON.stringify(join(dir,'descendant'))},'utf8'));if(descendant)break;} catch {}
  await new Promise(resolve=>setTimeout(resolve,10));
}
if (!descendant) throw new Error('Descendant did not start');
const closed = await tree.stop();
try {
  const stat=await readFile('/proc/'+descendant+'/stat','utf8');
  if (stat.slice(stat.lastIndexOf(')')+2).split(' ')[0] !== 'Z') throw new Error('Descendant still running');
} catch(error) {if(error.code !== 'ENOENT')throw error;}
await writeFile(${JSON.stringify(report)},JSON.stringify({identity:worker.identity,launcher:worker.child.pid,metrics,done,output,closed}));
`);
      controller = await startManagedController({argv:[process.execPath,harness],cwd:dir,
        stdout:join(dir,'controller.log'),stderr:join(dir,'controller.log')},{slice,hostMemory:{max_bytes:cap},controllerMemoryBytes:48*1024**2});
      const deadline = Date.now()+20000; let result;
      while (Date.now()<deadline) {
        try {result=JSON.parse(await readFile(report,'utf8'));break;} catch (error) {if(error.code!=='ENOENT')throw error;}
        await new Promise(resolve=>setTimeout(resolve,100));
      }
      if (!result) throw new Error('No report: ' + await readFile(join(dir,'controller.log'),'utf8'));
      assert.deepEqual(result.output.slice(0,3),[['$LITERAL','space " quote','%h'],'$HOME %h',dir]);
      assert.equal(result.identity.pid,result.output[3]); assert.notEqual(result.identity.pid,result.launcher);
      assert(result.metrics.peak_bytes>0); assert.equal(result.metrics.events.oom_kill,0);
      assert(result.done.closed); assert.equal(result.done.code,0); assert(result.closed.closed);
    } finally {
      try {if (controller) await controller.stop();} finally {
      await systemctl(['stop',slice]); await rm(unitFile,{force:true}); await systemctl(['daemon-reload']);
      await rm(dir,{recursive:true,force:true});
      }
    }
  });
