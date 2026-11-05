import test from 'node:test';
import assert from 'node:assert/strict';
import {configureHostMemory} from '../../cli/host-memory.mjs';

function fixture({state='inactive',cap=4096,units={},stop=true}={}) {
  const calls=[],mutations=[],files=new Map(),directories=new Set();
  const group='/user.slice/user-1.slice/user@1.service/rsi.slice';
  let stopped=stop;
  const fs={
    mkdir:async(path,options)=>{mutations.push(['mkdir',path]);
      if(directories.has(path)&&!options.recursive)throw Object.assign(new Error('locked'),{code:'EEXIST'});
      directories.add(path);},
    writeFile:async(path,text)=>{mutations.push(['write',path]);files.set(path,text);},
    rename:async(from,to)=>{mutations.push(['rename',to]);files.set(to,files.get(from));files.delete(from);},
    rm:async path=>{files.delete(path);},rmdir:async path=>{directories.delete(path);},
    readFile:async path=>{
      if(path==='/sys/fs/cgroup/cgroup.controllers')return 'cpu memory';
      if(path.endsWith('/memory.max'))return String(cap);
      if(path.endsWith('/memory.swap.max'))return '0';
      throw new Error('Unexpected read '+path);
    },
  };
  const deps={fs,configHome:'/user-config',verifyInstanceStopped:async()=>stopped,
    execFile:async(file,args,options)=>{
      calls.push({file,args,options});
      if(args[1]==='show') {
        const unit=args[2];
        if(unit!=='rsi.slice')return {stdout:units[unit]};
        return {stdout:`ActiveState=${state}\nSlice=user.slice\nControlGroup=${group}\nMemoryAccounting=yes\nMemoryMax=${cap}\nMemorySwapMax=0`};
      }
      if(args[1]==='list-units')return {stdout:Object.keys(units).map(unit=>unit+' loaded active running').join('\n')};
      if(args[1]==='daemon-reload') {
        cap=Number(files.get('/user-config/systemd/user/rsi.slice').match(/MemoryMax=(\d+)/)[1]);
        return {stdout:''};
      }
      if(args[1]==='start'){state='active';return {stdout:''};}
      throw new Error('Unexpected command '+args);
    }};
  return {deps,calls,mutations,files,directories,setStopped:value=>stopped=value};
}

test('stopped inactive provisioning writes configured unit, reloads, starts and verifies kernel cap',async()=>{
  const f=fixture();
  const result=await configureHostMemory({max_bytes:8192},f.deps);
  assert.equal(result.max_bytes,8192);
  assert.match(f.files.get('/user-config/systemd/user/rsi.slice'),/MemoryMax=8192\n/);
  assert.deepEqual(f.calls.filter(c=>['daemon-reload','start'].includes(c.args[1])).map(c=>c.args),
    [['--user','daemon-reload'],['--user','start','rsi.slice']]);
  assert.ok(f.calls.every(c=>c.options.timeout===5000));
  assert.ok(!f.directories.has('/user-config/systemd/user/.rsi-memory-config.lock'));
});

test('active matching verifies without writes or stopped check; conflicting cap refuses',async()=>{
  const matching=fixture({state:'active',stop:false});
  await configureHostMemory({max_bytes:4096},matching.deps);
  assert.deepEqual(matching.mutations,[]);
  const conflicting=fixture({state:'active'});
  await assert.rejects(configureHostMemory({max_bytes:8192},conflicting.deps),/enforcement unavailable/);
  assert.deepEqual(conflicting.mutations,[]);
});

test('inactive configuration requires stopped proof and rejects another instance or nested slice',async()=>{
  const running=fixture({stop:false});
  await assert.rejects(configureHostMemory({},running.deps),/stopped instance/);
  assert.deepEqual(running.mutations,[]);
  for(const Slice of ['rsi.slice','rsi-nested.slice','app.slice']) {
    const name=Slice==='app.slice'?'rsi-worker-other.service':'other.service';
    const f=fixture({units:{[name]:`ActiveState=active\nSlice=${Slice}`}});
    await assert.rejects(configureHostMemory({},f.deps),/Other active RSI unit/);
    assert.ok(!f.calls.some(c=>c.args[1]==='daemon-reload'));
    assert.ok(!f.mutations.some(m=>m[0]==='write'));
    assert.ok(!f.directories.has('/user-config/systemd/user/.rsi-memory-config.lock'));
  }
});

test('provision lock blocks competing writers and remains owned by original caller',async()=>{
  const f=fixture();
  const lock='/user-config/systemd/user/.rsi-memory-config.lock';
  f.directories.add(lock);
  await assert.rejects(configureHostMemory({},f.deps),{code:'EEXIST'});
  assert.ok(f.directories.has(lock));
  assert.ok(!f.mutations.some(m=>m[0]==='write'));
});

test('manager errors propagate and release provisioning lock without launching a slice',async()=>{
  const f=fixture();
  const original=f.deps.execFile;
  f.deps.execFile=async(file,args,options)=>{
    if(args[1]==='list-units')throw new Error('manager unavailable');
    return original(file,args,options);
  };
  await assert.rejects(configureHostMemory({},f.deps),/manager unavailable/);
  assert.ok(!f.directories.has('/user-config/systemd/user/.rsi-memory-config.lock'));
  assert.ok(!f.calls.some(c=>c.args[1]==='start'));
});

test('stopped proof is refreshed under lock and active kernel mismatches fail closed',async()=>{
  const f=fixture();let checks=0;
  f.deps.verifyInstanceStopped=async()=>++checks===1;
  await assert.rejects(configureHostMemory({},f.deps),/started during/);
  assert.equal(checks,2);
  assert.ok(!f.mutations.some(m=>m[0]==='write'));
  const active=fixture({state:'active'});
  const read=active.deps.fs.readFile;
  active.deps.fs.readFile=async path=>path.endsWith('/memory.max')?'max':read(path);
  await assert.rejects(configureHostMemory({max_bytes:4096},active.deps),/limit mismatch/);
  assert.deepEqual(active.mutations,[]);
});
