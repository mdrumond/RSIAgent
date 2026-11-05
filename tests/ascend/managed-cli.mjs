import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {DatabaseSync} from 'node:sqlite';
import {campaignFixture,waitFor} from './resilience/campaign-fixture.mjs';
import {controllerUnit} from '../../cli/managed-runtime.mjs';

// Explicit local gate; caller holds the shared validation lock. No rsi.slice writes.
test('public CPU CLI uses managed workers/KDB and preserves pause through controller restart',
  {skip:process.env.RSI_MANAGED_CLI_TEST!=='1',timeout:90000},async t=>{
    const directory=await mkdtemp(join(tmpdir(),'managed-cli-')),cleanups=[];
    const slice='rsiclitest'+process.pid+Date.now()+'.slice',cap=512*1024**2;
    const run=promisify(execFile),systemctl=args=>run('systemctl',['--user',...args],{timeout:5000,maxBuffer:65536});
    const unitDir=join(process.env.XDG_RUNTIME_DIR,'systemd/user'),unitFile=join(unitDir,slice);
    const abort=new AbortController();let passed=false;
    t.after(async()=>{
      try {for(const cleanup of cleanups.reverse())await cleanup({remainingSeconds:15});}
      finally {
        await systemctl(['stop',slice]);await rm(unitFile,{force:true});await systemctl(['daemon-reload']);
        if(passed)await rm(directory,{recursive:true,force:true});else console.error('Retained managed CLI evidence: '+directory);
      }
    });
    await mkdir(unitDir,{recursive:true});
    await writeFile(unitFile,'[Slice]\nMemoryAccounting=yes\nMemoryMax='+cap+'\nMemorySwapMax=0\n');
    await systemctl(['daemon-reload']);await systemctl(['start',slice]);
    const f=await campaignFixture({directory,signal:abort.signal,onCleanup:fn=>cleanups.push(fn)},'recover',['recover','unrelated']);
    await f.config({completion_target:3,host_memory:{max_bytes:cap},managed_runtime:{slice,controller_memory_bytes:256*1024**2}});
    const running=f.start('campaign');
    await waitFor(async()=>{
      if(running.child.exitCode!==null)await running.output;
      return f.rows('SELECT status FROM projects').every(row=>row.status==='passed');
    },undefined,50);
    const workers=f.rows("SELECT data FROM events WHERE type='managed-worker-started'").map(row=>JSON.parse(row.data));
    assert(workers.length>=4);
    const memory=f.rows("SELECT data FROM events WHERE type='worker-memory'").map(row=>JSON.parse(row.data));
    assert(memory.some(row=>row.closed&&row.peak_bytes>0&&row.events?.oom_kill===0));
    const receipt=JSON.parse(await readFile(join(f.instance,'knowledge-service.json'),'utf8'));
    const owner=JSON.parse(f.rows("SELECT value FROM control WHERE key='owner'")[0].value);
    const ownGroup=await readFile('/proc/'+owner.pid+'/cgroup','utf8');
    assert(ownGroup.includes('/'+slice+'/'+controllerUnit(f.instance)));
    assert.equal(await readFile('/proc/'+receipt.owner.pid+'/cgroup','utf8'),ownGroup);
    assert(workers.every(worker=>worker.pid!==owner.pid&&worker.cgroup.includes('/'+slice+'/rsi-worker-')));
    const current=JSON.parse(await readFile(join(f.instance,'instance.json'),'utf8'));
    const countBefore=f.rows('SELECT COUNT(*) AS n FROM attempts')[0].n;
    await f.config({host_memory:{max_bytes:cap/2},projects:[...current.projects,{...current.projects[0],id:'after-ceiling-change'}]});
    await waitFor(()=>f.rows("SELECT value FROM control WHERE key='wait'").some(row=>JSON.parse(row.value)?.reason==='memory-enforcement'));
    assert.equal(f.rows('SELECT COUNT(*) AS n FROM attempts')[0].n,countBefore);
    assert.equal(f.rows("SELECT status FROM projects WHERE id='after-ceiling-change'")[0].status,'queued');
    // Set durable pause then kill in the same synchronous host turn, before the 100ms watcher.
    const db=new DatabaseSync(join(f.instance,'state.sqlite'));db.exec('PRAGMA busy_timeout=5000');
    db.prepare("INSERT OR REPLACE INTO control(key,value) VALUES ('pause','true')").run();db.close();
    process.kill(owner.pid,'SIGKILL');
    const result=await running.output;
    assert.equal(result.reason,'paused');
    assert.equal(JSON.parse(f.rows("SELECT value FROM control WHERE key='pause'")[0].value),true);
    const count=f.rows('SELECT COUNT(*) AS n FROM attempts')[0].n;
    assert.equal((await f.run('wave')).reason,'paused');
    assert.equal(f.rows('SELECT COUNT(*) AS n FROM attempts')[0].n,count);
    await f.config({host_memory:{max_bytes:cap}});
    await f.run('resume');
    assert.equal(JSON.parse(f.rows("SELECT value FROM control WHERE key='pause'")[0].value),false);
    await f.run('stop');
    const fresh=await f.run('wave');
    assert(fresh.results.some(row=>row.outcome==='passed'));
    assert.equal(f.rows("SELECT status FROM projects WHERE id='after-ceiling-change'")[0].status,'passed');
    assert.equal(JSON.parse(f.rows("SELECT value FROM control WHERE key='stop'")[0].value),false);
    passed=true;
  });
