import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {ownerStopped} from '../../controller/lifecycle.mjs';
import {DatabaseSync} from 'node:sqlite';
import {campaignFixture,waitFor} from './resilience/campaign-fixture.mjs';
import {controllerUnit,knowledgeUnit} from '../../cli/managed-runtime.mjs';

// Explicit local gate; caller holds the shared validation lock. No rsi.slice writes.
test('public CPU CLI uses managed workers/KDB and preserves pause through controller restart',
  {skip:process.env.RSI_MANAGED_CLI_TEST!=='1',timeout:90000},async t=>{
    const directory=await mkdtemp(join(tmpdir(),'managed-cli-')),cleanups=[];
    const parent=process.env.RSI_TEST_SLICE;
    assert(parent?.startsWith('rsivalidation'),'Use run-managed-gate.mjs to bound all services under the owned 4GiB test slice');
    const slice=parent.slice(0,-6)+'-cli'+process.pid+Date.now()+'.slice',cap=768*1024**2;
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
    const f=await campaignFixture({directory,signal:abort.signal,onCleanup:fn=>cleanups.push(fn)},'recover',['recover','unrelated'],{managed:true});
    await f.config({completion_target:3,host_memory:{max_bytes:cap},managed_runtime:{slice,control_memory_bytes:384*1024**2,controller_service_bytes:192*1024**2}});
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
    assert((await readFile('/proc/'+receipt.owner.pid+'/cgroup','utf8')).includes('/'+slice+'/'+knowledgeUnit(f.instance)));
    assert.notEqual(await readFile('/proc/'+receipt.owner.pid+'/cgroup','utf8'),ownGroup);
    assert(workers.every(worker=>worker.pid!==owner.pid&&worker.cgroup.includes('/'+slice+'/rsi-worker-')));
    // Hold an actual target process, then kill the service owner rather than the CLI launcher.
    const entered=join(directory,'target-entered'),release=join(directory,'target-release');
    cleanups.push(()=>writeFile(release,'release'));
    const wrapper=join(f.instance,'machinery','managed-held.mjs');
    const holdCode="from pathlib import Path; import time; Path("+JSON.stringify(entered)+").touch();\nwhile not Path("+JSON.stringify(release)+").exists(): time.sleep(.01)";
    const holdCommand="python -c '"+holdCode.replaceAll("'","'\\''")+"'";
    await writeFile(wrapper,`import {createEnvironment as base} from './integrations/environments/index.mjs';
import {readFile} from 'node:fs/promises';
export async function createEnvironment(options,context){const env=await base(options,context);return {...env,
 resources:async()=>[{id:'slot-0',available:true},{id:'slot-1',available:true}],
 execute:async(command,cwd,opts)=>{
 const job=JSON.parse(await readFile(context.instancePath+'/job.json','utf8'));
 return env.execute(job.project?.id==='held'&&command.includes('import answer')?${JSON.stringify(holdCommand)}:command,cwd,opts);
 }};}
`);
    const beforeHeld=JSON.parse(await readFile(join(f.instance,'instance.json'),'utf8'));
    const held={...beforeHeld.projects[1],id:'held',goal:'unrelated-held'};
    const freshAfter={...beforeHeld.projects[1],id:'after-crash',goal:'unrelated-after-crash'};
    await f.config({completion_target:100,concurrency:'auto',environment:{module:'machinery/managed-held.mjs',options:{mode:'local'}},
      curriculum:{phase1:[...beforeHeld.projects.map(project=>project.id),'held']},projects:[...beforeHeld.projects,held]});
    await waitFor(async()=>{try{await readFile(entered);return true;}catch(error){if(error.code!=='ENOENT')throw error;}},undefined,20);
    const heldBefore=f.rows("SELECT * FROM attempts WHERE project='held'")[0];
    const crashSequence=f.rows('SELECT MAX(sequence) AS n FROM events')[0].n;
    const crashOwner=JSON.parse(f.rows("SELECT value FROM control WHERE key='owner'")[0].value);
    process.kill(crashOwner.pid,'SIGKILL');
    await waitFor(()=>{
      const active=JSON.parse(f.rows("SELECT value FROM control WHERE key='owner'")[0].value);
      return active&&active.pid!==crashOwner.pid;
    },undefined,10);
    assert.equal(await ownerStopped(heldBefore.owner),false);
    const sameKdb=JSON.parse(await readFile(join(f.instance,'knowledge-service.json'),'utf8'));
    assert.equal(sameKdb.url,receipt.url);assert.equal(sameKdb.owner.pid,receipt.owner.pid);
    await f.config({curriculum:{phase1:[...beforeHeld.projects.map(project=>project.id),'held','after-crash']},
      projects:[...beforeHeld.projects,held,freshAfter]});
    await waitFor(()=>f.rows("SELECT status FROM projects WHERE id='after-crash'")[0]?.status==='passed',undefined,20);
    assert.equal(f.rows("SELECT outcome FROM attempts WHERE id='"+heldBefore.id+"'")[0].outcome,null);
    assert.equal(await ownerStopped(heldBefore.owner),false);
    await writeFile(release,'release');
    await waitFor(async()=>{
      const attempt=f.rows("SELECT * FROM attempts WHERE id='"+heldBefore.id+"'")[0];
      return attempt.outcome==='passed'&&await ownerStopped(attempt.owner)&&
        f.rows("SELECT 1 FROM events WHERE attempt='"+heldBefore.id+"' AND type='postprocess-completed'").length;
    },undefined,20);
    assert.equal(f.rows("SELECT COUNT(*) AS n FROM attempts WHERE project='held'")[0].n,1);
    const heldSettled=f.rows("SELECT * FROM attempts WHERE id='"+heldBefore.id+"'")[0];
    assert.equal(heldSettled.session,heldBefore.session);
    const sessionBytes=await readFile(heldSettled.session);
    const postCrashSearches=f.rows("SELECT data FROM events WHERE attempt='"+heldBefore.id+"' AND type='knowledge-tool-result' AND sequence>"+crashSequence).map(row=>JSON.parse(row.data));
    assert(postCrashSearches.some(event=>event.tool.includes('search')&&!event.failed));
    assert.equal(f.rows("SELECT COUNT(*) AS n FROM events WHERE attempt='"+heldBefore.id+"' AND type='resource-lease-reclaimed'")[0].n,1);
    assert.equal(f.rows("SELECT COUNT(*) AS n FROM resource_leases WHERE attempt='"+heldBefore.id+"'")[0].n,0);
    assert(JSON.parse(f.rows("SELECT value FROM control WHERE key='actor-capacity'")[0].value)>=2);
    const current=JSON.parse(await readFile(join(f.instance,'instance.json'),'utf8'));
    const countBefore=f.rows('SELECT COUNT(*) AS n FROM attempts')[0].n;
    await f.config({host_memory:{max_bytes:cap/2},curriculum:{phase1:[...current.projects.map(project=>project.id),'after-ceiling-change']},projects:[...current.projects,{...current.projects[0],id:'after-ceiling-change'}]});
    await waitFor(()=>f.rows("SELECT value FROM control WHERE key='wait'").some(row=>JSON.parse(row.value)?.reason==='memory-enforcement'));
    assert.equal(f.rows('SELECT COUNT(*) AS n FROM attempts')[0].n,countBefore);
    assert.equal(f.rows("SELECT status FROM projects WHERE id='after-ceiling-change'")[0].status,'queued');
    // Set durable pause then kill in the same synchronous host turn, before the 100ms watcher.
    const db=new DatabaseSync(join(f.instance,'state.sqlite'));db.exec('PRAGMA busy_timeout=5000');
    db.prepare("INSERT OR REPLACE INTO control(key,value) VALUES ('pause','true')").run();db.close();
    process.kill(JSON.parse(f.rows("SELECT value FROM control WHERE key='owner'")[0].value).pid,'SIGKILL');
    const result=await running.output;
    assert.equal(result.reason,'paused');
    assert.equal(JSON.parse(f.rows("SELECT value FROM control WHERE key='pause'")[0].value),true);
    const count=f.rows('SELECT COUNT(*) AS n FROM attempts')[0].n;
    assert.equal((await f.run('wave')).reason,'paused');
    assert.equal(f.rows('SELECT COUNT(*) AS n FROM attempts')[0].n,count);
    await f.config({host_memory:{max_bytes:cap}});
    const continued=f.start('resume');
    await waitFor(()=>f.rows("SELECT status FROM projects WHERE id='after-ceiling-change'")[0].status==='passed',undefined,20);
    assert.equal(JSON.parse(f.rows("SELECT value FROM control WHERE key='pause'")[0].value),false);
    await f.run('stop');await continued.output;
    const stoppedConfig=JSON.parse(await readFile(join(f.instance,'instance.json'),'utf8'));
    await f.config({projects:[...stoppedConfig.projects,{...stoppedConfig.projects[1],id:'after-stop',goal:'unrelated-after-stop'}],
      curriculum:{phase1:[...stoppedConfig.projects.map(project=>project.id),'after-stop']}});
    const fresh=await f.run('wave');
    assert(fresh.results.some(row=>row.outcome==='passed'));
    assert.equal(f.rows("SELECT status FROM projects WHERE id='after-stop'")[0].status,'passed');
    assert.equal(JSON.parse(f.rows("SELECT value FROM control WHERE key='stop'")[0].value),false);
    assert.deepEqual(await readFile(heldSettled.session),sessionBytes);
    assert.deepEqual(f.rows("SELECT * FROM attempts WHERE id='"+heldBefore.id+"'")[0],heldSettled);
    // A resumed campaign remains continuous and shares the same elapsed cycle after restart.
    await f.config({campaign_seconds:5,completion_target:100});
    const budgetRun=f.start('campaign');
    await waitFor(()=>f.rows("SELECT value FROM control WHERE key='campaign-budget'").length);
    await f.run('pause');assert.equal((await budgetRun.output).reason,'paused');
    const budgetBefore=JSON.parse(f.rows("SELECT value FROM control WHERE key='campaign-budget'")[0].value);
    const resumed=f.start('resume');
    await waitFor(()=>JSON.parse(f.rows("SELECT value FROM control WHERE key='campaign-budget'")[0].value).running);
    await waitFor(()=>JSON.parse(f.rows("SELECT value FROM control WHERE key='campaign-budget'")[0].value).elapsed>=2,undefined,5);
    const restartOwner=JSON.parse(f.rows("SELECT value FROM control WHERE key='owner'")[0].value);
    process.kill(restartOwner.pid,'SIGKILL');
    const elapsedResult=await resumed.output;
    assert.equal(elapsedResult.reason,'elapsed');
    const budgetAfter=JSON.parse(f.rows("SELECT value FROM control WHERE key='campaign-budget'")[0].value);
    assert.equal(budgetAfter.cycle,budgetBefore.cycle);assert(budgetAfter.elapsed>=5&&budgetAfter.elapsed<6.5);
    if(process.env.RSI_MANAGED_PROOF)await writeFile(process.env.RSI_MANAGED_PROOF,JSON.stringify({
      passed:true,aggregate_bytes:cap,controller_unit:controllerUnit(f.instance),kdb_unit:knowledgeUnit(f.instance),
      kdb_pid:receipt.owner.pid,kdb_url:receipt.url,retained_attempt:heldBefore.id,
      fresh_finished_before_retained:true,post_crash_native_searches:postCrashSearches.filter(event=>!event.failed).length,
      session_preserved:true,lease_reclaimed_once:true,budget_before:budgetBefore,budget_after:budgetAfter},null,2));
    passed=true;
  });
