import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {ownerStopped} from '../../controller/lifecycle.mjs';
import {State} from '../../controller/state.mjs';
import {CampaignStorage} from '../../controller/storage.mjs';
import {campaignFixture,waitFor,digest} from './resilience/campaign-fixture.mjs';

// Real public dispatch, native Pi tools, SQLite and Restic. The provider and
// embedding model are CPU fixtures; process supervision is not substituted.
test('public CLI preserves recovery, maintenance, pause and explicit controller restart',
  {timeout:90000},async t=>{
    const directory=await mkdtemp(join(tmpdir(),'public-cli-')),cleanups=[];
    const abort=new AbortController();let passed=false;
    t.after(async()=>{
      for(const cleanup of cleanups.reverse())await cleanup({remainingSeconds:15});
      if(passed)await rm(directory,{recursive:true,force:true});
      else t.diagnostic('Retained public CLI evidence: '+directory);
    });
    const f=await campaignFixture({directory,signal:abort.signal,onCleanup:fn=>cleanups.push(fn)},
      'recover',['recover','unrelated']);
    await f.config({completion_target:100,campaign_seconds:60});

    // Maintenance also works before the first actor starts.
    const repo=join(directory,'knowledge-input');await mkdir(repo);
    await writeFile(join(repo,'reference.py'),'def before_campaign_oracle(): return 42\n');
    const actionFile=join(directory,'maintenance.json'),collection='before-campaign';
    await writeFile(actionFile,JSON.stringify({tool:'add_repo',arguments:{path:repo,collection,
      language:'python',target:'cpu',paths:['reference.py']}}));
    const submitted=await f.run('kdb','--config',actionFile,'--detach');
    const requestRow=id=>f.rows('SELECT * FROM requests WHERE id=?','state.sqlite',[id])[0];
    await waitFor(async()=>{
      const row=requestRow(submitted.request);
      if(row.status==='failed')throw new Error(row.error);
      return row.status==='passed'&&await ownerStopped(row.owner);
    },undefined,15);
    const receipt=JSON.parse(await readFile(join(f.instance,'knowledge-service.json'),'utf8'));
    await writeFile(actionFile,JSON.stringify({tool:'search',arguments:{query:'before campaign oracle',
      kind:'good',target:'cpu',limit:20}}));
    const query=await f.run('kdb','--config',actionFile);
    assert.equal(query.isError,false);assert(JSON.stringify(query).includes(collection));
    await writeFile(actionFile,JSON.stringify({}));
    assert.deepEqual(await f.run('tune','--config',actionFile),{analyse:false,evaluate:false});

    const evidence=join(f.instance,'artifacts','evidence.jsonl');
    await mkdir(join(f.instance,'artifacts'),{recursive:true});
    const bytes='{"type":"fixture","value":42}\n';await writeFile(evidence,bytes);
    const state=new State(join(f.instance,'state.sqlite'));let artifact;
    try {artifact=await CampaignStorage.open(state,f.instance).register(evidence,{kind:'cited-text'});}
    finally {state.close();}
    // Ordinary backups keep the working path. Only explicit pruning deletes it.
    await f.run('storage','compress','--artifact',artifact.id);
    assert.equal(await readFile(evidence,'utf8'),bytes);
    assert.equal((await f.run('storage','verify','--artifact',artifact.id))[0].verified,true);
    const inventory=await f.run('storage','prune','--artifact',artifact.id);
    const inventoryFile=join(f.instance,'prune.json');await writeFile(inventoryFile,JSON.stringify(inventory));
    const removed=await f.run('storage','prune','--apply','--inventory',inventoryFile);
    assert.deepEqual(removed.removed,[artifact.id]);
    await assert.rejects(readFile(evidence),{code:'ENOENT'});
    assert.equal((await f.run('storage','verify','--artifact',artifact.id))[0].verified,true);

    const running=f.start('campaign');
    await waitFor(async()=>{
      if(running.child.exitCode!==null)await running.output;
      return f.rows('SELECT status FROM projects').every(row=>row.status==='passed');
    },undefined,50);
    await waitFor(()=>f.rows("SELECT id FROM attempts a WHERE NOT EXISTS (SELECT 1 FROM events e WHERE e.attempt=a.id AND e.type='postprocess-completed')").length===0);
    assert(f.rows('SELECT * FROM recovery_attempts').length>0);
    const lookups=f.rows("SELECT data FROM events WHERE type='knowledge-tool-result'").map(row=>JSON.parse(row.data));
    assert(lookups.some(event=>event.tool.includes('search')&&!event.failed));
    await f.run('pause');assert.equal((await running.output).reason,'paused');
    const settled=f.rows('SELECT * FROM attempts'),sessionBytes=new Map();
    for(const attempt of settled)if(attempt.session)sessionBytes.set(attempt.session,await readFile(attempt.session));
    assert(sessionBytes.size>0);
    assert.equal((await f.run('wave')).reason,'paused');
    assert.deepEqual(f.rows('SELECT * FROM attempts'),settled);
    const control=key=>JSON.parse(f.rows('SELECT value FROM control WHERE key=?','state.sqlite',[key])[0]?.value??'null');
    const budgetBefore=control('campaign-budget');

    const resumed=f.start('resume');
    await waitFor(async()=>control('campaign-budget')?.running&&control('owner')&&!await ownerStopped(control('owner')));
    const interrupted=control('owner');process.kill(interrupted.pid,'SIGKILL');
    await assert.rejects(resumed.output);
    await waitFor(()=>ownerStopped(interrupted));
    const countBefore=f.server.requests.length;
    // Restart is an explicit public operation, not an automatic service action.
    const restarted=f.start('resume');
    await waitFor(()=>control('owner')?.pid!==interrupted.pid&&control('campaign-budget')?.running);
    await f.run('pause');assert.equal((await restarted.output).reason,'paused');
    const budgetAfter=control('campaign-budget');
    assert.equal(budgetAfter.cycle,budgetBefore.cycle);
    assert(budgetAfter.elapsed>=budgetBefore.elapsed);
    assert.deepEqual(f.rows('SELECT * FROM attempts'),settled);
    assert.equal(f.server.requests.length,countBefore);
    const sameKdb=JSON.parse(await readFile(join(f.instance,'knowledge-service.json'),'utf8'));
    assert.equal(sameKdb.url,receipt.url);assert.equal(sameKdb.owner.pid,receipt.owner.pid);
    for(const [path,bytes] of sessionBytes) {
      assert.equal(digest(await readFile(path)),digest(bytes));
    }
    // Consume the remaining elapsed allocation after restart, without new work.
    const allocation=budgetAfter.elapsed+1;
    await f.config({campaign_seconds:allocation});
    const expired=await f.run('resume');
    assert.equal(expired.reason,'elapsed');
    assert(control('campaign-budget').elapsed>=allocation);
    assert.equal(control('campaign-budget').cycle,budgetBefore.cycle);
    assert.deepEqual(f.rows('SELECT * FROM attempts'),settled);
    passed=true;
  });
