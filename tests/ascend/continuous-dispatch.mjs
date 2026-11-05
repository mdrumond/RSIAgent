import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {campaignFixture,waitFor} from './resilience/campaign-fixture.mjs';

test('continuous campaign completes canonical recovery without restarting and stays available for input',{timeout:90000},async t=>{
  const directory=await mkdtemp(join(tmpdir(),'continuous-campaign-')),cleanups=[];
  let passed=false;
  const abort=new AbortController();
  t.after(async()=>{
    try{for(const cleanup of cleanups.reverse())await cleanup({remainingSeconds:20});}
    finally{if(passed)await rm(directory,{recursive:true,force:true});else console.error('Retained campaign evidence: '+directory);}
  });
  const f=await campaignFixture({directory,signal:abort.signal,onCleanup:fn=>cleanups.push(fn)},'recover',['recover','unrelated']);
  await f.config({completion_target:3});
  const running=f.start('campaign');
  await waitFor(async()=>{
    if(running.child.exitCode!==null)await running.output;
    const results=f.rows('SELECT status FROM projects');
    return results.length===2&&results.every(row=>row.status==='passed');
  },undefined,60);
  assert.equal(f.rows("SELECT COUNT(*) AS n FROM recovery_attempts WHERE kind='retry'")[0].n,1);
  assert.equal(f.rows("SELECT COUNT(*) AS n FROM recovery_attempts WHERE kind='investigation'")[0].n,1);
  await waitFor(()=>f.rows("SELECT value FROM control WHERE key='wait'").some(row=>JSON.parse(row.value)?.reason==='inputs-or-recovery'));
  await f.run('stop');await running.output;
  assert.equal(f.rows("SELECT value FROM control WHERE key='owner'")[0].value,'null');
  assert.ok(f.rows('SELECT * FROM recovery_lessons').every(row=>row.recorded===1));
  passed=true;
});
