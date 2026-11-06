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

test('an unavailable performance project does not block an ordinary project',{timeout:15000},async t=>{
  const {mkdir,writeFile}=await import('node:fs/promises');
  const {operate}=await import('../../cli/runtime.mjs');
  const {State}=await import('../../controller/state.mjs');
  const root=await mkdtemp(join(tmpdir(),'dispatch-admission-'));t.after(()=>rm(root,{recursive:true,force:true}));
  await mkdir(join(root,'skills','versions','v'),{recursive:true});
  await writeFile(join(root,'environment.mjs'),`export function createEnvironment(){
    if(process.argv[2]==='worker')process.exit(0);
    return {resources:async({purpose})=>purpose==='performance'?[]:[{id:'cpu',available:true}]};
  }`);
  const config={format:1,root,mode:'phase1',skills:{activeVersion:'v'},recovery:false,python:join(root,'missing-python'),
    environment:{module:'environment.mjs'},projects:[
      {id:'a',goal:'performance',actor_seconds:10,performance_goal:'benchmark',verifier:{files:[],command:'true',performance_command:'true'}},
      {id:'b',goal:'ordinary',actor_seconds:10}]};
  await writeFile(join(root,'instance.json'),JSON.stringify(config));
  const result=await operate('wave',config);
  const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  assert.equal(result.results.length,1);assert.equal(state.project('a').status,'queued');assert.equal(state.project('b').status,'failed');
});

test('public campaign executes repeated batches once, persists cursor and accepts an appended batch',{timeout:90000},async t=>{
  const {readFile}=await import('node:fs/promises');
  const directory=await mkdtemp(join(tmpdir(),'planned-campaign-')),cleanups=[];
  let passed=false;
  t.after(async()=>{
    try{for(const cleanup of cleanups.reverse())await cleanup({remainingSeconds:20});}
    finally{if(passed)await rm(directory,{recursive:true,force:true});else console.error('Retained campaign evidence: '+directory);}
  });
  const f=await campaignFixture({directory,signal:new AbortController().signal,onCleanup:fn=>cleanups.push(fn)},
    'phase2',['one','two','three','four']);
  const plan=['phase1','phase2','phase1','phase2'].map((phase,index)=>({id:'r'+index,phase,new_projects:1}));
  const config=JSON.parse(await readFile(join(f.instance,'instance.json'),'utf8'));
  const projects=config.projects.map((project,index)=>({...project,campaign_entry:plan[index].id,phase:plan[index].phase}));
  await f.config({campaign_plan:plan,projects,skills:{...config.skills,tuningDisabled:true}});
  await f.run('campaign');
  const originals=f.rows("SELECT json_extract(data,'$.campaign_entry') entry FROM events WHERE type='job-admitted' ORDER BY sequence");
  assert.deepEqual(originals.map(row=>row.entry),plan.map(entry=>entry.id));
  assert.equal(f.rows('SELECT * FROM attempts').length,4);
  assert.ok(f.rows('SELECT * FROM recovery_lessons').every(row=>row.recorded===1));
  assert.equal(JSON.parse(f.rows("SELECT value FROM control WHERE key='campaign-plan'")[0].value).index,4);
  await f.run('campaign');
  assert.equal(f.rows('SELECT * FROM attempts').length,4,'restart does not replay originals');
  await f.config({campaign_plan:[...plan,{id:'appended',phase:'phase1',new_projects:1}]});
  const continuation=f.start('campaign');
  await waitFor(()=>f.rows("SELECT value FROM control WHERE key='wait'").some(row=>JSON.parse(row.value)?.reason==='waiting-inputs'));
  assert.equal(f.rows('SELECT * FROM attempts').length,4);
  await f.run('stop');await continuation.output;
  passed=true;
});

test('finite campaign leaves final failed originals queued for recovery',{timeout:60000},async t=>{
  const {readFile}=await import('node:fs/promises');
  const directory=await mkdtemp(join(tmpdir(),'final-campaign-')),cleanups=[];
  let passed=false;
  t.after(async()=>{
    try{for(const cleanup of cleanups.reverse())await cleanup({remainingSeconds:20});}
    finally{if(passed)await rm(directory,{recursive:true,force:true});else console.error('Retained campaign evidence: '+directory);}
  });
  const f=await campaignFixture({directory,signal:new AbortController().signal,onCleanup:fn=>cleanups.push(fn)},'freeze',['freeze']);
  const config=JSON.parse(await readFile(join(f.instance,'instance.json'),'utf8'));
  await f.config({campaign_plan:[{id:'final',phase:'phase1',new_projects:1}],
    projects:config.projects.map(project=>({...project,campaign_entry:'final'}))});
  await f.run('campaign');
  assert.equal(f.rows('SELECT * FROM attempts').length,1);
  assert.equal(f.rows('SELECT * FROM recovery_queue').length,1);
  assert.equal(f.rows('SELECT * FROM recovery_attempts').length,0);
  assert.ok(f.rows('SELECT * FROM recovery_lessons').every(row=>row.recorded===1));
  passed=true;
});
