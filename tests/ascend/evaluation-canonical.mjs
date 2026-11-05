import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,cp,readFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {setup,rows} from './driver_setup.mjs';
import {readConfig} from '../../cli/instance.mjs';
import {bootstrap,wave} from '../../cli/evaluation.mjs';
import {ownerStopped} from '../../controller/lifecycle.mjs';

test('isolated evaluation runs its requested canonical project outside the production curriculum',{timeout:120000},async t=>{
  const f=await setup(t,'recover',['unrelated']);
  const initial=await readConfig(f.root),plan=join(f.root,'plan.json');
  await writeFile(plan,JSON.stringify({campaign_plan:[{id:'first',phase:'phase1',new_projects:1},
    {id:'later',phase:'phase2',new_projects:1}],projects:initial.projects.map(project=>({...project,campaign_entry:'later',phase:'phase2'}))}));
  await f.run('config','--config',plan);
  const config=await readConfig(f.root),round=join(f.root,'round');await mkdir(round);
  const credentials=join(round,'fixture.env');await writeFile(credentials,'DRIVER_FIXTURE_KEY=fixture-only\n');
  const requested={...config,mode:'phase2',curriculum:{phase2:[]},actor:{...config.actor,credentials_file:credentials}};
  const instanceRoot=join(round,'sample');
  await bootstrap({sourceInstanceRoot:f.root,instanceRoot,projectId:'unrelated',projectIds:['unrelated'],skillVersion:'baseline',config:requested});
  assert.equal((await readConfig(instanceRoot)).campaign_plan,undefined);
  try {
    const result=await wave({instanceRoot,config:requested,projectIds:['unrelated']});
    assert.equal(result.correct,true);assert.equal(result.attempts.length,1);
    assert.equal(rows(instanceRoot,"SELECT count(*) AS n FROM events WHERE type='stage-completed'")[0].n,8);
    assert.ok(f.server.requests.some(r=>r.messages.some(m=>m.role==='tool')));
    const unavailable=join(round,'unavailable'),before=f.server.requests.length;
    await bootstrap({sourceInstanceRoot:f.root,instanceRoot:unavailable,projectId:'unrelated',projectIds:['unrelated'],skillVersion:'baseline',
      config:{...requested,resource_ids:['unavailable-fixture-resource']}});
    const skipped=await wave({instanceRoot:unavailable,config:requested});
    assert.equal(skipped.correct,null);assert.deepEqual(skipped.attempts,[]);
    assert.equal(f.server.requests.length,before);
  } finally {
    try{
      const receipt=JSON.parse(await readFile(join(instanceRoot,'knowledge-service.json'),'utf8'));
      if(!await ownerStopped(receipt.owner))process.kill(receipt.owner.pid,'SIGTERM');
    }catch(error){if(error.code!=='ENOENT'&&error.code!=='ESRCH')throw error;}
  }
  f.pass();
});


test('phase2 completed tasks write skills and run a real eight-arm canonical A/B round',{timeout:120000},async t=>{
  const f=await setup(t,'phase2',['unrelated','second']);
  const input=join(f.root,'phase2.json');
  await writeFile(input,JSON.stringify({mode:'phase2',tune_every:2,skills:{...(await readConfig(f.root)).skills,evaluationSeconds:120},curriculum:{phase2:['unrelated','second']}}));
  await f.run('config','--config',input);await f.run('wave','--mode','phase2');
  const status=await f.run('status');assert.ok(status.projects.every(p=>p.status==='passed'));
  const evolution=JSON.parse(await readFile(join(f.root,'skills','evolution.json'),'utf8'));
  assert.equal(evolution.history.length,2);assert.equal(evolution.lastEvaluated,2);assert.equal(evolution.pending,null);
  assert.equal(rows(f.root,"SELECT count(*) AS n FROM events WHERE type='skills-updated'")[0].n,2);
  assert.equal(rows(f.root,"SELECT count(*) AS n FROM events WHERE type='skill-update-error'")[0].n,0);
  const evaluations=rows(f.root,"SELECT data FROM events WHERE type='evaluation-started'");assert.equal(evaluations.length,8);
  for(const event of evaluations){
    const child=JSON.parse(event.data).instanceRoot;
    assert.equal(rows(child,"SELECT count(*) AS n FROM attempts WHERE outcome='passed'")[0].n,1);
    assert.equal(rows(child,"SELECT count(*) AS n FROM events WHERE type='stage-completed'")[0].n,8);
    const receipt=JSON.parse(await readFile(join(child,'knowledge-service.json'),'utf8'));
    if(!await ownerStopped(receipt.owner))process.kill(receipt.owner.pid,'SIGTERM');
  }
  assert.ok(f.server.requests.some(r=>r.messages.some(m=>m.tool_calls?.some(c=>c.function.name==='submit_changes'))));
  f.pass();
});

test('standalone resume finishes recorded outcome postprocessing without replaying its actor',{timeout:120000},async t=>{
  const {DatabaseSync}=await import('node:sqlite');
  const f=await setup(t,'phase2',['unrelated']);
  const file=join(f.root,'phase2.json');await writeFile(file,JSON.stringify({mode:'phase2',tune_every:8}));
  await f.run('config','--config',file);await f.run('wave');
  const before=await f.run('status'),session=rows(f.root,'SELECT session FROM attempts')[0].session;
  assert.ok(session);assert.equal(before.attempts.length,1);
  const history=JSON.parse(await readFile(join(f.root,'skills','evolution.json'),'utf8'));
  const db=new DatabaseSync(join(f.root,'state.sqlite'));
  try{db.prepare("DELETE FROM events WHERE type IN ('postprocess-completed','skills-updated')").run();}finally{db.close();}
  await f.run('resume');
  const after=await f.run('status');assert.equal(after.attempts.length,1);assert.equal(rows(f.root,'SELECT session FROM attempts')[0].session,session);
  assert.equal(rows(f.root,'SELECT COUNT(*) n FROM recovery_lessons')[0].n,1);
  assert.equal(rows(f.root,"SELECT COUNT(*) n FROM events WHERE type='postprocess-completed'")[0].n,1);
  assert.equal(rows(f.root,"SELECT COUNT(*) n FROM events WHERE type='skills-updated'")[0].n,1);
  assert.deepEqual(JSON.parse(await readFile(join(f.root,'skills','evolution.json'),'utf8')),history);
  await f.run('resume');assert.equal((await f.run('status')).attempts.length,1);
  f.pass();
});
