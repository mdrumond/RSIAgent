import test from 'node:test';
import assert from 'node:assert/strict';
import {writeFile,readFile,mkdir,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {setup,rows,until} from './driver_setup.mjs';
import {readConfig} from '../../cli/instance.mjs';
import {runBackgroundSkills} from '../../cli/skill-background.mjs';

test('storage-backed recovery selects accepted skills while an unapproved candidate holds the A/B writer lock',
  {timeout:120000},async t=>{
    let release=()=>{},block;
    t.after(async()=>{release();if(block)await rm(block,{force:true});});
    const f=await setup(t,'recover',['recover','unrelated']);
    const environment=join(f.root,'two-slots.mjs'),settings=join(f.root,'recovery-background.json');
    block=join(f.root,'hold-admission');const ready=join(f.root,'admission-ready');
    await writeFile(environment,`import {access,writeFile} from 'node:fs/promises';
import {createEnvironment as local} from './machinery/integrations/environments/index.mjs';
export const createEnvironment=(options,context)=>local(options,{...context,resources:async()=>{while(true){context.signal?.throwIfAborted();try{await access(${JSON.stringify(block)});}catch{break;}await writeFile(${JSON.stringify(ready)},'ready');await new Promise(resolve=>setTimeout(resolve,20));}return [0,1].map(i=>({id:'local'+i,available:true,metadata:{}}));}});\n`);
    const initial=await readConfig(f.root);
    await writeFile(settings,JSON.stringify({mode:'phase2',recovery:true,concurrency:2,tune_every:1,
      environment:{module:environment,options:{mode:'local'}},skills:{...initial.skills,background:true,tuningProjectIds:['recover','unrelated']}}));
    await f.run('config','--config',settings);await f.run('wave');
    assert.equal(rows(f.root,"SELECT status FROM projects WHERE id='recover'")[0].status,'failed');
    const original=rows(f.root,"SELECT id,session FROM attempts WHERE project='recover'")[0];
    await writeFile(block,'hold');const recoveryWave=f.start('wave');
    await until(async()=>{try{await readFile(ready);return true;}catch{return false;}});
    let entered;
    const gate=new Promise(resolve=>{release=resolve;}),started=new Promise(resolve=>{entered=resolve;});
    const runtime={bootstrap:async request=>{await mkdir(request.instanceRoot,{recursive:true});},
      wave:async()=>{entered();await gate;return {correct:true,score:1};},settled:async()=>true};
    runtime.resume=runtime.wave;
    const running=runBackgroundSkills(await readConfig(f.root),{resource:{id:'local0'},remainingSeconds:90,
      runtime,generate:async()=>[]});
    // Register rejection immediately so a failed assertion cannot leak a task.
    const completed=running.then(value=>({value}),error=>({error}));
    await started;
    const evolution=JSON.parse(await readFile(join(f.root,'skills','evolution.json'),'utf8'));
    const candidate=evolution.candidate;
    const mapping=rows(f.root,`SELECT revision FROM storage_skill_versions WHERE version='${candidate}'`)[0];
    assert.ok(mapping?.revision);
    const accepted=rows(f.root,`SELECT revision FROM storage_skill_versions WHERE version='${initial.skills.activeVersion}'`)[0];
    assert.ok(rows(f.root,"SELECT value FROM control WHERE key='skill-writer'")[0]);
    try {
      await rm(block,{force:true});await recoveryWave.output;
      const recovery=rows(f.root,"SELECT attempt FROM recovery_attempts WHERE kind='investigation'")[0];
      assert.ok(recovery);
      const job=JSON.parse(await readFile(join(f.root,'attempts',recovery.attempt,'job.json'),'utf8'));
      assert.equal(job.skillVersion,initial.skills.activeVersion);
      assert.equal(rows(f.root,`SELECT skill_revision FROM storage_attempts WHERE attempt='${recovery.attempt}'`)[0].skill_revision,accepted.revision);
      assert.equal(rows(f.root,`SELECT session FROM attempts WHERE id='${original.id}'`)[0].session,original.session);
      assert.equal(rows(f.root,"SELECT count(*) AS n FROM attempts WHERE project='recover'")[0].n,2);
      assert.equal(JSON.parse(await readFile(join(f.root,'skills','evolution.json'),'utf8')).pending.candidate,candidate);
    }finally{release();}
    const result=await completed;if(result.error)throw result.error;
    f.pass();
  });
