import test from 'node:test';
import assert from 'node:assert/strict';
import {writeFile,readFile,access} from 'node:fs/promises';
import {join} from 'node:path';
import {setup,rows,until} from './driver_setup.mjs';
import {readConfig} from '../../cli/instance.mjs';
import {operate,instanceStopped} from '../../cli/runtime.mjs';
import {ownerStopped,processIdentity} from '../../controller/lifecycle.mjs';

// All eight A/B samples use the production evaluator, SDK and Python verifier.
// Only the provider and embedding backend are scripted, as in the CLI fixtures.
const timeout=Number(process.env.RSI_CANONICAL_TEST_TIMEOUT_MS??600000);
test('background canonical eight-sample evaluation closes while foreground memory advances',
  {timeout},async t=>{
    const deadline=performance.now()+timeout;
    let f,release,round;
    const cleanup=async()=>{
      if(!f)return;
      try {
        await writeFile(release,'released');await f.run('stop');
        await until(()=>instanceStopped({...f.config,root:f.root}),15000);
        let evolution;try{evolution=JSON.parse(await readFile(join(f.root,'skills','evolution.json'),'utf8'));}
        catch(error){if(error.code==='ENOENT')return;throw error;}
        const path=evolution.pending?.evaluationRoot??evolution.lastEvaluationRoot;
        if(!path)return;
        const journal=JSON.parse(await readFile(join(path,'round.json'),'utf8'));
        for(const sample of journal.samples) {
          try {await access(join(sample.instanceRoot,'instance.json'));}
          catch(error){if(error.code==='ENOENT')continue;throw error;}
          const config=await readConfig(sample.instanceRoot);await operate('stop',config);
          await until(()=>instanceStopped(config),15000);
          await until(async()=>{
            if(!rows(config.root,"SELECT 1 FROM sqlite_master WHERE name='requests'").length)return true;
            for(const request of rows(config.root,'SELECT status,owner FROM requests'))
              if(['queued','running'].includes(request.status)||(request.owner&&!await ownerStopped(request.owner)))return false;
            return true;
          },15000);
          const receipt=JSON.parse(await readFile(join(config.root,'knowledge-service.json'),'utf8'));
          if(!await ownerStopped(receipt.owner))process.kill(receipt.owner.pid,'SIGTERM');
          await until(()=>ownerStopped(receipt.owner),15000);
        }
      } finally {
        try {
          const owner=rows(f.root,"SELECT value FROM control WHERE key='owner'")[0];
          const stopped=(!owner||await ownerStopped(JSON.parse(owner.value)))&&
            (await Promise.all(rows(f.root,'SELECT owner FROM attempts').map(row=>ownerStopped(row.owner)))).every(Boolean);
          if(stopped) {
            const receipt=JSON.parse(await readFile(join(f.root,'knowledge-service.json'),'utf8'));
            if(!await ownerStopped(receipt.owner)) {
              const actual=await processIdentity(receipt.owner.pid);
              assert.equal(actual.start,receipt.owner.start);assert.equal(actual.boot,receipt.owner.boot);
              process.kill(receipt.owner.pid,'SIGTERM');
              await until(()=>ownerStopped(receipt.owner),5000);
            }
          }
        } finally {await f.server.close();}
      }
    };
    t.after(cleanup); // Runs before setup removes any parent/source files.
    f=await setup(t,'phase2',['fa','gdn','free','hold'],{providerOptions:{resolveCall:async(call,body)=>{
      const content=body.messages.find(message=>message.role==='user')?.content;
      const prompt=typeof content==='string'?content:(content??[]).map(block=>block.text??'').join('');
      if(prompt.startsWith('Goal: hold')&&call?.name==='rsi_target')return {...call,args:{command:
        `python -c "import pathlib,time; p=pathlib.Path(${JSON.stringify(release).replaceAll('"',"'")}); p.parent.joinpath('hold-started').write_text('held'); print('foreground held',flush=True); exec('while not p.exists(): time.sleep(.05)')"`}};
      return call;
    }}});
    release=join(f.root,'release-foreground');
    const environment=join(f.root,'slots.mjs');
    await writeFile(environment,`import {createEnvironment as local} from './machinery/integrations/environments/index.mjs';
export const createEnvironment=(options,context)=>local(options,{...context,resources:async()=>[0,1].map(i=>({id:'local'+i,available:true,metadata:{}}))});\n`);
    const initial=await readConfig(f.root),plan=['phase1','phase2','phase1','phase2'].map((phase,index)=>({id:'entry'+index,phase,new_projects:1}));
    const file=join(f.root,'canonical-background.json');await writeFile(file,JSON.stringify({campaign_plan:plan,
      campaign_seconds:600,recovery:false,concurrency:2,tune_every:1,environment:{module:environment,options:{mode:'local'}},
      projects:initial.projects.map((project,index)=>({...project,phase:plan[index].phase,campaign_entry:plan[index].id,actor_seconds:project.id==='hold'?500:120})),
      skills:{...initial.skills,background:true,runsPerArm:2,tuningProjectIds:['fa','gdn']}}));
    await f.run('config','--config',file);f.config=await readConfig(f.root);
    const campaign=f.start('campaign');
    await until(()=>rows(f.root,"SELECT id FROM attempts WHERE project='hold'").length,180000);
    await until(async()=>{try{await access(join(f.root,'hold-started'));return true;}catch{return false;}},30000);
    assert.equal(rows(f.root,"SELECT status FROM projects WHERE id='free'")[0].status,'passed');
    await until(async()=>{
      try {
        const evolution=JSON.parse(await readFile(join(f.root,'skills','evolution.json'),'utf8'));
        const path=evolution.pending?.evaluationRoot??evolution.lastEvaluationRoot;
        if(!path)return false;
        round=JSON.parse(await readFile(join(path,'round.json'),'utf8'));
        return round.status==='completed'&&round.samples.filter(sample=>sample.result).length===8;
      }catch(error){if(error.code==='ENOENT')return false;throw error;}
    },Math.max(1,deadline-performance.now()-30000));
    assert.equal(rows(f.root,"SELECT outcome FROM attempts WHERE project='hold'")[0].outcome,null);
    assert.equal(new Set(round.samples.map(sample=>sample.instanceRoot)).size,8);
    const startingSources=new Map();
    for(const sample of round.samples) {
      assert.equal(sample.result.correct,true);
      const attempts=rows(sample.instanceRoot,'SELECT id,session,outcome FROM attempts');assert.equal(attempts.length,1);
      assert.equal(attempts[0].outcome,'passed');assert.ok(attempts[0].session);
      assert.equal(await instanceStopped(await readConfig(sample.instanceRoot)),true);
      const service=JSON.parse(await readFile(join(sample.instanceRoot,'knowledge-service.json'),'utf8'));
      assert.equal(await ownerStopped(service.owner),true,'completed sample service closed by evaluator');
      const identity=JSON.parse(rows(sample.instanceRoot,"SELECT data FROM events WHERE type='skill-evidence-context'")[0].data);
      const previous=startingSources.get(sample.projectId);if(previous)assert.deepEqual(identity.startingSource,previous);
      else startingSources.set(sample.projectId,identity.startingSource);
    }
    await writeFile(release,'released');await campaign.output;
    assert.equal(rows(f.root,'SELECT count(*) n FROM attempts')[0].n,4);
    assert.equal(rows(f.root,"SELECT value FROM control WHERE key='skill-background-resource'")[0].value,'null');
    await cleanup();f.pass();f=null;
  });
