import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,cp} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {skillPromotionPending,publishedPromotionGate} from '../../cli/skill-promotion-gate.mjs';
import {publishSkillVersion} from '../../cli/skills.mjs';
import {skillWorker} from '../../cli/skill-worker.mjs';
import {State} from '../../controller/state.mjs';
import {admitJob} from '../../controller/recovery.mjs';
import {setup,rows,until} from './driver_setup.mjs';

const load=async path=>JSON.parse(await readFile(path,'utf8'));
const save=(path,value)=>writeFile(path,JSON.stringify(value));

test('only successful baseline-matched publication releases the persisted gate',async t=>{
  const root=await mkdtemp(join(tmpdir(),'skill-promotion-gate-'));t.after(()=>rm(root,{recursive:true,force:true}));
  await mkdir(join(root,'skills'),{recursive:true});
  const skills={activeVersion:'baseline',promotionGate:{baseVersion:'baseline',status:'waiting'}};
  await save(join(root,'instance.json'),{format:1,skills});
  assert.equal(await publishSkillVersion(root,{active:'stale',candidate:'candidate'}),null);
  assert.equal(skillPromotionPending(await load(join(root,'instance.json'))),true);
  await publishSkillVersion(root,{active:'baseline',candidate:'baseline'});
  assert.equal(skillPromotionPending(await load(join(root,'instance.json'))),true);
  const abort=new AbortController();abort.abort();
  await assert.rejects(publishSkillVersion(root,{active:'baseline',candidate:'candidate',signal:abort.signal}));
  assert.equal(skillPromotionPending(await load(join(root,'instance.json'))),true);
  await publishSkillVersion(root,{active:'baseline',candidate:'candidate'});
  let config=await load(join(root,'instance.json'));
  assert.deepEqual(config.skills.promotionGate,{baseVersion:'baseline',status:'released',acceptedVersion:'candidate'});
  assert.equal(skillPromotionPending(config),false);
  await publishSkillVersion(root,{active:'candidate',candidate:'next'});
  config=await load(join(root,'instance.json'));assert.equal(skillPromotionPending(config),false);
  assert.equal(config.skills.activeVersion,'next');
  assert.equal(skillPromotionPending({skills:{...skills,activeVersion:'manual-change'}}),true);
  assert.equal(publishedPromotionGate({...skills,promotionGate:{baseVersion:'other',status:'waiting'}},{active:'baseline',candidate:'candidate'}).status,'waiting');
  assert.equal(skillPromotionPending({skills:{activeVersion:'baseline'}}),false);
});

test('a promotion gate starts background evaluation even before the first ordinary iteration',async()=>{
  const ledger=new Map([['campaign-plan',{index:0}]]);let called=0;
  const config={campaign_plan:[{}],skills:{background:true,promotionGate:{baseVersion:'baseline',status:'waiting'}}};
  const worker=skillWorker({control:{get:k=>ledger.get(k),set:(k,v)=>ledger.set(k,v),leased:()=>false},
    config:()=>config,remaining:()=>20,event:()=>{},operations:{backgroundSkillStatus:async()=>({hasWork:true}),
      backgroundSkillsSettled:async()=>true,runBackgroundSkills:async()=>{called++;}}});
  await worker.poll([{id:'ab',available:true,metadata:{backgroundSlot:true}}],0,1);await worker.running;
  assert.equal(called,1);assert.equal(worker.reserved,true);await worker.stop();
});

test('early gate release returns the settled background claim but retains running or unresolved work',async()=>{
  for(const unresolved of [false,true]) {
    const ledger=new Map([['campaign-plan',{index:0}]]);let release,closed=false;
    const config={campaign_plan:[{}],skills:{background:true,promotionGate:{baseVersion:'baseline',status:'waiting'}}};
    const worker=skillWorker({control:{get:k=>ledger.get(k),set:(k,v)=>ledger.set(k,v),leased:()=>false},
      config:()=>config,remaining:()=>20,event:()=>{},operations:{backgroundSkillStatus:async()=>({hasWork:true}),
        backgroundSkillsSettled:async()=>closed,runBackgroundSkills:async()=>new Promise(r=>{release=r;})}});
    // Qualification proves prior work closed; the next comparison remains in flight.
    closed=true;const resources=[{id:'only',available:true}];await worker.poll(resources,0,1);
    closed=false;config.skills.promotionGate.status='released';config.skills.promotionGate.acceptedVersion='candidate';
    await worker.poll(resources,0,1);assert.equal(worker.resource.id,'only');assert.equal(worker.reserved,true);
    release();await worker.running;closed=!unresolved;
    await worker.poll(resources,0,1);
    if(unresolved) {
      assert.equal(worker.resource.id,'only');assert.equal(ledger.get('skill-background-resource').id,'only');
      assert.equal(worker.reserved,true);closed=true;await worker.poll(resources,0,1);
    }
    assert.equal(worker.resource,null);assert.equal(ledger.get('skill-background-resource'),null);assert.equal(worker.reserved,false);
    assert.equal(worker.capacity,1);await worker.stop();
  }
});

test('first campaign entry admits its actor on the only resource after gate publication',{timeout:120000},async t=>{
  const f=await setup(t,'phase2',['queued']),initial=await load(join(f.root,'instance.json'));
  const environment=join(f.root,'one-slot.mjs'),settings=join(f.root,'gate-settings.json');
  await writeFile(environment,`import {createEnvironment as local} from './machinery/integrations/environments/index.mjs';
export const createEnvironment=(options,context)=>local(options,{...context,resources:async()=>[{id:'only',available:true,metadata:{}}]});\n`);
  await save(settings,{mode:'phase1',recovery:false,concurrency:1,campaign_seconds:120,pause_grace_seconds:0,
    environment:{module:environment,options:{mode:'local'}},campaign_plan:[{id:'first',phase:'phase1',new_projects:1}],
    projects:initial.projects.map(p=>({...p,phase:'phase1',campaign_entry:'first'})),
    skills:{...initial.skills,background:true,promotionGate:{baseVersion:'baseline',status:'waiting'}}});
  await f.run('config','--config',settings);const running=f.start('campaign');
  await until(()=>rows(f.root,"SELECT 1 FROM sqlite_master WHERE name='control'").length&&
    rows(f.root,"SELECT value FROM control WHERE key='skill-background-resource'").some(r=>JSON.parse(r.value)?.id==='only'));
  assert.equal(rows(f.root,'SELECT COUNT(*) n FROM attempts')[0].n,0);
  assert.equal(JSON.parse(rows(f.root,"SELECT value FROM control WHERE key='campaign-plan'")[0].value).index,0);
  const candidate='approved-candidate';await cp(join(f.root,'skills/versions/baseline'),join(f.root,'skills/versions',candidate),{recursive:true});
  await publishSkillVersion(f.root,{active:'baseline',candidate});await running.output;
  assert.equal(rows(f.root,"SELECT status FROM projects WHERE id='queued'")[0].status,'passed');
  const [attempt]=rows(f.root,"SELECT id FROM attempts WHERE project='queued'");
  const job=await load(join(f.root,'attempts',attempt.id,'job.json'));assert.equal(job.resource.id,'only');assert.equal(job.skillVersion,candidate);
  assert.equal(JSON.parse(rows(f.root,"SELECT value FROM control WHERE key='skill-background-resource'")[0].value),null);f.pass();
});

test('native campaign holds actors and curriculum across pause/resume until skill publication',{timeout:120000},async t=>{
  const f=await setup(t,'phase2',['held','queued']);
  const initial=await load(join(f.root,'instance.json')),settings=join(f.root,'gate-settings.json');
  const environment=join(f.root,'slots.mjs');
  await writeFile(environment,`import {createEnvironment as local} from './machinery/integrations/environments/index.mjs';
export const createEnvironment=(options,context)=>local(options,{...context,resources:async()=>[0,1].map(i=>({id:'local'+i,available:true,metadata:{}}))});\n`);
  await save(settings,{mode:'phase2',recovery:false,concurrency:2,campaign_seconds:120,pause_grace_seconds:0,
    environment:{module:environment,options:{mode:'local'}},
    curriculum:{phase2:['held','queued'],generation:{enabled:true,topics:['must not start while skill gate is armed']}},
    skills:{...initial.skills,background:true,promotionGate:{baseVersion:'baseline',status:'waiting'}}});
  await f.run('config','--config',settings);
  const waitGate=()=>until(()=>rows(f.root,"SELECT 1 FROM sqlite_master WHERE name='control'").length&&rows(f.root,"SELECT value FROM control WHERE key='wait'").some(r=>JSON.parse(r.value)?.reason==='skill-update-required'));
  let running=f.start('campaign');await waitGate();
  assert.equal(rows(f.root,'SELECT COUNT(*) n FROM attempts')[0].n,0);
  assert.equal(f.server.requests.length,0);
  assert.equal(rows(f.root,"SELECT COUNT(*) n FROM events WHERE type='curriculum-session'")[0].n,0);
  const before=JSON.parse(rows(f.root,"SELECT value FROM control WHERE key='campaign-budget'")[0].value);
  await f.run('pause');await running.output;
  assert.equal(skillPromotionPending(await load(join(f.root,'instance.json'))),true);
  const state=new State(join(f.root,'state.sqlite'));
  const held=admitJob(state,{projectId:'held',wave:1,owner:{pid:2147483647,start:'0',boot:'old'},resourceId:'local1',mode:'phase2'});
  state.checkpoint(held.attempt.id,3);state.close();
  const path=join(f.root,'attempts',held.attempt.id);await mkdir(path,{recursive:true});
  await save(join(path,'job.json'),{...held,wave:1,mode:'phase2',options:{},skillVersion:'baseline',resource:{id:'local1',available:true}});
  running=f.start('resume');await waitGate();
  const resumed=JSON.parse(rows(f.root,"SELECT value FROM control WHERE key='campaign-budget'")[0].value);
  assert.equal(resumed.cycle,before.cycle);assert.ok(resumed.elapsed>=before.elapsed);
  assert.equal(rows(f.root,'SELECT COUNT(*) n FROM attempts')[0].n,1);
  assert.equal(rows(f.root,`SELECT elapsed FROM attempts WHERE id='${held.attempt.id}'`)[0].elapsed,3);
  assert.equal(f.server.requests.length,0);
  assert.equal(rows(f.root,"SELECT COUNT(*) n FROM events WHERE type='curriculum-session'")[0].n,0);
  // Freeze the same one-project workload after proving generation could not run.
  await save(settings,{curriculum:{phase2:['held','queued'],generation:{enabled:false}}});await f.run('config','--config',settings);
  const version='approved-candidate';await cp(join(f.root,'skills/versions/baseline'),join(f.root,'skills/versions',version),{recursive:true});
  const published=await publishSkillVersion(f.root,{active:'baseline',candidate:version});
  assert.equal(published.skills.promotionGate.status,'released');
  await running.output;
  assert.equal(rows(f.root,"SELECT status FROM projects WHERE id='queued'")[0].status,'passed');
  const [attempt]=rows(f.root,"SELECT id FROM attempts WHERE project='queued'");
  const job=await load(join(f.root,'attempts',attempt.id,'job.json'));assert.equal(job.skillVersion,version);
  assert.equal(rows(f.root,'SELECT COUNT(*) n FROM attempts')[0].n,2);
  assert.equal(rows(f.root,`SELECT id FROM attempts WHERE project='held'`)[0].id,held.attempt.id);
  assert.equal((await load(join(path,'job.json'))).skillVersion,'baseline');
  f.pass();
});
