import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {syncCampaignPlan,currentCampaignEntry,advanceCampaignEntry,campaignProgress} from '../../controller/progress.mjs';
import {CampaignStorage} from '../../controller/storage.mjs';
import {prepareSourceGit} from '../../controller/source-git.mjs';
import {buildCampaignManifest,campaignEntryConfig,qualificationGate,bootstrapFreshCampaign,workloads} from '../../cli/campaign-manifest.mjs';
import {buildProjectCatalog,listProjectFiles,readProjectSource,projectCatalogExtension} from '../../cli/project-catalog.mjs';

const profile={baseline:'/private/base.sqlite',skills:'/private/baseline-skills',
  environment:{module:'machinery/integrations/environments/index.mjs',options:{mode:'local',runtime:'fixture-runtime'}},
  campaign_plan:[{id:'old',phase:'phase1',new_projects:99}],projects:[{id:'old'}],
  skillSettings:{activeVersion:'old-candidate',tuningDisabled:true,evaluationSeconds:999}};
const phase2Projects=Object.fromEntries(workloads.map(workload=>[workload,{goal:workload,source:'/private/'+workload,
  verifier:{files:[],command:'true',performance_command:'true'},performance_goal:'improve frozen baseline'}]));
const manifest=target=>buildCampaignManifest({campaignId:'fresh-study',target,profile,phase2Projects});

test('supplied-only native campaign entries retain configured actor budgets without a curriculum',()=>{
  const config={actor_seconds:17,campaign_plan:[{id:'supplied',phase:'phase2',new_projects:1}]};
  const current=campaignEntryConfig(config,'supplied');
  assert.equal(current.mode,'phase2');assert.equal(current.actor_seconds,17);
  assert.equal(current.curriculum.generation.enabled,undefined);assert.equal(config.curriculum,undefined);
});

test('both manifests drive exactly128 settled Phase1 and25 Phase2 originals through native progress',()=>{
  const before=structuredClone({profile,phase2Projects});
  for(const target of ['a3','a5']) {
    const config=manifest(target),state=new State(':memory:'),queue=new RecoveryQueue(state);
    try {
      syncCampaignPlan(state,config);
      for(const project of config.projects)state.add(project);
      const counts={phase1:0,phase2:0},topics=new Map(),ids=new Set();
      for(const entry of config.campaign_plan) {
        const active=campaignEntryConfig(config,entry.id);
        assert.equal(currentCampaignEntry(state,config).id,entry.id);
        let projects=config.projects.filter(project=>project.campaign_entry===entry.id);
        if(entry.phase==='phase1') {
          assert.equal(active.curriculum.generation.topics.length,8);
          projects=active.curriculum.generation.topics.map((topic,index)=>{
            topics.set(topic,(topics.get(topic)??0)+1);
            const project={id:entry.id+'-'+index,phase:'phase1',campaign_entry:entry.id,
              goal:topic,actor_seconds:active.actor_seconds};state.add(project);return project;
          });
          assert.equal(Object.values(active.curriculum.generation.requirements)[0].performance,true);
          assert.equal(Object.values(active.curriculum.generation.requirements)[0].capabilityEvidence,true);
        }
        for(const project of projects) {
          assert.equal(ids.has(project.id),false);ids.add(project.id);
          assert.equal(project.actor_seconds,entry.phase==='phase1'?7200:28800);
          const job=queue.claim(project.id,0);
          queue.recordOriginal(job.attempt.id,{passed:counts[entry.phase]%2===0,findings:'observed',evidence:[{collection:'fixture',entry_id:job.attempt.id}]},0);
          assert.equal(currentCampaignEntry(state,config).finishedOriginals,projects.indexOf(project));
          queue.acknowledgeLesson(queue.lessons().find(lesson=>lesson.attempt_id===job.attempt.id).id);
          state.event(job.attempt.id,'postprocess-completed',{});counts[entry.phase]++;
        }
        assert.equal(currentCampaignEntry(state,config).remainingOriginals,0);
        assert.equal(advanceCampaignEntry(state,config),true);
      }
      assert.deepEqual(counts,{phase1:128,phase2:25});assert.equal(campaignProgress(state,config).done,true);
      assert.deepEqual([...topics.values()].sort((a,b)=>a-b),[14,14,14,14,14,14,14,14,16]);
      const controls=config.skillSettings.tuningProjectIds.map(id=>config.projects.find(project=>project.id===id));
      assert.deepEqual(controls.map(project=>project.campaign_workload),['fa','gdn']);
      assert.equal(config.curriculum_seconds,1800);assert.equal(config.recovery,false);assert.equal(config.concurrency,8);
      assert.equal(config.skillSettings.background,true);assert.equal(config.skillSettings.tuningDisabled,false);
      assert.equal(config.skillSettings.evaluationSeconds,undefined);assert.equal(config.skillSettings.activeVersion,'baseline');
      for(const project of config.projects)assert.deepEqual(project.verifier,phase2Projects[project.campaign_workload].verifier);
    }finally{state.close();}
  }
  assert.deepEqual({profile,phase2Projects},before);
});

function qualification() {
  const expected={sourceRevision:'frozen-source'},receipts=[];
  for(const target of ['a3','a5']) {
    expected[target]={};
    for(const workload of workloads) {
      expected[target][workload]={sourceHash:'source',caseHash:'cases',verifierHash:'verifier',independentCheckHash:'checker',runtimeFingerprint:target+'-runtime',caseIds:[1,2]};
      receipts.push({format:1,qualified:true,target,workload,sourceHash:'source',caseHash:'cases',verifierHash:'verifier',
        runtime:{fingerprint:target+'-runtime'},independentChecks:{workload,passed:true,checkerHash:'checker'},
        cases:[1,2].map(id=>({id,passed:true,referenceHash:'reference',outputHash:'output',maxAbsError:0})),
        timing:{passed:true,samples:[1,2].map(id=>({id,seconds:[.1,.2]}))}});
    }
  }
  return {localValidation:{passed:true,sourceRevision:'frozen-source'},expected,receipts};
}

test('launch gate rejects stale local checks and missing, mismatched, failed or incomplete target evidence before bootstrap',async()=>{
  const cases=[q=>q.localValidation.passed=false,q=>q.localValidation.sourceRevision='old',q=>q.receipts.pop(),
    q=>q.receipts[0].runtime.fingerprint='other-runtime',q=>q.receipts[0].sourceHash='changed',
    q=>q.receipts[0].cases[0].passed=false,q=>q.receipts[0].cases.pop(),
    q=>q.receipts[0].timing.samples[0].seconds=[0],q=>q.receipts[0].timing.samples.pop(),
    q=>delete q.receipts[0].independentChecks,q=>q.receipts[0].independentChecks.checkerHash='different',
    q=>q.receipts[0].independentChecks.passed=false];
  for(const change of cases) {
    const q=qualification();change(q);
    await assert.rejects(bootstrapFreshCampaign({destination:'/not-created',campaignId:'fresh-study',target:'a3',
      profile,phase2Projects,qualification:q},{create:()=>assert.fail('bootstrap admitted without qualification')}));
  }
  const inputs=[];
  for(const target of ['a3','a5'])await bootstrapFreshCampaign({destination:'/new/'+target,
    campaignId:'fresh-study',target,profile,phase2Projects,qualification:qualification()},
  {create:async(destination,config)=>inputs.push({destination,config})});
  assert.deepEqual(inputs.map(input=>input.destination),['/new/a3','/new/a5']);
  assert.notEqual(inputs[0].config,inputs[1].config);
  assert.equal(inputs.every(input=>input.config.baseline===profile.baseline&&input.config.skills===profile.skills),true);
  assert.equal(inputs.some(input=>input.config.projects.some(project=>project.id==='old')),false);
  assert.equal(qualificationGate(qualification()).qualified,14);
});

test('catalog reads immutable prior source from both fresh targets, withholding live or unsettled work',async t=>{
  const root=await mkdtemp(join(tmpdir(),'fresh-project-catalog-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const campaigns=[],ledgers=[];
  t.after(()=>ledgers.forEach(state=>state.close()));
  for(const target of ['a3','a5']) {
    const path=join(root,target),source=join(path,'projects','p','source'),gitDir=join(path,'source-history','p');
    await mkdir(source,{recursive:true});await writeFile(join(source,'kernel.py'),'answer = 42\n');
    const state=new State(join(path,'state.sqlite'));ledgers.push(state);
    const queue=new RecoveryQueue(state),storage=new CampaignStorage(state,path);
    await storage.initialize({id:'fresh-'+target,target});
    state.add({id:'p',goal:'reference',actor_seconds:10,source,phase:'phase1'});
    const job=queue.claim('p',0);storage.attempt(job.attempt.id,{phase:'phase1'});
    await mkdir(join(path,'attempts',job.attempt.id),{recursive:true});
    await writeFile(join(path,'attempts',job.attempt.id,'job.json'),JSON.stringify({resource:{metadata:{runtime:'actual-runtime',product:target}}}));
    const repo=await prepareSourceGit({source,gitDir,attemptId:job.attempt.id});
    const snapshot=await repo.captureTree();await storage.register(gitDir,{project:'p',kind:'source-git'});
    storage.iteration(job.attempt.id,{stage:'optimisation',sourceHash:snapshot.tree_oid,sourceCommit:snapshot.head});
    await writeFile(join(path,'instance.json'),JSON.stringify({campaign_cohort:'new-study',target,kernel_language:target==='a3'?'triton':'catlass-dsl',
      storage:{campaign_id:'fresh-'+target},environment:profile.environment}));
    campaigns.push({root:path,campaignId:'fresh-'+target,target});
    queue.recordOriginal(job.attempt.id,{passed:false,findings:'observed',evidence:[{collection:'fixture',entry_id:job.attempt.id}]},0);
    queue.acknowledgeLesson(queue.lessons()[0].id);
    state.event(job.attempt.id,'postprocess-completed',{});
    state.event(job.attempt.id,'failure-settlement',{closed:target==='a3'});
    await writeFile(join(source,'kernel.py'),'answer = 99\n');
  }
  const options={cohort:'new-study',campaigns};
  let entries=await buildProjectCatalog(options);assert.equal(entries.length,1);
  assert.deepEqual(await listProjectFiles(entries[0]),['kernel.py']);
  assert.equal(await readProjectSource(entries[0],'kernel.py'),'answer = 42\n');
  await assert.rejects(readProjectSource(entries[0],'../instance.json'),/outside/);
  const before=await readFile(join(campaigns[0].root,'instance.json'),'utf8');
  const a5=ledgers[1],attempt=a5.db.prepare('SELECT id FROM attempts').get().id;
  a5.event(attempt,'failure-settlement',{closed:true});entries=await buildProjectCatalog(options);
  assert.deepEqual(entries.map(entry=>entry.target),['a3','a5']);
  assert.equal(entries.every(entry=>entry.runtime==='actual-runtime'&&entry.product===entry.target&&entry.outcome==='failed'),true);
  const events=[];let tool;await projectCatalogExtension({...options,event:(type,data)=>events.push({type,data})})({registerTool:value=>tool=value});
  const first=(await tool.execute('call',{action:'list',limit:1})).details;
  assert.equal(first.projects.length,1);assert.equal(first.total,2);assert.equal(first.nextOffset,1);
  const second=(await tool.execute('call',{action:'list',offset:first.nextOffset,limit:1})).details;
  assert.equal(second.projects[0].target,'a5');assert.equal(second.nextOffset,null);
  const selected=entries[1];assert.equal((await tool.execute('call',{action:'read',id:selected.id,path:'kernel.py'})).details,'answer = 42\n');
  assert.deepEqual(events.at(-1).data,{action:'read',campaignId:selected.campaignId,projectId:selected.projectId,
    attemptId:selected.attemptId,target:'a5',runtime:'actual-runtime',tree:selected.source.tree,commit:selected.source.commit,path:'kernel.py'});
  const choice=(await tool.execute('call',{action:'select',id:selected.id})).details;
  assert.equal(choice.tree,selected.source.tree);assert.equal(events.at(-1).type,'starter-selected');
  await tool.execute('call',{action:'select',starter:'scratch'});assert.deepEqual(events.at(-1).data,{starter:'scratch'});
  await tool.execute('call',{action:'select',starter:'baseline'});assert.deepEqual(events.at(-1).data,{starter:'baseline'});
  assert.equal(await readFile(join(campaigns[0].root,'instance.json'),'utf8'),before);
  await assert.rejects(buildProjectCatalog({...options,cohort:'historical'}),/outside/);
});
