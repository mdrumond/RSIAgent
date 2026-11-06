import {test} from 'node:test';
import assert from 'node:assert/strict';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {campaignProgress} from '../../controller/progress.mjs';

function fixture(t) {
  const state = new State(':memory:');
  t.after(() => state.close());
  const queue = new RecoveryQueue(state);
  function start(id, phase = 'phase1', mode) {
    state.add({id, goal: 'test', actor_seconds: 10, ...(phase ? {phase} : {})});
    const job = queue.claim(id, 0);
    if (mode !== undefined) state.db.prepare("UPDATE events SET data=json_set(data,'$.mode',?) WHERE attempt=? AND type='job-admitted'")
      .run(mode, job.attempt.id);
    else state.db.prepare("UPDATE events SET data=json_remove(data,'$.mode') WHERE attempt=? AND type='job-admitted'")
      .run(job.attempt.id); // Historical admissions exercise spec/list fallback.
    return job.attempt.id;
  }
  function finish(id, passed = true, ack = true, post = true) {
    queue.recordOriginal(id, {passed, findings: 'observed behavior', evidence: [{collection: 'memory', entry_id: id}]}, 0);
    if (ack) queue.acknowledgeLesson(queue.lessons().find(lesson => lesson.attempt_id === id).id);
    if (post) state.event(id, 'postprocess-completed', {});
  }
  return {state, queue, start, finish};
}

test('completion requires terminal outcome, memory acknowledgement, postprocessing and latest settlement', t => {
  const {state, queue, start, finish} = fixture(t);
  const id = start('one');
  const config = {mode: 'phase1', completion_target: 1, phase_sequence: ['phase1', 'phase2']};
  assert.equal(campaignProgress(state, config).finishedOriginals, 0);
  finish(id, false, false, false);
  assert.equal(campaignProgress(state, config).phaseComplete, false);
  queue.acknowledgeLesson(queue.lessons()[0].id);
  assert.equal(campaignProgress(state, config).phaseComplete, false);
  state.event(id, 'postprocess-completed', {});
  state.event(id, 'failure-settlement', {closed: false});
  assert.equal(campaignProgress(state, config).finishedOriginals, 0);
  state.event(id, 'failure-settlement', {closed: true});
  const result = campaignProgress(state, config);
  assert.equal(result.finishedOriginals, 1);
  assert.equal(result.nextMode, 'phase2');
  assert.equal(result.done, false);
  assert.ok(queue.entry('one'), 'pending recovery does not block advance');
  const investigation = queue.claim('one', 1);
  queue.complete(investigation.attempt.id, {passed: true, useful: true, findings: 'retry supported',
    evidence: [{collection: 'memory', entry_id: 'investigation'}]}, 1);
  queue.acknowledgeLesson(queue.lessons()[0].id);
  state.event(investigation.attempt.id, 'postprocess-completed', {});
  assert.equal(campaignProgress(state, config).finishedOriginals, 1);
  const retry = queue.claim('one', 2);
  queue.complete(retry.attempt.id, {passed: true, findings: 'passed', evidence: [{collection: 'memory', entry_id: 'retry'}]}, 2);
  queue.acknowledgeLesson(queue.lessons()[0].id);
  state.event(retry.attempt.id, 'postprocess-completed', {});
  assert.equal(campaignProgress(state, config).totalOriginals, 1);
});

test('phase scope uses admission then spec then unambiguous configured IDs; unknown stays unknown', t => {
  const {state, start, finish} = fixture(t);
  finish(start('admission', 'phase1', 'phase2'));
  finish(start('spec', 'phase1'));
  finish(start('fallback', null));
  finish(start('ambiguous', null));
  finish(start('unknown', null));
  finish(start('invalid', 'bogus'));
  const result = campaignProgress(state, {mode: 'phase1', curriculum: {
    phase1: ['fallback', 'ambiguous', 'spec'], phase2: ['ambiguous', 'spec']}});
  assert.deepEqual(result.phaseCounts, {
    phase1: {finishedOriginals: 2, totalOriginals: 2},
    phase2: {finishedOriginals: 1, totalOriginals: 1},
    inference: {finishedOriginals: 0, totalOriginals: 0}});
  assert.equal(result.phaseComplete, false);
});

test('explicit targets override list completion; generated unbounded phases keep running', t => {
  const {state, start, finish} = fixture(t);
  finish(start('one'));
  const config = {mode: 'phase1', curriculum: {phase1: ['one', 'one']}};
  assert.equal(campaignProgress(state, config).done, true);
  assert.equal(campaignProgress(state, {...config, completion_target: 2}).phaseComplete, false);
  assert.equal(campaignProgress(state, {...config, completion_target: 2, phase_targets: {phase1: 1}}).done, true);
  const generated = {...config, curriculum: {...config.curriculum, generation: {enabled: true}}};
  assert.equal(campaignProgress(state, generated).phaseComplete, false);
  assert.equal(campaignProgress(state, {...generated, completion_target: 1}).done, true);
  assert.equal(campaignProgress(state, {mode: 'phase1'}).target, null);
  assert.equal(campaignProgress(state, {mode: 'phase1'}).phaseComplete, false);
});

test('malformed settlement and admission cannot create completed originals', t => {
  const {state, start, finish} = fixture(t);
  for (const [index, data] of ['{', '{}', '{"closed":null}', '{"closed":0}'].entries()) {
    const id = start(`bad-${index}`);
    finish(id);
    state.event(id, 'failure-settlement', {});
    state.db.prepare("UPDATE events SET data=? WHERE attempt=? AND type='failure-settlement'").run(data, id);
  }
  const malformed = start('bad-admission'); finish(malformed);
  state.db.prepare("UPDATE events SET data='{' WHERE attempt=? AND type='job-admitted'").run(malformed);
  assert.equal(campaignProgress(state, {mode: 'phase1'}).finishedOriginals, 0);
});

test('duplicate original attempts count once and retain original admission phase', t => {
  const {state, start, finish} = fixture(t);
  finish(start('one'));
  state.db.prepare("UPDATE projects SET status='queued' WHERE id='one'").run();
  const duplicate = state.admit('one', {});
  state.event(duplicate.id, 'job-admitted', {kind: 'original', mode: 'phase2'});
  finish(duplicate.id);
  state.events = () => {throw new Error('unbounded history API');};
  const result = campaignProgress(state, {mode: 'phase1'});
  assert.equal(result.finishedOriginals, 1);
  assert.equal(result.phaseCounts.phase2.totalOriginals, 0);
});

test('validates configuration without imposing a built-in target', t => {
  const {state} = fixture(t);
  for (const config of [
    {phase_sequence: []}, {phase_sequence: ['phase1', 'phase1']}, {phase_sequence: ['bad']},
    {phase_sequence: ['phase2']}, {completion_target: 0}, {completion_target: 1.5},
    {phase_targets: {phase1: -1}}, {phase_targets: {bad: 1}},
  ]) assert.throws(() => campaignProgress(state, {mode: 'phase1', ...config}));
  assert.equal(campaignProgress(state, {mode: 'phase1', completion_target: 128}).target, 128);
});

test('fresh State is readable before RecoveryQueue setup', t => {
  const state = new State(':memory:'); t.after(() => state.close());
  state.add({id: 'one', phase: 'phase1', goal: 'test', actor_seconds: 1});
  const attempt = state.admit('one', {});
  state.finish(attempt.id, 'passed');
  state.event(attempt.id, 'postprocess-completed', {});
  assert.equal(campaignProgress(state, {mode: 'phase1'}).finishedOriginals, 0);
});

test('historical event noise and later settlement reopening preserve completion gates', t => {
  const {state, start, finish} = fixture(t);
  const id = start('one'); finish(id);
  for (let index = 0; index < 250; index++) state.event(id, 'actor-message', {index});
  state.event(id, 'failure-settlement', {closed: false});
  state.events = () => {throw new Error('history must remain in SQL');};
  state.queryEvents = () => {throw new Error('history must remain in SQL');};
  assert.equal(campaignProgress(state, {mode: 'phase1', completion_target: 1}).phaseComplete, false);
  state.event(id, 'failure-settlement', {closed: true});
  assert.equal(campaignProgress(state, {mode: 'phase1', completion_target: 1}).done, true);
});

test('campaign entries repeat phases without recounting originals or draining recovery', async t => {
  const {currentCampaignEntry, syncCampaignPlan, advanceCampaignEntry} = await import('../../controller/progress.mjs');
  const {state, queue} = fixture(t);
  const config = {mode:'phase1', campaign_plan:[
    {id:'r1p1',phase:'phase1',new_projects:1}, {id:'r1p2',phase:'phase2',new_projects:1},
    {id:'r2p1',phase:'phase1',new_projects:1}, {id:'r2p2',phase:'phase2',new_projects:1}], projects:[]};
  for (const [index, entry] of config.campaign_plan.entries()) {
    const spec = {id:'project'+index, goal:'test',actor_seconds:10,campaign_entry:entry.id,phase:entry.phase};
    config.projects.push(spec);state.add(spec);
  }
  syncCampaignPlan(state,config);
  for (const [index, entry] of config.campaign_plan.entries()) {
    assert.equal(currentCampaignEntry(state,config).id,entry.id);
    assert.equal(currentCampaignEntry(state,config).finishedOriginals,0);
    if(index<3)assert.throws(()=>queue.claim('project'+(index+1),index),/outside/);
    const job=queue.claim('project'+index,index);
    assert.equal(state.events(job.attempt.id).find(e=>e.type==='job-admitted').data.campaign_entry,entry.id);
    assert.equal(currentCampaignEntry(state,config).remainingAdmissions,0);
    assert.equal(currentCampaignEntry(state,config).remainingOriginals,1);
    queue.recordOriginal(job.attempt.id,{passed:false,findings:'observed',evidence:[{collection:'memory',entry_id:'one'}]},index);
    assert.equal(advanceCampaignEntry(state,config),false);
    queue.acknowledgeLesson(queue.lessons()[0].id);
    assert.equal(advanceCampaignEntry(state,config),false);
    state.event(job.attempt.id,'postprocess-completed',{});
    state.event(job.attempt.id,'failure-settlement',{closed:false});
    assert.equal(advanceCampaignEntry(state,config),false);
    state.event(job.attempt.id,'failure-settlement',{closed:true});
    assert.equal(currentCampaignEntry(state,config).remainingOriginals,0);
    assert.equal(advanceCampaignEntry(state,config),true);
  }
  assert.equal(currentCampaignEntry(state,config),null);
  assert.equal(campaignProgress(state,config).done,true);
  assert.equal(queue.ready(9).length,4);
});

test('campaign freezes started entries and admitted membership but allows future edits and append', async t=>{
  const {currentCampaignEntry,syncCampaignPlan,startCampaignEntry,advanceCampaignEntry}=await import('../../controller/progress.mjs');
  const {state,queue}=fixture(t);
  const first={id:'first',phase:'phase1',new_projects:1};
  const config={mode:'phase1',campaign_plan:[first,{id:'future',phase:'phase2',new_projects:2}],projects:[]};
  currentCampaignEntry(state,config);
  startCampaignEntry(state,config);
  assert.throws(()=>syncCampaignPlan(state,{...config,campaign_plan:[{...first,new_projects:2}]}),/started/);
  assert.throws(()=>syncCampaignPlan(state,{mode:'phase1'}),/remove/);
  config.campaign_plan[1]={id:'future',phase:'phase1',new_projects:3};
  syncCampaignPlan(state,config);
  const spec={id:'one',goal:'test',actor_seconds:10,campaign_entry:'first'};
  state.add(spec);config.projects=[spec];
  const job=queue.claim('one',0);
  assert.throws(()=>syncCampaignPlan(state,{...config,projects:[{...spec,campaign_entry:'future'}]}),/admitted/);
  queue.recordOriginal(job.attempt.id,{passed:true,findings:'observed',evidence:[{collection:'memory',entry_id:'one'}]},0);
  queue.acknowledgeLesson(queue.lessons()[0].id);state.event(job.attempt.id,'postprocess-completed',{});
  advanceCampaignEntry(state,config);
  assert.equal(currentCampaignEntry(state,config).id,'future');
  startCampaignEntry(state,config);
  config.campaign_plan.push({id:'appended',phase:'phase2',new_projects:1});
  assert.equal(currentCampaignEntry(state,config).remainingOriginals,3);
  assert.throws(()=>syncCampaignPlan(state,{...config,campaign_plan:[first]}),/started/);
});

test('status does not start entries; admission freezes membership and caps original slots', async t=>{
  const {currentCampaignEntry,syncCampaignPlan}=await import('../../controller/progress.mjs');
  const {state,queue}=fixture(t);
  const first={id:'first',phase:'phase1',new_projects:1},second={id:'second',phase:'phase2',new_projects:1};
  let config={mode:'phase1',campaign_plan:[first,second]};
  assert.equal(currentCampaignEntry(state,config).id,'first');
  assert.equal(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get(),undefined);
  syncCampaignPlan(state,config);
  assert.equal(campaignProgress(state,config).plan.lockedPrefix,0);
  config={...config,campaign_plan:[second,first]};
  syncCampaignPlan(state,config);
  assert.equal(currentCampaignEntry(state,config).id,'second');
  for(const id of ['one','two'])state.add({id,goal:'test',actor_seconds:10,campaign_entry:'second'});
  const job=queue.claim('one',0);
  assert.equal(campaignProgress(state,config).plan.lockedPrefix,1);
  assert.equal(campaignProgress(state,config).plan.entries[0].state,'started');
  assert.throws(()=>queue.claim('two',0),/outside/);
  assert.throws(()=>syncCampaignPlan(state,{...config,campaign_plan:[first,second]}),/started/);
  // Frozen admission metadata keeps counts stable even if a spec is changed outside the API.
  state.db.prepare("UPDATE projects SET spec=json_set(spec,'$.campaign_entry','first') WHERE id='one'").run();
  assert.equal(currentCampaignEntry(state,config).totalOriginals,1);
  assert.throws(()=>syncCampaignPlan(state,{...config,projects:[{id:'one',campaign_entry:'first'}]}),/admitted/);
  queue.recordOriginal(job.attempt.id,{passed:true,findings:'observed',evidence:[{collection:'memory',entry_id:'one'}]},0);
  queue.acknowledgeLesson(queue.lessons()[0].id);state.event(job.attempt.id,'postprocess-completed',{});
  assert.equal(currentCampaignEntry(state,config).finishedOriginals,1);
});

test('configuration rejects invalid assignments and frozen edits transactionally',async t=>{
  const {mkdtemp,writeFile,rm}=await import('node:fs/promises');
  const {join}=await import('node:path');
  const {tmpdir}=await import('node:os');
  const {updateConfig,readConfig}=await import('../../cli/instance.mjs');
  const {startCampaignEntry}=await import('../../controller/progress.mjs');
  const root=await mkdtemp(join(tmpdir(),'plan-config-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const config={format:1,mode:'phase1',campaign_plan:[{id:'first',phase:'phase1',new_projects:1}],projects:[]};
  await writeFile(join(root,'instance.json'),JSON.stringify(config));
  await updateConfig(root,current=>({...current}));
  // Unstarted first entry can change through the actual configuration API.
  await updateConfig(root,current=>({...current,campaign_plan:[{id:'renamed',phase:'phase2',new_projects:2}]}));
  await assert.rejects(updateConfig(root,(current,ledger)=>{
    ledger.add({id:'invalid',goal:'test',actor_seconds:10,campaign_entry:'missing'});
    return {...current,projects:[{id:'invalid',campaign_entry:'missing'}]};
  }),/matching/);
  const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  assert.equal(state.project('invalid'),undefined);
  startCampaignEntry(state,await readConfig(root));
  await assert.rejects(updateConfig(root,current=>({...current,campaign_plan:[{id:'changed',phase:'phase1',new_projects:1}]})),/started/);
  assert.equal((await readConfig(root)).campaign_plan[0].id,'renamed');
});

test('a new plan attaches to unassigned history without recounting it or excluding recovery',async t=>{
  const {mkdtemp,writeFile,rm}=await import('node:fs/promises');
  const {join}=await import('node:path');
  const {tmpdir}=await import('node:os');
  const {updateConfig,readConfig}=await import('../../cli/instance.mjs');
  const {currentCampaignEntry}=await import('../../controller/progress.mjs');
  const {selectAdmission}=await import('../../controller/dispatch.mjs');
  const root=await mkdtemp(join(tmpdir(),'plan-continuation-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  const projects=['passed','failed','queued'].map(id=>({id,goal:id,actor_seconds:10,
    phase:id==='failed'?'phase2':'phase1'}));
  for(const project of projects)state.add(project);
  const queue=new RecoveryQueue(state),originQueue=new RecoveryQueue(state,{mode:'phase2'});
  for(const [id,owner] of [['passed',queue],['failed',originQueue]]) {
    const job=owner.claim(id,0);
    owner.recordOriginal(job.attempt.id,{passed:id==='passed',findings:'observed',
      evidence:[{collection:'memory',entry_id:id}]},0);
    owner.acknowledgeLesson(owner.lessons()[0].id);
    state.event(job.attempt.id,'postprocess-completed',{});
  }
  const oldRecovery=queue.entry('failed');
  await writeFile(join(root,'instance.json'),JSON.stringify({format:1,mode:'phase1',projects}));
  const fresh={id:'fresh',goal:'new batch',actor_seconds:10,phase:'phase1',campaign_entry:'next'};
  const config=await updateConfig(root,(current,ledger)=>{
    ledger.add(fresh);
    return {...current,projects:[...current.projects,fresh],
      campaign_plan:[{id:'next',phase:'phase1',new_projects:1}]};
  });
  assert.equal(currentCampaignEntry(state,config).finishedOriginals,0);
  assert.equal(currentCampaignEntry(state,config).remainingOriginals,1);
  assert.equal(currentCampaignEntry(state,config).totalOriginals,0);
  assert.throws(()=>queue.claim('queued',1),/outside/);
  assert.equal(state.project('queued').status,'queued');
  const jobs=queue.jobs(1),recovery=jobs.find(job=>job.projectId==='failed');
  assert.equal(recovery.kind,'investigation');assert.equal(recovery.origin_attempt,oldRecovery.origin);
  assert.equal(selectAdmission([jobs.find(job=>job.projectId==='fresh'),recovery],
    {freshSinceRecovery:3}).projectId,'failed','carried recovery retains scheduler fairness');
  const admitted=queue.claim('fresh',1);
  assert.equal(admitted.kind,'original');
  assert.equal(currentCampaignEntry(state,config).remainingAdmissions,0);
  const originAdmission=state.events(oldRecovery.origin).find(event=>event.type==='job-admitted');
  assert.equal(originAdmission.data.mode,'phase2');
  assert.equal(state.project('failed').spec.campaign_entry,undefined);
  assert.deepEqual(queue.entry('failed'),oldRecovery);
  await assert.rejects(updateConfig(root,(current,ledger)=>{
    // Even a caller updating the mutable spec cannot rewrite admission ownership.
    ledger.db.prepare("UPDATE projects SET spec=json_set(spec,'$.campaign_entry','next') WHERE id='passed'").run();
    return {...current,projects:current.projects.map(project=>project.id==='passed'?{...project,campaign_entry:'next'}:project)};
  }),/admitted/);
  assert.equal(state.project('passed').spec.campaign_entry,undefined);
  for(const assignment of [null,'missing'])await assert.rejects(updateConfig(root,current=>({...current,
    projects:current.projects.map(project=>project.id==='queued'?{...project,campaign_entry:assignment}:project)})),/matching/);
  assert.equal((await readConfig(root)).projects.find(project=>project.id==='passed').campaign_entry,undefined);
});
