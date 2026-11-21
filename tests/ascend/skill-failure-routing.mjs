import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,rm,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {State} from '../../controller/state.mjs';
import {claimLearningRequest} from '../../controller/skill-review.mjs';
import {SkillBacklog} from '../../controller/skill-backlog.mjs';
import {analyzeBacklog} from '../../cli/skill-backlog-analysis.mjs';
import {runCampaignFailureStep} from '../../cli/skill-failure-campaign.mjs';

async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'skill-routing-'));
 const state=new State(join(root,'state.sqlite'));
 t.after(async()=>{state.close();await rm(root,{recursive:true,force:true});});
 const directory=join(root,'analysis'),version=join(root,'skills','versions','base');
 await mkdir(version,{recursive:true});await writeFile(join(version,'guidance.json'),'[]');
 const config={root,target:'a5',skills:{activeVersion:'base',failureFirst:{enabled:true}}};
 const backlog=new SkillBacklog(state,{target:'a5'});
 const analyze=classify=>analyzeBacklog({config,backlog,directory,classificationOnly:true},{classify});
 const start=id=>{state.add({id,phase:'phase1',source:root,goal:'Verify a kernel',actor_seconds:180});return state.admit(id,{pid:1,start:'fixture',boot:'fixture'}).id;};
 const verify=async(attempt,path,log)=>{
  const evidence=join(root,path);await writeFile(evidence,log);
  state.event(attempt,'verification',{passed:false,correctness:{passed:false,exitCode:1,evidence,evidence_sha256:createHash('sha256').update(log).digest('hex')}});
 };
 return {state,config,backlog,analyze,start,verify};
}
const classification=(group,{bucket='unknown',cause='unknown'}={})=>({classifications:[{groupId:group.id,bucket,cause,
 reason:'Observed runtime status',coverage:'unknown',guidanceIds:[],evidenceRefs:[group.occurrences[0].reference]}]});

test('a hash-bound verifier traceback does not establish ownership; corrected submission is retained without a device trial',async t=>{
 const f=await fixture(t),attempt=f.start('first');
 await f.verify(attempt,'first.log','RuntimeError: got npu:0 and cpu tensors');
 f.backlog.ingest();
 const {items:[item]}=await f.analyze(({groups:[group],validate})=>{
  assert.throws(()=>validate(classification(group,{bucket:'correctness',cause:'actor'})),/runtime boundary/);
  assert.throws(()=>validate(classification(group,{bucket:'harness',cause:'harness'})),/runtime boundary/);
  const result=classification(group);
  result.classifications[0].triage={route:'baseline'};
  return validate(result);
 });
 assert.equal(item.status,'deferred');
 assert.equal(item.objective,'unknown');
 assert.equal(item.preparation.classifications[item.groupIds[0]].triage.route,'unresolved');
 assert.equal(f.backlog.claim({base:'base',workflow:'failure-first-v1'}),null);
});

test('raw verifier exits retain a replay question while hash-bound candidate verdicts open baseline screening',async t=>{
 for(const receipt of [false,true]){
  const f=await fixture(t),attempt=f.start('first');
  const output=receipt?JSON.stringify({correct:false,kernelSha256:'a'.repeat(64)}):'TypeError: incompatible operand type';
  await f.verify(attempt,'verifier.log',output);
  f.backlog.ingest();
  const {items:[item]}=await f.analyze(({groups:[group],validate})=>validate(classification(group,{bucket:'correctness'})));
  assert.equal(item.status,receipt?'queued':'deferred');
  const {route,question}=item.preparation.classifications[item.groupIds[0]].triage;
  assert.equal(route,receipt?'baseline':'unresolved');
  assert.equal(Boolean(question),!receipt);
  assert.equal(item.proposal,null);
 }
});

test('a failed target baseline creates a Phase 1 learning request and no runnable A/B item',async t=>{
 const f=await fixture(t),attempt=f.start('first');
 f.state.event(attempt,'project-invalid',{reason:'target-baseline-failed',passed:false,exitCode:1});
 f.backlog.ingest();
 const {items:[item]}=await f.analyze(({groups:[group],validate})=>validate(classification(group,{bucket:'correctness',cause:'specification'})));
 assert.equal(item.status,'learning');
 const [request]=f.state.db.prepare('SELECT data,status FROM skill_learning_requests').all();
 assert.equal(request.status,'pending');
 assert.equal(JSON.parse(request.data).backlogId,item.id);
 assert.equal(JSON.parse(request.data).workKind,'phase1');
 assert.equal(claimLearningRequest(f.state,f.config,{draft:'draft',target:'a5'}).backlogId,item.id);
 assert.equal(f.backlog.claim({base:'base',workflow:'failure-first-v1'}),null);
});

test('identical verifier diagnostics preserve all records and count distinct attempts once',async t=>{
 const f=await fixture(t),first=f.start('first'),second=f.start('second');
 await f.verify(first,'verify-1.log','Same diagnostic');
 f.backlog.ingest();
 const [original]=f.backlog.failures();f.backlog.acknowledge(original.id);
 await f.verify(first,'verify-2.log','Same diagnostic');
 f.backlog.ingest();
 assert.equal(f.backlog.failures().length,0);
 await f.verify(second,'verify-3.log','Same diagnostic');
 f.backlog.ingest();
 const [group,...other]=f.backlog.failures();
 assert.equal(group.id,original.id);
 assert.equal(other.length,0);
 assert.equal(group.count,3);
 assert.deepEqual(group.frequency,{projects:2,attempts:2,drafts:0});
});

test('old queued ownership is sent back to classification before consuming a device family',async t=>{
 const f=await fixture(t),attempt=f.start('first');
 await f.verify(attempt,'old.log','RuntimeError: got npu:0 and cpu tensors');
 f.backlog.ingest();
 const [group]=f.backlog.failures();f.backlog.acknowledge(group.id);
 const item=f.backlog.upsert({key:'failure-first:'+group.id,workflow:'failure-first-v1',objective:'correctness',base:'base',priority:10,
  groupIds:[group.id],evidence:{groups:[group]},preparation:{classifications:{[group.id]:classification(group).classifications[0]}}});
 const result=await runCampaignFailureStep(f.config,{slots:[0,1,2,3],remainingSeconds:100,memory:{}},{
  analyze:()=>f.analyze(({groups:[current],validate})=>validate(classification(current))),
  family:()=>assert.fail('Unwitnessed ownership must not consume an experiment')});
 assert.equal(result.status,'idle');
 assert.equal(f.backlog.get(item.id).status,'deferred');
 assert.equal(f.backlog.failures().length,0);
});

test('optimization ranks verified speed gains on related successes and keeps the performance failure out of correctness screening',async t=>{
 const f=await fixture(t);
 for(const id of ['faster','unmeasured']){
  const attempt=f.start(id);
  f.state.event(attempt,'verification',{correctness:{passed:true,exitCode:0},performance:{passed:false,exitCode:1,detail:id}});
  if(id==='faster'){
   const evidence=join(f.config.root,'perf.log'),log=JSON.stringify({geometricMeanSpeedup:2});
   await writeFile(evidence,log);
   f.state.event(attempt,'verification',{passed:true,correctness:{passed:true,exitCode:0},
    performance:{passed:true,exitCode:0,evidence,evidence_sha256:createHash('sha256').update(log).digest('hex')}});
  }
 }
 f.backlog.ingest();
 const {items}=await f.analyze(({groups,validate})=>validate({classifications:groups.flatMap(group=>classification(group,{bucket:'optimization'}).classifications)}));
 const [measured,unmeasured]=items.sort((a,b)=>b.priority-a.priority);
 assert.equal(measured.status,'deferred');
 assert.equal(unmeasured.status,'deferred');
 assert.equal(measured.priority-unmeasured.priority,10);
 assert.equal(measured.preparation.measurements[0].geometricMeanSpeedup,2);
 assert.equal(measured.evidence.groups[0].occurrences[0].projectId,'faster');
 assert.equal(f.backlog.claim({base:'base',workflow:'failure-first-v1'}),null);
});

test('a native provider failure stays in human review even when the actor reports a correctness hypothesis',async t=>{
 const f=await fixture(t),attempt=f.start('provider-stop');
 f.state.event(attempt,'actor-blocker-reported',{categoryHypothesis:'correctness'});
 f.state.event(attempt,'rejected',{actorStatus:{kind:'provider-error',report:'submitted'}});
 f.backlog.ingest();
 const {items:[item]}=await f.analyze(({groups:[group],validate})=>validate(classification(group,{bucket:'correctness',cause:'provider'})));
 assert.equal(item.objective,'harness');
 assert.equal(item.status,'user-required');
 assert.equal(item.preparation.classifications[item.groupIds[0]].triage.observed.kind,'provider');
 assert.equal(f.backlog.claim({base:'base',workflow:'failure-first-v1'}),null);
});
