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
import {failureCampaignStatus,runCampaignFailureStep} from '../../cli/skill-failure-campaign.mjs';

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

test('a verified declaration-only rejection reaches specification review without an experiment or curriculum work',async t=>{
 const f=await fixture(t),attempt=f.start('declaration');
 const verdict={passed:false,mode:'verify',flag_value:false,cases:10,failures:[{case:'<declaration>'}]};
 await f.verify(attempt,'verify.log',JSON.stringify(verdict));
 f.backlog.ingest();
 const result=await runCampaignFailureStep(f.config,{slots:[0,1,2,3],remainingSeconds:100,memory:{}},{
  analyze:()=>f.analyze(({groups:[group],validate})=>validate(classification(group,{bucket:'correctness',cause:'specification'}))),
  family:()=>assert.fail('A declaration-only rejection must not use an experiment slot')});
 assert.equal(result.status,'idle');
 const [item]=f.backlog.list({status:'user-required'});
 assert.equal(item.preparation.classifications[item.groupIds[0]].triage.route,'specification');
 assert.equal(f.state.db.prepare("SELECT name FROM sqlite_master WHERE name='skill_learning_requests'").get(),undefined);
});

test('a trusted unresolved oracle queues one Phase 1 question with independently specified expected outputs',async t=>{
 const f=await fixture(t),attempt=f.start('oracle-gap');
 const receipt={correct:null,evidence:{oracle:{status:'unresolved'}}};
 await f.verify(attempt,'verify.log',JSON.stringify(receipt));
 f.backlog.ingest();
 const {items:[item]}=await f.analyze(({groups:[group],validate})=>validate(classification(group,{bucket:'correctness'})));
 assert.equal(item.preparation.classifications[item.groupIds[0]].triage.route,'phase1');
 assert.equal(item.status,'learning');
 const question=claimLearningRequest(f.state,f.config,{draft:'oracle-study',target:'a5'});
 assert.equal(question.backlogId,item.id);
 assert.equal(question.workKind,'phase1');
 assert.equal(question.verificationReference,item.evidence.groups[0].occurrences[0].reference);
 const repeated=f.start('oracle-gap-repeated');
 await f.verify(repeated,'repeated.log',JSON.stringify(receipt));
 f.backlog.ingest();
 await f.analyze(({groups:[group],validate})=>validate(classification(group,{bucket:'correctness'})));
 assert.equal(f.state.db.prepare('SELECT COUNT(*) AS count FROM skill_learning_requests').get().count,1);
 assert.equal(claimLearningRequest(f.state,f.config,{draft:'second',target:'a5'}),null);
 assert.equal(f.backlog.claim({base:'base',workflow:'failure-first-v1'}),null);
});

test('numerical evidence still queues baseline isolation when a declaration also fails',async t=>{
 const f=await fixture(t),attempt=f.start('compound');
 const receipt={correct:false,kernelSha256:'a'.repeat(64),evidence:{caseResults:[
  {id:'declaration',kind:'declaration',correct:false},
  {id:'numerical',kind:'numerical',correct:false},
  {id:'control',kind:'numerical',correct:true}
 ]}};
 await f.verify(attempt,'verify.log',JSON.stringify(receipt));
 f.backlog.ingest();
 const {items:[item]}=await f.analyze(({groups:[group],validate})=>validate(classification(group,{bucket:'correctness'})));
 assert.equal(item.preparation.classifications[item.groupIds[0]].triage.route,'baseline');
 assert.equal(item.status,'queued');
});

test('a complete coverage report opens baseline screening only for measured kernel-path undercoverage',async t=>{
 for(const [kernel_path,contrib_covered,expected] of [[true,384,'baseline'],[false,384,'unresolved'],[true,24576,'unresolved']]){
  const f=await fixture(t),attempt=f.start('coverage');
  const report={mode:'coverage',passed:false,kernel:'kernel.py',cases:[{case:'corr-6',sql:{needed:24576,contrib_covered,kernel_path}}]};
  await f.verify(attempt,'coverage.log',JSON.stringify(report,null,2));
  f.backlog.ingest();
  const {items:[item]}=await f.analyze(({groups:[group],validate})=>validate(classification(group,{bucket:'correctness'})));
  assert.equal(item.preparation.classifications[item.groupIds[0]].triage.route,expected);
  assert.equal(item.status,expected==='baseline'?'queued':'deferred');
 }
});

test('an upgrade wakes a previously deferred coverage failure for exactly one new family',async t=>{
 const f=await fixture(t),attempt=f.start('legacy');
 const report={mode:'coverage',passed:false,kernel:'kernel.py',cases:[{sql:{kernel_path:true,needed:24576,contrib_covered:384}}]};
 await f.verify(attempt,'legacy.log',JSON.stringify(report,null,2));
 f.backlog.ingest();
 const [group]=f.backlog.failures(),observed={kind:'verifier-exit',reference:group.occurrences[0].reference};
 const old={...classification(group).classifications[0],triage:{route:'unresolved',observed}};
 f.backlog.upsert({key:'failure-first:'+group.id,workflow:'failure-first-v1',objective:'unknown',status:'deferred',
  groupIds:[group.id],priority:1,base:'base',preparation:{failureRoutingRevision:'legacy',classifications:{[group.id]:old}}});
 f.backlog.acknowledge(group.id,{classification:old,count:group.count});
 assert.equal(f.backlog.status().dirtyFailures,0);
 assert.equal((await failureCampaignStatus(f.config)).hasWork,true);
 let classifications=0,families=0;
 const step=()=>runCampaignFailureStep(f.config,{slots:[0,1,2,3],remainingSeconds:100,memory:{}},{
  analyze:()=>f.analyze(({groups:[current],validate})=>{
   classifications++;return validate(classification(current,{bucket:'correctness'}));
  }),
  input:async()=>({text:'base',sha256:'base'}),model:async()=>({id:'model'}),
  family:async()=>{families++;return {status:'needs-evidence',reason:'No retained source fixture'};}
 });
 await step();
 assert.equal((await failureCampaignStatus(f.config)).hasWork,false);
 assert.equal((await step()).status,'idle');
 assert.deepEqual([classifications,families],[1,1]);
});

test('an empty campaign stays idle across the admission upgrade',async t=>{
 const f=await fixture(t);
 assert.equal((await failureCampaignStatus(f.config)).hasWork,false);
});

test('a timed-out classification page retains its grant and stops requesting devices when only blocked pages remain',async t=>{
 const f=await fixture(t),attempt=f.start('page');
 await f.verify(attempt,'one.log','first failure');
 const values={slots:[0,1,2,3],remainingSeconds:1000,memory:{}};
 let milliseconds=0;
 const grants=[];
 const step=classify=>runCampaignFailureStep(f.config,values,{
  analyze:args=>analyzeBacklog({...args,remainingSeconds:500},{now:()=>milliseconds,classify})
 });
 const timeout=seconds=>({seconds:grant})=>{
  grants.push(grant);milliseconds+=seconds*1000;
  throw Object.assign(Error('Review timed out'),{reviewSession:{kind:'elapsed'}});
 };
 assert.equal((await step(timeout(40))).classificationPage,'retry');
 assert.equal((await failureCampaignStatus(f.config)).hasWork,true);
 assert.equal((await step(timeout(80))).classificationPage,'blocked');
 assert.deepEqual(grants,[120,80]);
 assert.equal((await failureCampaignStatus(f.config)).hasWork,false);
 assert.equal(f.backlog.failures().length,1);
 await f.verify(attempt,'two.log','second failure');
 // The new journal row must wake the controller before the ingestion cursor reaches it.
 assert.equal((await failureCampaignStatus(f.config)).hasWork,true);
 f.backlog.ingest();
 assert.equal((await failureCampaignStatus(f.config)).hasWork,true);
 await step(({groups:[fresh],validate})=>validate(classification(fresh)));
 assert.equal((await failureCampaignStatus(f.config)).hasWork,false);
});

test('unexpected classifier errors propagate to the campaign worker',async t=>{
 const f=await fixture(t);
 await f.verify(f.start('broken'),'one.log','first failure');
 await assert.rejects(runCampaignFailureStep(f.config,{slots:[0,1,2,3],remainingSeconds:1000,memory:{}},{
  analyze:args=>analyzeBacklog(args,{classify:()=>{throw Error('unexpected database error');}})
 }),/unexpected database error/);
});

test('elapsed page preparation returns to the worker before a classifier request is started',async t=>{
 const f=await fixture(t);
 await f.verify(f.start('clock'),'one.log','first failure');
 let now=0;
 const step=()=>runCampaignFailureStep(f.config,{slots:[0,1,2,3],remainingSeconds:100,memory:{}},{
  analyze:args=>analyzeBacklog({...args,remainingSeconds:1},{now:()=>{const before=now;now+=1000;return before;},
   classify:()=>assert.fail('Expired page must not start a provider request')})
 });
 const result=await step();
 assert.equal(result.status,'idle');
 assert.equal(result.classificationPage,'retry');
 assert.equal((await failureCampaignStatus(f.config)).hasWork,true);
});
