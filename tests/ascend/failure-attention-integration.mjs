import {DatabaseSync} from 'node:sqlite';
import {enqueueLearningRequest,recordLearningOutcome} from '../../controller/skill-review.mjs';
import {freezeAttentionBinding} from '../../cli/skill-priority.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {State} from '../../controller/state.mjs';
import {SkillBacklog} from '../../controller/skill-backlog.mjs';
import {initializeAttention} from '../../controller/failure-attention.mjs';
import {auditQueueRebuild,rebuildQueue} from '../../cli/failure-queue.mjs';
import {projectSummary} from '../../cli/project-summary.mjs';
import {diagnosticContext} from '../../cli/curriculum.mjs';
import {retrievalIdentity} from '../../controller/skill-objectives.mjs';
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'attention-integration-')),state=new State(join(root,'state.sqlite'));t.after(async()=>{state.close();await rm(root,{recursive:true,force:true});});
 const backlog=new SkillBacklog(state,{target:'a3'});state.add({id:'p',goal:'test',actor_seconds:60});const attempt=state.admit('p',{});
 const event=state.event(attempt.id,'attempt-error',{message:'gather index placement'});backlog.ingest();const group=backlog.failures()[0];
 return {state,backlog,attempt,group,config:{root,target:'a3',skills:{activeVersion:'v1'}},event};
}
test('suppressed diagnostics leave scientific verification and raw history available',async t=>{
 const {state,attempt,group}=await fixture(t);state.event(attempt.id,'verification',{passed:true,correctness:{passed:true}});
 initializeAttention(state.db);state.db.prepare('INSERT INTO failure_attention_actions VALUES(1,?,?,?)').run('incident','retired','{}');state.db.prepare('INSERT INTO failure_attention_effects VALUES(1,1,?,?,?)').run('occurrence',group.occurrences[0].reference,'retired');
 const summary=await projectSummary({state,projectId:'p'}),historical=await projectSummary({state,projectId:'p',includeRetired:true});
 assert.equal(summary.references.length,1);assert.equal(summary.verifications.length,1);assert.equal(historical.references.length,2);assert.equal(historical.references[0].attention,'retired');
 assert.equal(diagnosticContext(state,attempt.id,20).diagnostics.length,1);assert.equal(state.db.prepare('SELECT count(*) n FROM events').get().n,2);
});
test('operator rebuild archives planning only, focuses analysis, preserves human alerts and epochs new requests',async t=>{
 const {state,backlog,group,config}=await fixture(t);
 const old=backlog.upsert({key:'old',groupIds:[group.id],objective:'correctness',proposal:{action:'learn',question:'old question'},base:'v1',status:'learning'});
 const human=backlog.upsert({key:'human',groupIds:[group.id],objective:'harness',proposal:{action:'no-change'},base:'v1'});
 const oldRequest=enqueueLearningRequest(state,{target:'a3',question:'stale request',requiredEvidence:'old evidence'});
 const input={action:'rebuild-audit',id:'round2',operator:{reason:'Replace stale planning'},seeds:[{id:'gather',objective:'correctness',groupIds:[group.id],evidenceRefs:[group.occurrences[0].reference],question:'Does CPU index placement reproduce?',requiredEvidence:'CPU and NPU placement replay',learningTarget:'device'}]};
 const deps={readiness:async()=>({ready:false,supported:true,reason:'needs device evidence'})};
 const reader=new DatabaseSync(join(config.root,'state.sqlite'),{readOnly:true});
 const audit=await auditQueueRebuild(config,reader,input,deps);reader.close();assert.equal(backlog.get(old.id).status,'learning');
 const result=await rebuildQueue(config,state.db,{...input,action:'rebuild',auditSignature:audit.signature},deps);
 assert.equal(result.archivedItems,1);assert.equal(result.archivedRequests,1);
 recordLearningOutcome(state,{id:'late',generation:{draft:null,learningRequest:{id:oldRequest}}},{id:'late-attempt',outcome:'passed'});
 assert.equal(state.db.prepare('SELECT status FROM skill_learning_requests WHERE id=?').get(oldRequest).status,'archived');assert.equal(backlog.get(old.id).status,'archived');assert.equal(backlog.get(human.id).status,'user-required');assert.equal(backlog.failure(group.id).count,1);
 assert.equal(backlog.get(result.seedIds[0]).status,'learning');assert.ok(backlog.get(result.seedIds[0]).key.startsWith('epoch:round2:'));
 assert.equal(state.db.prepare('SELECT count(*) n FROM skill_queue_archives').get().n,1);
 await assert.rejects(rebuildQueue(config,state.db,{...input,auditSignature:audit.signature},deps),/unchanged dry-run signature/);
 state.event(null,'attempt-error',{message:'new failure after rebuild'});backlog.ingest();assert.equal(backlog.failures().length,2);
});
test('attention revision is part of retrieval identity for both immutable and legacy endpoints',()=>{
 assert.notDeepEqual(retrievalIdentity({url:'http://localhost',attentionRevision:1}),retrievalIdentity({url:'http://localhost',attentionRevision:2}));
 assert.deepEqual(retrievalIdentity({identity:'snapshot',attentionRevision:2}),{identity:'snapshot',readOnly:true,attentionRevision:2});
});

test('a pair pins its initial attention revision while historical in-flight evaluation keeps its identity',async t=>{
 const {state,backlog}=await fixture(t);initializeAttention(state.db);
 state.db.prepare('INSERT INTO failure_attention_actions VALUES(1,?,?,?)').run('one','retired','{}');
 const checkpoint={},cases={binding:{target:'cpu',knowledge:{identity:'memory',url:'http://localhost'}}};
 const first=freezeAttentionBinding(backlog,checkpoint,cases);
 state.db.prepare('INSERT INTO failure_attention_actions VALUES(2,?,?,?)').run('two','retired','{}');
 assert.deepEqual(freezeAttentionBinding(backlog,checkpoint,cases),first);assert.equal(first.binding.knowledge.attentionRevision,1);
 assert.deepEqual(freezeAttentionBinding(backlog,{evaluationDirectory:'old',evaluation:{binding:cases.binding}},cases),cases);
});
test('rebuild rejects changed evidence and active comparisons, preserves infrastructure review',async t=>{
 const {state,backlog,group,config}=await fixture(t),deps={readiness:async()=>({ready:false,supported:false})};
 const requestId=enqueueLearningRequest(state,{target:'a3',question:'blocked device',requiredEvidence:'native preflight'});
 state.db.prepare("UPDATE skill_learning_requests SET status='inconclusive' WHERE id=?").run(requestId);
 const input={id:'review',operator:{reason:'Reduce stale planning'},seeds:[],openInfrastructureBlocked:[{requestId,diagnosis:'DEVICE_LEASE_BLOCKED',evidenceRefs:['child:preactor']}]};
 const audit=await auditQueueRebuild(config,state.db,input,deps);assert.equal(audit.humanReview[0].retired,false);assert.equal(audit.snapshot.inconclusiveRequests[0].id,requestId);
 state.event(null,'attempt-error',{message:'new arrival'});
 await assert.rejects(rebuildQueue(config,state.db,{...input,auditSignature:audit.signature},deps),/unchanged dry-run signature/);
 const item=backlog.upsert({key:'active',groupIds:[group.id],objective:'correctness',proposal:{action:'update',changes:[]},base:'v1'});backlog.claim({base:'v1',itemId:item.id});
 await assert.rejects(auditQueueRebuild(config,state.db,input,deps),/Settle active/);assert.equal(backlog.get(item.id).status,'running');
});

test('an acknowledged failure cannot become an unready update with no executable learning request',async t=>{
 const {state,backlog,group,config}=await fixture(t);backlog.acknowledge(group.id);assert.equal(backlog.status().dirtyFailures,0);
 const input={id:'missing-cases',operator:{reason:'Rebuild actionable work'},seeds:[{id:'gather',objective:'correctness',groupIds:[group.id],evidenceRefs:[group.occurrences[0].reference],proposal:{action:'update',changes:[{action:'add',scope:'implementation',text:'Place indices on the tensor device'}]}}]};
 const unready={readiness:async()=>({ready:false,supported:true})};
 await assert.rejects(auditQueueRebuild(config,state.db,input,unready),/requires ready evaluation cases/);
 await assert.rejects(rebuildQueue(config,state.db,{...input,auditSignature:'unused'},unready),/requires ready evaluation cases/);
 assert.equal(backlog.list().length,0);assert.equal(backlog.meta('queue-epoch'),null);assert.equal(backlog.status().dirtyFailures,0);
 const ready={readiness:async()=>({ready:true,supported:true})},audit=await auditQueueRebuild(config,state.db,input,ready);
 assert.equal(audit.seeds[0].readiness.ready,true);
 const rebuilt=await rebuildQueue(config,state.db,{...input,auditSignature:audit.signature},ready),item=backlog.get(rebuilt.seedIds[0]);
 assert.equal(item.status,'queued');assert.equal(item.priority,100);assert.equal(backlog.claim({base:'v1',itemId:item.id}).item.id,item.id);
});
