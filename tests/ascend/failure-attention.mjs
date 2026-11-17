import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {DatabaseSync} from 'node:sqlite';
import {State} from '../../controller/state.mjs';
import {SkillBacklog} from '../../controller/skill-backlog.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {processIdentity,operationsStopped} from '../../controller/lifecycle.mjs';
import {auditResolution,applyResolution,reopenResolution,attentionRevision,attentionSnapshot,occurrenceAttention,projectAttention,attemptAttention} from '../../controller/failure-attention.mjs';
import {main} from '../../cli/main.mjs';
import {executionDigest,administrativeHealthHistory} from '../../controller/portable-proof.mjs';

async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'failure-attention-')),state=new State(join(root,'state.sqlite')),db=state.db;
 t.after(async()=>{state.close();await rm(root,{recursive:true,force:true});});
 const backlog=new SkillBacklog(state,{target:'a3'}),queue=new RecoveryQueue(state);
 state.add({id:'project',goal:'immutable kernel experiment',actor_seconds:3600,phase:'phase1'});
 db.prepare("INSERT INTO attempts(id,project,outcome,created) VALUES('attempt','project','failed','now')").run();
 db.prepare("UPDATE projects SET status='failed'").run();
 const event=(type,data)=>Number(db.prepare("INSERT INTO events(attempt,type,data,created) VALUES('attempt',?,?,'now')").run(type,JSON.stringify(data)).lastInsertRowid);
 const ref='events:'+event('attempt-error',{message:'A specific controller store failure'});
 event('failure-settlement',{closed:true});backlog.ingest();const group=backlog.failures()[0];
 const item=backlog.upsert({key:'lesson',groupIds:[group.id],objective:'correctness',skill:'preflight',base:'base',priority:50,proposal:{action:'learn',question:'Inspect cause'}});
 const proof=join(root,'validation.json'),bytes=JSON.stringify({passed:true,target:'a3',sourceRevision:'a'.repeat(40)});await writeFile(proof,bytes);
 const manifest={id:'store-fix',target:'a3',diagnosis:'Confirmed harness-only controller store failure',operator:{reason:'Explicit user-authorized cleanup'},attention:'retired',references:[ref],projectIds:['project'],attemptIds:['attempt'],fixRevision:'a'.repeat(40),watchGroupIds:[group.id],validation:[{path:proof,sha256:createHash('sha256').update(bytes).digest('hex')}]};
 return {root,state,db,backlog,queue,event,ref,group,item,manifest};
}

test('operator audit is read-only and retirement hides attention without changing scientific history',async t=>{
 const f=await fixture(t);f.queue.enqueue('attempt',0);const before={projects:f.db.prepare('SELECT * FROM projects').all(),attempts:f.db.prepare('SELECT * FROM attempts').all(),events:f.db.prepare('SELECT * FROM events').all()};
 assert.equal(projectAttention(f.db,'project'),'active');assert.equal(attentionRevision(f.db),0);
 await auditResolution(f.db,f.manifest);assert.equal(f.db.prepare("SELECT 1 FROM sqlite_master WHERE name='failure_attention_actions'").get(),undefined);
 assert.equal(f.queue.ready(1).length,1);
 const result=await applyResolution(f.db,f.manifest);assert.equal(result.applied,true);assert.equal(result.revision,1);
 assert.equal(occurrenceAttention(f.db,f.ref),'retired');assert.equal(projectAttention(f.db,'project'),'retired');assert.equal(attemptAttention(f.db,'attempt'),'retired');
 assert.equal(f.queue.enqueue('attempt',1),null);assert.deepEqual(f.queue.ready(1),[]);assert.throws(()=>f.queue.claim('project',1),/not eligible/);
 assert.deepEqual(f.backlog.failures(),[]);assert.deepEqual(f.backlog.list(),[]);assert.equal(f.backlog.claim({base:'base',itemId:f.item.id}),null);
 assert.equal(f.backlog.status().dirtyFailures,0);assert.equal(f.backlog.status().failures,0);assert.equal(f.backlog.status().historicalFailures,1);
 assert.equal(f.backlog.failure(f.group.id,{includeRetired:true}).occurrences[0].reference,f.ref);
 assert.deepEqual(f.db.prepare('SELECT * FROM projects').all(),before.projects);assert.deepEqual(f.db.prepare('SELECT * FROM attempts').all(),before.attempts);assert.deepEqual(f.db.prepare('SELECT * FROM events').all(),before.events);
 assert.equal((await applyResolution(f.db,f.manifest)).applied,false);assert.equal(attentionRevision(f.db),1);
 const ro=new DatabaseSync(join(f.root,'state.sqlite'),{readOnly:true});try{assert.equal(attentionSnapshot(ro).projects[0].attention,'retired');}finally{ro.close();}
});

test('mixed genuine failures stay visible and unknown or live execution cannot retire a project',async t=>{
 const f=await fixture(t);f.event('verification',{passed:false,correctness:{passed:false},fingerprint:'numeric'});
 await assert.rejects(applyResolution(f.db,f.manifest),/Uncovered verification/);
 f.backlog.ingest();await assert.rejects(applyResolution(f.db,f.manifest),/Mixed or uncovered/);
 await applyResolution(f.db,{...f.manifest,projectIds:[],attemptIds:[]});
 assert.equal(projectAttention(f.db,'project'),'active');assert.equal(f.backlog.failures().length,1);assert.equal(f.backlog.failures()[0].type,'verification');
 const g=await fixture(t);g.event('failure-settlement',{closed:false});await assert.rejects(applyResolution(g.db,g.manifest),/positive failure settlement/);
 const h=await fixture(t);h.db.prepare('UPDATE attempts SET owner=?').run(JSON.stringify(await processIdentity()));await assert.rejects(applyResolution(h.db,h.manifest),/remains live/);
 assert.equal(attentionRevision(g.db),0);assert.equal(attentionRevision(h.db),0);
});

test('reingestion preserves retired history and fresh watched recurrence reopens only new attention',async t=>{
 const f=await fixture(t);await applyResolution(f.db,f.manifest);
 f.db.prepare('DELETE FROM skill_backlog_occurrences').run();f.backlog.meta('cursor:events',null);
 const reopened=new SkillBacklog(f.state,{target:'a3'});reopened.ingest();assert.equal(reopened.failures().length,0);assert.equal(attentionRevision(f.db),1);
 const ref='events:'+f.event('attempt-error',{message:'A specific controller store failure'});reopened.ingest();
 assert.equal(attentionRevision(f.db),2);assert.equal(occurrenceAttention(f.db,f.ref),'retired');assert.equal(occurrenceAttention(f.db,ref),'active');assert.equal(projectAttention(f.db,'project'),'active');
 assert.equal(reopened.failure(f.group.id).count,1);assert.equal(reopened.failure(f.group.id).historicalCount,2);
 assert.equal(attentionSnapshot(f.db).incidents[0].action,'regression');reopened.ingest();assert.equal(attentionRevision(f.db),2);
 reopenResolution(f.db,{id:f.manifest.id,reason:'Operator reviewing original diagnosis',references:[f.ref]});assert.equal(reopened.failure(f.group.id).count,2);
 assert.equal(f.db.prepare('SELECT count(*) n FROM failure_attention_actions').get().n,3);
});

test('retirement rejects missing fix proof, changed evidence, broad scopes and target mismatch',async t=>{
 const f=await fixture(t);
 await assert.rejects(applyResolution(f.db,{...f.manifest,fixRevision:null}),/fix revision/);
 await assert.rejects(applyResolution(f.db,{...f.manifest,references:[]}),/exact unique references/);
 await assert.rejects(applyResolution(f.db,{...f.manifest,projectIds:['other']}),/scopes/);
 await assert.rejects(applyResolution(f.db,f.manifest,{target:'a5'}),/target differs/);
 const negative=JSON.stringify({passed:false,target:'a3',sourceRevision:'a'.repeat(40)});await writeFile(f.manifest.validation[0].path,negative);
 await assert.rejects(applyResolution(f.db,{...f.manifest,validation:[{path:f.manifest.validation[0].path,sha256:createHash('sha256').update(negative).digest('hex')}]}),/passed validation/);
 await writeFile(f.manifest.validation[0].path,'changed');await assert.rejects(applyResolution(f.db,f.manifest),/checksum/);
 assert.equal(attentionRevision(f.db),0);
});

test('quarantined unfixed incidents remain explicitly open and are never reported as resolved',async t=>{
 const f=await fixture(t);await applyResolution(f.db,{...f.manifest,attention:'quarantined',fixRevision:undefined});
 const report=f.backlog.status();assert.equal(report.attention.openIncidents[0].id,f.manifest.id);assert.equal(report.attention.quarantinedOccurrences,1);assert.equal(report.attention.retiredOccurrences,0);
 assert.equal(attentionSnapshot(f.db).incidents[0].state,'open');assert.equal(f.state.project('project').status,'failed');
});

test('explicit nonrecurrence retirement preserves its basis and requires exact scoped evidence',async t=>{
 const f=await fixture(t),proof={retirementBasis:'not-expected-to-recur',expectedRecurrence:false,kind:'obsolete-benchmark',target:'a3',reason:'The benchmark version is abandoned; its frozen source will not be readmitted',references:[f.ref]},bytes=JSON.stringify(proof);
 await writeFile(f.manifest.validation[0].path,bytes);const manifest={...f.manifest,fixRevision:undefined,retirementBasis:proof.retirementBasis,expectedRecurrence:false,validation:[{path:f.manifest.validation[0].path,sha256:createHash('sha256').update(bytes).digest('hex')}]};
 await assert.rejects(applyResolution(f.db,{...manifest,expectedRecurrence:undefined}),/Nonrecurrence/);
 await applyResolution(f.db,manifest);assert.equal(attentionSnapshot(f.db).incidents[0].manifest.retirementBasis,'not-expected-to-recur');assert.equal(projectAttention(f.db,'project'),'retired');
});

test('passed settled investigations do not prevent retiring a failed original, but accepted retries and unmatched operations do',async t=>{
 const f=await fixture(t);
 f.db.prepare("INSERT INTO attempts(id,project,outcome,created) VALUES('investigation','project','passed','now')").run();
 f.db.prepare("INSERT INTO recovery_attempts(attempt,project,kind,wave,completed) VALUES('investigation','project','investigation',1,1)").run();
 f.state.event('investigation','postprocess-completed',{});
 await applyResolution(f.db,f.manifest);assert.equal(projectAttention(f.db,'project'),'retired');assert.equal(f.state.attempt('investigation').outcome,'passed');
 const g=await fixture(t);g.event('native-bash-started',{operation:'pending-shell'});await assert.rejects(applyResolution(g.db,g.manifest),/Unmatched execution/);
 const h=await fixture(t);h.db.prepare("INSERT INTO attempts(id,project,outcome,created) VALUES('accepted','project','passed','now')").run();
 h.db.prepare("INSERT INTO recovery_attempts(attempt,project,kind,wave,completed) VALUES('accepted','project','retry',1,1)").run();h.state.event('accepted','postprocess-completed',{});
 await assert.rejects(applyResolution(h.db,h.manifest),/positive failure settlement/);
});

test('native reconciliation may recover terminal handles after closure without permitting a later launch',async t=>{
 const f=await fixture(t),handle='remote:fixture:job:recovered';
 f.event('execution-starting',{operation:'recovered-operation',actor_elapsed:0});
 assert.equal(await operationsStopped(f.state,f.state.attempt('attempt'),{
  reconcileOperation:async()=>({closed:true,executionSeconds:1,handles:[handle]}),
  observe:async()=>({state:'stopped',nativeState:'completed',executionSeconds:1})}),true);
 f.event('failure-settlement',{closed:true});
 const recovered=f.state.events('attempt'),closure=recovered.findIndex(e=>e.type==='execution-reconciled'),identity=recovered.findIndex(e=>e.type==='execution-handle');
 assert.ok(identity>closure);await auditResolution(f.db,f.manifest);
 f.event('execution-handle',{operation:'recovered-operation',handle:'remote:fixture:job:unproved'});
 await assert.rejects(auditResolution(f.db,f.manifest),/Unmatched execution/);
 f.event('execution-state',{handle:'remote:fixture:job:unproved',state:'stopped',nativeState:'completed'});
 await auditResolution(f.db,f.manifest);
 f.event('execution-starting',{operation:'recovered-operation',actor_elapsed:1});
 await assert.rejects(auditResolution(f.db,f.manifest),/Unmatched execution/);
});

test('legacy transfer handles require exact submitted operation mapping and native terminal proof',async t=>{
 const f=await fixture(t),handle='remote:fixture:transfer:legacy',operation='transfer-operation';
 f.event('execution-submitting',{submission:operation,operation});
 f.event('execution-submitted',{submission:operation,operation,handle});
 f.event('execution-handle',{handle,purpose:'transfer'});
 f.event('execution-state',{handle,state:'stopped',nativeState:'completed'});
 f.event('execution-reconciled',{operation,closed:true});
 f.event('execution-handle',{operation,handle});
 await auditResolution(f.db,f.manifest);
 f.event('execution-handle',{handle:'remote:fixture:transfer:unknown',purpose:'transfer'});
 await assert.rejects(auditResolution(f.db,f.manifest),/Unmatched execution/);
 const g=await fixture(t);
 g.event('execution-submitted',{submission:operation,operation,handle});
 g.event('execution-handle',{handle,purpose:'transfer'});
 g.event('execution-reconciled',{operation,closed:true});
 await assert.rejects(auditResolution(g.db,g.manifest),/Unmatched execution/);
 g.event('execution-state',{handle,state:'stopped',nativeState:'completed'});
 await auditResolution(g.db,g.manifest);
 g.event('execution-submitting',{submission:operation,operation});
 await assert.rejects(auditResolution(g.db,g.manifest),/Unmatched execution/);
});

test('retirement honors only validated portable administrative boundaries and fences destination work',async t=>{
 for(const kind of ['valid','fake-digest','payload','destination-start']){
  const f=await fixture(t),handle='local:exec-health';
  f.event('execution-starting',{operation:'source-health'});
  f.event('execution-submitting',{operation:'source-health',submission:'health',kind:'run',accountingExcluded:true});
  f.event('execution-submitted',{operation:'source-health',submission:'health',handle});
  f.event('execution-handle',{operation:'source-health',handle,purpose:'resource-health'});
  const prefix=f.state.events('attempt'),proof={eventCount:prefix.length,executionSha256:executionDigest(prefix),healthHandles:administrativeHealthHistory(prefix),sourceShutdown:{sha256:'a'.repeat(64)}};
  if(kind==='fake-digest')proof.executionSha256='0'.repeat(64);
  if(kind==='payload'){
   f.event('native-bash-started',{operation:'source-payload'});
   const altered=f.state.events('attempt');proof.eventCount=altered.length;proof.executionSha256=executionDigest(altered);
  }
  f.event('portable-bound',proof);
  if(kind==='destination-start')f.event('execution-starting',{operation:'destination-payload'});
  if(kind==='valid')await auditResolution(f.db,f.manifest);
  else await assert.rejects(auditResolution(f.db,f.manifest),/Unmatched execution/);
 }
});

test('occurrence-only attention cannot hide live or unresolved attempts and does not require whole-attempt coverage',async t=>{
 for(const defect of ['running','live-owner','open-settlement','unmatched-shell']){
  const f=await fixture(t),manifest={...f.manifest,projectIds:[],attemptIds:[]};
  if(defect==='running')f.db.prepare('UPDATE attempts SET outcome=NULL').run();
  if(defect==='live-owner')f.db.prepare('UPDATE attempts SET owner=?').run(JSON.stringify(await processIdentity()));
  if(defect==='open-settlement')f.event('failure-settlement',{closed:false});
  if(defect==='unmatched-shell')f.event('native-bash-started',{operation:'still-running'});
  await assert.rejects(applyResolution(f.db,manifest),/positive failure settlement|remains live|Unmatched execution/);
  assert.equal(attentionRevision(f.db),0);assert.equal(occurrenceAttention(f.db,f.ref),'active');
 }
 const f=await fixture(t);f.event('verification',{passed:false,correctness:{passed:false}});f.backlog.ingest();
 await applyResolution(f.db,{...f.manifest,projectIds:[],attemptIds:[]});
 assert.equal(occurrenceAttention(f.db,f.ref),'retired');assert.equal(f.backlog.failures().length,1);assert.equal(projectAttention(f.db,'project'),'active');
});

test('public failures maintenance defaults to inspection/audit and requires stopped writers to apply',async t=>{
 const f=await fixture(t);await writeFile(join(f.root,'instance.json'),JSON.stringify({format:1,target:'a3'}));
 const path=join(f.root,'operator.json');await writeFile(path,JSON.stringify({manifest:f.manifest}));
 assert.equal((await main(['failures','--instance',f.root])).revision,0);
 assert.equal((await main(['failures','--instance',f.root,'--config',path])).occurrences.length,1);assert.equal(attentionRevision(f.db),0);
 f.db.exec('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT)');f.db.prepare('INSERT INTO control VALUES(?,?)').run('owner',JSON.stringify(await processIdentity()));
 await writeFile(path,JSON.stringify({action:'apply',manifest:f.manifest}));await assert.rejects(main(['failures','--instance',f.root,'--config',path]),/Checkpoint live/);
 f.db.prepare("UPDATE control SET value='null' WHERE key='owner'").run();assert.equal((await main(['failures','--instance',f.root,'--config',path])).applied,true);
 await writeFile(path,JSON.stringify({action:'reopen',id:f.manifest.id,reason:'Review correction',references:[f.ref],projectIds:['project'],attemptIds:['attempt']}));
 await main(['failures','--instance',f.root,'--config',path]);assert.equal(projectAttention(f.db,'project'),'active');assert.equal(f.queue.enqueue('attempt',1).project,'project');
 assert.equal(JSON.parse(await readFile(join(f.root,'instance.json'))).target,'a3');
});
