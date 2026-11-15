import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {SkillBacklog} from '../../controller/skill-backlog.mjs';
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'skill-backlog-')),state=new State(join(root,'state.sqlite'));
 t.after(async()=>{state.close();await rm(root,{recursive:true,force:true});});
 const b=new SkillBacklog(state,{target:'a3',now:()=>100});
 const event=(type,data={},attempt=null)=>state.db.prepare('INSERT INTO events(attempt,type,data,created) VALUES(?,?,?,?)').run(attempt,type,JSON.stringify(data),'now');
 const attempt=(id,phase='phase1',outcome=null)=>{state.add({id,goal:'fixture',actor_seconds:60,phase});state.db.prepare('INSERT INTO attempts(id,project,outcome,created) VALUES(?,?,?,?)').run(id,id,outcome,'now');event('job-admitted',{kind:'original',mode:phase},id);};
 const seed=()=>{event('attempt-error',{message:'Tail store exceeds logical extent'});b.ingest();return b.failures()[0].id;};
 const item=(key,group,patch={})=>b.upsert({key,groupIds:[group],objective:'correctness',skill:'implementation',proposal:{text:'Mask tail stores'},evidence:{observation:'Tail lane 31 overwrites adjacent row'},base:'baseline',priority:0,...patch});
 return{state,b,event,attempt,seed,item};
}

test('ingests every failure source without admitting negative controls or explicit operator pauses',async t=>{
 const{state,b,event,attempt}=await fixture(t);
 attempt('p1','phase1','failed');attempt('p2','phase2','passed');attempt('paused','phase1','failed');
 event('verification',{correct:false,reason:'Invalid ragged store',expected_exit:0,exitCode:0},'p2');
 event('attempt-error',{message:'Interrupted',reason:'operator-pause'},'paused');
 event('generation-rejected',{expected_exit:1,exitCode:1});
 state.db.exec('CREATE TABLE generation_events(sequence INTEGER PRIMARY KEY,draft TEXT,type TEXT,data TEXT,created TEXT);CREATE TABLE generation_drafts(id TEXT PRIMARY KEY,data TEXT);');
 state.db.prepare('INSERT INTO generation_events VALUES(1,?,?,?,?)').run('draft','generation-error',JSON.stringify({message:'Reference is invalid'}),'now');
 state.db.prepare('INSERT INTO generation_drafts VALUES(?,?)').run('draft',JSON.stringify({phase:'phase1',status:'exhausted',topic:'layout'}));
 b.ingest();const groups=b.failures({dirty:false});
 assert.deepEqual(new Set(groups.map(g=>g.type)),new Set(['verification','failed-outcome','generation-error','generation-exhausted']));
 assert.equal(groups.find(g=>g.type==='verification').occurrences[0].phase,'phase2');
 assert.equal(groups.find(g=>g.type==='failed-outcome').occurrences[0].phase,'phase1');
 assert.ok(groups.every(g=>g.occurrences.every(o=>o.attemptId!=='paused')));
 const count=b.status().failures;b.ingest();assert.equal(b.status().failures,count);
});

test('fresh diagnostics bypass historical telemetry and watermarks resume after reopening',async t=>{
 const{state,b,event}=await fixture(t);
 state.db.exec('BEGIN');for(let i=0;i<1000;i++)state.db.prepare("INSERT INTO events(type,data,created) VALUES('pi-event','not parsed','now')").run();state.db.exec('COMMIT');
 assert.equal(b.status().ingestion.initialPending,true);assert.equal(b.status().backfillComplete,false);
 b.ingest({limit:1});assert.equal(b.status().backfillComplete,false);
 event('attempt-error',{message:'Urgent new compiler error'});
 const reopened=new SkillBacklog(state,{target:'a3'});const report=reopened.ingest({limit:1});
 assert.equal(report.observed,1);assert.equal(reopened.failures()[0].message,'Urgent new compiler error');
 for(let i=0;i<12;i++)reopened.ingest({limit:1});assert.equal(reopened.status().backfillComplete,true);
 assert.equal(reopened.status().failures,1);assert.equal(reopened.status().ingestion.freshPending,false);
});

test('terminal command failures remain evidence after successful recovery, excluding polls and controls',async t=>{
 const{b,event,attempt}=await fixture(t);attempt('recovered','phase1','passed');
 event('execution-state',{state:'running',exitCode:1},'recovered');
 event('execution-state',{state:'stopped',exitCode:1,accountingExcluded:true},'recovered');
 event('execution-state',{state:'stopped',exitCode:1,purpose:'resource-health'},'recovered');
 event('execution-state',{state:'stopped',exitCode:1,expected_exit:1},'recovered');
 event('execution-state',{state:'stopped',exitCode:1,handle:'native:failed-compile',message:'Compiler rejected stride'},'recovered');
 event('execution-state',{state:'stopped',exitCode:0},'recovered');b.ingest();
 const groups=b.failures();assert.equal(groups.length,1);assert.equal(groups[0].type,'execution-state');assert.equal(groups[0].occurrences[0].attemptId,'recovered');assert.equal(groups[0].occurrences[0].data.handle,'native:failed-compile');
});

test('failure provenance uses the skill actually selected before that occurrence',async t=>{
 const{b,event,attempt}=await fixture(t);attempt('versioned','phase1','passed');
 event('skills-selected',{version:'first-version'},'versioned');event('attempt-error',{message:'First diagnostic'},'versioned');
 event('skills-selected',{version:'second-version'},'versioned');event('verification',{correct:false,reason:'Second diagnostic'},'versioned');
 event('skills-selected',{version:'later-version'},'versioned');b.ingest();
 const versions=new Map(b.failures().map(group=>[group.message,group.occurrences[0].skillVersion]));
 assert.equal(versions.get('First diagnostic'),'first-version');assert.equal(versions.get('Second diagnostic'),'second-version');
});

test('an interrupted ingestion transaction cannot advance a cursor past uncaptured evidence',async t=>{
 const{state,b,event}=await fixture(t);event('attempt-error',{message:'Must survive rollback'});
 state.db.exec("CREATE TRIGGER fail_capture BEFORE INSERT ON skill_backlog_occurrences BEGIN SELECT RAISE(ABORT,'capture interrupted');END;");
 assert.throws(()=>b.ingest(),/capture interrupted/);assert.equal(b.status().failures,0);assert.equal(b.status().ingestion.initialPending,true);
 state.db.exec('DROP TRIGGER fail_capture');b.ingest();assert.equal(b.status().failures,1);assert.equal(b.failures()[0].message,'Must survive rollback');
});

test('recurrence retains every reference with bounded representatives and acknowledgement avoids lost arrivals',async t=>{
 const{b,event}=await fixture(t);
 for(let i=0;i<25;i++)event('attempt-error',{message:'Same tail mismatch',elapsed:i});
 b.ingest();const group=b.failures({occurrenceLimit:2})[0];assert.equal(group.count,25);assert.equal(group.occurrences.length,2);
 assert.equal(new Set(b.failure(group.id,{limit:100}).occurrences.map(o=>o.reference)).size,25);
 event('attempt-error',{message:'Same tail mismatch',elapsed:30});b.ingest();b.acknowledge(group.id,{classification:{bucket:'correctness'},count:25});
 assert.equal(b.failure(group.id).dirty,true);assert.equal(b.failure(group.id).classification.bucket,'correctness');
 b.acknowledge(group.id,{count:26});assert.equal(b.failures().length,0);
});

test('mutable drafts and failed outcomes are revisited after their initial nonterminal sweep',async t=>{
 const{state,b,attempt}=await fixture(t);attempt('pending');
 state.db.exec('CREATE TABLE generation_drafts(id TEXT PRIMARY KEY,data TEXT)');state.db.prepare('INSERT INTO generation_drafts VALUES(?,?)').run('d',JSON.stringify({status:'draft'}));
 b.ingest();state.db.prepare("UPDATE attempts SET outcome='failed' WHERE id='pending'").run();state.db.prepare('UPDATE generation_drafts SET data=?').run(JSON.stringify({status:'invalid',error:'Broken spec'}));
 b.ingest();b.ingest();assert.deepEqual(new Set(b.failures().map(g=>g.type)),new Set(['failed-outcome','generation-invalid']));
});

test('new urgent hypotheses overtake queued work but never preempt a retained comparison',async t=>{
 const{b,seed,item}=await fixture(t),group=seed(),low=item('low',group,{priority:1});
 const first=b.claim({base:'baseline',owner:{pid:1}});assert.equal(first.item.id,low.id);
 const urgent=item('urgent',group,{priority:100,proposal:{text:'Fix the reduction identity before optimizing'}});b.checkpoint(first.run.id,{elapsed:120,checkpoint:{runningAt:1000,stage:'qa'}});
 const resumed=b.claim({base:'baseline',seconds:30,owner:{pid:2}});
 assert.equal(resumed.run.id,first.run.id);assert.equal(resumed.remaining,1680);assert.equal(resumed.run.seconds,1800);assert.deepEqual(resumed.run.owner,{pid:2});assert.equal(resumed.run.checkpoint.stage,'qa');
 b.finish(first.run.id,{status:'rejected',result:{correct:false}});assert.equal(b.claim({base:'baseline'}).item.id,urgent.id);
});

test('rejected identical instructions do not gain a fresh allowance from new occurrence IDs or priority',async t=>{
 const{b,seed,item}=await fixture(t),group=seed();const h=item('h',group,{evidence:[{reference:'events:1',data:{message:'tail mismatch'},count:1}]});
 const run=b.claim({base:'baseline'}).run;b.checkpoint(run.id,{elapsed:300});b.finish(run.id,{status:'rejected'});
 b.revise(h.id,{priority:100,evidence:[{reference:'events:99',data:{message:'tail mismatch'},count:2},{reference:'events:100',data:{message:'tail mismatch'},count:3}]});
 assert.equal(b.get(h.id).status,'rejected');assert.equal(b.claim({base:'baseline'}),null);assert.equal(b.run(run.id).elapsed,300);
 b.revise(h.id,{proposal:{text:'Mask row and column tails independently'}});const revised=b.claim({base:'baseline'});
 assert.notEqual(revised.run.id,run.id);assert.equal(revised.run.revision,2);assert.equal(revised.remaining,1800);
});

test('real new diagnostic evidence can reopen an item while expired work cannot promote',async t=>{
 const{b,seed,item}=await fixture(t),group=seed(),h=item('h',group),first=b.claim({base:'baseline'});
 b.checkpoint(first.run.id,{elapsed:1800});assert.throws(()=>b.finish(first.run.id,{status:'accepted'}),/Expired/);b.finish(first.run.id,{status:'inconclusive'});
 b.revise(h.id,{evidence:{observation:'Column mask also required at column 17'}});const next=b.claim({base:'baseline'});assert.equal(next.run.revision,2);
 assert.throws(()=>b.checkpoint(next.run.id,{elapsed:-1}),/monotonic/);b.checkpoint(next.run.id,{elapsed:10});assert.throws(()=>b.checkpoint(next.run.id,{elapsed:9}),/monotonic/);
});

test('evidence ordering and snapshot locations are telemetry, while changed test contracts are revisions',async t=>{
 const{b,seed,item}=await fixture(t),h=item('h',seed(),{evidence:[{path:'/first/copy',sha256:'one',text:'row boundary'},{path:'/first/other',sha256:'two',text:'column boundary'}]});
 const first=b.claim({base:'baseline'});b.finish(first.run.id,{status:'rejected'});
 b.revise(h.id,{preparation:{proposalDirectory:'/new/snapshot',proposalElapsedSeconds:50},evidence:[{path:'/second/other',sha256:'two',text:'column boundary'},{path:'/second/copy',sha256:'one',text:'row boundary'}]});
 assert.equal(b.get(h.id).status,'rejected');assert.equal(b.get(h.id).revision,1);
 b.revise(h.id,{tests:{cases:['partial-row','partial-column']}});assert.equal(b.get(h.id).revision,2);assert.ok(b.claim({base:'baseline'}));
});

test('reworded proposal rationales and retrieval scores cannot retry an identical rejected instruction',async t=>{
 const{b,seed,item}=await fixture(t),proposal={action:'update',reason:'The last tile overwrites its neighbour',memoryRefs:['m:1'],
  changes:[{action:'upsert',id:'tail',scope:'implementation',text:'Mask stores on the logical row extent',critical:true,negative:false}],learningRequests:[{question:'Is another layout affected?'}]};
 const evidence={groups:[{id:'same-diagnostic',classification:{reason:'Original review narrative'}}],memoryEvidence:[{collection:'m',entry_id:'1',text:'Observed row overflow',sha256:'same-content',score:.9}]};
 const h=item('update',seed(),{proposal,evidence}),first=b.claim({base:'baseline'});b.finish(first.run.id,{status:'rejected'});
 b.revise(h.id,{proposal:{...proposal,reason:'A boundary mask is needed',learningRequests:[]},evidence:{groups:[{id:'same-diagnostic',classification:{reason:'Reworded review narrative'}}],memoryEvidence:[{...evidence.memoryEvidence[0],score:.3,rank:9}]}});
 const retained=b.get(h.id);assert.equal(retained.status,'rejected');assert.equal(retained.revision,1);assert.equal(retained.proposal.reason,'A boundary mask is needed');assert.equal(retained.evidence.memoryEvidence[0].score,.3);
 assert.equal(b.claim({base:'baseline'}),null);assert.equal(b.run(first.run.id).snapshot.proposal.reason,proposal.reason);
 b.revise(h.id,{proposal:{...proposal,changes:[{...proposal.changes[0],text:'Mask both row and column logical extents'}]}});assert.equal(b.get(h.id).revision,2);assert.ok(b.claim({base:'baseline'}));
});

test('a generated instruction ID and queue key cannot buy the same failed hypothesis another experiment',async t=>{
 const{b,seed,item}=await fixture(t),group=seed(),proposal=id=>({action:'update',memoryRefs:['m:1'],changes:[{action:'upsert',id,scope:'implementation',text:'Mask logical tail stores',critical:true,negative:false}]});
 const evidence={observation:'An unchanged diagnosed failure',memoryEvidence:[{ref:'m:1',text:'Observed boundary failure',sha256:'same'}]};
 const h=item('first-key',group,{proposal:proposal('first-name'),evidence}),first=b.claim({base:'baseline'});b.finish(first.run.id,{status:'rejected'});
 const renamed=proposal('fresh-name');renamed.memoryRefs=['m:2'];renamed.changes[0].text=' Mask  logical\n tail stores ';
 const duplicate=item('second-key',group,{proposal:renamed,evidence:{...evidence,memoryEvidence:[{ref:'m:2',text:'Observed boundary failure',sha256:'same'}]},priority:100});
 assert.equal(duplicate.id,h.id);assert.equal(b.get('second-key').id,h.id);assert.equal(duplicate.status,'rejected');assert.equal(b.list().length,1);assert.equal(b.claim({base:'baseline'}),null);
 const meaningful=item('new-evidence',group,{proposal:proposal('third-name'),evidence:{observation:'Transposed strides introduce a second independent overwrite'}});
 assert.notEqual(meaningful.id,h.id);assert.equal(b.claim({base:'baseline'}).item.id,meaningful.id);
});

test('returning from terminal revision A through B to A preserves the first experiment result and allowance',async t=>{
 const{b,seed,item}=await fixture(t),group=seed(),proposal=text=>({action:'update',changes:[{action:'upsert',id:'instruction',scope:'implementation',text}]});
 const h=item('hypothesis',group,{proposal:proposal('Instruction A')}),a=b.claim({base:'baseline'});b.checkpoint(a.run.id,{elapsed:73});b.finish(a.run.id,{status:'rejected'});
 b.revise(h.id,{proposal:proposal('Instruction B')});const z=b.claim({base:'baseline'});b.checkpoint(z.run.id,{elapsed:22});b.finish(z.run.id,{status:'inconclusive'});
 const returned=b.revise(h.id,{proposal:proposal('Instruction A')});assert.equal(returned.status,'rejected');assert.equal(returned.previousRun,a.run.id);assert.equal(b.claim({base:'baseline'}),null);assert.equal(b.run(a.run.id).elapsed,73);assert.equal(b.run(z.run.id).elapsed,22);
 b.revise(h.id,{evidence:{observation:'A new independently verified failing boundary'}});const fresh=b.claim({base:'baseline'});assert.equal(fresh.remaining,1800);assert.notEqual(fresh.run.id,a.run.id);
});

test('removing different existing guidance entries remains a distinct change',async t=>{
 const{b,seed,item}=await fixture(t),group=seed(),proposal=id=>({action:'update',changes:[{action:'remove',id,scope:'implementation'}]});
 item('remove-a',group,{proposal:proposal('a')});const a=b.claim({base:'baseline'});b.finish(a.run.id,{status:'rejected'});
 item('remove-b',group,{proposal:proposal('b')});assert.ok(b.claim({base:'baseline'}));
});

test('learning identity follows the actual question and evidence criterion rather than review narrative',async t=>{
 const{b,seed,item}=await fixture(t),proposal={action:'learn',question:'Which layouts need a mask?',hypothesis:'Partial tiles require a mask',requiredEvidence:'Check a partial tile',learningTarget:'device',reason:'Missing observation'};
 const h=item('learn',seed(),{proposal,status:'learning'});b.revise(h.id,{proposal:{...proposal,reason:'Needs a target experiment',learningRequests:[]},status:'learning'});assert.equal(b.get(h.id).revision,1);
 b.revise(h.id,{proposal:{...proposal,requiredEvidence:'Check a partial tile and transposed strides'},status:'learning'});assert.equal(b.get(h.id).revision,2);
});

test('active-version changes require requalification and retain all old experiment evidence',async t=>{
 const{b,seed,item}=await fixture(t),h=item('h',seed());const first=b.claim({base:'baseline'});b.checkpoint(first.run.id,{elapsed:42});
 assert.throws(()=>b.rebase(h.id,'promoted'),/Settle/);assert.throws(()=>b.claim({base:'promoted'}),/Settle/);
 b.finish(first.run.id,{status:'rejected'});b.rebase(h.id,'promoted');assert.equal(b.claim({base:'promoted'}),null);assert.equal(b.get(h.id).status,'needs-review');
 b.revise(h.id,{status:'queued'});const next=b.claim({base:'promoted'});assert.equal(next.remaining,1800);assert.equal(b.run(first.run.id).elapsed,42);assert.equal(b.run(first.run.id).snapshot.base,'baseline');
});

test('priority changes are durable and same-priority work uses oldest insertion first',async t=>{
 const{state,b,seed,item}=await fixture(t),g=seed(),a=item('first',g),c=item('second',g,{proposal:{text:'Fix the independent stride assumption'}});
 const reopened=new SkillBacklog(state,{target:'a3'});reopened.rank([{id:c.id,priority:10,reason:'Blocks the current Phase 2'}]);assert.equal(reopened.list()[0].id,c.id);
 reopened.rank([{id:a.id,priority:10,reason:'Equally severe'}]);assert.equal(reopened.claim({base:'baseline'}).item.id,a.id);
});

test('harness items remain user handled and each campaign has independent claims',async t=>{
 const a=await fixture(t),z=await fixture(t),h=a.item('harness',a.seed(),{objective:'harness'});assert.equal(h.status,'user-required');assert.equal(a.b.claim({base:'baseline'}),null);
 a.item('actor',h.groupIds[0]);z.item('actor',z.seed());assert.ok(a.b.claim({base:'baseline'}));assert.ok(z.b.claim({base:'baseline'}));
});

test('a revised hypothesis cannot alter an already frozen experiment snapshot',async t=>{
 const{b,seed,item}=await fixture(t),h=item('h',seed()),old=b.claim({base:'baseline'});b.revise(h.id,{proposal:{text:'New supported instruction'}});
 assert.equal(b.run(old.run.id).snapshot.proposal.text,'Mask tail stores');b.finish(old.run.id,{status:'rejected'});assert.equal(b.get(h.id).status,'queued');
 const next=b.claim({base:'baseline'});assert.equal(next.run.revision,2);assert.equal(next.run.snapshot.proposal.text,'New supported instruction');
});

test('audited case-selection repair preserves unchanged hypothesis history and original allocation',async t=>{
 const {state,b,seed,item}=await fixture(t),h=item('handoff',seed()),old=b.claim({base:'baseline'}).run;
 b.checkpoint(old.id,{elapsed:125,checkpoint:{proposalCharged:true,resource:{id:'reserved'}}});b.finish(old.id,{status:'inconclusive',result:{status:'unsupported',missingCapability:true,reason:'Case details were not delivered to the selector'}});
 const frozen=state.db.prepare('SELECT * FROM skill_backlog_runs WHERE id=?').get(old.id),repair={fromRevision:'a'.repeat(40),toRevision:'b'.repeat(40),evidence:{path:'/tmp/selection-repair-proof.json',sha256:'c'.repeat(64)}};
 const continued=b.continue(old.id,{base:'baseline',procedure:'Operator-authorized handoff repair after validated code change',caseSelectionRepair:repair});
 assert.equal(continued.item_id,h.id);assert.equal(continued.seconds,1800);assert.equal(continued.elapsed,125);assert.equal(continued.checkpoint.continuedFrom,old.id);assert.deepEqual(continued.checkpoint.caseSelectionRepair,repair);
 repair.evidence.sha256='d'.repeat(64);assert.equal(b.run(continued.id).checkpoint.caseSelectionRepair.evidence.sha256,'c'.repeat(64));assert.deepEqual(state.db.prepare('SELECT * FROM skill_backlog_runs WHERE id=?').get(old.id),frozen);
 assert.equal(b.claim({base:'baseline'}).remaining,1675);
});
test('case-selection repair refuses scientific outcomes, QA work, unresolved custody and malformed provenance',async t=>{
 const repair={fromRevision:'a'.repeat(40),toRevision:'b'.repeat(40),evidence:{path:'/tmp/selection-repair-proof.json',sha256:'c'.repeat(64)}};
 for(const [name,status,checkpoint,result]of [
  ['rejected','rejected',{}, {status:'unsupported',missingCapability:true}],
  ['evaluated','inconclusive',{}, {status:'unsupported',missingCapability:true,report:{results:[{correct:false}]}}],
  ['evaluation-directory','inconclusive',{evaluationDirectory:'/tmp/evaluated'}, {status:'unsupported',missingCapability:true}],
  ['qa','inconclusive',{prepared:{status:'inconclusive',qa:{elapsedSeconds:30}}}, {status:'unsupported',missingCapability:true,reason:'qa-elapsed'}],
  ['prior-qa','inconclusive',{priorQaElapsedSeconds:30}, {status:'unsupported',missingCapability:true}],
  ['unresolved','inconclusive',{unresolved:true}, {status:'unsupported',missingCapability:true}],
  ['unsupported-without-proof','inconclusive',{}, {status:'unsupported',missingCapability:false}]
 ]){
  const {state,b,seed,item}=await fixture(t);item(name,seed());const old=b.claim({base:'baseline'}).run;b.checkpoint(old.id,{elapsed:50,checkpoint});b.finish(old.id,{status,result});
  assert.throws(()=>b.continue(old.id,{base:'baseline',procedure:'Explicit operator repair',caseSelectionRepair:repair}),/Only audited pre-evaluation/);assert.equal(state.db.prepare('SELECT count(*) n FROM skill_backlog_runs').get().n,1);
 }
 for(const invalid of [{...repair,toRevision:repair.fromRevision},{...repair,evidence:{path:'relative.json',sha256:'c'.repeat(64)}},{...repair,evidence:{path:'/tmp/proof',sha256:'not-a-hash'}},{}]){
  const {b,seed,item}=await fixture(t);item('invalid-provenance',seed());const old=b.claim({base:'baseline'}).run;b.finish(old.id,{status:'inconclusive',result:{status:'unsupported',missingCapability:true}});
  assert.throws(()=>b.continue(old.id,{base:'baseline',procedure:'Explicit repair',caseSelectionRepair:invalid}),/immutable evidence/);
 }
});
