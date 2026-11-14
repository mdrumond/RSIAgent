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
 const urgent=item('urgent',group,{priority:100});b.checkpoint(first.run.id,{elapsed:120,checkpoint:{runningAt:1000,stage:'qa'}});
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

test('active-version changes require requalification and retain all old experiment evidence',async t=>{
 const{b,seed,item}=await fixture(t),h=item('h',seed());const first=b.claim({base:'baseline'});b.checkpoint(first.run.id,{elapsed:42});
 assert.throws(()=>b.rebase(h.id,'promoted'),/Settle/);assert.throws(()=>b.claim({base:'promoted'}),/Settle/);
 b.finish(first.run.id,{status:'rejected'});b.rebase(h.id,'promoted');assert.equal(b.claim({base:'promoted'}),null);assert.equal(b.get(h.id).status,'needs-review');
 b.revise(h.id,{status:'queued'});const next=b.claim({base:'promoted'});assert.equal(next.remaining,1800);assert.equal(b.run(first.run.id).elapsed,42);assert.equal(b.run(first.run.id).snapshot.base,'baseline');
});

test('priority changes are durable and same-priority work uses oldest insertion first',async t=>{
 const{state,b,seed,item}=await fixture(t),g=seed(),a=item('first',g),c=item('second',g);
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
