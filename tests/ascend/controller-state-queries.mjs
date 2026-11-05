import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {DatabaseSync} from 'node:sqlite';
import {State} from '../../controller/state.mjs';
import {operationsStopped} from '../../controller/lifecycle.mjs';

function admit(state,id='p') {
  state.add({id,goal:'Observe numeric behavior',actor_seconds:30});
  return state.admit(id,{});
}
function fixture(t) {
  const state=new State(':memory:');t.after(()=>state.close());
  return {state,attempt:admit(state)};
}
const payloads=rows=>rows.map(row=>row.data.n);

test('bounded filters select matching events before limiting, in either direction',t=>{
  const {state,attempt}=fixture(t),other=admit(state,'other');
  for(let n=0;n<24;n++) {
    state.event(attempt.id,n%3===0?'verification':'pi-event',{n});
    state.event(other.id,'verification',{n:100+n});
    state.event(null,'verification',{n:200+n});
  }
  assert.deepEqual(payloads(state.queryEvents(attempt.id,{types:['verification'],limit:3})),[0,3,6]);
  const newest=state.queryEvents(attempt.id,{types:['verification'],order:'desc',limit:3});
  assert.deepEqual(payloads(newest),[21,18,15]);
  assert.deepEqual(payloads(state.queryEvents(attempt.id,{excludeTypes:['pi-event'],order:'desc',limit:3})),[21,18,15]);
  assert.deepEqual(payloads(state.queryEvents(null,{limit:2})),[200,201]);
  assert.deepEqual(state.queryEvents('missing'),[]);
  assert.deepEqual(state.queryEvents(attempt.id,{types:[]}),[]);
  assert.equal(state.queryEvents(attempt.id,{excludeTypes:[]}).length,24);
  assert.ok(newest.every(row=>Number.isSafeInteger(row.sequence)&&Number.isFinite(Date.parse(row.created))));
  assert.equal(state.events(attempt.id).length,24);
  assert.deepEqual(Object.keys(state.events(attempt.id)[0]).sort(),['data','type']);
});

test('exclusive sequence cursors page without gaps across interleaved attempts',t=>{
  const {state,attempt}=fixture(t),other=admit(state,'other');
  for(let n=0;n<11;n++) {
    state.event(attempt.id,n%2?'rejected':'verification',{n});
    state.event(other.id,'verification',{n:-n});
  }
  const ascending=[],descending=[];
  let after,before;
  for(;;) {
    const page=state.queryEvents(attempt.id,{after,limit:3});
    if(!page.length)break;
    ascending.push(...page);after=page.at(-1).sequence;
  }
  for(;;) {
    const page=state.queryEvents(attempt.id,{before,order:'desc',limit:4});
    if(!page.length)break;
    descending.push(...page);before=page.at(-1).sequence;
  }
  assert.deepEqual(payloads(ascending),Array.from({length:11},(_,n)=>n));
  assert.deepEqual(descending.reverse(),ascending);
  assert.deepEqual(payloads(state.queryEvents(attempt.id,{
    after:ascending[2].sequence,before:ascending[8].sequence,
    types:['verification','rejected'],excludeTypes:['rejected'],limit:10})),[4,6]);
});

test('queries do not decode excluded events or records beyond the SQL limit',t=>{
  const {state,attempt}=fixture(t);
  state.db.prepare('INSERT INTO events(attempt,type,data,created) VALUES (?,?,?,?)')
    .run(attempt.id,'pi-event','not JSON','2026-01-01');
  state.event(attempt.id,'verification',{n:1});
  state.event(attempt.id,'verification',{n:2});
  assert.deepEqual(payloads(state.queryEvents(attempt.id,{types:['verification'],limit:1})),[1]);
  assert.deepEqual(payloads(state.queryEvents(attempt.id,{order:'desc',limit:1})),[2]);
});

test('invalid query bounds fail rather than accidentally returning all history',t=>{
  const {state,attempt}=fixture(t);
  for(const limit of [0,-1,1.5,1001,Infinity,'10',null])
    assert.throws(()=>state.queryEvents(attempt.id,{limit}));
  for(const options of [{order:'random'},{after:-1},{before:1.2},{types:'verification'},
    {types:[null]},{excludeTypes:['']},{after:'1'}])
    assert.throws(()=>state.queryEvents(attempt.id,options));
  for(let n=0;n<105;n++)state.event(attempt.id,'sample',{n});
  assert.equal(state.queryEvents(attempt.id).length,100);
  assert.equal(state.queryEvents(attempt.id,{limit:1000}).length,105);
});

test('version-one databases gain query support without rewriting event history',t=>{
  const root=mkdtempSync(join(tmpdir(),'state-query-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
  const file=join(root,'state.sqlite');
  let state=new State(file);
  const attempt=admit(state);
  state.event(attempt.id,'verification',{n:1});state.close();
  const db=new DatabaseSync(file);
  db.exec('DROP INDEX events_attempt_sequence; DROP INDEX events_attempt_type_sequence;');db.close();
  state=new State(file);t.after(()=>state.close());
  assert.deepEqual(payloads(state.queryEvents(attempt.id,{types:['verification'],limit:1})),[1]);
  state.event(attempt.id,'verification',{n:2});
  assert.deepEqual(payloads(state.queryEvents(attempt.id,{order:'desc',limit:1})),[2]);
  assert.equal(state.project('p').status,'running');
});

test('factual failures survive reopen and never settle or authorize unsafe resume',async t=>{
  const root=mkdtempSync(join(tmpdir(),'state-failure-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
  const file=join(root,'state.sqlite');let state=new State(file);
  const attempt=admit(state);state.checkpoint(attempt.id,4,'actor.jsonl');
  state.event(attempt.id,'native-bash-started',{operation:'shell'});
  const fact={phase:'execution',kind:'transport',message:'Observation timed out',operation:'shell',
    evidence:[{path:'observe.log',sha256:'a'.repeat(64)}]};
  const sequence=state.recordFailure(attempt.id,fact);
  fact.evidence[0].path='changed';
  state.close();state=new State(file);t.after(()=>state.close());
  const records=state.failures(attempt.id,{order:'desc',limit:1});
  assert.equal(records[0].sequence,sequence);
  assert.deepEqual(records[0].data,{...fact,evidence:[{path:'observe.log',sha256:'a'.repeat(64)}]});
  assert.equal(state.attempt(attempt.id).outcome,null);
  assert.equal(state.attempt(attempt.id).elapsed,4);
  assert.equal(state.attempt(attempt.id).session,'actor.jsonl');
  assert.equal(state.project('p').status,'running');
  assert.equal(await operationsStopped(state,state.attempt(attempt.id),{}),false);
  await assert.rejects(state.resume(attempt.id,async()=>false,{}));
  state.event(attempt.id,'native-bash-completed',{operation:'shell',actor_elapsed:5});
  assert.equal(await operationsStopped(state,state.attempt(attempt.id),{}),true);
  state.finish(attempt.id,'failed');
  state.recordFailure(attempt.id,{phase:'summary',kind:'provider',message:'Summary unavailable'});
  assert.equal(state.failures(attempt.id).length,2);
  assert.equal(state.attempt(attempt.id).outcome,'failed');
  assert.equal(state.project('p').status,'failed');
});

test('invalid failure records leave the ledger and attempt unchanged',t=>{
  const {state,attempt}=fixture(t),fact={phase:'setup',kind:'io',message:'Missing file'};
  for(const change of [{phase:''},{kind:4},{message:null},{evidence:['file']},
    {evidence:[{path:''}]},{evidence:[{path:'file',sha256:'bad'}]},{operation:''}])
    assert.throws(()=>state.recordFailure(attempt.id,{...fact,...change}));
  assert.throws(()=>state.recordFailure(null,fact));
  assert.throws(()=>state.recordFailure('unknown',fact));
  assert.deepEqual(state.events(attempt.id),[]);
  assert.equal(state.project('p').status,'running');
});
