import test from 'node:test';
import {randomUUID} from 'node:crypto';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {failureInvestigations,claimFailureInvestigation} from '../../controller/failure-patterns.mjs';
const failure={phase:'verification',kind:'mismatch',message:'Expected 42, observed 0',evidence:[{path:'verify.log'}]};
const attempt=(n,extra={})=>({id:`a${n}`,project:`p${n}`,outcome:'failed',target:'a5',failures:[failure],...extra});
test('three distinct matching failures within latest ten completed attempts',()=>{
  assert.deepEqual(failureInvestigations([attempt(1),attempt(2)]),[]);
  assert.deepEqual(failureInvestigations([attempt(1,{failures:[failure,failure,failure]})]),[]);
  const rows=[attempt(1),attempt(2),attempt(3)];
  const [candidate]=failureInvestigations(rows);
  assert.deepEqual(candidate.attempts.map(a=>a.id),['a1','a2','a3']);
  assert.deepEqual(candidate.attempts[0].evidence,failure.evidence);
  assert.deepEqual(failureInvestigations(rows,[candidate]),[]);
  assert.equal(failureInvestigations([...rows,...Array.from({length:7},(_,i)=>attempt(i+4,{outcome:'passed'})),attempt(20,{outcome:null})]).length,1);
  assert.deepEqual(failureInvestigations([...rows,...Array.from({length:8},(_,i)=>attempt(i+4,{outcome:'passed'}))]),[]);
});
test('target and factual observations remain distinct; passing attempts never count',()=>{
  assert.deepEqual(failureInvestigations([attempt(1),attempt(2),attempt(3,{target:'a3'})]),[]);
  assert.deepEqual(failureInvestigations([attempt(1),attempt(2),attempt(3,{outcome:'passed'})]),[]);
  assert.deepEqual(failureInvestigations([attempt(1),attempt(2),attempt(3,{failures:[{...failure,message:'Expected 42, observed 1'}]})]),[]);
});
test('durable claim deduplicates and leaves controller project and attempt untouched',()=>{
  const state=new State(':memory:');
  try {
    state.add({id:'p',goal:'original',actor_seconds:120});
    const active=state.admit('p',{}),before=state.project('p');
    const [candidate]=failureInvestigations([attempt(1),attempt(2),attempt(3)]);
    assert.equal(claimFailureInvestigation(state,candidate),true);
    assert.equal(claimFailureInvestigation(state,JSON.parse(JSON.stringify(candidate))),false);
    assert.equal(state.events(null).length,1);
    assert.deepEqual(state.project('p'),before);
    assert.deepEqual(state.attempt(active.id),active);
  }finally{state.close();}
});

test('claim remains deduplicated after reopening controller state',()=>{
  const root=mkdtempSync(join(tmpdir(),'failure-patterns-'));
  let state=new State(join(root,'state.sqlite'));
  try {
    const [candidate]=failureInvestigations([attempt(1),attempt(2),attempt(3)]);
    assert.equal(claimFailureInvestigation(state,candidate),true);
    state.close();state=new State(join(root,'state.sqlite'));
    assert.equal(claimFailureInvestigation(state,candidate),false);
    assert.deepEqual(failureInvestigations([attempt(1),attempt(2),attempt(3)],state.events(null).map(e=>e.data)),[]);
  }finally{state.close();rmSync(root,{recursive:true,force:true});}
});

test('same observation across distinct executions matches and retains raw operation IDs',()=>{
  const operations=Array.from({length:3},()=>randomUUID());
  const rows=operations.map((operation,i)=>attempt(i,{failures:[{...failure,operation}]}));
  const candidates=failureInvestigations(rows);
  assert.equal(candidates.length,1);
  assert.deepEqual(candidates[0].attempts.map(a=>a.operation),operations);
  assert.deepEqual(candidates[0].attempts.map(a=>a.evidence),operations.map(()=>failure.evidence));
  const [withoutOperations]=failureInvestigations(rows.map(a=>({...a,failures:[failure]})));
  assert.equal(candidates[0].key,withoutOperations.key);
  assert.deepEqual(failureInvestigations(rows,[withoutOperations]),[]);
});
