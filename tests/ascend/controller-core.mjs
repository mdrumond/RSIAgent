import test from 'node:test';
import assert from 'node:assert/strict';
import { ActorClock } from '../../controller/budget.mjs';
import { State } from '../../controller/state.mjs';
import { Workflow, stages } from '../../controller/workflow.mjs';

test('elapsed accounting excludes nested transfers but counts compilation and thinking',async()=>{
  let now=0;
  const clock=new ActorClock(10,2,()=>now);
  now=1000;
  await clock.exclude('setup',async()=>{
    now+=5000;
    await clock.exclude('transfer',async()=>{now+=10000;});
  });
  now+=2000;
  assert.equal(clock.elapsed,5);
  assert.equal(clock.remaining,5);
  now+=5000;
  assert.throws(()=>clock.check());
});

test('admission persists before setup; resume preserves attempt and elapsed',async()=>{
  const state=new State(':memory:');
  try {
    state.add({id:'p',goal:'numeric Python',actor_seconds:240});
    const attempt=state.admit('p',{pid:123});
    assert.equal(state.project('p').status,'running');
    assert.throws(()=>state.admit('p',{}));
    state.checkpoint(attempt.id,12,'session.jsonl');
    await assert.rejects(state.resume(attempt.id,async()=>false,{}));
    const resumed=await state.resume(attempt.id,async()=>true,{pid:456});
    assert.equal(resumed.id,attempt.id);
    assert.equal(resumed.elapsed,12);
    assert.equal(resumed.session,'session.jsonl');
    assert.throws(()=>state.checkpoint(attempt.id,5));
    state.finish(attempt.id,'failed');
    assert.equal(state.project('p').status,'failed');
    assert.throws(()=>state.finish(attempt.id,'passed'));
  } finally {state.close();}
});

test('workflow enforces real retrieval, step skills and current-source numeric verification',async()=>{
  let source='a',passed=false;
  const events=[],queries=[];
  const workflow=new Workflow({clock:new ActorClock(240),event:async(...e)=>events.push(e),
    fingerprint:async()=>source,verify:async()=>({passed}),
    readSkill:async stage=>({content:stage,evidence:{path:stage}}),
    search:async query=>{queries.push(query);return {content:[],evidence:[]};}});
  for (const stage of stages) {
    assert.equal(workflow.stage,stage);
    await assert.rejects(workflow.advance());
    await workflow.skill();
    if (['design','optimisation'].includes(stage)) {
      await assert.rejects(workflow.advance({reflection:'prior attempt failed'}));
      await workflow.query('prior successes','good');
      await assert.rejects(workflow.advance({reflection:'prior attempt failed'}));
      await workflow.query('prior failures','negative');
    }
    if (stage==='validation') {
      await workflow.validate();
      await assert.rejects(workflow.advance());
      passed=true;
      await workflow.validate();
    }
    if (stage==='acceptance') {
      source='changed';
      await assert.rejects(workflow.advance());
      source='a';
    }
    await workflow.advance({reflection:'Prior evidence rules out the failed strategy'});
  }
  assert.equal(queries.length,4);
  assert.equal(events.at(-1)[0],'accepted');
});

test('resumed workflow retains stage and prior evidence without replaying execution',async()=>{
  const events=[];
  const options={clock:new ActorClock(240),event:async(type,data)=>events.push({type,data}),
    fingerprint:async()=> 'source',verify:async()=>({passed:true}),
    readSkill:async()=>({content:'skill',evidence:{}}),search:async()=>({content:[],evidence:[]})};
  const first=new Workflow(options);
  await first.skill();await first.advance();
  await first.skill();await first.advance();
  await first.skill();await first.query('success','good');
  const resumed=new Workflow(options).restore(events);
  assert.equal(resumed.stage,'design');
  await assert.rejects(resumed.advance({reflection:'still need negative evidence'}));
  await resumed.query('failure','negative');
  await resumed.advance({reflection:'prior strategy failed; use simpler operation'});
  assert.equal(resumed.stage,'implementation');
  assert.equal(resumed.history.length,3);
});

test('live owners block resume and unverifiable ownership remains unknown',async()=>{
  const {processIdentity,ownerStopped,attemptStopped}=await import('../../controller/lifecycle.mjs');
  assert.equal(await ownerStopped(await processIdentity()),false);
  assert.equal(await ownerStopped({pid:123}),false);
  const owner={...await processIdentity(),start:'different-process'};
  const state={events:()=>[{type:'execution-handle',data:{handle:'known'}}]};
  assert.equal(await attemptStopped(state,{id:'a',owner},{observe:async()=>({state:'unknown'})}),false);
  assert.equal(await attemptStopped(state,{id:'a',owner},{observe:async()=>({state:'stopped'})}),true);
});

test('verifier mutation is detected across original retries and resumed attempts',async()=>{
  const {mkdtemp,mkdir,writeFile,rm}=await import('node:fs/promises');
  const {join}=await import('node:path');const {tmpdir}=await import('node:os');
  const {freezeFiles}=await import('../../controller/evidence.mjs');
  const root=await mkdtemp(join(tmpdir(),'ascend-verifier-')),state=new State(':memory:');
  try {
    const source=join(root,'source');await mkdir(source);
    const verifier=join(root,'verify.py');await writeFile(verifier,'assert False');
    state.add({id:'p',goal:'numeric',actor_seconds:240,source,verifier:{command:'python verify.py',files:[verifier]}});
    const hashes=state.project('p').spec.verifier.hashes;
    const check=await freezeFiles([verifier],source,hashes);await check();
    await writeFile(verifier,'assert True');
    await assert.rejects(check());
    await assert.rejects(freezeFiles([verifier],source,hashes));
  }finally{state.close();await rm(root,{recursive:true,force:true});}
});
