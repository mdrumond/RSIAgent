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

test('private provider setup reads its key without exposing dotenv values to actor processes',async()=>{
  const {mkdtemp,writeFile,rm}=await import('node:fs/promises');
  const {join}=await import('node:path');const {tmpdir}=await import('node:os');
  const {modelFromProfile}=await import('../../controller/model.mjs');
  const root=await mkdtemp(join(tmpdir(),'ascend-provider-'));
  const name='ASCEND_FIXTURE_PROVIDER_KEY',other='ASCEND_FIXTURE_UNRELATED';
  try {
    const file=join(root,'credentials.env');await writeFile(file,`${name}=fixture-only\n${other}=private-fixture\n`);
    const {model}=await modelFromProfile({provider:'deepseek',model:'deepseek-flash',
      base_url:'http://127.0.0.1',credential_env:name,credentials_file:file,max_tokens:1024});
    assert.equal(model.maxTokens,1024);
    assert.equal(process.env[name],undefined);assert.equal(process.env[other],undefined);
  }finally{await rm(root,{recursive:true,force:true});}
});

test('stale acceptance resumes through revision and verifies edited source afresh',async()=>{
  let source='original',verified=0;const events=[];
  const options={clock:new ActorClock(240),event:async(type,data)=>events.push({type,data}),
    fingerprint:async()=>source,verify:async()=>{verified++;return {passed:true};},
    readSkill:async()=>({content:'guidance',evidence:{}}),search:async()=>({content:[],evidence:[]})};
  async function step(flow) {
    await flow.skill();
    if(['design','optimisation'].includes(flow.stage))for(const kind of ['good','negative'])await flow.query('prior evidence',kind);
    if(flow.stage==='validation')await flow.validate();
    return flow.advance({reflection:'Use prior failures to validate the revised source'});
  }
  const first=new Workflow(options);
  while(first.stage!=='acceptance')await step(first);
  source='edited';const resumed=new Workflow(options).restore(events);await resumed.skill();
  await assert.rejects(resumed.advance());
  assert.equal(resumed.stage,'acceptance');
  await assert.rejects(resumed.validate());
  assert.equal(verified,1);assert.equal(resumed.stage,'acceptance');
  assert.equal(events.some(e=>e.type==='accepted'),false);
  await resumed.advance({next:'implementation',reflection:'Final edit invalidated prior verification'});
  while(resumed.stage!=='acceptance')await step(resumed);
  assert.equal((await step(resumed)).accepted,true);assert.equal(verified,2);
});

test('source fingerprints distinguish file boundaries from arbitrary binary content',async()=>{
  const {mkdtemp,writeFile,rm}=await import('node:fs/promises');
  const {join}=await import('node:path');const {tmpdir}=await import('node:os');
  const {fingerprint}=await import('../../controller/evidence.mjs');
  const root=await mkdtemp(join(tmpdir(),'ascend-tree-'));
  try {
    await writeFile(join(root,'a'),'x');await writeFile(join(root,'b'),'y');
    const first=await fingerprint(root);
    await rm(join(root,'b'));await writeFile(join(root,'a'),Buffer.from('xb\0y'));
    assert.notEqual(await fingerprint(root),first);
  }finally{await rm(root,{recursive:true,force:true});}
});

test('parallel workers retain all evidence in the shared WAL ledger',async()=>{
  const {mkdtemp,rm}=await import('node:fs/promises');const {spawn}=await import('node:child_process');
  const {once}=await import('node:events');const {join}=await import('node:path');const {tmpdir}=await import('node:os');
  const root=await mkdtemp(join(tmpdir(),'ascend-ledger-')),path=join(root,'state.sqlite');
  const state=new State(path);
  try {
    const module=new URL('../../controller/state.mjs',import.meta.url).href;
    const code="const {State}=await import(process.argv[2]);const s=new State(process.argv[1]);for(let i=0;i<100;i++)s.event(null,'parallel',{pid:process.pid,i});s.close();";
    const children=Array.from({length:2},()=>spawn(process.execPath,['--input-type=module','-e',code,path,module],{stdio:'ignore'}));
    for(const result of await Promise.all(children.map(child=>once(child,'exit'))))assert.equal(result[0],0);
    assert.equal(state.db.prepare("SELECT count(*) AS n FROM events WHERE type='parallel'").get().n,200);
  }finally{state.close();await rm(root,{recursive:true,force:true});}
});

test('orphan closure charges actual execution once without charging observer downtime',async()=>{
  const state=new State(':memory:');
  try {
    state.add({id:'p',goal:'numeric',actor_seconds:120});
    const attempt=state.admit('p',{...await (await import('../../controller/lifecycle.mjs')).processIdentity(),start:'previous-process'});
    state.checkpoint(attempt.id,16);
    state.event(attempt.id,'execution-starting',{operation:'op',startElapsed:10});
    state.event(attempt.id,'execution-submitting',{operation:'op',submission:'native',actor_elapsed:16});
    const {attemptStopped}=await import('../../controller/lifecycle.mjs');
    let observations=0;
    const environment={observe:async()=>({state:'stopped'}),reconcileOperation:async()=>{
      observations++;return {closed:true,executionSeconds:30,handles:['native:job']};
    }};
    assert.equal(await attemptStopped(state,state.attempt(attempt.id),environment),true);
    assert.equal(state.attempt(attempt.id).elapsed,40);
    assert.equal(await attemptStopped(state,state.attempt(attempt.id),environment),true);
    assert.equal(state.attempt(attempt.id).elapsed,40);assert.equal(observations,1);
    state.event(attempt.id,'execution-starting',{operation:'missing',startElapsed:40});
    assert.equal(await attemptStopped(state,state.attempt(attempt.id),{...environment,
      reconcileOperation:async()=>({closed:false,accountingUnresolved:true})}),false);
  }finally{state.close();}
});

test('confirmed excluded setup resumes without an actor origin and does not change its budget',async()=>{
  const {processIdentity,attemptStopped}=await import('../../controller/lifecycle.mjs');
  const state=new State(':memory:');
  try {
    state.add({id:'p',goal:'numeric',actor_seconds:120});
    const attempt=state.admit('p',{...await processIdentity(),start:'previous-process'});
    state.checkpoint(attempt.id,23,'retained-native-session');
    state.event(attempt.id,'execution-submitting',{
      submission:'setup',kind:'download',accountingExcluded:true});
    let reconciliations=0;
    const environment={observe:async()=>({state:'stopped'}),reconcileOperation:async()=>{
      reconciliations++;return {closed:true,executionSeconds:0,handles:['native:transfer']};
    }};
    const resumed=await state.resume(attempt.id,
      current=>attemptStopped(state,current,environment),await processIdentity());
    assert.equal(resumed.id,attempt.id);
    assert.equal(resumed.elapsed,23);
    assert.equal(resumed.session,'retained-native-session');
    assert.equal(resumed.outcome,null);
    assert.equal(await attemptStopped(state,state.attempt(attempt.id),environment),false);
    // A repeated closure observation after this owner stops is idempotent.
    const stopped={...resumed,owner:attempt.owner};
    assert.equal(await attemptStopped(state,stopped,environment),true);
    assert.equal(reconciliations,1);
    assert.equal(state.attempt(attempt.id).elapsed,23);
  }finally{state.close();}
});

test('missing actor origins still block unexcluded, mixed, active and unresolved work',async t=>{
  const {processIdentity,attemptStopped}=await import('../../controller/lifecycle.mjs');
  for(const scenario of [
    {name:'unclassified'}, {name:'charged',excluded:false},
    {name:'mixed',excluded:true,mixed:true},
    {name:'nonzero',excluded:true,seconds:1},
    {name:'unresolved',excluded:true,closed:false},
    {name:'actor boundary',excluded:true,started:true},
  ])await t.test(scenario.name,async()=>{
    const state=new State(':memory:');
    try {
      state.add({id:'p',goal:'numeric',actor_seconds:120});
      const attempt=state.admit('p',{...await processIdentity(),start:'previous-process'});
      state.checkpoint(attempt.id,23,'retained-native-session');
      state.event(attempt.id,'execution-submitting',{
        operation:'op',submission:'setup',accountingExcluded:scenario.excluded});
      if(scenario.mixed)state.event(attempt.id,'execution-submitting',{
        operation:'op',submission:'workload',accountingExcluded:false});
      if(scenario.started)state.event(attempt.id,'execution-starting',{operation:'op'});
      const environment={observe:async()=>({state:'stopped'}),reconcileOperation:async()=>({
        closed:scenario.closed??true,executionSeconds:scenario.seconds??0,handles:['native:job']})};
      await assert.rejects(state.resume(attempt.id,
        current=>attemptStopped(state,current,environment),await processIdentity()));
      assert.equal(state.attempt(attempt.id).elapsed,23);
      assert.equal(state.attempt(attempt.id).session,'retained-native-session');
      assert.equal(state.events(attempt.id).some(e=>e.type==='execution-reconciled'),false);
    }finally{state.close();}
  });
});
