import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {Worker} from 'node:worker_threads';
import {once} from 'node:events';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {syncCampaignPlan,startCampaignEntry,advanceCampaignEntry,currentCampaignEntry} from '../../controller/progress.mjs';

async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'campaign-transition-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const file=join(root,'state.sqlite'),state=new State(file),queue=new RecoveryQueue(state);
  t.after(()=>state.close());
  const config={campaign_plan:[{id:'first',phase:'phase1',new_projects:1},{id:'second',phase:'phase2',new_projects:1}],projects:[]};
  const project={id:'original',goal:'test',actor_seconds:10,campaign_entry:'first'};
  state.add(project);config.projects.push(project);syncCampaignPlan(state,config);
  const job=queue.claim(project.id,0);
  queue.recordOriginal(job.attempt.id,{passed:true,findings:'observed',evidence:[{collection:'memory',entry_id:'one'}]},0);
  queue.acknowledgeLesson(queue.lessons()[0].id);state.event(job.attempt.id,'postprocess-completed',{});
  return {state,file,config,attempt:job.attempt.id};
}

// Wake an already-open independent writer after the transition reads its plan.
// The assertion concerns an actual event commit, not SQL wording or query plans.
function duringPlanRead(state,action) {
  const prepare=state.db.prepare.bind(state.db);let fired=false;
  state.db.prepare=(...args)=>{
    const statement=prepare(...args),get=statement.get.bind(statement);
    statement.get=(...parameters)=>{
      const value=get(...parameters);
      if(!fired&&state.db.isTransaction&&value?.value) {
        let saved;try{saved=JSON.parse(value.value);}catch{}
        if(Array.isArray(saved?.plan)){fired=true;action();}
      }
      return value;
    };
    return statement;
  };
  return ()=>{state.db.prepare=prepare;};
}

test('campaign transition lets an unrelated actor event commit despite a long historical event ledger',async t=>{
  const f=await fixture(t);
  // Historical admissions occur after unrelated telemetry, as in a long campaign.
  f.state.db.exec(`WITH RECURSIVE rows(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM rows WHERE n<150000)
    INSERT INTO events(attempt,type,data,created) SELECT NULL,'pi-event','{}','2026-01-01' FROM rows`);
  f.state.db.exec('BEGIN IMMEDIATE');
  for(let n=0;n<80;n++) {
    const project={id:'history-'+n,goal:'historical',actor_seconds:10,campaign_entry:'first'};
    f.config.projects.push(project);f.state.add(project);
    const id='history-attempt-'+n;
    f.state.db.prepare("INSERT INTO attempts(id,project,outcome,created) VALUES (?,?,'passed','2026-01-01')").run(id,project.id);
    f.state.event(id,'job-admitted',{kind:'original',campaign_entry:'first'});
  }
  f.state.db.exec('COMMIT');
  const gate=new SharedArrayBuffer(4),signal=new Int32Array(gate);
  const worker=new Worker(`const {parentPort,workerData}=await import('node:worker_threads');
    const {State}=await import(workerData.module);
    const state=new State(workerData.file);state.db.exec('PRAGMA busy_timeout=100');
    parentPort.postMessage({ready:true});Atomics.wait(new Int32Array(workerData.gate),0,0);
    try{state.event(workerData.attempt,'actor-observation',{retained:true});parentPort.postMessage({written:true});}
    catch(error){parentPort.postMessage({written:false,error:error.message});}
    finally{state.close();}`,{eval:true,workerData:{module:new URL('../../controller/state.mjs',import.meta.url).href,
      file:f.file,attempt:f.attempt,gate}});
  t.after(()=>worker.terminate());
  await once(worker,'message');
  const result=once(worker,'message');
  const restore=duringPlanRead(f.state,()=>{Atomics.store(signal,0,1);Atomics.notify(signal,0);});
  try{assert.equal(advanceCampaignEntry(f.state,f.config),true);}finally{restore();}
  const [observation]=await result;
  assert.equal(observation.written,true,observation.error);
  assert.equal(f.state.queryEvents(f.attempt,{types:['actor-observation']}).length,1);
  assert.equal(currentCampaignEntry(f.state,f.config).id,'second');
  assert.equal(f.state.queryEvents(null,{types:['campaign-entry-completed']}).length,1);
});

test('plan validation still rolls back a frozen membership edit without advancing the cursor',async t=>{
  const f=await fixture(t);startCampaignEntry(f.state,f.config);
  const invalid={...f.config,projects:[{...f.config.projects[0],campaign_entry:'second'}]};
  assert.throws(()=>advanceCampaignEntry(f.state,invalid),/admitted/);
  assert.equal(currentCampaignEntry(f.state,f.config).id,'first');
  assert.equal(f.state.queryEvents(null,{types:['campaign-entry-completed']}).length,0);
  assert.equal(advanceCampaignEntry(f.state,f.config),true);
});

test('two connections cannot advance the same entry twice or admit outside the current entry',async t=>{
  const f=await fixture(t),other=new State(f.file),queue=new RecoveryQueue(other);t.after(()=>other.close());
  assert.equal(advanceCampaignEntry(f.state,f.config),true);
  assert.equal(advanceCampaignEntry(other,f.config),false);
  other.add({id:'late-first',goal:'test',actor_seconds:10,campaign_entry:'first'});
  assert.throws(()=>queue.claim('late-first',0),/outside/);
  assert.equal(f.state.queryEvents(null,{types:['campaign-entry-completed']}).length,1);
  assert.throws(()=>syncCampaignPlan(other,{...f.config,campaign_plan:[{...f.config.campaign_plan[0],new_projects:2},f.config.campaign_plan[1]]}),/started/);
  assert.equal(currentCampaignEntry(other,f.config).id,'second');
});

test('membership validation keeps the earliest admission across repeated and legacy attempts',async t=>{
  const f=await fixture(t);
  // A later attempt's admission must not replace the project's original batch.
  f.state.db.prepare("INSERT INTO attempts(id,project,outcome,created) VALUES ('later','original','failed','2026-01-02')").run();
  f.state.event('later','job-admitted',{kind:'original',campaign_entry:'second'});
  assert.doesNotThrow(()=>syncCampaignPlan(f.state,f.config));
  assert.throws(()=>syncCampaignPlan(f.state,{...f.config,projects:[{...f.config.projects[0],campaign_entry:'second'}]}),/admitted/);
  const legacy={id:'legacy',goal:'test',actor_seconds:10,campaign_entry:'first'};
  f.state.add(legacy);
  f.state.db.prepare("INSERT INTO attempts(id,project,outcome,created) VALUES ('legacy-attempt','legacy','failed','2026-01-02')").run();
  assert.doesNotThrow(()=>syncCampaignPlan(f.state,{...f.config,projects:[...f.config.projects,legacy]}));
  assert.throws(()=>syncCampaignPlan(f.state,{...f.config,projects:[...f.config.projects,{...legacy,campaign_entry:'second'}]}),/admitted/);
});
