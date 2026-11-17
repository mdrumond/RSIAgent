import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,mkdir,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {processIdentity} from '../../controller/lifecycle.mjs';
import {writeAttemptJob,readAttemptJob} from '../../cli/attempt-job.mjs';
import {migrateForegroundCheckpoint} from '../../cli/foreground-resource.mjs';
import {setup,rows,until} from './driver_setup.mjs';
import {pathToFileURL} from 'node:url';

async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'foreground-slot-')),state=new State(join(root,'state.sqlite'));
  t.after(async()=>{state.close();await rm(root,{recursive:true,force:true});});
  state.add({id:'p',goal:'retained work',actor_seconds:3600});new RecoveryQueue(state);
  const attempt=state.admit('p',{pid:2147483647,start:'old',boot:'old'});
  state.checkpoint(attempt.id,123.75,'unchanged-native-session');
  state.db.exec('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
  state.db.prepare('INSERT INTO control VALUES (?,?)').run('owner',JSON.stringify(await processIdentity()));
  const resources=[0,1,2].map(hostSlot=>({id:'actual-slot-'+hostSlot,available:true,
    metadata:{kind:'host-slot',hostSlot,backgroundSlot:hostSlot===0,pool:'fixture-pool',target:'fixture-target'}}));
  const config={format:1,root,environment:{module:'registered-environment.mjs',options:{mode:'cpl',target:'fixture-target',runtime:'registered'}}};
  await writeFile(join(root,'instance.json'),JSON.stringify(config));
  const job={attempt:{...state.attempt(attempt.id)},projectId:'p',kind:'retry',resume:true,wave:21,
    actor_seconds:3600,skillVersion:'accepted',options:{continuous:true},resource:resources[0]};
  await writeAttemptJob(state,root,attempt.id,job);
  state.db.prepare('INSERT INTO resource_leases VALUES (?,?)').run(resources[0].id,attempt.id);
  const calls=[],environment={reconcileLease:async()=>{calls.push('physical-closure');return true;}};
  const input={state,config,job,resources,reserved:resources[0],createEnvironment:async prior=>{calls.push(prior.id);return environment;}};
  return {root,state,attempt,job,resources,config,input,environment,calls,file:join(root,'attempts',attempt.id,'job.json')};
}

test('a stopped foreground checkpoint moves to a real free slot without replacing its work or elapsed allocation',async t=>{
  const f=await fixture(t),before=f.state.attempt(f.attempt.id);
  assert.equal((await migrateForegroundCheckpoint(f.input)).id,f.resources[1].id);
  assert.deepEqual(f.calls,[f.resources[0].id,'physical-closure']);
  assert.deepEqual(f.state.attempt(f.attempt.id),before);
  assert.equal(f.state.db.prepare('SELECT resource FROM resource_leases WHERE attempt=?').get(f.attempt.id).resource,f.resources[1].id);
  assert.deepEqual(await readAttemptJob(f.state,f.root,f.attempt.id),{...f.job,resource:f.resources[1]});
  assert.equal(f.state.queryEvents(f.attempt.id,{types:['foreground-resource-migrated']}).length,1);
  assert.equal(await migrateForegroundCheckpoint(f.input),null);
});

test('unresolved native work, an unknown physical lease, or a live owner never yields the reserved slot',async t=>{
  for(const kind of ['native','lease','owner'])await t.test(kind,async t=>{
    const f=await fixture(t),before=await readFile(f.file,'utf8');
    if(kind==='native'){
      f.state.event(f.attempt.id,'execution-starting',{operation:'pending',startElapsed:123.75});
      f.environment.reconcileOperation=async()=>({closed:false});
    }else if(kind==='lease')f.environment.reconcileLease=async()=>false;
    else f.state.db.prepare('UPDATE attempts SET owner=? WHERE id=?').run(JSON.stringify(await processIdentity()),f.attempt.id);
    assert.equal(await migrateForegroundCheckpoint(f.input),null);
    assert.equal(f.state.db.prepare('SELECT resource FROM resource_leases').get().resource,f.resources[0].id);
    assert.equal(await readFile(f.file,'utf8'),before);
    assert.equal(f.state.queryEvents(f.attempt.id,{types:['foreground-resource-migrated']}).length,0);
  });
});

test('migration refuses a changed runtime binding or another controller',async t=>{
  for(const kind of ['runtime','controller'])await t.test(kind,async t=>{
    const f=await fixture(t);
    if(kind==='runtime')for(const r of f.resources.slice(1))r.metadata.runtime='other-runtime';
    else f.state.db.prepare("UPDATE control SET value=? WHERE key='owner'").run(JSON.stringify({pid:2147483647,start:'other',boot:'other'}));
    assert.equal(await migrateForegroundCheckpoint(f.input),null);assert.deepEqual(f.calls,[]);
  });
});

test('a foreground slot claimed during closure is not stolen and a failed snapshot write rolls back the lease',async t=>{
  for(const kind of ['contended','publication'])await t.test(kind,async t=>{
    const f=await fixture(t);
    if(kind==='contended'){
      f.state.add({id:'q',goal:'another owner',actor_seconds:60});const other=f.state.admit('q',{pid:2147483647,start:'old',boot:'old'});
      f.environment.reconcileLease=async()=>{f.state.db.prepare('INSERT INTO resource_leases VALUES (?,?)').run(f.resources[1].id,other.id);return true;};
      assert.equal(await migrateForegroundCheckpoint(f.input),null);
    }else{
      f.state.db.exec("CREATE TRIGGER prevent_job_write BEFORE INSERT ON attempt_job_manifests BEGIN SELECT RAISE(FAIL,'snapshot unavailable'); END");
      await assert.rejects(migrateForegroundCheckpoint(f.input),/snapshot unavailable/);
    }
    assert.equal(f.state.db.prepare('SELECT resource FROM resource_leases WHERE attempt=?').get(f.attempt.id).resource,f.resources[0].id);
    assert.deepEqual(await readAttemptJob(f.state,f.root,f.attempt.id),f.job);
  });
});

test('a file-publication failure cannot revive the old reserved resource after the atomic transfer',async t=>{
  const f=await fixture(t);
  f.environment.reconcileLease=async()=>{await rm(f.file);await mkdir(f.file);return true;};
  assert.equal(await migrateForegroundCheckpoint(f.input),null);
  assert.equal(f.state.db.prepare('SELECT resource FROM resource_leases WHERE attempt=?').get(f.attempt.id).resource,f.resources[1].id);
  assert.equal(f.state.queryEvents(f.attempt.id,{types:['attempt-metadata-unavailable']}).length,1);
  await rm(f.file,{recursive:true});
  assert.deepEqual(await readAttemptJob(f.state,f.root,f.attempt.id),{...f.job,resource:f.resources[1]});
});

test('the native campaign resume migrates its retained foreground checkpoint before launching the actor',async t=>{
  const f=await setup(t,'phase2',['retained']);
  const current=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));
  const environment=join(f.root,'marked-environment.mjs');
  const generic=pathToFileURL(join(f.root,'machinery','integrations/environments/index.mjs')).href;
  const resources=[0,1].map(hostSlot=>({id:'fixture-slot-'+hostSlot,available:true,
    metadata:{kind:'host-slot',pool:'fixture-pool',hostSlot,backgroundSlot:hostSlot===0,target:'fixture-target'}}));
  await writeFile(environment,`import {createEnvironment as generic} from ${JSON.stringify(generic)};
    export async function createEnvironment(options,context){return {...await generic(options,context),
      resources:async()=>${JSON.stringify(resources)},reconcileLease:async()=>true};}`);
  const input=join(f.root,'resume-settings.json');
  await writeFile(input,JSON.stringify({environment:{module:environment,options:{mode:'local'}},
    skills:{...current.skills,background:true,tuningDisabled:true},campaign_seconds:30,concurrency:2,recovery:false}));
  await f.run('config','--config',input);
  const state=new State(join(f.root,'state.sqlite'));let id;
  try {
    const attempt=state.admit('retained',{pid:2147483647,start:'old',boot:'old'});id=attempt.id;
    state.checkpoint(id,7.25);new RecoveryQueue(state);
    state.db.exec('CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
    state.db.prepare('INSERT INTO resource_leases VALUES (?,?)').run(resources[0].id,id);
    state.db.prepare("INSERT OR REPLACE INTO control VALUES ('operation','\"campaign\"')").run();
    await writeAttemptJob(state,f.root,id,{attempt:{...state.attempt(id)},project:state.project('retained'),
      kind:'original',wave:1,actor_seconds:120,mode:'phase1',skillVersion:current.skills.activeVersion,
      options:{continuous:true},resource:resources[0],deferFinalization:true});
  }finally{state.close();}
  await f.run('resume');
  const after=new State(join(f.root,'state.sqlite'));
  try {
    assert.equal(after.project('retained').status,'passed');
    assert.ok(after.attempt(id).elapsed>=7.25);
    assert.equal(after.queryEvents(id,{types:['foreground-resource-migrated']}).length,1);
    assert.equal((await readAttemptJob(after,f.root,id)).resource.id,resources[1].id);
  }finally{after.close();}
  f.pass();
});

test('a cached resume whose metadata becomes unwritable cannot abort an unrelated native actor',async t=>{
  const f=await setup(t,'phase2',['blocked','healthy']);
  const config=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));
  const state=new State(join(f.root,'state.sqlite')),ids=[];
  const resources=[0,1].map(hostSlot=>({id:'fixture-slot-'+hostSlot,available:true,
    metadata:{kind:'host-slot',pool:'fixture-pool',hostSlot,backgroundSlot:false,target:'fixture-target'}}));
  try {
    new RecoveryQueue(state);
    for(const [index,project]of ['blocked','healthy'].entries()){
      const a=state.admit(project,{pid:2147483647,start:'old',boot:'old'});ids.push(a.id);
      state.checkpoint(a.id,7.25);
      state.db.prepare('INSERT INTO resource_leases VALUES (?,?)').run(resources[index].id,a.id);
      await writeAttemptJob(state,f.root,a.id,{attempt:{...state.attempt(a.id)},project:state.project(project),
        kind:'original',wave:1,actor_seconds:120,mode:'phase1',skillVersion:config.skills.activeVersion,
        options:{},resource:resources[index]});
    }
  }finally{state.close();}
  const environment=join(f.root,'fault-environment.mjs'),broken=join(f.root,'attempts',ids[0],'job.json');
  const generic=pathToFileURL(join(f.root,'machinery','integrations/environments/index.mjs')).href;
  await writeFile(environment,`import {createEnvironment as generic} from ${JSON.stringify(generic)};
    import {rm,mkdir} from 'node:fs/promises';let corrupted=false;
    export async function createEnvironment(options,context){return {...await generic(options,context),
      resources:async()=>{if(!corrupted){corrupted=true;await rm(${JSON.stringify(broken)});await mkdir(${JSON.stringify(broken)});}
        return ${JSON.stringify(resources)};}};}`);
  const input=join(f.root,'fault-settings.json');
  await writeFile(input,JSON.stringify({environment:{module:environment,options:{mode:'local'}},concurrency:2,recovery:false}));
  await f.run('config','--config',input);await f.run('resume');
  const after=new State(join(f.root,'state.sqlite'));
  try {
    assert.equal(after.project('healthy').status,'passed');
    assert.equal(after.attempt(ids[0]).outcome,null);assert.equal(after.attempt(ids[0]).elapsed,7.25);
    assert.equal(after.db.prepare('SELECT resource FROM resource_leases WHERE attempt=?').get(ids[0]).resource,resources[0].id);
    assert.equal(after.queryEvents(ids[0],{types:['attempt-metadata-unavailable']}).length,1);
    assert.equal(after.queryEvents(null,{types:['scheduler-error']}).length,0);
  }finally{after.close();}
  f.pass();
});

test('the promotion gate frees the A/B slot without starting the retained foreground actor',async t=>{
  const f=await setup(t,'phase2',['retained']);
  const config=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));
  const resources=[0,1].map(hostSlot=>({id:'fixture-slot-'+hostSlot,available:true,
    metadata:{kind:'host-slot',pool:'fixture-pool',hostSlot,backgroundSlot:hostSlot===0,target:'fixture-target'}}));
  const environment=join(f.root,'gated-environment.mjs');
  const generic=pathToFileURL(join(f.root,'machinery','integrations/environments/index.mjs')).href;
  await writeFile(environment,`import {createEnvironment as generic} from ${JSON.stringify(generic)};
    export async function createEnvironment(options,context){return {...await generic(options,context),
      resources:async()=>${JSON.stringify(resources)},reconcileLease:async()=>true};}`);
  const input=join(f.root,'gate-settings.json');
  await writeFile(input,JSON.stringify({environment:{module:environment,options:{mode:'local'}},campaign_seconds:30,concurrency:2,
    skills:{...config.skills,background:true,tuningDisabled:true,promotionGate:{baseVersion:config.skills.activeVersion,status:'waiting'}}}));
  await f.run('config','--config',input);
  const state=new State(join(f.root,'state.sqlite'));let id;
  try {
    const a=state.admit('retained',{pid:2147483647,start:'old',boot:'old'});id=a.id;state.checkpoint(id,7.25);new RecoveryQueue(state);
    state.db.exec('CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
    state.db.prepare("INSERT OR REPLACE INTO control VALUES ('operation','\"campaign\"')").run();
    state.db.prepare('INSERT INTO resource_leases VALUES (?,?)').run(resources[0].id,id);
    await writeAttemptJob(state,f.root,id,{attempt:{...state.attempt(id)},project:state.project('retained'),
      kind:'original',wave:1,actor_seconds:120,mode:'phase1',skillVersion:config.skills.activeVersion,
      options:{continuous:true},resource:resources[0],deferFinalization:true});
  }finally{state.close();}
  const running=f.start('resume');
  try {
    await until(()=>rows(f.root,"SELECT 1 FROM events WHERE type='foreground-resource-migrated'").length>0,10000);
    const checkpoint=rows(f.root,'SELECT outcome,session,elapsed FROM attempts')[0];
    assert.equal(checkpoint.outcome,null);assert.equal(checkpoint.session,null);assert.equal(checkpoint.elapsed,7.25);
    assert.equal(rows(f.root,"SELECT 1 FROM events WHERE type='task-constructed'").length,0);
    assert.equal(rows(f.root,'SELECT resource FROM resource_leases')[0].resource,resources[1].id);
    assert.equal(f.server.requests.length,0);
  }finally{await f.run('pause');await running.output;}
  f.pass();
});
