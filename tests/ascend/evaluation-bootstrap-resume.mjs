import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,readFile,chmod,mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {evaluateRound} from '../../controller/evaluation-round.mjs';
import {join} from 'node:path';
import {DatabaseSync} from 'node:sqlite';
import {setup,rows,until} from './driver_setup.mjs';
import {State} from '../../controller/state.mjs';
import {bootstrap as instance,readConfig} from '../../cli/instance.mjs';
import {bootstrap,resume,settled} from '../../cli/evaluation.mjs';
import {bootstrapSettled,managedBootstrap} from '../../cli/evaluation-bootstrap.mjs';
import {processIdentity} from '../../controller/lifecycle.mjs';

const quote=text=>"'"+text.replaceAll("'","'\\''")+"'";
test('cancelled setup resumes the same frozen sample and runs exactly one canonical actor',
  {timeout:180000},async t=>{
    const f=await setup(t,'recover',['unrelated']);
    const config=await readConfig(f.root),round=join(f.root,'round');await mkdir(round);
    const marker=join(round,'setup-started'),release=join(round,'release'),python=join(round,'python');
    await writeFile(python,`#!/bin/bash\nif [[ "$*" == '-m ascend_tools.restic_install --install' ]]; then\n touch ${quote(marker)}\n while [[ ! -f ${quote(release)} ]]; do sleep 0.05; done\nfi\nexec ${quote(config.python)} "$@"\n`);
    await chmod(python,0o700);
    const credentials=join(round,'fixture.env');await writeFile(credentials,'DRIVER_FIXTURE_KEY=fixture-only\n');
    const request={sourceInstanceRoot:f.root,instanceRoot:join(round,'sample'),projectId:'unrelated',projectIds:['unrelated'],
      skillVersion:'baseline',remainingSeconds:120,config:{...config,python,actor:{...config.actor,credentials_file:credentials}}};
    const abort=new AbortController();t.after(()=>abort.abort());
    const interrupted=assert.rejects(bootstrap({...request,signal:abort.signal}),{name:'AbortError'});
    await until(async()=>{try{await readFile(marker);return true;}catch{return false;}},30000);
    assert.equal(rows(request.instanceRoot,"SELECT count(*) AS n FROM sqlite_master WHERE name='later'",'knowledge.sqlite')[0].n,0);
    const before=JSON.parse(await readFile(request.instanceRoot+'.bootstrap.json','utf8'));
    assert.equal(await bootstrapSettled(request.instanceRoot),false);
    abort.abort();await interrupted;
    assert.equal(await settled(request),true);
    assert.equal((await readConfig(join(round,'inputs'))).projects.length,1);
    await assert.rejects(bootstrap({...request,skillVersion:'baseline',config:{...request.config,actor_seconds:119}}),/binding changed/);
    const db=new DatabaseSync(join(f.root,'knowledge.sqlite'));db.exec('CREATE TABLE later(value)');db.close();
    await writeFile(release,'go');
    const result=await resume(request);
    assert.equal(result.correct,true);assert.equal(result.attempts.length,1);
    assert.equal(rows(request.instanceRoot,'SELECT count(*) AS n FROM attempts')[0].n,1);
    assert.equal(rows(request.instanceRoot,"SELECT count(*) AS n FROM sqlite_master WHERE name='later'",'knowledge.sqlite')[0].n,0);
    const after=JSON.parse(await readFile(request.instanceRoot+'.bootstrap.json','utf8'));
    assert.equal(after.signature,before.signature);assert.equal(after.status,'ready');
    assert.equal(after.handles.length,3); // cancelled restic, resumed restic, dependency installation
    assert.equal(await settled(request),true);
    f.pass();
  });

test('managed bootstrap does not replay completed setup and fences unknown handles or live owners',async t=>{
  const f=await setup(t,'recover',['unrelated']),root=join(f.root,'managed'),counter=join(f.root,'counter');
  const options={binding:{source:'frozen'},remainingSeconds:10};
  const cancelled=new AbortController();cancelled.abort();let dispatched=false;
  await assert.rejects(managedBootstrap(root,{...options,signal:cancelled.signal},async()=>{dispatched=true;}),{name:'AbortError'});
  assert.equal(dispatched,false);assert.equal(await settled({instanceRoot:root}),true);
  const initialize=async initialization=>{
    await initialization.setup('one',process.execPath,['-e',`require('fs').appendFileSync(${JSON.stringify(counter)},'x')`],f.root,{});
    throw new Error('pause between steps');
  };
  await assert.rejects(managedBootstrap(root,options,initialize),/pause between/);
  await assert.rejects(managedBootstrap(root,options,initialize),/pause between/);
  assert.equal(await readFile(counter,'utf8'),'x');
  const path=root+'.bootstrap.json',receipt=JSON.parse(await readFile(path,'utf8'));
  receipt.status='preparing';receipt.owner=await processIdentity();await writeFile(path,JSON.stringify(receipt));
  await assert.rejects(managedBootstrap(root,options,initialize),/unresolved/);
  receipt.owner={pid:2147483647,start:'0',boot:'old'};receipt.phase='storage';await writeFile(path,JSON.stringify(receipt));
  assert.equal(await bootstrapSettled(root),false);
  await assert.rejects(managedBootstrap(root,options,initialize),/unresolved/);
  receipt.status='stopped';receipt.handles.push('local:exec-unknown');await writeFile(path,JSON.stringify(receipt));
  assert.equal(await bootstrapSettled(root),false);
  await assert.rejects(managedBootstrap(root,options,initialize),/unresolved/);
  const config=await readConfig(f.root),actor=join(f.root,'actor');await mkdir(actor);
  const state=new State(join(actor,'state.sqlite'));state.add({id:'p',goal:'retain actor',actor_seconds:1});
  state.admit('p',await processIdentity());state.close();
  const parameters={baseline:join(f.root,'knowledge.sqlite'),skills:join(f.root,'skills','versions','baseline'),
    environment:config.environment,initialization:{resume:true,setup:async()=>{}},install:false};
  await assert.rejects(instance(actor,parameters),/existing actor work/);
  assert.equal(rows(actor,'SELECT count(*) AS n FROM attempts')[0].n,1);
  await assert.rejects(instance(f.root,parameters),/ready instance/);
  const legacy=join(f.root,'legacy');await mkdir(legacy);
  await assert.rejects(managedBootstrap(legacy,options,initialize),/Unreceipted/);
  f.pass();
});


test('zero-dispatch pause clears the running checkpoint and resumes the same eight-sample journal',async t=>{
  const root=await mkdtemp(join(tmpdir(),'evaluation-zero-dispatch-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const pending={evaluationRoot:join(root,'round'),sourceInstanceRoot:root,projectIds:['fa','gdn'],runs:2,
    active:'baseline',candidate:'candidate',evaluationSeconds:10};
  const abort=new AbortController();let identity;
  await assert.rejects(evaluateRound({pending,config:{},signal:abort.signal,evaluate:async request=>{
    identity=request.instanceRoot;abort.abort();
    try{await bootstrap(request);}catch(error){error.evaluationStopped=await settled(request);throw error;}
  }}),{name:'AbortError'});
  const before=JSON.parse(await readFile(join(pending.evaluationRoot,'round.json'),'utf8'));
  assert.equal(before.runningAt,undefined);assert.equal(before.samples.length,8);
  await assert.rejects(readFile(identity+'.bootstrap.json'),{code:'ENOENT'});
  await assert.rejects(readConfig(identity),{code:'ENOENT'});
  const ids=[];
  const after=await evaluateRound({pending,config:{},evaluate:async request=>{
    ids.push(request.instanceRoot);return {correct:true,score:1};
  }});
  assert.equal(ids[0],identity);assert.equal(new Set(ids).size,8);assert.equal(after.status,'completed');
  assert.deepEqual(after.samples.map(sample=>sample.instanceRoot),before.samples.map(sample=>sample.instanceRoot));
  assert.equal(after.samples.filter(sample=>sample.result).length,8);
});


test('ready bootstrap receipts cannot settle live controllers, workers or unclosed execution',async t=>{
  const round=await mkdtemp(join(tmpdir(),'evaluation-ready-receipt-'));t.after(()=>rm(round,{recursive:true,force:true}));
  const root=join(round,'sample'),inputs=join(round,'inputs'),live=await processIdentity();
  const stopped={pid:2147483647,start:'0',boot:'old-boot'};
  for(const path of [inputs,root])await managedBootstrap(path,{binding:{source:'frozen'}},async()=>{await mkdir(path);});
  await writeFile(join(root,'instance.json'),JSON.stringify({format:1,mode:'phase2'}));
  const state=new State(join(root,'state.sqlite'));
  try {
    state.add({id:'p',goal:'positive actor closure',actor_seconds:5});
    const attempt=state.admit('p',stopped);state.event(attempt.id,'accepted',{});state.finish(attempt.id,'passed');
    state.db.exec("CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT NOT NULL)");
    state.db.prepare("INSERT INTO control VALUES ('owner',?)").run(JSON.stringify(live));
    assert.equal(await bootstrapSettled(root),true);assert.equal(await bootstrapSettled(inputs),true);
    assert.equal(await settled({instanceRoot:root}),false);
    state.db.prepare("UPDATE control SET value='null' WHERE key='owner'").run();
    state.db.prepare('UPDATE attempts SET owner=? WHERE id=?').run(JSON.stringify(live),attempt.id);
    assert.equal(await settled({instanceRoot:root}),false);
    state.db.prepare('UPDATE attempts SET owner=? WHERE id=?').run(JSON.stringify(stopped),attempt.id);
    assert.equal(await settled({instanceRoot:root}),true);
    state.event(attempt.id,'native-bash-started',{operation:'still-unresolved'});
    assert.equal(await settled({instanceRoot:root}),false);
    state.event(attempt.id,'native-bash-completed',{operation:'still-unresolved',actor_elapsed:0});
    assert.equal(await settled({instanceRoot:root}),true);
  }finally{state.close();}
});

test('a minted local handle without a process receipt remains fenced despite stopped host owners',async t=>{
  const round=await mkdtemp(join(tmpdir(),'evaluation-prelaunch-fence-'));t.after(()=>rm(round,{recursive:true,force:true}));
  const root=join(round,'sample'),inputs=join(round,'inputs');
  for(const path of [inputs,root])await managedBootstrap(path,{binding:{source:'frozen'}},async()=>{await mkdir(path);});
  await writeFile(join(root,'instance.json'),JSON.stringify({format:1,mode:'phase2',environment:{
    module:new URL('../../integrations/environments/index.mjs',import.meta.url).pathname,options:{mode:'local'}}}));
  const state=new State(join(root,'state.sqlite'));
  try {
    state.add({id:'p',goal:'retain prelaunch uncertainty',actor_seconds:5});
    const attempt=state.admit('p',{pid:2147483647,start:'old',boot:'old'});
    state.event(attempt.id,'accepted',{});state.finish(attempt.id,'passed');
    const path=join(root,'attempts',attempt.id),execution=join(path,'.executions','exec-prelaunch');
    await mkdir(execution,{recursive:true});await writeFile(join(path,'job.json'),JSON.stringify({resource:{id:'local'}}));
    const identity=await processIdentity();
    await writeFile(join(execution,'execution.json'),JSON.stringify({boot:identity.boot,cwd:root}));
    state.event(attempt.id,'execution-handle',{handle:'local:exec-prelaunch'});
    assert.equal(await bootstrapSettled(root),true);assert.equal(await bootstrapSettled(inputs),true);
    assert.equal(await settled({instanceRoot:root}),false);
    await assert.rejects(resume({instanceRoot:root}),/execution remains unresolved/);
    await assert.rejects(readFile(join(execution,'identity')),{code:'ENOENT'});
    assert.equal(state.db.prepare('SELECT count(*) AS n FROM attempts').get().n,1);
  }finally{state.close();}
});
