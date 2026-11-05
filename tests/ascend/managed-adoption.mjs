import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {operate,dispatchManaged} from '../../cli/runtime.mjs';
import {waitFor} from './resilience/campaign-fixture.mjs';
const MiB=1024**2,dead={pid:2147483647,start:'1',boot:'old'};
async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'managed-adoption-'));
  const config={format:1,root,mode:'phase1',host_memory:{max_bytes:512*MiB},skills:{activeVersion:'v'},
    environment:{module:'environment.mjs'},projects:['old','fresh'].map(id=>({id,goal:id,actor_seconds:5}))};
  await writeFile(join(root,'instance.json'),JSON.stringify(config));
  await writeFile(join(root,'environment.mjs'),`export function createEnvironment(){return {
    resources:async()=>[{id:'old-slot',available:true},{id:'fresh-slot',available:true}]};}`);
  const state=new State(join(root,'state.sqlite')),queue=new RecoveryQueue(state);
  for(const project of config.projects)state.add(project);
  const admitted=queue.claim('old',1,dead,'old-slot'),attempt=admitted.attempt;
  state.db.prepare('UPDATE attempts SET session=?,elapsed=2 WHERE id=?').run('original-session',attempt.id);
  await mkdir(join(root,'attempts',attempt.id),{recursive:true});
  await writeFile(join(root,'attempts',attempt.id,'job.json'),JSON.stringify({...admitted,options:{},mode:'phase1',resource:{id:'old-slot'}}));
  state.db.exec('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
  const set=(key,value)=>state.db.prepare('INSERT OR REPLACE INTO control VALUES (?,?)').run(key,JSON.stringify(value));
  const get=key=>JSON.parse(state.db.prepare('SELECT value FROM control WHERE key=?').get(key)?.value??'null');
  const manager={verifyProcessMembership:async()=>{},ensureKnowledgeService:async()=>'',
    measureManagedMemory:async()=>({current_bytes:64*MiB,peak_bytes:64*MiB}),
    spawnManagedWorker:async prepare=>{
      const descriptor=await prepare(),id=descriptor.argv.at(-1);state.finish(id,'passed');
      return {identity:{unit:'rsi-worker-fresh.service',pid:19},wait:async()=>({closed:true,code:0,metrics:{peak_bytes:64*MiB}})};
    }};
  t.after(async()=>{state.close();await rm(root,{recursive:true,force:true});});
  return {root,config,state,attempt,set,get,manager};
}
for(const [concurrency,reserved,freshBefore] of [[2,64,true],[2,256,false],[1,64,false]])
  test(`retained worker adoption honors concurrency=${concurrency} and ${reserved}MiB reservation`,async t=>{
    const f=await fixture(t);f.set('worker-peak',64*MiB);
    f.state.event(f.attempt.id,'managed-worker-started',{unit:'rsi-worker-retained.service',invocation:'old',memory_bytes:reserved*MiB});
    let release,finished=false;
    const done=new Promise(resolve=>release=resolve);
    f.manager.observeManagedWorker=async()=>({wait:async()=>{
      await done;finished=true;f.state.finish(f.attempt.id,'passed');
      return {closed:true,metrics:{peak_bytes:64*MiB}};
    }});
    let spawnedWhileRetained;
    const spawn=f.manager.spawnManagedWorker;
    f.manager.spawnManagedWorker=async prepare=>{spawnedWhileRetained=!finished;return spawn(prepare);};
    const running=operate('wave',f.config,{managed:true,concurrency},f.manager);
    try {
      if(freshBefore)await waitFor(()=>spawnedWhileRetained===true,undefined,3);
      else {
        await waitFor(()=>['memory-capacity','actor-capacity'].includes(f.get('wait')?.reason),undefined,3);
        assert.equal(spawnedWhileRetained,undefined);
      }
    } finally {release();}
    await running;
    assert.equal(spawnedWhileRetained,freshBefore);
    assert.equal(f.state.db.prepare('SELECT COUNT(*) AS n FROM attempts').get().n,2);
    assert.equal(f.state.queryEvents(f.attempt.id,{types:['worker-memory']})[0].data.reconciled,true);
  });

test('explicit fresh start clears stop after preflight and never resumes interrupted original',async t=>{
  const f=await fixture(t);f.set('stop',true);
  let checked=false;
  f.manager.configureHostMemory=async(_config,{verifyInstanceStopped})=>{assert(await verifyInstanceStopped());checked=true;};
  f.manager.startManagedController=async()=>{assert(checked);assert.equal(f.get('stop'),false);return {identity:{unit:'rsi-controller-test.service'}};};
  await dispatchManaged('wave',f.config,{detach:true},f.manager);
  assert.deepEqual(f.get('managed-excluded'),[f.attempt.id]);
  await operate('wave',f.config,{managed:true},f.manager);
  assert.equal(f.state.attempt(f.attempt.id).session,'original-session');
  assert.equal(f.state.attempt(f.attempt.id).elapsed,2);
  assert.equal(f.state.attempt(f.attempt.id).outcome,null);
  assert.equal(f.state.project('fresh').status,'passed');
  // Internal service restart preserves a later user stop without touching an actor.
  f.set('stop',true);
  assert.equal((await operate('wave',f.config,{managed:true},f.manager)).reason,'stopped');
  assert.equal(f.get('stop'),true);
});
