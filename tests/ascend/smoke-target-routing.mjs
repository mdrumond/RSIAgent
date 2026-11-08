import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {smokeEnvironment} from '../../examples/ascend-smoke/run.mjs';
import {createEnvironment} from '../../integrations/environments/ascend.mjs';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';

for(const [target,alias,mode] of [['a3','fixture-gz-a3','cpl'],['a5','fixture-bz-a5','local']])
  test(`${target} smoke resource admission preserves product and registered ${mode} alias`,async t=>{
    const root=await mkdtemp(join(tmpdir(),'smoke-routing-'));
    t.after(()=>rm(root,{recursive:true,force:true}));
    const instructionFile=join(root,'instructions.md');await writeFile(instructionFile,'Fixture access policy');
    const environment={module:'machinery/integrations/environments/ascend.mjs',options:{
      mode,target:alias,runtime:'fixture-runtime',instructionFile,health:{product:target}}};
    const calls=[],events=[];
    const context=path=>({instancePath:join(root,path),event:async(type,data)=>events.push({type,data}),
      exclude:async(_label,operation)=>operation(),client:async args=>{
        calls.push(args);assert.deepEqual(args,['npu','list',alias]);
        return {state:'completed',profiles:[{target:alias,pool:'fixture-'+target,
          devices:[{id:0,enabled:true},{id:1,enabled:false}]}]};
      }});
    // The raw alias-only draft explains the original pre-attempt rejection.
    const raw=await createEnvironment(environment.options,context('raw'));
    assert.deepEqual(await raw.resources({target}),[]);assert.equal(calls.length,0);
    const config={target,environment:smokeEnvironment(environment,target)};
    const bound=await createEnvironment(config.environment.options,context('normalized'));
    const resources=await bound.resources({target:config.target});
    assert.equal(resources.length,1);assert.equal(resources[0].available,true);
    assert.equal(resources[0].metadata.target,alias);
    assert.equal(config.target,target);assert.equal(config.environment.options.target,alias);
    assert.equal(environment.options.profile,undefined,'Input profile must remain unchanged');
    assert.deepEqual(await bound.resources({target:alias}),resources);
    assert.deepEqual(await bound.resources({target:target==='a3'?'a5':'a3'}),[]);
    assert.equal(calls.length,2);
    const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
    state.add({id:'hello',target:config.target,goal:'fixed smoke',source:root,actor_seconds:300,
      verifier:{files:[],command:'true'}});
    const job=new RecoveryQueue(state).claim('hello',1,{},resources[0].id);
    assert.equal(job.project.spec.target,target);assert.equal(state.project('hello').status,'running');
    assert.equal(state.db.prepare('SELECT count(*) AS n FROM attempts').get().n,1);
    assert.equal(events.some(event=>event.type==='execution-handle'),false);
  });

test('CPU smoke retains its original generic execution binding',()=>{
  const environment={module:'machinery/integrations/environments/index.mjs',options:{mode:'local'}};
  assert.equal(smokeEnvironment(environment,'cpu'),environment);
});
