import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {resume,operate} from '../../cli/runtime.mjs';
import {readConfig} from '../../cli/instance.mjs';

async function fixture(t,{operation,budget=true,unbound=false}={}) {
 const root=await mkdtemp(join(tmpdir(),'portable-resume-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const config={format:1,revision:1,root,mode:'phase1',projects:[],concurrency:1,campaign_seconds:10,
  binding_required:unbound,environment:{module:'environment.mjs',options:{}}};
 await writeFile(join(root,'instance.json'),JSON.stringify(config));
 await writeFile(join(root,'environment.mjs'),'export function createEnvironment(){return {resources:async()=>[]};}');
 const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
 state.db.exec('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
 const set=(key,value)=>state.db.prepare('INSERT OR REPLACE INTO control VALUES (?,?)').run(key,JSON.stringify(value));
 if(operation!==undefined)set('operation',operation);
 const saved={cycle:'retained-authentic-cycle',seconds:10,elapsed:12.5,running:false,boot:'source',monotonic:8};
 if(budget)set('campaign-budget',saved);
 const get=key=>JSON.parse(state.db.prepare('SELECT value FROM control WHERE key=?').get(key)?.value??'null');
 return {root,config,state,set,get,saved};
}

test('portable resume recovers continuous dispatch from its retained cycle without renewing elapsed allocation',async t=>{
 const f=await fixture(t);
 f.state.add({id:'historical',goal:'retained result',actor_seconds:30});
 const a=f.state.admit('historical',{pid:2147483647,start:'0',boot:'source'});
 f.state.checkpoint(a.id,7.25,'retained-session');f.state.finish(a.id,'passed');
 f.state.event(a.id,'postprocess-completed',{});
 const before=f.state.attempt(a.id);
 const result=await resume(await readConfig(f.root));
 assert.equal(result.reason,'elapsed');assert.equal(result.waves[0].reason,'elapsed');
 assert.equal(f.get('operation'),'campaign');assert.equal(f.get('campaign-budget').cycle,f.saved.cycle);
 assert.equal(f.get('campaign-budget').seconds,10);assert.ok(f.get('campaign-budget').elapsed>=12.5);
 assert.equal(f.get('campaign-budget').running,false);assert.deepEqual(f.state.attempt(a.id),before);
 const again=await operate('resume',await readConfig(f.root));assert.equal(again.reason,'elapsed');
 assert.equal(f.get('campaign-budget').cycle,f.saved.cycle);
});

for(const options of [{budget:false},{operation:'wave'}])test('ordinary wave resume stays one wave '+JSON.stringify(options),async t=>{
 const f=await fixture(t,options),before=f.get('campaign-budget');
 const result=await resume(await readConfig(f.root));
 assert.equal(result.wave,0);assert.deepEqual(result.results,[]);assert.equal(f.get('operation'),options.operation??null);
 assert.deepEqual(f.get('campaign-budget'),before);
});

test('internal resume dispatch also recovers a missing campaign operation',async t=>{
 const f=await fixture(t),result=await operate('resume',await readConfig(f.root));
 assert.equal(result.reason,'elapsed');assert.equal(f.get('operation'),'campaign');
 assert.equal(f.get('campaign-budget').cycle,f.saved.cycle);assert.ok(f.get('campaign-budget').elapsed>=f.saved.elapsed);
});

test('an unbound portable cycle cannot claim ownership or change its operation and budget',async t=>{
 const f=await fixture(t,{unbound:true});
 await assert.rejects(resume(await readConfig(f.root)),/Bind the portable campaign/);
 assert.equal(f.get('owner'),null);assert.equal(f.get('operation'),null);assert.deepEqual(f.get('campaign-budget'),f.saved);
});
