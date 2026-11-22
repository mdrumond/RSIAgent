import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {DatabaseSync} from 'node:sqlite';
import {State} from '../../controller/state.mjs';
import {runReview} from '../../cli/skill-review.mjs';
import {wave} from '../../cli/evaluation.mjs';

async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'skill-context-entry-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const source=join(root,'source'),version=join(source,'skills/versions/baseline');await mkdir(version,{recursive:true});
 const config={format:1,target:'a3',mode:'phase2',actor:{provider:'fixture',model:'fixture'},skills:{activeVersion:'baseline'}};
 await writeFile(join(source,'instance.json'),JSON.stringify(config));
 await writeFile(join(version,'skill-map.json'),JSON.stringify({implementation:'implementation.md'}));
 await writeFile(join(version,'guidance.json'),JSON.stringify([{id:'baseline',scope:'implementation',text:'Retained guidance',critical:true,negative:false}]));
 await writeFile(join(version,'implementation.md'),'Retained guidance');new DatabaseSync(join(source,'knowledge.sqlite')).close();
 const state=new State(join(source,'state.sqlite'));state.add({id:'p',goal:'fixture',phase:'phase1',actor_seconds:30});
 const attempt=state.admit('p',{pid:1,start:'fixture',boot:'fixture'});state.event(attempt.id,'failure-recorded',{message:'Wrong result'});state.db.prepare("UPDATE attempts SET outcome='failed' WHERE id=?").run(attempt.id);state.close();
 return {root,source,options:{sources:[{id:'a3',root:source}],output:join(root,'review'),seconds:30,stage:'classify'}};
}
test('public isolated review constructs capped writer profiles from an uncapped source campaign',async t=>{
 const f=await fixture(t);let observed;
 const before=await readFile(join(f.source,'instance.json'),'utf8');
 await runReview({...f.options,stage:'inventory'});
 await runReview(f.options,{classify:async({group,config,validate})=>{observed=config;return validate({cause:'actor',coverage:'unknown',guidanceIds:[],evidenceRefs:[group.occurrences[0].reference],reason:'Observed incorrect implementation'});}});
 assert.equal(observed.actor.max_context_tokens,200000);assert.equal(observed.skill_writer_actor.max_context_tokens,200000);
 assert.equal(await readFile(join(f.source,'instance.json'),'utf8'),before);
});
test('legacy uncapped frozen full-project comparisons are stopped before actor dispatch',async t=>{
 const f=await fixture(t);
 await assert.rejects(wave({instanceRoot:f.source,config:{mode:'phase2'}}),/Frozen evaluation context policy differs/);
 await assert.rejects(readFile(join(f.source,'evaluation-wave.log')),{code:'ENOENT'});
});
