import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {join} from 'node:path';
import {setup} from './driver_setup.mjs';
import {readConfig} from '../../cli/instance.mjs';
import {State} from '../../controller/state.mjs';
import {SkillBacklog} from '../../controller/skill-backlog.mjs';
import {enqueueLearningRequest} from '../../controller/skill-review.mjs';
import {microlearningSettled} from '../../cli/skill-microlearning.mjs';
import {releaseKnowledgeService} from '../../cli/campaign-memory.mjs';

test('reserved microlearning executes a preplanned project through native Pi and Python', {timeout:120000},async t=>{
 const f=await setup(t,'phase2',[]),config=await readConfig(f.root),state=new State(join(f.root,'state.sqlite'));t.after(()=>state.close());
 const native=await import(pathToFileURL(join(f.root,'machinery','cli','skill-microlearning.mjs')));
 const {ensureKnowledgeService:ensureNative}=await import(pathToFileURL(join(f.root,'machinery','cli','knowledge.mjs')));
 const credentials=join(f.root,'fixture.env');await writeFile(credentials,'DRIVER_FIXTURE_KEY=fixture-only\n');config.actor={...config.actor,credentials_file:credentials};
 const source=join(f.root,'preplanned','source'),trusted=join(f.root,'preplanned','trusted');await mkdir(source,{recursive:true});await mkdir(trusted);
 await writeFile(join(source,'answer.py'),'def answer(): return 0\n');const verifier=join(trusted,'verify.py');
 await writeFile(verifier,"import importlib.util\nfrom pathlib import Path\ns=importlib.util.spec_from_file_location('candidate',Path.cwd()/'answer.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nassert m.answer()==sum(6 for _ in range(7))\n");
 const backlog=new SkillBacklog(state,{target:'cpu'});enqueueLearningRequest(state,{target:'cpu',workKind:'microlearning',learningTarget:'cpu',question:'Check arithmetic oracle',requiredEvidence:'Independent Python checker'});
 const url=await ensureNative(config),before=JSON.parse(await readFile(join(f.root,'knowledge-service.json')));
 const result=await native.runMicrolearning({config,backlog,resource:{id:'local'},remainingSeconds:900},{inspect:async()=>({status:'experiment',answer:'Execute the tiny checker',refs:[]}),
  construct:async()=>[{id:'tiny',goal:'unrelated tiny arithmetic oracle',source,actor_seconds:90,verifier:{files:[verifier],command:{entrypoint:verifier,args:[]}}}]});
 assert.equal(result.status,'verified');const child=await readConfig(result.findings.instanceRoot);assert.deepEqual(child.resource_ids,['local']);
 assert.equal(await microlearningSettled(config,{directory:child.root}),true);
 const childState=new State(join(child.root,'state.sqlite'));try{
  assert.equal(childState.db.prepare("SELECT COUNT(*) n FROM attempts WHERE outcome='passed'").get().n,1);
  assert(childState.db.prepare("SELECT 1 FROM events WHERE type='postprocess-completed'").get());
 }finally{childState.close();}
 await releaseKnowledgeService(child);assert.equal(await ensureNative(config),url);
 assert.deepEqual(JSON.parse(await readFile(join(f.root,'knowledge-service.json'))).owner,before.owner);assert(f.server.requests.length>0);f.pass();
});
