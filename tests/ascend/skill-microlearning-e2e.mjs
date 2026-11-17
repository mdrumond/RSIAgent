import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,writeFile,cp} from 'node:fs/promises';
import {join} from 'node:path';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {processIdentity} from '../../controller/lifecycle.mjs';
import {pathToFileURL} from 'node:url';
import {readConfig} from '../../cli/instance.mjs';
import {setup} from './driver_setup.mjs';
import {generationPlan,samplePlan} from './fixtures/microlearning-native.mjs';
import {State} from '../../controller/state.mjs';
import {SkillBacklog} from '../../controller/skill-backlog.mjs';
import {enqueueLearningRequest} from '../../controller/skill-review.mjs';
import {runMicrolearning} from '../../cli/skill-microlearning.mjs';
import {qualifiedSuites} from '../../cli/skill-priority-cases.mjs';
import {evaluateHypothesis} from '../../cli/skill-hypothesis-evaluation.mjs';
import {publishSkillVersion} from '../../cli/skills.mjs';
const json=async path=>JSON.parse(await readFile(path,'utf8'));
async function fixture(t,options={}){
 const f=await setup(t,'phase2',[],{providerOptions:{generationPlan:()=>generationPlan(options),samplePlan:body=>samplePlan(body,options)}});
 const config=await readConfig(f.root);
 process.env.DRIVER_FIXTURE_KEY='fixture-only';t.after(()=>delete process.env.DRIVER_FIXTURE_KEY);
 const knowledge=await import(pathToFileURL(join(f.root,'machinery','cli','knowledge.mjs')));await knowledge.ensureKnowledgeService(config);
 config.skills.priority={enabled:true};await writeFile(join(f.root,'instance.json'),JSON.stringify(config));
 const state=new State(join(f.root,'state.sqlite'));t.after(()=>state.close());
 const backlog=new SkillBacklog(state,{target:'cpu'}),resource={id:'local-background'};
 const request={target:'cpu',learningTarget:'cpu',question:'Construct five independent arithmetic correctness cases',requiredEvidence:'Reference accepts and negatives reject',workKind:'suite'};
 const id=enqueueLearningRequest(state,request);
 return {...f,config,state,backlog,resource,id,request};
}

test('native CPU learner executes actor, qualifies five cases, then A/B verifies and publishes only improvement',async t=>{
 const f=await fixture(t);
 const result=await runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:180});
 assert.equal(result.status,'verified',JSON.stringify(result));
 assert.equal(result.qualification.status,'qualified');
 const [suite]=await qualifiedSuites({config:f.config});assert.equal(suite.tasks.length,5);
 const qualification=await json(join(f.backlog.meta('microlearning-last').qualificationDirectory,'micro','results.json'));
 assert.equal(qualification.results.length,10);assert.ok(qualification.results.some(row=>row.verification?.exitCode===1||row.exitCode===1));
 const active=f.config.skills.activeVersion,candidate='cpu-candidate',activePath=join(f.root,'skills','versions',active),candidatePath=join(f.root,'skills','versions',candidate);
 await cp(activePath,candidatePath,{recursive:true});
 const map=await json(join(candidatePath,'skill-map.json'));
 await writeFile(join(candidatePath,map.implementation),'USE_INDEPENDENT_ORACLE: implement answer() with six times seven.');
 const evaluated=await evaluateHypothesis({directory:join(f.root,'isolated-ab'),config:f.config,objective:'correctness',target:'cpu',tasks:suite.tasks,binding:suite.binding,skills:{active:activePath,candidate:candidatePath},remainingSeconds:180,resource:f.resource});
 assert.equal(evaluated.status,'complete');assert.equal(evaluated.report.results.length,20);
 assert.deepEqual(evaluated.decision.passes,{active:0,candidate:10});assert.equal(evaluated.decision.promote,true);
 assert.ok(evaluated.report.results.every(row=>Number.isInteger(row.verification.exitCode)));
 const regression=await evaluateHypothesis({directory:join(f.root,'regression-ab'),config:f.config,objective:'correctness',target:'cpu',tasks:suite.tasks,binding:suite.binding,skills:{active:candidatePath,candidate:activePath},remainingSeconds:180,resource:f.resource});
 assert.deepEqual(regression.decision.passes,{active:10,candidate:0});assert.equal(regression.decision.promote,false);
 assert.equal((await json(join(f.root,'instance.json'))).skills.activeVersion,active);
 await publishSkillVersion(f.root,{active,candidate});
 assert.equal((await json(join(f.root,'instance.json'))).skills.activeVersion,candidate);
 const requests=f.server.requests.length;
 const replay=await evaluateHypothesis({directory:join(f.root,'isolated-ab'),config:f.config,objective:'correctness',target:'cpu',tasks:suite.tasks,binding:suite.binding,skills:{active:activePath,candidate:candidatePath},remainingSeconds:100,resource:f.resource});
 assert.equal(replay.identity,evaluated.identity);assert.equal(f.server.requests.length,requests,'completed samples must not replay');
 f.pass();
});

test('native suite negative controls cannot qualify a broken discrimination test',async t=>{
 const f=await fixture(t,{badControl:true});
 const result=await runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:180});
 assert.equal(result.status,'inconclusive');assert.equal(result.qualification.status,'inconclusive');
 assert.deepEqual(await qualifiedSuites({config:f.config}),[]);
 assert.equal((await json(join(f.root,'instance.json'))).skills.activeVersion,f.config.skills.activeVersion);
 assert.equal(f.backlog.meta('microlearning-active'),null);f.pass();
});

test('cached invalid ready artifact settles once, preserves evidence and lets the next request run',async t=>{
 const f=await fixture(t),abort=new AbortController();let cached;
 const {refillProjects}=await import('../../cli/project-generation.mjs');
 await assert.rejects(runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:180,signal:abort.signal},{construct:async input=>{
  const specs=await refillProjects(input);cached=specs[0];assert.ok(cached);abort.abort(Error('pause before import'));return specs;
 }}),/pause before import/);
 const retained=f.backlog.meta('microlearning-active'),childState=new State(join(retained.directory,'state.sqlite'));
 t.after(()=>childState.close());
 assert.equal(childState.db.prepare('SELECT count(*) AS n FROM projects').get().n,0);
 const ready=JSON.parse(childState.db.prepare('SELECT data FROM generation_drafts WHERE id=?').get(cached.generation.draft).data);
 const sessions=childState.db.prepare("SELECT data FROM generation_events WHERE draft=? AND type='curriculum-session'").all(cached.generation.draft);
 assert.equal(sessions.length,1);assert.ok(await readFile(JSON.parse(sessions[0].data).session,'utf8'));
 const manifest=cached.verifier.files.find(file=>file.endsWith('/skill-evaluation.json'));
 // Same invalid wire shape as the archived ready draft: cases is not tasks.
 const bytes=await readFile(new URL('./fixtures/captured-malformed-skill-evaluation.json',import.meta.url),'utf8');
 await writeFile(manifest,bytes);
 const requests=f.server.requests.length;
 const {runBackgroundSkills}=await import('../../cli/skill-background.mjs');
 // A retained live owner is an actual local Bash process, not a canned
 // settled=true/false response. Scheduler polls may not release its resource.
 const process=spawn('/bin/bash',['-c','exec sleep 30'],{stdio:'ignore'});await once(process,'spawn');
 const owner=await processIdentity(process.pid);t.after(()=>{try{process.kill('SIGKILL');}catch{}});
 childState.db.prepare('INSERT OR REPLACE INTO control(key,value) VALUES (?,?)').run('owner',JSON.stringify(owner));
 for(let i=0;i<3;i++){
  const blocked=await runBackgroundSkills(f.config,{resource:f.resource,remainingSeconds:180});
  assert.equal(blocked.status,'blocked');
  assert.equal(f.backlog.meta('microlearning-active').id,f.id);
  assert.equal(f.backlog.meta('microlearning-active').elapsed,retained.elapsed);
 }
 assert.equal(f.server.requests.length,requests);
 const stopped=once(process,'close');process.kill('SIGTERM');await stopped;
 const invalid=await runBackgroundSkills(f.config,{resource:f.resource,remainingSeconds:180});
 assert.equal(invalid.status,'inconclusive');assert.equal(invalid.reason,'INVALID_SKILL_EVALUATION');
 assert.equal(f.server.requests.length,requests,'invalid imported contract must not invoke an actor');
 assert.equal(f.backlog.meta('microlearning-active'),null);
 assert.equal(await readFile(manifest,'utf8'),bytes,'malformed evidence remains immutable');
 const draft=JSON.parse(childState.db.prepare('SELECT data FROM generation_drafts WHERE id=?').get(cached.generation.draft).data);
 assert.equal(draft.status,'invalid');assert.deepEqual(childState.db.prepare("SELECT data FROM generation_events WHERE draft=? AND type='curriculum-session'").all(cached.generation.draft),sessions);assert.equal(draft.elapsed,ready.elapsed);
 assert.equal(childState.db.prepare('SELECT count(*) AS n FROM attempts').get().n,0);
 const feedback=JSON.parse(f.state.db.prepare('SELECT data FROM skill_learning_requests WHERE id=?').get(f.id).data);
 assert.equal(feedback.findings.validationFailure.code,'INVALID_SKILL_EVALUATION');
 // Repeated production learner polling has no request to reclaim or replay.
 for(let i=0;i<3;i++)assert.equal((await runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:180})).status,'idle');
 assert.equal(f.server.requests.length,requests);
 const next=enqueueLearningRequest(f.state,{...f.request,question:'A different bounded arithmetic question'});
 assert.notEqual(next,f.id);
 const succeeded=await runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:180});
 assert.equal(succeeded.status,'verified');f.pass();
});
