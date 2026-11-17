import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {SkillBacklog} from '../../controller/skill-backlog.mjs';
import {enqueueLearningRequest,claimLearningRequest} from '../../controller/skill-review.mjs';
import {microlearningConfig,runMicrolearning,pendingMicrolearning,validateArchivalInspection} from '../../cli/skill-microlearning.mjs';
import {prioritySkillStatus} from '../../cli/skill-priority.mjs';
import {qualifiedSuites} from '../../cli/skill-priority-cases.mjs';
import {releaseKnowledgeService} from '../../cli/campaign-memory.mjs';
import {processIdentity} from '../../controller/lifecycle.mjs';
import {SkillEvaluationContractError} from '../../cli/skill-evaluation-contract.mjs';
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'microlearning-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());const backlog=new SkillBacklog(state,{target:'a3'});
 const config={root,target:'a3',mode:'phase1',concurrency:5,actor_seconds_by_phase:{phase1:3600},phase_targets:{phase1:128},
  environment:{module:'/existing/environment.mjs',options:{runtime:'pinned-runtime'}},skills:{activeVersion:'baseline',background:true,priority:{enabled:true}}};
 await mkdir(join(root,'skills','versions','baseline'),{recursive:true});await writeFile(join(root,'skills','versions','baseline','rules.md'),'pinned');
 await writeFile(join(root,'knowledge-service.json'),JSON.stringify({url:'http://borrowed',owner:await processIdentity()}));
 const request={target:'a3',question:'Construct a small controlled tail experiment',requiredEvidence:'Independent reference and negatives',workKind:'suite'};
 const id=enqueueLearningRequest(state,request),resource={id:'existing-background'};
 return {root,state,backlog,config,request,id,resource};
}
test('minimal child uses exactly the admitted resource and cannot recursively tune',async t=>{
 const f=await fixture(t),child=microlearningConfig(f.config,{directory:join(f.root,'child'),request:f.request,resource:f.resource,seconds:900});
 assert.deepEqual(child.resource_ids,[f.resource.id]);assert.equal(child.concurrency,1);assert.equal(child.skills.background,false);assert.equal(child.skills.priority.enabled,false);
 assert.equal(child.phase_targets.phase1,1);assert.equal(child.actor_seconds_by_phase,undefined);assert.equal(child.campaign_seconds,900);assert.equal(child.environment.options.runtime,'pinned-runtime');
 assert.equal(claimLearningRequest(f.state,f.config,{draft:'foreground',target:'a3'}),null);assert.equal(pendingMicrolearning(f.backlog)[0].id,f.id);
});
test('one bounded child borrows parent KDB and verifies postprocessed provenance',async t=>{
 const f=await fixture(t);let constructed=0,operated=0;
 const ops={qualify:async()=>({status:'qualified'}),inspect:async()=>({status:'experiment',answer:'A suite must be constructed',refs:[]}),ensureKnowledgeService:async()=> 'http://borrowed',createKdbBridge:()=>({}),construct:async({config,concurrency})=>{
  constructed++;assert.equal((await prioritySkillStatus(f.config)).activeWork.stage,'construction');assert.equal(concurrency,1);assert.deepEqual(config.resource_ids,[f.resource.id]);return [{id:'question',goal:'Tiny question',actor_seconds:20}];},
  importProject:async(root,spec)=>spec,operate:async(command,config)=>{operated++;const s=new State(join(config.root,'state.sqlite'));
   s.db.prepare("INSERT INTO attempts(id,project,outcome,created) VALUES('attempt','question','passed','now')").run();s.event('attempt','postprocess-completed',{});s.close();},stopped:async()=>true};
 const result=await runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:1800},ops);
 assert.equal(result.status,'verified');assert.equal(constructed,1);assert.equal(operated,1);assert.equal(result.findings.projectId,'question');
 const child=JSON.parse(await readFile(join(result.findings.instanceRoot,'instance.json')));assert.equal(child.skills.tuningDisabled,true);
 await releaseKnowledgeService(child);process.kill(process.pid,0);
 const receipt=JSON.parse(await readFile(join(child.root,'knowledge-service.json')));assert.equal(receipt.ownership,'borrowed');
 const active=await readFile(join(child.root,'skills','versions','baseline','rules.md'),'utf8');assert.equal(active,'pinned');
 assert.equal(f.backlog.meta('microlearning-active'),null);assert.equal(f.backlog.meta('microlearning-last').seconds,900);
});
test('crash retains construction elapsed and never claims an additional request or device',async t=>{
 const f=await fixture(t);let clock=0;
 await assert.rejects(runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:1000},{
  inspect:async()=>({status:'experiment',answer:'A suite must be constructed',refs:[]}),now:()=>clock,ensureKnowledgeService:async()=> 'http://borrowed',createKdbBridge:()=>({}),construct:async()=>{clock=400000;throw Error('transport interrupted');}}),/transport interrupted/);
 const work=f.backlog.meta('microlearning-active');assert.equal(work.elapsed,400);assert.equal(work.seconds,900);
 let dispatched=false;
 const result=await runMicrolearning({config:f.config,backlog:f.backlog,resource:{id:'other'},remainingSeconds:1000},{operate:async()=>{dispatched=true;}});
 assert.equal(result.status,'blocked');assert.equal(dispatched,false);
 const blocked=await runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:1000},{stopped:async()=>false});
 assert.equal(blocked.status,'blocked');assert.equal(f.backlog.meta('microlearning-active').elapsed,400);
});
test('explicit infrastructure continuation preserves old history and original allocation',async t=>{
 const f=await fixture(t);f.state.event(null,'attempt-error',{message:'tail'});f.backlog.ingest();const group=f.backlog.failures()[0];
 const item=f.backlog.upsert({key:'hypothesis',objective:'correctness',skill:'implementation',groupIds:[group.id],base:'baseline',proposal:{text:'mask tails'}});
 const claimed=f.backlog.claim({base:'baseline'});f.backlog.checkpoint(claimed.run.id,{elapsed:400,checkpoint:{}});
 f.backlog.finish(claimed.run.id,{status:'inconclusive',result:{reason:'qa-elapsed'}});
 const resumed=f.backlog.continue(claimed.run.id,{procedure:'Installed fixed QA executor',base:'baseline'});
 assert.equal(resumed.elapsed,400);assert.equal(resumed.seconds,1800);assert.equal(f.backlog.run(claimed.run.id).status,'inconclusive');assert.equal(resumed.checkpoint.continuedFrom,claimed.run.id);
 f.backlog.finish(resumed.id,{status:'rejected',result:{reason:'insufficient-correctness-improvement',report:{results:[{correct:false}]}}});
 assert.throws(()=>f.backlog.continue(resumed.id,{procedure:'Try again',base:'baseline'}),/Only audited/);assert.equal(f.backlog.get(item.id).status,'rejected');
});
test('read-only archival findings reenter analysis without claiming device verification',async t=>{
 const f=await fixture(t);f.state.event(null,'attempt-error',{message:'A recorded compiler diagnostic'});f.backlog.ingest();const group=f.backlog.failures()[0];
 const request={...f.request,workKind:'microlearning',groupIds:[group.id]};f.state.db.prepare('UPDATE skill_learning_requests SET data=? WHERE id=?').run(JSON.stringify(request),f.id);
 let constructed=false;
 const result=await runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:1000},{ensureKnowledgeService:async()=> 'http://borrowed',
  inspect:async args=>args.validate({status:'resolved',answer:'The recorded diagnostic establishes a static API constraint',refs:[group.occurrences[0].reference]}),construct:async()=>{constructed=true;}});
 assert.equal(result.status,'archival-resolved');assert.equal(constructed,false);assert.equal(f.backlog.failure(group.id).dirty,true);
 assert.equal(f.state.db.prepare('SELECT status FROM skill_learning_requests WHERE id=?').get(f.id).status,'inconclusive');
 assert.equal(f.backlog.failure(group.id).archivalInspection.answer,result.inspection.answer);
});
test('expired retained microlearning settles without creating another project or service',async t=>{
 const f=await fixture(t);const directory=join(f.root,'child');await mkdir(directory);
 f.backlog.meta('microlearning-active',{id:f.id,request:f.request,directory,resource:f.resource,elapsed:901,seconds:900,stage:'construction'});
 let service=false;
 const result=await runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:1000},{ensureKnowledgeService:async()=>{service=true;}});
 assert.equal(result.reason,'microlearning-elapsed');assert.equal(service,false);assert.equal(f.backlog.meta('microlearning-active'),null);
});

for(const oracle of ['correct','always-pass'])test('suite export must pass actual Python reference and negative controls: '+oracle,async t=>{
 const f=await fixture(t),trusted=join(f.root,'trusted'),editable=join(f.root,'editable');await mkdir(trusted);await mkdir(editable);
 await writeFile(join(trusted,'checker.py'),oracle==='always-pass'?'raise SystemExit(0)\n':"import runpy,sys\nfrom pathlib import Path\nraise SystemExit(0 if runpy.run_path(str(Path(sys.argv[1])/'kernel.py')).get('answer')==42 else 1)\n");
 await writeFile(join(trusted,'good.py'),'answer=42\n');await writeFile(join(trusted,'bad.py'),'answer=0\n');
 const tasks=Array.from({length:5},(_,n)=>({id:'case-'+n,target:'cpu',kind:'hypothesis-case',bucket:'correctness',prompt:'Repair boundary answer',files:{'kernel.py':'bad.py'},checker:'checker.py',controls:{reference:{files:{'kernel.py':'good.py'}},negatives:[{files:{'kernel.py':'bad.py'}}]}}));
 const manifest=join(trusted,'skill-evaluation.json');await writeFile(manifest,JSON.stringify({target:'cpu',objective:'correctness',tasks}));
 let constructed=0;
 const result=await runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:1000},{ensureKnowledgeService:async()=> 'http://borrowed',createKdbBridge:()=>({}),inspect:async()=>{throw Error('Suite work must not invoke factual inspection');},
  construct:async({config,concurrency})=>{constructed++;assert.equal(concurrency,1);assert.deepEqual(config.resource_ids,[f.resource.id]);return [{id:'suite',goal:'Build suite',source:editable,actor_seconds:20,verifier:{files:[manifest,join(trusted,'checker.py'),join(trusted,'good.py'),join(trusted,'bad.py')]}}];},importProject:async(root,spec)=>spec,
  operate:async(command,config)=>{const child=new State(join(config.root,'state.sqlite'));child.db.prepare("INSERT INTO attempts(id,project,outcome,created) VALUES('attempt','suite','passed','now')").run();child.event('attempt','postprocess-completed',{});child.close();},stopped:async()=>true});
 assert.equal(result.status,oracle==='correct'?'verified':'inconclusive');assert.equal(constructed,1);
 const episode=f.backlog.meta('microlearning-last');assert.equal(episode.seconds,900);assert.deepEqual(episode.resource,f.resource);assert.equal(episode.routing.modelReview,false);assert.equal(episode.inspection,undefined);
 const qualification=JSON.parse(await readFile(join(episode.directory,'qualification','micro','results.json')));assert.equal(qualification.results.length,oracle==='correct'?10:2);
 const suites=await qualifiedSuites({config:f.config});assert.equal(suites.length,oracle==='correct'?1:0);
 if(suites.length){assert.equal(suites[0].tasks.length,5);assert.equal(suites[0].target,'cpu');}
});
test('passed child without protected suite export cannot satisfy suite learning',async t=>{
 const f=await fixture(t);
 const result=await runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:1000},{ensureKnowledgeService:async()=> 'http://borrowed',createKdbBridge:()=>({}),inspect:async()=>({status:'experiment',answer:'Need suite',refs:[]}),
 construct:async()=>[{id:'question',goal:'Only an answer',actor_seconds:20}],importProject:async(root,spec)=>spec,
 operate:async(command,config)=>{const child=new State(join(config.root,'state.sqlite'));child.db.prepare("INSERT INTO attempts(id,project,outcome,created) VALUES('attempt','question','passed','now')").run();child.event('attempt','postprocess-completed',{});child.close();},stopped:async()=>true});
 assert.equal(result.status,'inconclusive');assert.match(result.qualification.reason,/no trusted skill-evaluation.json/i);
});

test('archival inspection accepts only actual nested KDB entries, never invented retrieval IDs',()=>{
 const request={workKind:'microlearning'},refs=new Set(),retrievals=[{arguments:{collection:'lessons'},entries:[{chunk_id:'observed-chunk',text:'Actual fact'}]}];
 const value={status:'resolved',answer:'A retrieved archived fact',refs:['lessons:observed-chunk']};
 assert.equal(validateArchivalInspection(value,{request,refs,retrievals}),value);
 assert.throws(()=>validateArchivalInspection({...value,refs:['lessons:invented']},{request,refs,retrievals}),/Cited archival/);
 assert.throws(()=>validateArchivalInspection(value,{request:{workKind:'suite'},refs,retrievals}),/Cited archival/);
});

test('retained allowance includes closure proof waits and keeps its live owner until proof completes',async t=>{
 const f=await fixture(t);let clock=0,failed=false;
 await assert.rejects(runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:1000},{now:()=>clock,ensureKnowledgeService:async()=> 'http://borrowed',createKdbBridge:()=>({}),inspect:async()=>({status:'experiment',answer:'Need controls',refs:[]}),
  construct:async()=>[{id:'question',goal:'Small question',actor_seconds:20}],importProject:async(root,spec)=>spec,
  operate:async(command,config)=>{const child=new State(join(config.root,'state.sqlite'));child.db.prepare("INSERT INTO attempts(id,project,outcome,created) VALUES('attempt','question','passed','now')").run();child.event('attempt','postprocess-completed',{});child.close();},
  qualify:async()=>{clock=100000;failed=true;throw Error('qualification interrupted');},
  stopped:async()=>{if(failed){assert(f.backlog.meta('microlearning-active').owner);clock+=20000;}return true;},
  settled:async()=>{assert(f.backlog.meta('microlearning-active').owner);clock+=30000;return true;}}),/qualification interrupted/);
 const work=f.backlog.meta('microlearning-active');assert.equal(work.elapsed,150);assert.equal(work.seconds,900);assert.equal(work.owner,undefined);assert.equal(work.runningAt,undefined);
 const resumed=await runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:1000},{stopped:async()=>false,settled:async()=>true});
 assert.equal(resumed.status,'blocked');assert.equal(f.backlog.meta('microlearning-active').elapsed,150);assert.equal(f.backlog.meta('microlearning-active').seconds,900);
});

test('a deterministic contract failure defers invalidation until positive closure without reconstructing',async t=>{
 const f=await fixture(t);let closed=false,constructed=0,clock=0;
 const ops={now:()=>clock,ensureKnowledgeService:async()=> 'http://borrowed',createKdbBridge:()=>({}),stopped:async()=>closed,
  construct:async()=>{constructed++;clock=12000;throw new SkillEvaluationContractError('malformed protected artifact');}};
 await assert.rejects(runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:1000},ops),SkillEvaluationContractError);
 const retained=f.backlog.meta('microlearning-active');assert.equal(retained.invalidContract.code,'INVALID_SKILL_EVALUATION');assert.equal(retained.elapsed,12);
 assert.equal(f.state.db.prepare('SELECT status FROM skill_learning_requests WHERE id=?').get(f.id).status,'background');
 assert.equal((await runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:1000},ops)).status,'blocked');
 closed=true;
 const result=await runMicrolearning({config:f.config,backlog:f.backlog,resource:f.resource,remainingSeconds:1000},ops);
 assert.equal(result.status,'inconclusive');assert.equal(constructed,1);assert.equal(f.backlog.meta('microlearning-active'),null);
 assert.ok(f.backlog.meta('microlearning-last').elapsed>=12);
});
