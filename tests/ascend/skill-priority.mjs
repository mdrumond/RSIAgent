import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {State} from '../../controller/state.mjs';
import {SkillBacklog} from '../../controller/skill-backlog.mjs';
import {runPrioritySkills,prioritySkillStatus,migratePriorityThroughput,wakeQualifiedSuiteWaiters} from '../../cli/skill-priority.mjs';
import {backgroundSkillStatus} from '../../cli/skill-background.mjs';
import {resolveHypothesisCases,registerQualifiedSuite,hypothesisCaseReadiness} from '../../cli/skill-priority-cases.mjs';
import {selectCohort,claimCohort} from '../../cli/skill-cohort.mjs';
import {claimLearningRequest,enqueueLearningRequest,recordLearningOutcome} from '../../controller/skill-review.mjs';
import {qualifyHypothesisTasks} from '../../cli/skill-hypothesis-evaluation.mjs';
import {publishSkillVersion} from '../../cli/skills.mjs';

async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'priority-controller-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const config={format:1,root,target:'a3',mode:'phase1',projects:[],concurrency:5,environment:{options:{}},
  skills:{activeVersion:'baseline',background:true,priority:{enabled:true}}};
 await writeFile(join(root,'instance.json'),JSON.stringify(config));await mkdir(join(root,'skills'),{recursive:true});
 const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());state.event(null,'generation-rejected',{message:'bad frozen shape'});
 const backlog=new SkillBacklog(state,{target:'a3'});backlog.ingest();const group=backlog.failures()[0];backlog.acknowledge(group.id);
 const add=(key,priority=1,extra={})=>backlog.upsert({key,priority,reason:'Observed recurring numeric failure',objective:'correctness',skill:'implementation',base:'baseline',
  groupIds:[group.id],proposal:{action:'update',changes:[{id:key,text:'Handle boundary '+key+' before the final tile load'}]},evidence:{},...extra});
 const calls=[];
 const ops={analyze:async()=>{},prioritize:async()=>{},prepare:async({item})=>{calls.push(['prepare',item.key]);return {status:'ready',active:item.base,candidate:'candidate-'+item.key};},
  resolveCases:async()=>({status:'ready',target:'cpu',tasks:[],binding:{target:'cpu'}}),
  evaluate:async args=>{calls.push(['evaluate',args.skills,args.remainingSeconds]);return {status:'complete',decision:{promote:false,reason:'insufficient-correctness-improvement'}};},settled:async()=>true};
 const values={resource:{id:'reserved'},remainingSeconds:3600};return{root,config,state,backlog,group,add,ops,calls,values};
}
test('one Phase 1 failure drives multiple comparisons without another Phase 2 completion',async t=>{
 const f=await fixture(t);f.add('first',20);f.add('second',10);
 assert.equal((await backgroundSkillStatus(f.config)).queued,2);
 await runPrioritySkills(f.config,f.values,f.ops);await runPrioritySkills(f.config,f.values,f.ops);
 assert.deepEqual(f.calls.filter(c=>c[0]==='prepare').map(c=>c[1]),['first','second']);
 assert.equal(f.backlog.get('first').status,'rejected');assert.equal(f.backlog.get('second').status,'rejected');
 assert.equal((await backgroundSkillStatus(f.config)).pending,false);
});
test('new urgent item waits until the selected cohort finishes',async t=>{
 const f=await fixture(t);f.add('running',20);f.add('waiting',10);let release,started;
 const gate=new Promise(r=>release=r),ready=new Promise(r=>started=r);
 const running=runPrioritySkills(f.config,f.values,{...f.ops,evaluate:async args=>{started();await gate;return f.ops.evaluate(args);}});
 await ready;try{f.add('urgent',100);const status=await prioritySkillStatus(f.config);
 assert.equal(status.current.id,f.backlog.get('running').id);assert.equal(status.backlog[0].id,f.backlog.get('urgent').id);
 }finally{release();await running;}await runPrioritySkills(f.config,f.values,f.ops);
 assert.deepEqual(f.calls.filter(c=>c[0]==='prepare').map(c=>c[1]),['running','waiting']);
 await runPrioritySkills(f.config,f.values,f.ops);assert.equal(f.calls.filter(c=>c[0]==='prepare').at(-1)[1],'urgent');
});
test('partial analysis and ranking timeouts preserve ready work and expose diagnostics',async t=>{
 const f=await fixture(t);
 await runPrioritySkills(f.config,f.values,{...f.ops,analyze:async()=>{f.add('ready',20);f.add('new-ready',90);throw Error('Analysis elapsed');},prioritize:async()=>{throw Error('Ranking elapsed');}});
 assert.equal(f.backlog.get('new-ready').status,'rejected');assert.equal(f.backlog.get('ready').status,'queued');
 const status=await prioritySkillStatus(f.config);assert.equal(status.analysisError.message,'Analysis elapsed');assert.equal(status.rankingError.message,'Ranking elapsed');
});
test('a real rejected comparison feeds new diagnostic evidence back into analysis',async t=>{
 const f=await fixture(t),item=f.add('first-proposal',20);
 await runPrioritySkills(f.config,f.values,{...f.ops,evaluate:async()=>({status:'complete',decision:{promote:false,reason:'insufficient-correctness-improvement'},
  report:{results:[{projectId:'tail',run:0,arm:'candidate',correct:false,verification:{error:'Tail case differs'}}]}})});
 const feedback=f.state.db.prepare("SELECT data FROM events WHERE type='skill-hypothesis-rejected'").get();assert(feedback);
 assert.equal(JSON.parse(feedback.data).hypothesisId,item.id);
 f.backlog.ingest();const group=f.backlog.failures().find(g=>g.type==='skill-hypothesis-rejected');assert(group);assert.equal(group.occurrences[0].data.diagnostics[0].projectId,'tail');
 await runPrioritySkills(f.config,f.values,{...f.ops,analyze:async()=>{f.add('revised-proposal',90,{proposal:{changes:[{scope:'implementation',text:'Mask the final tile before loading'}]}});}});
 assert.equal(f.backlog.get('revised-proposal').status,'rejected');
});
test('operator pause retains comparison identity, preparation and spent allowance',async t=>{
 const f=await fixture(t),item=f.add('resume',1,{preparation:{proposalElapsedSeconds:12}}),abort=new AbortController();
 await assert.rejects(runPrioritySkills(f.config,{...f.values,signal:abort.signal},{...f.ops,evaluate:async()=>{abort.abort(Error('operator pause'));throw abort.signal.reason;}}),/operator pause/);
 const retained=f.backlog.run(item.id);assert.equal(retained.status,'running');assert(retained.elapsed>=12);assert(!retained.checkpoint.runningAt);
 f.add('urgent',100);await runPrioritySkills(f.config,f.values,f.ops);
 assert.equal(f.backlog.run(retained.id).status,'rejected');assert.equal(f.calls.filter(c=>c[0]==='prepare').length,1);
 assert(f.calls.find(c=>c[0]==='evaluate')[2]<1800-12);assert.equal(f.backlog.get('urgent').status,'queued');
});
test('expired writing allowance prevents preparation and promotion',async t=>{
 const f=await fixture(t),item=f.add('spent',1,{preparation:{proposalElapsedSeconds:1801}});
 await runPrioritySkills(f.config,f.values,f.ops);assert.equal(f.backlog.get(item.id).status,'inconclusive');assert.deepEqual(f.calls,[]);
});
test('accepted candidate uses public publication and next item requalifies against new active skills',async t=>{
 const f=await fixture(t);f.add('improve',20);f.add('later',10);
 await runPrioritySkills(f.config,f.values,{...f.ops,evaluate:async()=>({status:'complete',decision:{promote:true,reason:'correctness-improved'}})});
 assert.equal(JSON.parse(await readFile(join(f.root,'instance.json'),'utf8')).skills.activeVersion,'candidate-improve');
 assert.equal(f.backlog.get('improve').status,'accepted');
 await runPrioritySkills(f.config,f.values,f.ops);await runPrioritySkills(f.config,f.values,f.ops);
 const later=f.backlog.get('later');assert.equal(later.base,'candidate-improve');assert.equal(later.status,'rejected');
});
test('baseline changed while comparing cannot publish stale candidate',async t=>{
 const f=await fixture(t);f.add('stale');
 await runPrioritySkills(f.config,f.values,{...f.ops,evaluate:async()=>{
  const c=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));c.skills.activeVersion='new-active';await writeFile(join(f.root,'instance.json'),JSON.stringify(c));
  return {status:'complete',decision:{promote:true}};
 }});
 assert.equal(f.backlog.get('stale').status,'inconclusive');assert.equal(JSON.parse(await readFile(join(f.root,'instance.json'),'utf8')).skills.activeVersion,'new-active');
});
test('restart recovers an already committed promotion without evaluating again',async t=>{
 const f=await fixture(t);f.add('committed');
 await assert.rejects(runPrioritySkills(f.config,f.values,{...f.ops,evaluate:async()=>({status:'complete',decision:{promote:true}}),
  publish:async(...args)=>{await publishSkillVersion(...args);throw Error('Exit after publication');}}),/Exit after publication/);
 assert.equal(f.backlog.get('committed').status,'running');
 await runPrioritySkills(f.config,f.values,{...f.ops,evaluate:()=>{throw Error('Accepted comparison replayed');}});
 assert.equal(f.backlog.get('committed').status,'accepted');
});
test('missing relevant cases report a capability gap without constructing another project',async t=>{
 const f=await fixture(t),item=f.add('new-mechanism',90);
 const result=await resolveHypothesisCases({config:f.config,backlog:f.backlog,item,directory:join(f.root,'cases'),seconds:5},
  {prepareTasks:async()=>({manifest:{hypotheses:[]},byHypothesis:{}})});
 assert.equal(result.status,'unsupported');assert.equal(result.missingCapability,true);
 assert.equal(f.state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='skill_learning_requests'").get()?f.state.db.prepare('SELECT COUNT(*) n FROM skill_learning_requests').get().n:0,0);
});
test('legacy pending comparison keeps the old execution path during opt-in migration',async t=>{
 const f=await fixture(t);await writeFile(join(f.root,'skills','evolution.json'),JSON.stringify({history:[],pending:{evaluationRoot:join(f.root,'old'),projectIds:[],runs:2,evaluationSeconds:100},evaluationStatus:'deferred'}));
 const status=await backgroundSkillStatus(f.config);assert.equal(status.pending,true);assert.equal(status.backlog,undefined);
});
test('learned cases must pass real reference and negative controls before admission',async t=>{
 const f=await fixture(t),item=f.add('learned',80),trusted=join(f.root,'trusted');await mkdir(trusted);
 const checker=join(trusted,'checker.py');await writeFile(checker,"import runpy,sys\nfrom pathlib import Path\nraise SystemExit(0 if runpy.run_path(str(Path(sys.argv[1])/'kernel.py')).get('answer')==42 else 1)\n");
 await writeFile(join(trusted,'reference.py'),'answer=42\n');await writeFile(join(trusted,'starter.py'),'answer=0\n');
 const contract={target:'cpu',objective:'correctness',tasks:Array.from({length:5},(_,n)=>({id:'case-'+n,target:'cpu',kind:'hypothesis-case',bucket:'correctness',prompt:'Repair the computed answer',files:{'kernel.py':'starter.py'},checker:'checker.py',
  controls:{reference:{files:{'kernel.py':'reference.py'}},negatives:[{files:{'kernel.py':'starter.py'}}]}}))};
 const manifest=join(trusted,'skill-evaluation.json');await writeFile(manifest,JSON.stringify(contract));
 const editable=join(f.root,'editable');await mkdir(editable);
 f.state.add({id:'learning-project',goal:'Qualify small cases',actor_seconds:30,source:editable,verifier:{files:[manifest,checker,join(trusted,'reference.py'),join(trusted,'starter.py')]}});
 const request=enqueueLearningRequest(f.state,{backlogId:item.id,target:'a3',question:'Qualify cases',requiredEvidence:'Reference and controls',findings:{projectId:'learning-project',outcome:'passed'}});
 f.state.db.prepare("UPDATE skill_learning_requests SET status='verified',draft='draft' WHERE id=?").run(request);
 const admitted=[];let acknowledged;
 const args={config:f.config,backlog:f.backlog,item,directory:join(f.root,'learned-cases'),seconds:120,beforeQualification:async directory=>{await Promise.resolve();acknowledged=directory;admitted.push(directory);}};
 const deps={prepareTasks:async()=>({manifest:{hypotheses:[]},byHypothesis:{}}),select:async()=>({value:{catalogueId:'learning-learning-project',reason:'These controls exercise the hypothesis'}}),qualify:async input=>{assert.equal(acknowledged,input.directory);acknowledged=null;return (await import('../../cli/skill-hypothesis-evaluation.mjs')).qualifyHypothesisTasks(input);}};
 const result=await resolveHypothesisCases(args,deps);assert.equal(result.status,'ready');
 const report=JSON.parse(await readFile(join(args.directory,'qualification','micro','results.json'),'utf8'));
 assert.equal(report.status,'qualified');assert.equal(report.results.length,10);assert(report.results.every(r=>r.passed));
 await writeFile(checker,'raise SystemExit(0)\n');
 assert.equal((await resolveHypothesisCases(args,deps)).status,'unsupported');
 assert.equal(admitted[0],join(args.directory,'qualification'));assert(admitted[1].startsWith(join(args.directory,'qualification')+'/'));assert.notEqual(admitted[0],admitted[1]);
 assert.notEqual(await readFile(result.tasks[0].checker,'utf8'),await readFile(checker,'utf8'));
 const bad=await resolveHypothesisCases({...args,directory:join(f.root,'bad-cases')},deps);
 assert.equal(bad.status,'unsupported');assert.equal(bad.missingCapability,true);
});
test('revised feedback reuses its frozen qualified cases without another learning project',async t=>{
 const f=await fixture(t),home=join(f.root,'skills','priority','hypotheses','prior','cases');await mkdir(home,{recursive:true});
 const checker=join(home,'checker.py');await writeFile(checker,'raise SystemExit(0)\n');
 const tasks=Array.from({length:5},(_,n)=>({id:'case-'+n,target:'cpu',kind:'hypothesis-case',bucket:'correctness',prompt:'Repair the answer',files:{'kernel.py':'answer=0\n'},checker}));
 const path=join(home,'selected.json');await writeFile(path,JSON.stringify({status:'ready',target:'cpu',objective:'correctness',binding:{target:'cpu'},tasks}));
 const item=f.add('revision',90,{evidence:{groups:[{occurrences:[{data:{hypothesisId:'prior',evaluation:{path}}}]}]}});
 const args={config:f.config,backlog:f.backlog,item,directory:join(f.root,'revised-cases'),seconds:10};
 const result=await resolveHypothesisCases(args,{prepareTasks:async()=>({manifest:{hypotheses:[]},byHypothesis:{}}),select:async()=>({value:{catalogueId:'previous-0',reason:'Same boundary cases for the substantive revision'}}),qualify:()=>{throw Error('Already qualified suite replayed');}});
 assert.equal(result.status,'ready');assert.equal(result.tasks.length,5);assert.notEqual(result.tasks[0].checker,checker);
 await writeFile(path,JSON.stringify({status:'ready',target:'a5',objective:'correctness',binding:{target:'a5',environment:{different:true}},tasks}));
 const incompatible=await resolveHypothesisCases({...args,directory:join(f.root,'incompatible-cases')},{prepareTasks:async()=>({manifest:{hypotheses:[]},byHypothesis:{}})});
 assert.equal(incompatible.status,'unsupported');
});

test('crash resume charges only time after the latest elapsed checkpoint',async t=>{
 const f=await fixture(t),item=f.add('crash-clock');let release,started;const ready=new Promise(r=>started=r),gate=new Promise(r=>release=r);
 let monotonic=0;t.mock.timers.enable({apis:['setInterval','Date'],now:Date.now()});
 const running=runPrioritySkills(f.config,f.values,{...f.ops,now:()=>monotonic,evaluate:async()=>{started();await gate;return {status:'complete',decision:{promote:false}};}});
 await ready;monotonic=900000;t.mock.timers.tick(900000);
 const retained=f.backlog.run(item.id);assert.equal(retained.elapsed,900);assert.equal(retained.checkpoint.runningAt,Date.now());
 release();await running;
 // Restore the persisted crash checkpoint, as if no orderly shutdown had happened.
 f.state.db.prepare("UPDATE skill_backlog_runs SET status='running',elapsed=?,checkpoint=?,owner=NULL WHERE id=?").run(retained.elapsed,JSON.stringify(retained.checkpoint),retained.id);
 f.backlog.store({...f.backlog.get(item.id),status:'running'});t.mock.timers.tick(10000);
 let available;await runPrioritySkills(f.config,f.values,{...f.ops,evaluate:async args=>{available=args.remainingSeconds;return {status:'complete',decision:{promote:false}};}});
 assert(available>880&&available<=890,'Only the ten-second uncheckpointed tail is added to 900 seconds');
});

test('pending learning pages keep the background worker eligible and new findings restart the scan',async t=>{
 const f=await fixture(t);f.backlog.meta('learning-seen','[]');
 assert.equal((await prioritySkillStatus(f.config)).hasWork,false);
 f.backlog.meta('learning-pending',true);assert.equal((await prioritySkillStatus(f.config)).hasWork,true);
 f.backlog.meta('learning-scan',{ids:['already-consumed'],offset:1});f.backlog.meta('learning-seen','older findings');
 let scan;await runPrioritySkills(f.config,f.values,{...f.ops,analyze:async()=>{scan=f.backlog.meta('learning-scan');assert.equal(f.backlog.meta('learning-pending'),true);f.backlog.meta('learning-pending',false);}});
 assert.equal(scan,null);assert.equal((await prioritySkillStatus(f.config)).hasWork,false);
});

test('learning outcomes arriving during analysis remain visible for the next scan',async t=>{
 const f=await fixture(t);f.backlog.meta('learning-seen','[]');
 await runPrioritySkills(f.config,f.values,{...f.ops,analyze:async()=>{
  const id=enqueueLearningRequest(f.state,{target:'a3',question:'A new boundary',requiredEvidence:'Actual learning result'});
  claimLearningRequest(f.state,f.config,{draft:'new-draft',target:'a3'});
  recordLearningOutcome(f.state,{id:'new-project',generation:{draft:'new-draft',learningRequest:{id}}},{id:'new-attempt',outcome:'passed'});
  f.backlog.meta('learning-pending',false);
 }});
 assert.equal((await prioritySkillStatus(f.config)).hasWork,true);
 await runPrioritySkills(f.config,f.values,{...f.ops,analyze:async()=>{assert.equal(f.backlog.meta('learning-scan'),null);f.backlog.meta('learning-pending',false);}});
 assert.equal((await prioritySkillStatus(f.config)).hasWork,false);
});
test('missing cases defer to learning without spending QA allowance',async t=>{
 const f=await fixture(t);f.add('missing-suite');let prepared=false,evaluated=false;
 await runPrioritySkills(f.config,f.values,{...f.ops,resolveCases:async()=>({status:'needs-learning',reason:'missing-cases'}),
  prepare:async()=>{prepared=true;},evaluate:async()=>{evaluated=true;}});
 assert.equal(prepared,false);assert.equal(evaluated,false);assert.equal(f.backlog.get('missing-suite').status,'learning');
});
test('bounded factual learning uses the existing background resource after cohort comparisons',async t=>{
 const f=await fixture(t);f.add('ready');let learnt=0;
 enqueueLearningRequest(f.state,{target:'a3',question:'Measure a bounded factual boundary',requiredEvidence:'Actual observation',workKind:'microlearning'});
 await runPrioritySkills(f.config,f.values,{...f.ops,learn:async({resource})=>{assert.equal(resource.id,'reserved');learnt++;return {status:'verified'};}});
 assert.equal(learnt,0);
 await runPrioritySkills(f.config,f.values,{...f.ops,learn:async({resource})=>{assert.equal(resource.id,'reserved');learnt++;return {status:'verified'};}});
 assert.equal(learnt,1);
});

test('audited migration preserves claimed projects and queues only pre-evaluation infrastructure continuations',async t=>{
 const f=await fixture(t),item=f.add('legacy',90),old=f.backlog.claim({base:'baseline'});f.backlog.checkpoint(old.run.id,{elapsed:450,checkpoint:{prepared:{status:'ready',elapsedSeconds:100}}});
 const request=enqueueLearningRequest(f.state,{target:'a3',backlogId:item.id,question:'Qualify an experiment for implementation: legacy',requiredEvidence:'Five controls'});
 const claimed=enqueueLearningRequest(f.state,{target:'a3',question:'Already claimed original phase one',requiredEvidence:'Large experiment'});
 f.state.db.prepare("UPDATE skill_learning_requests SET status='claimed',draft='existing-draft' WHERE id=?").run(claimed);
 f.backlog.finish(old.run.id,{status:'learning',result:{reason:'No target-compatible evaluation suite',learningRequest:request}});
 const audit=await migratePriorityThroughput(f.config,{procedure:'Enable qualified suite lane v2'});assert.deepEqual(audit.tagged,[request]);assert.deepEqual(audit.continuations,[old.run.id]);
 assert.equal(f.state.db.prepare('SELECT status FROM skill_learning_requests WHERE id=?').get(claimed).status,'claimed');assert.equal(f.backlog.run(old.run.id).elapsed,450);
 await runPrioritySkills(f.config,f.values,f.ops);const runs=f.state.db.prepare('SELECT id,elapsed,status FROM skill_backlog_runs ORDER BY rowid').all();
 assert.equal(runs.length,2);assert.equal(runs[0].status,'learning');assert(runs[1].elapsed>=450);assert.equal(runs[1].status,'rejected');assert.equal(f.backlog.meta('procedural-continuations').length,0);
 assert.deepEqual(await migratePriorityThroughput(f.config,{procedure:'Repeated migration'}),audit);
});
test('legacy cached preparation cannot skip v2 contract review',async t=>{
 const f=await fixture(t),item=f.add('legacy-ready'),old=f.backlog.claim({base:'baseline'});
 f.backlog.checkpoint(old.run.id,{elapsed:120,checkpoint:{prepared:{status:'ready',active:'baseline',candidate:'old-candidate',qa:{elapsedSeconds:60}}}});
 let budget;
 await runPrioritySkills(f.config,f.values,{...f.ops,prepare:async args=>{budget=args.priorElapsedSeconds;assert(args.evaluationContract);return f.ops.prepare(args);}});
 assert.equal(budget,60);assert.equal(f.calls.filter(c=>c[0]==='prepare').length,1);assert.equal(f.backlog.get(item.id).status,'rejected');
});
test('qualification custody is checkpointed before a revised suite dispatches',async t=>{
 const f=await fixture(t),item=f.add('revised-suite'),actual=join(f.root,'cases','qualification','new-contract');
 let checked;
 await assert.rejects(runPrioritySkills(f.config,f.values,{...f.ops,
  resolveCases:async({beforeQualification})=>{await beforeQualification(actual);assert.equal(f.backlog.run(item.id).checkpoint.qualificationDirectory,actual);throw Error('qualification transport interrupted');},
  settled:async({directory})=>{checked=directory;return false;}}),/qualification transport interrupted/);
 assert.equal(checked,actual);assert.equal(f.backlog.run(item.id).checkpoint.unresolved,true);assert.equal(f.backlog.run(item.id).status,'running');
 assert.equal(f.calls.length,0);
});

test('higher priority fresh queued work precedes lower priority procedural continuations',async t=>{
 const f=await fixture(t);f.add('old-low',10);const old=f.backlog.claim({base:'baseline'});f.backlog.checkpoint(old.run.id,{elapsed:50,checkpoint:{}});f.backlog.finish(old.run.id,{status:'inconclusive',result:{reason:'qa-elapsed'}});
 f.backlog.meta('procedural-continuations',[{id:old.run.id,procedure:'Upgrade QA',base:'baseline'}]);f.add('fresh-urgent',100);
 await runPrioritySkills(f.config,f.values,f.ops);assert.equal(f.calls.find(c=>c[0]==='prepare')[1],'fresh-urgent');assert.equal(f.backlog.meta('procedural-continuations').length,1);
 await runPrioritySkills(f.config,f.values,f.ops);assert.equal(f.calls.filter(c=>c[0]==='prepare')[1][1],'old-low');
});
test('a newly qualified shared suite wakes waiters once and semantic relevance controls reuse',async t=>{
 const f=await fixture(t),related=f.add('related',20),unrelated=f.add('unrelated',10);
 for(const item of [related,unrelated]){
  const run=f.backlog.claim({base:'baseline'});const request=enqueueLearningRequest(f.state,{target:'a3',workKind:'suite',backlogId:item.id,question:'Qualify an experiment for '+item.key,requiredEvidence:'Five references and negatives'});
  f.backlog.checkpoint(run.run.id,{elapsed:90,checkpoint:{}});f.backlog.finish(run.run.id,{status:'learning',result:{reason:'No target-compatible evaluation suite',learningRequest:request}});
 }
 const checker=join(f.root,'checker.py');await writeFile(checker,"import runpy,sys\nfrom pathlib import Path\nraise SystemExit(0 if runpy.run_path(str(Path(sys.argv[1])/'kernel.py')).get('answer')==42 else 1)\n");
 const suite={target:'cpu',objective:'correctness',mechanism:'related boundary mechanism',binding:{target:'cpu'},tasks:Array.from({length:5},(_,n)=>({id:'shared-'+n,kind:'hypothesis-case',target:'cpu',bucket:'correctness',prompt:'Check related boundary',checker,files:{'kernel.py':'answer=0'},controls:{reference:{files:{'kernel.py':'answer=42'}},negatives:[{files:{'kernel.py':'answer=0'}}]}}))};
 const qualification=await qualifyHypothesisTasks({...suite,config:f.config,directory:join(f.root,'shared-qualification'),remainingSeconds:120});assert.equal(qualification.status,'qualified');
 await registerQualifiedSuite({config:f.config,suite,qualification});
 assert.equal((await wakeQualifiedSuiteWaiters({config:f.config,backlog:f.backlog})).length,2);assert.equal((await wakeQualifiedSuiteWaiters({config:f.config,backlog:f.backlog})).length,0);
 let constructed=0,selected=0;
 const resolver=args=>resolveHypothesisCases(args,{prepareTasks:async()=>({manifest:{hypotheses:[]},byHypothesis:{}}),select:async({prompt})=>{selected++;return {value:{catalogueId:args.item.key==='related'?prompt.catalogues.find(c=>c.id.startsWith('shared-')).id:null,reason:args.item.key==='related'?'Exercises the boundary mechanism':'Different mechanism'}};}});
 await runPrioritySkills(f.config,f.values,{...f.ops,resolveCases:resolver,learn:async()=>{constructed++;}});
 assert.equal(f.backlog.get(related.id).status,'rejected');assert.equal(constructed,0);assert(selected>0);
 const relatedRequest=JSON.parse(f.state.db.prepare("SELECT data FROM skill_learning_requests WHERE status='satisfied'").get().data);assert.equal(relatedRequest.backlogId,related.id);
 await runPrioritySkills(f.config,f.values,{...f.ops,resolveCases:resolver,learn:async()=>{constructed++;}});
 assert.equal(f.backlog.get(unrelated.id).status,'inconclusive');assert.equal(constructed,0);
 const unrelatedRun=f.backlog.run(f.state.db.prepare('SELECT id FROM skill_backlog_runs WHERE item_id=? ORDER BY rowid DESC LIMIT 1').get(unrelated.id).id);
 assert.equal(unrelatedRun.result.status,'unsupported');assert.equal(unrelatedRun.result.missingCapability,true);assert.equal(unrelatedRun.result.reason,'Different mechanism');
 assert.equal(f.calls.filter(c=>c[0]==='evaluate').length,1,'Only the semantically related hypothesis reaches A/B');
 assert.equal(f.state.db.prepare('SELECT count(*) n FROM skill_learning_requests').get().n,2,'No bespoke request is created for an unrelated shared suite');
 const pending=f.state.db.prepare("SELECT data FROM skill_learning_requests WHERE status='pending'").get();assert.equal(JSON.parse(pending.data).backlogId,unrelated.id);
 assert(unrelatedRun.elapsed>=90);assert.equal(unrelatedRun.seconds,1800);
 const cohort=(await prioritySkillStatus(f.config)).cohort;assert.equal(cohort.counts.completed,1);assert.equal(cohort.counts.blocked,1);
});
test('standalone scheduler checkpoints the exact nested qualification path before dispatch',async t=>{
 const f=await fixture(t);f.add('custody',20);let checked=false;
 await runPrioritySkills(f.config,f.values,{...f.ops,resolveCases:async({beforeQualification,directory})=>{
  const actual=join(directory,'qualification','content-identity');await beforeQualification(actual);
  const row=f.state.db.prepare("SELECT checkpoint FROM skill_backlog_runs WHERE status='running'").get();
  assert.equal(JSON.parse(row.checkpoint).qualificationDirectory,actual);checked=true;return {status:'needs-learning',reason:'No cases'};
 }});
 assert.equal(checked,true);
});

test('five-member cohort finishes despite fresh failures and more urgent hypotheses arriving',async t=>{
 const f=await fixture(t);for(let i=0;i<6;i++)f.add('chosen-'+i,60-i);let analyzed=0,ranked=0;
 const ops={...f.ops,analyze:async()=>{analyzed++;},prioritize:async()=>{ranked++;}};
 for(let i=0;i<5;i++){
  await runPrioritySkills(f.config,f.values,ops);
  f.state.event(null,'attempt-error',{message:'New failure '+i});f.add('urgent-'+i,100+i);
  const status=await prioritySkillStatus(f.config);assert.equal(status.cohort.members.length,5);assert.equal(status.cohort.counts.completed,i+1);
 }
 assert.equal(analyzed,0);assert.equal(ranked,0);assert.deepEqual(f.calls.filter(c=>c[0]==='prepare').map(c=>c[1]),Array.from({length:5},(_,i)=>'chosen-'+i));
 await runPrioritySkills(f.config,f.values,ops);assert.equal(analyzed,1);assert.equal(ranked,1);assert.equal(f.calls.filter(c=>c[0]==='prepare').at(-1)[1],'urgent-4');
 assert((await prioritySkillStatus(f.config)).cohort.members.every(m=>!m.id.startsWith('chosen-')));
});
test('ready cohort cases precede higher-priority unsupported members and unrelated continuations',async t=>{
 const f=await fixture(t);f.add('unrelated-old',1);const old=f.backlog.claim({base:'baseline'});f.backlog.checkpoint(old.run.id,{elapsed:100});f.backlog.finish(old.run.id,{status:'inconclusive',result:{reason:'qa-elapsed'}});
 f.backlog.meta('procedural-continuations',[{id:old.run.id,base:'baseline',procedure:'Earlier audited upgrade'}]);
 for(let i=0;i<5;i++)f.add('member-'+i,90-i);let learned=0;
 const ops={...f.ops,caseReadiness:async({item})=>({ready:item.key==='member-4'}),resolveCases:async({item})=>item.key==='member-0'?{status:'unsupported',reason:'No matching mechanism'}:f.ops.resolveCases(),learn:async()=>{learned++;}};
 await runPrioritySkills(f.config,f.values,ops);assert.equal(f.calls.find(c=>c[0]==='prepare')[1],'member-4');
 await runPrioritySkills(f.config,f.values,ops);assert.equal(f.backlog.get('member-0').status,'inconclusive');assert.equal(learned,0);
 const status=await prioritySkillStatus(f.config);assert.equal(status.cohort.counts.blocked,1);assert.equal(status.cohort.members.find(m=>m.id===f.backlog.get('member-0').id).reason,'No matching mechanism');
 assert.equal(f.backlog.meta('procedural-continuations').length,1);assert.equal(f.backlog.run(old.run.id).elapsed,100);
 await runPrioritySkills(f.config,f.values,ops);assert.equal(f.backlog.get('member-1').status,'rejected');
});
test('case qualification for a cohort member precedes a sixth unrelated hypothesis',async t=>{
 const f=await fixture(t);for(let i=0;i<6;i++)f.add('case-'+i,20-i);const resolved=[];
 const ops={...f.ops,resolveCases:async({item,beforeQualification})=>{resolved.push(item.key);await beforeQualification(join(f.root,'qualified',item.key));return f.ops.resolveCases();},analyze:()=>{throw Error('New data must wait');}};
 for(let i=0;i<5;i++)await runPrioritySkills(f.config,f.values,ops);
 assert.deepEqual([...new Set(resolved)],Array.from({length:5},(_,i)=>'case-'+i));assert.equal(f.backlog.get('case-5').status,'queued');
});
test('operator pause retains cohort membership and original continuation spend',async t=>{
 const f=await fixture(t);f.add('retained',20);f.add('peer',10);const abort=new AbortController();
 await assert.rejects(runPrioritySkills(f.config,{...f.values,signal:abort.signal},{...f.ops,evaluate:async()=>{abort.abort(Error('pause'));throw abort.signal.reason;}}),/pause/);
 const first=f.backlog.run(f.backlog.get('retained').id),cohort=f.backlog.meta('hypothesis-cohort');f.backlog.checkpoint(first.id,{elapsed:130,checkpoint:{...first.checkpoint,runningAt:undefined}});
 f.add('outside-urgent',1000);let remaining;
 await runPrioritySkills(f.config,f.values,{...f.ops,evaluate:async args=>{remaining=args.remainingSeconds;return f.ops.evaluate(args);},analyze:()=>{throw Error('Cohort resumed through analysis');}});
 assert(remaining<=1670&&remaining>1660);assert.equal(f.backlog.run(first.id).seconds,1800);assert.equal(f.backlog.meta('hypothesis-cohort').id,cohort.id);
 assert.deepEqual(f.backlog.meta('hypothesis-cohort').members.map(m=>m.id),cohort.members.map(m=>m.id));
 await runPrioritySkills(f.config,f.values,f.ops);assert.equal(f.backlog.get('peer').status,'rejected');assert.equal(f.backlog.get('outside-urgent').status,'queued');
});
test('unfinished comparison rebases without renewing elapsed or reusing old-baseline samples',async t=>{
 const f=await fixture(t);f.add('rebase',20,{preparation:{proposalElapsedSeconds:12}});const abort=new AbortController();
 await assert.rejects(runPrioritySkills(f.config,{...f.values,signal:abort.signal},{...f.ops,evaluate:async()=>{abort.abort(Error('pause'));throw abort.signal.reason;}}),/pause/);
 const old=f.backlog.run(f.backlog.get('rebase').id);await mkdir(old.checkpoint.evaluationDirectory,{recursive:true});await writeFile(join(old.checkpoint.evaluationDirectory,'frozen-samples.json'),'["old-baseline-sample"]');
 f.backlog.checkpoint(old.id,{elapsed:230,checkpoint:{...old.checkpoint,runningAt:undefined,prepared:{...old.checkpoint.prepared,qa:{elapsedSeconds:100}}}});
 const config=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));config.skills.activeVersion='promoted';await writeFile(join(f.root,'instance.json'),JSON.stringify(config));
 await runPrioritySkills(f.config,f.values,f.ops);assert.equal(f.backlog.run(old.id).result.reason,'active-skill-changed');
 let remaining,directory;await runPrioritySkills(f.config,f.values,{...f.ops,prepare:async args=>{assert.equal(args.priorElapsedSeconds,100);return f.ops.prepare(args);},evaluate:async args=>{remaining=args.remainingSeconds;directory=args.directory;return f.ops.evaluate(args);}});
 const latest=f.state.db.prepare('SELECT id,elapsed,seconds,checkpoint,base FROM skill_backlog_runs ORDER BY rowid DESC LIMIT 1').get();
 assert.notEqual(latest.id,old.id);assert.equal(latest.base,'promoted');assert.equal(latest.seconds,1800);assert(latest.elapsed>=230);assert(remaining<=1570&&remaining>1560);
 assert.equal(JSON.parse(latest.checkpoint).requalifiedFrom,old.id);assert.notEqual(directory,old.checkpoint.evaluationDirectory);assert.equal(f.backlog.run(old.id).elapsed,230);assert.equal(await readFile(join(old.checkpoint.evaluationDirectory,'frozen-samples.json'),'utf8'),'["old-baseline-sample"]');
});
test('settled infrastructure failure blocks one member and allows its peer without automatic retry',async t=>{
 const f=await fixture(t);f.add('failed-qa',30);f.add('peer',20);const ops={...f.ops,prepare:async args=>args.item.key==='failed-qa'?{status:'inconclusive',retryable:true,reason:'qa-execution-failed',qa:{elapsedSeconds:90}}:f.ops.prepare(args)};
 await runPrioritySkills(f.config,f.values,ops);assert.equal(f.backlog.get('failed-qa').status,'inconclusive');
 await runPrioritySkills(f.config,f.values,ops);assert.equal(f.backlog.get('peer').status,'rejected');assert.equal(f.state.db.prepare("SELECT count(*) n FROM skill_backlog_runs WHERE item_id=?").get(f.backlog.get('failed-qa').id).n,1);
});
test('idle factual learning cannot claim a legacy bespoke-suite request',async t=>{
 const f=await fixture(t),suite=enqueueLearningRequest(f.state,{target:'a3',question:'Build an old suite',workKind:'suite',requiredEvidence:'Five cases'}),fact=enqueueLearningRequest(f.state,{target:'a3',question:'Check a small archived fact',workKind:'microlearning',requiredEvidence:'Observation'});let picked;
 await runPrioritySkills(f.config,f.values,{...f.ops,learn:async args=>{picked=args.requestIds;return {status:'verified'};}});
 assert.deepEqual(picked,[fact]);assert.equal(f.state.db.prepare('SELECT status FROM skill_learning_requests WHERE id=?').get(suite).status,'pending');
});
test('two campaigns retain independent cohorts even with identical hypothesis keys',async t=>{
 const a=await fixture(t),b=await fixture(t);for(let i=0;i<3;i++){a.add('same-'+i,30-i);b.add('same-'+i,30-i);}
 await runPrioritySkills(a.config,a.values,a.ops);await runPrioritySkills(b.config,b.values,b.ops);
 const before=b.backlog.meta('hypothesis-cohort');await runPrioritySkills(a.config,a.values,a.ops);
 assert.deepEqual(b.backlog.meta('hypothesis-cohort'),before);assert.notEqual(a.backlog.meta('hypothesis-cohort').id,before.id);
 assert.equal(a.backlog.get('same-1').status,'rejected');assert.equal(b.backlog.get('same-1').status,'queued');
});
test('an obsolete procedural continuation blocks explicitly and is not selected again',async t=>{
 const f=await fixture(t),item=f.add('obsolete',20),old=f.backlog.claim({base:'baseline'});f.backlog.finish(old.run.id,{status:'inconclusive',result:{reason:'qa-elapsed'}});
 const continued=f.backlog.continue(old.run.id,{base:'baseline',procedure:'Earlier valid procedure'});f.backlog.finish(continued.id,{status:'rejected',result:{reason:'Scientifically rejected'}});
 f.backlog.meta('procedural-continuations',[{id:old.run.id,base:'baseline',procedure:'Stale queue entry'}]);await runPrioritySkills(f.config,f.values,f.ops);
 assert.equal(f.backlog.meta('procedural-continuations').length,0);assert.equal((await prioritySkillStatus(f.config)).cohort.counts.blocked,1);assert.equal(f.backlog.get(item.id).status,'rejected');
 await runPrioritySkills(f.config,f.values,f.ops);assert.equal(f.state.db.prepare('SELECT count(*) n FROM skill_backlog_runs').get().n,2);
});
test('supported premade qualification precedes unrelated unsupported hypotheses',async t=>{
 const f=await fixture(t);for(let i=0;i<105;i++)f.add('unknown-'+i,200-i);f.add('premade',1);let qualified;
 await runPrioritySkills(f.config,f.values,{...f.ops,caseReadiness:async({item,backlog})=>{assert.equal(backlog.db.location(),f.backlog.db.location());return {ready:false,supported:item.key==='premade'};},resolveCases:async({item,beforeQualification})=>{qualified=item.key;await beforeQualification(join(f.root,'pack-qualification'));return f.ops.resolveCases();}});
 assert.equal(qualified,'premade');assert.equal(f.backlog.get('premade').status,'rejected');assert.equal(f.backlog.get('unknown-0').status,'queued');assert.equal(f.backlog.meta('hypothesis-cohort').members.length,5);
});
test('unresolved case qualification retains its slot until positive closure, then peers proceed',async t=>{
 const f=await fixture(t);f.add('qualifying',30);f.add('peer',20);let dispatched;
 const blocked=await runPrioritySkills(f.config,f.values,{...f.ops,resolveCases:async({beforeQualification})=>{dispatched=join(f.root,'actual-qualification');await beforeQualification(dispatched);return {status:'blocked',reason:'Native command unknown'};},settled:async({directory})=>{assert.equal(directory,dispatched);return false;}});
 assert.equal(blocked.status,'blocked');assert.equal(f.backlog.get('qualifying').status,'running');assert.equal(f.backlog.run(f.backlog.get('qualifying').id).checkpoint.unresolved,true);assert.equal(f.backlog.get('peer').status,'queued');
 await runPrioritySkills(f.config,f.values,f.ops);await runPrioritySkills(f.config,f.values,f.ops);
 assert.equal(f.backlog.get('qualifying').status,'rejected');assert.equal(f.backlog.get('peer').status,'rejected');
});
test('promotion retires outside-cohort continuations without renewing or modifying historical trials',async t=>{
 const f=await fixture(t),queue=[];
 for(let i=0;i<6;i++){
  f.add('legacy-'+i,60-i);const {run}=f.backlog.claim({base:'baseline'});f.backlog.checkpoint(run.id,{elapsed:100+i});f.backlog.finish(run.id,{status:'inconclusive',result:{reason:'qa-elapsed'}});
  queue.push({id:run.id,base:'baseline',procedure:'Earlier audited infrastructure continuation'});
 }
 const oldRows=queue.map(({id})=>f.state.db.prepare('SELECT * FROM skill_backlog_runs WHERE id=?').get(id));f.backlog.meta('procedural-continuations',queue);f.add('promoter',100);
 await runPrioritySkills(f.config,f.values,{...f.ops,evaluate:async()=>({status:'complete',decision:{promote:true,reason:'correctness-improved'}})});
 const current={...JSON.parse(await readFile(join(f.root,'instance.json'),'utf8')),root:f.root},status=await prioritySkillStatus(current);
 assert.equal(status.cohort.counts.pending,4);assert.equal(status.continuationQueued,4,'Only selected pending members remain admissible after promotion');
 for(let i=0;i<4;i++)await runPrioritySkills(current,f.values,f.ops);
 const retired=f.state.db.prepare("SELECT data FROM events WHERE type='skill-continuation-retired'").all().map(row=>JSON.parse(row.data));
 assert.deepEqual(new Set(retired.map(row=>row.id)),new Set(queue.slice(4).map(row=>row.id)));assert(retired.every(row=>row.reason==='active-skill-changed-outside-cohort'));
 assert.deepEqual(queue.map(({id})=>f.state.db.prepare('SELECT * FROM skill_backlog_runs WHERE id=?').get(id)),oldRows);
 for(let i=4;i<6;i++){const item=f.backlog.get('legacy-'+i);assert.equal(item.base,'baseline');assert.equal(f.state.db.prepare('SELECT count(*) n FROM skill_backlog_runs WHERE item_id=?').get(item.id).n,1);}
 let analyses=0;const ops={...f.ops,analyze:async()=>{analyses++;f.backlog.meta('learning-pending',false);for(const group of f.backlog.failures())f.backlog.acknowledge(group.id);}};
 await runPrioritySkills(current,f.values,ops);assert.equal(f.backlog.meta('procedural-continuations').length,0);assert.equal((await prioritySkillStatus(current)).hasWork,false);
 f.state.event(null,'attempt-error',{message:'New post-promotion diagnostic'});
 await runPrioritySkills(current,f.values,{...ops,analyze:async()=>{await ops.analyze();f.add('new-post-promotion',100,{base:current.skills.activeVersion});}});
 assert.equal(analyses,2);assert.equal(f.backlog.get('new-post-promotion').status,'rejected');
});
test('read-only status excludes stale outside-cohort continuations before cleanup',async t=>{
 const f=await fixture(t);f.add('old-baseline');const {run}=f.backlog.claim({base:'baseline'});f.backlog.checkpoint(run.id,{elapsed:120});f.backlog.finish(run.id,{status:'inconclusive',result:{reason:'qa-elapsed'}});
 f.backlog.meta('procedural-continuations',[{id:run.id,base:'baseline',procedure:'Previous infrastructure upgrade'}]);f.backlog.meta('learning-seen','[]');
 const config={...f.config,skills:{...f.config.skills,activeVersion:'promoted'}},status=await prioritySkillStatus(config);
 assert.equal(status.continuationQueued,0);assert.equal(status.hasWork,false);assert.equal(f.backlog.meta('procedural-continuations').length,1,'Observation leaves cleanup to the controller');assert.equal(f.backlog.run(run.id).elapsed,120);
 f.state.event(null,'attempt-error',{message:'New actionable failure'});assert.equal((await prioritySkillStatus(config)).hasWork,true);
});

test('actual catalogue readiness admits a scoped custom CPU suite ahead of unrelated high-priority work',async t=>{
 const f=await fixture(t);for(let i=0;i<7;i++)f.add('unrelated-'+i,100+i);
 const ready=f.add('custom-low-priority',1,{proposal:{action:'update',changes:[{id:'custom-rule',scope:'implementation',text:'Compute the returned integer from the supplied report'}]}}),checker=join(f.root,'custom-checker.py');
 await writeFile(checker,"import runpy,sys\nfrom pathlib import Path\nraise SystemExit(0 if runpy.run_path(str(Path(sys.argv[1])/'kernel.py')).get('answer')==42 else 1)\n");
 const suite={id:'custom-cpu',target:'cpu',objective:'correctness',proposalIds:['custom-rule'],backlogIds:[ready.id],scopes:['implementation'],binding:{target:'cpu'},
  tasks:Array.from({length:5},(_,i)=>({id:'custom-'+i,kind:'hypothesis-case',target:'cpu',bucket:'correctness',prompt:'Repair the computed integer',checker,files:{'kernel.py':'answer=0'},controls:{reference:{files:{'kernel.py':'answer=42'}},negatives:[{files:{'kernel.py':'answer=0'}}]}}))};
 assert.equal((await qualifyHypothesisTasks({...suite,config:f.config,directory:join(f.root,'custom-qualification'),remainingSeconds:120})).status,'qualified');
 f.config.skills.priority.evaluations=[suite];
 const cohort=await selectCohort(f.backlog,f.config,hypothesisCaseReadiness);
 assert.equal(cohort.members.length,5);assert.equal(cohort.members[0].id,ready.id);assert.equal(cohort.members[0].ready,true);
 assert.equal(cohort.members.some(member=>member.id===f.backlog.get('unrelated-0').id),false);
 const claimed=await claimCohort(f.backlog,f.config,undefined,hypothesisCaseReadiness);
 assert.equal(claimed.item.id,ready.id);assert.equal(claimed.run.item_id,ready.id);assert.equal(f.backlog.run(claimed.run.id).status,'running');
 assert.equal(f.backlog.get('unrelated-6').status,'queued');assert.equal(f.backlog.meta('hypothesis-cohort').id,cohort.id);
});
