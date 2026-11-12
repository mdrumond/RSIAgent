import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {DatabaseSync} from 'node:sqlite';
import {State} from '../../controller/state.mjs';
import {inventoryFailures,validateClassification,claimLearningRequest,enqueueLearningRequest,recordLearningOutcome} from '../../controller/skill-review.mjs';
import {runReview} from '../../cli/skill-review.mjs';
import {run as skillInput} from '../../cli/skill-input.mjs';
import {evaluateNonInferiority} from '../../controller/skill-noninferiority.mjs';
import {stages} from '../../controller/workflow.mjs';
const load=async path=>JSON.parse(await readFile(path,'utf8'));
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'skill-review-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const source=join(root,'source');await mkdir(join(source,'skills/versions/baseline'),{recursive:true});
 const config={format:1,target:'a3',mode:'phase2',actor:{provider:'fixture',model:'fixture'},skills:{activeVersion:'baseline'}};
 await writeFile(join(source,'instance.json'),JSON.stringify(config));
 const entries=['global-kdb',...stages].map(scope=>({id:scope+'-baseline',scope,text:'Existing '+scope,critical:true,negative:false}));
 const map={};for(const e of entries){map[e.scope]=e.scope+'.md';await writeFile(join(source,'skills/versions/baseline',map[e.scope]),e.text);}
 await writeFile(join(source,'skills/versions/baseline/skill-map.json'),JSON.stringify(map));await writeFile(join(source,'skills/versions/baseline/guidance.json'),JSON.stringify(entries));
 new DatabaseSync(join(source,'knowledge.sqlite')).close();
 const state=new State(join(source,'state.sqlite'));state.add({id:'p1',goal:'Test',phase:'phase1',actor_seconds:60});
 const attempt=state.admit('p1',{pid:1,start:'fixture',boot:'fixture'});state.event(attempt.id,'failure-recorded',{message:'Wrong result'});
 state.event(attempt.id,'verification',{correct:false});state.db.prepare("UPDATE attempts SET outcome='passed' WHERE id=?").run(attempt.id);
 state.db.exec('CREATE TABLE generation_events(sequence INTEGER PRIMARY KEY,draft TEXT,type TEXT,data TEXT)');
 state.db.prepare('INSERT INTO generation_events(draft,type,data) VALUES (?,?,?)').run('draft','generation-rejected',JSON.stringify({error:'Bad control'}));state.close();
 return {root,source,options:{sources:[{id:'a3',root:source}],output:join(root,'review'),seconds:7200}};
}
const classify=async({group,validate})=>validate({cause:'actor',coverage:'covered',guidanceIds:['implementation-baseline'],evidenceRefs:[group.occurrences[0].reference],reason:'Observed incorrect implementation'});
const update=({groups})=>({action:'update',reason:'Stored repair evidence applies',groupIds:groups.map(g=>g.id),memoryRefs:['memory:one'],changes:[{action:'upsert',id:'check-boundaries',scope:'implementation',text:'Validate independent boundary cases before tuning.',critical:false,negative:false}]});
const evaluation=(correct=true,ratio=1)=>({decision:evaluateNonInferiority(['a','b'].flatMap(projectId=>[0,1].flatMap(run=>['active','candidate'].map(arm=>({projectId,run,arm,correct:arm==='active'||correct,elapsedSeconds:arm==='candidate'?ratio:1,tokens:10})))),{projectIds:['a','b'],runs:2})});

test('inventory pages all journals and includes observations in passed attempts',async t=>{
 const f=await fixture(t),db=new DatabaseSync(join(f.source,'state.sqlite'));
 const attempt=db.prepare('SELECT id FROM attempts').get().id;
 for(let i=0;i<1002;i++)db.prepare('INSERT INTO events(attempt,type,data,created) VALUES (?,?,?,?)').run(attempt,'execution-unresolved',JSON.stringify({message:'Unresolved command'}),'now');db.close();
 const inventory=await inventoryFailures(f.options.sources);
 assert.equal(inventory.groups.reduce((n,g)=>n+g.occurrences.length,0),1005);
 assert.ok(inventory.groups.some(g=>g.type==='failure-recorded'));assert.ok(inventory.groups.some(g=>g.type==='generation-rejected'));
 assert.equal(new Set(inventory.groups.flatMap(g=>g.occurrences.map(o=>o.reference))).size,1005);
});
test('classification separates unknown/uncovered/harness intervention and rejects invented references',async t=>{
 const f=await fixture(t),g=(await inventoryFailures(f.options.sources)).groups[0];
 const value={cause:'harness',coverage:'uncovered',guidanceIds:[],evidenceRefs:[g.occurrences[0].reference],reason:'Execution wrapper fault'};
 assert.equal(validateClassification(value,g,[]).intervention,'user-required');
 assert.equal(validateClassification({...value,cause:'unknown',coverage:'unknown'},g,[]).intervention,null);
 assert.throws(()=>validateClassification({...value,evidenceRefs:['invented']},g,[]));
});
test('manufactured evidence updates immutable skills through evaluation and fresh actor skill reads',async t=>{
 const f=await fixture(t),before=await readFile(join(f.source,'instance.json'),'utf8');let comparisons=0;
 const report=await runReview(f.options,{classify,review:async args=>update(args),evaluate:async()=>{comparisons++;return evaluation();}});
 assert.equal(report.candidates.length,1);assert.equal(report.candidates[0].published,true);
 const state=await load(join(f.options.output,'review.json')),candidate=Object.values(state.candidates)[0],cfg=await load(join(candidate.directory,'instance.json'));
 assert.equal(cfg.skills.activeVersion,candidate.version);
 const fresh=await skillInput({...cfg,root:candidate.directory},{scope:'implementation'});
 const acceptedMap=await load(join(candidate.directory,'skills','versions',candidate.version,'skill-map.json'));
 assert.equal(fresh.content,await readFile(join(candidate.directory,'skills','versions',candidate.version,acceptedMap.implementation),'utf8'));
 assert.equal(await readFile(join(f.source,'instance.json'),'utf8'),before);
 await runReview(f.options,{classify:async()=>{throw Error('Replay');},review:async()=>{throw Error('Replay');},evaluate:async()=>{throw Error('Replay');}});
 assert.equal(comparisons,1);
});
test('incorrect, slower and protected-entry updates cannot publish',async t=>{
 for(const [kind,result]of [['wrong',evaluation(false)],['slow',evaluation(true,1.2)],['protected',evaluation()]]){
  const f=await fixture(t);f.options.output=join(f.root,kind);
  const report=await runReview(f.options,{classify,review:async args=>{const r=update(args);if(kind==='protected')r.changes[0].id='implementation-baseline';return r;},evaluate:async()=>result});
  assert.equal(report.candidates[0].published,false);assert.equal(report.candidates[0].decision.promote,false);
 }
});
test('CPU learning evidence triggers another proposal, device questions stay pending',async t=>{
 const f=await fixture(t);let reviews=0,learned=0;
 const report=await runReview(f.options,{classify,review:async args=>{reviews++;return args.learning.length?update(args):{action:'learn',reason:'Need independent boundary evidence',groupIds:args.groups.map(g=>g.id),question:'Which boundary fails?',hypothesis:'The tail must be masked',requiredEvidence:'Independent CPU control',learningTarget:'cpu'};},
  learn:async()=>{learned++;return {status:'verified-cpu',evidence:[{exitCode:0},{exitCode:1}]};},evaluate:async()=>evaluation()});
 assert.equal(reviews,2);assert.equal(learned,1);assert.equal(report.candidates[0].published,true);
 const g=await fixture(t);const deferred=await runReview(g.options,{classify,review:async args=>({action:'learn',reason:'Need device timing',groupIds:args.groups.map(g=>g.id),question:'Does it compile?',hypothesis:'Compiler limitation',requiredEvidence:'Compiler on A3',learningTarget:'device'}),learn:async()=>{throw Error('Must not fabricate CPU proof');}});
 assert.equal(deferred.learning[0].status,'device-evidence-needed');
});
test('interrupted stage resumes without duplicate classifications or resetting elapsed allowance',async t=>{
 const f=await fixture(t);let calls=0;
 await assert.rejects(runReview({...f.options,stage:'all'},{classify:async args=>{if(++calls===2)throw Error('Interrupted fixture');return classify(args);}}),/Interrupted/);
 const before=await load(join(f.options.output,'review.json'));assert.equal(Object.keys(before.classifications).length,1);
 await runReview({...f.options,stage:'classify'},{classify});const after=await load(join(f.options.output,'review.json'));
 assert.equal(after.startedAt,before.startedAt);assert.equal(Object.keys(after.classifications).length,3);
 await assert.rejects(runReview({...f.options,seconds:7201,stage:'report'}),/Frozen review/);
});
test('learning requests are idempotent, opt-in, and preserve topic/target admission',async t=>{
 const f=await fixture(t),state=new State(join(f.root,'learning.sqlite'));t.after(()=>state.close());
 const r={question:'Tail behavior',requiredEvidence:'CPU boundary controls',target:'a3',topic:'reductions'};
 const id=enqueueLearningRequest(state,r);assert.equal(enqueueLearningRequest(state,r),id);
 const args={draft:'draft-a',topic:'reductions',target:'a3'};
 assert.equal(claimLearningRequest(state,{},args),null);
 const cfg={skills:{review:{enabled:true}}};assert.equal(claimLearningRequest(state,cfg,{...args,target:'a5'}),null);
 assert.equal(claimLearningRequest(state,cfg,{...args,topic:'layout'}),null);
 assert.equal(claimLearningRequest(state,cfg,args).id,id);assert.equal(claimLearningRequest(state,cfg,{...args,draft:'other'}),null);
 recordLearningOutcome(state,{id:'project',generation:{draft:args.draft,learningRequest:{id}}},{id:'attempt',outcome:'passed'});
 const row=state.db.prepare('SELECT status,data FROM skill_learning_requests WHERE id=?').get(id);assert.equal(row.status,'verified');assert.equal(JSON.parse(row.data).findings.attemptId,'attempt');
});

test('native Pi classification submits validated results and resumes without another provider call',async t=>{
 const {createServer}=await import('node:http'),{once}=await import('node:events'),{nativeClassify}=await import('../../cli/skill-review-native.mjs');
 const f=await fixture(t),group=(await inventoryFailures(f.options.sources)).groups[0],directory=join(f.root,'native-classify');let requests=0;
 const value={cause:'unknown',coverage:'uncovered',guidanceIds:[],evidenceRefs:[group.occurrences[0].reference],reason:'Observed failure does not establish cause'};
 const server=createServer(async(req,res)=>{
  let raw='';for await(const chunk of req)raw+=chunk;const body=JSON.parse(raw);requests++;
  const used=body.messages.some(m=>m.role==='tool');res.writeHead(200,{'Content-Type':'text/event-stream'});
  const frame=delta=>res.write('data: '+JSON.stringify({id:'review-fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason:null}]})+'\n\n');
  frame({role:'assistant',...(used?{content:'Done'}:{tool_calls:[{index:0,id:'submit',type:'function',function:{name:'submit_review',arguments:JSON.stringify({result:value})}}]})});
  res.write('data: '+JSON.stringify({id:'review-fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:{},finish_reason:used?'stop':'tool_calls'}],usage:{prompt_tokens:10,completion_tokens:5,total_tokens:15}})+'\n\n');res.end('data: [DONE]\n\n');
 });server.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>{server.closeAllConnections();server.close();});
 const credentials_file=join(f.root,'fixture.env');await writeFile(credentials_file,'REVIEW_FIXTURE=fixture-only\n');
 const profile={provider:'review-fixture',model:'scripted',credential_env:'REVIEW_FIXTURE',credentials_file,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:32000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
 const input={directory,config:{actor:profile},group,entries:[],seconds:20,validate:r=>validateClassification(r,group,[])};
 const result=await nativeClassify(input);assert.equal(result.value.cause,'unknown');assert.ok(result.stats.tokens.total>0);assert.equal(requests,2);
 await nativeClassify(input);assert.equal(requests,2);
});

test('learning backlog retains every question while only two CPU projects execute',async t=>{
 const f=await fixture(t);let executed=0;
 const questions=[0,1,2,3].map(n=>({question:'Question '+n,hypothesis:'Hypothesis '+n,requiredEvidence:'Independent evidence '+n,learningTarget:n===3?'device':'cpu'}));
 const report=await runReview(f.options,{classify,review:async({groups})=>({action:'no-change',reason:'Need more evidence',groupIds:groups.map(g=>g.id),learningRequests:questions}),learn:async()=>{executed++;return {status:'verified-cpu',evidence:[]};}});
 assert.equal(report.learning.length,4);assert.equal(executed,2);
 assert.equal(report.learning.filter(r=>r.status==='verified-cpu').length,2);
 assert.equal(report.learning.filter(r=>r.status==='pending').length,1);
 assert.equal(report.learning.filter(r=>r.status==='device-evidence-needed').length,1);
});


test('failed learning stays inconclusive while other learning and candidate evaluation continue',async t=>{
 const f=await fixture(t);let learned=0,evaluated=0;
 const questions=[0,1,2].map(n=>({question:'Independent question '+n,hypothesis:'Hypothesis '+n,requiredEvidence:'CPU observation '+n,learningTarget:'cpu'}));
 const report=await runReview(f.options,{classify,review:async args=>({...update(args),learningRequests:questions}),
  learn:async()=>{if(++learned===1)throw Error('Constructor configuration rejected');return {status:'verified-cpu',evidence:[{exitCode:0},{exitCode:1}]};},
  evaluate:async()=>{evaluated++;return evaluation();}});
 assert.equal(learned,2);assert.equal(evaluated,1);assert.equal(report.candidates[0].published,true);
 assert.equal(report.learning.filter(r=>r.status==='inconclusive').length,1);
 assert.equal(report.learning.filter(r=>r.status==='verified-cpu').length,1);
 assert.equal(report.learning.filter(r=>r.status==='pending').length,1);
 assert.equal(report.learning.find(r=>r.status==='inconclusive').error.name,'Error');
 const state=await load(join(f.options.output,'review.json'));assert.equal(state.errors.filter(e=>e.stage==='learn').length,1);
});
