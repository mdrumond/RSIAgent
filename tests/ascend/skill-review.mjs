import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {DatabaseSync} from 'node:sqlite';
import {State} from '../../controller/state.mjs';
import {inventoryFailures,validateClassification,validateClassificationBatch,claimLearningRequest,enqueueLearningRequest,recordLearningOutcome} from '../../controller/skill-review.mjs';
import {runReview,validateReviewProposal} from '../../cli/skill-review.mjs';
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
  const used=body.messages.filter(m=>m.role==='tool').length;res.writeHead(200,{'Content-Type':'text/event-stream'});
  const frame=delta=>res.write('data: '+JSON.stringify({id:'review-fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason:null}]})+'\n\n');
  frame({role:'assistant',...(used>1?{content:'Done'}:{tool_calls:[{index:0,id:used?'submit':'read',type:'function',function:used?{name:'submit_review',arguments:JSON.stringify({result:value})}:{name:'read',arguments:JSON.stringify({path:'input.json'})}}]})});
  res.write('data: '+JSON.stringify({id:'review-fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:{},finish_reason:used>1?'stop':'tool_calls'}],usage:{prompt_tokens:10,completion_tokens:5,total_tokens:15}})+'\n\n');res.end('data: [DONE]\n\n');
 });server.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>{server.closeAllConnections();server.close();});
 const credentials_file=join(f.root,'fixture.env');await writeFile(credentials_file,'REVIEW_FIXTURE=fixture-only\n');
 const profile={provider:'review-fixture',model:'scripted',credential_env:'REVIEW_FIXTURE',credentials_file,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:32000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
 const input={directory,config:{actor:profile},group,entries:[],seconds:20,validate:r=>validateClassification(r,group,[])};
 const result=await nativeClassify(input);assert.equal(result.value.cause,'unknown');assert.ok(result.stats.tokens.total>0);assert.equal(requests,3);
 await nativeClassify(input);assert.equal(requests,3);
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


test('admitted learning resumes the original actor and source before considering another draft',async t=>{
 const {nativeLearn}=await import('../../cli/skill-review-native.mjs'),f=await fixture(t),directory=join(f.root,'learning-resume');await mkdir(directory);
 const source=join(f.root,'tiny-source');await mkdir(source);await writeFile(join(source,'answer.py'),'answer=1\n');
 let constructed=0,launched=0,identity,allocation;
 const ops={ensureKnowledgeService:async()=> 'http://127.0.0.1:1/mcp',createKdbBridge:()=>({}),
  construct:async()=>{constructed++;return [{id:'tiny',goal:'CPU experiment',source,actor_seconds:60,generation:{checks:[]}}];},importProject:async(_root,spec)=>spec,
  operate:async(command,cfg)=>{const state=new State(join(directory,'state.sqlite'));try{
   if(++launched===1){assert.equal(command,'wave');allocation=state.project('tiny').spec.actor_seconds;identity=state.admit('tiny',{pid:1,start:'fixture',boot:'fixture'}).id;
    state.checkpoint(identity,17.5,'same-native-session');await writeFile(join(source,'answer.py'),'answer=2\n');throw Error('Interrupted after actor launch');}
   assert.equal(command,'resume');const a=state.attempt(identity);assert.equal(a.elapsed,17.5);assert.equal(a.session,'same-native-session');
   assert.equal(cfg.actor_seconds,allocation);assert.equal(await readFile(join(source,'answer.py'),'utf8'),'answer=2\n');
   state.db.prepare("UPDATE attempts SET outcome='passed' WHERE id=?").run(identity);state.event(identity,'postprocess-completed',{});
  }finally{state.close();}}};
 const input={directory,config:{...await load(join(f.source,'instance.json')),root:f.source},request:{id:'question',question:'CPU boundary'},seconds:100};
 await assert.rejects(nativeLearn(input,ops),/Interrupted/);
 const resumed=await nativeLearn({...input,seconds:30},ops);assert.equal(resumed.status,'verified-cpu');assert.equal(resumed.attempts[0].id,identity);assert.equal(constructed,1);
 await nativeLearn({...input,seconds:20},ops);assert.equal(launched,2);
});

test('interrupted fork publication reuses the copied memory database without replacing its contents',async t=>{
 const f=await fixture(t);await assert.rejects(runReview({...f.options,stage:'classify'},{classify}),/inventory/);
 await runReview({...f.options,stage:'inventory'});
 await assert.rejects(runReview({...f.options,stage:'classify'},{classify:async()=>{throw Error('Stopped after fork publication');}}),/Stopped/);
 const copy=join(f.options.output,'campaigns','a3');await rm(join(copy,'instance.json'));
 const db=new DatabaseSync(join(copy,'knowledge.sqlite'));db.exec('CREATE TABLE retained(value TEXT); INSERT INTO retained VALUES (\'preserved\')');db.close();
 await runReview({...f.options,stage:'classify'},{classify});
 const retained=new DatabaseSync(join(copy,'knowledge.sqlite'),{readOnly:true});try{assert.equal(retained.prepare('SELECT value FROM retained').get().value,'preserved');}finally{retained.close();}
});

test('a settled comparison timeout retains partial results and does not block the next candidate',async t=>{
 const f=await fixture(t);let calls=0;
 const report=await runReview(f.options,{classify:async args=>({...await classify(args),guidanceIds:[args.group.type==='generation-rejected'?'debug-baseline':'implementation-baseline']}),review:async args=>update(args),
  evaluate:async({directory})=>{if(++calls===1){await mkdir(directory,{recursive:true});await writeFile(join(directory,'results.json'),JSON.stringify({status:'elapsed',elapsedSeconds:900,results:[{correct:true,arm:'active'}]}));throw new DOMException('Comparison elapsed','TimeoutError');}return evaluation();}});
 assert.equal(calls,2);assert.equal(report.candidates[0].decision.reason,'evaluation-elapsed');assert.equal(report.candidates[1].published,true);
 const state=await load(join(f.options.output,'review.json'));assert.equal(Object.values(state.candidates)[0].results.length,1);
});

test('unresolved comparison execution stops dispatch without publishing an inconclusive closure',async t=>{
 const f=await fixture(t);let calls=0;
 await assert.rejects(runReview(f.options,{classify,review:async args=>update(args),evaluate:async({directory})=>{calls++;await mkdir(directory,{recursive:true});await writeFile(join(directory,'results.json'),JSON.stringify({runningSample:{key:'uncertain'},results:[]}));throw new DOMException('Comparison elapsed','TimeoutError');}}),{name:'TimeoutError'});
 const state=await load(join(f.options.output,'review.json'));assert.equal(calls,1);assert.equal(Object.values(state.candidates)[0].decision,undefined);
});

test('a revised proposal receives a new candidate while prior decisions and versions remain frozen',async t=>{
 const f=await fixture(t);let comparisons=0;
 const deps={classify,review:async args=>({...update(args),changes:[{...update(args).changes[0],text:args.learning.length?'Learned revised guidance':'Original evidence-backed guidance'}],learningRequests:[{question:'Boundary question',hypothesis:'Boundary hypothesis',requiredEvidence:'CPU evidence',learningTarget:'cpu'}]}),learn:async()=>({status:'verified-cpu',evidence:[]}),evaluate:async()=>{comparisons++;return evaluation();}};
 for(const stage of ['inventory','classify','propose','evaluate'])await runReview({...f.options,stage},deps);
 const first=Object.values((await load(join(f.options.output,'review.json'))).candidates)[0],before=await readFile(join(first.directory,'skills','versions',first.version,'guidance.json'),'utf8');
 for(const stage of ['learn','propose','evaluate'])await runReview({...f.options,stage},deps);
 const candidates=Object.values((await load(join(f.options.output,'review.json'))).candidates);assert.equal(candidates.length,2);assert.equal(comparisons,2);assert.notEqual(candidates[0].version,candidates[1].version);
 assert.equal(await readFile(join(first.directory,'skills','versions',first.version,'guidance.json'),'utf8'),before);assert.equal(candidates[0].decision.promote,true);
});

async function scriptedReview(t,root,plan){
 const {createServer}=await import('node:http'),{once}=await import('node:events');let requests=0;
 const server=createServer(async(req,res)=>{
  let raw='';for await(const chunk of req)raw+=chunk;const body=JSON.parse(raw),tool=await plan(body,requests++);
  res.writeHead(200,{'Content-Type':'text/event-stream'});
  const delta={role:'assistant',...(tool?{tool_calls:[{index:0,id:'review-'+requests,type:'function',function:{name:tool.name,arguments:JSON.stringify(tool.args)}}]}:{content:'Done'})};
  const frame=value=>res.write('data: '+JSON.stringify(value)+'\n\n');
  frame({id:'review-pages',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason:null}]});
  frame({id:'review-pages',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:{},finish_reason:tool?'tool_calls':'stop'}],usage:{prompt_tokens:10,completion_tokens:5,total_tokens:15}});res.end('data: [DONE]\n\n');
 });server.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>{server.closeAllConnections();server.close();});
 const credentials_file=join(root,'extra.env');await writeFile(credentials_file,'REVIEW_EXTRA=fixture-only\n');
 return {requests:()=>requests,profile:{provider:'review-extra',model:'scripted',credential_env:'REVIEW_EXTRA',credentials_file,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:32000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}}};
}

test('native reads page beyond the preview before classifying a late distinct failure',async t=>{
 const {nativeClassify}=await import('../../cli/skill-review-native.mjs'),f=await fixture(t),directory=join(f.root,'late-diagnostic');
 const group={id:'late-group',message:'Verification failed',occurrences:Array.from({length:7},(_,i)=>({reference:'failure:'+i,data:{message:i===6?'Incorrect strided layout':'Distinct boundary '+i},artifacts:[]}))};
 const decision=cause=>({cause,reason:'Observed diagnostic evidence',coverage:'uncovered',guidanceIds:[],evidenceRefs:[cause==='actor'?'failure:6':'failure:0']});
 const provider=await scriptedReview(t,f.root,(_body,n)=>[
  {name:'read',args:{path:'input.json'}},
  {name:'submit_review',args:{result:decision('unknown')}},
  {name:'read',args:{path:join(directory,'evidence','late-group-diagnostics.jsonl'),offset:7,limit:1}},
  {name:'submit_review',args:{result:decision('actor')}}
 ][n]);
 const result=await nativeClassify({directory,config:{actor:provider.profile},group,entries:[],seconds:30,validate:r=>validateClassification(r,group,[])});
 assert.equal(result.value.cause,'actor');assert.equal(provider.requests(),5);
 assert.equal((await load(join(directory,'read-evidence.json'))).length,7);
 const full=(await readFile(join(directory,'evidence','late-group-occurrences.jsonl'),'utf8')).trim().split('\n').map(line=>JSON.parse(line));assert.equal(full.length,7);assert.equal(full.at(-1).reference,'failure:6');
});

test('a resumed native review validates saved retrieval citations without querying again',async t=>{
 const {reviewSession}=await import('../../cli/skill-review-native.mjs'),f=await fixture(t),directory=join(f.root,'retrieval-resume');let phase=0;
 const provider=await scriptedReview(t,f.root,body=>phase&& !body.messages.some(m=>m.role==='tool')?{name:'submit_review',args:{result:{memoryRefs:['memory:old-entry']}}}:null);
 const validate=(result,records)=>{assert.ok(records.some(r=>r.entries.some(e=>e.collection==='memory'&&e.chunk_id==='old-entry')));return result;};
 const input={directory,config:{actor:provider.profile},prompt:{task:'Retain retrieved references'},validate,seconds:30};
 await assert.rejects(reviewSession(input),/did not submit/);const session=await load(join(directory,'native-session.json'));
 await writeFile(join(directory,'retrievals.jsonl'),JSON.stringify({entries:[{collection:'memory',chunk_id:'old-entry'}]})+'\n');phase=1;
 const resumed=await reviewSession(input);assert.deepEqual(resumed.value.memoryRefs,['memory:old-entry']);assert.equal(resumed.retrievals.length,1);
 assert.deepEqual(await load(join(directory,'native-session.json')),session);assert.equal(provider.requests(),3);
});


test('harness learning remains a user intervention and cannot consume a Phase 1 slot',async t=>{
 const f=await fixture(t);let learned=0;
 const report=await runReview(f.options,{classify:async args=>({...await classify(args),cause:'harness',coverage:'uncovered',guidanceIds:[]}),
  review:async({groups})=>({action:'learn',reason:'Harness closure needs operator attention',groupIds:groups.map(g=>g.id),question:'Repair a missing harness receipt?',hypothesis:'Owned execution proof is missing',requiredEvidence:'Operator closure',learningTarget:'cpu'}),
  learn:async()=>{learned++;throw Error('Harness requests must not reach an actor');}});
 assert.equal(learned,0);assert.equal(report.learning[0].status,'user-required');assert.equal(report.learning[0].intervention,'user-required');
 const request=Object.values((await load(join(f.options.output,'review.json'))).learning)[0],state=new State(join(f.root,'harness-queue.sqlite'));try{
  assert.throws(()=>enqueueLearningRequest(state,request));
  assert.equal(claimLearningRequest(state,{skills:{review:{enabled:true}}},{draft:'unused',target:'a3',topic:'debug'}),null);
 }finally{state.close();}
});

test('negative baseline and stage verdicts retain real diagnostics while expected negative controls do not become failures',async t=>{
 const f=await fixture(t),state=new State(join(f.source,'state.sqlite')),attempt=state.db.prepare('SELECT id FROM attempts').get().id;
 const original='/original/campaign/attempts/'+attempt+'/baseline.log',copied=join(f.source,'evidence','baseline.log'),body='RuntimeError: PyTorch baseline dtype mismatch\n';
 await mkdir(join(f.source,'evidence'));await writeFile(copied,body);
 const {createHash}=await import('node:crypto'),sha256=createHash('sha256').update(body).digest('hex');
 await writeFile(join(f.source,'evidence-files.json'),JSON.stringify({[original]:{path:'evidence/baseline.log',sha256,size:body.length}}));
 const verdict={passed:false,exitCode:1,evidence:{path:original,sha256}};
 state.event(attempt,'attempt-error',{stage:'baseline-preflight',message:'Generated project target baseline failed before actor startup'});
 state.event(attempt,'baseline-preflight-completed',verdict);state.event(attempt,'project-invalid',{reason:'target-baseline-failed',...verdict});
 state.event(attempt,'other-stage-completed',{result:{correct:false}});
 const write=state.db.prepare('INSERT INTO generation_events(draft,type,data) VALUES (?,?,?)');
 write.run('matched-negative','generation-construction-check',JSON.stringify({kind:'negative',expected_exit:1,exitCode:1,passed:false}));
 write.run('bad-positive','generation-construction-check',JSON.stringify({kind:'reference',expected_exit:0,exitCode:1}));
 write.run('bad-negative','generation-construction-check',JSON.stringify({kind:'negative',expected_exit:1,exitCode:0}));state.close();
 const inventory=await inventoryFailures(f.options.sources);
 for(const type of ['baseline-preflight-completed','project-invalid','other-stage-completed'])assert.ok(inventory.groups.some(g=>g.type===type));
 assert.equal(inventory.groups.some(g=>g.occurrences.some(o=>o.draftId==='matched-negative')),false);
 for(const draft of ['bad-positive','bad-negative'])assert.ok(inventory.groups.some(g=>g.occurrences.some(o=>o.draftId===draft)));
 const label=inventory.groups.find(g=>g.type==='attempt-error'),occurrence=label.occurrences[0];
 assert.equal(occurrence.artifacts[0].path,copied);assert.equal(occurrence.artifacts[0].sha256,sha256);
 assert.ok(label.linkedDiagnostics.some(d=>d.type==='baseline-preflight-completed'&&d.data.passed===false));
 const report=await runReview({...f.options,stage:'inventory'});assert.ok(report.failureGroups>3);
 await runReview({...f.options,stage:'classify'},{classify:async args=>{
  if(args.group.type==='attempt-error'){
   const evidence=args.group.linkedDiagnostics.find(d=>d.type==='baseline-preflight-completed');assert.equal(await readFile(evidence.artifacts[0].path,'utf8'),body);
   return args.validate({cause:'specification',coverage:'covered',guidanceIds:['task-construction-baseline'],reason:'The actual baseline failed its mathematical/API contract',evidenceRefs:[evidence.reference]});
  }return classify(args);
 }});
 const reviewed=await load(join(f.options.output,'review.json'));assert.equal(reviewed.classifications[label.id].cause,'specification');assert.equal(reviewed.classifications[label.id].intervention,null);
});

test('native Pi blocks an outside workspace grep before execution and accepts corrected classification fields',async t=>{
 const {nativeClassify}=await import('../../cli/skill-review-native.mjs'),{default:fs}=await import('node:fs'),{syncBuiltinESMExports}=await import('node:module');
 const f=await fixture(t),group=(await inventoryFailures(f.options.sources)).groups[0],directory=join(f.root,'bounded-native'),outside=join(f.root,'outside-workspace.txt');await writeFile(outside,'Unrelated workspace content');
 let outsideActions=0;const originalStat=fs.promises.stat;
 fs.promises.stat=async function(path,...args){if(String(path)===outside)outsideActions++;return originalStat.call(this,path,...args);};syncBuiltinESMExports();
 t.after(()=>{fs.promises.stat=originalStat;syncBuiltinESMExports();});
 const invalid={cause:'unknown',coverage:'unknown',guidanceIds:['implementation-baseline'],reason:'Need observed diagnostics',evidenceRefs:[group.occurrences[0].reference]};
 const entries=[{id:'implementation-baseline',scope:'implementation'}];
 const provider=await scriptedReview(t,f.root,(_body,n)=>[
  {name:'read',args:{path:'input.json'}},
  {name:'submit_review',args:{result:invalid}},
  {name:'grep',args:{path:outside,pattern:'classification',limit:20}},
  {name:'submit_review',args:{result:{...invalid,coverage:'covered'}}}
 ][n]);
 const result=await nativeClassify({directory,config:{actor:provider.profile},group,entries,seconds:30,validate:r=>validateClassification(r,group,entries)});
 assert.equal(result.value.coverage,'covered');assert.equal(provider.requests(),5);assert.equal(outsideActions,0);
 const blocks=(await readFile(join(directory,'tool-policy.jsonl'),'utf8')).trim().split('\n').map(line=>JSON.parse(line));
 assert.equal(blocks.length,1);assert.equal(blocks[0].tool,'grep');assert.equal(blocks[0].blocked,true);assert.equal(blocks[0].path,outside);
});


test('native classification batches validate all group IDs and correct missing or duplicate outputs',async t=>{
 const {nativeClassifyBatch}=await import('../../cli/skill-review-native.mjs'),f=await fixture(t),directory=join(f.root,'native-batch');
 const groups=['first','second'].map(id=>({id,occurrences:[{reference:id+':event',data:{message:id+' failure'},artifacts:[]}]}));
 const rows=groups.map(group=>({groupId:group.id,cause:'unknown',coverage:'uncovered',guidanceIds:[],reason:'The evidence is inconclusive.',evidenceRefs:[group.occurrences[0].reference]}));
 const provider=await scriptedReview(t,f.root,(_body,n)=>[
  {name:'read',args:{path:'input.json'}},
  {name:'submit_review',args:{result:{classifications:rows.slice(0,1)}}},
  {name:'submit_review',args:{result:{classifications:[rows[0],rows[0]]}}},
  {name:'submit_review',args:{result:{classifications:rows}}}
 ][n]);
 const result=await nativeClassifyBatch({directory,config:{actor:provider.profile},groups,entries:[],seconds:30,validate:r=>validateClassificationBatch(r,groups,[])});
 assert.deepEqual(result.value.classifications.map(r=>r.groupId),groups.map(g=>g.id));assert.equal(provider.requests(),5);
});

test('classification batches resume unfinished groups without repeating completed outputs or resetting the deadline',async t=>{
 const f=await fixture(t),state=new State(join(f.source,'state.sqlite')),attempt=state.db.prepare('SELECT id FROM attempts').get().id;
 for(let i=0;i<10;i++)state.event(attempt,'failure-recorded',{message:'Distinct failure '+i});state.close();
 await runReview({...f.options,stage:'inventory'});let calls=0;const batches=[];
 const classifyBatch=async({groups,entries,validate})=>{batches.push(groups.map(g=>g.id));if(++calls===2)throw Error('Interrupted batch');
  return {value:validate({classifications:await Promise.all(groups.map(async group=>({...await classify({group,validate:r=>r}),groupId:group.id})))}),stats:{tokens:{total:groups.length*10}}};};
 await assert.rejects(runReview({...f.options,stage:'classify'},{classifyBatch}),/Interrupted batch/);
 const before=await load(join(f.options.output,'review.json')),completed=Object.keys(before.classifications);assert.ok(completed.length>1);
 await runReview({...f.options,stage:'classify'},{classifyBatch});
 const after=await load(join(f.options.output,'review.json'));assert.equal(after.startedAt,before.startedAt);assert.equal(Object.keys(after.classifications).length,after.inventory.groups.length);
 assert.ok(batches.at(-1).every(id=>!completed.includes(id)));
 const recorded=after.timings.find(t=>t.stage==='classifyBatch');assert.equal(recorded.groupCount,recorded.groupIds.length);assert.equal(recorded.sourceId,'a3');assert.equal(recorded.guidanceVersion,'baseline');
 assert.ok(after.stats.some(s=>s.stage==='classifyBatch'&&s.groupCount===s.groupIds.length));
});

test('batch rollout resumes a pending legacy group session before batching newer work',async t=>{
 const f=await fixture(t);await runReview({...f.options,stage:'inventory'});const initial=await load(join(f.options.output,'review.json')),group=initial.inventory.groups[0];
 const directory=join(f.options.output,'classifications',group.id);await mkdir(directory,{recursive:true});const retained={path:'retained-native-session.jsonl'};await writeFile(join(directory,'native-session.json'),JSON.stringify(retained));
 let singles=0,batches=0;
 await runReview({...f.options,stage:'classify'},{classify:async args=>{singles++;assert.equal(args.group.id,group.id);assert.equal(args.directory,directory);return classify(args);},classifyBatch:async args=>{batches++;return args.validate({classifications:await Promise.all(args.groups.map(async g=>({...await classify({group:g,validate:r=>r}),groupId:g.id})))});}});
 assert.equal(singles,1);assert.equal(batches,1);assert.deepEqual(await load(join(directory,'native-session.json')),retained);
});


test('an inconclusive writer bucket retains its session and does not starve supported candidates',async t=>{
 const f=await fixture(t);let reviewed=0,evaluated=0,failedSession;
 const deps={classify:async args=>({...await classify(args),guidanceIds:[args.group.type==='generation-rejected'?'debug-baseline':'implementation-baseline']}),
  review:async args=>{if(++reviewed===1){failedSession=join(args.directory,'session.native.jsonl');await writeFile(failedSession,JSON.stringify({role:'assistant',partial:true})+'\n');throw new DOMException('Writer elapsed','TimeoutError');}return update(args);},
  evaluate:async()=>{evaluated++;return evaluation();}};
 const report=await runReview(f.options,deps);assert.equal(reviewed,2);assert.equal(evaluated,1);assert.equal(report.candidates[0].published,true);
 assert.equal(report.dispositions.inconclusive,1);assert.equal(report.reviewFailures[0].error.name,'TimeoutError');
 assert.equal(JSON.parse((await readFile(failedSession,'utf8')).trim()).partial,true);
 await runReview({...f.options,stage:'propose'},deps);assert.equal(reviewed,2);
});

test('propose persists learning metadata immediately and report can recover old raw questions without executing them',async t=>{
 const f=await fixture(t);let learned=0;const deps={classify,review:async args=>({...update(args),learningRequests:[{question:'Unanswered boundary',hypothesis:'Tail behavior differs',requiredEvidence:'CPU controls',learningTarget:'cpu'}]}),learn:async()=>{learned++;throw Error('Metadata stages must not execute learning');}};
 for(const stage of ['inventory','classify'])await runReview({...f.options,stage},deps);
 const proposed=await runReview({...f.options,stage:'propose'},deps);assert.equal(proposed.learning.length,1);assert.equal(proposed.learning[0].status,'pending');assert.equal(learned,0);
 const file=join(f.options.output,'review.json'),legacy=await load(file),started=legacy.startedAt;legacy.learning={};await writeFile(file,JSON.stringify(legacy));
 const recovered=await runReview({...f.options,stage:'report'},deps);assert.equal(recovered.learning.length,1);assert.equal(learned,0);assert.equal((await load(file)).startedAt,started);
 const elapsed=await runReview({...f.options,stage:'evaluate'},{...deps,evaluate:async({directory})=>{await mkdir(directory,{recursive:true});await writeFile(join(directory,'results.json'),JSON.stringify({status:'elapsed',elapsedSeconds:900,results:[]}));throw new DOMException('Comparison elapsed','TimeoutError');}});
 assert.equal(elapsed.learning.length,1);assert.equal(learned,0);assert.equal(elapsed.candidates[0].decision.promote,false);
});

test('native writer corrects blocker scope before authoritative grounding admits its update',async t=>{
 const {reviewSession}=await import('../../cli/skill-review-native.mjs'),f=await fixture(t),directory=join(f.root,'scope-correction');
 const groups=[{id:'harness',occurrences:[{reference:'harness:event',data:{message:'Missing receipt'},artifacts:[]}]}],classifications={harness:{cause:'harness'}};
 const proposal={action:'update',reason:'Report evidenced harness blockers',groupIds:['harness'],memoryRefs:['memory:chunk'],changes:[{action:'upsert',id:'safe-blocker',scope:'implementation',text:'Preserve source and report.',critical:false,negative:false,response:'report-blocker',evidenceRefs:['harness:event']}]};
 const corrected=structuredClone(proposal);corrected.changes[0].scope='preflight';
 const provider=await scriptedReview(t,f.root,(_body,n)=>[
  {name:'read',args:{path:'input.json'}},{name:'submit_review',args:{result:proposal}},{name:'submit_review',args:{result:corrected}}
 ][n]);
 const result=await reviewSession({directory,config:{actor:provider.profile},prompt:{task:'Submit evidence-backed guidance',groups},evidenceGroups:groups,seconds:30,validate:r=>validateReviewProposal(r,groups,classifications)});
 assert.equal(result.value.changes[0].scope,'preflight');assert.equal(provider.requests(),4);
 const actor=structuredClone(corrected);actor.changes[0].response='actor-change';actor.changes[0].scope='implementation';assert.equal(validateReviewProposal(actor,groups,{harness:{cause:'actor'}}).changes[0].response,'actor-change');
 const unknown=structuredClone(corrected),ref='unknown-'+Date.now();unknown.changes[0].evidenceRefs=[ref];assert.throws(()=>validateReviewProposal(unknown,groups,classifications),error=>error.message.includes(ref));
});


test('question-specific origins keep actor learning eligible in a mixed harness bucket',async t=>{
 const f=await fixture(t);let learned=0;
 const deps={classify:async args=>({...await classify(args),guidanceIds:['preflight-baseline'],cause:args.group.type==='generation-rejected'?'harness':'actor'}),
  review:async({groups})=>({action:'no-change',reason:'Different questions need different evidence',groupIds:groups.map(g=>g.id),learningRequests:[
   {groupIds:[groups.find(g=>g.type!=='generation-rejected').id],question:'CPU conformance',hypothesis:'Boundary differs',requiredEvidence:'CPU controls',learningTarget:'cpu'},
   {groupIds:[groups.find(g=>g.type==='generation-rejected').id],question:'Harness closure',hypothesis:'Missing receipt',requiredEvidence:'Operator investigation',learningTarget:'cpu'}]}),learn:async()=>{learned++;return {status:'inconclusive',evidence:[]};}};
 for(const stage of ['inventory','classify','propose'])await runReview({...f.options,stage},deps);
 const pending=await runReview({...f.options,stage:'report'},deps);assert.equal(pending.learning.find(r=>r.question==='CPU conformance').status,'pending');assert.equal(pending.learning.find(r=>r.question==='Harness closure').status,'user-required');
 await runReview({...f.options,stage:'learn'},deps);assert.equal(learned,1);
});


test('resume freezes attempt jobs, evidence indexes and actual copied-log bytes',async t=>{
 const {createHash}=await import('node:crypto');
 for(const changed of ['job','index','log'])await t.test(changed,async t=>{
  const f=await fixture(t),db=new DatabaseSync(join(f.source,'state.sqlite'),{readOnly:true}),attempt=db.prepare('SELECT id FROM attempts').get().id;db.close();
  const job=join(f.source,'attempts',attempt,'job.json'),log=join(f.source,'evidence','copied.log'),index=join(f.source,'evidence-files.json');
  await mkdir(dirname(job),{recursive:true});await writeFile(job,JSON.stringify({kind:'original',mode:'phase1',skillVersion:'baseline'}));await mkdir(dirname(log),{recursive:true});await writeFile(log,'original diagnostic');
  const digest=createHash('sha256').update('original diagnostic').digest('hex');await writeFile(index,JSON.stringify({'/original/log':{path:'evidence/copied.log',sha256:digest}}));
  await runReview({...f.options,stage:'inventory'});const before=await load(join(f.options.output,'review.json'));
  if(changed==='job')await writeFile(job,JSON.stringify({kind:'retry',mode:'phase1',skillVersion:'baseline'}));
  if(changed==='index'){const metadata=await load(index);metadata['/original/log'].size=19;await writeFile(index,JSON.stringify(metadata));}
  if(changed==='log')await writeFile(log,'changed diagnostic');
  await assert.rejects(runReview({...f.options,stage:'report'}));assert.equal((await load(join(f.options.output,'review.json'))).startedAt,before.startedAt);
 });
});

test('a copied log must match its declared SHA256 before the first review starts',async t=>{
 const f=await fixture(t);await mkdir(join(f.source,'evidence'));await writeFile(join(f.source,'evidence','bad.log'),'actual bytes');
 await writeFile(join(f.source,'evidence-files.json'),JSON.stringify({'/original/bad.log':{path:'evidence/bad.log',sha256:'0'.repeat(64)}}));
 await assert.rejects(runReview({...f.options,stage:'inventory'}));
 await assert.rejects(readFile(join(f.options.output,'review.json')),{code:'ENOENT'});
});


async function demoteFingerprint(file){
 const state=await load(file);state.signature=state.legacySignature;delete state.fingerprintVersion;delete state.legacySignature;await writeFile(file,JSON.stringify(state));return state;
}

test('unchanged legacy reviews migrate fingerprints without restarting clocks or replacing results and sessions',async t=>{
 const f=await fixture(t);for(const stage of ['inventory','classify'])await runReview({...f.options,stage},{classify});
 const file=join(f.options.output,'review.json'),old=await demoteFingerprint(file),session=join(f.options.output,'retained.native.jsonl');await writeFile(session,JSON.stringify({original:true})+'\n');
 let replayed=0;await runReview({...f.options,stage:'classify'},{classify:async()=>{replayed++;throw Error('Completed classifications must not replay');}});
 const migrated=await load(file);assert.equal(replayed,0);assert.equal(migrated.startedAt,old.startedAt);assert.equal(migrated.seconds,old.seconds);assert.deepEqual(migrated.classifications,old.classifications);assert.deepEqual(migrated.candidates,old.candidates);assert.equal(migrated.fingerprintVersion,2);assert.equal(migrated.fingerprintMigrations.length,1);assert.equal(JSON.parse((await readFile(session,'utf8')).trim()).original,true);
});

test('unprovable legacy job or copied-log changes reject model resume but permit read-only history inspection',async t=>{
 const {createHash}=await import('node:crypto');
 for(const changed of ['job','log'])await t.test(changed,async t=>{
  const f=await fixture(t),db=new State(join(f.source,'state.sqlite')),attempt=db.db.prepare('SELECT id FROM attempts').get().id;
  const job=join(f.source,'attempts',attempt,'job.json'),log=join(f.source,'evidence','legacy.log');await mkdir(dirname(job),{recursive:true});await writeFile(job,JSON.stringify({kind:'original',mode:'phase1'}));await mkdir(dirname(log),{recursive:true});await writeFile(log,'retained bytes');
  const sha256=createHash('sha256').update('retained bytes').digest('hex');await writeFile(join(f.source,'evidence-files.json'),JSON.stringify({'/original/legacy.log':{path:'evidence/legacy.log',sha256}}));db.event(attempt,'attempt-error',{message:'Baseline failed',evidence:{path:'/original/legacy.log'}});db.close();
  await runReview({...f.options,stage:'inventory'});const file=join(f.options.output,'review.json');await demoteFingerprint(file);const original=await readFile(file,'utf8');
  if(changed==='job')await writeFile(job,JSON.stringify({kind:'retry',mode:'phase1'}));else await writeFile(log,'different bytes');
  await assert.rejects(runReview({...f.options,stage:'classify'},{classify}));
  const inspected=await runReview({...f.options,stage:'report'});assert.equal(inspected.legacyInspection,true);assert.equal(inspected.resumeAllowed,false);assert.equal(await readFile(file,'utf8'),original);
 });
});
test('bucket policy requires explicit bucket and validates objective bindings before provider work',async t=>{
 const f=await fixture(t);let calls=0;
 await assert.rejects(runReview({...f.options,bucketPolicy:true},{classify:async()=>{calls++;}}),/Five frozen tasks/);assert.equal(calls,0);
 await runReview({...f.options,bucketPolicy:true,stage:'inventory'});
 await assert.rejects(runReview({...f.options,bucketPolicy:true,stage:'classify'},{classify}),/Explicit failure bucket/);
});
test('bucket review carries configured A3 tasks through fork, native evaluation and final paired promotion',async t=>{
 const f=await fixture(t),{nativeEvaluate}=await import('../../cli/skill-review-native.mjs'),{runMicroeval}=await import('../../cli/skill-microeval.mjs');
 const checker=join(f.root,'verify.py'),pairedChecker=join(f.root,'paired.py');
 await writeFile(checker,"import sys,json,pathlib,hashlib\np=pathlib.Path(sys.argv[1])/'kernel.py'\nprint(json.dumps(dict(correct=True,kernelSha256=hashlib.sha256(p.read_bytes()).hexdigest(),latencySeconds=100,evidence={'fixture':True})))\n");
 await writeFile(pairedChecker,"import sys,json,pathlib,hashlib\na,b=[hashlib.sha256((pathlib.Path(x)/'kernel.py').read_bytes()).hexdigest() for x in sys.argv[1:]]\nprint(json.dumps(dict(correct=True,activeKernelSha256=a,candidateKernelSha256=b,activeLatencySeconds=1,candidateLatencySeconds=.9,evidence={'alternating':True})))\n");
 const evaluations=Object.fromEntries(['correctness','optimization'].map(objective=>[objective,{binding:{target:'CPU-fixture-not-device-evidence'},knowledge:{url:'http://127.0.0.1:1234/mcp',identity:'logical-snapshot'},developmentSeconds:30,pairedSeconds:15,tasks:['a','b','c','d','e'].map(id=>({id,bucket:objective,kind:'a3-kernel',prompt:'Fixture',files:{'kernel.py':'answer=42\n'},checker,pairedChecker}))}]));
 const options={...f.options,bucketPolicy:true,evaluations};let evaluated=0;
 const dependencies={classify:args=>args.validate({cause:'actor',bucket:'optimization',coverage:'covered',guidanceIds:['implementation-baseline'],reason:'Actual timing evidence',evidenceRefs:[args.group.occurrences[0].reference]}),review:async args=>update(args),evaluate:async args=>{
  evaluated++;if(evaluated===1)throw Error('Interrupted candidate fixture');assert.equal(args.objective,'optimization');assert.equal(args.config.skill_review_evaluations.optimization.tasks.length,5);assert.ok(args.seconds>44&&args.seconds<=45);
  assert.equal(args.config.skill_review_evaluations.optimization.knowledge.url,'http://127.0.0.1:5678/mcp');
  return nativeEvaluate(args,{runMicroeval:input=>runMicroeval(input,{sample:async()=>({tokens:1})})});
 },evaluationSettled:async()=>false};
 for(const stage of ['inventory','classify','propose'])await runReview({...f.options,bucketPolicy:true,stage},dependencies);
 const startedAt=(await load(join(options.output,'review.json'))).startedAt;
 await assert.rejects(runReview({...options,stage:'evaluate'},dependencies),/Interrupted candidate fixture/);
 const interrupted=Object.values((await load(join(options.output,'review.json'))).candidates)[0];
 assert.equal((await load(join(interrupted.directory,'instance.json'))).skill_review_evaluations.optimization.knowledge.url,'http://127.0.0.1:1234/mcp');
 for(const settings of Object.values(evaluations))settings.knowledge.url='http://127.0.0.1:5678/mcp';
 const report=await runReview({...options,stage:'evaluate'},dependencies);
 assert.equal((await load(join(options.output,'review.json'))).startedAt,startedAt);
 assert.equal(evaluated,2);assert.equal(report.candidates[0].version,interrupted.version);assert.equal(report.candidates[0].decision.promote,true);assert.equal(report.candidates[0].published,true);assert.equal(report.buckets.optimization,3);
 const state=await load(join(options.output,'review.json')),candidate=Object.values(state.candidates)[0],result=await load(join(candidate.directory,'evaluation','results.json'));
 assert.equal(result.pairedTimings.length,10);assert.equal(result.totalSeconds,30);assert.equal(result.pairedReplay.seconds,15);assert.equal(result.binding.target,'CPU-fixture-not-device-evidence');
 for(const settings of Object.values(evaluations))settings.knowledge.url='http://127.0.0.1:5678/mcp';
 await runReview({...options,stage:'report'});
 evaluations.optimization.knowledge.identity='different';await assert.rejects(runReview({...options,stage:'report'}),/Frozen review evaluation/);evaluations.optimization.knowledge.identity='logical-snapshot';
 await writeFile(pairedChecker,'changed trusted checker');await assert.rejects(runReview({...options,stage:'evaluate'}),/Frozen review/);
});

test('explicit harness ownership cannot be routed into correctness A/B',async t=>{
 const f=await fixture(t);let compared=0;
 for(const stage of ['inventory'])await runReview({...f.options,bucketPolicy:true,stage});
 await assert.rejects(runReview({...f.options,bucketPolicy:true,stage:'classify'},{classify:args=>args.validate({cause:'harness',bucket:'correctness',coverage:'uncovered',guidanceIds:[],reason:'Execution wrapper fault',evidenceRefs:[args.group.occurrences[0].reference]}),evaluate:async()=>{compared++;return evaluation();}}),/Harness cause/);
 const saved=await load(join(f.options.output,'review.json'));assert.equal(Object.keys(saved.classifications).length,0);assert.equal(Object.keys(saved.candidates).length,0);assert.equal(compared,0);
});
test('native writer corrects evaluation bucket used as guidance scope before admission',async t=>{
 const {reviewSession}=await import('../../cli/skill-review-native.mjs'),f=await fixture(t),directory=join(f.root,'guidance-scope-correction');
 const groups=[{id:'actor',occurrences:[{reference:'actor:event',data:{message:'Wrong reduction'},artifacts:[]}]}],classifications={actor:{cause:'actor'}};
 const proposal={action:'update',reason:'Measured repair applies',groupIds:['actor'],memoryRefs:['memory:chunk'],changes:[{action:'upsert',id:'reduction-repair',scope:'correctness',text:'Check reduction axes before broadcasting.',critical:false,negative:false,response:'actor-change',evidenceRefs:['actor:event']}]};
 const corrected=structuredClone(proposal);corrected.changes[0].scope='implementation';
 const provider=await scriptedReview(t,f.root,(_body,n)=>[{name:'read',args:{path:'input.json'}},{name:'submit_review',args:{result:proposal}},{name:'submit_review',args:{result:corrected}}][n]);
 const result=await reviewSession({directory,config:{actor:provider.profile},prompt:{task:'Submit evidence-backed guidance',groups},evidenceGroups:groups,seconds:30,validate:r=>validateReviewProposal(r,groups,classifications)});
 assert.equal(provider.requests(),4);assert.equal(result.value.changes[0].scope,'implementation');assert.equal(proposal.changes[0].scope,'correctness');
 assert.equal((await load(join(directory,'submission.json'))).changes[0].scope,'implementation');
 const pending={...proposal,action:'learn',question:'Which reduction axes?',hypothesis:'Axis order matters',requiredEvidence:'Independent check',learningTarget:'cpu'};
 assert.equal(validateReviewProposal(pending,groups,classifications).action,'learn');assert.equal(pending.changes[0].scope,'correctness');
});
test('traceback ownership survives inventory grouping and native diagnostic deduplication',async t=>{
 const f=await fixture(t),state=new State(join(f.source,'state.sqlite')),attempt=state.db.prepare('SELECT id FROM attempts').get().id;
 state.db.exec('DELETE FROM events; DELETE FROM generation_events');
 for(const path of ['/tmp/work-a/candidate/kernel.py','/tmp/work-b/candidate/kernel.py','/tmp/work-c/trusted/reference.py'])state.event(attempt,'verification',{correct:false,error:`Traceback: File "${path}", line 7: ValueError: incorrect argument`});state.close();
 const inventory=await inventoryFailures(f.options.sources);assert.equal(inventory.groups.length,2);assert.deepEqual(inventory.groups.map(g=>g.occurrences.length).sort(),[1,2]);
 const group={id:'ownership',occurrences:inventory.groups.flatMap(g=>g.occurrences)},directory=join(f.root,'ownership-review');
 const decision={cause:'unknown',reason:'Observed failures have different source ownership',coverage:'uncovered',guidanceIds:[],evidenceRefs:group.occurrences.map(o=>o.reference)};
 const provider=await scriptedReview(t,f.root,(_body,n)=>[{name:'read',args:{path:'input.json'}},{name:'submit_review',args:{result:decision}}][n]);
 const {nativeClassify}=await import('../../cli/skill-review-native.mjs');await nativeClassify({directory,config:{actor:provider.profile},group,entries:[],seconds:30,validate:r=>validateClassification(r,group,[])});
 const input=await load(join(directory,'input.json'));assert.equal(input.group.distinctDiagnostics,2);
 assert.equal(input.group.occurrences.filter(o=>o.data.error.includes('/candidate/kernel.py')).length,1);
 assert.equal(input.group.occurrences.filter(o=>o.data.error.includes('/trusted/reference.py')).length,1);
 assert.equal((await load(join(directory,'read-evidence.json'))).length,2);
});
