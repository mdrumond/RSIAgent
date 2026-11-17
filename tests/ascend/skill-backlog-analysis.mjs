import test from 'node:test';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {SkillBacklog} from '../../controller/skill-backlog.mjs';
import {stages} from '../../controller/workflow.mjs';
import {enqueueLearningRequest} from '../../controller/skill-review.mjs';
import {analyzeBacklog,prioritizeBacklog,prepareHypothesis} from '../../cli/skill-backlog-analysis.mjs';
async function fixture(t,target='a3'){
 const root=await mkdtemp(join(tmpdir(),'backlog-analysis-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const version=join(root,'skills','versions','baseline');await mkdir(version,{recursive:true});
 const entries=['global-kdb',...stages].map(scope=>({id:scope+'-baseline',scope,text:'Existing '+scope,critical:true,negative:false})),map={};
 for(const entry of entries){map[entry.scope]=entry.scope+'.md';await writeFile(join(version,map[entry.scope]),entry.text);}
 await writeFile(join(version,'skill-map.json'),JSON.stringify(map));await writeFile(join(version,'guidance.json'),JSON.stringify(entries));
 const config={format:1,root,target,actor:{provider:'fixture',model:'fixture'},skills:{activeVersion:'baseline'}};
 const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
 state.add({id:'p1',phase:'phase1',actor_seconds:60,goal:'Boundary'});const a=state.admit('p1',{pid:1,start:'fixture',boot:'fixture'});
 state.event(a.id,'job-admitted',{kind:'original',mode:'phase1'});const log=join(root,'verifier.log');await writeFile(log,'Independent tail mismatch');
 state.event(a.id,'verification',{correct:false,evidence:{path:log}});const backlog=new SkillBacklog(state,{target});backlog.ingest();
 return {root,state,backlog,config,log,directory:join(root,'analysis')};
}
const change=(id,reference)=>({action:'upsert',id,scope:'implementation',text:'Repair '+id+' only at its documented boundary.',critical:false,negative:false,response:'actor-change',evidenceRefs:[reference]});
const classify=async({groups,validate})=>validate({classifications:groups.map(g=>({groupId:g.id,cause:'actor',bucket:'correctness',coverage:'uncovered',guidanceIds:[],evidenceRefs:[g.occurrences[0].reference],reason:'Actual failing output identifies an uncovered boundary'}))});
const review=(ids=['tail-mask'])=>async({directory,groups,validate})=>{
 await mkdir(directory,{recursive:true});await writeFile(join(directory,'retrievals.jsonl'),JSON.stringify({entries:[{collection:'memory',entry_id:'repair',text:'Observed boundary repair'}]})+'\n');
 return validate({action:'update',reason:'Stored repair evidence applies',groupIds:groups.map(g=>g.id),memoryRefs:['memory:repair'],changes:ids.map(id=>change(id,groups[0].occurrences[0].reference))});
};
const qaAnswer=async()=>({answers:Object.fromEntries(['action','mechanism','example','evidence','objective'].map(k=>[k,{text:'Concrete supported '+k,refs:[]}]))});
const qaReview=async()=>({verdict:'approved',reason:'Actual final instruction contains the mechanism and boundary'});
async function completeLearning(f,request,label){
 const source=join(f.root,'learning-'+label);await mkdir(source);await writeFile(join(source,'kernel.py'),'def run(x): return x');
 const id='learning-'+label;f.state.add({id,source,phase:'phase1',goal:'Independent learned fact',actor_seconds:60});const attempt=f.state.admit(id,{pid:1,start:'fixture',boot:'fixture'});
 f.state.event(attempt.id,'verification',{passed:true,correctness:{passed:true,exitCode:0}});f.state.db.prepare("UPDATE attempts SET outcome='passed' WHERE id=?").run(attempt.id);f.state.event(attempt.id,'postprocess-completed',{});
 const data=JSON.parse(request.data);data.findings={attemptId:attempt.id,projectId:id,outcome:'passed'};f.state.db.prepare("UPDATE skill_learning_requests SET status='verified',data=? WHERE id=?").run(JSON.stringify(data),request.id);
}


test('one Phase1 failure admits independent grounded hypotheses and freezes copied evidence',async t=>{
 const f=await fixture(t),result=await analyzeBacklog(f,{classify,review:review(['tail-mask','alignment'])});
 assert.equal(result.items.length,2);assert.notEqual(result.items[0].id,result.items[1].id);
 assert.deepEqual(result.items.map(i=>i.proposal.changes.length),[1,1]);assert.deepEqual(result.items.map(i=>i.status),['queued','queued']);
 assert.equal(result.items[0].evidence.groups[0].occurrences[0].phase,'phase1');
 const copied=result.items[0].evidence.groups[0].occurrences[0].artifacts[0];assert.notEqual(copied.path,f.log);
 await writeFile(f.log,'Changed live log');assert.equal(await readFile(copied.path,'utf8'),'Independent tail mismatch');
 assert.equal(f.backlog.failures().length,0);assert.deepEqual(result.items[0].evidence.memoryEvidence.map(e=>e.entry_id),['repair']);
});

test('unretrieved memory does not admit a hypothesis or acknowledge its failure',async t=>{
 const f=await fixture(t);
 await assert.rejects(analyzeBacklog(f,{classify,review:async({groups})=>({action:'update',reason:'Unsupported',groupIds:groups.map(g=>g.id),memoryRefs:['invented:repair'],changes:[change('tail-mask',groups[0].occurrences[0].reference)]})}),/retrieved/);
 assert.equal(f.backlog.list().length,0);assert.equal(f.backlog.failures().length,1);
});

test('harness faults become user-required findings without model repairs or learning projects',async t=>{
 const f=await fixture(t);let proposed=0;
 const result=await analyzeBacklog(f,{classify:async({groups,validate})=>validate({classifications:groups.map(g=>({groupId:g.id,cause:'harness',bucket:'harness',coverage:'uncovered',guidanceIds:[],evidenceRefs:[g.occurrences[0].reference],reason:'Copied diagnostic proves wrapper exception'}))}),review:async()=>{proposed++;}});
 assert.equal(result.items[0].status,'user-required');assert.equal(proposed,0);assert.equal(f.backlog.claim({base:'baseline'}),null);
 assert.equal(f.state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='skill_learning_requests'").get(),undefined);
});

test('priority decisions move urgent new hypotheses ahead without changing active comparisons',async t=>{
 const f=await fixture(t);await analyzeBacklog(f,{classify,review:review(['old','urgent'])});const [old,urgent]=f.backlog.list();
 f.backlog.rank([{id:old.id,priority:90,reason:'Initial blocker'},{id:urgent.id,priority:30,reason:'Later question'}]);
 const active=f.backlog.claim({base:'baseline'});assert.equal(active.item.id,old.id);
 await prioritizeBacklog(f,{session:async({prompt,validate})=>validate({priorities:prompt.items.map(i=>({id:i.id,priority:100,reason:'New urgent diagnostic'}))})});
 assert.equal(f.backlog.run(active.run.id).snapshot.id,old.id);f.backlog.finish(active.run.id,{status:'rejected',result:{reason:'No improvement'}});
 assert.equal(f.backlog.claim({base:'baseline'}).item.id,urgent.id);
});

test('Q&A inspects final text; candidates are immutable and reused on resume without publication',async t=>{
 const f=await fixture(t),{items:[item]}=await analyzeBacklog(f,{classify,review:review()});let answers=0;
 const args={...f,item,directory:join(f.root,'prepare'),remainingSeconds:100};
 const result=await prepareHypothesis(args,{qaAnswer:async input=>{answers++;assert.equal(input.proposal.changes[0].id,'tail-mask');return {...await qaAnswer(),changes:[{...input.proposal.changes[0],text:'Mask only the partial tail; preserve full interior tiles.'}]};},qaReview});
 assert.equal(result.status,'ready');assert.notEqual(result.candidate,'baseline');assert.equal(f.config.skills.activeVersion,'baseline');
 const candidate=JSON.parse(await readFile(join(f.root,'skills','versions',result.candidate,'guidance.json'),'utf8'));assert.ok(candidate.some(e=>e.id==='tail-mask'&&e.text==='Mask only the partial tail; preserve full interior tiles.'));
 const resumed=await prepareHypothesis(args,{qaAnswer:async()=>{throw Error('Must not repeat');},qaReview});assert.equal(resumed.candidate,result.candidate);assert.equal(answers,1);
 await assert.rejects(prepareHypothesis({...args,item:{...item,revision:item.revision+1}}),/inputs changed/);
 await assert.rejects(prepareHypothesis({...args,config:{...f.config,skills:{activeVersion:'new-active'}}}),/Rebase and requalify/);
});

test('QA knowledge gaps enqueue compatible learning and prohibit premature candidate creation',async t=>{
 const f=await fixture(t),{items:[item]}=await analyzeBacklog(f,{classify,review:review()});
 const prepared=await prepareHypothesis({...f,item,directory:join(f.root,'prepare'),remainingSeconds:100},{qaAnswer,qaReview:async()=>({verdict:'learn',reason:'Need target evidence',learningRequests:[{learningTarget:'device',question:'Does the mask compile?',hypothesis:'Tail layout matters',requiredEvidence:'Independent target compile'}]})});
 assert.equal(prepared.status,'learning');assert.equal(prepared.candidate,undefined);
 const row=f.state.db.prepare('SELECT data FROM skill_learning_requests').get(),request=JSON.parse(row.data);assert.equal(request.backlogId,item.id);assert.equal(request.target,'a3');
});

test('verified curriculum learning reopens its linked item with new evidence and a fresh review',async t=>{
 const f=await fixture(t),question={action:'learn',reason:'Need evidence',question:'Which tail is safe?',hypothesis:'Mask the final partial tile',requiredEvidence:'Independent target result',learningTarget:'device'};
 const first=await analyzeBacklog(f,{classify,review:async({groups,validate})=>validate({...question,groupIds:groups.map(g=>g.id)})});
 const item=first.items[0];assert.equal(item.status,'learning');const request=f.state.db.prepare('SELECT id,data FROM skill_learning_requests').get();
 const source=join(f.root,'learning-source');await mkdir(source);await writeFile(join(source,'kernel.py'),'def run(x): return x');
 f.state.add({id:'learning-project',phase:'phase1',goal:'Actual target evidence',actor_seconds:60,source});const learned=f.state.admit('learning-project',{pid:1,start:'fixture',boot:'fixture'});
 const learnedLog=join(f.root,'learning-verifier.log');await writeFile(learnedLog,'Actual passing target verifier');
 f.state.event(learned.id,'verification',{passed:true,fingerprint:'a'.repeat(64),correctness:{passed:true,exitCode:0,evidence:learnedLog,receipt:{correct:true,kernelSha256:createHash('sha256').update('def run(x): return x').digest('hex'),evidence:{device:'npu:0'}}}});f.state.db.prepare("UPDATE attempts SET outcome='passed' WHERE id=?").run(learned.id);f.state.event(learned.id,'postprocess-completed',{});
 await mkdir(join(f.root,'attempts',learned.id),{recursive:true});await writeFile(join(f.root,'attempts',learned.id,'job.json'),JSON.stringify({config:{target:'a3',environment:{options:{mode:'ascend',runtime:'fixture-runtime'}}}}));
 const data=JSON.parse(request.data);data.findings={attemptId:learned.id,projectId:'learning-project',outcome:'passed'};
 f.state.db.prepare("UPDATE skill_learning_requests SET status='verified',data=? WHERE id=?").run(JSON.stringify(data),request.id);
 let observed;const second=await analyzeBacklog(f,{classify,review:async input=>{observed=input.learning;return review()(input);}});
 assert.equal(second.items[0].id,item.id);assert.equal(second.items[0].status,'queued');assert.equal(observed[0][1].result.status,'verified-device');
 assert.equal(observed[0][1].result.findings.attemptId,learned.id);assert.equal(observed[0][1].result.binding.runtime,'fixture-runtime');assert.notEqual(observed[0][1].result.project.source,source);assert.equal(await readFile(join(observed[0][1].result.project.source,'kernel.py'),'utf8'),'def run(x): return x');assert.notEqual(second.items[0].preparation.proposalDirectory,item.preparation.proposalDirectory);
});


test('interrupted proposal writing retains its spent elapsed time on retry',async t=>{
 const f=await fixture(t);let proposalDirectory;
 await assert.rejects(analyzeBacklog(f,{classify,review:async input=>{proposalDirectory=input.directory;await new Promise(r=>setTimeout(r,20));throw Error('Provider disconnected');}}),/Provider disconnected/);
 const spent=JSON.parse(await readFile(join(proposalDirectory,'writing-spend.json'),'utf8')).elapsedSeconds;assert.ok(spent>0);
 const result=await analyzeBacklog(f,{classify,review:review()});assert.ok(result.items[0].preparation.proposalElapsedSeconds>=spent);
});

test('shared learning questions retain all linked hypotheses instead of losing a waiter',async t=>{
 const f=await fixture(t),{items}=await analyzeBacklog(f,{classify,review:review(['mask','alignment'])});
 const missing={verdict:'learn',reason:'Need compiler evidence',learningRequests:[{learningTarget:'device',question:'Does the tail compile?',hypothesis:'Target tail is constrained',requiredEvidence:'Independent target compile'}]};
 for(const item of items)await prepareHypothesis({...f,item,directory:join(f.root,'prepare',item.id),remainingSeconds:100},{qaAnswer,qaReview:async()=>missing});
 const rows=f.state.db.prepare('SELECT data FROM skill_learning_requests').all();assert.equal(rows.length,1);
 assert.deepEqual(new Set(JSON.parse(rows[0].data).backlogIds),new Set(items.map(i=>i.id)));
});


test('identical rejected guidance is not retried because analysis elapsed or archive paths changed',async t=>{
 const f=await fixture(t),{items:[item]}=await analyzeBacklog(f,{classify,review:review()});const claim=f.backlog.claim({base:'baseline'});f.backlog.finish(claim.run.id,{status:'rejected',result:{reason:'No gain'}});
 const attempt=f.state.db.prepare('SELECT id FROM attempts LIMIT 1').get().id;
 f.state.event(attempt,'verification',{correct:false,evidence:{path:f.log}});f.backlog.ingest();
 const second=await analyzeBacklog(f,{classify,review:review()});assert.equal(second.items[0].id,item.id);assert.equal(second.items[0].revision,item.revision);assert.equal(second.items[0].status,'rejected');assert.equal(f.backlog.claim({base:'baseline'}),null);
});

test('bounded diagnostic snapshots retain compiler failures at the end of long logs',async t=>{
 const f=await fixture(t);await writeFile(f.log,'setup\n'+'x'.repeat(40000)+'\nFINAL COMPILER FAILURE');
 const {items:[item]}=await analyzeBacklog(f,{classify,review:review()});const artifact=item.evidence.groups[0].occurrences[0].artifacts[0],text=await readFile(artifact.path,'utf8');
 assert.equal(artifact.truncated,true);assert.ok(text.startsWith('setup\n'));assert.ok(text.endsWith('FINAL COMPILER FAILURE'));assert.ok(text.length<40000);
});


test('completed high-priority history cannot hide queued work from reranking',async t=>{
 const f=await fixture(t),{items:[item]}=await analyzeBacklog(f,{classify,review:review()});
 for(let i=0;i<1001;i++)f.backlog.upsert({...item,key:'historical-'+i,id:undefined,proposal:{...item.proposal,changes:[{...item.proposal.changes[0],text:'Historical instruction '+i}]},priority:100,status:'rejected'});
 let submitted=[];await prioritizeBacklog(f,{session:async({prompt,validate})=>{submitted=prompt.items;return validate({priorities:prompt.items.map(i=>({id:i.id,priority:80,reason:'Actionable queued failure'}))});}});
 assert.deepEqual(submitted.map(i=>i.id),[item.id]);assert.equal(f.backlog.get(item.id).priority,80);
});


test('a passed project without a verification receipt does not qualify device learning or loop forever',async t=>{
 const f=await fixture(t),question={action:'learn',reason:'Need device evidence',question:'Does the tail compile?',hypothesis:'Tail constraints differ',requiredEvidence:'Verified target receipt',learningTarget:'device'};
 const first=await analyzeBacklog(f,{classify,review:async({groups,validate})=>validate({...question,groupIds:groups.map(g=>g.id)})});
 const source=join(f.root,'learning-source');await mkdir(source);await writeFile(join(source,'kernel.py'),'def run(x): return x');f.state.add({id:'learning',goal:'Learning',phase:'phase1',actor_seconds:60,source});const a=f.state.admit('learning',{pid:1,start:'fixture',boot:'fixture'});
 f.state.db.prepare("UPDATE attempts SET outcome='passed' WHERE id=?").run(a.id);f.state.event(a.id,'postprocess-completed',{});await mkdir(join(f.root,'attempts',a.id),{recursive:true});await writeFile(join(f.root,'attempts',a.id,'job.json'),JSON.stringify({config:{target:'a3',environment:{options:{mode:'ascend',runtime:'fixture-runtime'}}}}));
 const row=f.state.db.prepare('SELECT id,data FROM skill_learning_requests').get(),data=JSON.parse(row.data);data.findings={attemptId:a.id,projectId:'learning',outcome:'passed'};f.state.db.prepare("UPDATE skill_learning_requests SET status='verified',data=? WHERE id=?").run(JSON.stringify(data),row.id);
 let observed;await analyzeBacklog(f,{classify,review:async({groups,learning,validate})=>{observed=learning;return validate({action:'no-change',reason:'No qualifying target receipt',groupIds:groups.map(g=>g.id)});}});
 assert.equal(observed[0][1].result.status,'verified-project');assert.equal(f.backlog.get(first.items[0].id).status,'inconclusive');
 const next=await analyzeBacklog(f,{classify,review:()=>{throw Error('Must not repeat answered question');}});assert.equal(next.analyzed,0);
});


test('actual verifier receipt and captured-stream paths expose bounded diagnostic head and tail',async t=>{
 const f=await fixture(t),correctness=join(f.root,'verify-correctness.log'),performance=join(f.root,'verify-performance.log');
 await writeFile(correctness,'CORRECTNESS HEADER\n'+'x'.repeat(40000)+'\nFINAL CORRECTNESS FAILURE');await writeFile(performance,'PERFORMANCE HEADER\n'+'y'.repeat(40000)+'\nFINAL PERFORMANCE FAILURE');
 const a=f.state.db.prepare('SELECT id FROM attempts LIMIT 1').get().id;
 f.state.event(a,'verification',{passed:false,correctness:{passed:false,evidence:correctness},performance:{passed:false,evidence:performance},evidence:{correctness,performance}});
 f.state.event(a,'attempt-error',{message:'Captured streams',stdout:correctness,stderr:performance});f.state.event(a,'attempt-error',{message:'Raw output is not a filename',stdout:'/compiler/source.cpp\n'+'z'.repeat(6000)});f.backlog.ingest();
 const items=[];while(f.backlog.failures().length)items.push(...(await analyzeBacklog(f,{classify,review:review()})).items);assert.ok(items.some(i=>i.evidence.groups[0].message==='Raw output is not a filename'&&i.evidence.groups[0].occurrences[0].artifacts.length===0));
 for(const type of ['verification','attempt-error']){
  const item=items.find(i=>i.evidence.groups[0].type===type&&i.evidence.groups[0].occurrences[0].artifacts.length===2);assert.ok(item);
  const texts=await Promise.all(item.evidence.groups[0].occurrences[0].artifacts.map(a=>readFile(a.path,'utf8')));
  assert.ok(texts.some(s=>s.startsWith('CORRECTNESS HEADER')&&s.endsWith('FINAL CORRECTNESS FAILURE')));
  assert.ok(texts.some(s=>s.startsWith('PERFORMANCE HEADER')&&s.endsWith('FINAL PERFORMANCE FAILURE')));
  assert.ok(item.evidence.groups[0].occurrences[0].artifacts.every(a=>a.truncated&&a.layout==='head-and-tail'));
 }
});


test('A5 learning requires an explicit matching NPU receipt, not a passing CPU log on an A5 host',async t=>{
 for(const [kind,device,expected]of [['structured','npu:0','verified-device'],['json','npu:0','verified-device'],['json','cpu','verified-project']])await t.test(kind+' '+device,async t=>{
  const f=await fixture(t,'a5'),question={action:'learn',reason:'Need target proof',question:'Does it execute on NPU?',hypothesis:'Device executes the kernel',requiredEvidence:'Bound NPU receipt',learningTarget:'device'};
  await analyzeBacklog(f,{classify,review:async({groups,validate})=>validate({...question,groupIds:groups.map(g=>g.id)})});
  const source=join(f.root,'learning-source'),code='def run(x): return x';await mkdir(source);await writeFile(join(source,'kernel.py'),code);f.state.add({id:'learning',goal:'Learning',phase:'phase1',actor_seconds:60,source});const a=f.state.admit('learning',{pid:1,start:'fixture',boot:'fixture'});
  const receipt={correct:true,kernelSha256:createHash('sha256').update(code).digest('hex'),evidence:{device}},log=join(f.root,'learning-verifier.log');await writeFile(log,kind==='json'?'Verifier output\n'+JSON.stringify(receipt)+'\n':'Passed generic verification');
  f.state.event(a.id,'verification',{passed:true,fingerprint:'a'.repeat(64),correctness:{passed:true,exitCode:0,evidence:log,...(kind==='structured'?{receipt}:{})}});f.state.db.prepare("UPDATE attempts SET outcome='passed' WHERE id=?").run(a.id);f.state.event(a.id,'postprocess-completed',{});
  await mkdir(join(f.root,'attempts',a.id),{recursive:true});await writeFile(join(f.root,'attempts',a.id,'job.json'),JSON.stringify({config:{target:'a5',environment:{options:{mode:'ascend',runtime:'fixture-a5'}}}}));
  const row=f.state.db.prepare('SELECT id,data FROM skill_learning_requests').get(),data=JSON.parse(row.data);data.findings={attemptId:a.id,projectId:'learning',outcome:'passed'};f.state.db.prepare("UPDATE skill_learning_requests SET status='verified',data=? WHERE id=?").run(JSON.stringify(data),row.id);
  let seen;await analyzeBacklog(f,{classify,review:async({groups,learning,validate})=>{seen=learning;return validate({action:'no-change',reason:'Inspect observed receipt',groupIds:groups.map(g=>g.id)});}});
  assert.equal(seen[0][1].result.status,expected);if(expected==='verified-device')assert.equal(seen[0][1].result.deviceEvidence.device,'npu:0');
 });
});


test('rejected comparison feedback cannot enqueue an unchanged effective instruction under a new ID',async t=>{
 for(const revised of [false,true])await t.test(revised?'substantive revision':'metadata-only revision',async t=>{
  const f=await fixture(t);await analyzeBacklog(f,{classify,review:review()});
  const claim=f.backlog.claim({base:'baseline'});f.backlog.finish(claim.run.id,{status:'rejected',result:{reason:'Correctness regression'}});
  const previous={...claim.item.proposal,changes:[{...claim.item.proposal.changes[0],text:'Mask only the final partial tile.'}]};
  f.state.event(null,'skill-hypothesis-rejected',{hypothesisId:claim.item.id,groupIds:claim.item.groupIds,previousProposal:previous,objective:'correctness',reason:'Candidate failed a tail case',diagnostics:[{case:'tail',active:true,candidate:false}]});f.backlog.ingest();
  const result=await analyzeBacklog(f,{classify,review:async input=>{
   const proposal=await review(['different-id'])(input);proposal.changes[0].text=revised?'Mask both the final partial tile and its load.':'  Mask only  the final partial tile.  ';proposal.reason='Different explanation';return proposal;
  }});
  if(revised){assert.equal(result.items[0].status,'queued');assert.equal(f.backlog.claim({base:'baseline'}).item.id,result.items[0].id);}
  else{assert.equal(result.items[0].proposal.action,'no-change');assert.equal(result.items[0].status,'inconclusive');assert.equal(f.backlog.claim({base:'baseline'}),null);}
 });
});


test('every verified learning contributor is consumed for fewer updates, follow-up questions and harness findings',async t=>{
 for(const disposition of ['fewer-updates','follow-up','harness'])await t.test(disposition,async t=>{
  const f=await fixture(t),{items}=await analyzeBacklog(f,{classify,review:review(['mask','alignment'])});
  const question={learningTarget:'cpu',question:'Which boundary is wrong?',hypothesis:'The tail differs',requiredEvidence:'Independent CPU boundary result'};
  for(const item of items){const claim=f.backlog.claim({base:'baseline'});assert.equal(claim.item.id,item.id);await prepareHypothesis({...f,item,directory:join(f.root,'prepare',item.id),remainingSeconds:100},{qaAnswer,qaReview:async()=>({verdict:'learn',reason:'Need boundary fact',learningRequests:[question]})});f.backlog.finish(claim.run.id,{status:'learning'});}
  const request=f.state.db.prepare('SELECT id,data FROM skill_learning_requests').get();await completeLearning(f,request,'initial');
  const classifyResult=disposition==='harness'?async input=>{const result=await classify(input);result.classifications.forEach(c=>{c.cause='harness';c.bucket='harness';});return input.validate(result);}:classify;
  await analyzeBacklog(f,{classify:classifyResult,review:disposition==='follow-up'?async({groups,validate})=>validate({action:'learn',reason:'A narrower fact is missing',groupIds:groups.map(g=>g.id),...question,question:'Which alignment also matters?'}):review(['single-supported-revision'])});
  for(const item of items)assert.ok(f.backlog.get(item.id).evidence.learningIdentity);
  const next=await analyzeBacklog(f,{classify:()=>{throw Error('Verified results were already consumed');},review:()=>{throw Error('Do not reclassify consumed results');}});assert.equal(next.analyzed,0);
  if(disposition==='follow-up'){const follow=f.state.db.prepare("SELECT id,data FROM skill_learning_requests WHERE status='pending'").get();assert.ok(follow);await completeLearning(f,follow,'follow-up');const resumed=await analyzeBacklog(f,{classify,review:review(['follow-up-supported'])});assert.ok(resumed.items.some(i=>i.status==='queued'));}
 });
});

test('feedback permits unchanged guidance only with genuinely changed factual grounding',async t=>{
 for(const kind of ['renamed-memory','new-memory','verified-learning'])await t.test(kind,async t=>{
  const f=await fixture(t),{items:[item]}=await analyzeBacklog(f,{classify,review:review(['original-id'])});const claim=f.backlog.claim({base:'baseline'});f.backlog.finish(claim.run.id,{status:'rejected'});
  f.state.event(null,'skill-hypothesis-rejected',{hypothesisId:item.id,groupIds:item.groupIds,previousProposal:item.proposal,previousEvidence:{memoryEvidence:item.evidence.memoryEvidence,learningIdentity:null},objective:'correctness',reason:'New rejection reference is not new knowledge',diagnostics:[{case:'tail',candidate:false}]});f.backlog.ingest();
  if(kind==='verified-learning'){const group=f.backlog.failures().find(g=>g.type==='skill-hypothesis-rejected'),question={question:'Which boundary is proven?',hypothesis:'The tail is proven',requiredEvidence:'Independent CPU fact',learningTarget:'cpu'};const learning=f.backlog.upsert({key:'feedback-learning',groupIds:[group.id],objective:'correctness',base:'baseline',status:'learning',proposal:{action:'learn',...question},evidence:{groups:[group]}});const id=enqueueLearningRequest(f.state,{...question,target:'a3',backlogId:learning.id});await completeLearning(f,f.state.db.prepare('SELECT id,data FROM skill_learning_requests WHERE id=?').get(id),'feedback');}
  const result=await analyzeBacklog(f,{classify,review:async({directory,groups,validate})=>{
   await mkdir(directory,{recursive:true});await writeFile(join(directory,'retrievals.jsonl'),JSON.stringify({entries:[{collection:'memory',entry_id:'renamed-ref',text:kind==='new-memory'?'Independent new boundary evidence':'Observed boundary repair'}]})+'\n');
   return validate({...item.proposal,groupIds:groups.map(g=>g.id),memoryRefs:['memory:renamed-ref'],changes:[{...item.proposal.changes[0],id:'renamed-change',evidenceRefs:[groups[0].occurrences[0].reference]}]});
  }});
  assert.equal(result.items.some(i=>i.status==='queued'),kind!=='renamed-memory');
 });
});

test('renaming a change during original-failure reanalysis cannot bypass an identical rejection',async t=>{
 const f=await fixture(t),{items:[item]}=await analyzeBacklog(f,{classify,review:review(['old-id'])});const claim=f.backlog.claim({base:'baseline'});f.backlog.finish(claim.run.id,{status:'rejected'});
 const attempt=f.state.db.prepare('SELECT id FROM attempts LIMIT 1').get().id;f.state.event(attempt,'verification',{correct:false,evidence:{path:f.log}});f.backlog.ingest();
 const next=await analyzeBacklog(f,{classify,review:async input=>{const proposal=await review(['new-id'])(input);proposal.changes[0].text=item.proposal.changes[0].text;return proposal;}});
 assert.equal(next.items[0].id,item.id);assert.equal(next.items[0].status,'rejected');assert.equal(f.backlog.claim({base:'baseline'}),null);
});


test('bounded learning pages pass consumed high-priority history and retain pending work until the newer result is analyzed',async t=>{
 const f=await fixture(t),question={action:'learn',reason:'Need boundary evidence',question:'Which tail is safe?',hypothesis:'Tail differs',requiredEvidence:'Independent CPU result',learningTarget:'cpu'};
 const initial=await analyzeBacklog(f,{classify,review:async({groups,validate})=>validate({...question,groupIds:groups.map(g=>g.id)})}),item=initial.items[0];
 const row=f.state.db.prepare('SELECT id,data FROM skill_learning_requests').get();await completeLearning(f,row,'page-proof');
 await analyzeBacklog(f,{classify,review:async({groups,validate})=>validate({action:'no-change',reason:'Fact already represented',groupIds:groups.map(g=>g.id)})});
 const consumed=f.backlog.get(item.id),linked=[];
 for(let i=0;i<101;i++)linked.push(f.backlog.upsert({...consumed,key:'consumed-'+i,id:undefined,status:'learning',priority:90,proposal:{...question,question:'Consumed question '+i}}).id);
 const newer=f.backlog.upsert({...consumed,key:'newer-learning',id:undefined,status:'learning',priority:1,proposal:{...question,question:'A newer boundary question'},evidence:{...consumed.evidence,learningIdentity:null}});linked.push(newer.id);
 const data=JSON.parse(f.state.db.prepare('SELECT data FROM skill_learning_requests WHERE id=?').get(row.id).data);data.backlogIds=[item.id,...linked];f.state.db.prepare('UPDATE skill_learning_requests SET data=? WHERE id=?').run(JSON.stringify(data),row.id);
 f.backlog.meta('learning-scan',null);let calls=0;
 for(let page=0;page<20&&!calls;page++){
  const result=await analyzeBacklog(f,{classify,review:async input=>{calls++;return review(['newer-supported'])(input);}});
  if(!calls){assert.equal(result.analyzed,0);assert.equal(f.backlog.meta('learning-pending'),true);}
 }
 assert.equal(calls,1);assert.equal(f.backlog.get(newer.id).status,'queued');assert.equal(f.backlog.meta('learning-pending'),false);
});
test('ranking cache replay normalizes legacy arrays and persists an idempotent object',async t=>{
 const f=await fixture(t);await analyzeBacklog(f,{classify,review:review(['cached-rank'])});
 let cached,providerCalls=0;
 const session=async({prompt,validate})=>{
  if(cached)return {value:validate(JSON.parse(cached))};
  providerCalls++;const old=prompt.items.map(item=>({id:item.id,priority:95,reason:'Severe observed failure'}));
  cached=JSON.stringify(validate(old));return {value:JSON.parse(cached)};
 };
 await prioritizeBacklog(f,{session});await prioritizeBacklog(f,{session});
 assert.equal(providerCalls,1);assert.equal(Array.isArray(JSON.parse(cached).priorities),true);assert.equal(f.backlog.list()[0].priority,95);
 await assert.rejects(prioritizeBacklog(f,{session:async({validate})=>validate({priorities:[]})}));assert.equal(f.backlog.list()[0].priority,95);
});

test('classification spending does not clip independent proposal allowance',async t=>{
 const f=await fixture(t);let milliseconds=0,proposalSeconds;
 const result=await analyzeBacklog({...f,seconds:120,remainingSeconds:2000},{now:()=>milliseconds,
  classify:async input=>{milliseconds+=105000;return classify(input);},
  review:async input=>{proposalSeconds=input.seconds;milliseconds+=30000;return review()(input);}});
 assert.equal(proposalSeconds,1800);assert.equal(result.items[0].preparation.proposalElapsedSeconds,30);
 assert.equal(result.elapsedSeconds,135);
});

test('resuming proposal reuses accepted classifications and charges only remaining writing time',async t=>{
 const f=await fixture(t);let milliseconds=0,calls=0;
 const dependencies={now:()=>milliseconds,classify:async input=>{calls++;return classify(input);}};
 await assert.rejects(analyzeBacklog({...f,remainingSeconds:2000},{...dependencies,review:async()=>{milliseconds+=25000;throw Error('Provider disconnected');}}),/disconnected/);
 const stages=Object.values(f.backlog.meta('analysis-stages'));assert.equal(stages[0].stage,'proposal');assert.equal(stages[0].status,'retry');
 let allowed;const result=await analyzeBacklog({...f,remainingSeconds:2000},{...dependencies,review:async input=>{allowed=input.seconds;milliseconds+=10000;return review()(input);}});
 assert.equal(calls,1);assert.equal(allowed,1775);assert.equal(result.items[0].preparation.proposalElapsedSeconds,35);
 assert.equal(Object.values(f.backlog.meta('analysis-stages'))[0].status,'complete');
});

test('exhausted contract stages remain visible and do not reopen native sessions automatically',async t=>{
 const f=await fixture(t);let calls=0;
 await assert.rejects(analyzeBacklog(f,{classify:async()=>{calls++;throw Object.assign(Error('Schema contract rejected'),{reviewSession:{kind:'rejected-submissions'}});}}),/contract rejected/);
 const resumed=await analyzeBacklog(f,{classify:()=>{throw Error('Do not replay blocked session');}});
 assert.equal(resumed.analyzed,0);assert.equal(calls,1);assert.equal(f.backlog.failures().length,1);
 assert.equal(Object.values(f.backlog.meta('analysis-stages'))[0].status,'blocked');
 // A fresh explicit queue epoch is the operator-controlled retry boundary.
 f.backlog.meta('queue-epoch','fresh');const fresh=await analyzeBacklog(f,{classify,review:review()});assert.equal(fresh.items.length,1);
});

test('classification elapsed exhaustion is retained without granting a fresh allowance',async t=>{
 const f=await fixture(t);let milliseconds=0;
 await assert.rejects(analyzeBacklog(f,{now:()=>milliseconds,classify:async()=>{milliseconds+=120000;throw Object.assign(Error('Elapsed'),{reviewSession:{kind:'elapsed'}});}}),/Elapsed/);
 assert.equal(Object.values(f.backlog.meta('analysis-stages'))[0].status,'blocked');
 assert.equal((await analyzeBacklog(f,{classify:()=>{throw Error('Must not retry');}})).analyzed,0);
});

test('new urgent evidence does not reset interrupted writing or repeat accepted classification',async t=>{
 const f=await fixture(t);let milliseconds=0,originalId,originalDirectory;
 await assert.rejects(analyzeBacklog({...f,remainingSeconds:3000},{now:()=>milliseconds,classify,review:async input=>{originalId=input.groups[0].id;originalDirectory=input.directory;milliseconds+=50000;throw Error('Disconnected');}}),/Disconnected/);
 const a=f.state.db.prepare('SELECT id FROM attempts LIMIT 1').get().id;f.state.event(a,'attempt-error',{message:'New independent compile failure'});f.backlog.ingest();
 const classified=[],allowances=[];
 await analyzeBacklog({...f,remainingSeconds:3000},{now:()=>milliseconds,classify:async input=>{classified.push(...input.groups.map(g=>g.id));return classify(input);},review:async input=>{if(input.groups[0].id===originalId){assert.equal(input.directory,originalDirectory);allowances.push(input.seconds);}return review()(input);}});
 assert.ok(!classified.includes(originalId));assert.deepEqual(allowances,[1750]);assert.equal(f.backlog.failures().length,0);
});

test('blocked first two hundred groups do not hide later unclassified evidence',async t=>{
 const f=await fixture(t);for(let i=0;i<201;i++)f.state.event(null,'generation-rejected',{message:'Independent diagnostic '+i});
 while(f.backlog.ingest({limit:100}).progress){}
 await assert.rejects(analyzeBacklog({...f,limit:200},{classify:async()=>{throw Object.assign(Error('Contract failed'),{reviewSession:{kind:'rejected-submissions'}});}}),/Contract failed/);
 let classified=0;const result=await analyzeBacklog(f,{classify:async input=>{classified=input.groups.length;return classify(input);},review:review()});
 assert.equal(classified,2);assert.equal(result.items.length,2);
});

test('an unclosed durable writing marker blocks rather than renewing unknown elapsed spending',async t=>{
 const f=await fixture(t);let directory;
 await assert.rejects(analyzeBacklog(f,{classify,review:async input=>{directory=input.directory;const ledger=JSON.parse(await readFile(join(directory,'writing-spend.json'),'utf8'));assert.equal(ledger.running,true);throw Error('Interrupted');}}),/Interrupted/);
 const path=join(directory,'writing-spend.json'),ledger=JSON.parse(await readFile(path,'utf8'));assert.equal(ledger.running,false);
 await writeFile(path,JSON.stringify({elapsedSeconds:17,running:true}));
 const result=await analyzeBacklog(f,{classify:()=>{throw Error('Do not repeat');},review:()=>{throw Error('Unknown spend must not grant work');}});
 assert.equal(result.items.length,0);assert.equal(f.backlog.failures().length,1);
 const stage=Object.values(f.backlog.meta('analysis-stages'))[0];assert.equal(stage.status,'blocked');assert.equal(stage.elapsedSeconds,17);
});

test('classification spending with unknown elapsed custody is blocked without a fresh model grant',async t=>{
 const f=await fixture(t);let directory;
 await assert.rejects(analyzeBacklog(f,{classify:async input=>{directory=input.directory;throw Error('Disconnected');}}),/Disconnected/);
 await writeFile(join(directory,'..','classification-spend.json'),JSON.stringify({elapsedSeconds:12,running:true}));
 await assert.rejects(analyzeBacklog(f,{classify:()=>{throw Error('Must not call model');}}),/custody is unknown/);
 const stage=Object.values(f.backlog.meta('analysis-stages'))[0];assert.equal(stage.status,'blocked');assert.ok(stage.elapsedSeconds>=12);
});

test('interrupted proposal has periodic durable elapsed checkpoints before completion',async t=>{
 const f=await fixture(t);
 await analyzeBacklog(f,{classify,review:async input=>{
  await new Promise(resolve=>setTimeout(resolve,1100));
  const ledger=JSON.parse(await readFile(join(input.directory,'writing-spend.json'),'utf8'));
  assert.equal(ledger.running,true);assert.ok(ledger.elapsedSeconds>=.5);
  return review()(input);
 }});
});
