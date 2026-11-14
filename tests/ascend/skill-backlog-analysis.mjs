import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {SkillBacklog} from '../../controller/skill-backlog.mjs';
import {stages} from '../../controller/workflow.mjs';
import {analyzeBacklog,prioritizeBacklog,prepareHypothesis} from '../../cli/skill-backlog-analysis.mjs';
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'backlog-analysis-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const version=join(root,'skills','versions','baseline');await mkdir(version,{recursive:true});
 const entries=['global-kdb',...stages].map(scope=>({id:scope+'-baseline',scope,text:'Existing '+scope,critical:true,negative:false})),map={};
 for(const entry of entries){map[entry.scope]=entry.scope+'.md';await writeFile(join(version,map[entry.scope]),entry.text);}
 await writeFile(join(version,'skill-map.json'),JSON.stringify(map));await writeFile(join(version,'guidance.json'),JSON.stringify(entries));
 const config={format:1,root,target:'a3',actor:{provider:'fixture',model:'fixture'},skills:{activeVersion:'baseline'}};
 const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
 state.add({id:'p1',phase:'phase1',actor_seconds:60,goal:'Boundary'});const a=state.admit('p1',{pid:1,start:'fixture',boot:'fixture'});
 state.event(a.id,'job-admitted',{kind:'original',mode:'phase1'});const log=join(root,'verifier.log');await writeFile(log,'Independent tail mismatch');
 state.event(a.id,'verification',{correct:false,evidence:{path:log}});const backlog=new SkillBacklog(state,{target:'a3'});backlog.ingest();
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
 f.state.event(learned.id,'verification',{passed:true,fingerprint:'a'.repeat(64),correctness:{passed:true,exitCode:0,evidence:learnedLog}});f.state.db.prepare("UPDATE attempts SET outcome='passed' WHERE id=?").run(learned.id);f.state.event(learned.id,'postprocess-completed',{});
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
 for(let i=0;i<1001;i++)f.backlog.upsert({...item,key:'historical-'+i,id:undefined,priority:100,status:'rejected'});
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
