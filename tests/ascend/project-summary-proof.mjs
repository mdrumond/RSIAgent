import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,readdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {State} from '../../controller/state.mjs';
import {projectSummary,projectSummaries,validateClaims,validateClaimedResult,claimEvidence} from '../../cli/project-summary.mjs';
import {findingsFor} from '../../cli/findings.mjs';
import {frozenEvidencePacket} from '../../cli/skill-review-evidence.mjs';
import {nativeClassify,reviewSession} from '../../cli/skill-review-native.mjs';
import {runProposalQa} from '../../cli/skill-review-qa.mjs';
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'project-summary-proof-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());state.add({id:'p',goal:'test',actor_seconds:60});
 const attempt=state.admit('p',{});return {root,state,attempt};
}
const proof=(fact,text,path='reason')=>({path,text,kind:'observation',proofs:[{reference:fact.reference,quote:fact.text,factId:fact.factId,value:fact.value}]});
test('all project attempts and verifications survive final failure and diagnostic tail limits',async t=>{
 const f=await fixture(t);f.state.db.prepare('INSERT INTO attempts(id,project,outcome,created) VALUES (?,?,?,?)').run('older','p','failed','2000-01-01');
 f.state.event('older','verification',{passed:true,correctness:{passed:true}});
 for(let i=0;i<15;i++)f.state.event(f.attempt.id,'verification',{passed:false,correctness:{passed:false}});
 f.state.finish(f.attempt.id,'failed');const s=await projectSummary({state:f.state,projectId:'p'});
 assert.equal(s.projectStatus,'failed');assert.equal(s.attempts.length,2);assert.equal(s.verifications.length,16);assert.equal(s.facts.find(x=>x.reference.endsWith(':passingVerificationCount')).value,1);assert.ok(s.eventCutoff>0);
 const groups=[{occurrences:[{projectId:'p',attemptId:f.attempt.id}]}];assert.deepEqual((await projectSummaries({root:f.root},groups))[0],s);
});
test('aggregate gain, candidate identity and exact artifact binding remain distinct from case ratios',async t=>{
 const f=await fixture(t),path=join(f.root,'performance.log'),result={passed:true,geometricMeanSpeedup:1.86,cases:[{id:'one',speedup:1.638}],candidateHash:'same-candidate',caseHash:'same-cases',runtimeFingerprint:'runtime'};
 const text=JSON.stringify(result)+'\nwarning after receipt\n';await writeFile(path,text);
 f.state.event(f.attempt.id,'verification',{passed:true,correctness:{passed:true},performance:{passed:true,evidence:path,evidence_sha256:createHash('sha256').update(text).digest('hex')}});
 const s=await projectSummary({state:f.state,projectId:'p'});assert.equal(s.measurements[0].geometricMeanSpeedup,1.86);assert.equal(s.measurements[0].candidateHash,'same-candidate');assert.equal(s.measurements[0].sourceHash,null);assert.equal(s.measurements[0].verified,true);
 const fact=s.facts.find(x=>x.reference.endsWith(':geometricMeanSpeedup'));assert.throws(()=>validateClaims([{...proof(fact,'Wrong gain'),proofs:[{reference:fact.reference,quote:fact.text,factId:fact.factId,value:1.638}]}],s.facts),/Structured fact assertion/);
});
test('missing, changed and contradictory artifact evidence cannot become verified measurements',async t=>{
 const f=await fixture(t),path=join(f.root,'performance.log');await writeFile(path,JSON.stringify({passed:false,geometricMeanSpeedup:2}));
 f.state.event(f.attempt.id,'verification',{correctness:{passed:true},performance:{passed:true,evidence:path}});
 f.state.event(f.attempt.id,'verification',{passed:true,performance:{passed:true,evidence:path,evidence_sha256:'wrong'}});
 f.state.event(f.attempt.id,'verification',{passed:true,performance:{passed:true,evidence:join(f.root,'absent')}});
 const s=await projectSummary({state:f.state,projectId:'p'});assert.equal(s.verifications[0].passed,null);assert.equal(s.measurements[0].verified,false);assert.equal(s.unknowns.length,3);assert.equal(s.facts.find(x=>x.reference.endsWith(':unknownVerificationCount')).value,1);
});
test('legacy cached findings acquire fresh project facts and old narrative is retained',async t=>{
 const f=await fixture(t),path=join(f.root,'attempt');await mkdir(path);await writeFile(join(path,'findings.json'),JSON.stringify({findings:'Legacy narrative',useful:false,evidence:[]}));
 f.state.event(f.attempt.id,'project-invalid',{evidence:'baseline.log',message:'bad baseline'});
 const options={state:f.state,job:{attempt:f.attempt,kind:'original'},path,passed:false};const a=await findingsFor(options);
 assert.equal(a.projectSummary.verifications.length,0);assert.equal((await readdir(join(path,'findings-history'))).length,1);
 f.state.event(f.attempt.id,'verification',{passed:true});const b=await findingsFor(options);assert.notEqual(a.projectSummary.identity,b.projectSummary.identity);assert.equal(b.projectSummary.verifications.length,1);
 assert.deepEqual(JSON.parse(await readFile(join(path,'project-summary.json'),'utf8')),b.projectSummary);
});
test('proof admission rejects wrong references, crossing omitted bytes, absent claims and typed contradictions',async t=>{
 const path=join(tmpdir(),'proof-fragment-'+Date.now());await writeFile(path,'HEAD----omitted----TAIL');t.after(()=>rm(path,{force:true}));
 const packet=await frozenEvidencePacket([{occurrences:[{reference:'failure',artifacts:[{path}]}]}],{limit:8}),a=packet.artifacts[0];assert.deepEqual(a.segments,['HEAD','TAIL']);
 const record={reference:'artifact',text:a.text,segments:a.segments};
 assert.throws(()=>validateClaims([{text:'bad',kind:'observation',proofs:[{reference:'artifact',quote:'ADTA'}]}],[record]),/Quote does not match/);
 assert.throws(()=>validateClaims([{text:'bad',kind:'observation',proofs:[{reference:'other',quote:'HEAD'}]}],[record]),/Unknown proof reference/);
 assert.throws(()=>validateClaimedResult({reason:'uncited'},[record]),/proof-bearing claims/);
 const result={reason:'observed head',claims:[{path:'reason',text:'observed head',kind:'observation',proofs:[{reference:'artifact',quote:'HEAD'}]}]};assert.equal(validateClaimedResult(result,[record]),result);
 assert.throws(()=>validateClaimedResult({...result,reason:'different uncited statement'},[record]),/Missing exact proof-bearing claim/);
});
async function provider(t,plan){
 let requests=0;const server=createServer(async(req,res)=>{let raw='';for await(const c of req)raw+=c;const body=JSON.parse(raw),item=plan(requests++,body);res.writeHead(200,{'content-type':'text/event-stream'});
  const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'proof',object:'chat.completion.chunk',created:1,model:'proof',choices:[{index:0,delta,finish_reason}]})+'\n\n');
  if(item){const [name,args]=item;frame({role:'assistant',tool_calls:[{index:0,id:'c'+requests,type:'function',function:{name,arguments:JSON.stringify(args)}}]});frame({},'tool_calls');}else{frame({role:'assistant',content:'Done'});frame({},'stop');}res.end('data: [DONE]\n\n');});
 server.listen(0,'127.0.0.1');await once(server,'listening');t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));});
 const key='PROJECT_PROOF_FIXTURE_KEY',old=process.env[key];process.env[key]='fixture';t.after(()=>{if(old===undefined)delete process.env[key];else process.env[key]=old;});
 return {requests:()=>requests,profile:{provider:'proof-fixture',model:'proof',credential_env:key,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'proof',name:'proof',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}}};
}
test('native classification gets complete facts inline and repairs rejected proof in same Pi session',async t=>{
 const f=await fixture(t);f.state.event(f.attempt.id,'verification',{passed:true});f.state.finish(f.attempt.id,'failed');
 const summary=await projectSummary({state:f.state,projectId:'p'}),fact=summary.facts.find(x=>x.reference.endsWith(':passingVerificationCount'));
 const value={cause:'unknown',reason:'One passing verification exists',coverage:'unknown',guidanceIds:[],evidenceRefs:['failure:1'],gap:'unknown',claims:[proof(fact,'One passing verification exists')]};
 const p=await provider(t,(n,body)=>{if(n===0)return ['read',{path:'input.json'}];if(n===1)return ['submit_review',{result:{...value,claims:[{...value.claims[0],proofs:[{...value.claims[0].proofs[0],value:0}]}]}}];if(n===2){assert.match(JSON.stringify(body.messages),/Structured fact assertion/);return ['submit_review',{result:value}];}});
 const directory=join(f.root,'review'),input={directory,config:{root:f.root,actor:p.profile},group:{id:'failure',occurrences:[{reference:'failure:1',projectId:'p',attemptId:f.attempt.id,message:'failed attempt'}]},entries:[],seconds:15,validate:r=>r};
 const out=await nativeClassify(input);assert.equal(out.value.reason,value.reason);assert.equal(p.requests(),4);
 const context=JSON.parse(await readFile(join(directory,'input.json'),'utf8'));assert.equal(context.projectSummaries[0].projectStatus,'failed');assert.equal(context.proofEvidence.find(x=>x.factId===fact.factId).value,1);
 const session=JSON.parse(await readFile(join(directory,'native-session.json'),'utf8'));const rows=(await readFile(session.path,'utf8')).split('\n').filter(Boolean).map(JSON.parse);assert.equal(rows.filter(x=>x.type==='session').length,1);
 await nativeClassify(input);assert.equal(p.requests(),4);
});
test('cached legacy submission is rejected and repaired rather than bypassing proof mode',async t=>{
 const f=await fixture(t),directory=join(f.root,'legacy');await mkdir(directory);await writeFile(join(directory,'submission.json'),JSON.stringify({reason:'old uncited'}));
 const record={reference:'r',text:'observed'},value={reason:'observed',claims:[{path:'reason',text:'observed',kind:'observation',proofs:[{reference:'r',quote:'observed'}]}]};
 const p=await provider(t,n=>n===0?['submit_review',{result:value}]:undefined);
 const out=await reviewSession({directory,config:{actor:p.profile},prompt:{entries:[{id:'r',text:'observed'}]},seconds:10,proofRequired:true,validate:x=>x});assert.equal(out.value.reason,'observed');assert.equal(p.requests(),2);assert.equal(JSON.parse((await readFile(join(directory,'submission-rejections.jsonl'),'utf8')).trim().split('\n')[0]).cached,true);
});
test('legacy approved QA cannot bypass proof-required answer and semantic re-review',async t=>{
 const f=await fixture(t),directory=join(f.root,'qa');await mkdir(directory);const proposal={changes:[],memoryRefs:[]},input={directory,proposal,entries:[],groups:[],target:'a3',validateProposal:x=>x};
 const {qaIdentity}=await import('../../cli/skill-review-qa.mjs');await writeFile(join(directory,'qa.json'),JSON.stringify({format:2,protocol:'skill-qa-v2',inputIdentity:qaIdentity(input),seconds:300,elapsedSeconds:25,verdict:'approved',rounds:[{answers:{old:true}}]}));
 let answered=0,reviewed=0;
 const result=await runProposalQa(input,{qaAnswer:async()=>{answered++;return{answers:Object.fromEntries(['action','mechanism','example','evidence','objective'].map(k=>[k,{text:'unknown',refs:[]}])),claims:[{proof:'new'}]};},qaReview:async options=>{reviewed++;assert.deepEqual(options.answerClaims,[{proof:'new'}]);return{verdict:'inconclusive',reason:'No evidence'};}});
 assert.equal(answered,1);assert.equal(reviewed,1);assert.equal(result.verdict,'inconclusive');assert.ok(result.elapsedSeconds>=25);assert.equal(result.proofUpgrade.verdict,'approved');
});

test('new project facts invalidate approved QA without resetting its elapsed allocation',async t=>{
 const f=await fixture(t),directory=join(f.root,'qa-facts');await mkdir(directory);let clock=100000,calls=0;
 const base={directory,proposal:{changes:[],memoryRefs:[]},entries:[],groups:[],target:'a3',validateProposal:x=>x};
 const ops={now:()=>clock,qaAnswer:async()=>{calls++;clock+=1000;return{answers:Object.fromEntries(['action','mechanism','example','evidence','objective'].map(k=>[k,{text:'Observed count',refs:[]}])),claims:[{proof:'checked'}]};},qaReview:async options=>{clock+=1000;assert.equal(options.projectSummaries[0].facts[0].value,calls-1);return{verdict:'approved',reason:'Checked against facts'};}};
 const a=await runProposalQa({...base,projectSummaries:[{facts:[{factId:'count',value:0}]}]},ops);
 const b=await runProposalQa({...base,projectSummaries:[{facts:[{factId:'count',value:1}]}]},ops);
 assert.equal(calls,2);assert.notEqual(a.projectFactsIdentity,b.projectFactsIdentity);assert.ok(b.elapsedSeconds>=a.elapsedSeconds+2);assert.equal(b.proofUpgrade.verdict,'approved');
});

test('structured QA issue prose and inline citations cannot bypass proof admission',()=>{
 const fact={reference:'count',factId:'count',text:'1',value:1},records=[fact];
 const result={verdict:'revise',reason:'Count is one',issues:[{path:'proposal.changes.0.text',problem:'The draft says zero',description:'One was recorded',reference:'count',quote:'1'}],claims:[proof(fact,'Count is one')]};
 assert.throws(()=>validateClaimedResult(result,records),error=>error.message.includes('issues.0.problem')&&error.message.includes('issues.0.description'));
 result.claims.push(proof(fact,result.issues[0].problem,'issues.0.problem'),proof(fact,result.issues[0].description,'issues.0.description'));
 result.issues[0].quote='0';assert.throws(()=>validateClaimedResult(result,records),/Quote does not match/);
 result.issues[0].quote='1';assert.throws(()=>validateClaimedResult(result,records),/Structured fact assertion/);
 Object.assign(result.issues[0],{factId:'count',value:1});assert.equal(validateClaimedResult(result,records),result);
 result.claims.push(proof(fact,'revise','verdict'));assert.throws(()=>validateClaimedResult(result,records),/Claim path\/text does not match/);
});

test('reviewer can prove what the submitted draft and answers actually say',()=>{
 const records=claimEvidence({prompt:{proposal:{text:'No passes'},answers:{evidence:{text:'No passes'}},answerClaims:[{text:'No passes'}]}});
 const claims=[{text:'The submitted draft says no passes',kind:'observation',proofs:[{reference:'input:proposal',quote:'No passes'}]}];
 assert.equal(validateClaims(claims,records),claims);
 assert.throws(()=>validateClaims([{...claims[0],proofs:[{reference:'input:answers',quote:'One pass'}]}],records),/Quote does not match/);
});
