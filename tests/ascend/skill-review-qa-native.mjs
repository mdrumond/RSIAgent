import test from 'node:test';
import {createHash} from 'node:crypto';
import {Type} from 'typebox';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {nativeQaAnswer,nativeQaReview,nativeQaPanelAudit,reviewSession,REVIEW_CONTRACT_VERSION} from '../../cli/skill-review-native.mjs';
import {runProposalQa,qaTopics,qaIdentity} from '../../cli/skill-review-qa.mjs';
import {scriptedReview} from './helpers/scripted-review.mjs';
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'qa-native-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const proposalDirectory=join(root,'writer');await mkdir(proposalDirectory);
 await writeFile(join(proposalDirectory,'retrievals.jsonl'),JSON.stringify({entries:[{collection:'memory',entry_id:'seen',text:'Observed boundary repair'},{collection:'memory',entry_id:'irrelevant',text:'Other fact'}]})+'\n');
 return {root,input:{directory:join(root,'0-qaAnswer'),proposalDirectory,seconds:120,config:{actor:{provider:'fixture',model:'fixture'}},
  proposal:{changes:[{text:'Mask the last tile',evidenceRefs:['failure:1']}],memoryRefs:['memory:seen']},
  entries:[{id:'baseline',text:'Baseline'}],objective:'correctness',target:'a3',groups:[{id:'group',occurrences:[{reference:'failure:1',message:'Tail mismatch'},{reference:'failure:unrelated',message:'Different failure'}],linkedDiagnostics:[{reference:'failure:root',message:'Actual baseline API error'}]}]}};
}
test('native writer uses read-only supplied retrieved evidence and invokes admission validation',async t=>{
 const {input}=await fixture(t),answer={answers:{action:{text:'Mask tail',refs:['memory:seen']}}};let validated=0;
 input.validate=value=>{validated++;assert.equal(value,answer);return {...value,validated:true};};
 const result=await nativeQaAnswer(input,{session:async options=>{
  assert.equal(options.knowledge,false);assert.equal(options.directory,input.directory);
  assert.deepEqual(options.prompt.memory,[{collection:'memory',entry_id:'seen',text:'Observed boundary repair'}]);
  assert.deepEqual(options.evidenceGroups[0].occurrences.map(o=>o.reference),['failure:1']);
  assert.deepEqual(options.evidenceGroups[0].linkedDiagnostics.map(o=>o.reference),['failure:root']);
  assert.equal(options.seconds,120);return {value:options.validate(answer)};
 }});
 assert.equal(validated,1);assert.equal(result.value.validated,true);
});
test('independent reviewer receives final changed skill text and its fresh round directory',async t=>{
 const {root,input}=await fixture(t),answers={action:{text:'Explain the final boundary',refs:['memory:seen']}};
 const changed={...input.proposal,changes:[{text:'Mask only the partial final tile; retain interior tiles',evidenceRefs:['failure:1']}]};
 const reviewed=[];
 for(const index of [0,1]){
  const directory=join(root,index+'-qaReview');
  await nativeQaReview({...input,directory,proposal:changed,answers,validate:value=>value},{session:async options=>{
   reviewed.push(options.directory);assert.equal(options.knowledge,false);assert.equal(options.prompt.proposal,changed);assert.equal(options.prompt.answers,answers);
   assert.deepEqual(options.prompt.baseline,input.entries);assert.equal(options.prompt.objective,'correctness');assert.equal(options.prompt.target,'a3');
   return {value:options.validate({verdict:'approved',reason:'Final text includes the boundary'})};
  }});
 }
 assert.equal(new Set(reviewed).size,2);
});
test('explicit frozen memory evidence replaces archived retrievals without live lookup',async t=>{
 const {input}=await fixture(t),memoryEvidence=[{ref:'memory:seen',text:'Frozen original observation',sha256:'fixture-digest'}];
 await nativeQaReview({...input,memoryEvidence,validate:value=>value},{session:async options=>{
  assert.deepEqual(options.prompt.memory,memoryEvidence);assert.equal(options.knowledge,false);
  return {value:options.validate({verdict:'learn',reason:'Missing device boundary',learningRequests:[]})};
 }});
});
test('QA sessions can inspect verified learning used to revise the skill',async t=>{
 const {input,root}=await fixture(t),source=join(root,'learning.py');await writeFile(source,'assert 2 + 2 == 4');
 const learning=[['question',{result:{status:'verified-cpu',project:{source},evidence:[{log:source}]}}]];
 await nativeQaReview({...input,learning,validate:value=>value},{session:async options=>{
  assert.deepEqual(options.prompt.learning,learning);assert.ok(options.allowedPaths.includes(source));
  return {value:options.validate({verdict:'approved',reason:'Evidence inspected'})};
 }});
});


test('QA can inspect copied device learning evidence without presenting it as CPU evidence',async t=>{
 const {input,root}=await fixture(t),source=join(root,'device-learning');await writeFile(source,'actual device observation');
 const learning=[['device-question',{result:{status:'verified-device',binding:{target:'a5',runtime:'fixture'},project:{source},evidence:[]}}]];
 await nativeQaReview({...input,learning,validate:value=>value},{session:async options=>{assert.ok(options.allowedPaths.includes(source));assert.equal(options.prompt.learning[0][1].result.status,'verified-device');return{value:options.validate({verdict:'approved',reason:'Copied device evidence supports instruction'})};}});
});
test('native cached submissions replay validation without opening a provider session',async t=>{
 const {reviewSession}=await import('../../cli/skill-review-native.mjs');
 const {root}=await fixture(t),directory=join(root,'ranking');await mkdir(directory);
 await writeFile(join(directory,'submission.json'),JSON.stringify([{id:'one',priority:80,reason:'Observed recurrence'}]));await writeFile(join(directory,'review-contract.json'),JSON.stringify({version:REVIEW_CONTRACT_VERSION,schema:Type.Unknown()}));
 let validations=0;
 const result=await reviewSession({directory,config:{},prompt:{},seconds:1,validate:value=>{validations++;return {priorities:Array.isArray(value)?value:value.priorities};}});
 assert.equal(result.recovered,true);assert.equal(validations,1);assert.equal(result.value.priorities[0].priority,80);assert.equal(JSON.parse(await readFile(join(directory,'submission.json'))).priorities[0].id,'one');
});

test('native reviewer validates typed summary repair targets against the current evidence summary',async t=>{
 const {input}=await fixture(t);const result={verdict:'revise',reason:'Recheck pointer',issues:[{problem:'Wrong line',summaryItemId:'known-summary'}]};
 await nativeQaReview({...input,evidenceSummary:{items:[{id:'known-summary',text:'Observed failure',kind:'observation',proofs:[]}]},sourceCatalogue:{directory:input.directory,indexFile:'unused',facts:[],records:[],pages:[],sources:[]},validate:x=>x},{session:async options=>{
  assert.equal(options.validate(result),result);
  assert.throws(()=>options.validate({...result,issues:[{problem:'Wrong line',summaryItemId:'invented'}]}),/Unknown summaryItemId/);
  return {value:result};
 }});
});
test('native reviewer receives all previous issues without dropping them from compact evidence mode',async t=>{
 const {input}=await fixture(t),previousReview={verdict:'revise',reason:'Multiple defects',issues:[{problem:'Direction'},{problem:'Unsupported timing'}]};
 await nativeQaReview({...input,previousReview,sourceCatalogue:{directory:'/copied-primary',indexFile:'/copied-primary/catalogue.json',facts:[],records:[]},evidenceSummary:{items:[]},validate:x=>x},{session:async options=>{
  assert.deepEqual(options.prompt.previousReview,previousReview);assert.deepEqual(options.allowedPaths,['/copied-primary']);return {value:{verdict:'inconclusive',reason:'Fixture'}};
 }});
});
test('session input exposes the same canonical paths used for filesystem admission',async t=>{
 const {root}=await fixture(t),directory=join(root,'cached'),primary=join(root,'primary');await mkdir(directory);await mkdir(primary);const source=join(primary,'source.txt');await writeFile(source,'Observed fact');
 await writeFile(join(directory,'submission.json'),JSON.stringify({ok:true}));await writeFile(join(directory,'review-contract.json'),JSON.stringify({version:REVIEW_CONTRACT_VERSION,schema:Type.Unknown()}));const {reviewSession}=await import('../../cli/skill-review-native.mjs');
 const out=await reviewSession({directory,config:{},prompt:{},allowedPaths:[primary,source],validate:x=>x});assert.equal(out.recovered,true);
 const input=JSON.parse(await readFile(join(directory,'input.json'),'utf8'));
 assert.deepEqual(input.filesystemAccess.allowed,[{path:directory,directory:true},{path:primary,directory:true},{path:source,directory:false}]);assert.deepEqual(input.filesystemAccess.requiredReads,[]);
});

for(const kind of ['available','missing','hash-mismatch','invalid-reference','out-of-scope','forged-proof'])test('native QA scoped artifact reference: '+kind,async t=>{
 const {root,input}=await fixture(t);
 const artifact=join(root,'verification.txt'),body='Verifier reported a numerical mismatch\n';
 const sha256=createHash('sha256').update(body).digest('hex');
 if(kind!=='missing')await writeFile(artifact,kind==='hash-mismatch'?'Changed verification\n':body);
 const evidence={path:artifact,sha256};
 const observations=input.groups[0].occurrences;
 (kind==='out-of-scope'?observations[1]:input.groups[0].linkedDiagnostics[0]).artifacts=[evidence];
 const artifactRef='artifact:'+sha256,ref=kind==='invalid-reference'?'artifact:'+'0'.repeat(64):artifactRef;
 const artifactProof=kind==='available'||kind==='forged-proof';
 const answer={
  answers:Object.fromEntries(qaTopics.map(topic=>[topic,{text:'Evidence supports this boundary',refs:['baseline',ref]}])),
  claims:qaTopics.map(topic=>({path:'answers.'+topic+'.text',kind:'observation',proofs:[{
   reference:artifactProof?artifactRef:'baseline',startLine:1,endLine:1,
   ...(kind==='forged-proof'?{quote:'Invented verifier output'}:{}),
  }]})),
 };
 const provider=await scriptedReview(t,root,(_message,n)=>n===0?{name:'submit_review',args:{result:answer}}:null);
 let sessionInput,reviewed=0;
 const report=await runProposalQa({...input,config:{actor:provider.profile},projectSummaries:[],learningProofEvidence:[],validateProposal:value=>value},{
  qaAnswer:async args=>(await nativeQaAnswer(args,{session:async options=>{
   sessionInput=join(options.directory,'input.json');return reviewSession(options);
  }})).value,
  qaReview:async()=>{reviewed++;return {verdict:'approved',reason:'Fixture reviewed the accepted answers'};},
 });
 const native=JSON.parse(await readFile(sessionInput,'utf8'));
 assert.equal(native.proofSelectors.some(selector=>selector.reference===artifactRef),['available','invalid-reference','forged-proof'].includes(kind));
 if(kind==='available'){
  assert.equal(report.verdict,'approved');assert.equal(reviewed,1);
  assert.deepEqual(report.rounds[0].answers.action.refs,['baseline',artifactRef]);
  const proof=report.rounds[0].claims[0].proofs[0];
  assert.equal(proof.sourceSha256,sha256);assert.equal(proof.quote,body.trim());
 }else{
  assert.equal(report.verdict,'inconclusive');assert.equal(reviewed,0);
  assert.match(report.error.reviewSession.lastRejection,kind==='forged-proof'?/quote does not match/i:/actual evidence references/);
 }
});

async function artifactQaFixture(t){
 const {root,input}=await fixture(t),path=join(root,'artifact.log'),text='Observed boundary failure';
 const sha256=createHash('sha256').update(text).digest('hex'),reference='artifact:'+sha256;
 await writeFile(path,text);
 input.groups[0].linkedDiagnostics[0].artifacts=[{path,sha256}];
 const proof={reference,startLine:1,endLine:1};
 const answer=selected=>({
  answers:Object.fromEntries(qaTopics.map(topic=>[topic,{text,refs:[reference]}])),
  claims:qaTopics.map(topic=>({path:'answers.'+topic+'.text',kind:'observation',proofs:[selected]})),
 });
 const review=selected=>({verdict:'approved',reason:text,claims:[{path:'reason',kind:'observation',proofs:[selected]}]});
 return {root,input,text,sha256,proof,answer,review};
}

test('failure-first native answer and review derive quotes from verified artifact selectors',async t=>{
 const {root,input,text,sha256,proof,answer,review}=await artifactQaFixture(t);
 const provider=await scriptedReview(t,root,body=>{
  const schema=body.tools.find(tool=>tool.function.name==='submit_review').function.parameters.properties.result;
  const replies=body.messages.filter(message=>message.role==='tool');
  if(replies.at(-1)?.content.includes('Accepted. Finish'))return null;
  const build=schema.properties.answers?answer:review;
  const selected=replies.length?proof:{...proof,quote:text};
  return {name:'submit_review',args:{result:build(selected)}};
 });
 let answering,reviewing;
 const report=await runProposalQa({...input,config:{actor:provider.profile},evaluationContract:{workflow:'failure-first-v1'},
  projectSummaries:[],learningProofEvidence:[],validateProposal:value=>value},{
  qaAnswer:async args=>{answering=args.directory;return (await nativeQaAnswer(args)).value;},
  qaReview:async args=>{reviewing=args.directory;return (await nativeQaReview(args)).value;},
 });
 assert.equal(report.verdict,'approved');
 assert.equal(provider.requests(),6);
 for(const directory of [answering,reviewing]){
  const nativeInput=JSON.parse(await readFile(join(directory,'input.json'),'utf8'));
  assert(nativeInput.frozenEvidence.artifacts.some(artifact=>artifact.id===proof.reference&&artifact.text===text));
  const settlement=JSON.parse(await readFile(join(directory,'termination.json'),'utf8'));
  assert.equal(settlement.rejectedSubmissions,1);
 }
 for(const claim of [report.rounds[0].claims[0],report.rounds[0].review.claims[0]]){
  assert.equal(claim.proofs[0].quote,text);
  assert.equal(claim.proofs[0].sourceSha256,sha256);
 }
});

test('failure-first native QA revalidates a legacy cached literal proof in place',async t=>{
 const {root,input,text,proof,answer,review}=await artifactQaFixture(t);
 const provider=await scriptedReview(t,root,body=>{
  if(body.messages.some(message=>message.role==='tool'))return null;
  const schema=body.tools.find(tool=>tool.function.name==='submit_review').function.parameters.properties.result;
  const build=schema.properties.answers?answer:review;
  return {name:'submit_review',args:{result:build({reference:proof.reference,quote:text})}};
 });
 for(const [name,adapter]of [['answer',nativeQaAnswer],['review',nativeQaReview]]){
  const request={...input,directory:join(root,name),config:{actor:provider.profile},validate:value=>value};
  const cached=await adapter(request);
  const resumed=await adapter({...request,config:{},evaluationContract:{workflow:'failure-first-v1'}});
  assert.equal(resumed.recovered,true);
  assert.equal(resumed.session,cached.session);
  assert.deepEqual(resumed.value,cached.value);
 }
 assert.equal(provider.requests(),4);
});

test('failure-first native QA repairs an answer in fresh sessions and keeps the approved draft',async t=>{
 const {root,input,text,proof,answer,review}=await artifactQaFixture(t);
 const answerPath='answers.example.text';
 const issue={kind:'answer',path:answerPath,problem:'The archived evidence does not show how the passing kernel was edited'};
 let answerSession=0,reviewSession=0;
 const provider=await scriptedReview(t,root,body=>{
  const schema=body.tools.find(tool=>tool.function.name==='submit_review').function.parameters.properties.result;
  const replies=body.messages.filter(message=>message.role==='tool');
  if(replies.at(-1)?.content.includes('Accepted. Finish'))return null;
  let result;
  if(schema.properties.answers){
   result=answer(proof);
   if(answerSession===1)result.answers.example.text='The passing kernel changed its loop';
   if(answerSession===2&&!replies.length){
    result.reason='A new rationale for the draft';
    result.claims.push({path:'reason',kind:'hypothesis',proofs:[proof]});
   }
  }else{
   result=review(proof);
   if(reviewSession===1){
    result.verdict='revise';
    result.issues=[issue];
    result.claims.push({path:'issues.0.problem',kind:'observation',proofs:[proof]});
   }else{
    result.issues=[{kind:'advisory',problem:'An additional example is optional'}];
    result.claims.push({path:'issues.0.problem',kind:'hypothesis',proofs:[proof]});
   }
  }
  return {name:'submit_review',args:{result}};
 });
 const answerDirectories=[],reviewDirectories=[];
 const report=await runProposalQa({...input,qaPolicy:'split',config:{actor:provider.profile},
  evaluationContract:{workflow:'failure-first-v1'},projectSummaries:[],learningProofEvidence:[],allowWriterRevision:true,
  validateProposal:value=>value},{
  qaAnswer:async args=>{
   answerSession++;
   answerDirectories.push(args.directory);
   if(answerSession===2){
    assert.equal(args.answerCorrection.previousAnswers.example.text,'The passing kernel changed its loop');
    assert.deepEqual(args.answerCorrection.findings.map(({kind,path})=>({kind,path})),[{kind:'answer',path:answerPath}]);
   }
   return (await nativeQaAnswer(args)).value;
  },
  qaReview:async args=>{
   reviewSession++;
   reviewDirectories.push(args.directory);
   assert.deepEqual(args.proposal,input.proposal);
   assert.equal(args.answers.example.text,reviewSession===1?'The passing kernel changed its loop':text);
   return (await nativeQaReview(args)).value;
  },
 });
 assert.equal(report.verdict,'approved');
 assert.equal(report.rounds.length,2);
 assert.deepEqual(report.proposal,input.proposal);
 assert.equal(report.rounds[1].review.issues[0].kind,'advisory');
 assert.equal(new Set(answerDirectories).size,2);
 assert.equal(new Set(reviewDirectories).size,2);
 const second=JSON.parse(await readFile(join(answerDirectories[1],'termination.json'),'utf8'));
 assert.equal(second.rejectedSubmissions,1);
 assert.equal(provider.requests(),9);
});

test('five isolated native reviewers decide from frozen answers and a primary-source audit',async t=>{
 const {root,input,proof,answer,review}=await artifactQaFixture(t);
 const extra={reference:'check:execution',text:'The mismatching output belongs to the intermediate edit.'};
 const issue={path:'changes.0.text',kind:'draft',problem:'The mismatch belongs to the starter'};
 let issued=0,objections;
 const stages=new Set();
 const provider=await scriptedReview(t,root,body=>{
  if(body.messages.some(m=>m.role==='tool'))return null;
  const schema=body.tools.find(tool=>tool.function.name==='submit_review').function.parameters.properties.result;
  let result;
  if(schema.properties.audits){
   result={audits:[{claimId:objections[0].id,disposition:'disproved',reason:extra.text}],claims:[{path:'audits.0.reason',kind:'observation',proofs:[{reference:extra.reference,startLine:1,endLine:1}]}]};
  }else{
   result=review(proof);
   if(++issued===1){result.verdict='revise';result.issues=[issue];result.claims.push({path:'issues.0.problem',kind:'observation',proofs:[proof]});}
  }
  return {name:'submit_review',args:{result}};
 });
 const request={...input,config:{actor:provider.profile},evaluationContract:{workflow:'failure-first-v1'},qaPolicy:'panel',seconds:600,
  panelAuditEvidence:[extra],projectSummaries:[],learningProofEvidence:[],validateProposal:value=>value};
 request.frozenQa={inputIdentity:qaIdentity(request),draft:request.proposal,proposal:request.proposal,...answer(proof)};
 const out=await runProposalQa(request,{
  qaReview:async args=>{stages.add(args.directory);return nativeQaReview(args);},
  qaPanelAudit:async args=>{objections=args.objections;return nativeQaPanelAudit(args);},
 });
 assert.equal(out.verdict,'approved',out.error?.message??out.reason);
 const panel=out.rounds[0].panel;
 assert.equal(panel.decision.approvals,4);assert.equal(stages.size,5);assert.equal(provider.requests(),12);
 assert.equal(panel.decision.claims[0].audit.proofs[0].quote,extra.text);
 for(const directory of stages){
  const seen=JSON.parse(await readFile(join(directory,'input.json')));
  assert.deepEqual(seen.answers,request.frozenQa.answers);assert.equal(seen.primaryAuditEvidence,undefined);
  assert.equal(seen.objections,undefined);
 }
 await assert.rejects(runProposalQa({...request,frozenQa:{...request.frozenQa,answers:{}}},{}),/changed on resume/);
});
