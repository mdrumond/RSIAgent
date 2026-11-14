import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {runProposalQa,qaIdentity,qaTopics} from '../../cli/skill-review-qa.mjs';
async function fixture(t){
 const directory=await mkdtemp(join(tmpdir(),'proposal-qa-'));t.after(()=>rm(directory,{recursive:true,force:true}));
 return {directory,proposal:{changes:[{text:'Reuse the prepared workspace'}],memoryRefs:['memory:1']},entries:[{id:'baseline',text:'Prior guidance'}],objective:'optimization',target:'a3',groups:[{occurrences:[{reference:'event:1',message:'Repeated setup'}]}],validateProposal:value=>value};
}
const answers=()=>({answers:Object.fromEntries(qaTopics.map(topic=>[topic,{text:'Specific explanation for '+topic,refs:['event:1']}]))});
test('approval binds final revised text, baseline, objective, target and evidence',async t=>{
 const input=await fixture(t);let calls=0;
 const result=await runProposalQa(input,{qaAnswer:async()=>({...answers(),...(calls?{changes:[{text:'Reuse only a matching prepared workspace'}]}:{})}),qaReview:async()=>++calls===1?{verdict:'revise',reason:'Unbounded reuse',issues:['Require matching runtime']}:{verdict:'approved',reason:'Boundary is explicit'}});
 assert.equal(result.rounds.length,2);assert.equal(result.verdict,'approved');assert.equal(result.approval,qaIdentity(input,result.proposal));
 for(const changed of [{target:'a5'},{objective:'correctness'},{entries:[]},{groups:[]}])assert.notEqual(qaIdentity({...input,...changed},result.proposal),result.approval);
 assert.notEqual(qaIdentity(input),result.approval);
 const resumed=await runProposalQa(input,{qaAnswer:()=>{throw Error('Replay');}});assert.deepEqual(resumed,result);
});
test('learning and inconclusive material issues never receive approval',async t=>{
 for(const verdict of ['learn','inconclusive']){
  const input=await fixture(t),result=await runProposalQa(input,{qaAnswer:async()=>answers(),qaReview:async()=>({verdict,reason:'Unknown target boundary',...(verdict==='learn'?{learningRequests:[{learningTarget:'device',question:'Does dispatch support it?',hypothesis:'Boundary differs',requiredEvidence:'Device control'}]}:{})})});
  assert.equal(result.verdict,verdict);assert.equal(result.approval,undefined);
 }
});
test('elapsed allowance is shared across revisions and retained on interruption',async t=>{
 const input=await fixture(t);let clock=1000,calls=0;
 const result=await runProposalQa(input,{now:()=>clock,qaAnswer:async()=>{clock+=50000;return answers();},qaReview:async()=>{calls++;clock+=20000;return {verdict:'revise',reason:'Missing boundary',issues:['Boundary']};}});
 assert.equal(result.verdict,'inconclusive');assert.equal(result.reason,'qa-elapsed');assert.equal(calls,1);assert.equal(result.startedAt,1000);
 const saved=JSON.parse(await readFile(join(input.directory,'qa.json')));assert.equal(saved.seconds,120);
});
test('invented answer evidence is rejected before independent assessment',async t=>{
 const input=await fixture(t);let reviewed=false;
 const result=await runProposalQa(input,{qaAnswer:async()=>{const value=answers();value.answers.evidence.refs=['invented'];return value;},qaReview:async()=>{reviewed=true;}});
 assert.equal(reviewed,false);assert.equal(result.verdict,'inconclusive');
});
test('interruption resumes the same proposal clock without refilling its allowance',async t=>{
 const input=await fixture(t),controller=new AbortController();let clock=1000;
 await assert.rejects(runProposalQa({...input,signal:controller.signal},{now:()=>clock,qaAnswer:async()=>{
  clock+=90000;controller.abort(Error('External stop'));throw controller.signal.reason;
 }}),/External stop/);
 let allowance;
 const resumed=await runProposalQa(input,{now:()=>clock,qaAnswer:async({seconds})=>{allowance=seconds;return answers();},qaReview:async()=>({verdict:'approved',reason:'No material issue'})});
 assert.equal(allowance,30);assert.equal(resumed.startedAt,1000);assert.equal(resumed.verdict,'approved');
});
test('writer revisions cannot introduce memory that was never retrieved',async t=>{
 const input=await fixture(t);let reviewed=false;
 const result=await runProposalQa(input,{qaAnswer:async()=>({...answers(),changes:[{text:'Claim an unsupported repair'}],memoryRefs:['memory:invented']}),qaReview:async()=>{reviewed=true;return {verdict:'approved',reason:'Rubber stamp'};}});
 assert.equal(result.verdict,'inconclusive');assert.equal(reviewed,false);assert.equal(result.approval,undefined);
});
test('changed proposal invalidates persisted admission instead of reusing old approval',async t=>{
 const input=await fixture(t);
 await runProposalQa(input,{qaAnswer:async()=>answers(),qaReview:async()=>({verdict:'approved',reason:'Supported'})});
 await assert.rejects(runProposalQa({...input,proposal:{...input.proposal,changes:[{text:'Different instruction'}]}},{}),/QA input changed/);
});
test('native submission receives change-validation feedback and can repair within the original clock',async t=>{
 const input=await fixture(t);let clock=1000,writerCalls=0,reviewed;
 input.validateProposal=value=>{
  if(value.changes.some(change=>change.evidenceRefs?.some(ref=>ref!=='event:1')))throw Error('Unknown occurrence evidence reference');
  return value;
 };
 const result=await runProposalQa(input,{now:()=>clock,qaAnswer:async({validate,seconds})=>{
  writerCalls++;assert.equal(seconds,120);clock+=20000;
  assert.throws(()=>validate({...answers(),changes:[{text:'Reuse setup',evidenceRefs:['memory:1']}]}),/Unknown occurrence/);
  clock+=15000;
  return validate({...answers(),changes:[{text:'Reuse matching setup',evidenceRefs:['event:1']}]});
 },qaReview:async({proposal,seconds})=>{
  reviewed=proposal;assert.equal(seconds,85);return {verdict:'approved',reason:'Corrected evidence and scoped instruction'};
 }});
 assert.equal(writerCalls,1);assert.deepEqual(reviewed.changes,[{text:'Reuse matching setup',evidenceRefs:['event:1']}]);
 assert.equal(result.verdict,'approved');assert.equal(result.startedAt,1000);assert.equal(result.elapsedSeconds,35);
});
test('reviewer can repair malformed learning before submission within the same allocation',async t=>{
 const input=await fixture(t);let clock=1000;
 input.validateProposal=value=>{if(value.learningRequests?.some(request=>!request.hypothesis))throw Error('Learning hypothesis required');return value;};
 const result=await runProposalQa(input,{now:()=>clock,qaAnswer:async()=>{clock+=20000;return answers();},qaReview:async({validate,seconds})=>{
  assert.equal(seconds,100);assert.throws(()=>validate({verdict:'learn',reason:'Missing fact',learningRequests:[{question:'Boundary?'}]}),/hypothesis required/);
  clock+=10000;return validate({verdict:'learn',reason:'Missing device fact',learningRequests:[{question:'Boundary?',hypothesis:'Target differs',requiredEvidence:'Device comparison',learningTarget:'device'}]});
 }});
 assert.equal(result.verdict,'learn');assert.equal(result.elapsedSeconds,30);assert.equal(result.learningRequests.length,1);
});
