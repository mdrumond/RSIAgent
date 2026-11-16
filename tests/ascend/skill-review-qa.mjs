import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
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
 assert.equal(result.verdict,'inconclusive');assert.equal(result.reason,'qa-elapsed');assert.equal(calls,4);assert.ok(result.elapsedSeconds>=300);assert.equal(result.startedAt,1000);
 const saved=JSON.parse(await readFile(join(input.directory,'qa.json')));assert.equal(saved.seconds,300);
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
 assert.equal(allowance,210);assert.equal(resumed.startedAt,1000);assert.equal(resumed.verdict,'approved');
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
  writerCalls++;assert.equal(seconds,300);clock+=20000;
  assert.throws(()=>validate({...answers(),changes:[{text:'Reuse setup',evidenceRefs:['memory:1']}]}),/Unknown occurrence/);
  clock+=15000;
  return validate({...answers(),changes:[{text:'Reuse matching setup',evidenceRefs:['event:1']}]});
 },qaReview:async({proposal,seconds})=>{
  reviewed=proposal;assert.equal(seconds,265);return {verdict:'approved',reason:'Corrected evidence and scoped instruction'};
 }});
 assert.equal(writerCalls,1);assert.deepEqual(reviewed.changes,[{text:'Reuse matching setup',evidenceRefs:['event:1']}]);
 assert.equal(result.verdict,'approved');assert.equal(result.startedAt,1000);assert.equal(result.elapsedSeconds,35);
});
test('reviewer can repair malformed learning before submission within the same allocation',async t=>{
 const input=await fixture(t);let clock=1000;
 input.validateProposal=value=>{if(value.learningRequests?.some(request=>!request.hypothesis))throw Error('Learning hypothesis required');return value;};
 const result=await runProposalQa(input,{now:()=>clock,qaAnswer:async()=>{clock+=20000;return answers();},qaReview:async({validate,seconds})=>{
  assert.equal(seconds,280);assert.throws(()=>validate({verdict:'learn',reason:'Missing fact',learningRequests:[{question:'Boundary?'}]}),/hypothesis required/);
  clock+=10000;return validate({verdict:'learn',reason:'Missing device fact',learningRequests:[{question:'Boundary?',hypothesis:'Target differs',requiredEvidence:'Device comparison',learningTarget:'device'}]});
 }});
 assert.equal(result.verdict,'learn');assert.equal(result.elapsedSeconds,30);assert.equal(result.learningRequests.length,1);
});


test('Q&A may cite supplied verified learning references but not invented findings',async t=>{
 const input=await fixture(t),learning=[['request',{result:{status:'verified-device',evidence:[{reference:'learning:event:1',log:'copied.log'}]}}]];
 const result=await runProposalQa({...input,learning},{qaAnswer:async()=>{const value=answers();value.answers.evidence.refs=['learning:event:1'];return value;},qaReview:async()=>({verdict:'approved',reason:'Final text grounded in supplied learning'})});
 assert.equal(result.verdict,'approved');
});
test('paused wall time is excluded and caller interruption retains charged review work',async t=>{
 const input=await fixture(t),stop=new AbortController();let time=0;
 await assert.rejects(runProposalQa({...input,signal:stop.signal},{now:()=>time,qaAnswer:async()=>{time+=40000;stop.abort(Error('pause'));throw stop.signal.reason;}}),/pause/);
 time+=90000000;
 const result=await runProposalQa(input,{now:()=>time,qaAnswer:async({seconds})=>{assert.equal(seconds,260);return answers();},qaReview:async()=>({verdict:'approved',reason:'Supported'})});
 assert.equal(result.elapsedSeconds,40);assert.equal(result.protocol,'skill-qa-v2');
});
test('case contract changes invalidate approval identity',async t=>{
 const input=await fixture(t);assert.notEqual(qaIdentity({...input,evaluationContract:{id:'first'}}),qaIdentity({...input,evaluationContract:{id:'other'}}));
});

test('legacy timeout remains immutable and a v2 continuation charges its original spend',async t=>{
 const input=await fixture(t),file=join(input.directory,'qa.json'),legacy=JSON.stringify({format:1,elapsedSeconds:120,verdict:'inconclusive',reason:'qa-elapsed'});
 await writeFile(file,legacy);
 const result=await runProposalQa(input,{qaAnswer:async({seconds})=>{assert(seconds<=180&&seconds>179);return answers();},qaReview:async()=>({verdict:'approved',reason:'Supported'})});
 assert.equal(result.verdict,'approved');assert.equal(await readFile(file,'utf8'),legacy);assert.equal(JSON.parse(await readFile(join(input.directory,'qa-v2.json'))).protocol,'skill-qa-v2');
});
test('SIGKILL during provider work retains charged slices without charging paused downtime',async t=>{
 const {spawn}=await import('node:child_process'),{once}=await import('node:events'),{setTimeout:pause}=await import('node:timers/promises');
 const input=await fixture(t),module=new URL('../../cli/skill-review-qa.mjs',import.meta.url).href;
 const child=spawn(process.execPath,['--input-type=module','-e',`import {runProposalQa} from ${JSON.stringify(module)};const input=${JSON.stringify(input)};input.validateProposal=x=>x;await runProposalQa(input,{qaAnswer:async()=>new Promise(()=>{})});`],{stdio:'ignore'});
 t.after(()=>{if(child.exitCode===null)child.kill('SIGKILL');});
 let checkpoint;
 for(let i=0;i<300;i++){
  try{checkpoint=JSON.parse(await readFile(join(input.directory,'qa.json'),'utf8'));if(checkpoint.inFlight&&checkpoint.elapsedSeconds>=1)break;}catch(error){if(error.code!=='ENOENT')throw error;}
  await pause(10);
 }
 assert(checkpoint?.inFlight);assert(checkpoint.elapsedSeconds>=1);const exited=once(child,'exit');child.kill('SIGKILL');await exited;
 const reserved=checkpoint.inFlight.reservedThrough;assert(reserved>=6&&reserved<10);
 const later=Date.now()+86400000;
 const resumed=await runProposalQa(input,{now:()=>later,qaAnswer:async({seconds})=>{assert(seconds<=300-reserved);assert(seconds>289);return answers();},qaReview:async()=>({verdict:'approved',reason:'Supported'})});
 assert.equal(resumed.verdict,'approved');assert(resumed.elapsedSeconds>=reserved&&resumed.elapsedSeconds<10);assert.equal(resumed.inFlight,undefined);assert.equal(resumed.seconds,300);
});

test('legacy cached native submissions cannot approve a changed contract in a v2 continuation',async t=>{
 const input=await fixture(t),legacy={format:1,elapsedSeconds:120,verdict:'inconclusive',reason:'qa-elapsed'};
 await writeFile(join(input.directory,'qa.json'),JSON.stringify(legacy));
 const oldAnswer=join(input.directory,'0-qaAnswer'),oldReview=join(input.directory,'0-qaReview');await mkdir(oldAnswer);await mkdir(oldReview);
 await writeFile(join(oldAnswer,'submission.json'),JSON.stringify(answers()));await writeFile(join(oldReview,'submission.json'),JSON.stringify({verdict:'approved',reason:'Old contract'}));
 let calls=0;
 const native=async({directory},fresh)=>{try{return JSON.parse(await readFile(join(directory,'submission.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}calls++;return fresh;};
 const result=await runProposalQa({...input,evaluationContract:{id:'new-contract'}},{qaAnswer:args=>native(args,answers()),qaReview:args=>native(args,{verdict:'learn',reason:'New boundary requires facts',learningRequests:[{question:'New boundary?',hypothesis:'Untested'}]})});
 assert.equal(calls,2);assert.equal(result.verdict,'learn');assert.equal(result.approval,undefined);assert.equal(JSON.parse(await readFile(join(oldReview,'submission.json'),'utf8')).reason,'Old contract');
});
