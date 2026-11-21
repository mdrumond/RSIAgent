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
  try{checkpoint=JSON.parse(await readFile(join(input.directory,'qa.json'),'utf8'));if(checkpoint.inFlightOperations&&checkpoint.elapsedSeconds>=1)break;}catch(error){if(error.code!=='ENOENT')throw error;}
  await pause(10);
 }
 assert(checkpoint?.inFlightOperations);assert(checkpoint.elapsedSeconds>=1);const exited=once(child,'exit');child.kill('SIGKILL');await exited;
 const reserved=Math.max(...Object.values(checkpoint.inFlightOperations).map(op=>op.reservedThrough));assert(reserved>=6&&reserved<10);
 const later=Date.now()+86400000;
 const resumed=await runProposalQa(input,{now:()=>later,qaAnswer:async({seconds})=>{assert(seconds<=300-reserved);assert(seconds>289);return answers();},qaReview:async()=>({verdict:'approved',reason:'Supported'})});
 assert.equal(resumed.verdict,'approved');assert(resumed.elapsedSeconds>=reserved&&resumed.elapsedSeconds<10);assert.equal(resumed.inFlight,undefined);assert.equal(resumed.inFlightOperations,undefined);assert.equal(resumed.seconds,300);
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
test('every review receives the complete previous issue list after a writer revision',async t=>{
 const input=await fixture(t),previous={verdict:'revise',reason:'Two material defects',issues:[{problem:'Wrong direction'},{problem:'Unsupported device claim'}]};let reviews=0;
 const result=await runProposalQa(input,{qaAnswer:async x=>{if(reviews)assert.deepEqual(x.previousReview,previous);return answers();},qaReview:async x=>{
  if(!reviews++){assert.equal(x.previousReview,undefined);return previous;}
  assert.deepEqual(x.previousReview,previous);return {verdict:'approved',reason:'Both defects corrected'};
 }});
 assert.equal(result.verdict,'approved');assert.equal(reviews,2);
});

const failureFirstQa=input=>({...input,evaluationContract:{workflow:'failure-first-v1'},allowWriterRevision:true});
const answerFinding={kind:'answer',path:'answers.mechanism.text',problem:'Name the observed boundary'};
const advisoryFinding={kind:'advisory',problem:'An extra example could help'};

test('split QA runs one fresh answer-only correction within cumulative clocks',async t=>{
 const input=failureFirstQa(await fixture(t));let clock=0,reviewed=0;
 const grants=[],directories=[];
 const original=answers();
 const corrected=answers();corrected.answers.mechanism.text='Observed only on the tail boundary';
 const report=await runProposalQa(input,{now:()=>clock,
  qaAnswer:async({seconds,directory,answerCorrection,proposal,validate})=>{
   grants.push(['answer',seconds]);directories.push(directory);assert.deepEqual(proposal,input.proposal);
   if(!answerCorrection){clock+=240000;return original;}
   assert.deepEqual(answerCorrection.findings,[answerFinding]);
   assert.deepEqual(answerCorrection.previousAnswers,original.answers);
   assert.throws(()=>validate({...corrected,changes:[{text:'Alter the draft'}]}),/must preserve the proposal/);
   clock+=40000;return corrected;
  },
  qaReview:async({seconds,answers:received,proposal,validate})=>{
   grants.push(['review',seconds]);assert.deepEqual(proposal,input.proposal);
   if(!reviewed++){
    assert.deepEqual(received,original.answers);clock+=220000;
    assert.throws(()=>validate({verdict:'approved',reason:'Incorrect approval',issues:[answerFinding,advisoryFinding]}),/cannot retain material findings/);
    assert.throws(()=>validate({verdict:'revise',reason:'Optional only',issues:[advisoryFinding]}),/material finding/);
    assert.throws(()=>validate({verdict:'revise',reason:'Unknown answer',issues:[{...answerFinding,path:'answers.other.text'}]}),/exact answers/);
    return {verdict:'revise',reason:'Answer needs boundary',issues:[answerFinding,advisoryFinding]};
   }
   assert.deepEqual(received,corrected.answers);clock+=70000;
   return {verdict:'approved',reason:'Supported',issues:[advisoryFinding]};
  },
 });
 assert.equal(report.qaPolicy,'split');assert.equal(report.seconds,600);
 assert.deepEqual(report.phaseSeconds,{answer:300,review:300});
 assert.deepEqual(report.phaseElapsed,{answer:280,review:290});
 assert.equal(report.elapsedSeconds,570);assert.equal(report.verdict,'approved');
 assert.equal(report.approval,qaIdentity(input));assert.deepEqual(report.proposal,input.proposal);
 assert.notEqual(directories[0],directories[1]);
 assert.deepEqual(grants,[['answer',300],['review',300],['answer',60],['review',80]]);
});

test('typed answer correction does not overwrite the proposal or repeat indefinitely',async t=>{
 const input=failureFirstQa(await fixture(t));let corrections=0;
 const report=await runProposalQa(input,{
  qaAnswer:async args=>{if(args.answerCorrection)corrections++;return answers();},
  qaReview:async()=>({verdict:'revise',reason:'Answer remains unsupported',issues:[answerFinding]}),
 });
 assert.equal(corrections,1);assert.equal(report.rounds.length,2);
 assert.equal(report.verdict,'inconclusive');assert.equal(report.revisionFeedback,undefined);
 assert.equal(report.approval,undefined);
});

test('material draft issues go to the writer, while missing evidence takes precedence',async t=>{
 const draft={kind:'draft',problem:'Scope of the instruction is too broad'},evidence={kind:'evidence',problem:'No retained target check'};
 for(const [issues,writer]of [[[answerFinding,draft,advisoryFinding],true],[[draft,evidence],false]]){
  const input=failureFirstQa(await fixture(t));
  const report=await runProposalQa(input,{
   qaAnswer:async args=>{assert.equal(args.answerCorrection,undefined);return answers();},
   qaReview:async()=>({verdict:'revise',reason:'Review found material issues',issues}),
  });
  assert.equal(report.verdict,'inconclusive');
  assert.equal(report.revisionFeedback,writer?'Review found material issues':undefined);
  assert.equal(report.rounds.length,1);
 }
});

test('exhausted answer phase cannot borrow review time or trigger a correction',async t=>{
 const input=failureFirstQa(await fixture(t));let clock=0,answered=0;
 const report=await runProposalQa(input,{now:()=>clock,
  qaAnswer:async({seconds})=>{assert.equal(seconds,300);answered++;clock+=300000;return answers();},
  qaReview:async({seconds})=>{assert.equal(seconds,300);clock+=1000;return {verdict:'revise',reason:'Answer still missing facts',issues:[answerFinding]};},
 });
 assert.equal(report.verdict,'inconclusive');assert.equal(report.approval,undefined);
 assert.equal(report.revisionFeedback,undefined);assert.equal(answered,1);
 assert.deepEqual(report.phaseElapsed,{answer:300,review:1});
});

test('experiment override and legacy failure-first checkpoint use the shared spent clock',async t=>{
 const input=failureFirstQa(await fixture(t)),stop=new AbortController();let clock=0;
 const shared={...input,qaPolicy:'shared'};
 await assert.rejects(runProposalQa({...shared,signal:stop.signal},{now:()=>clock,
  qaAnswer:async({seconds})=>{assert.equal(seconds,300);clock+=80000;stop.abort(Error('pause'));throw stop.signal.reason;},
 }),/pause/);
 const path=join(input.directory,'qa.json'),legacy=JSON.parse(await readFile(path,'utf8'));
 assert.equal(legacy.qaPolicy,'shared');delete legacy.qaPolicy;await writeFile(path,JSON.stringify(legacy));
 clock+=90000000;
 const resumed=await runProposalQa(input,{now:()=>clock,
  qaAnswer:async({seconds,answerCorrection})=>{assert.equal(seconds,220);assert.equal(answerCorrection,undefined);return answers();},
  qaReview:async({seconds})=>{assert.equal(seconds,220);return {verdict:'revise',reason:'Legacy writer feedback',issues:[answerFinding]};},
 });
 assert.equal(resumed.qaPolicy,'shared');assert.equal(resumed.seconds,300);assert.equal(resumed.elapsedSeconds,80);
 assert.equal(resumed.revisionFeedback,'Legacy writer feedback');assert.equal(resumed.rounds.length,1);
});

test('split QA restores the charged phase reservation and excludes paused wall time',async t=>{
 const input=failureFirstQa(await fixture(t)),stop=new AbortController();let clock=0;
 await assert.rejects(runProposalQa({...input,signal:stop.signal},{now:()=>clock,
  qaAnswer:async()=>{clock+=80000;stop.abort(Error('pause'));throw stop.signal.reason;},
 }),/pause/);
 const path=join(input.directory,'qa.json'),saved=JSON.parse(await readFile(path,'utf8'));
 assert.deepEqual(saved.phaseElapsed,{answer:80,review:0});
 saved.inFlightOperations={'0-qaAnswer':{stage:'qaAnswer',lane:'answer',reservedThrough:85,reservedPhaseThrough:85}};
 await writeFile(path,JSON.stringify(saved));clock+=90000000;
 const report=await runProposalQa(input,{now:()=>clock,
  qaAnswer:async({seconds})=>{assert.equal(seconds,215);return answers();},
  qaReview:async({seconds})=>{assert.equal(seconds,300);return {verdict:'approved',reason:'Supported'};},
 });
 assert.equal(report.verdict,'approved');assert.equal(report.elapsedSeconds,85);
 assert.deepEqual(report.phaseElapsed,{answer:85,review:0});
});

test('split QA discards a successful response returned after the parent stopped',async t=>{
 const input=failureFirstQa(await fixture(t)),stop=new AbortController(),cause=Error('Operator stopped review');
 let clock=0;
 await assert.rejects(runProposalQa({...input,signal:stop.signal},{now:()=>clock,
  qaAnswer:async()=>answers(),
  qaReview:async()=>{clock+=20000;stop.abort(cause);return {verdict:'approved',reason:'Late'};},
 }),error=>error===cause);
 const checkpoint=JSON.parse(await readFile(join(input.directory,'qa.json'),'utf8'));
 assert.equal(checkpoint.approval,undefined);assert.equal(checkpoint.rounds[0].review,undefined);
 assert.deepEqual(checkpoint.phaseElapsed,{answer:0,review:20});
 const resumed=await runProposalQa(input,{now:()=>clock,
  qaAnswer:()=>assert.fail('Retain the completed answers'),
  qaReview:async({seconds})=>{assert.equal(seconds,280);return {verdict:'approved',reason:'Fresh review'};},
 });
 assert.equal(resumed.verdict,'approved');
});

test('panel interruption retains submitted votes and resumes only unfinished slots',async t=>{
 const input={...await fixture(t),qaPolicy:'panel',evaluationContract:{workflow:'failure-first-v1'},projectSummaries:[],learningProofEvidence:[]};
 const controller=new AbortController(),paths=[];
 let active=0,peak=0;
 const vote={verdict:'approved',reason:'Bounded by observed evidence'};
 const call=async({directory,signal})=>{
  paths.push(directory);peak=Math.max(peak,++active);
  try{
   if(directory.endsWith('-0')||directory.endsWith('-1'))return vote;
   await new Promise((resolve,reject)=>signal.addEventListener('abort',()=>reject(signal.reason),{once:true}));
  }finally{active--;}
 };
 const job=runProposalQa({...input,signal:controller.signal},{qaAnswer:async()=>answers(),qaReview:call});
 let saved;
 for(let i=0;i<300;i++){
  try{saved=JSON.parse(await readFile(join(input.directory,'qa.json')));}catch{}
  if(saved?.rounds[0]?.panel?.slots.filter(row=>row?.status==='submitted').length===2&&paths.length===5)break;
  await new Promise(resolve=>setTimeout(resolve,10));
 }
 controller.abort(Error('Pause'));
 await assert.rejects(job,/Pause/);assert.equal(paths.length,5);assert(peak>=3);
 const reran=[];
 const out=await runProposalQa(input,{qaReview:async({directory})=>{reran.push(directory);return vote;}});
 assert.equal(out.verdict,'approved');assert.equal(reran.length,3);
 assert(!reran.some(path=>path.endsWith('-0')||path.endsWith('-1')));
});

test('lower concurrency retries unresolved provider votes while preserving submitted verdicts and time',async t=>{
 const input={...await fixture(t),qaPolicy:'panel',evaluationContract:{workflow:'failure-first-v1'}};
 const approved={verdict:'approved',reason:'Supported by execution'};
 let clock=0;
 const first=await runProposalQa(input,{now:()=>clock,qaAnswer:async()=>answers(),qaReview:async({directory})=>{
  clock+=5000;
  if(directory.endsWith('-0')||directory.endsWith('-1'))return approved;
  throw Object.assign(Error('Provider unavailable'),{reviewSession:{kind:'provider-error'}});
 }});
 assert.equal(first.reasonKind,'infrastructure');assert.equal(first.rounds[0].panel.decision.approvals,2);
 const saved=first.rounds[0].panel.slots.slice(0,2);
 let calls=0,active=0;
 const out=await runProposalQa({...input,qaPanelConcurrency:1,retryMissingPanelVotes:true},{now:()=>clock,qaReview:async({seconds})=>{
  assert.equal(++active,1);assert(seconds<600);await new Promise(r=>setImmediate(r));active--;calls++;return approved;
 }});
 assert.equal(calls,3);assert.equal(out.verdict,'approved');
 assert.deepEqual(out.rounds[0].panel.slots.slice(0,2),saved);assert.equal(out.rounds[0].panel.history[0].decision.approvals,2);
});

test('one lost quorum-deciding vote and scientifically unresolved evidence have distinct outcomes',async t=>{
 for(const disposition of ['disproved','unresolved']){
  const input={...await fixture(t),qaPolicy:'panel',evaluationContract:{workflow:'failure-first-v1'}};
  const result=await runProposalQa(input,{
   qaAnswer:async()=>answers(),qaReview:async({directory})=>{
    if(directory.endsWith('-0')||directory.endsWith('-1'))return {verdict:'approved',reason:'Supported'};
    if(directory.endsWith('-2'))throw Error('Provider disconnected');
    return {verdict:'revise',reason:'Inspect source role',issues:[{kind:'draft',problem:'Source belongs to an intermediate edit'}]};
   },qaPanelAudit:async({objections})=>({audits:[{claimId:objections[0].id,disposition,reason:'Intermediate source role checked'}],
    claims:[{path:'audits.0.reason',proofs:[{reference:'e:1',quote:'Scored output',recordSha256:'fixture'}]}]}),
  });
  assert.equal(result.verdict,'inconclusive');
  assert.deepEqual(result.rounds[0].panel.decision.missingSlots,[3]);
  assert.equal(result.reasonKind,disposition==='disproved'?'infrastructure':'scientific');
 }
});

test('explicit recovery can retry an interrupted audit while keeping every original ballot',async t=>{
 const input={...await fixture(t),qaPolicy:'panel',evaluationContract:{workflow:'failure-first-v1'}};
 let clock=0;
 const first=await runProposalQa(input,{now:()=>clock,qaAnswer:async()=>answers(),qaReview:async({directory})=>directory.endsWith('-4')?
  {verdict:'revise',reason:'Wrong source attribution',issues:[{kind:'draft',problem:'Mismatch is from the starter'}]}:{verdict:'approved',reason:'Supported'},
  qaPanelAudit:async()=>{clock+=30000;throw Error('Audit provider disconnected');}
 });
 assert.equal(first.reasonKind,'infrastructure');assert.equal(first.rounds[0].panel.decision.missingSlots.length,0);
 const resumed=await runProposalQa({...input,retryMissingPanelVotes:true},{now:()=>clock,
  qaReview:()=>assert.fail('Every vote already submitted'),qaPanelAudit:async({objections,seconds})=>{
   assert.equal(seconds,90);
   return {audits:[{claimId:objections[0].id,disposition:'disproved',reason:'The output belongs to the intermediate source'}],
    claims:[{path:'audits.0.reason',proofs:[{reference:'e:1',quote:'Intermediate output',recordSha256:'fixture'}]}]};
  }});
 assert.equal(resumed.verdict,'approved');const panel=resumed.rounds[0].panel;
 assert.deepEqual(panel.slots,first.rounds[0].panel.slots);assert.equal(panel.history[0].auditElapsedSeconds,30);
 assert.equal(panel.history[0].auditError.message,'Audit provider disconnected');
});
