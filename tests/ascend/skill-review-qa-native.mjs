import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {nativeQaAnswer,nativeQaReview} from '../../cli/skill-review-native.mjs';
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
 await writeFile(join(directory,'submission.json'),JSON.stringify([{id:'one',priority:80,reason:'Observed recurrence'}]));
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
 await writeFile(join(directory,'submission.json'),JSON.stringify({ok:true}));const {reviewSession}=await import('../../cli/skill-review-native.mjs');
 const out=await reviewSession({directory,config:{},prompt:{},allowedPaths:[primary,source],validate:x=>x});assert.equal(out.recovered,true);
 const input=JSON.parse(await readFile(join(directory,'input.json'),'utf8'));
 assert.deepEqual(input.filesystemAccess.allowed,[{path:directory,directory:true},{path:primary,directory:true},{path:source,directory:false}]);assert.deepEqual(input.filesystemAccess.requiredReads,[]);
});
