import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm,readFile,writeFile,readdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {Check} from 'typebox/value';
import {classificationSchema,classificationBatchSchema,proposalSchema,nativeClassify} from '../../cli/skill-review-native.mjs';
import {proofSelectorCatalogue,materializeProof} from '../../cli/claim-proofs.mjs';
import {scriptedReview} from './helpers/scripted-review.mjs';
const row={cause:'unknown',coverage:'uncovered',guidanceIds:[],evidenceRefs:['events:1267893'],reason:'The verifier failed.'};
const claim={path:'reason',kind:'observation',proofs:[{reference:'events:1267893',startLine:1,endLine:1}]};
test('claims-only and nested live wrappers fail admission before prose traversal',()=>{
 const good={...row,claims:[claim]};assert(Check(classificationSchema,good));
 for(const bad of [{claims:[claim]},{...good,proposal:good},{...good,answer:{classifications:[row]}}])assert.equal(Check(classificationSchema,bad),false);
 assert.equal(Check(classificationBatchSchema,{claims:[{...claim,path:'classifications.0.reason'}]}),false);
 assert.equal(Check(proposalSchema,{claims:[claim]}),false);
 assert(Check(proposalSchema,{action:'no-change',reason:row.reason,groupIds:['g'],claims:[claim]}));
});
test('source catalogue and field rejection repair share actual supported selectors',()=>{
 const records=[{reference:'events:1267893',text:'{"performance":{"passed":false}}'},{reference:'fact',factId:'passed',value:false,text:'false'},{reference:'input:proposal',text:'{"reason":"observed"}'}];
 const catalogue=proofSelectorCatalogue(records);assert.deepEqual(catalogue.map(r=>r.selectors),[['lines'],['reference'],['lines','field']]);
 let detail;try{materializeProof({reference:records[0].reference,field:'data.performance.passed',segment:1},records);}catch(error){detail=JSON.parse(error.message.split('Validation details: ')[1]);}
 assert.equal(materializeProof(detail.repair,records).quote,records[0].text);
 assert.throws(()=>materializeProof({reference:'input:proposal',field:'reason'},records),/RFC6901/);
 assert.equal(materializeProof({reference:'input:proposal',field:'/reason'},records).value,'observed');
 assert.throws(()=>materializeProof({reference:'fact',factId:'other'},records),/does not match/);
});
test('native raw live rejection can repair selectors without provider-generated claims',async t=>{
 const root=await mkdtemp(join(tmpdir(),'contract-replay-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const directory=join(root,'classify'),group={id:'g',occurrences:[{reference:'events:1267893',data:{performance:{passed:false}},artifacts:[]}]};
 const raw={...row,claims:[{...claim,proofs:[{reference:'events:1267893',field:'data.performance.passed',segment:1}]}]};let corrected;
 const provider=await scriptedReview(t,root,(body,n)=>{
  if(n===0)return {name:'read',args:{path:'input.json'}};
  if(n===1)return {name:'submit_review',args:{result:{claims:[claim]}}};
  if(n===2)return {name:'submit_review',args:{result:raw}};
  if(n===3){const feedback=body.messages.filter(m=>m.role==='tool').at(-1).content;const detail=JSON.parse(feedback.split('Validation details: ')[1].split(' Required prose claim paths: ')[0]);corrected={...row,claims:[{...claim,proofs:[detail.repair]}]};return {name:'submit_review',args:{result:corrected}};}
  return null;
 });
 const input={directory,config:{actor:provider.profile},group,entries:[],seconds:20,validate:x=>x};
 const out=await nativeClassify(input);assert.equal(out.value.reason,row.reason);assert.equal(provider.requests(),5);assert.equal(raw.claims[0].text,undefined);assert.equal(raw.claims[0].proofs[0].field,'data.performance.passed');
 const rejections=(await readFile(join(directory,'submission-rejections.jsonl'),'utf8')).trim().split('\n').map(JSON.parse);assert.equal(rejections.length,2);assert.equal(rejections[1].result.claims[0].proofs[0].field,'data.performance.passed');
 assert.equal((await nativeClassify(input)).recovered,true);
 // A contract change must not replay an accepted stale result or its rejected conversation.
 const retained=JSON.parse(await readFile(join(directory,'native-session.json'),'utf8'));
 await writeFile(join(directory,'review-contract.json'),JSON.stringify({version:0}));
 await assert.rejects(nativeClassify(input),error=>error.reviewSession?.kind==='no-submit');
 const next=JSON.parse(await readFile(join(directory,'native-session.json'),'utf8'));assert.notEqual(next.path,retained.path);assert((await readFile(retained.path,'utf8')).length>0);
 const history=join(directory,'contract-history');assert((await readdir(history)).length>=2);
 await assert.rejects(nativeClassify(input),error=>error.reviewSession?.kind==='no-submit');
});
