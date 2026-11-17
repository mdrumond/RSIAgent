import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {Type} from 'typebox';
import {reviewSession,nativeQaReview} from '../../cli/skill-review-native.mjs';
import {runProposalQa,qaTopics} from '../../cli/skill-review-qa.mjs';
const claim=path=>({path,kind:'observation',proofs:[{reference:'raw',startLine:1,endLine:1}]});
const valid=()=>({reason:'Observed failure',claims:[claim('reason')]});
async function fixture(t,script){
 const directory=await mkdtemp(join(tmpdir(),'native-termination-'));t.after(()=>rm(directory,{recursive:true,force:true}));const bodies=[];
 const server=createServer(async(req,res)=>{let text='';for await(const chunk of req)text+=chunk;bodies.push(JSON.parse(text));const next=await script(bodies.length,bodies.at(-1),req,res);
  if(res.writableEnded||next===undefined)return;
  if(next.status){res.writeHead(next.status,{'content-type':'application/json'});res.end(JSON.stringify({error:{message:next.error,type:'permission_error'}}));return;}
  res.writeHead(200,{'content-type':'text/event-stream'});const tool=next.result!==undefined,delta=tool?{role:'assistant',tool_calls:[{index:0,id:'call'+bodies.length,type:'function',function:{name:'submit_review',arguments:JSON.stringify({result:next.result})}}]}:{role:'assistant',content:'Finished without submitting'};
  res.end('data: '+JSON.stringify({id:'test',object:'chat.completion.chunk',created:1,model:'termination',choices:[{index:0,delta,finish_reason:tool?'tool_calls':'stop'}]})+'\n\ndata: [DONE]\n\n');
 });server.listen(0,'127.0.0.1');await once(server,'listening');t.after(async()=>{server.closeAllConnections();await new Promise(resolve=>server.close(resolve));});
 const env='QA_TERMINATION_FIXTURE';const previous=process.env[env];process.env[env]='fixture';t.after(()=>{if(previous===undefined)delete process.env[env];else process.env[env]=previous;});
 const actor={provider:'termination-fixture',model:'termination',credential_env:env,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'termination',name:'termination',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
 return {directory,bodies,config:{actor},seconds:5,prompt:{},proofRequired:true,selectorOnly:true,proofRecords:[{reference:'raw',text:'Observed failure\n'}],resultSchema:Type.Object({reason:Type.String(),claims:Type.Array(Type.Object({path:Type.String(),kind:Type.Literal('observation'),proofs:Type.Array(Type.Object({reference:Type.String(),startLine:Type.Integer(),endLine:Type.Integer()},{additionalProperties:false}))},{additionalProperties:false}))},{additionalProperties:false}),validate:x=>x};
}
test('actual Pi SDK returns validation feedback, accepts selector issue fields, and normalizes cached proofs',async t=>{
 const f=await fixture(t,(n,body)=>{
  if(n===1)return {result:{verdict:'revise',reason:'Observed failure',issues:[{problem:'Observed failure',field:'changes.0.text',summaryItemId:'summary-item'}],claims:[claim('reason')]}};
  if(n===2){assert(body.messages.some(m=>m.role==='tool'&&JSON.stringify(m).includes('issues.0.problem')));return {result:{verdict:'revise',reason:'Observed failure',issues:[{problem:'Observed failure',field:'changes.0.text',summaryItemId:'summary-item'}],claims:[claim('reason'),claim('issues.0.problem')]}};}
  return {};
 });
 const input={...f,groups:[],entries:[],proposal:{changes:[],memoryRefs:[]},answers:{},sourceCatalogue:{directory:f.directory,indexFile:join(f.directory,'unused.json'),records:f.proofRecords,facts:[],pages:[],sources:[]},evidenceSummary:{items:[{id:'summary-item',text:'Observed failure',kind:'observation',proofs:[{reference:'raw',startLine:1,endLine:1}]}]}};
 const result=await nativeQaReview(input);assert.equal(result.value.verdict,'revise');assert.equal(result.value.issues[0].summaryItemId,'summary-item');assert.equal(f.bodies.length,3);assert.equal(result.value.claims[1].proofs[0].quote,'Observed failure');
 const cached=await nativeQaReview({...input,config:{}});assert.equal(cached.recovered,true);assert.equal(f.bodies.length,3);assert.deepEqual(cached.value,result.value);
 const termination=JSON.parse(await readFile(join(f.directory,'termination.json')));assert.equal(termination.kind,'submitted');assert.equal(termination.rejectedSubmissions,1);
 const legacy={...result.value,issues:[{problem:'Observed failure',reference:'raw',quote:'Observed failure'}]};await writeFile(join(f.directory,'submission.json'),JSON.stringify(legacy));
 const historical=await nativeQaReview({...input,config:{}});assert.equal(historical.recovered,true);assert.equal(historical.value.issues[0].reference,'raw');assert.equal(historical.value.issues[0].sourceSha256.length,64);assert.equal(f.bodies.length,3);
});
test('actual Pi SDK preserves sanitized proxy403 separately from no-submit',async t=>{
 const f=await fixture(t,()=>({status:403,error:'Proxy denied Bearer secret-token https://proxy.invalid/path?token=private'}));
 await assert.rejects(reviewSession(f),error=>{assert.equal(error.reviewSession.kind,'provider-error');assert.match(error.message,/403|Proxy denied/);assert(!error.message.includes('secret-token'));assert(!error.message.includes('token=private'));return true;});
 const termination=JSON.parse(await readFile(join(f.directory,'termination.json')));assert.equal(termination.kind,'provider-error');assert(termination.providerErrors>=1);assert.equal(termination.rejectedSubmissions,0);
 const qa=await runProposalQa({directory:join(f.directory,'qa'),proposal:{changes:[],memoryRefs:[]},entries:[],groups:[],projectSummaries:[],learningProofEvidence:[],validateProposal:x=>x},{qaAnswer:()=>reviewSession(f)});assert.equal(qa.error.reviewSession.kind,'provider-error');assert(qa.error.reviewSession.providerErrors>=1);
});
test('actual Pi SDK distinguishes rejected submissions from an ordinary no-submit finish',async t=>{
 for(const rejected of [false,true]){
  const f=await fixture(t,n=>rejected&&n===1?{result:{reason:'Observed failure',claims:[{...claim('reason'),proofs:[{reference:'missing',startLine:1,endLine:1}]}]}}:{});
  await assert.rejects(reviewSession(f),error=>{assert.equal(error.reviewSession.kind,rejected?'rejected-submissions':'no-submit');assert.equal(error.reviewSession.rejectedSubmissions,rejected?1:0);if(rejected)assert.match(error.reviewSession.lastRejection,/missing/);return true;});
 }
});
test('actual Pi SDK cancellation drains HTTP work and preserves the proposal elapsed budget on resume',async t=>{
 let release;const arrived=new Promise(resolve=>{release=resolve;});let resume=false;
 const f=await fixture(t,async(n,body,req,res)=>{if(!resume){release();return undefined;}return n===2?{result:valid()}:{};});
 const stop=new AbortController(),proposal={changes:[{text:'Preserve evidence'}],memoryRefs:[]};let clock=0;
 const input={directory:join(f.directory,'proposal'),proposal,entries:[],groups:[],projectSummaries:[],learningProofEvidence:[],target:'a3',objective:'correctness',validateProposal:x=>x};
 const running=runProposalQa({...input,signal:stop.signal},{now:()=>clock,qaAnswer:args=>reviewSession({...f,directory:args.directory,signal:args.signal,seconds:args.seconds}).then(()=>{throw Error('Unexpected submission');})});
 await arrived;clock=40000;stop.abort(Error('Pause native QA'));await assert.rejects(running,/Pause native QA/);
 resume=true;const out=await runProposalQa(input,{now:()=>clock,qaAnswer:async({seconds,directory,signal})=>{assert.equal(seconds,260);await reviewSession({...f,directory,signal,seconds});return {answers:Object.fromEntries(qaTopics.map(k=>[k,{text:'Supported '+k,refs:[]}]))};},qaReview:async()=>({verdict:'approved',reason:'Checked'})});
 assert.equal(out.elapsedSeconds,40);assert.equal(out.verdict,'approved');assert.equal(f.bodies.length,3);
});

test('actual Pi SDK elapsed termination is distinct from cancellation and no-submit',async t=>{
 const f=await fixture(t,()=>undefined);await assert.rejects(reviewSession({...f,seconds:0.05}),error=>{assert.equal(error.name,'TimeoutError');assert.equal(error.reviewSession.kind,'elapsed');return true;});
 const termination=JSON.parse(await readFile(join(f.directory,'termination.json')));assert.equal(termination.kind,'elapsed');assert.equal(termination.rejectedSubmissions,0);
});
test('actual Pi SDK schema rejection before execute counts exactly one rejected submission',async t=>{
 const f=await fixture(t,n=>n===1?{result:{reason:'Observed failure'}}:{});
 await assert.rejects(reviewSession(f),error=>{assert.equal(error.reviewSession.kind,'rejected-submissions');assert.equal(error.reviewSession.rejectedSubmissions,1);assert(error.reviewSession.lastRejection);return true;});
 const rows=(await readFile(join(f.directory,'submission-rejections.jsonl'),'utf8')).trim().split('\n').map(JSON.parse);assert.equal(rows.length,1);
});
test('actual Pi SDK inherited timeout signal records elapsed rather than deliberate cancellation',async t=>{
 const f=await fixture(t,()=>undefined);
 await assert.rejects(reviewSession({...f,signal:AbortSignal.timeout(50)}),error=>{assert.equal(error.name,'TimeoutError');assert.equal(error.reviewSession.kind,'elapsed');return true;});
 const termination=JSON.parse(await readFile(join(f.directory,'termination.json')));assert.equal(termination.kind,'elapsed');
});
