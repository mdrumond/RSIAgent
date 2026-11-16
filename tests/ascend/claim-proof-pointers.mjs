import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {materializeProof,validateClaims,validateClaimedResult,claimEvidence} from '../../cli/project-summary.mjs';
import {reviewSession} from '../../cli/skill-review-native.mjs';

test('line pointers materialize exact excerpts and reject invalid or gap-crossing ranges',()=>{
 const records=[{reference:'file',text:'first\nsecond\nthird\n',paths:['retained.txt'],sourceRefs:['original:1']}];
 const p=materializeProof({reference:'file',startLine:2,endLine:3},records);assert.equal(p.quote,'second\nthird');assert.deepEqual(p.sourcePaths,['retained.txt']);assert.deepEqual(p.sourceRefs,['original:1']);assert.throws(()=>materializeProof({reference:'file',startLine:1,endLine:1,factId:'invented',value:0},records),/Structured assertions/);assert.equal(p.recordSha256.length,64);assert.equal(p.sourceSha256.length,64);
 for(const range of [[0,1],[2,1],[1,5],[1.5,2]])assert.throws(()=>materializeProof({reference:'file',startLine:range[0],endLine:range[1]},records),/range/);
 const excerpt=[{reference:'excerpt',text:'head\ntail',segments:['head','tail']}];assert.throws(()=>materializeProof({reference:'excerpt',startLine:1,endLine:2},excerpt),/explicit segment/);
 assert.equal(materializeProof({reference:'excerpt',segment:2,startLine:1,endLine:1},excerpt).quote,'tail');assert.throws(()=>materializeProof({reference:'excerpt',segment:1,startLine:1,endLine:2},excerpt),/exceeds/);assert.throws(()=>materializeProof({reference:'excerpt',quote:'head\ntail'},excerpt),/Quote/);assert.throws(()=>materializeProof({reference:'excerpt',segment:0,startLine:1,endLine:1},excerpt),/segment/);
});
test('JSON field pointers resolve actual quoted strings, newlines, escaped keys and arrays',()=>{
 const original={changes:[{text:'Say "hello"\nnext line'}],'slash/key':{'~key':false}},records=claimEvidence({prompt:{proposal:original}});
 const p=materializeProof({reference:'input:proposal',field:'/changes/0/text'},records);assert.equal(p.quote,JSON.stringify(original.changes[0].text));assert.equal(p.value,original.changes[0].text);assert.equal(materializeProof({reference:'input:proposal',field:'/slash~1key/~0key'},records).value,false);
 for(const field of ['/missing','/changes/01','/changes/length','/slash~2key'])assert.throws(()=>materializeProof({reference:'input:proposal',field},records),/field|pointer/);
 assert.throws(()=>materializeProof({reference:'input:proposal',field:'/changes/0/text',value:'invented'},records),/assertion/);assert.throws(()=>materializeProof({reference:'input:proposal',field:'/changes/0/text',quote:'invented'},records),/quote/);
 assert.throws(()=>materializeProof({reference:'unknown',field:'/x'},records),/Unknown proof reference/);
});
test('whole typed facts fill assertions while rejecting wrong supplied values and fact IDs',()=>{
 const records=[{reference:'count',factId:'count',value:2,text:'2'}];const p=materializeProof({reference:'count'},records);assert.equal(p.quote,'2');assert.equal(p.factId,'count');assert.equal(p.value,2);
 assert.throws(()=>materializeProof({reference:'count',value:0},records),/Structured fact/);assert.throws(()=>materializeProof({reference:'count',factId:'other'},records),/Structured fact/);assert.throws(()=>materializeProof({reference:'count',startLine:1,endLine:1},records),/whole reference/);
 const value={reason:'Two records exist',claims:[{path:'reason',kind:'observation',proofs:[{reference:'count'}]}]};assert.equal(validateClaimedResult(value,records),value);assert.equal(value.claims[0].text,value.reason);assert.equal(value.claims[0].proofs[0].value,2);
 assert.throws(()=>validateClaims([{kind:'observation',proofs:[{reference:'count'}]}],records),/requires text/);
});
test('materialized proofs stay bound to current full record and do not silently refresh',()=>{
 const old=[{reference:'file',text:'same selected line\nold context'}],p=materializeProof({reference:'file',startLine:1,endLine:1},old);assert.deepEqual(materializeProof(p,old),p);
 assert.throws(()=>materializeProof(p,[{reference:'file',text:'same selected line\nchanged context'}]),/source changed/);assert.throws(()=>materializeProof({...p,quote:'invented'},old),/quote/);
 assert.equal(materializeProof({reference:'file',quote:'same selected line'},old).quote,'same selected line');
});
test('native cached selector claims revalidate current source instead of recovering stale proof',async t=>{
 const root=await mkdtemp(join(tmpdir(),'proof-pointers-'));t.after(()=>rm(root,{recursive:true,force:true}));const directory=join(root,'review');await mkdir(directory);
 const record={reference:'count',factId:'count',text:'2',value:2},result={reason:'Two records exist',claims:[{path:'reason',kind:'observation',proofs:[{reference:'count'}]}]};validateClaimedResult(result,[record]);await writeFile(join(directory,'submission.json'),JSON.stringify(result));
 const input={directory,config:{},proofRequired:true,prompt:{projectSummaries:[{facts:[record]}]},seconds:5,validate:x=>x};assert.equal((await reviewSession(input)).recovered,true);
 // A source-only change invalidates the cached proof before provider construction.
 await assert.rejects(reviewSession({...input,prompt:{projectSummaries:[{facts:[{...record,sourceRefs:['new-snapshot']}]}]}}));
 const rejections=(await readFile(join(directory,'submission-rejections.jsonl'),'utf8')).trim().split('\n').map(JSON.parse);assert.equal(rejections[0].cached,true);assert.match(rejections[0].error,/source changed/);
});

test('native typed submission accepts compact selectors and derives exact output prose',async t=>{
 const root=await mkdtemp(join(tmpdir(),'native-proof-pointers-'));t.after(()=>rm(root,{recursive:true,force:true}));let requests=0;
 const text='Keep "This"\nOn Next Line',result={reason:text,claims:[{path:'reason',kind:'observation',proofs:[{reference:'input:proposal',field:'/changes/0/text'},{reference:'source',startLine:2,endLine:2},{reference:'count'}]}]};
 const server=createServer(async(req,res)=>{for await(const chunk of req){}requests++;res.writeHead(200,{'content-type':'text/event-stream'});const call=requests===1;const delta=call?{role:'assistant',tool_calls:[{index:0,id:'compact',type:'function',function:{name:'submit_review',arguments:JSON.stringify({result})}}]}:{role:'assistant',content:'Done'};res.write('data: '+JSON.stringify({id:'pointer',object:'chat.completion.chunk',created:1,model:'pointer',choices:[{index:0,delta,finish_reason:call?'tool_calls':'stop'}]})+'\n\ndata: [DONE]\n\n');res.end();});
 server.listen(0,'127.0.0.1');await once(server,'listening');t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));});const env='COMPACT_PROOF_FIXTURE',old=process.env[env];process.env[env]='fixture';t.after(()=>{if(old===undefined)delete process.env[env];else process.env[env]=old;});
 const actor={provider:'pointer-fixture',model:'pointer',credential_env:env,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'pointer',name:'pointer',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
 const out=await reviewSession({directory:join(root,'review'),config:{actor},proofRequired:true,seconds:10,prompt:{proposal:{changes:[{text}]},entries:[{id:'source',text:'First Line\nExact SECOND line'}],projectSummaries:[{facts:[{reference:'count',factId:'count',text:'1',value:1}]}]},validate:x=>x});
 assert.equal(requests,2);assert.equal(out.value.claims[0].text,text);assert.equal(out.value.claims[0].proofs[0].quote,JSON.stringify(text));assert.equal(out.value.claims[0].proofs[1].quote,'Exact SECOND line');assert.equal(out.value.claims[0].proofs[2].value,1);
});
