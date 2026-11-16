import test from 'node:test';
import {Check} from 'typebox/value';
import assert from 'node:assert/strict';
import {mkdtemp,rm,readFile,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {reviewPages} from '../../cli/skill-review-pages.mjs';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {materializeProof} from '../../cli/project-summary.mjs';
import {runProposalQa,qaTopics} from '../../cli/skill-review-qa.mjs';
import {prepareQaSources,validateSummary,summaryItems,compactSummary} from '../../cli/skill-evidence-summary.mjs';
import {nativeQaAnswer,nativeQaReview,nativeQaAuditSummary,nativeQaSummarize,reviewSession} from '../../cli/skill-review-native.mjs';
async function fixture(t){const directory=await mkdtemp(join(tmpdir(),'evidence-summary-'));t.after(()=>rm(directory,{recursive:true,force:true}));return {directory,proposal:{reason:'Observed failure',changes:[{text:'Check source',evidenceRefs:['raw']}],memoryRefs:[]},entries:[],objective:'correctness',target:'fixture',groups:[{id:'g',occurrences:[{reference:'raw',type:'tool-result',message:'PASS one case\nunknown target'}]}],memoryEvidence:[],projectSummaries:[{facts:[{reference:'count',factId:'count',value:1,text:'1'}],references:[]}],learningProofEvidence:[],validateProposal:x=>x};}
const summary=text=>({items:[{text,kind:'observation'}],claims:[{path:'items.0.text',kind:'observation',proofs:[{reference:'raw',startLine:1,endLine:1}]}]});
const answers=()=>({answers:Object.fromEntries(qaTopics.map(k=>[k,{text:'Checked '+k,refs:['raw']}]))});
const approve=()=>({verdict:'approved',reason:'Supported'});
test('primary records stay outside compact prompt and summaries cannot prove themselves',async t=>{
 const f=await fixture(t);f.groups[0].occurrences[0].message+='\n'+'large source context'.repeat(20000);
 const c=await prepareQaSources(f),value=validateSummary(summary('A passing case is recorded'),c),items=summaryItems(value,c.pages[0].id),compact=compactSummary({items},c);
 assert.ok(JSON.stringify(compact).length<4000);assert.ok((await readFile(c.sources.find(s=>s.reference==='raw').segments[0].path,'utf8')).length>200000);
 await nativeQaAnswer({...f,evidenceSummary:{items},sourceCatalogue:c,validate:x=>x},{session:async options=>{assert.equal(options.compactEvidence,true);assert.equal(options.evidenceGroups.length,0);assert.ok(JSON.stringify(options.prompt).length<8000);assert.ok(options.proofRecords.some(r=>r.text.length>200000));assert.ok(options.allowedPaths.includes(c.directory));return {value:{}};}});
 assert.throws(()=>validateSummary({items:[{text:'False fact',kind:'observation'}],claims:[{path:'items.0.text',kind:'observation',proofs:[{reference:items[0].id,startLine:1,endLine:1}]}]},c),/Unknown proof/);
});
test('source-bound but false summary is returned before answers; each audit uses a fresh context',async t=>{
 const f=await fixture(t),dirs=[],events=[];let clock=1000,writes=0;
 const result=await runProposalQa(f,{now:()=>clock,qaSummarize:async x=>{events.push('summary');clock+=1000;return summary(writes++?'One case passed':'Zero cases passed');},qaAuditSummary:async x=>{dirs.push(x.directory);events.push('audit');clock+=1000;return {verdict:x.item.text==='Zero cases passed'?'revise':'supported',reason:'Compare the observed PASS'};},qaAnswer:async x=>{events.push('answers');assert.equal(x.evidenceSummary.items[0].text,'One case passed');return answers();},qaReview:async()=>{events.push('review');return approve();}});
 assert.equal(result.verdict,'approved');assert.deepEqual(events,['summary','audit','summary','audit','answers','review']);assert.equal(new Set(dirs).size,2);assert.equal(result.summary.pages[Object.keys(result.summary.pages)[0]].revisions.length,1);
});
test('summary audit sees source provenance and requires direct primary file access',async t=>{
 const f=await fixture(t);f.groups[0].occurrences[0].type='source-code';f.groups[0].occurrences[0].message='// This backend is broken';
 const c=await prepareQaSources(f),item=summaryItems(validateSummary(summary('A code comment claims a backend defect'),c),c.pages[0].id)[0];
 await nativeQaAuditSummary({...f,item,sourceCatalogue:c,validate:x=>x},{session:async options=>{assert.equal(options.prompt.sources[0].provenance,'code-text');assert.ok(options.requiredReads.includes(c.sources.find(s=>s.reference==='raw').segments[0].path));assert.equal(options.proofRecords.some(r=>r.reference===item.id),false);assert.equal(options.prompt.item.proofs[0].quote,'// This backend is broken');return {value:options.validate({verdict:'revise',reason:'Comment is not execution evidence'})};}});
 await nativeQaSummarize({...f,page:c.pages[0],sourceCatalogue:c,validate:x=>x},{session:async options=>{assert.equal(options.compactEvidence,true);assert.ok(options.prompt.authoritativeFacts.some(s=>s.reference==='count'&&s.value===1));return {value:{}};}});
});
test('source-only changes invalidate summary and approval while preserving the elapsed ledger',async t=>{
 const f=await fixture(t),artifact=join(f.directory,'log.txt');await writeFile(artifact,'same line\nold context');f.groups[0].occurrences[0].artifacts=[{path:artifact}];let clock=1000,produced=0,audited=0;
 const ops={now:()=>clock,qaSummarize:async()=>{produced++;clock+=1000;return summary('One case passed');},qaAuditSummary:async()=>{audited++;clock+=1000;return {verdict:'supported',reason:'Primary supports it'};},qaAnswer:async()=>answers(),qaReview:async()=>approve()};
 const first=await runProposalQa(f,ops);const firstCount=produced;await writeFile(artifact,'same line\nnew context');const second=await runProposalQa(f,ops);
 assert.equal(second.verdict,'approved');assert.ok(produced>firstCount);assert.ok(audited>=2);assert.notEqual(first.sourceIdentity,second.sourceIdentity);assert.notEqual(first.approval,second.approval);assert.ok(second.elapsedSeconds>=first.elapsedSeconds);assert.equal(second.proofUpgrades.at(-1).verdict,'approved');
});
test('legacy 85-second approval gets only215seconds; expired300-second QA calls no providers',async t=>{
 for(const spent of [85,300]){
  const f=await fixture(t);let clock=1000;
  await runProposalQa(f,{now:()=>clock,qaAnswer:async()=>{clock+=spent*1000;return answers();},qaReview:async()=>approve()});
  const path=join(f.directory,'qa.json'),old=JSON.parse(await readFile(path));old.elapsedSeconds=spent;await writeFile(path,JSON.stringify(old));let calls=0,allowance;
  const out=await runProposalQa(f,{now:()=>clock,qaSummarize:async x=>{calls++;allowance=x.seconds;return summary('One case passed');},qaAuditSummary:async()=>({verdict:'supported',reason:'Checked'}),qaAnswer:async()=>answers(),qaReview:async()=>approve()});
  if(spent===85){assert.equal(allowance,215);assert.equal(out.verdict,'approved');}else{assert.equal(calls,0);assert.equal(out.verdict,'inconclusive');assert.equal(out.reason,'qa-elapsed');}
  assert.ok(out.elapsedSeconds>=spent);assert.equal(out.seconds,300);
 }
});
test('interrupted summary audits resume completed work and preserve all elapsed charges',async t=>{
 const f=await fixture(t),stop=new AbortController();let clock=1000,produced=0;
 await assert.rejects(runProposalQa({...f,signal:stop.signal},{now:()=>clock,qaSummarize:async()=>{produced++;clock+=10000;return summary('One case passed');},qaAuditSummary:async()=>{clock+=20000;stop.abort(Error('Pause'));throw stop.signal.reason;}}),/Pause/);
 let left;const out=await runProposalQa(f,{now:()=>clock,qaSummarize:()=>{throw Error('Repeated completed summary');},qaAuditSummary:async x=>{left=x.seconds;return {verdict:'supported',reason:'Checked'};},qaAnswer:async()=>answers(),qaReview:async()=>approve()});
 assert.equal(produced,1);assert.equal(left,270);assert.ok(out.elapsedSeconds>=30);assert.equal(out.verdict,'approved');
});
test('writer can revise the proposal reason with a source-bound claim',async t=>{
 const f=await fixture(t);const out=await runProposalQa(f,{qaAnswer:async()=>({...answers(),reason:'One passing case is recorded',claims:[{path:'reason',text:'One passing case is recorded',kind:'observation',proofs:[{reference:'raw',startLine:1,endLine:1}]}]}),qaReview:async x=>{assert.equal(x.proposal.reason,'One passing case is recorded');return approve();}});assert.equal(out.proposal.reason,'One passing case is recorded');
});

test('huge single-line sources have bounded literal pages with complete byte recovery',async t=>{
 const f=await fixture(t),text='α'.repeat(60000);f.groups[0].occurrences[0].message=text;
 const c=await prepareQaSources(f),aliases=c.sources.filter(s=>s.parent?.reference==='raw');assert.ok(aliases.length>4);
 let reconstructed='';for(const s of aliases){const part=await readFile(s.segments[0].path,'utf8');assert.ok(Buffer.byteLength(part)<=16384);reconstructed+=part;}
 assert.equal(reconstructed,c.records.find(r=>r.reference==='raw').text);for(const p of c.pages)assert.ok(p.sources.reduce((n,s)=>n+s.segments.reduce((n,x)=>n+x.bytes,0),0)<=16384);
 const alias=aliases[0];assert.doesNotThrow(()=>validateSummary({items:[{text:'Literal source',kind:'observation'}],claims:[{path:'items.0.text',kind:'observation',proofs:[{reference:alias.reference,startLine:1,endLine:1}]}]},c));
 assert.throws(()=>validateSummary(summary('x'.repeat(9000)),c),/byte limit/);
});
test('summary pages select cited evidence and linked diagnostics while keeping full primary catalogue',async t=>{
 const f=await fixture(t);f.groups[0].occurrences.push({reference:'irrelevant',message:'uncited metadata'});f.groups[0].linkedDiagnostics=[{reference:'negative',message:'contradictory diagnostic'}];
 const c=await prepareQaSources(f),refs=c.pages.flatMap(p=>p.sources.map(s=>s.reference));assert.ok(refs.includes('raw'));assert.ok(refs.includes('negative'));assert.ok(!refs.includes('irrelevant'));assert.ok(c.records.some(r=>r.reference==='irrelevant'));
});

test('native audit rejects submission until a stock read opens its primary source',async t=>{
 const f=await fixture(t),primary=join(f.directory,'primary.txt');await writeFile(primary,'literal primary\nunselected private source context');let requests=0;
 const result={verdict:'supported',reason:'Recorded text',claims:[{path:'reason',kind:'observation',proofs:[{reference:'raw',startLine:1,endLine:1}]}]};
 const server=createServer(async(req,res)=>{for await(const chunk of req){}requests++;const call=requests<5,tool=requests===3?'read':'submit_review',args=tool==='read'?{path:primary}:{result:requests===2?{...result,claims:[{...result.claims[0],path:'item.text'}]}:result};res.writeHead(200,{'content-type':'text/event-stream'});res.end('data: '+JSON.stringify({id:'audit',object:'chat.completion.chunk',created:1,model:'audit',choices:[{index:0,delta:call?{role:'assistant',tool_calls:[{index:0,id:'call'+requests,type:'function',function:{name:tool,arguments:JSON.stringify(args)}}]}:{role:'assistant',content:'Done'},finish_reason:call?'tool_calls':'stop'}]})+'\n\ndata: [DONE]\n\n');});
 server.listen(0,'127.0.0.1');await once(server,'listening');t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));});
 const env='SUMMARY_AUDIT_FIXTURE',old=process.env[env];process.env[env]='fixture';t.after(()=>{if(old===undefined)delete process.env[env];else process.env[env]=old;});
 const actor={provider:'audit-fixture',model:'audit',credential_env:env,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'audit',name:'audit',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}},directory=join(f.directory,'native');
 const catalogue={directory:f.directory,records:[{reference:'raw',text:'literal primary\nunselected private source context'}],sources:[{reference:'raw',segments:[{path:primary}]}],facts:[],pages:[]};const auditInput={directory,config:{actor},sourceCatalogue:catalogue,item:{text:'Recorded primary',kind:'observation',proofs:[{reference:'raw',startLine:1,endLine:1,quote:'literal primary'}]},seconds:10,validate:x=>x};const out=await nativeQaAuditSummary(auditInput,{session:async options=>{assert.equal(Check(options.resultSchema,result),true);assert.equal(Check(options.resultSchema,{...result,claims:[{...result.claims[0],path:'item.text'}]}),false);assert.equal(Check(options.resultSchema,{...result,claims:[{...result.claims[0],text:result.reason}]}),false);assert.equal(Check(options.resultSchema,{...result,claims:[{...result.claims[0],proofs:[{...result.claims[0].proofs[0],quote:'literal primary'}]}]}),false);return reviewSession(options);}});
 assert.equal(requests,5);assert.equal(out.value.claims[0].proofs[0].quote,'literal primary');assert.ok(!(await readFile(join(directory,'input.json'),'utf8')).includes('unselected private source context'));const rejected=(await readFile(join(directory,'submission-rejections.jsonl'),'utf8')).trim().split('\n').map(JSON.parse);assert.equal(rejected.length,1);assert.match(rejected[0].error,/Read these primary/);assert.equal(out.value.claims[0].text,'Recorded text');const recovered=await nativeQaAuditSummary({...auditInput,config:{}});assert.equal(recovered.recovered,true);assert.deepEqual(JSON.parse(await readFile(join(directory,'read-evidence.json'),'utf8')),['file:'+primary]);
});

test('discovery paging preserves middle primary evidence, scoped access and stale-source binding',async t=>{
 const f=await fixture(t),marker='DEVICE_MIDDLE_VERIFIED_18923';f.groups[0].occurrences[0].message='prefix '.repeat(10000)+'\n'+marker+'\n'+'suffix '.repeat(10000);f.groups[0].occurrences[0].data={sourceCall:{tool:'rsi_target'}};
 const pages=await reviewPages(f.groups,{directory:join(f.directory,'discovery')}),groups=pages.flatMap(p=>p.groups),c=await prepareQaSources({...f,groups});
 const middle=c.records.find(r=>r.parent?.reference==='raw'&&r.text.includes(marker));assert.ok(middle);assert.equal(middle.provenance,'target-tool-record');assert.ok(c.pages.some(p=>p.sources.some(s=>s.reference===middle.reference)));
 const line=middle.text.split('\n').findIndex(s=>s.includes(marker))+1,value={items:[{text:'A device result is recorded in the middle',kind:'observation'}],claims:[{path:'items.0.text',kind:'observation',proofs:[{reference:middle.reference,startLine:line,endLine:line}]}]};validateSummary(value,c);
 const item=summaryItems(value,c.pages.find(p=>p.sources.some(s=>s.reference===middle.reference)).id)[0];
 await nativeQaAuditSummary({...f,item,sourceCatalogue:c,validate:x=>x},{session:async options=>{for(const path of options.requiredReads){assert.ok(path.startsWith(c.directory+'/'));assert.ok((await readFile(path,'utf8')).includes(marker));}assert.ok(options.prompt.item.proofs[0].quote.includes(marker));return {value:{}};}});
 f.groups[0].occurrences[0].message+=' changed later context';const nextPages=await reviewPages(f.groups,{directory:join(f.directory,'discovery-next')}),next=await prepareQaSources({...f,groups:nextPages.flatMap(p=>p.groups)});assert.notEqual(next.identity,c.identity);assert.throws(()=>validateSummary(value,next),/source changed/);
 const pointer=groups[0].occurrences[0].data.evidencePointer;await writeFile(pointer.path,(await readFile(pointer.path,'utf8')).replace(marker,'DIFFERENT_RESULT'));
 await assert.rejects(prepareQaSources({...f,groups}),/primary source changed/);
});

test('native compact summary accepts selector paths and restores normalized cached results',async t=>{
 const f=await fixture(t),c=await prepareQaSources(f),value=summary('One passing case is recorded');await writeFile(join(f.directory,'submission.json'),JSON.stringify(value));
 const input={...f,config:{},sourceCatalogue:c,page:c.pages[0],seconds:5,validate:x=>validateSummary(x,c)},first=await nativeQaSummarize(input);assert.equal(first.recovered,true);assert.equal(first.value.claims[0].text,value.items[0].text);assert.equal(first.value.claims[0].proofs[0].quote,'PASS one case');
 const second=await nativeQaSummarize(input);assert.equal(second.recovered,true);assert.deepEqual(second.value,first.value);
});
test('selector-schema migration archives old semantic work and retains in-flight elapsed reservation',async t=>{
 const f=await fixture(t),ops={now:()=>1000,qaSummarize:async()=>summary('One case passed'),qaAuditSummary:async()=>({verdict:'supported',reason:'Checked'}),qaAnswer:async()=>answers(),qaReview:async()=>approve()};await runProposalQa(f,ops);
 const path=join(f.directory,'qa.json'),old=JSON.parse(await readFile(path,'utf8'));old.proofProtocol='primary-summary-v2';old.elapsedSeconds=225;old.inFlight={reservedThrough:230,stage:'qaAuditSummary'};await writeFile(path,JSON.stringify(old));let available;
 const out=await runProposalQa(f,{...ops,qaSummarize:async x=>{available=x.seconds;return summary('One case passed');}});assert.equal(available,70);assert.equal(out.elapsedSeconds,230);assert.ok(out.proofUpgrades.at(-1).summary);assert.ok(out.proofUpgrades.at(-1).rounds.length);assert.equal(out.verdict,'approved');
});

test('long cited memory spans reach summary pages and primary audit selectors',async t=>{
 const f=await fixture(t),reference='good:long-memory',marker='MIDDLE_MEMORY_OBSERVATION';f.proposal.memoryRefs=[reference];f.memoryEvidence=[{ref:reference,text:'prefix '.repeat(5000)+'\n'+marker+'\n'+'suffix '.repeat(5000)}];
 const c=await prepareQaSources(f),record=c.records.find(r=>r.parent?.reference===reference&&r.text.includes(marker)),page=c.pages.find(p=>p.sources.some(s=>s.reference===record?.reference));assert.ok(page);assert.equal(record.provenance,'actor-memory-report');
 const line=record.text.split('\n').findIndex(s=>s.includes(marker))+1,value={items:[{text:'Memory reports an observation',kind:'observation'}],claims:[{path:'items.0.text',kind:'observation',proofs:[{reference:record.reference,startLine:line,endLine:line}]}]};validateSummary(value,c);const item=summaryItems(value,page.id)[0];
 await nativeQaAuditSummary({...f,item,sourceCatalogue:c,validate:x=>x},{session:async options=>{assert.equal(options.prompt.item.proofs[0].quote,marker);assert.equal(options.prompt.sources[0].provenance,'actor-memory-report');assert.ok(options.requiredReads.length);for(const path of options.requiredReads)assert.ok((await readFile(path,'utf8')).includes(marker));return {value:{verdict:'supported'}};}});
});
test('advertised primary line ranges match proof renderer with and without terminal newline',async t=>{
 for(const text of ['first\nsecond','first\nsecond\n']){const f=await fixture(t),reference='good:lines';f.proposal.memoryRefs=[reference];f.memoryEvidence=[{ref:reference,text}];const c=await prepareQaSources(f),source=c.sources.find(s=>s.reference===reference),segment=source.segments[0];
  const proof=materializeProof({reference,startLine:1,endLine:segment.lines},c.records);assert.equal(proof.quote,'first\nsecond');assert.throws(()=>materializeProof({reference,startLine:1,endLine:segment.lines+1},c.records),/exceeds/);
 }
});

test('compact final review keeps normalized answer proof text out of inline context',async t=>{
 const f=await fixture(t),raw='hidden primary '.repeat(10000);f.groups[0].occurrences[0].message=raw;const c=await prepareQaSources(f),value=validateSummary({items:[{text:'Recorded evidence',kind:'observation'}],claims:[{path:'items.0.text',kind:'observation',proofs:[{reference:'raw',quote:'hidden primary'}]}]},c),items=summaryItems(value,c.pages[0].id),answerClaims=[{path:'answers.evidence.text',text:'Observed text',kind:'observation',proofs:[materializeProof({reference:'raw',startLine:1,endLine:1},c.records)]}];
 await nativeQaReview({...f,sourceCatalogue:c,evidenceSummary:{items},answers:answers().answers,answerClaims,validate:x=>x},{session:async options=>{assert.ok(JSON.stringify(options.prompt).length<12000);assert.deepEqual(options.prompt.answerClaims[0].proofs,[{reference:'raw',startLine:1,endLine:1,segment:1}]);assert.equal(Object.hasOwn(options.prompt.answerClaims[0],'text'),false);assert.ok(options.proofRecords.some(r=>r.reference==='raw'&&r.text===raw));assert.equal(materializeProof({reference:'raw',quote:'hidden primary'},options.proofRecords).quote,'hidden primary');assert.throws(()=>materializeProof({reference:'raw',quote:'invented execution'},options.proofRecords),/Quote/);return {value:{}};}});
});

test('verified learning file aliases enter summary and audit without admitting unavailable or unrelated sources',async t=>{
 const f=await fixture(t),reference='file:/verified/experiment.log',marker='VERIFIED_LEARNING_MIDDLE';f.learningProofEvidence=[{reference,text:'prefix '.repeat(5000)+'\n'+marker+'\n'+'suffix '.repeat(5000),learningId:'verified-one',status:'verified-device'},{reference:'file:/unavailable',text:'not available',available:false}];f.projectSummaries[0].references=[{reference:'learning:unrelated',text:'not a verified learning source'}];
 const c=await prepareQaSources(f),record=c.records.find(r=>r.parent?.reference===reference&&r.text.includes(marker)),page=c.pages.find(p=>p.sources.some(s=>s.reference===record?.reference));assert.ok(page);assert.ok(!c.pages.some(p=>p.sources.some(s=>['file:/unavailable','learning:unrelated'].includes(s.reference))));assert.ok(!c.records.some(r=>r.reference==='file:/unavailable'));
 const line=record.text.split('\n').findIndex(s=>s.includes(marker))+1,value={items:[{text:'Verified learning records an observation',kind:'observation'}],claims:[{path:'items.0.text',kind:'observation',proofs:[{reference:record.reference,startLine:line,endLine:line}]}]};validateSummary(value,c);
 await nativeQaAuditSummary({...f,item:summaryItems(value,page.id)[0],sourceCatalogue:c,validate:x=>x},{session:async options=>{assert.equal(options.prompt.item.proofs[0].quote,marker);assert.equal(options.prompt.sources[0].provenance,'learning-artifact');for(const path of options.requiredReads)assert.ok((await readFile(path,'utf8')).includes(marker));return {value:{}};}});
});
