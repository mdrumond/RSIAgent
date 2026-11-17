import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {setTimeout as pause} from 'node:timers/promises';
import {runProposalQa,qaTopics} from '../../cli/skill-review-qa.mjs';
async function fixture(t,count=6){
 const directory=await mkdtemp(join(tmpdir(),'parallel-qa-'));t.after(()=>rm(directory,{recursive:true,force:true}));
 const occurrences=Array.from({length:count},(_,i)=>({reference:'raw'+i,message:('primary '+i+'\n').repeat(1200)}));
 return {directory,qaConcurrency:4,proposal:{changes:[{text:'Use the observed procedure',evidenceRefs:occurrences.map(o=>o.reference)}],memoryRefs:[]},entries:[],groups:[{id:'g',occurrences}],projectSummaries:[],learningProofEvidence:[],target:'a3',objective:'correctness',validateProposal:x=>x};
}
const answers=()=>({answers:Object.fromEntries(qaTopics.map(k=>[k,{text:'Observed '+k,refs:['raw0']}]))});
function summary(page,count=2){return {items:Array.from({length:count},(_,i)=>({text:'Observed item '+i,kind:'observation'})),claims:Array.from({length:count},(_,i)=>({path:`items.${i}.text`,kind:'observation',proofs:[{reference:page.sources[0].reference,startLine:1,endLine:1}]}))};}
const approved=()=>({verdict:'approved',reason:'Supported by evidence'});
test('four sessions overlap across pages and audits; writing waits for all audits and preserves catalogue order',async t=>{
 const input=await fixture(t);let active=0,peak=0,audits=0,pages=0;
 const session=async work=>{active++;peak=Math.max(peak,active);try{await pause(8);return work();}finally{active--;}};
 const out=await runProposalQa(input,{qaSummarize:({page})=>session(()=>{pages++;return summary(page);}),qaAuditSummary:()=>session(()=>{audits++;return {verdict:'supported',reason:'Primary agrees'};}),qaAnswer:async x=>{
  assert.equal(active,0);assert.equal(audits,pages*2);assert.equal(x.evidenceSummary.items.length,audits);
  assert.deepEqual(x.evidenceSummary.items.map(v=>v.text),x.sourceCatalogue.pages.flatMap(()=>['Observed item 0','Observed item 1']));return answers();
 },qaReview:async()=>approved()});
 assert.equal(out.verdict,'approved');assert.equal(peak,4);assert.equal(active,0);
});
test('default scheduling is serial and rejects invalid concurrency before provider calls',async t=>{
 const input=await fixture(t,2);delete input.qaConcurrency;let active=0,peak=0;
 const tracked=async result=>{active++;peak=Math.max(peak,active);await pause(2);active--;return result;};
 const ops={qaSummarize:({page})=>tracked(summary(page)),qaAuditSummary:()=>tracked({verdict:'supported',reason:'Checked'}),qaAnswer:()=>tracked(answers()),qaReview:()=>tracked(approved())};
 assert.equal((await runProposalQa(input,ops)).verdict,'approved');assert.equal(peak,1);
 for(const qaConcurrency of [0,-1,1.5,Infinity])await assert.rejects(runProposalQa({...input,qaConcurrency},{}),/positive integer/);
});
test('page revisions wait for every own audit; other pages continue independently',async t=>{
 const input=await fixture(t,3),counts=new Map(),events=[];let first;
 const out=await runProposalQa(input,{qaSummarize:async({page,previousReview})=>{
  first??=page.id;const n=(counts.get(page.id)??0)+1;counts.set(page.id,n);
  if(page.id===first&&n===2){assert.equal(previousReview.length,2);assert(events.includes('slow-audit-finished'));}
  const value=summary(page);if(n>1)for(const item of value.items)item.text+=' corrected';return value;
 },qaAuditSummary:async({item})=>{
  if(item.pageId===first&&item.text.endsWith('1')){await pause(15);events.push('slow-audit-finished');}
  const verdict=item.pageId===first&&counts.get(first)===1?'revise':'supported';
  return {verdict,reason:'Checked'};
 },qaAnswer:async()=>answers(),qaReview:async()=>approved()});
 assert.equal(out.verdict,'approved');assert.equal(counts.get(first),2);assert.equal(Object.values(out.summary.pages).filter(p=>p.revisions.length).length,1);
});
test('sibling failure stops admission, settles running calls, and preserves completed audits for resume',async t=>{
 const input=await fixture(t,2);let active=0,started=0,completed=0,failed=false;
 const ops={qaSummarize:async({page})=>summary(page,1),qaAuditSummary:async()=>{
  active++;const order=++started;try{await pause(order===1?10:25);if(order===1&&!failed){failed=true;throw Error('Provider unavailable');}completed++;return {verdict:'supported',reason:'Primary read completed'};}finally{active--;}
 },qaAnswer:async()=>{throw Error('Must not write after a failed audit');},qaReview:async()=>approved()};
 const first=await runProposalQa(input,ops);assert.equal(first.reason,'qa-execution-failed');assert.equal(active,0);assert.equal(completed,1);
 const saved=JSON.parse(await readFile(join(input.directory,'qa.json')));assert.equal(Object.keys(saved.summary.audits).length,1);assert.equal(saved.inFlightOperations,undefined);
 const second=await runProposalQa(input,{...ops,qaAnswer:async()=>answers()});assert.equal(second.verdict,'approved');assert.equal(started,3);assert.equal(active,0);
});
test('parallel reservations charge maximum elapsed wall time, preserve legacy markers, and exclude paused time',async t=>{
 const input=await fixture(t,4);let clock=0,started=0;const waiters=[];
 const out=await runProposalQa(input,{now:()=>clock,qaSummarize:async({page})=>summary(page,1),qaAuditSummary:async()=>{
  started++;await new Promise(resolve=>{waiters.push(resolve);if(started===4)setImmediate(async()=>{
   const checkpoint=JSON.parse(await readFile(join(input.directory,'qa.json')));assert.equal(Object.keys(checkpoint.inFlightOperations).length,4);
   clock=20000;for(const done of waiters)done();
  });});return {verdict:'supported',reason:'Checked'};
 },qaAnswer:async()=>answers(),qaReview:async()=>approved()});
 assert.equal(out.elapsedSeconds,20);
 const file=join(input.directory,'qa.json'),saved=JSON.parse(await readFile(file));
 saved.elapsedSeconds=100;saved.inFlight={stage:'qaReview',reservedThrough:150};saved.inFlightOperations={one:{stage:'qaReview',reservedThrough:160},two:{stage:'qaReview',reservedThrough:155}};
 saved.verdict='inconclusive';saved.retryable=true;delete saved.rounds[0].review;await writeFile(file,JSON.stringify(saved));clock=90000000;
 const resumed=await runProposalQa(input,{now:()=>clock,qaSummarize:()=>{throw Error('Must reuse summaries');},qaAuditSummary:()=>{throw Error('Must reuse audits');},qaReview:async({seconds})=>{assert.equal(seconds,140);return approved();}});
 assert.equal(resumed.elapsedSeconds,160);assert.equal(resumed.verdict,'approved');assert.equal(resumed.inFlight,undefined);assert.equal(resumed.inFlightOperations,undefined);
});
test('caller cancellation drains running sessions and never admits queued work',async t=>{
 const input=await fixture(t,6),stop=new AbortController();input.qaConcurrency=2;let active=0,started=0,finished=0;
 await assert.rejects(runProposalQa({...input,signal:stop.signal},{qaSummarize:async({page,signal})=>{
  active++;started++;try{
   if(started===2)setImmediate(()=>stop.abort(Error('Pause QA')));
   await new Promise((resolve,reject)=>signal.addEventListener('abort',()=>{setTimeout(()=>reject(signal.reason),10);},{once:true}));
   return summary(page);
  }finally{active--;finished++;}
 },qaAuditSummary:async()=>{throw Error('No audits after cancellation');}}),/Pause QA/);
 assert.equal(started,2);assert.equal(finished,2);assert.equal(active,0);
 const saved=JSON.parse(await readFile(join(input.directory,'qa.json')));assert.equal(saved.inFlightOperations,undefined);
});
test('audit completion order does not change approval for the same evidence and answers',async t=>{
 const input=await fixture(t,3),ops={now:()=>0,qaSummarize:async({page})=>summary(page),qaAnswer:async()=>answers(),qaReview:async()=>approved()};let reverse=false;
 const audit=async({item})=>{await pause(item.text.endsWith(reverse?'0':'1')?10:1);return {verdict:'supported',reason:'Checked'};};
 const first=await runProposalQa(input,{...ops,qaAuditSummary:audit});
 const path=join(input.directory,'qa.json'),saved=JSON.parse(await readFile(path));delete saved.verdict;delete saved.approval;delete saved.rounds[0].answers;delete saved.rounds[0].review;saved.summary.audits={};await writeFile(path,JSON.stringify(saved));reverse=true;
 const second=await runProposalQa(input,{...ops,qaAuditSummary:audit});assert.equal(first.approval,second.approval);
});
