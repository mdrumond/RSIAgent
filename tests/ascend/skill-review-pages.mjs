import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {DatabaseSync} from 'node:sqlite';
import {State} from '../../controller/state.mjs';
import {reviewPages,reviewPageLimits} from '../../cli/skill-review-pages.mjs';
import {runReview} from '../../cli/skill-review.mjs';

const load=async path=>JSON.parse(await readFile(path,'utf8'));
const sha=text=>createHash('sha256').update(text).digest('hex');
const observation=(reference,projectId='p1')=>({reference,projectId,data:{message:'Wrong numeric result'},artifacts:[]});
const group=(id,count=1)=>({id,sourceId:'a3',type:'verification',message:'Wrong numeric result',
 occurrences:Array.from({length:count},(_,i)=>observation(id+':'+i)),linkedDiagnostics:[]});
async function directory(t){const path=await mkdtemp(join(tmpdir(),'review-pages-'));t.after(()=>rm(path,{recursive:true,force:true}));return path;}

test('pages retain original groups, all observations and successful linked diagnostics without increasing recurrence',async t=>{
 const source=[group('a',9),group('b',2)];source[0].linkedDiagnostics=[{reference:'positive',projectId:'p1',data:{passed:true},artifacts:[]}];
 const pages=await reviewPages(source,{directory:await directory(t),limits:{maxRecords:3,maxGroups:1,maxBytes:2048}});
 assert.ok(pages.length>1);assert.ok(pages.every(p=>p.metadata.bytes<=2048&&p.groups.length<=1));
 const refs=pages.flatMap(p=>p.groups.flatMap(g=>[...g.occurrences,...g.linkedDiagnostics].map(o=>o.reference)));
 assert.deepEqual(refs,source.flatMap(g=>[...g.occurrences,...g.linkedDiagnostics].map(o=>o.reference)));
 assert.equal(new Set(pages.flatMap(p=>p.metadata.projects)).size,1);
 assert.ok(pages.some(p=>p.groups.some(g=>g.linkedDiagnostics.some(o=>o.data.passed===true))));
 assert.equal(pages[0].metadata.continuation[0].total.occurrences,9);
 assert.equal(source[0].occurrences.length,9);
});
test('oversized record is copied completely behind an immutable scoped file pointer',async t=>{
 const source=[group('large')];source[0].occurrences[0].data.message='first\n'+'α'.repeat(35000)+'\nlast';
 const root=await directory(t),pages=await reviewPages(source,{directory:root,limits:{maxBytes:2048}});
 const record=pages[0].groups[0].occurrences[0],pointer=record.data.evidencePointer;
 assert.ok(pages[0].metadata.bytes<=2048);assert.equal(pointer.reference,source[0].occurrences[0].reference);
 const bytes=await readFile(pointer.path);assert.equal(sha(bytes),pointer.sha256);
 assert.deepEqual(JSON.parse(bytes),source[0].occurrences[0]);
 assert.ok(record.artifacts.some(a=>a.path===pointer.path&&a.sha256===pointer.sha256));
 const index=(await readFile(pointer.readableText.path,'utf8')).trim().split('\n').map(JSON.parse),parts=[];
 for(const item of index){assert.equal(item.field,'/data/message');assert.ok(item.bytes<=16384);const bytes=await readFile(item.path);assert.equal(sha(bytes),item.sha256);parts.push(bytes);}
 assert.equal(Buffer.concat(parts).toString('utf8'),source[0].occurrences[0].data.message);
 assert.ok(record.artifacts.some(a=>a.path===pointer.readableText.directory));
 assert.deepEqual((await reviewPages(source,{directory:root,limits:{maxBytes:2048}})).map(p=>p.id),pages.map(p=>p.id));
 await writeFile(pointer.path,'changed');await assert.rejects(reviewPages(source,{directory:root,limits:{maxBytes:2048}}),/Retained discovery record changed/);
});
test('page identities follow content rather than only IDs, and impossible metadata is explicit',async t=>{
 const root=await directory(t),source=[group('a')],first=await reviewPages(source,{directory:root});
 source[0].occurrences[0].data.message='Different failure';const changed=await reviewPages(source,{directory:root});
 assert.notEqual(first[0].id,changed[0].id);
 await assert.rejects(reviewPages([{...group('big'),message:'x'.repeat(3000)}],{directory:root,limits:{maxBytes:1024}}),/metadata exceeds/);
 await assert.rejects(reviewPages([{...group('dup'),occurrences:[observation('same'),observation('same')]}],{directory:root}),/Duplicate discovery reference/);
 assert.throws(()=>reviewPageLimits({maxBytes:0}));
});
test('linked-only continuation keeps its source project and attempt through actual relationship pointers',async t=>{
 const source=[group('a')];source[0].occurrences[0].attemptId='attempt';source[0].occurrences[0].relatedDiagnosticRefs=['positive'];
 source[0].linkedDiagnostics=[{reference:'positive',data:{passed:true},artifacts:[]}];
 const pages=await reviewPages(source,{directory:await directory(t),limits:{maxRecords:1}}),linked=pages[1].groups[0].linkedDiagnostics[0];
 assert.equal(pages[1].groups[0].occurrences.length,0);assert.equal(linked.projectId,'p1');assert.equal(linked.attemptId,'attempt');
 assert.deepEqual(linked.pageOriginRefs,['a:0']);assert.deepEqual(pages[1].metadata.projects,['p1']);
 assert.equal(source[0].linkedDiagnostics[0].projectId,undefined);
});

async function fixture(t){
 const root=await directory(t),source=join(root,'source');await mkdir(join(source,'skills/versions/baseline'),{recursive:true});
 await writeFile(join(source,'instance.json'),JSON.stringify({format:1,target:'a3',actor:{provider:'fixture',model:'fixture'},skills:{activeVersion:'baseline'}}));
 await writeFile(join(source,'skills/versions/baseline/guidance.json'),JSON.stringify([{id:'implementation-baseline',scope:'implementation',text:'Existing implementation',critical:true}]));
 new DatabaseSync(join(source,'knowledge.sqlite')).close();const state=new State(join(source,'state.sqlite'));
 state.add({id:'p1',goal:'Test source-bound pages',phase:'phase1',actor_seconds:60});const attempt=state.admit('p1',{pid:1,start:'fixture',boot:'fixture'});
 for(let i=0;i<11;i++)state.event(attempt.id,'failure-recorded',{message:'Repeated failure',detail:'same'});
 state.close();
 return {root,source,options:{sources:[{id:'a3',root:source}],output:join(root,'review'),seconds:7200,discoveryPages:{maxRecords:3,maxGroups:1,maxBytes:4096}}};
}
const classify=async({group,validate})=>validate({cause:'actor',coverage:'covered',guidanceIds:['implementation-baseline'],evidenceRefs:[group.occurrences[0].reference],reason:'Observed failure'});
const noChange=async({groups})=>({action:'no-change',reason:'No supported update',groupIds:groups.map(g=>g.id)});

test('runReview makes fresh page calls, persists coverage, and resumes without replaying finished pages',async t=>{
 const f=await fixture(t);for(const stage of ['inventory','classify'])await runReview({...f.options,discoveryPages:undefined,stage},{classify});
 const calls=[];const first=await runReview({...f.options,stage:'propose'},{review:async input=>{calls.push(input);return noChange(input);}});
 assert.equal(calls.length,4);assert.equal(new Set(calls.map(c=>c.directory)).size,4);
 assert.equal(new Set(calls.flatMap(c=>c.groups.flatMap(g=>g.occurrences.map(o=>o.reference)))).size,11);
 assert.equal(new Set(calls.flatMap(c=>c.groups.map(g=>g.id))).size,1);
 assert.equal(first.failureObservations,11);assert.equal(first.discoveryPages.length,4);
 const before=await load(join(f.options.output,'review.json'));
 await runReview({...f.options,stage:'propose'},{review:async()=>{throw Error('Finished page replayed');}});
 const after=await load(join(f.options.output,'review.json'));assert.equal(before.startedAt,after.startedAt);
 assert.deepEqual(after.discoveryPages,before.discoveryPages);
 await assert.rejects(runReview({...f.options,stage:'propose',discoveryPages:{maxRecords:4}},{review:noChange}),/Frozen discovery page limits changed/);
});
test('elapsed exhaustion prevents pending page provider calls without resetting the ledger',async t=>{
 const f=await fixture(t);for(const stage of ['inventory','classify'])await runReview({...f.options,stage},{classify});
 const path=join(f.options.output,'review.json'),state=await load(path);state.startedAt=Date.now()-7201000;await writeFile(path,JSON.stringify(state));
 let calls=0;await assert.rejects(runReview({...f.options,stage:'propose'},{review:async()=>{calls++;return noChange({groups:[]});}}),/elapsed/);
 const after=await load(path);assert.equal(calls,0);assert.equal(after.startedAt,state.startedAt);assert.ok(after.elapsedSeconds>=7200);
});
test('interrupted pending page resumes its own session without repeating completed pages',async t=>{
 const f=await fixture(t);for(const stage of ['inventory','classify'])await runReview({...f.options,stage},{classify});
 const abort=new AbortController(),seen=[];let pending;
 await assert.rejects(runReview({...f.options,stage:'propose',signal:abort.signal},{review:async input=>{
  seen.push(input.directory);
  if(seen.length===2){pending=input.directory;await writeFile(join(pending,'native-session.json'),JSON.stringify({path:'retained-page-session'}));abort.abort(Error('Interrupted page'));throw abort.signal.reason;}
  return noChange(input);
 }}),/Interrupted page/);
 const before=await load(join(f.options.output,'review.json'));assert.equal(Object.keys(before.reviews).length,1);
 const resumed=[];await runReview({...f.options,stage:'propose'},{review:async input=>{resumed.push(input.directory);return noChange(input);}});
 assert.equal(resumed.length,3);assert.equal(resumed[0],pending);assert.ok(!resumed.includes(seen[0]));
 assert.equal((await load(join(pending,'native-session.json'))).path,'retained-page-session');
 assert.equal((await load(join(f.options.output,'review.json'))).startedAt,before.startedAt);
});
test('legacy completed bucket and native session resumes retain their original context directory',async t=>{
 const f=await fixture(t);for(const stage of ['inventory','classify'])await runReview({...f.options,stage},{classify});
 const path=join(f.options.output,'review.json'),state=await load(path),key='a3:implementation-baseline';
 state.reviews[key]={action:'no-change',reason:'Completed before paging',groupIds:state.inventory.groups.map(g=>g.id),sourceId:'a3',guidanceId:'implementation-baseline',learningRevision:0};
 await writeFile(path,JSON.stringify(state));let calls=0;await runReview({...f.options,stage:'propose'},{review:async()=>{calls++;return noChange({groups:[]});}});assert.equal(calls,0);
 const saved=await load(path);delete saved.reviews[key];const old=join(f.options.output,'proposals',sha(JSON.stringify([key,0])));await mkdir(old,{recursive:true});await writeFile(join(old,'native-session.json'),JSON.stringify({path:'retained-session'}));await writeFile(path,JSON.stringify(saved));
 await runReview({...f.options,stage:'propose'},{review:async input=>{calls++;assert.equal(input.directory,old);assert.equal(input.groups[0].occurrences.length,11);return noChange(input);}});assert.equal(calls,1);
});
