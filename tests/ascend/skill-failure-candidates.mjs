import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,readdir,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {stages} from '../../controller/workflow.mjs';
import {skillInput} from '../../cli/skill-microeval.mjs';
import {readConfig} from '../../cli/instance.mjs';
import {publishSkillVersion} from '../../cli/skills.mjs';
import {runFailurePilot} from '../../cli/skill-failure-pilot.mjs';
import {qaTopics} from '../../cli/skill-review-qa.mjs';
import {createFailureDesign,publishFailureWinner} from '../../cli/skill-failure-candidates.mjs';
import {reviewSession} from '../../cli/skill-review-native.mjs';
import {validateClaimedResult} from '../../cli/claim-proofs.mjs';
import {scriptedReview} from './helpers/scripted-review.mjs';

const result=correct=>({correct,mechanismMatch:!correct,caseResults:[{id:'mechanism',correct},{id:'control',correct:true}],sourceSha256:'a'.repeat(64),artifacts:[],handles:[],failure:correct?null:{phase:'candidate',type:'CompilerFailure'}});
const task=id=>({id,prompt:'Repair the isolated operation',caseIds:['mechanism','control'],controlCaseIds:['control'],files:{'kernel.py':'retained failing source'},checker:'/private/oracle.py',controls:{reference:{files:{'kernel.py':'private reference'}}}});
const family=()=>({id:'retained',development:task('development'),transfer:[task('stride-transfer'),task('reduce-transfer')],evidence:[]});
const baseline=()=>Array.from({length:5},(_,i)=>result(i<2));
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'failure-candidates-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const base='base',directory=join(root,'skills/versions',base),scopeNames=['global-kdb',...stages],map=Object.fromEntries(scopeNames.map(scope=>[scope,scope+'.md']));await mkdir(directory,{recursive:true});
 const entries=scopeNames.map(scope=>({id:scope+'-base',scope,text:'Baseline '+scope+'.',critical:true,negative:false}));
 for(const entry of entries)await writeFile(join(directory,map[entry.scope]),'# '+entry.scope+'\n\n'+entry.text+'\n');
 await writeFile(join(directory,'skill-map.json'),JSON.stringify(map));await writeFile(join(directory,'guidance.json'),JSON.stringify(entries));
 const config={format:1,revision:0,root,target:'a3',skills:{activeVersion:base}},item={id:'item',base,objective:'correctness',groupIds:['failure'],evidence:{groups:[{id:'failure',occurrences:[{reference:'events:1',projectId:'project',attemptId:'attempt',data:{correct:false}}],linkedDiagnostics:[]}]},preparation:{classifications:{failure:{bucket:'correctness',cause:'actor'}}}};
 await writeFile(join(root,'instance.json'),JSON.stringify(config));const state=new State(join(root,'state.sqlite'));state.db.exec('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');state.close();
 const calls=[],qaCalls=[];let active=0,peak=0;
 const session=async args=>{
  active++;peak=Math.max(peak,active);calls.push(args);await new Promise(resolve=>setTimeout(resolve,5));active--;
  const id=args.directory.split('/').at(-2),value={text:'When a compiler cannot handle an implementation, consider a simpler representation while preserving the mathematical contract. '+id+'.',scope:'implementation',memoryRefs:['memory:chunk'],claims:[{path:'text',kind:'hypothesis',proofs:[{reference:'baseline-0',startLine:1,endLine:1}]}]},retrievals=[{entries:[{collection:'memory',chunk_id:'chunk',text:'A successful implementation preserved the contract using a bounded representation.'}]}];
  validateClaimedResult(value,args.proofRecords);return {value:args.validate(value,retrievals),retrievals};
 };
 const qaAnswer=async input=>{qaCalls.push(input);return {answers:Object.fromEntries(qaTopics.map(topic=>[topic,{text:'Evidence-supported applicability boundary',refs:['events:1']}]))};};
 const qaReview=async()=>({verdict:'approved',reason:'General instruction with a stated boundary'});
 const input={identities:{model:'fixed-model',runtime:'fixed-runtime',memory:'fixed-memory'},family:{id:'retained',development:family().development,evidence:[]},baselineResults:baseline(),directory:join(root,'design'),remainingSeconds:1200};
 return {root,config,item,input,calls,qaCalls,deps:{session,qaAnswer,qaReview},peak:()=>peak};
}
async function study(t,overrides={}){
 const f=await fixture(t),directory=join(f.root,'pilot'),manifest={baseline:(await skillInput(join(f.root,'skills/versions/base'))).text,identities:{model:'fixed-model',runtime:'fixed-runtime',memory:'fixed-memory'},families:[family()]};
 const report=await runFailurePilot({directory,manifest,trialSeconds:1,seconds:7200,concurrency:4},{qualify:async()=>({qualified:true,referencePassed:true,negativeFailed:true}),design:createFailureDesign({config:f.config,item:f.item},f.deps),trial:async input=>{
  if(input.phase==='screen')return baseline()[input.repetition];
  const correct=input.arm==='baseline'?input.repetition<2:input.arm==='principle-a'?true:input.arm==='principle-b'?input.repetition<4:input.repetition<3;
  return result(correct);
 },...overrides});return {...f,directory,report};
}

test('baseline gate prevents any instruction inference or candidate creation',async t=>{
 const f=await fixture(t),design=createFailureDesign({config:f.config,item:f.item},f.deps);
 await assert.rejects(design({...f.input,baselineResults:Array.from({length:5},()=>result(true))}),/matching failures/);assert.equal(f.calls.length,0);
 await assert.rejects(design({...f.input,baselineResults:[...baseline().slice(1),{infrastructure:true,correct:null,reason:'Unavailable'}]}),/clean baselines/);assert.equal(f.calls.length,0);
});
test('three writers run independently, cannot see controls, and freeze exact publishable guidance',async t=>{
 const f=await fixture(t),design=createFailureDesign({config:f.config,item:f.item},f.deps),rows=await design(f.input);
 assert.equal(rows.length,3);assert.equal(f.peak(),3);assert.equal(new Set(f.calls.map(call=>call.directory)).size,3);
 for(const call of f.calls){assert.deepEqual(Object.keys(call.prompt.development).sort(),['caseIds','files','id','prompt']);assert.equal(call.proofRequired,true);assert.equal(call.selectorOnly,true);assert.equal(call.knowledge,true);assert.equal(call.config.project_evidence_root,f.root);}
 for(const call of f.qaCalls){assert.deepEqual(call.groups[0].linkedDiagnostics.map(row=>row.data.correct),baseline().map(row=>row.correct));assert.equal(call.evaluationContract.development.controls,undefined);assert.equal(call.evaluationContract.transfer,undefined);}
 const original=await skillInput(join(f.root,'skills/versions/base'));
 for(const candidate of rows){const actual=await skillInput(join(f.root,'skills/versions',candidate.version));assert.equal(candidate.text,actual.text);assert.equal(candidate.skillSha256,actual.sha256);assert.equal(candidate.qa.verdict,'approved');assert(candidate.evidence.length);for(const [scope,text]of Object.entries(original.files))if(scope!=='implementation')assert.equal(actual.files[scope],text);}
 const calls=f.calls.length,versions=(await readdir(join(f.root,'skills/versions'))).length;assert.deepEqual(await design(f.input),rows);assert.equal(f.calls.length,calls);assert.equal((await readdir(join(f.root,'skills/versions'))).length,versions);
 assert.equal((await readConfig(f.root)).skills.activeVersion,'base');
});
test('QA revisions are frozen before testing rather than publishing an earlier draft',async t=>{
 const f=await fixture(t);f.deps.qaAnswer=async()=>({answers:Object.fromEntries(qaTopics.map(topic=>[topic,{text:'Supported',refs:['events:1']}])),changes:[{action:'upsert',id:'invalid',scope:'implementation',text:'Replacement',critical:false,negative:false,evidenceRefs:['events:1']}]});
 await assert.rejects(createFailureDesign({config:f.config,item:f.item},f.deps)(f.input),/additive|qa-execution-failed/);assert.equal((await readConfig(f.root)).skills.activeVersion,'base');
 const g=await fixture(t);g.deps.qaAnswer=async input=>({answers:Object.fromEntries(qaTopics.map(topic=>[topic,{text:'Supported',refs:['events:1']}])),changes:input.proposal.changes.map(change=>({...change,text:change.text+' Verify the preserved behavior independently.'}))});
 const candidates=await createFailureDesign({config:g.config,item:g.item},g.deps)(g.input);for(const candidate of candidates)assert.equal(candidate.instruction,candidate.proposal.changes[0].text);
});
test('invented memory is rejected before review and no versions publish',async t=>{
 const f=await fixture(t);f.deps.session=async args=>({value:args.validate({text:'Use bounded representations.',scope:'implementation',memoryRefs:['invented'],claims:[]},[]),retrievals:[]});
 await assert.rejects(createFailureDesign({config:f.config,item:f.item},f.deps)(f.input),/retrieved/);assert.equal(f.qaCalls.length,0);assert.deepEqual(await readdir(join(f.root,'skills/versions')),['base']);
});
test('native Pi submission accepts selector-only prose and preserves reviewed guidance',async t=>{
 const f=await fixture(t);let submitted=0;
 const provider=await scriptedReview(t,f.root,body=>{
  const tools=body.messages.filter(message=>message.role==='tool');
  if(!tools.length)return {name:'read',args:{path:'input.json'}};
  if(String(tools.at(-1).content).includes('Accepted. Finish'))return null;
  const value={text:'Keep the mathematical contract while selecting a representation suited to the execution environment. Consider independent checks before expanding implementation complexity. Variant '+(++submitted)+'.',scope:'implementation',memoryRefs:['memory:chunk'],claims:[{path:'text',kind:'hypothesis',proofs:[{reference:'baseline-0',startLine:1,endLine:1}]}]};return {name:'submit_review',args:{result:value}};
 });
 // Real Pi/proof/submission contract, with synthetic retrieved memory and no live KDB/provider.
 const retrievals=[{entries:[{collection:'memory',chunk_id:'chunk',text:'Successful bounded formulation'}]}];
 f.deps.session=async args=>{const reviewed=await reviewSession({...args,knowledge:false,config:{...args.config,actor:provider.profile,skill_writer_actor:provider.profile},validate:value=>args.validate(value,retrievals)});return {...reviewed,retrievals};};
 const rows=await createFailureDesign({config:f.config,item:f.item},f.deps)(f.input);assert.equal(rows.length,3);assert.equal(submitted,3);assert.equal(provider.requests(),9);for(const row of rows)assert.equal(row.text,(await skillInput(join(f.root,'skills/versions',row.version))).text);
});
test('memory transport failures exclude even an otherwise supported instruction',async t=>{
 const f=await fixture(t),session=f.deps.session;f.deps.session=async args=>{const value=await session(args),file=join(args.directory,'tainted.jsonl');await mkdir(args.directory,{recursive:true});await writeFile(file,JSON.stringify({message:{role:'toolResult',toolName:'mcp__kdb__search',isError:true,content:[{type:'text',text:'MCP request timed out after 60000ms'}]}})+'\n');return {...value,session:file};};
 await assert.rejects(createFailureDesign({config:f.config,item:f.item},f.deps)(f.input),/infrastructure-contaminated/);assert.equal(f.qaCalls.length,0);assert.deepEqual(await readdir(join(f.root,'skills/versions')),['base']);
});
test('scientific QA rejection is a completed disposition with no repeated inference',async t=>{
 const f=await fixture(t);f.deps.qaReview=async()=>({verdict:'learn',reason:'Evidence does not support a general instruction',learningRequests:[{question:'Need a different observed failure',hypothesis:'Current principle is already known',requiredEvidence:'Independent failure',learningTarget:'device'}]});
 const design=createFailureDesign({config:f.config,item:f.item},f.deps),out=await design(f.input);assert.equal(out.status,'rejected');const requests=f.calls.length;assert.deepEqual(await design(f.input),out);assert.equal(f.calls.length,requests);assert.equal((await readConfig(f.root)).skills.activeVersion,'base');
});
test('tampered reviewed guidance cannot be recovered or published',async t=>{
 const f=await fixture(t),design=createFailureDesign({config:f.config,item:f.item},f.deps),rows=await design(f.input),file=join(f.root,'skills/versions',rows[0].version,'skill-map.json'),map=JSON.parse(await readFile(file));await writeFile(join(f.root,'skills/versions',rows[0].version,map.implementation),'Changed after review');
 await assert.rejects(design(f.input),/changed after review/);
});
test('writer resumes spend only its retained balance and unknown elapsed custody blocks inference',async t=>{
 const f=await fixture(t);for(const id of ['a','b','c']){const directory=join(f.input.directory,'principle-'+id);await mkdir(directory,{recursive:true});await writeFile(join(directory,'writer-spend.json'),JSON.stringify({seconds:180,elapsedSeconds:100,running:false}));}
 await createFailureDesign({config:f.config,item:f.item},f.deps)(f.input);assert(f.calls.every(call=>call.seconds===80));
 const g=await fixture(t);for(const id of ['a','b','c']){const directory=join(g.input.directory,'principle-'+id);await mkdir(directory,{recursive:true});await writeFile(join(directory,'writer-spend.json'),JSON.stringify({seconds:180,elapsedSeconds:30,running:true}));}
 await assert.rejects(createFailureDesign({config:g.config,item:g.item},g.deps)(g.input),/elapsed custody is unknown/);assert.equal(g.calls.length,0);
});
test('a partial writer conversation cannot be reused for changed evidence or memory',async t=>{
 const f=await fixture(t);f.deps.session=async()=>{throw Error('Interrupted provider');};const design=createFailureDesign({config:f.config,item:f.item},f.deps);await assert.rejects(design(f.input),/Interrupted provider/);
 let calls=0;f.deps.session=async()=>{calls++;};await assert.rejects(createFailureDesign({config:f.config,item:f.item},f.deps)({...f.input,identities:{...f.input.identities,memory:'different'}}),/evidence changed/);assert.equal(calls,0);
});
test('actual pilot gates permit only its reviewed development winner to publish once',async t=>{
 const f=await study(t);assert.equal(f.report.promising,true);assert.equal(f.report.winner.id,'principle-a');const before=await readConfig(f.root),published=await publishFailureWinner({config:f.config,item:f.item,result:f.report,directory:f.directory});assert.equal(published.published,true);assert.equal(published.memoryRecorded,false);assert.equal((await readConfig(f.root)).skills.activeVersion,f.report.winner.version);assert.equal(before.skills.activeVersion,'base');
 const recovered=await publishFailureWinner({config:f.config,item:f.item,result:f.report,directory:f.directory},{publish:()=>{throw Error('Must not republish');}});assert.equal(recovered.published,true);assert.equal(recovered.recovered,true);
});
test('failed transfer, partial results and changed approval cannot publish',async t=>{
 const f=await study(t);const input={config:f.config,item:f.item,directory:f.directory};
 const partial=structuredClone(f.report);partial.confirmation['stride-transfer'].baseline.pop();await writeFile(join(f.directory,'pilot.json'),JSON.stringify(partial));await assert.rejects(publishFailureWinner({...input,result:partial}),/five trials/);
 const bad=structuredClone(f.report);bad.confirmation['reduce-transfer'][bad.winner.id]=Array.from({length:5},()=>result(false));await writeFile(join(f.directory,'pilot.json'),JSON.stringify(bad));await assert.rejects(publishFailureWinner({...input,result:bad}),/Both transfers/);
 const approval=structuredClone(f.report);approval.winner.qa.approval='unreviewed';approval.candidates.find(row=>row.id===approval.winner.id).qa.approval='unreviewed';await writeFile(join(f.directory,'pilot.json'),JSON.stringify(approval));await assert.rejects(publishFailureWinner({...input,result:approval}),/reviewed version/);assert.equal((await readConfig(f.root)).skills.activeVersion,'base');
});
test('changed active baseline refuses publication and a lost publisher acknowledgement recovers idempotently',async t=>{
 const f=await study(t);let calls=0;const stale=await publishFailureWinner({config:f.config,item:f.item,result:f.report,directory:f.directory},{readConfig:async()=>({skills:{activeVersion:'someone-else'}}),publish:()=>calls++});assert.equal(stale.published,false);assert.equal(calls,0);
 const g=await study(t),input={config:g.config,item:g.item,result:g.report,directory:g.directory};await assert.rejects(publishFailureWinner(input,{publish:async(root,request)=>{calls++;await publishSkillVersion(root,request);throw Error('Lost acknowledgement');}}),/Lost acknowledgement/);
 const recovered=await publishFailureWinner(input,{publish:()=>{throw Error('Duplicate');}});assert.equal(recovered.published,true);assert.equal(recovered.recovered,true);assert.equal(calls,1);
});
test('accepted storage recording happens only after CAS and recovers without republishing',async t=>{
 const f=await study(t);f.config.storage={enabled:true};let publicationCalls=0,recordCalls=0;const input={config:f.config,item:f.item,result:f.report,directory:f.directory};
 const dependencies={publish:async(root,request)=>{publicationCalls++;return publishSkillVersion(root,request);},record:async request=>{recordCalls++;assert.equal((await readConfig(f.root)).skills.activeVersion,f.report.winner.version);assert.equal(request.base,'base');assert.equal(request.candidate,f.report.winner.version);if(recordCalls===1)throw Error('Storage temporarily unavailable');}};
 await assert.rejects(publishFailureWinner(input,dependencies),/Storage temporarily/);assert.equal(publicationCalls,1);assert.equal(recordCalls,1);
 const recovered=await publishFailureWinner(input,dependencies);assert.equal(recovered.published,true);assert.equal(recovered.memoryRecorded,true);assert.equal(recordCalls,2);assert.equal(publicationCalls,1);
 await publishFailureWinner(input,dependencies);assert.equal(recordCalls,2);assert.equal(publicationCalls,1);
});
