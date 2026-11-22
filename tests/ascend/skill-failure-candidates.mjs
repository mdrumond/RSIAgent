import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,readdir,rm} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {stages} from '../../controller/workflow.mjs';
import {skillInput} from '../../cli/skill-microeval.mjs';
import {readConfig} from '../../cli/instance.mjs';
import {publishSkillVersion} from '../../cli/skills.mjs';
import {runFailurePilot,validatePilotTrial} from '../../cli/skill-failure-pilot.mjs';
import {qaTopics} from '../../cli/skill-review-qa.mjs';
import {createFailureDesign,publishFailureWinner,verifyPanelApproval} from '../../cli/skill-failure-candidates.mjs';
import {reviewSession} from '../../cli/skill-review-native.mjs';
import {validateClaimedResult} from '../../cli/claim-proofs.mjs';
import {scriptedReview} from './helpers/scripted-review.mjs';
import {timeoutReviewStream} from './helpers/review-timeout.mjs';

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
 const input={baseline:(await skillInput(directory)).text,identities:{model:'fixed-model',runtime:'fixed-runtime',memory:'fixed-memory'},family:{id:'retained',development:family().development,evidence:[]},baselineResults:baseline(),directory:join(root,'design'),remainingSeconds:1200};
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
test('different tested baseline refuses inference before any writer starts',async t=>{
 const f=await fixture(t);await assert.rejects(createFailureDesign({config:f.config,item:f.item},f.deps)({...f.input,baseline:f.input.baseline+'Additional guidance'}),/Tested baseline guidance differs/);assert.equal(f.calls.length,0);assert.equal(f.qaCalls.length,0);assert.deepEqual(await readdir(join(f.root,'skills/versions')),['base']);
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
 const disposition=await createFailureDesign({config:f.config,item:f.item},f.deps)(f.input);assert.equal(disposition.status,'infrastructure');assert.match(disposition.reason,/additive|qa-execution-failed/);assert.equal((await readConfig(f.root)).skills.activeVersion,'base');
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
test('the native design writer cites verified campaign evidence alongside retained baselines',async t=>{
 const f=await fixture(t),valid=join(f.root,'valid.log'),changed=join(f.root,'changed.log');
 const sha=text=>createHash('sha256').update(text).digest('hex');
 const validText='Verifier observed an output mismatch',changedText='Old verifier output';
 const validRef='artifact:'+sha(validText),changedRef='artifact:'+sha(changedText),unknown='events:9999';
 await writeFile(valid,validText);
 await writeFile(changed,'Different output');
 f.item.evidence.groups[0].occurrences[0].artifacts=[
  {path:valid,sha256:sha(validText)},
  {path:changed,sha256:sha(changedText)},
 ];
 const retrievals=[{entries:[{collection:'memory',chunk_id:'chunk',text:'Successful bounded formulation'}]}];
 const directories=[];
 let submitted=0;
 const provider=await scriptedReview(t,f.root,body=>{
  const tools=body.messages.filter(message=>message.role==='tool');
  if(tools.length===3)return null;
  const references=tools.length===0?[unknown]:tools.length===1?[changedRef]:['events:1',validRef,'baseline-0'];
  const value={text:'Check that a simpler representation preserves the original contract. Variant '+(++submitted)+'.',
   scope:'implementation',memoryRefs:['memory:chunk'],claims:[{path:'text',kind:'hypothesis',
    proofs:references.map(reference=>({reference,startLine:1,endLine:1})),
   }]};
  return {name:'submit_review',args:{result:value}};
 });
 f.deps.session=async args=>{
  directories.push(args.directory);
  const reviewed=await reviewSession({...args,knowledge:false,
   config:{...args.config,actor:provider.profile},validate:value=>args.validate(value,retrievals),
  });
  return {...reviewed,retrievals};
 };
 const candidates=await createFailureDesign({config:f.config,item:f.item},f.deps)(f.input);
 assert.equal(candidates.length,3);
 assert.equal(provider.requests(),12);
 for(const directory of directories){
  const native=JSON.parse(await readFile(join(directory,'input.json'),'utf8'));
  const refs=new Set(native.proofSelectors.map(record=>record.reference));
  assert(refs.has('events:1')&&refs.has(validRef)&&refs.has('baseline-0'));
  assert(!refs.has(unknown)&&!refs.has(changedRef));
  const rejected=(await readFile(join(directory,'submission-rejections.jsonl'),'utf8')).trim().split('\n').map(JSON.parse);
  assert(rejected.some(row=>row.error.includes(unknown)));
  assert(rejected.some(row=>row.error.includes(changedRef)));
 }
 for(const candidate of candidates){
  assert.deepEqual(candidate.evidence.map(proof=>proof.reference),['events:1',validRef,'baseline-0']);
  assert.equal(candidate.evidence[1].quote,validText);
  assert.equal(candidate.evidence[1].sourceSha256,sha(validText));
 }
});

test('default native QA envelopes pass five proof-backed answers and review through proposal QA',async t=>{
 const f=await fixture(t);let answers=0,reviews=0;
 const provider=await scriptedReview(t,f.root,body=>{
  const tools=body.messages.filter(message=>message.role==='tool');if(!tools.length)return {name:'read',args:{path:'input.json'}};
  if(String(tools.at(-1).content).includes('Accepted. Finish'))return null;
  const first=JSON.parse(tools[0].content),paths=first.groups.flatMap(group=>[group.diagnosticsFile,group.linkedDiagnosticsFile]);
  const readPaths=body.messages.filter(message=>message.role==='assistant').flatMap(message=>message.tool_calls??[]).filter(call=>call.function.name==='read').map(call=>JSON.parse(call.function.arguments).path);
  const missing=paths.find(path=>!readPaths.includes(path));if(missing)return {name:'read',args:{path:missing}};
  const schema=body.tools.find(tool=>tool.function.name==='submit_review').function.parameters.properties.result;
  const proof={reference:'input:proposal',field:'/reason'},text='Instruction grounded in reproduced baseline failures';let value;
  if(schema.properties.answers){answers++;value={answers:Object.fromEntries(qaTopics.map(topic=>[topic,{text,refs:['events:1']}])),claims:qaTopics.map(topic=>({path:'answers.'+topic+'.text',kind:'observation',proofs:[proof]}))};}
  else{reviews++;value={verdict:'approved',reason:text,claims:[{path:'reason',kind:'observation',proofs:[proof]}]};}
  return {name:'submit_review',args:{result:value}};
 });
 const rows=await createFailureDesign({config:f.config,item:f.item,knowledgeConfig:{...f.config,actor:provider.profile,skill_writer_actor:provider.profile}},{session:f.deps.session})(f.input);
 assert.equal(rows.length,3);assert.equal(answers,3);assert.equal(reviews,3);for(const row of rows){assert.equal(row.qa.verdict,'approved');assert(row.qa.approval);}
});
test('memory transport failures exclude even an otherwise supported instruction',async t=>{
 const f=await fixture(t),session=f.deps.session;f.deps.session=async args=>{const value=await session(args),file=join(args.directory,'tainted.jsonl');await mkdir(args.directory,{recursive:true});await writeFile(file,JSON.stringify({message:{role:'toolResult',toolName:'mcp__kdb__search',isError:true,content:[{type:'text',text:'MCP request timed out after 60000ms'}]}})+'\n');return {...value,session:file};};
 await assert.rejects(createFailureDesign({config:f.config,item:f.item},f.deps)(f.input),/infrastructure-contaminated/);assert.equal(f.qaCalls.length,0);assert.deepEqual(await readdir(join(f.root,'skills/versions')),['base']);
});
test('scientific QA rejection is a completed disposition with no repeated inference',async t=>{
 const f=await fixture(t);f.deps.qaReview=async()=>({verdict:'learn',reason:'Evidence does not support a general instruction',learningRequests:[{question:'Need a different observed failure',hypothesis:'Current principle is already known',requiredEvidence:'Independent failure',learningTarget:'device'}]});
 const design=createFailureDesign({config:f.config,item:f.item},f.deps),out=await design(f.input);assert.equal(out.status,'rejected');const requests=f.calls.length;assert.deepEqual(await design(f.input),out);assert.equal(f.calls.length,requests);assert.equal((await readConfig(f.root)).skills.activeVersion,'base');
});
test('one scientific rejection lets pending approved siblings finish without infrastructure cancellation',async t=>{
 const f=await fixture(t);let entered=0,completed=0,cancelled=0,ready;const allEntered=new Promise(resolve=>{ready=resolve;});
 f.deps.qa=async input=>{if(++entered===3)ready();await allEntered;
  if(input.directory.endsWith('principle-a/qa'))return {verdict:'learn',reason:'Unsupported principle',reasonKind:'scientific'};
  await new Promise((resolve,reject)=>{const aborted=()=>{cancelled++;reject(new DOMException('Sibling cancelled','AbortError'));};if(input.signal.aborted)return aborted();input.signal.addEventListener('abort',aborted,{once:true});setTimeout(()=>{input.signal.removeEventListener('abort',aborted);resolve();},25);});
  completed++;return {verdict:'approved',approval:'reviewed',proposal:input.proposal};
 };
 const out=await createFailureDesign({config:f.config,item:f.item},f.deps)(f.input);assert.deepEqual(out.map(candidate=>candidate.id),['principle-b','principle-c']);assert.equal(completed,2);assert.equal(cancelled,0);assert.equal((await readConfig(f.root)).skills.activeVersion,'base');
});
test('a peer QA infrastructure interruption leaves independent reviewed drafts eligible',async t=>{
 const f=await fixture(t);
 f.deps.qa=async input=>{
  if(input.directory.endsWith('principle-c/qa'))return {verdict:'inconclusive',reason:'Memory transport disconnected',reasonKind:'infrastructure'};
  await new Promise(resolve=>setTimeout(resolve,25));input.signal.throwIfAborted();
  return {verdict:'approved',approval:'reviewed',proposal:input.proposal};
 };
 const rows=await createFailureDesign({config:f.config,item:f.item},f.deps)(f.input);
 assert.deepEqual(rows.map(row=>row.id),['principle-a','principle-b']);
});
test('a closed native writer timeout cohort is inconclusive and retains its five pilot baselines',async t=>{
 const f=await fixture(t),directory=join(f.root,'timeout-pilot');
 const manifest={baseline:f.input.baseline,identities:f.input.identities,families:[family()]};
 const designDirectory=join(directory,'operations','design--retained');
 const provider=await scriptedReview(t,f.root,()=>null),writers=[];
 for(const id of ['principle-a','principle-b','principle-c']){
  const directory=join(designDirectory,id);
  await mkdir(directory,{recursive:true});
  await writeFile(join(directory,'writer-spend.json'),JSON.stringify({seconds:180,elapsedSeconds:179,running:false}));
 }
 let baselines=0,qualifications=0;
 const options={directory,manifest,trialSeconds:1,seconds:120,concurrency:4};
 const deps={
  qualify:async()=>{qualifications++;return {qualified:true,referencePassed:true,negativeFailed:true};},
  trial:async({phase,repetition})=>{assert.equal(phase,'screen');baselines++;return baseline()[repetition];},
  design:createFailureDesign({config:f.config,item:f.item},{
   session:async input=>{
    writers.push(input);
    return reviewSession({...input,config:{actor:provider.profile},knowledge:false},{
     retrySettings:{maxRetries:1,baseDelayMs:4000},wrapStream:real=>timeoutReviewStream(real,()=>true),
    });
   },
   qa:()=>assert.fail('A timed-out writer submitted no proposal to review'),
  }),
 };
 const report=await runFailurePilot(options,deps);
 assert.equal(report.status,'inconclusive');
 assert.equal(report.unresolvedOperation,undefined);
 assert.equal(report.candidates,null);
 assert.deepEqual(report.screens.retained,baseline().map(row=>validatePilotTrial(row,manifest.families[0].development)));
 assert.equal(report.operations['design--retained'].status,'complete');
 assert.equal(report.operations['design--retained'].result.errorCode,'FAILURE_DESIGN_PROVIDER_TIMEOUT');
 assert.equal(report.operations['design--retained'].result.writers.length,3);
 assert(report.operations['design--retained'].result.writers.every(w=>w.kind==='elapsed'));
 for(const input of writers){
  assert.equal(input.seconds,1);
  const spend=JSON.parse(await readFile(join(input.directory,'..','writer-spend.json'),'utf8'));
  assert.equal(spend.running,false);
  assert(spend.elapsedSeconds>=179.9);
 }
 assert.equal(provider.requests(),0);
 assert.equal(baselines,5);
 assert.equal(qualifications,1);
 const rerun=await runFailurePilot(options,{
  qualify:()=>assert.fail('Reuse qualified controls'),
  trial:()=>assert.fail('Reuse all five settled baseline trials'),
  design:()=>assert.fail('Reuse the settled design outcome'),
  settle:()=>assert.fail('Closed writer cohort has no unresolved custody'),
 });
 assert.equal(rerun.status,'inconclusive');
 assert.deepEqual(rerun.screens,report.screens);
 assert.deepEqual(rerun.operations,report.operations);
 assert.deepEqual(await readdir(join(f.root,'skills/versions')),['base']);
});

test('an unknown design error keeps the pilot operation unresolved',async t=>{
 const f=await fixture(t),directory=join(f.root,'error-pilot');
 const options={directory,manifest:{baseline:f.input.baseline,identities:f.input.identities,families:[family()]},trialSeconds:1,seconds:120};
 const report=await runFailurePilot(options,{
  qualify:async()=>({qualified:true,referencePassed:true,negativeFailed:true}),
  trial:async({repetition})=>baseline()[repetition],
  design:createFailureDesign({config:f.config,item:f.item},{session:async()=>{throw Error('Writer closure unknown');}}),
 });
 assert.equal(report.status,'needs-settlement');
 assert.equal(report.reason,'Writer closure unknown');
 assert.equal(report.operations['design--retained'].status,'running');
 assert.equal(report.candidates,null);
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
 const approval=structuredClone(f.report);approval.winner.qa.approval='unreviewed';approval.candidates.find(row=>row.id===approval.winner.id).qa.approval='unreviewed';await writeFile(join(f.directory,'pilot.json'),JSON.stringify(approval));await assert.rejects(publishFailureWinner({...input,result:approval}),/reviewed version|cohort changed/);assert.equal((await readConfig(f.root)).skills.activeVersion,'base');
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

test('coherently persisted pilot with a different baseline cannot publish',async t=>{
 const f=await study(t),manifest=JSON.parse(await readFile(join(f.directory,'manifest.json'),'utf8'));manifest.baseline+='Additional untested baseline';await writeFile(join(f.directory,'manifest.json'),JSON.stringify(manifest));
 const report={...f.report,identity:'coherently-replaced-study'};await writeFile(join(f.directory,'pilot.json'),JSON.stringify(report));let published=0;
 await assert.rejects(publishFailureWinner({config:f.config,item:f.item,result:report,directory:f.directory},{publish:async()=>{published++;return true;}}),/Tested baseline guidance differs/);assert.equal(published,0);assert.equal((await readConfig(f.root)).skills.activeVersion,'base');
});

test('focused writing requires one exact failure and room for two additional passes',async t=>{
 const f=await fixture(t),design=createFailureDesign({config:f.config,item:f.item},f.deps);
 const makeInput=baselineResults=>({...f.input,mode:'focused-correctness',baselineResults});
 const other={...result(false),mechanismMatch:false};
 await assert.rejects(design(makeInput([result(true),other,other,other,other])),/writing gate/);
 await assert.rejects(design(makeInput([result(true),result(true),result(true),result(true),result(false)])),/writing gate/);
 assert.equal(f.calls.length,0);
 const rows=await design(makeInput([result(true),result(false),other,other,other]));
 assert.equal(rows.length,3);
});

test('a single focused panel design writes only the approved candidate and replays all votes for scoring',async t=>{
 const f=await fixture(t);
 const design=createFailureDesign({config:f.config,item:f.item},f.deps);
 const input={...f.input,mode:'focused-correctness',qaPolicy:'panel',reviewCandidate:'principle-b'};
 const candidates=await design(input);
 assert.deepEqual(candidates.map(candidate=>candidate.id),['principle-b']);
 assert.equal(f.calls.length,1);
 const record=JSON.parse(await readFile(join(input.directory,'principle-b/qa/qa.json'),'utf8'));
 const candidate=candidates[0];
 assert.equal(record.rounds.at(-1).panel.decision.approvals,5);
 verifyPanelApproval(record,candidate.proposal);
 const changed=structuredClone(record);
 for(const slot of changed.rounds.at(-1).panel.slots.slice(0,3))slot.status='missing';
 assert.throws(()=>verifyPanelApproval(changed,candidate.proposal),/votes do not approve/);
 assert.throws(()=>verifyPanelApproval(record,{...candidate.proposal,reason:'different draft'}),/different proposal/);
 assert.deepEqual(await design(input),candidates);
 assert.equal(f.calls.length,1);
 await assert.rejects(design({...input,reviewCandidate:'principle-a'}),/evidence changed/);
 await writeFile(join(input.directory,'principle-b/qa/qa.json'),JSON.stringify(changed));
 await assert.rejects(design(input),/votes do not approve/);
 assert.equal((await readConfig(f.root)).skills.activeVersion,'base');
});

test('focused panel recovery retries missing reviewers with the same audit sources',async t=>{
 const f=await fixture(t),source={reference:'verification:final',text:'The scored output passed.'};
 let calls=0;
 f.deps.qaReview=async()=>{
  calls++;
  if(calls<=2)return {verdict:'approved',reason:'General applicable guidance'};
  if(calls===3)return {verdict:'revise',reason:'Check final status',issues:[{kind:'draft',path:'changes.0.text',problem:'The final output failed.'}]};
  if(calls<=5)throw Error('Provider timeout');
  return {verdict:'approved',reason:'Supported by the frozen source'};
 };
 f.deps.qaPanelAudit=async request=>{
  assert.deepEqual(request.panelAuditEvidence,[source]);
  return {audits:request.objections.map(claim=>({claimId:claim.id,disposition:'disproved',reason:source.text})),
   claims:[{path:'audits.0.reason',proofs:[{reference:source.reference,quote:source.text,recordSha256:createHash('sha256').update(source.text).digest('hex')}]}]};
 };
 const design=createFailureDesign({config:f.config,item:f.item},f.deps);
 const input={...f.input,mode:'focused-correctness',qaPolicy:'panel',reviewCandidate:'principle-a',panelAuditEvidence:[source]};
 await assert.rejects(design(input),error=>error.code==='FAILURE_DESIGN_INFRASTRUCTURE');
 assert.equal(calls,5);
 const [approved]=await design({...input,retryMissingPanelVotes:true,qaPanelConcurrency:1});
 assert.equal(approved.qa.verdict,'approved');
 assert.equal(calls,7);
 await assert.rejects(design({...input,panelAuditEvidence:[]}),/evidence changed/);
});

test('a fresh panel writer can cite the primary source that established a verified defect',async t=>{
 const f=await fixture(t),record={reference:'final-score:1',text:'The terminal verifier failed.'};
 const irrelevant={reference:'unused:2',text:'Different experiment.'};
 const input={...f.input,mode:'focused-correctness',qaPolicy:'panel',reviewCandidate:'principle-a',panelAuditEvidence:[record,irrelevant]};
 const design=createFailureDesign({config:f.config,item:f.item},{
  ...f.deps,
  qaReview:async request=>request.directory.includes('second-writer')?{verdict:'approved',reason:'Corrected.'}:
   {verdict:'revise',reason:'Specify the boundary.',issues:[{kind:'draft',path:'changes.0.text',problem:'The final verifier is misidentified.'}]},
  qaPanelAudit:async request=>({audits:request.objections.map(claim=>({claimId:claim.id,disposition:'verified',reason:record.text})),
   claims:[{path:'audits.0.reason',proofs:[{reference:record.reference,quote:record.text,recordSha256:createHash('sha256').update(record.text).digest('hex'),startLine:1,endLine:1}]}]}),
  session:async args=>{
   const written=await f.deps.session(args);
   if(args.prompt.revisionFeedback){
    assert.deepEqual(args.prompt.revisionSources,[record]);
    assert(!args.proofRecords.some(row=>row.reference===irrelevant.reference));
    const value={...written.value,text:'Check the terminal verification result before attributing a repair.'};
    value.claims=[{path:'text',kind:'hypothesis',proofs:[{reference:record.reference,startLine:1,endLine:1}]}];
    validateClaimedResult(value,args.proofRecords);
    return {...written,value};
   }
   return written;
  },
 });
 const first=await design(input);
 assert.equal(first.status,'rejected');
 const [revised]=await design({...input,directory:join(f.root,'second-writer'),revisionFeedback:first.revisionFeedback});
 assert.equal(revised.evidence[0].reference,record.reference);
 assert.equal(f.calls.length,2);
});

test('focused review returns concrete feedback for one fresh writing attempt',async t=>{
 const f=await fixture(t);
 let reject=true;
 f.deps.qaReview=async()=>reject
  ?{verdict:'revise',reason:'Specify where the type of the branch result must remain stable',issues:[{issue:'Type stability boundary'}]}
  :{verdict:'approved',reason:'General instruction with a stated boundary'};
 const design=createFailureDesign({config:f.config,item:f.item},f.deps),input={...f.input,mode:'focused-correctness'};
 const first=await design(input);
 assert.equal(first.status,'rejected');
 assert.match(first.revisionFeedback,/branch result/);
 assert.deepEqual(await design(input),first);
 reject=false;
 const second=await design({...input,directory:join(f.root,'design-revision'),revisionFeedback:first.revisionFeedback});
 assert.equal(second.length,3);
 assert(f.calls.slice(3).every(call=>call.prompt.revisionFeedback===first.revisionFeedback));
});

async function focusedStudy(t,{redesign=false,partial=false}={}){
 const f=await fixture(t),directory=join(f.root,'pilot'),chosen=family();
 for(const [index,task] of [chosen.development,...chosen.transfer].entries()){
  task.files={'kernel.py':'retained failing source for '+task.id};
  task.checker=join(f.root,task.id+'-checker.py');
  await writeFile(task.checker,'# Frozen private oracle for '+task.id+'\n');
  task.controls.calibration={files:{'kernel.py':'private calibration for '+task.id}};
  if(index){task.changeType=index===1?'layout':'operation';task.relation='Independent adaptation for '+task.id;}
 }
 if(partial){
  const session=f.deps.session;
  f.deps.session=async input=>{
   if(input.directory.includes('principle-a/'))return session(input);
   throw Object.assign(Error('Writer exhausted'),{reviewSession:{kind:'elapsed'}});
  };
 }
 if(redesign){
  const approve=f.deps.qaReview;
  f.deps.qaReview=async input=>input.directory.includes('/operations/design--')?{
   verdict:'revise',reason:'Specify the applicability boundary',issues:[{issue:'Scope is incomplete'}],
  }:approve(input);
 }
 const manifest={mode:'focused-correctness',baseline:f.input.baseline,identities:f.input.identities,families:[chosen]};
 const report=await runFailurePilot({directory,manifest,trialSeconds:1,seconds:120,concurrency:4},{
  qualify:async()=>({qualified:true,referencePassed:true,negativeFailed:true,calibrationPassed:true}),
  design:createFailureDesign({config:f.config,item:f.item},f.deps),
  trial:async input=>{
   if(input.arm==='baseline')return input.repetition<2?result(true):input.repetition===2?result(false):{...result(false),mechanismMatch:false};
   return result(input.arm==='principle-a');
  },
 });
 assert.equal(report.promising,true,report.reason);
 return {...f,directory,manifest,chosen,report};
}

test('focused pilot publishes the exact reviewed revision after both admitted transfers',async t=>{
 const f=await focusedStudy(t,{redesign:true}),input={config:f.config,item:f.item,result:f.report,directory:f.directory};
 assert(f.report.operations['redesign--retained']);
 const receipt=await publishFailureWinner(input);
 assert.equal(receipt.published,true);
 assert.equal((await readConfig(f.root)).skills.activeVersion,f.report.winner.version);
});

test('focused publication refuses changed transfer fixture and never invokes the publisher',async t=>{
 const f=await focusedStudy(t),input={config:f.config,item:f.item,result:f.report,directory:f.directory};
 const manifest=structuredClone(f.manifest);
 manifest.families[0].transfer[0].files['kernel.py']='different untested transfer';
 await writeFile(join(f.directory,'manifest.json'),JSON.stringify(manifest));
 let calls=0;
 await assert.rejects(publishFailureWinner(input,{publish:()=>++calls}),/fixture|manifest/i);
 await writeFile(join(f.directory,'manifest.json'),JSON.stringify(f.manifest));
 await writeFile(f.chosen.transfer[0].checker,'# Changed oracle\n');
 await assert.rejects(publishFailureWinner(input,{publish:()=>++calls}),/Frozen checker changed/i);
 assert.equal(calls,0);
});

test('focused publication refuses independent changes to the QA receipt or completed operation result',async t=>{
 const f=await focusedStudy(t),input={config:f.config,item:f.item,result:f.report,directory:f.directory};
 const qa=join(f.report.operations['design--retained'].directory,f.report.winner.id,'qa','qa.json');
 const original=await readFile(qa,'utf8');
 const changed={...JSON.parse(original),approval:'changed'};
 await writeFile(qa,JSON.stringify(changed));
 let calls=0;
 await assert.rejects(publishFailureWinner(input,{publish:()=>++calls}),/QA approval/i);
 await writeFile(qa,original);
 const writer=join(f.report.operations['design--retained'].directory,f.report.winner.id,'writer-result.json');
 const writerSnapshot=await readFile(writer,'utf8'),replacedMemory=JSON.parse(writerSnapshot);
 replacedMemory.retrievals[0].entries[0].text='Different memory from what QA reviewed';
 await writeFile(writer,JSON.stringify(replacedMemory));
 await assert.rejects(publishFailureWinner(input,{publish:()=>++calls}),/QA approval/i);
 await writeFile(writer,writerSnapshot);
 const tampered=structuredClone(f.report);
 tampered.operations['confirmation--retained--stride-transfer--principle-a--0'].result=result(false);
 await writeFile(join(f.directory,'pilot.json'),JSON.stringify(tampered));
 await assert.rejects(publishFailureWinner({...input,result:tampered},{publish:()=>++calls}),/operation result/i);
 assert.equal(calls,0);
});

test('focused publication requires a separate exact baseline recurrence in each transfer',async t=>{
 const f=await focusedStudy(t),report=structuredClone(f.report);
 report.confirmation['stride-transfer'].baseline[2].mechanismMatch=false;
 report.operations['transfer-screen--retained--stride-transfer--baseline--2'].result.mechanismMatch=false;
 await writeFile(join(f.directory,'pilot.json'),JSON.stringify(report));
 let calls=0;
 await assert.rejects(publishFailureWinner({config:f.config,item:f.item,result:report,directory:f.directory},{publish:()=>++calls}),/Transfer baseline/i);
 assert.equal(calls,0);
});

test('an independently approved singleton reaches publication only after development and both transfers',async t=>{
 const f=await focusedStudy(t,{partial:true,redesign:true});
 assert.deepEqual(f.report.operations['design--retained'].result.revisionCandidates,['principle-a']);
 assert.equal(f.calls.filter(call=>call.directory.includes('/redesign--')).length,1);
 assert.deepEqual(f.report.candidates.map(candidate=>candidate.id),['principle-a']);
 assert.equal(Object.keys(f.report.confirmation).length,2);
 const saved=await publishFailureWinner({config:f.config,item:f.item,result:f.report,directory:f.directory});
 assert.equal(saved.published,true);
});

test('resume reuses a saved draft while both closed exhausted writers stay idle',async t=>{
 const f=await fixture(t),options={directory:join(f.root,'pilot'),manifest:{baseline:f.input.baseline,identities:f.input.identities,families:[family()]},trialSeconds:1,seconds:120};
 const session=f.deps.session;let checks=0,writes=0;
 const design=createFailureDesign({config:f.config,item:f.item},{...f.deps,session:async input=>{
  writes++;
  if(input.directory.includes('principle-c/'))return session(input);
  await mkdir(input.directory,{recursive:true});
  await writeFile(join(input.directory,'termination.json'),JSON.stringify({kind:'elapsed'}));
  throw Object.assign(Error('Writer exhausted'),{reviewSession:{kind:'elapsed'}});
 },qa:async()=>{throw Error('Interrupted after writer submitted');}});
 const base={qualify:async()=>({qualified:true,referencePassed:true,negativeFailed:true}),trial:async input=>{checks++;return input.phase==='screen'?baseline()[input.repetition]:result(input.arm!=='baseline');}};
 const first=await runFailurePilot(options,{...base,design});
 assert.equal(first.status,'needs-settlement');assert.equal(checks,5);
 const op=first.operations['design--retained'];
 for(const id of ['principle-a','principle-b']){
  await writeFile(join(op.directory,id,'writer-spend.json'),JSON.stringify({seconds:180,elapsedSeconds:180.01,running:false}));
 }
 const resume=createFailureDesign({config:f.config,item:f.item},{...f.deps,session:()=>assert.fail('Writer dispatched during recovery')});
 const qa=join(op.directory,'principle-c','qa');await mkdir(qa,{recursive:true});
 await writeFile(join(qa,'qa.json'),JSON.stringify({inFlightOperations:{request:{stage:'qaAnswer'}}}));
 assert.equal(await resume.canResume(op),false);
 await rm(join(qa,'qa.json'));
 const second=await runFailurePilot(options,{...base,design:resume,qualify:()=>assert.fail('Repeated controls'),trial:async input=>{assert.notEqual(input.phase,'screen');return result(input.arm!=='baseline');}});
 assert.equal(writes,3);assert.equal(second.promising,true,second.reason);assert.deepEqual(second.candidates.map(row=>row.id),['principle-c']);
});

test('closed unsubmitted writers leave independently approved guidance intact',async t=>{
 const f=await fixture(t),session=f.deps.session;
 f.deps.session=input=>{
  if(input.directory.includes('principle-c/'))return session(input);
  throw Object.assign(Error('No valid submission'),{reviewSession:{kind:input.directory.includes('principle-a/')?'no-submit':'rejected-submissions'}});
 };
 const rows=await createFailureDesign({config:f.config,item:f.item},f.deps)(f.input);
 assert.deepEqual(rows.map(row=>row.id),['principle-c']);
});

test('panel custody checks only the selected lane and requires a closed or frozen writer',async t=>{
 const f=await fixture(t),design=createFailureDesign({config:f.config,item:f.item},f.deps);
 const directory=f.input.directory,lane=join(directory,'principle-b');await mkdir(lane,{recursive:true});
 const input={qaPolicy:'panel',reviewCandidate:'principle-b'};
 await writeFile(join(directory,'design-input.json'),JSON.stringify(input));
 await writeFile(join(lane,'writer-result.json'),JSON.stringify({value:{}}));
 assert.equal(await design.canResume({directory}),false);
 await writeFile(join(lane,'writer-spend.json'),JSON.stringify({running:false}));
 assert.equal(await design.canResume({directory}),true);
 await rm(join(lane,'writer-spend.json'));
 await writeFile(join(directory,'design-input.json'),JSON.stringify({...input,frozenQaSha256:'frozen'}));
 assert.equal(await design.canResume({directory}),true);
});

test('QA repairs a changed review flag while retaining its substantive revision',async t=>{
 const f=await fixture(t);let repairs=0;
 f.deps.qa=async input=>{
  const proposal=structuredClone(input.proposal),change=proposal.changes[0];
  assert.throws(()=>input.validateProposal({...proposal,changes:[]}),/exactly one change.*received 0/);
  change.text+=' Include one independent check.';change.negative=true;change.id='replacement-id';
  try{input.validateProposal(proposal);assert.fail('Changed flag accepted');}catch(error){
   const fields=[...error.message.matchAll(/changes\[0\]\.(\w+) must be ("[^"]*"|false|true); received ("[^"]*"|false|true)/g)];
   assert.deepEqual(fields.map(match=>match[1]).sort(),['id','negative']);
   for(const [,key,expected,actual] of fields){
    assert.equal(JSON.parse(actual),change[key]);change[key]=JSON.parse(expected);
   }
   repairs++;
  }
  return {verdict:'approved',approval:'reviewed',proposal:input.validateProposal(proposal)};
 };
 const rows=await createFailureDesign({config:f.config,item:f.item},f.deps)(f.input);
 assert.equal(repairs,3);assert.equal(rows.length,3);
 for(const row of rows){assert.equal(row.proposal.changes[0].negative,false);assert(row.instruction.endsWith('Include one independent check.'));}
});
