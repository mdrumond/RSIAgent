import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {runSkillProjectAB,validateSkillProject,validateSkillTestDesign,groupedSkillVerdict} from '../../cli/skill-project-ab.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex');
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'skill-project-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const checker=join(root,'trusted.py');await writeFile(checker,'trusted fixture checker\n');
 const project={id:'retained-project',objective:'correctness',target:'a3',prompt:'Implement the fixed operation.',checker,files:{'kernel.py':'starter','CONTRACT.md':'contract'},cases:[0,1,2,3,4].map(n=>({id:'case-'+n,parameters:{n}})),controls:{reference:{files:{'kernel.py':'reference'}},negatives:[{files:{'kernel.py':'broken'}}]}};
 const evidence=[{reference:'events:41',text:'Measured failure with source identity.'}],design={decision:'propose',reason:'Measure the same project under two guidance arms.',target:'a3',objective:'correctness',evidenceRefs:['events:41'],caseIds:project.cases.map(c=>c.id),actorUnit:'one-project-five-cases',repetitions:2,arms:['baseline','candidate'],success:'Full candidate success with baseline failures.',disconfirmingOutcome:'Baseline ceiling or regression.',inconclusive:'No completed trusted verdict.',scoring:{candidateFullPasses:2,baselineFullPassesAtMost:1,noCorrectnessRegression:true}};
 const calls=[],budgets=[],outcomes={baseline:false,candidate:true},options={directory:join(root,'experiment'),config:{actor:{id:'actor'},skill_writer_actor:{id:'writer'}},project,evidence,guidance:{baseline:'base',candidate:'base plus scoped rule'},binding:{target:'a3',environment:{mode:'cpl',target:'registered-a3',runtime:'registered',health:{product:'a3'}}},resource:{id:'caller-bound'},seconds:120,actorSeconds:15};
 const deps={
  role:async args=>{
   calls.push(args);budgets.push(args.seconds);const cwd=join(args.directory,'work');await mkdir(cwd,{recursive:true});for(const[n,text]of Object.entries(args.files))await writeFile(join(cwd,n),text);
   if(args.submission)return {value:args.submission==='proposal.json'?structuredClone(design):{verdict:'approved',reason:'Fair and relevant.',disconfirmation:'Both arms may pass.',leakage:'Controls are not supplied.'},cwd};
   const arm=args.directory.endsWith('baseline')?'baseline':'candidate';await writeFile(join(cwd,'kernel.py'),arm);return {cwd,elapsedSeconds:.1,stats:{toolCalls:2}};
  },
  target:{qualifyBatch:async args=>{
   budgets.push(args.seconds);return args.entries.map(entry=>{
    let correct=entry.key==='reference'?true:entry.key.startsWith('negative')?false:outcomes[entry.key.split('/')[1]];
    const value={correct,kernelSha256:sha(entry.files['kernel.py']),evidence:correct?{device:'npu:0',checkedCases:project.cases.map(c=>c.id)}:{reason:'Numerical mismatch',type:'AssertionError'}};
    return {key:entry.key,exitCode:correct?0:1,sourceFilesMatch:true,checkerSha256:sha(entry.checker),output:JSON.stringify(value)};
   });
  }},settled:async()=>true
 };
 return {root,options,deps,calls,budgets,outcomes,design};
}
test('one project produces four actors with identical task inputs; scientific success remains shadow-only',async t=>{
 const f=await fixture(t),result=await runSkillProjectAB(f.options,f.deps);
 assert.equal(result.status,'complete');assert.equal(result.verdict.promote,true);assert.equal(result.shadow,true);
 const actors=f.calls.filter(c=>!c.submission);assert.equal(actors.length,4);assert.deepEqual(actors.map(a=>a.directory.split('/').at(-1)),['0-baseline','0-candidate','1-candidate','1-baseline']);
 for(const a of actors){assert.equal(a.files['kernel.py'],'starter');assert.equal(a.files['CONTRACT.md'],'contract');assert.deepEqual(JSON.parse(a.files['AB_CASES.json']),f.options.project.cases);assert.equal(a.editable[0],'kernel.py');assert.equal(a.profile.id,'actor');assert.equal(a.files['reference.py'],undefined);}
 assert.equal(new Set(actors.map(a=>a.files['AB_GUIDANCE.md'])).size,2);assert.equal(result.results.length,4);assert(f.budgets.every(s=>s>0&&s<120));
 await assert.rejects(runSkillProjectAB(f.options,f.deps),/EEXIST/);assert.equal(f.calls.length,6);
});
test('baseline ceiling never recommends publication',async t=>{const f=await fixture(t);f.outcomes.baseline=true;const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.verdict.promote,false);assert.equal(r.reason,'baseline-ceiling');});
test('partial candidate correctness and regressions cannot pass',()=>{
 const results=[{rep:0,arm:'baseline',correct:true},{rep:1,arm:'baseline',correct:false},{rep:0,arm:'candidate',correct:false},{rep:1,arm:'candidate',correct:true}];assert.equal(groupedSkillVerdict(results).promote,false);
 results[0].correct=null;assert.equal(groupedSkillVerdict(results).reason,'inconclusive');assert.equal(groupedSkillVerdict([...results,results[0]]).promote,false);
});
test('duplicate cases or a renamed duplicate parameter set are rejected before model work',async t=>{
 const f=await fixture(t);f.options.project.cases[1].parameters={n:0};await assert.rejects(runSkillProjectAB(f.options,f.deps),/distinct case/);assert.equal(f.calls.length,0);
});
test('parameter key order does not create an independent case',async t=>{const f=await fixture(t);f.options.project.cases[0].parameters={x:1,y:2};f.options.project.cases[1].parameters={y:2,x:1};assert.throws(()=>validateSkillProject(f.options.project),/distinct case/);});
test('optimization, harness and mismatched target bindings do not run a correctness comparison',async t=>{
 const f=await fixture(t);for(const objective of ['optimization','harness'])assert.throws(()=>validateSkillProject({...f.options.project,objective}),/correctness project/);
 f.options.binding.environment.health.product='a5';await assert.rejects(runSkillProjectAB(f.options,f.deps),/binding/);assert.equal(f.calls.length,0);
});
test('unknown references, changed arms, missing cases, and win-every-pair scoring fail design admission',async t=>{
 const f=await fixture(t),ctx={project:f.options.project,evidence:f.options.evidence};
 for(const patch of [{evidenceRefs:['events:4']},{arms:['candidate','baseline']},{caseIds:['case-0']},{scoring:{candidateFullPasses:2,baselineFullPassesAtMost:0,noCorrectnessRegression:true}}])assert.throws(()=>validateSkillTestDesign({...f.design,...patch},ctx));
});
test('learn/no-test decisions and independent QA rejection do not execute controls or actors',async t=>{
 for(const which of ['learn','no-test','qa']){
  const f=await fixture(t),original=f.deps.role;let batches=0;
  f.deps.role=async args=>{const r=await original(args);if(args.submission==='proposal.json'&&which!=='qa')r.value.decision=which;if(args.submission==='qa.json')r.value.verdict='reject';return r;};f.deps.target.qualifyBatch=async()=>{batches++;return [];};
  const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.status,'needs-learning');assert.equal(batches,0);assert.equal(f.calls.filter(c=>!c.submission).length,0);
 }
});
test('a now-working historical negative is retained and stops before paid actors',async t=>{
 const f=await fixture(t),batch=f.deps.target.qualifyBatch;f.deps.target.qualifyBatch=async args=>(await batch(args)).map(r=>r.key.startsWith('negative')?{...r,exitCode:0,output:JSON.stringify({correct:true,kernelSha256:sha('broken'),evidence:{device:'npu:0',checkedCases:f.options.project.cases.map(c=>c.id)}})}:r);
 const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.reason,'historical-negative-now-passes');assert.equal(r.controls.length,2);assert.equal(f.calls.length,2);assert.equal(r.verdict.promote,false);
});
test('missing coverage, wrong source hash and infrastructure refusals remain unscored',async t=>{
 for(const defect of ['coverage','source','infrastructure']){
  const f=await fixture(t),batch=f.deps.target.qualifyBatch;f.deps.target.qualifyBatch=async args=>(await batch(args)).map(r=>{
   if(!r.key.includes('/'))return r;const value=JSON.parse(r.output);if(defect==='coverage')value.evidence.checkedCases=['case-0'];if(defect==='source')value.kernelSha256='wrong';if(defect==='infrastructure'){r.exitCode=125;value.correct=null;value.bucket='harness';}return {...r,output:JSON.stringify(value)};
  });const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.verdict.promote,false);assert(r.results.some(row=>row.correct===null));
 }
});
test('missing trusted receipt cannot qualify controls',async t=>{const f=await fixture(t);f.deps.target.qualifyBatch=async()=>[{key:'reference',exitCode:0,sourceFilesMatch:true,checkerSha256:sha('trusted fixture checker\n'),output:'no receipt'}];const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.reason,'controls-inconclusive');assert.equal(f.calls.length,2);});
test('provider length stop is retained as infrastructure rather than kernel failure',async t=>{
 const f=await fixture(t);f.deps.role=async()=>({error:'provider-length',stopReason:'length',expired:false});const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.status,'inconclusive');assert.equal(r.error.roleResult.stopReason,'length');assert.equal(r.results.length,0);
});
test('unknown commands remain fenced with their handle and cannot be replayed',async t=>{
 const f=await fixture(t);f.deps.target.qualifyBatch=async()=>{throw Object.assign(Error('Execution closure unknown'),{code:'RSI_EXECUTION_UNRESOLVED',handle:'remote:test:job:retained'});};f.deps.settled=async()=>false;
 const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.status,'unresolved');assert.equal(r.error.handle,'remote:test:job:retained');assert.equal(r.verdict.promote,false);await assert.rejects(runSkillProjectAB(f.options,f.deps),/EEXIST/);
});
test('an unresolved final verification cannot recommend a skill',async t=>{const f=await fixture(t);let n=0;f.deps.settled=async()=>++n===1;const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.status,'unresolved');assert.equal(r.verdict.promote,false);});
test('actor changes to frozen task files are detected before verification',async t=>{
 const f=await fixture(t),role=f.deps.role;f.deps.role=async args=>{const r=await role(args);if(!args.submission)await writeFile(join(r.cwd,'AB_CASES.json'),'[]');return r;};const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.status,'inconclusive');assert.match(r.reason,/Protected actor task file changed/);assert.equal(r.verdict.promote,false);
});

test('settlement observation errors preserve unresolved state and exact custody handle',async t=>{
 const f=await fixture(t);f.deps.target.qualifyBatch=async()=>{throw Object.assign(Error('Execution closure unknown'),{code:'RSI_EXECUTION_UNRESOLVED',handle:'remote:test:job:retained'});};f.deps.settled=async()=>{throw Object.assign(Error('Observation unavailable'),{handle:'remote:test:job:retained'});};
 const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.status,'unresolved');assert.equal(r.custodyErrors[0].handle,'remote:test:job:retained');assert.equal(r.verdict.promote,false);const stored=JSON.parse(await readFile(join(f.options.directory,'result.json'),'utf8'));assert.equal(stored.status,'unresolved');assert.equal(stored.error.handle,'remote:test:job:retained');await assert.rejects(runSkillProjectAB(f.options,f.deps),/EEXIST/);
});

test('native Pi uses stock file tools, preserves protected files and submits JSON from disk',async t=>{
 const {createServer}=await import('node:http'),{once}=await import('node:events'),{nativeProjectRole}=await import('../../cli/skill-project-native.mjs');
 const f=await fixture(t),requests=[],actions=[
  {name:'read',arguments:{path:'missing.txt'}},
  {name:'write',arguments:{path:'CONTRACT.md',content:'tampered'}},
  {name:'write',arguments:{path:'proposal.json',content:JSON.stringify({decision:'learn',reason:'Needs controls.',evidenceRefs:['project.files.kernel.py']})}},
  {name:'write',arguments:{path:'proposal.json',content:JSON.stringify({decision:'learn',reason:'Needs controls.',evidenceRefs:['events:41']})}},
  {name:'edit',arguments:{path:'proposal.json',oldText:'events:41',newText:'invented'}}
 ];
 const server=createServer(async(req,res)=>{
  let raw='';for await(const chunk of req)raw+=chunk;requests.push(JSON.parse(raw));const action=actions[requests.length-1],delta=action?{role:'assistant',tool_calls:[{index:0,id:'step-'+requests.length,type:'function',function:{name:action.name,arguments:JSON.stringify(action.arguments)}}]}:{role:'assistant',content:'Submitted.'};
  res.writeHead(200,{'Content-Type':'text/event-stream'});
  for(const[value,finish]of [[delta,null],[{},action?'tool_calls':'stop']])res.write('data: '+JSON.stringify({id:'fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:value,finish_reason:finish}]})+'\n\n');res.end('data: [DONE]\n\n');
 });
 server.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>{server.closeAllConnections();server.close();});
 const key='RSI_PROJECT_TEST_KEY',old=process.env[key];process.env[key]='fixture';t.after(()=>{if(old===undefined)delete process.env[key];else process.env[key]=old;});
 const directory=join(f.root,'native'),profile={provider:'skill-project-test',model:'scripted',credential_env:key,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:2048,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
 const result=await nativeProjectRole({directory,files:{'input.json':'{}','CONTRACT.md':'protected'},prompt:'Use the provided file tools.',submission:'proposal.json',validate:value=>{if(value.evidenceRefs.some(ref=>ref!=='events:41'))throw Error('Unknown evidenceRefs: '+JSON.stringify(value.evidenceRefs));},profile,seconds:20});
 assert.equal(result.error,null);assert.deepEqual(result.value,{decision:'learn',reason:'Needs controls.',evidenceRefs:['events:41']});assert.equal(await readFile(join(result.cwd,'CONTRACT.md'),'utf8'),'protected');
 const policy=(await readFile(join(directory,'policy.jsonl'),'utf8')).trim().split('\n').map(JSON.parse);assert.deepEqual(policy.map(p=>[p.tool,p.path]),[['write','CONTRACT.md']]);assert(requests.some(r=>r.messages.some(m=>m.role==='tool'&&JSON.stringify(m.content).includes('ENOENT'))));
 const rejections=(await readFile(join(directory,'submission-rejections.jsonl'),'utf8')).trim().split('\n').map(JSON.parse);assert.equal(rejections.length,2);assert.deepEqual(rejections.map(r=>r.tool),['write','edit']);assert.equal(rejections[0].input.content,JSON.stringify({decision:'learn',reason:'Needs controls.',evidenceRefs:['project.files.kernel.py']}));assert.equal(result.stats.userMessages,1);assert(requests.some(r=>r.messages.some(m=>m.role==='tool'&&JSON.stringify(m.content).includes('Unknown evidenceRefs'))));assert.equal(requests.length,6);assert(requests.every(r=>r.tools.every(t=>['read','write','edit'].includes(t.function.name))));
});

test('large elapsed allocations do not overflow native timer limits into immediate expiry',async t=>{
 const f=await fixture(t);f.options.seconds=3_000_000;const role=f.deps.role;f.deps.role=async args=>{assert.equal(args.signal.aborted,false);return role(args);};const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.status,'complete');assert.equal(r.verdict.promote,true);
});

test('explicit compatibility basis propagates to frozen inputs, both model roles and result',async t=>{
 const f=await fixture(t);f.options.project.successBasis='compile-and-correct';f.design.successBasis='compile-and-correct';const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.successBasis,'compile-and-correct');assert.equal(r.verdict.successBasis,'compile-and-correct');assert.equal(r.reason,'compatibility-success-improved');for(const role of f.calls.filter(c=>c.submission))assert.equal(JSON.parse(role.files['input.json']).project.successBasis,'compile-and-correct');assert.equal(JSON.parse(await readFile(join(f.options.directory,'frozen.json'),'utf8')).project.successBasis,'compile-and-correct');
});
test('resource allowance outcomes are not numerical failures; explicit compatibility can score them',async t=>{
 for(const successBasis of [undefined,'compile-and-correct']){
  const f=await fixture(t);if(successBasis){f.options.project.successBasis=successBasis;f.design.successBasis=successBasis;}const batch=f.deps.target.qualifyBatch;f.deps.target.qualifyBatch=async args=>(await batch(args)).map(row=>{if(!row.key.endsWith('/baseline'))return row;const value=JSON.parse(row.output);value.evidence={kind:'case-resource-allowance',reason:'The declared case allowance expired after confirmed child closure'};return {...row,output:JSON.stringify(value)};});
  const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.verdict.promote,Boolean(successBasis));assert.equal(r.reason,successBasis?'compatibility-success-improved':'inconclusive');assert.equal(r.results.find(row=>row.arm==='baseline').correct,successBasis?false:null);
 }
});
test('compatibility basis cannot score infrastructure or unknown outcomes as source failures',async t=>{
 for(const kind of ['infrastructure','unknown-execution']){
  const f=await fixture(t);f.options.project.successBasis='compile-and-correct';f.design.successBasis='compile-and-correct';const batch=f.deps.target.qualifyBatch;f.deps.target.qualifyBatch=async args=>(await batch(args)).map(row=>{if(!row.key.endsWith('/baseline'))return row;const value=JSON.parse(row.output);value.evidence={kind,reason:'Execution status not established'};return {...row,output:JSON.stringify(value)};});
  const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.verdict.promote,false);assert.equal(r.reason,'inconclusive');assert(r.results.some(row=>row.correct===null));
 }
});
test('success basis is a frozen admission choice, not a model-redefinable metric',async t=>{
 const f=await fixture(t);assert.throws(()=>validateSkillProject({...f.options.project,successBasis:'latency'}));assert.throws(()=>validateSkillTestDesign({...f.design,successBasis:'compile-and-correct'},{project:f.options.project,evidence:f.options.evidence}));
});

test('explicit compatibility cannot turn unknown command custody into a scored failure',async t=>{
 const f=await fixture(t);f.options.project.successBasis='compile-and-correct';f.design.successBasis='compile-and-correct';f.deps.target.qualifyBatch=async()=>{throw Object.assign(Error('Execution closure unknown'),{code:'RSI_EXECUTION_UNRESOLVED',handle:'remote:test:job:compatibility'});};f.deps.settled=async()=>false;const r=await runSkillProjectAB(f.options,f.deps);assert.equal(r.status,'unresolved');assert.equal(r.verdict.promote,false);assert.equal(r.verdict.successBasis,'compile-and-correct');assert.equal(r.error.handle,'remote:test:job:compatibility');
});
