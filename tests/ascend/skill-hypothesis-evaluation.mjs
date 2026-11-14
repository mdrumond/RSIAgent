import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {prepareHypothesisEvaluation,evaluateHypothesis,hypothesisEvaluationSettled} from '../../cli/skill-hypothesis-evaluation.mjs';
import {runMicroeval} from '../../cli/skill-microeval.mjs';
import {evaluationSettings} from '../../controller/skill-objectives.mjs';
const digest=text=>createHash('sha256').update(text).digest('hex');
async function fixture(t,{target='cpu',objective='correctness'}={}){
 const root=await mkdtemp(join(tmpdir(),'hypothesis-eval-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const skills={};for(const arm of ['active','candidate']){const path=skills[arm]=join(root,arm);await mkdir(path);await writeFile(join(path,'skill-map.json'),JSON.stringify({implementation:'skill.md'}));await writeFile(join(path,'skill.md'),arm);}
 const checker=join(root,'checker.py');await writeFile(checker,"import sys,pathlib\np=pathlib.Path(sys.argv[1])/'kernel.py'\nraise SystemExit(0 if 'answer=42' in p.read_text() else 1)\n");
 const binding={target,...(target==='cpu'?{}:{environment:{mode:'local',target:'registered-'+target,runtime:'fixture-runtime',instructionFile:'fixture',health:{product:target}},python:'python3'})};
 const tasks=['one','two','three','four','five'].map(id=>({id,kind:target==='cpu'?'hypothesis-case':'target-kernel',target,bucket:objective,prompt:'Repair answer',checker,...(objective==='optimization'?{pairedChecker:checker}:{}),files:{'kernel.py':'answer=0\n','nested/config.json':'{}'},protectedFiles:{'nested/config.json':'json'}}));
 return {directory:join(root,'evaluation'),skills,config:{actor:{provider:'fixture',model:'fixture'},python:process.env.RSI_TEST_PYTHON??'python3'},objective,target,tasks,binding,remainingSeconds:900};
}
test('selection requires a compatible five-case suite and never substitutes another target',async t=>{
 const f=await fixture(t,{target:'a5'});
 assert.equal(prepareHypothesisEvaluation({...f,tasks:[]}).status,'needs-learning');
 assert.equal(prepareHypothesisEvaluation({...f,binding:{...f.binding,target:'a3'}}).status,'unsupported');
 assert.equal(prepareHypothesisEvaluation({...f,tasks:f.tasks.map(x=>({...x,target:'a3'}))}).status,'unsupported');
 assert.equal(prepareHypothesisEvaluation(f).status,'ready');
 assert.equal(prepareHypothesisEvaluation({...f,objective:'harness'}).status,'unsupported');
 await assert.rejects(runMicroeval({...f,profile:f.config.actor,binding:{target:'a3',environment:{health:{product:'a3'}}}},{sample:()=>{throw Error('Mismatched dispatch');}}),/target differs/);
 const legacy=f.tasks.map(x=>({...x,kind:'a3-kernel',target:undefined}));
 assert.equal(evaluationSettings({correctness:{binding:{target:'legacy-registry-name'},tasks:legacy}},'correctness').tasks.length,5);
 const cpu=await fixture(t);assert.equal(prepareHypothesisEvaluation({...cpu,objective:'optimization'}).status,'unsupported');
});
test('portable correctness executes both arms and resumes frozen results without another candidate run',async t=>{
 const f=await fixture(t);let calls=0;
 const deps={sample:async({cwd,guidance})=>{calls++;await writeFile(join(cwd,'kernel.py'),guidance.includes('candidate')||calls>8?'answer=42\n':'answer=0\n');return {tokens:1};}};
 const result=await evaluateHypothesis(f,deps);assert.equal(result.report.results.length,20);assert.equal(result.decision.promote,true);assert.equal(calls,20);
 const allocation=JSON.parse(await readFile(join(f.directory,'hypothesis.json'),'utf8'));
 await evaluateHypothesis({...f,remainingSeconds:800},{sample:()=>{throw Error('Completed sample replayed');}});
 assert.deepEqual(JSON.parse(await readFile(join(f.directory,'hypothesis.json'),'utf8')),allocation);
 await writeFile(join(f.skills.candidate,'skill.md'),'different candidate');
 await assert.rejects(evaluateHypothesis({...f,remainingSeconds:700},deps),/Frozen hypothesis/);
});
test('frozen trusted cases and runtime identity cannot change during resume',async t=>{
 const f=await fixture(t);await evaluateHypothesis(f,{sample:async()=>({tokens:1})});
 await writeFile(f.tasks[0].checker,'raise SystemExit(0)\n');
 await assert.rejects(evaluateHypothesis({...f,remainingSeconds:800}),/Frozen hypothesis/);
});
test('A5 samples and paired timings use the registered environment, reserved resource and native kernel identity',async t=>{
 const f=await fixture(t,{target:'a5',objective:'optimization'}),resource={id:'reserved',metadata:{hostSlot:0,backgroundSlot:true}};
 const seen=[],actorSignals=[];
 const deps={createEnvironment:async(options,context)=>{
  assert.equal(options.health.product,'a5');assert.equal(context.resource,resource);
  return {prepare:async value=>{assert.equal(value.project.bundle.root,f.directory);},execute:async(command,cwd,opts)=>{
   seen.push({purpose:opts.purpose,signal:opts.signal,stopSignal:opts.stopSignal});
   let receipt;if(opts.purpose==='performance')receipt={correct:true,activeKernelSha256:digest(await readFile(join(cwd,'active/kernel.py'))),candidateKernelSha256:digest(await readFile(join(cwd,'candidate/kernel.py'))),activeLatencySeconds:1,candidateLatencySeconds:.8,evidence:{fixture:true}};
   else receipt={correct:true,kernelSha256:digest(await readFile(join(cwd,'kernel.py'))),latencySeconds:1,evidence:{fixture:true}};
   opts.onData?.(JSON.stringify(receipt)+'\n');return {settled:true,exitCode:0,handle:'local:fixture'};
  }};
 },nativeSample:async args=>{assert.equal(args.targetBash,true);actorSignals.push(args.sampleStopSignal);await writeFile(join(args.cwd,'kernel.py'),'answer=42\n');return {tokens:1};}};
 const result=await evaluateHypothesis({...f,resource},deps);
 assert.equal(result.decision.promote,true);assert.equal(result.decision.passes.candidate,10);assert.equal(seen.filter(x=>x.purpose==='verification').length,20);assert.equal(seen.filter(x=>x.purpose==='performance').length,10);
 assert.ok(seen.every(x=>x.stopSignal instanceof AbortSignal));assert.ok(actorSignals.every(x=>x instanceof AbortSignal));
 assert.equal(await hypothesisEvaluationSettled({directory:f.directory}),true);
});
test('target execution with unknown closure stays retained and prevents a second dispatch',async t=>{
 const f=await fixture(t,{target:'a3'});let commands=0;
 const deps={createEnvironment:async()=>({prepare:async()=>{},execute:async()=>{commands++;return {settled:false,handle:'remote:fixture:job:same-id'};}}),nativeSample:async()=>({tokens:1})};
 await assert.rejects(evaluateHypothesis(f,deps),{code:'RSI_EXECUTION_UNRESOLVED'});
 assert.equal(await hypothesisEvaluationSettled({directory:f.directory}),false);assert.equal(commands,1);
 await assert.rejects(evaluateHypothesis({...f,remainingSeconds:800},deps),/unresolved/);assert.equal(commands,1);
});
test('no work is admitted when only cleanup allocation remains',async t=>{
 const f=await fixture(t);const result=await evaluateHypothesis({...f,remainingSeconds:20},{runMicroeval:()=>{throw Error('Late dispatch');}});
 assert.equal(result.status,'elapsed');assert.equal(result.decision.promote,false);
});
test('an incomplete comparison cannot promote even if partial input resembles a gain',async t=>{
 const f=await fixture(t),results=f.tasks.flatMap(task=>[0,1].flatMap(run=>['active','candidate'].map(arm=>({projectId:task.id,run,arm,correct:arm==='candidate'}))));
 const result=await evaluateHypothesis(f,{runMicroeval:async()=>({status:'elapsed',results})});
 assert.equal(result.decision.promote,false);assert.equal(result.decision.reason,'inconclusive');
});

test('retrieval snapshot reconnect retains its frozen identity rather than the transient URL',async t=>{
 const f=await fixture(t);f.binding.knowledge={url:'http://127.0.0.1:1111/mcp',identity:'frozen-snapshot'};
 await evaluateHypothesis(f,{sample:async()=>({tokens:1})});
 const result=await evaluateHypothesis({...f,binding:{...f.binding,knowledge:{...f.binding.knowledge,url:'http://127.0.0.1:2222/mcp'}},remainingSeconds:800},{sample:()=>{throw Error('Completed samples replayed');}});
 assert.equal(result.report.results.length,20);
 await assert.rejects(evaluateHypothesis({...f,binding:{...f.binding,knowledge:{...f.binding.knowledge,identity:'changed-snapshot'}},remainingSeconds:700}),/Frozen hypothesis/);
});
test('positive exact-command closure ends an interrupted sample without late credit or redispatch',async t=>{
 const f=await fixture(t,{target:'a3'});let commands=0,stopped=false;const handle='remote:fixture:job:original-id';
 const deps={createEnvironment:async()=>({prepare:async()=>{},execute:async()=>{commands++;return {settled:false,handle};}}),nativeSample:async()=>({tokens:1})};
 await assert.rejects(evaluateHypothesis(f,deps),{code:'RSI_EXECUTION_UNRESOLVED'});
 const file=join(f.directory,'micro/results.json'),before=JSON.parse(await readFile(file,'utf8')),observed=[];
 const observer={createEnvironment:async()=>({observe:async id=>{observed.push(id);return {state:stopped?'stopped':'running',exitCode:0};},reconcileLease:async()=>true,prepare:()=>{throw Error('Unexpected prepare');},execute:()=>{throw Error('Unexpected replay');}})};
 assert.equal(await hypothesisEvaluationSettled({directory:f.directory},observer),false);
 stopped=true;assert.equal(await hypothesisEvaluationSettled({directory:f.directory},observer),true);
 const after=JSON.parse(await readFile(file,'utf8'));assert.equal(after.elapsedSeconds,before.elapsedSeconds);assert.equal(after.results.length,0);assert.equal(after.runningSample,undefined);
 const proof=JSON.parse(await readFile(after.settledInterruption.closure,'utf8'));assert.equal(proof.marker.unresolvedExecution.handle,handle);assert.deepEqual(observed,[handle,handle]);
 const result=await evaluateHypothesis({...f,remainingSeconds:800},deps);assert.equal(result.status,'inconclusive');assert.equal(result.decision.promote,false);assert.equal(commands,1);
});
test('an unknown reservation remains blocked until its exact token has positive closure',async t=>{
 const f=await fixture(t,{target:'a5'}),reservation={physicalDevice:3,receipt:'/fixture/lease.json',token:'original-token'};
 await assert.rejects(evaluateHypothesis(f,{createEnvironment:async()=>({prepare:async()=>{},execute:async()=>{throw Object.assign(Error('Device reservations require reconciliation'),{code:'DEVICE_LEASE_BLOCKED',reservations:[reservation]});}}),nativeSample:async()=>({})}),{code:'RSI_EXECUTION_UNRESOLVED'});
 let closed=false;const deps={createEnvironment:async()=>({reconcileReservation:async(id,proof)=>{assert.equal(id,3);assert.deepEqual(proof,{receipt:reservation.receipt,token:reservation.token});return {closed};}})};
 assert.equal(await hypothesisEvaluationSettled({directory:f.directory},deps),false);closed=true;
 assert.equal(await hypothesisEvaluationSettled({directory:f.directory},deps),true);
});
test('paired replay retains the unresolved native command and closes without claiming late speedup',async t=>{
 const f=await fixture(t,{target:'a5',objective:'optimization'});let calls=0;
 const deps={createEnvironment:async()=>({prepare:async()=>{},execute:async(_command,cwd,opts)=>{
  calls++;if(opts.purpose==='performance')return {settled:false,handle:'remote:fixture:job:paired-original'};
  opts.onData(JSON.stringify({correct:true,kernelSha256:digest(await readFile(join(cwd,'kernel.py'))),latencySeconds:1,evidence:{fixture:true}})+'\n');return {settled:true,exitCode:0};
 }}),nativeSample:async()=>({tokens:1})};
 await assert.rejects(evaluateHypothesis(f,deps),{code:'RSI_EXECUTION_UNRESOLVED'});
 const before=JSON.parse(await readFile(join(f.directory,'micro/results.json'),'utf8'));assert.equal(before.pairedReplay.runningPair.unresolvedExecution.handle,'remote:fixture:job:paired-original');assert.equal(calls,21);
 assert.equal(await hypothesisEvaluationSettled({directory:f.directory},{createEnvironment:async()=>({observe:async handle=>{assert.equal(handle,'remote:fixture:job:paired-original');return {state:'stopped'};},reconcileLease:async()=>true})}),true);
 const result=await evaluateHypothesis({...f,remainingSeconds:800},deps);assert.equal(result.decision.promote,false);assert.equal(result.status,'inconclusive');assert.equal(calls,21);
});

test('device correctness verifies source identity without demanding latency evidence',async t=>{
 const f=await fixture(t,{target:'a5'});const result=await evaluateHypothesis(f,{createEnvironment:async()=>({prepare:async()=>{},execute:async(_command,cwd,opts)=>{
  opts.onData(JSON.stringify({correct:true,kernelSha256:digest(await readFile(join(cwd,'kernel.py'))),evidence:{numeric:true}})+'\n');return {settled:true,exitCode:0};
 }}),nativeSample:async()=>({tokens:1})});
 assert.ok(result.report.results.every(r=>r.correct===true&&r.kernelSha256&&r.latencySeconds===undefined));
 assert.equal(result.decision.reason,'insufficient-correctness-improvement');
});
