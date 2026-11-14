import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {prepareHypothesisEvaluation,evaluateHypothesis,hypothesisEvaluationSettled,qualifyHypothesisTasks,hypothesisTarget} from '../../cli/skill-hypothesis-evaluation.mjs';
import {runMicroeval,check,finalizeKernelPairs} from '../../cli/skill-microeval.mjs';
import {evaluationSettings} from '../../controller/skill-objectives.mjs';
const fixtureCommand=(python,checker,cwd,seconds)=>{const r=spawnSync(python,[checker,cwd],{cwd,encoding:'utf8',maxBuffer:1048576,timeout:Math.ceil(seconds*1000)});return {correct:r.status===0,exitCode:r.status,output:r.stdout,error:r.stderr};};
const digest=text=>createHash('sha256').update(text).digest('hex');
async function fixture(t,{target='cpu',objective='correctness'}={}){
 const root=await mkdtemp(join(tmpdir(),'hypothesis-eval-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const skills={};for(const arm of ['active','candidate']){const path=skills[arm]=join(root,arm);await mkdir(path);await writeFile(join(path,'skill-map.json'),JSON.stringify({implementation:'skill.md'}));await writeFile(join(path,'skill.md'),arm);}
 const checker=join(root,'checker.py');await writeFile(checker,"import sys,pathlib,runpy\np=pathlib.Path(sys.argv[1])/'kernel.py'\nraise SystemExit(0 if runpy.run_path(str(p)).get('answer')==42 else 1)\n");
 const binding={target,...(target==='cpu'?{}:{environment:{mode:'local',target:'registered-'+target,runtime:'fixture-runtime',instructionFile:'fixture',health:{product:target}},python:'python3'})};
 const pairedChecker=join(root,'paired.py');if(objective==='optimization')await writeFile(pairedChecker,"import sys,pathlib,json,hashlib\nsha=lambda p:hashlib.sha256((pathlib.Path(p)/'kernel.py').read_bytes()).hexdigest()\nprint(json.dumps(dict(correct=True,activeKernelSha256=sha(sys.argv[1]),candidateKernelSha256=sha(sys.argv[2]),activeLatencySeconds=1,candidateLatencySeconds=.8,evidence={'paired':True})))\n");
 const tasks=['one','two','three','four','five'].map(id=>({id,kind:target==='cpu'?'hypothesis-case':'target-kernel',target,bucket:objective,prompt:'Repair answer',checker,...(objective==='optimization'?{pairedChecker}:{}),files:{'kernel.py':'answer=0\n','nested/config.json':'{}'},protectedFiles:{'nested/config.json':'json'}}));
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
  let checker;return {prepare:async value=>{assert.equal(value.project.bundle.root,f.directory);checker=value.project.verifier.files[0];},execute:async(command,cwd,opts)=>{
   seen.push({purpose:opts.purpose,signal:opts.signal,stopSignal:opts.stopSignal});
   if(opts.purpose==='performance'){const result=fixtureCommand(f.config.python,checker,cwd,opts.timeout);opts.onData(result.output);return {...result,settled:true,handle:'local:fixture'};}
   const receipt={correct:true,kernelSha256:digest(await readFile(join(cwd,'kernel.py'))),latencySeconds:1,evidence:{fixture:true}};
   opts.onData?.(JSON.stringify(receipt)+'\n');return {settled:true,exitCode:0,handle:'local:fixture'};
  }};
 },nativeSample:async args=>{assert.equal(args.targetBash,true);actorSignals.push(args.sampleStopSignal);await writeFile(join(args.cwd,'kernel.py'),'answer=42\n');return {tokens:1};}};
 const result=await evaluateHypothesis({...f,resource},deps);
 assert.equal(result.decision.promote,true);assert.equal(result.decision.passes.candidate,10);assert.equal(seen.filter(x=>x.purpose==='verification').length,20);assert.equal(seen.filter(x=>x.purpose==='performance').length,1);assert.equal(result.report.pairedTimings.length,10);assert.ok(result.report.pairedTimings.every(r=>r.verification.handle==='local:fixture'&&r.verification.sourceFilesMatch));
 assert.ok(seen.every(x=>x.stopSignal instanceof AbortSignal));assert.ok(actorSignals.every(x=>x instanceof AbortSignal));
 assert.equal(await hypothesisEvaluationSettled({directory:f.directory}),true);
 const state=JSON.parse(await readFile(join(f.directory,'hypothesis.json'),'utf8')),file=join(f.directory,'micro/results.json'),partial=JSON.parse(await readFile(file,'utf8'));partial.pairedTimings=partial.pairedTimings.slice(0,8);await writeFile(file,JSON.stringify(partial));
 const resumed=await finalizeKernelPairs({directory:join(f.directory,'micro'),seconds:state.pairedSeconds},hypothesisTarget({root:f.directory,binding:f.binding,resource},deps));assert.equal(resumed.pairedTimings.length,10);assert.deepEqual(resumed.pairedTimings.slice(0,8),partial.pairedTimings);assert.equal(seen.filter(x=>x.purpose==='performance').length,2);
 await writeFile(join(f.directory,'micro/one/0-active/work/kernel.py'),'answer=43\n');await assert.rejects(finalizeKernelPairs({directory:join(f.directory,'micro'),seconds:state.pairedSeconds},{measureBatch:()=>{throw Error('Changed source reached target');}}),/kernel changed/);
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

const controls=tasks=>tasks.map(t=>({...t,controls:{reference:{files:{'kernel.py':'answer=42\n'}},negatives:[{files:{'kernel.py':'answer=0\n'}}]}}));
test('new cases need independently passing reference and rejected negative controls before admission',async t=>{
 const f=await fixture(t);assert.equal((await qualifyHypothesisTasks(f)).status,'needs-learning');f.tasks=controls(f.tasks);
 const result=await qualifyHypothesisTasks(f);assert.equal(result.status,'qualified');assert.equal(result.receipts.length,10);assert.ok(result.receipts.every(r=>r.passed));
 const saved=JSON.parse(await readFile(join(f.directory,'micro/results.json'),'utf8'));const resumed=await qualifyHypothesisTasks({...f,remainingSeconds:800},{verify:()=>{throw Error('Passed control replayed');}});assert.equal(resumed.status,'qualified');assert.equal(resumed.elapsedSeconds,saved.elapsedSeconds);
 f.tasks[0].controls.negatives[0].files['kernel.py']='answer=1\n';await assert.rejects(qualifyHypothesisTasks({...f,remainingSeconds:700}),/Frozen qualification/);
});
test('bad reference, always-passing checker and negative usage errors cannot qualify',async t=>{
 for(const fault of ['reference','accepts-all','usage']){
  const f=await fixture(t);f.tasks=controls(f.tasks);
  if(fault==='reference')f.tasks[0].controls.reference.files['kernel.py']='answer=0\n';
  if(fault==='accepts-all')await writeFile(f.tasks[0].checker,'raise SystemExit(0)\n');
  const result=await qualifyHypothesisTasks(f,fault==='usage'?{verify:async args=>({exitCode:args.cwd.includes('negative')?2:0,correct:!args.cwd.includes('negative')})}:{});
  assert.equal(result.status,'blocked');assert.ok(result.receipts.some(r=>!r.passed));
 }
});
test('A5 control qualification uses the bound environment and trusted source receipts',async t=>{
 const f=await fixture(t,{target:'a5'});f.tasks=controls(f.tasks);const resource={id:'reserved',metadata:{hostSlot:0}};let commands=0;const counts=[];
 await writeFile(f.tasks[0].checker,"import sys,pathlib,runpy,json,hashlib\np=pathlib.Path(sys.argv[1])/'kernel.py'\npassed=runpy.run_path(str(p)).get('answer')==42\nprint(json.dumps(dict(correct=passed,kernelSha256=hashlib.sha256(p.read_bytes()).hexdigest(),evidence={'numeric':True})))\nraise SystemExit(0 if passed else 1)\n");
 const deps={createEnvironment:async(options,context)=>{
  assert.equal(options.health.product,'a5');assert.equal(context.resource,resource);let checker;
  return {prepare:async value=>{checker=value.project.verifier.files[0];},execute:async(_command,cwd,opts)=>{commands++;const r=fixtureCommand(f.config.python,checker,cwd,opts.timeout);counts.push(r.output.trim().split('\n').length);opts.onData(r.output);return {...r,settled:true,handle:'local:qualified'};}};
 }};const result=await qualifyHypothesisTasks({...f,resource},deps);
 assert.equal(result.status,'qualified');assert.equal(commands,1);assert.ok(result.receipts.filter(r=>r.expectedExit===0).every(r=>r.verification.receipt.kernelSha256));
 const file=join(f.directory,'micro/results.json'),saved=JSON.parse(await readFile(file,'utf8'));saved.status='running';saved.results=saved.results.slice(0,8);await writeFile(file,JSON.stringify(saved));
 const resumed=await qualifyHypothesisTasks({...f,resource,remainingSeconds:800},deps);assert.equal(resumed.status,'qualified');assert.deepEqual(counts,[10,2]);assert.deepEqual(resumed.receipts.slice(0,8),saved.results);
 const bad=await fixture(t,{target:'a5'});bad.tasks=controls(bad.tasks);await writeFile(bad.tasks[0].checker,await readFile(f.tasks[0].checker));bad.tasks[0].controls.negatives[0].files['kernel.py']='answer=42\n';const rejected=await qualifyHypothesisTasks({...bad,resource},deps);assert.equal(rejected.status,'blocked');assert.ok(rejected.receipts.some(r=>r.expectedExit===1&&!r.passed));

});
test('unknown qualification execution is retained and positive closure never turns it into a passing control',async t=>{
 const f=await fixture(t,{target:'a3'});f.tasks=controls(f.tasks);let commands=0;
 const deps={createEnvironment:async()=>({prepare:async()=>{},execute:async()=>{commands++;return {settled:false,handle:'remote:fixture:job:control-original'};}})};
 await assert.rejects(qualifyHypothesisTasks(f,deps),{code:'RSI_EXECUTION_UNRESOLVED'});
 await assert.rejects(qualifyHypothesisTasks({...f,remainingSeconds:800},deps),{code:'RSI_EXECUTION_UNRESOLVED'});assert.equal(commands,1);
 assert.equal(await hypothesisEvaluationSettled({directory:f.directory},{createEnvironment:async()=>({observe:async id=>{assert.equal(id,'remote:fixture:job:control-original');return {state:'stopped',exitCode:0};},reconcileLease:async()=>true})}),true);
 const result=await qualifyHypothesisTasks({...f,remainingSeconds:700},deps);assert.equal(result.status,'blocked');assert.equal(result.receipts.length,0);assert.equal(commands,1);
});
test('qualification resumes unfinished controls while retaining passed receipts and elapsed spend',async t=>{
 const f=await fixture(t);f.tasks=controls(f.tasks);const abort=new AbortController();let calls=0;
 await assert.rejects(qualifyHypothesisTasks({...f,signal:abort.signal},{verify:async args=>{if(++calls===2){abort.abort(Error('operator pause'));throw abort.signal.reason;}return check(f.config.python,args.checker,args.cwd,args.seconds,args.signal);}}),/operator pause/);
 const before=JSON.parse(await readFile(join(f.directory,'micro/results.json'),'utf8'));assert.equal(before.results.length,1);let resumed=0;
 const result=await qualifyHypothesisTasks({...f,remainingSeconds:800},{verify:async args=>{resumed++;return check(f.config.python,args.checker,args.cwd,args.seconds,args.signal);}});
 const after=JSON.parse(await readFile(join(f.directory,'micro/results.json'),'utf8'));assert.equal(result.status,'qualified');assert.equal(resumed,9);assert.deepEqual(after.results[0],before.results[0]);assert.ok(after.elapsedSeconds>=before.elapsedSeconds);
});
