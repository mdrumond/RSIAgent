import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {evaluateSkillObjective} from '../../controller/skill-objectives.mjs';
import {runMicroeval,finalizeKernelPairs} from '../../cli/skill-microeval.mjs';
import {validateClassification} from '../../controller/skill-review.mjs';
const ids=['a','b','c','d','e'];
const rows=()=>ids.flatMap(projectId=>[0,1].flatMap(run=>['active','candidate'].map(arm=>({projectId,taskId:projectId,run,arm,correct:true,kernelSha256:arm,latencySeconds:999,tokens:1,elapsedSeconds:1}))));
const options=objective=>({objective,projectIds:ids,runs:2});
test('correctness objective admits two extra passes even with regressions and larger costs',()=>{
 const results=rows();results[0].correct=false;results[4].correct=false;results[8].correct=false;results[13].correct=false;
 results[1].tokens=1e9;assert.equal(evaluateSkillObjective(results,options('correctness')).promote,true);
 results[4].correct=true;assert.equal(evaluateSkillObjective(results,options('correctness')).promote,false);
 assert.equal(evaluateSkillObjective(results.slice(1),options('correctness')).reason,'inconclusive');
});
test('optimization requires correct paired kernels and aggregate speedup, not every pair',()=>{
 const results=rows(),pairedTimings=ids.flatMap(projectId=>[0,1].map(run=>({projectId,run,correct:true,evidence:{alternating:true},activeKernelSha256:'active',candidateKernelSha256:'candidate',activeLatencySeconds:1,candidateLatencySeconds:projectId==='a'?1.1:.8})));
 const config={...options('optimization'),pairedTimings};assert.equal(evaluateSkillObjective(results,config).promote,true);
 results[0].correct=false;assert.equal(evaluateSkillObjective(results,config).reason,'incorrect-kernel');results[0].correct=true;
 pairedTimings[0].candidateKernelSha256='other';assert.equal(evaluateSkillObjective(results,config).reason,'kernel-timing-identity-mismatch');
 assert.equal(evaluateSkillObjective(results,options('optimization')).reason,'missing-paired-kernel-timing');
 assert.throws(()=>evaluateSkillObjective(results,{...config,runs:1}));
 assert.equal(evaluateSkillObjective(results,options('harness')).reason,'user-intervention-required');
});
test('failure bucket is orthogonal to attribution and legacy classifications remain compatible',()=>{
 const group={id:'g',occurrences:[{reference:'e'}]},base={cause:'specification',reason:'actual diagnostic',coverage:'uncovered',guidanceIds:[],evidenceRefs:['e']};
 assert.equal(validateClassification({...base,bucket:'correctness'},group,[]).bucket,'correctness');
 assert.equal(validateClassification({...base,bucket:'harness'},group,[]).intervention,'user-required');
 assert.equal(validateClassification(base,group,[]).bucket,undefined);
 assert.throws(()=>validateClassification({...base,bucket:'other'},group,[]));
});
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'objective-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const skills={};for(const arm of ['active','candidate']){skills[arm]=join(root,arm);await mkdir(skills[arm]);await writeFile(join(skills[arm],'skill-map.json'),JSON.stringify({implementation:'skill.md'}));await writeFile(join(skills[arm],'skill.md'),arm);}
 const checker=join(root,'checker.py');await writeFile(checker,"import json,sys,pathlib,hashlib\np=pathlib.Path(sys.argv[1])/'kernel.py'\nprint(json.dumps(dict(correct=True,latencySeconds=1,kernelSha256=hashlib.sha256(p.read_bytes()).hexdigest(),evidence={'verified':True})))\n");
 return {directory:join(root,'evaluation'),skills,profile:{provider:'fixture',model:'fixture'},objective:'optimization',tasks:ids.map(id=>({id,kind:'a3-kernel',bucket:'optimization',prompt:'tiny fixture',checker,pairedChecker:checker,files:{'kernel.py':'answer=42\n','proposal.json':'{"a":1}'},protectedFiles:{'proposal.json':'json'}})),runs:2,seconds:5,python:process.env.RSI_TEST_PYTHON??'python3'};
}
test('native kernel verification binds actual source; JSON protection tolerates formatting, not value changes',async t=>{
 const f=await fixture(t);const report=await runMicroeval(f,{sample:async({cwd})=>{await writeFile(join(cwd,'proposal.json'),'{\n "a": 1\n}\n');return {tokens:1};}});
 assert.ok(report.results.every(r=>r.correct&&r.kernelSha256));assert.equal(report.decision.reason,'missing-paired-kernel-timing');
 let count=0;const measured=await finalizeKernelPairs({directory:f.directory,python:f.python,seconds:20},{measure:async({active,candidate})=>{count++;return {correct:true,activeKernelSha256:report.results[0].kernelSha256,candidateKernelSha256:report.results[0].kernelSha256,activeLatencySeconds:1,candidateLatencySeconds:.9,evidence:{alternating:true}};}});
 assert.equal(count,10);assert.equal(measured.decision.promote,true);
 await finalizeKernelPairs({directory:f.directory,seconds:20},{measure:()=>{throw Error('must not replay');}});
 await writeFile(join(f.directory,'a','0-active','work','kernel.py'),'changed=1\n');
 await assert.rejects(finalizeKernelPairs({directory:f.directory,seconds:20}),/kernel changed/);
 const changed=await fixture(t);const bad=await runMicroeval(changed,{sample:async({cwd})=>{await writeFile(join(cwd,'proposal.json'),'{"a":2}');return {};}});assert.ok(bad.results.every(r=>r.correct===null));
});
test('harness or mixed task objectives are rejected before a sample; legacy byte protection stays strict',async t=>{
 const f=await fixture(t);await assert.rejects(runMicroeval({...f,tasks:[{...f.tasks[0],bucket:'harness'}]}),/Harness/);
 await assert.rejects(runMicroeval({...f,tasks:[{...f.tasks[0],bucket:'correctness'}]}),/bucket/);
 const tasks=f.tasks.map(task=>({...task,protectedFiles:{'proposal.json':'bytes'}}));
 const report=await runMicroeval({...f,tasks},{sample:async({cwd})=>{await writeFile(join(cwd,'proposal.json'),'{"a":1}\n');return {};}});assert.ok(report.results.every(r=>r.correct===null));
});
test('kernel receipt identity cannot be supplied by a model; objective and binding stay frozen',async t=>{
 const f=await fixture(t);await writeFile(f.tasks[0].checker,"import json\nprint(json.dumps(dict(correct=True,latencySeconds=1,kernelSha256='forged',evidence={})))\n");
 const report=await runMicroeval({...f,binding:{target:'registered-fixture'}},{sample:async()=>({kernelSha256:'forged',latencySeconds:.01})});
 assert.ok(report.results.every(r=>r.correct===null));
 await assert.rejects(runMicroeval({...f,binding:{target:'changed'},resume:true},{sample:async()=>({})}),/Frozen/);
 await assert.rejects(runMicroeval({...f,objective:'correctness',resume:true},{sample:async()=>({})}),/bucket/);
});
test('paired replay preserves partial receipts and does not replay uncertain executions',async t=>{
 const f=await fixture(t);const report=await runMicroeval(f,{sample:async()=>({})});let count=0;
 await assert.rejects(finalizeKernelPairs({directory:f.directory,seconds:20},{measure:async()=>{
  if(++count===2)throw Error('interrupted');return {correct:true,activeKernelSha256:report.results[0].kernelSha256,candidateKernelSha256:report.results[0].kernelSha256,activeLatencySeconds:1,candidateLatencySeconds:.9,evidence:{alternating:true}};
 }}),/interrupted/);
 const saved=JSON.parse(await readFile(join(f.directory,'results.json')));assert.equal(saved.pairedTimings.length,1);assert.ok(saved.pairedReplay.runningPair);
 await assert.rejects(finalizeKernelPairs({directory:f.directory,seconds:20},{measure:()=>{throw Error('replay');}}),/unresolved/);
 const retained=JSON.parse(await readFile(join(f.directory,'results.json')));assert.equal(retained.pairedReplay.startedAt,saved.pairedReplay.startedAt);
});

test('trusted exit125 harness refusal stays inconclusive while ordinary incorrect kernels fail',async t=>{
 const f=await fixture(t);await writeFile(f.tasks[0].checker,"import json,sys,pathlib,hashlib\np=pathlib.Path(sys.argv[1])/'kernel.py'\nprint(json.dumps(dict(correct=None,bucket='harness',kernelSha256=hashlib.sha256(p.read_bytes()).hexdigest(),evidence={'reason':'foreign process'})))\nsys.exit(125)\n");
 const refused=await runMicroeval(f,{sample:async()=>({})});assert.ok(refused.results.every(r=>r.correct===null&&r.bucket==='harness'&&!r.latencySeconds));assert.equal(refused.decision.promote,false);assert.equal(refused.decision.reason,'inconclusive');
 const bad=await fixture(t);await writeFile(bad.tasks[0].checker,"import sys\nsys.exit(1)\n");
 const incorrect=await runMicroeval(bad,{sample:async()=>({})});assert.ok(incorrect.results.every(r=>r.correct===false&&!r.bucket));assert.equal(incorrect.decision.reason,'incorrect-kernel');
 const forged=await fixture(t);await writeFile(forged.tasks[0].checker,"import json,sys\nprint(json.dumps(dict(correct=None,bucket='harness',kernelSha256='wrong',evidence={})))\nsys.exit(125)\n");
 const invalid=await runMicroeval(forged,{sample:async()=>({})});assert.ok(invalid.results.every(r=>r.correct===false&&!r.bucket));
});
