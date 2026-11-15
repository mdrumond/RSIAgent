import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {frozenEvidencePacket,scopedBaseline} from '../../cli/skill-review-evidence.mjs';
test('frozen diagnostics deliver content, deduplicate, and distinguish missing and truncated evidence',async t=>{
 const root=await mkdtemp(join(tmpdir(),'review-packet-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const file=join(root,'copied.log');await writeFile(file,'HEAD'+'.'.repeat(80)+'TAIL');
 const packet=await frozenEvidencePacket([{occurrences:[{reference:'event:1',artifacts:[{path:file}]},{reference:'event:2',artifacts:[{path:file}]},{reference:'event:3',artifacts:[{path:join(root,'missing')}]},{reference:'event:4'}]}],{limit:16});
 assert.equal(packet.artifacts.length,2);assert.equal(packet.artifacts[0].text,'HEAD....'+'....TAIL');assert.equal(packet.artifacts[0].truncated,true);
 assert.deepEqual(packet.observations[0].artifacts,packet.observations[1].artifacts);assert.equal(packet.artifacts[1].available,false);assert.equal(packet.artifacts[1].reason,'ENOENT');assert.ok(packet.observations[3].unavailable);
});
test('scoped baseline retains global and protected constraints and relevant stage',()=>{
 const entries=[{id:'global',scope:'global-kdb'},{id:'safe',scope:'verify',critical:true},{id:'actor',scope:'actor'},{id:'unrelated',scope:'postprocess'}];
 assert.deepEqual(scopedBaseline(entries,{changes:[{scope:'actor'}]}).map(e=>e.id),['global','safe','actor']);
});
test('qualified suite reuse rejects changed source and checker content',async t=>{
 const {registerQualifiedSuite,qualifiedSuites}=await import('../../cli/skill-priority-cases.mjs');
 const root=await mkdtemp(join(tmpdir(),'review-suite-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const checker=join(root,'checker.py');await writeFile(checker,'raise SystemExit(0)');
 const suite={target:'cpu',objective:'correctness',binding:{target:'cpu'},mechanism:'Tail boundaries',tasks:Array.from({length:5},(_,i)=>({id:'case-'+i,checker,files:{'kernel.py':'answer=1'},controls:{reference:{files:{'kernel.py':'answer=2'}},negatives:[{files:{'kernel.py':'answer=1'}}]}}))};
 await assert.rejects(registerQualifiedSuite({config:{root},suite,qualification:{status:'failed'}}));
 await registerQualifiedSuite({config:{root},suite,qualification:{status:'qualified'},provenance:{projectId:'p'}});
 assert.equal((await qualifiedSuites({config:{root}})).length,1);
 await writeFile(checker,'raise SystemExit(1)');assert.equal((await qualifiedSuites({config:{root}})).length,0);
});
test('a final revised proposal must pass relevance selection again before cached cases are reused',async t=>{
 const {resolveHypothesisCases}=await import('../../cli/skill-priority-cases.mjs');
 const root=await mkdtemp(join(tmpdir(),'review-relevance-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const checker=join(root,'checker.py');await writeFile(checker,'raise SystemExit(0)');
 const tasks=Array.from({length:5},(_,i)=>({id:'case-'+i,kind:'hypothesis-case',target:'cpu',bucket:'correctness',prompt:'Check boundary',checker,files:{'kernel.py':'answer=0'},controls:{reference:{files:{'kernel.py':'answer=1'}},negatives:[{files:{'kernel.py':'answer=0'}}]}}));
 const config={root,target:'a3',environment:{options:{}},skills:{priority:{evaluations:[{id:'boundary',description:'Boundary contract',target:'cpu',objective:'correctness',binding:{target:'cpu'},tasks}]}}};
 const item={id:'item',objective:'correctness',proposal:{changes:[{id:'boundary',text:'Original boundary'}]},evidence:{groups:[]}};
 const seen=[],sessions=[],deps={prepareTasks:async()=>({manifest:{hypotheses:[]},byHypothesis:{}}),select:async({directory,prompt,validate})=>{
  const submission=join(directory,'submission.json');
  try{return validate(JSON.parse(await readFile(submission,'utf8')));}catch(error){if(error.code!=='ENOENT')throw error;}
  seen.push(prompt.hypothesis.proposal);sessions.push(submission);await mkdir(directory,{recursive:true});
  const result=validate({catalogueId:'boundary',reason:'This boundary is covered'});await writeFile(submission,JSON.stringify(result));return result;
 }};
 const args={config,backlog:{db:{prepare:()=>({get:()=>undefined})}},item,directory:join(root,'cases'),seconds:20};
 const first=await resolveHypothesisCases(args,deps);assert.equal(first.status,'ready');
 await resolveHypothesisCases(args,deps);assert.equal(seen.length,1);
 const final={changes:[{id:'boundary',text:'Revised boundary'}]};const second=await resolveHypothesisCases({...args,proposal:final},deps);
 assert.equal(second.status,'ready');assert.equal(seen.length,2);assert.deepEqual(seen[1],final);assert.notEqual(first.proposalIdentity,second.proposalIdentity);
 await writeFile(checker,'raise SystemExit(1)');
 const third=await resolveHypothesisCases({...args,proposal:final},deps);
 assert.equal(third.status,'ready');assert.equal(seen.length,3);assert.notEqual(second.catalogueIdentity,third.catalogueIdentity);
 assert.equal(new Set(sessions).size,3);for(const file of sessions)assert.equal(JSON.parse(await readFile(file,'utf8')).catalogueId,'boundary');
});
test('configured relative checker paths use the campaign root independently of controller cwd',async t=>{
 const {suiteFingerprint}=await import('../../cli/skill-priority-cases.mjs');
 const root=await mkdtemp(join(tmpdir(),'review-relative-'));t.after(()=>rm(root,{recursive:true,force:true}));assert.notEqual(root,process.cwd());
 await mkdir(join(root,'checks'));await writeFile(join(root,'checks','check.py'),'raise SystemExit(0)');
 const relative={target:'cpu',tasks:[{id:'case',checker:'checks/check.py',pairedChecker:'checks/check.py'}]};
 const absolute={...relative,tasks:[{id:'case',checker:join(root,'checks','check.py'),pairedChecker:join(root,'checks','check.py')}]};
 assert.equal(await suiteFingerprint(relative,root),await suiteFingerprint(absolute,root));
});
