import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {tmpdir} from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import {runFailurePilot,replayRetainedFailureTask,validatePilotManifest,validatePilotTrial,pilotWinner} from '../../cli/skill-failure-pilot.mjs';
const task=id=>({id,caseIds:['mechanism','control'],controlCaseIds:['control'],files:{'kernel.py':'public starter'}});
const family=id=>({id,development:task('development'),evidence:[{reference:'historical:'+id}],transfer:[{...task('transfer-a'),secret:'PRIVATE-A'},{...task('transfer-b'),secret:'PRIVATE-B'}]});
const candidate=(id,text=id)=>({id,text,evidence:[{reference:'historical:first',startLine:1,endLine:1}]});
const rows=[candidate('principle-a','Longer broadly applicable principle'),candidate('principle-b','Other broadly applicable principle'),candidate('principle-c','Short principle')];
const result=(correct,extra={})=>({correct,mechanismMatch:!correct,caseResults:[{id:'mechanism',correct},{id:'control',correct:true}],sourceSha256:'a'.repeat(64),artifacts:[],handles:[],...extra});
async function fixture(t){
 const directory=await mkdtemp(join(tmpdir(),'failure-pilot-'));t.after(()=>rm(directory,{recursive:true,force:true}));
 const manifest={baseline:'Current skills',identities:{model:{id:'frozen-model'},runtime:'a3-runtime',memory:'readonly-snapshot'},families:[family('first')]};
 const calls=[],ops={qualify:async()=>({qualified:true,referencePassed:true,negativeFailed:true}),design:async input=>{calls.push(['design',input]);return rows;},trial:async input=>{
  calls.push(['trial',input]);await delay(2);
  return result(input.phase==='screen'?false:input.arm==='baseline'?input.repetition===0:input.arm==='principle-b'?input.repetition<3:true);
 }};
 return {directory,manifest,calls,ops,options:{directory,manifest,seconds:120,trialSeconds:10,concurrency:4}};
}
test('parallel repeated trials freeze one winner and independently confirm both unseen tasks',async t=>{
 const f=await fixture(t);let active=0,peak=0;
 const report=await runFailurePilot(f.options,{...f.ops,trial:async input=>{
  active++;peak=Math.max(peak,active);
  try{if(input.phase==='confirmation'){const winner=JSON.parse(await readFile(join(f.directory,'winner.json')));assert.equal(input.arm==='baseline'||input.guidance===winner.text,true);}return await f.ops.trial(input);}finally{active--;}
 }});
 assert.equal(report.status,'complete');assert.equal(report.promising,true);assert.equal(report.winner.id,'principle-c');assert.equal(peak,4);assert.equal(active,0);
 const calls=f.calls.filter(([kind])=>kind==='trial').map(([,input])=>input);
 assert.equal(calls.length,45);assert.equal(new Set(calls.map(c=>c.directory)).size,45);
 assert.equal(calls.filter(c=>c.phase==='screen').length,5);assert.equal(calls.filter(c=>c.phase==='development').length,20);assert.equal(calls.filter(c=>c.phase==='confirmation').length,20);
 for(let repetition=0;repetition<5;repetition++){
  const arms=calls.filter(c=>c.phase==='development'&&c.repetition===repetition).map(c=>c.arm).sort();
  assert.deepEqual(arms,['baseline',...rows.map(r=>r.id)]);
 }
 for(const [kind,input]of f.calls){if(kind==='design'||input.phase!=='confirmation'){const encoded=JSON.stringify(input);assert(!encoded.includes('PRIVATE-A'));assert(!encoded.includes('PRIVATE-B'));}}
 for(const [,input]of f.calls.filter(([kind])=>kind==='trial'))assert.deepEqual(input.identities,f.manifest.identities);
 const count=f.calls.length,replayed=await runFailurePilot(f.options,{...f.ops,trial:()=>{throw Error('Must not replay');},design:()=>{throw Error('Must not redraft');}});
 assert.deepEqual(replayed,report);assert.equal(f.calls.length,count);
});
test('reference or negative qualification failure never starts actors or writers',async t=>{
 const f=await fixture(t);let calls=0;
 const report=await runFailurePilot(f.options,{qualify:async()=>({qualified:true,referencePassed:true,negativeFailed:false}),trial:async()=>{calls++;},design:async()=>{calls++;}});
 assert.equal(report.status,'complete');assert.equal(calls,0);assert.equal(report.candidates,null);
});
test('baseline ceiling advances the ordered shortlist without creating harder prompts',async t=>{
 const f=await fixture(t);f.manifest.families.push(family('second'));
 const report=await runFailurePilot(f.options,{...f.ops,trial:async input=>result(input.familyId==='first'?true:input.phase==='screen'?false:input.arm!=='baseline')});
 assert.equal(report.selectedFamily,'second');assert.equal(report.promising,true);
 const design=f.calls.find(([kind])=>kind==='design')[1];assert.equal(design.family.id,'second');assert.deepEqual(design.family.development,f.manifest.families[1].development);
});
test('infrastructure failures are inconclusive and do not qualify baseline struggle',async t=>{
 const f=await fixture(t);let designed=0;
 const report=await runFailurePilot(f.options,{...f.ops,trial:async()=>({infrastructure:true,correct:null,reason:'Target unavailable'}),design:async()=>{designed++;return rows;}});
 assert.equal(report.status,'inconclusive');assert.equal(designed,0);assert.equal(report.candidates,null);
});
test('held-out performance cannot select the development winner or trigger prompt revision',async t=>{
 const f=await fixture(t);let designs=0;
 const report=await runFailurePilot(f.options,{...f.ops,design:async input=>{designs++;return f.ops.design(input);},trial:async input=>result(input.phase==='screen'?false:input.phase==='confirmation'?input.arm==='baseline':input.arm!=='baseline')});
 assert.equal(report.status,'complete');assert.equal(report.promising,false);assert.equal(report.winner.id,'principle-c');assert.equal(designs,1);
});
test('control regression rejects an otherwise higher-scoring candidate',()=>{
 const base=Array.from({length:5},(_,i)=>result(i===0));const good=Array.from({length:5},()=>result(true));const bad=good.map((r,i)=>i? r:result(false,{caseResults:[{id:'mechanism',correct:false},{id:'control',correct:false,control:true}]}));
 for(const row of base)row.caseResults[1].control=true;
 assert.equal(pilotWinner({baseline:base,'principle-a':bad,'principle-b':good,'principle-c':base},rows).id,'principle-b');
});
test('frozen inputs cannot change on resume and the retained evidence stays byte-identical',async t=>{
 const f=await fixture(t);await runFailurePilot(f.options,f.ops);const before=await readFile(join(f.directory,'pilot.json'),'utf8');
 await assert.rejects(runFailurePilot({...f.options,manifest:{...f.manifest,baseline:'Changed guidance'}},f.ops),/Frozen pilot/);
 assert.equal(await readFile(join(f.directory,'pilot.json'),'utf8'),before);
});
test('interrupted cells require positive closure; recovered outputs reuse spending and do not rerun actors',async t=>{
 const f=await fixture(t);let clock=0,wall=10000,calls=0;
 const interrupted=await runFailurePilot({...f.options,concurrency:1},{...f.ops,now:()=>clock,wallNow:()=>wall,trial:async()=>{calls++;clock+=3000;throw Error('Transport disconnected');}});
 assert.equal(interrupted.status,'needs-settlement');assert.equal(interrupted.elapsedSeconds,3);const pending=Object.values(interrupted.operations).find(o=>o.status==='running');assert.equal(pending.elapsedSeconds,3);
 wall+=2000;let replay=0;
 const unresolved=await runFailurePilot({...f.options,concurrency:1},{...f.ops,now:()=>clock,wallNow:()=>wall,settle:async()=>({closed:false}),trial:async()=>{replay++;}});
 assert.equal(unresolved.status,'needs-settlement');assert.equal(unresolved.elapsedSeconds,5);assert.equal(replay,0);
 const restored=await runFailurePilot({...f.options,concurrency:1},{...f.ops,now:()=>clock,wallNow:()=>wall,settle:async op=>{assert.equal(op.id,pending.id);return {closed:true,result:result(false)};},trial:async input=>{replay++;clock+=1000;return f.ops.trial(input);}});
 assert.equal(restored.status,'complete');assert.equal(restored.promising,true);assert.equal(replay,44);assert(restored.elapsedSeconds>=49);
});
test('closed work without a trustworthy result is never silently replayed',async t=>{
 const f=await fixture(t);const once={...f.options,concurrency:1};
 await runFailurePilot(once,{...f.ops,trial:async()=>{throw Error('Lost process');}});
 let calls=0;const report=await runFailurePilot(once,{...f.ops,settle:async()=>({closed:true}),trial:async()=>{calls++;}});
 assert.equal(report.status,'inconclusive');assert.equal(calls,0);
});
test('elapsed expiry propagates remaining time and prevents renewed work on resume',async t=>{
 const f=await fixture(t);let clock=0,calls=0;
 const report=await runFailurePilot({...f.options,seconds:2,concurrency:1},{...f.ops,now:()=>clock,qualify:async input=>{assert.equal(input.remainingSeconds,2);clock=1500;return {qualified:true,referencePassed:true,negativeFailed:true};},trial:async()=>{calls++;return result(false);}});
 assert.equal(report.status,'inconclusive');assert.match(report.reason,/elapsed/);assert.equal(calls,0);
 const resumed=await runFailurePilot({...f.options,seconds:2,concurrency:1},{...f.ops,now:()=>clock,trial:async()=>{calls++;return result(false);}});assert.equal(calls,0);assert.equal(resumed.status,'inconclusive');
});
test('wire validation rejects missing cases, unknown cases and counterfeit successful outcomes',()=>{
 const t=task('public');assert.throws(()=>validatePilotTrial(result(true,{caseResults:[]}),t),/Successful trial/);
 assert.throws(()=>validatePilotTrial(result(true,{caseResults:[{id:'mechanism',correct:true}]}),t),/coverage/);
 assert.throws(()=>validatePilotTrial(result(false,{caseResults:[{id:'unknown',correct:false}]}),t),/coverage/);
 assert.throws(()=>validatePilotTrial({infrastructure:true,correct:false,reason:'Unknown'}),/correct:null/);
 const bad={baseline:'Current',identities:{model:'x',runtime:'x',memory:'x'},families:[family('f')]};bad.families[0].transfer[1].id='transfer-a';assert.throws(()=>validatePilotManifest(bad),/Unique/);
});

test('late successful callback cannot turn expired work into a valid comparison',async t=>{
 const f=await fixture(t);let clock=0;
 const report=await runFailurePilot({...f.options,trialSeconds:1,concurrency:1},{...f.ops,now:()=>clock,trial:async()=>{clock+=2000;return result(true);}});
 assert.equal(report.status,'inconclusive');assert.equal(report.candidates,null);assert(Object.values(report.operations).some(op=>op.result?.infrastructure===true));
});

test('caller-owned manifest and design objects cannot mutate frozen trial conditions',async t=>{
 const f=await fixture(t),authored=structuredClone(rows);let changed=false;
 const report=await runFailurePilot(f.options,{...f.ops,design:async()=>authored,trial:async input=>{
  if(input.phase==='screen')f.manifest.baseline='Changed external baseline';
  if(input.phase==='development'&&!changed){changed=true;authored[2].text='Changed external candidate';}
  if(input.arm==='baseline')assert.equal(input.guidance,'Current skills');
  if(input.arm==='principle-c')assert.equal(input.guidance,'Short principle');
  return f.ops.trial(input);
 }});
 assert.equal(report.promising,true);assert.equal(report.winner.text,'Short principle');
});

test('a concurrent invocation cannot dispatch work under the live pilot owner',async t=>{
 const f=await fixture(t);let release,entered;const gate=new Promise(resolve=>release=resolve),ready=new Promise(resolve=>entered=resolve);
 const first=runFailurePilot(f.options,{...f.ops,qualify:async()=>{entered();await gate;return {qualified:true,referencePassed:true,negativeFailed:true};}});
 await ready;
 try{const second=await runFailurePilot(f.options,{qualify:()=>{throw Error('Duplicate dispatch');},trial:f.ops.trial,design:f.ops.design});assert.equal(second.status,'needs-settlement');}finally{release();}
 assert.equal((await first).status,'complete');
});
test('an incomplete owner record is retained for settlement rather than overwritten',async t=>{
 const f=await fixture(t);await writeFile(join(f.directory,'pilot.lock'),'{');
 const report=await runFailurePilot(f.options,f.ops);assert.equal(report.status,'needs-settlement');assert.equal(await readFile(join(f.directory,'pilot.lock'),'utf8'),'{');assert.equal(f.calls.length,0);
});

test('qualification work already performed consumes the original pilot allocation',async t=>{
 const f=await fixture(t);let calls=0;
 const report=await runFailurePilot({...f.options,seconds:120,initialElapsedSeconds:120},{qualify:async()=>{calls++;},trial:f.ops.trial,design:f.ops.design});
 assert.equal(report.status,'inconclusive');assert.equal(calls,0);assert(report.elapsedSeconds>=120);
});

test('actor headroom and trusted verification have separate immutable grants',async t=>{
 const f=await fixture(t);let clock=0,calls=0;
 const options={...f.options,seconds:7200,trialSeconds:180,verificationSeconds:120,concurrency:1};
 const report=await runFailurePilot(options,{...f.ops,now:()=>clock,trial:async input=>{
  calls++;assert.equal(input.actorSeconds,180);assert.equal(input.verificationSeconds,120);assert.equal(input.remainingSeconds,300);clock+=299000;return result(true);
 }});
 assert.equal(report.status,'complete');assert.equal(calls,5);
 for(const op of Object.values(report.operations).filter(op=>op.kind==='trial')){assert.equal(op.actorSeconds,180);assert.equal(op.verificationSeconds,120);assert.equal(op.allocatedSeconds,300);assert.equal(op.elapsedSeconds,299);}
 await assert.rejects(runFailurePilot({...options,verificationSeconds:121},f.ops),/Frozen pilot/);
});
test('remaining campaign time cannot silently shorten actor or verification grants',async t=>{
 const f=await fixture(t);let calls=0;
 const report=await runFailurePilot({...f.options,seconds:299,trialSeconds:180,verificationSeconds:120},{...f.ops,trial:async()=>{calls++;return result(false);}});
 assert.equal(report.status,'inconclusive');assert.equal(calls,0);assert.match(report.reason,/full actor and verification allocation/);assert.equal(Object.values(report.operations).filter(op=>op.kind==='trial').length,0);
});

test('denying a new full grant does not cancel other already admitted trials',async t=>{
 const f=await fixture(t);let clock=0,calls=0;
 const report=await runFailurePilot({...f.options,seconds:15,trialSeconds:10,concurrency:2},{...f.ops,now:()=>clock,trial:async input=>{
  calls++;
  if(input.repetition===0){clock=6000;await delay(1);}else{await delay(10);clock=8000;assert.equal(input.signal.aborted,false);}
  return result(false);
 }});
 assert.equal(report.status,'inconclusive');assert.equal(calls,2);
 const trials=Object.values(report.operations).filter(op=>op.kind==='trial');assert.equal(trials.length,2);assert(trials.every(op=>op.status==='complete'&&op.result.correct===false));
});


test('immediate settlement does not charge an interrupted interval twice',async t=>{
 const f=await fixture(t);let clock=0,wall=10000;
 const options={...f.options,concurrency:1};
 const first=await runFailurePilot(options,{...f.ops,now:()=>clock,wallNow:()=>wall,trial:async()=>{clock+=6000;wall+=6000;throw Error('Disconnected');}});
 const pending=Object.values(first.operations).find(op=>op.status==='running');assert.equal(pending.elapsedSeconds,6);
 const resumed=await runFailurePilot(options,{...f.ops,now:()=>clock,wallNow:()=>wall,settle:async()=>({closed:true,result:result(false)})});
 assert.equal(resumed.status,'complete');assert.equal(resumed.operations[pending.id].elapsedSeconds,6);assert.equal(resumed.elapsedSeconds,6);
});
test('operation delimiter in family or task IDs is rejected before any dispatch',async t=>{
 for(const invalidId of ['first--second','first-','-first'])for(const field of ['family','development','transfer']){
  const f=await fixture(t);if(field==='family')f.manifest.families[0].id=invalidId;else if(field==='development')f.manifest.families[0].development.id=invalidId;else f.manifest.families[0].transfer[0].id=invalidId;
  let dispatched=0;await assert.rejects(runFailurePilot(f.options,{qualify:async()=>{dispatched++;},trial:async()=>{dispatched++;},design:async()=>{dispatched++;}}),/Unique/);assert.equal(dispatched,0);
 }
});
test('candidate IDs containing the operation delimiter never reach development trials',async t=>{
 const f=await fixture(t);let laterTrials=0;
 const report=await runFailurePilot(f.options,{...f.ops,design:async()=>[candidate('principle--a'),...rows.slice(1)],trial:async input=>{if(input.phase!=='screen')laterTrials++;return result(false);}});
 assert.equal(report.status,'inconclusive');assert.match(report.reason,/candidate IDs/);assert.equal(laterTrials,0);assert.equal(report.candidates,null);
});

test('candidate boundary hyphens cannot alias composite operation IDs',async t=>{
 for(const id of ['principle-','-principle']){const f=await fixture(t);let later=0;const report=await runFailurePilot(f.options,{...f.ops,design:async()=>[candidate(id),...rows.slice(1)],trial:async input=>{if(input.phase!=='screen')later++;return result(false);}});assert.equal(report.status,'inconclusive');assert.equal(later,0);}
});

test('focused correctness runs each qualified baseline before its treatment',async t=>{
 const f=await fixture(t);
 f.manifest.mode='focused-correctness';
 const qualified=[];
 const report=await runFailurePilot(f.options,{...f.ops,
  qualify:async ({task})=>{
   qualified.push(task.id);
   return {qualified:true,referencePassed:true,negativeFailed:true,calibrationPassed:true};
  },
  trial:async input=>{
   f.calls.push(['trial',input]);
   return result(input.arm==='baseline'?input.repetition!==0: true,
    input.arm==='baseline'&&input.repetition>1?{correct:null,reason:'outside scope'}:{});
  },
 });
 assert.deepEqual(qualified,['development','transfer-a','transfer-b']);
 assert.equal(report.promising,true);
 for(const taskId of qualified){
  const trials=f.calls.filter(([k,c])=>k==='trial'&&c.task.id===taskId).map(([,c])=>c);
  assert.equal(trials.filter(c=>c.arm==='baseline').length,5);
  assert(trials.slice(0,5).every(c=>c.arm==='baseline'));
 }
});

test('focused ceiling and unrelated diagnostics keep the writer idle',async t=>{
 const f=await fixture(t);
 f.manifest.mode='focused-correctness';
 f.manifest.families.push(family('unrelated'));
 const report=await runFailurePilot(f.options,{...f.ops,
  qualify:async ()=>({qualified:true,referencePassed:true,negativeFailed:true,calibrationPassed:true}),
  trial:async input=>input.familyId==='first'?result(input.repetition<4):result(false,{mechanismMatch:false}),
 });
 assert.equal(report.screenDecisions.first,'baseline-ceiling');
 assert.equal(report.screenDecisions.unrelated,'no-exact-recurrence');
 assert.equal(report.candidates,null);
 assert.equal(f.calls.length,0);
});

test('focused writer receives one retry for explicitly actionable review',async t=>{
 const f=await fixture(t);
 f.manifest.mode='focused-correctness';
 const feedback=[];
 const report=await runFailurePilot(f.options,{...f.ops,
  qualify:async ()=>({qualified:true,referencePassed:true,negativeFailed:true,calibrationPassed:true}),
  design:async input=>{
   feedback.push(input.revisionFeedback);
   return input.revisionFeedback?rows:{status:'rejected',reason:'Use only archived source',revisionFeedback:'Remove unsupported count'};
  },
 });
 assert.deepEqual(feedback,[undefined,'Remove unsupported count']);
 assert.equal(report.promising,true);
});

test('focused transfer stops on device failure without running its treatment',async t=>{
 const f=await fixture(t);
 f.manifest.mode='focused-correctness';
 const report=await runFailurePilot(f.options,{...f.ops,
  qualify:async ()=>({qualified:true,referencePassed:true,negativeFailed:true,calibrationPassed:true}),
  trial:async input=>{
   if(input.phase==='transfer-screen'&&input.repetition===1){
    return {correct:null,infrastructure:true,reason:'Device became unavailable'};
   }
   return f.ops.trial(input);
  },
 });
 assert.equal(report.status,'inconclusive');
 assert.equal(report.promising,false);
 assert.equal(report.reason,'Device became unavailable');
 assert.equal(report.operations['transfer-screen--first--transfer-a--baseline--1'].result.infrastructure,true);
 assert(!f.calls.some(([kind,input])=>kind==='trial'&&input.phase==='confirmation'));
});

test('focused calibration failure keeps all model roles idle',async t=>{
 const f=await fixture(t);
 f.manifest.mode='focused-correctness';
 const report=await runFailurePilot(f.options,{...f.ops,
  qualify:async ()=>({qualified:true,referencePassed:true,negativeFailed:true,calibrationPassed:false}),
 });
 assert.equal(report.status,'complete');
 assert.equal(report.promising,false);
 assert.equal(f.calls.length,0);
});

test('focused transfer controls use a bounded qualification batch before actor trials',async t=>{
 const f=await fixture(t);f.manifest.mode='focused-correctness';
 const settled=new Set();
 let active=0,peak=0;
 const report=await runFailurePilot({...f.options,concurrency:2},{...f.ops,
  qualify:async ({task})=>{
   active++;peak=Math.max(peak,active);
   try{await delay(5);settled.add(task.id);
    return {qualified:true,referencePassed:true,negativeFailed:true,calibrationPassed:true};
   }finally{active--;}
  },
  trial:async input=>{
   assert.equal(active,0,'Actors must not compete for qualification custody');
   if(input.task.id!=='development')assert.deepEqual([...settled].sort(),['development','transfer-a','transfer-b']);
   return f.ops.trial(input);
  },
 });
 assert.equal(report.status,'complete');
 assert.equal(report.promising,true);
 assert.equal(peak,2);
 assert.match(report.manifestSha256,/^[0-9a-f]{64}$/);
});

test('qualification exception settles all in-flight slots and never runs transfer actors',async t=>{
 const f=await fixture(t);f.manifest.mode='focused-correctness';
 let failed=false,settled=false;
 const report=await runFailurePilot(f.options,{...f.ops,
  qualify:async ({task})=>{
   if(task.id==='transfer-a'){failed=true;throw Error('Qualification transport failure');}
   if(task.id==='transfer-b'){await delay(5);settled=true;}
   return {qualified:true,referencePassed:true,negativeFailed:true,calibrationPassed:true};
  },
 });
 assert.equal(failed,true);
 assert.equal(settled,true);
 assert.equal(report.promising,false);
 assert.equal(report.status,'needs-settlement');
 assert(!f.calls.some(([kind,input])=>kind==='trial'&&input.phase==='transfer-screen'));
});

for(const wins of [true,false])test('retained replay binds all saved actors and controls even if the first checker '+(wins?'qualified a winner':'found no gain'),async t=>{
 const f=await fixture(t);f.manifest.mode='focused-correctness';
 const task=f.manifest.families[0].development;
 const hash=value=>createHash('sha256').update(value).digest('hex');
 task.checker=join(f.directory,'checker.py');
 await writeFile(task.checker,'original checker');
 task.controls={reference:{files:{'kernel.py':'reference'}},
  negatives:[{files:{'kernel.py':'starter'}}],calibration:{files:{'kernel.py':'calibration'}}};
 const report=await runFailurePilot(f.options,{...f.ops,
  qualify:async ()=>({qualified:true,referencePassed:true,negativeFailed:true,calibrationPassed:true}),
  trial:async input=>{
   const row=await f.ops.trial(input),source=input.arm+' '+input.repetition;
   const cwd=join(input.directory,'retained');
   await mkdir(cwd,{recursive:true});
   await writeFile(join(cwd,'kernel.py'),source);
   const files={'kernel.py':hash(source)};
   await writeFile(join(input.directory,'source-manifest.json'),JSON.stringify({cwd,files}));
   return {...(wins?row:result(false)),sourceSha256:hash(JSON.stringify(files))};
  },
 });
 assert.equal(report.promising,wins);
 const before=await readFile(join(f.directory,'pilot.json'),'utf8');
 const checked=[];
 const candidateId=wins?undefined:'principle-c';
 const replay=await replayRetainedFailureTask({directory:f.directory,taskId:'development',candidateId,checkerSha256:hash('corrected checker')},{
  score:async ({record,files})=>{
   checked.push(record.arm);
   const source=files['kernel.py'].toString();
   return {outcome:source==='starter'?'exact-failure':'pass'};
  }
 });
 assert.equal(replay.freshConfirmationRequired,true);
 assert.equal(checked.filter(a=>a==='baseline').length,5);
 assert.equal(checked.filter(a=>a==='candidate').length,5);
 assert(checked.includes('positive')&&checked.includes('negative')&&checked.includes('calibration'));
 assert.equal(await readFile(join(f.directory,'pilot.json'),'utf8'),before);
 const stored=JSON.parse(await readFile(replay.path,'utf8'));
 assert.equal(stored.records.length,13);
 const first=Object.values(report.operations).find(row=>row.id.startsWith('screen--'));
 await writeFile(join(first.directory,'retained','kernel.py'),'tampered');
 await assert.rejects(replayRetainedFailureTask({directory:f.directory,taskId:'development',candidateId,checkerSha256:hash('another checker')},
  {score:async ()=>assert.fail('No replay after source drift')}),/Source hash mismatch/);
});
