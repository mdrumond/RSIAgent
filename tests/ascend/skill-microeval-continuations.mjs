import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,appendFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {randomUUID,createHash} from 'node:crypto';
import {runMicroeval,check} from '../../cli/skill-microeval.mjs';
import {continueMicroevalSamples} from '../../cli/skill-microeval-continuations.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex'),python=process.env.RSI_TEST_PYTHON??'python3';
async function fixture(t,{correct=false,spent=.05}={}){
 const root=await mkdtemp(join(tmpdir(),'microeval-continuation-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const skills={};for(const arm of ['active','candidate']){const p=join(root,arm);await mkdir(p);await writeFile(join(p,'skill-map.json'),JSON.stringify({implementation:'mapped.md'}));await writeFile(join(p,'mapped.md'),arm+' guidance');skills[arm]=p;}
 const checker=join(root,'checker.py');await writeFile(checker,"import runpy,sys,pathlib\nm=runpy.run_path(str(pathlib.Path(sys.argv[1])/'kernel.py'))\nassert m['answer']()==42\n");
 const originDirectory=join(root,'original'),profile={provider:'fixture',model:'scripted'};
 await runMicroeval({directory:originDirectory,skills,profile,python,runs:1,seconds:1,tasks:[{id:'repair',scopes:['implementation'],prompt:'Repair answer',files:{'kernel.py':`def answer(): return ${correct?42:0}\n`,'case.json':'{"expected":42}'},checker,protectedFiles:{'case.json':'json'}}]},{sample:async args=>{
  const session=join(args.sessionDirectory,'original-session.jsonl');
  await writeFile(session,JSON.stringify({type:'session',version:3,id:randomUUID(),timestamp:new Date().toISOString(),cwd:args.cwd})+'\n'+JSON.stringify({type:'message',id:'first',parentId:null,timestamp:new Date().toISOString(),message:{role:'user',content:'Keep original history',timestamp:Date.now()}})+'\n');return {timedOut:true,session};
 }});
 const report=JSON.parse(await readFile(join(originDirectory,'results.json')));for(const row of report.results)row.elapsedSeconds=spent;
 await writeFile(join(originDirectory,'results.json'),JSON.stringify(report));
 const original=await readFile(join(originDirectory,'results.json')),source=join(originDirectory,'repair/0-active/work/kernel.py'),session=report.results[0].session;
 return {root,original,source,session,skills,profile,originDirectory,directory:join(root,'extended'),sampleKeys:['repair/0/active'],python,seconds:2,verificationSeconds:.2};
}
async function unchanged(f){assert.equal(sha(await readFile(join(f.originDirectory,'results.json'))),sha(f.original));}
const untilAbort=signal=>new Promise(resolve=>{const timer=setTimeout(resolve,3000),done=()=>{clearTimeout(timer);resolve();};if(signal.aborted)return done();signal.addEventListener('abort',done,{once:true});});
test('verification-only recovery preserves original outcome, source and Pi session',async t=>{
 const f=await fixture(t,{correct:true}),before=await readFile(f.session);let calls=0;
 const report=await continueMicroevalSamples(f,{sample:async()=>{calls++;throw Error('must not run');}}),row=report.samples['repair/0/active'];
 assert.equal(calls,0);assert.equal(row.correct,true);assert.equal(row.verificationOnly,true);assert.equal(row.originalElapsedSeconds,.05);
 assert.ok(row.elapsedSeconds>=row.originalElapsedSeconds);assert.deepEqual(await readFile(f.session),before);await unchanged(f);
 assert.equal((await readFile(f.source,'utf8')).trim(),'def answer(): return 42');assert.equal(report.decision,undefined);
});
test('a failed check forks the real Pi history and repairs only the continuation copy',async t=>{
 const f=await fixture(t),before=await readFile(f.session);let calls=0;
 const report=await continueMicroevalSamples(f,{sample:async args=>{
  calls++;assert.notEqual(args.resumeSession,f.session);const lines=(await readFile(args.resumeSession,'utf8')).trim().split('\n').map(JSON.parse);
  assert.equal(lines[0].parentSession,f.session);assert.equal(lines[0].cwd,args.cwd);assert.equal(lines[1].message.content,'Keep original history');
  assert.equal(args.guidance,'## implementation\nactive guidance');assert.ok(args.seconds<f.seconds-f.verificationSeconds);
  await appendFile(args.resumeSession,JSON.stringify({type:'continuation-test',value:'new'})+'\n');await writeFile(join(args.cwd,'kernel.py'),'def answer(): return 42\n');return {session:args.resumeSession};
 }});
 assert.equal(calls,1);assert.equal(report.samples['repair/0/active'].correct,true);assert.deepEqual(await readFile(f.session),before);assert.equal((await readFile(f.source,'utf8')).trim(),'def answer(): return 0');await unchanged(f);
});
test('original elapsed time reduces the actor allowance and final verification stays inside the remainder',async t=>{
 const f=await fixture(t,{spent:.24});f.seconds=.5;f.verificationSeconds=.08;let allocation;
 const report=await continueMicroevalSamples(f,{sample:async args=>{
  allocation=args.seconds;assert.ok(args.sampleSeconds<=.26);await untilAbort(args.signal);await writeFile(join(args.cwd,'kernel.py'),'def answer(): return 42\n');return {timedOut:true,session:args.resumeSession};
 }}),row=report.samples['repair/0/active'];
 assert.ok(allocation>0&&allocation<.18);assert.equal(row.actorTimedOut,true);assert.equal(row.correct,true);assert.ok(row.elapsedSeconds>=.24);assert.ok(row.elapsedSeconds<.6);await unchanged(f);
});
test('controlled interruption resumes the same fork with less time and without repeating the initial check',async t=>{
 const f=await fixture(t),abort=new AbortController();let calls=0,checks=0,forks=0,firstAllocation,session;
 const deps={verify:async args=>{checks++;return check(args.python,args.checker,args.cwd,args.seconds,args.signal);},forkSession:async args=>{
  forks++;const {SessionManager}=await import('@earendil-works/pi-coding-agent');return SessionManager.forkFrom(args.source,args.cwd,args.sessionDirectory).getSessionFile();
 },sample:async args=>{
  calls++;if(calls===1){firstAllocation=args.seconds;session=args.resumeSession;setTimeout(()=>abort.abort(Error('operator checkpoint')),30);await untilAbort(args.signal);return {session};}
  assert.equal(args.resumeSession,session);assert.ok(args.seconds<firstAllocation);await writeFile(join(args.cwd,'kernel.py'),'def answer(): return 42\n');return {session};
 }};
 await assert.rejects(continueMicroevalSamples({...f,signal:abort.signal},deps),/operator checkpoint/);
 const partial=JSON.parse(await readFile(join(f.directory,'results.json')));assert.equal(partial.samples['repair/0/active'].status,'interrupted');assert.equal(partial.runningSample,undefined);
 const report=await continueMicroevalSamples(f,deps);assert.equal(report.samples['repair/0/active'].correct,true);assert.equal(calls,2);assert.equal(forks,1);assert.equal(checks,2);await unchanged(f);
});
test('unknown native executions retain identity and prevent another admission',async t=>{
 for(const phase of ['verify','sample']){
  const f=await fixture(t),handle='remote:fixture:job:original-'+phase;let calls=0;
  const deps={verify:async args=>{if(phase==='verify')throw Object.assign(Error('Unknown target'),{code:'RSI_EXECUTION_UNRESOLVED',handle});return check(args.python,args.checker,args.cwd,args.seconds,args.signal);},sample:async()=>{calls++;return {unresolvedExecution:{handle}};}};
  await assert.rejects(continueMicroevalSamples(f,deps),{code:'RSI_EXECUTION_UNRESOLVED'});
  const report=JSON.parse(await readFile(join(f.directory,'results.json')));assert.equal(report.runningSample.unresolvedExecution.handle,handle);assert.equal(report.samples['repair/0/active'].status,'blocked');
  await assert.rejects(continueMicroevalSamples(f,deps),/unresolved/);assert.equal(calls,phase==='sample'?1:0);await unchanged(f);
 }
});
test('blocked reservations preserve their exact metadata without actor work',async t=>{
 const f=await fixture(t),reservations=[{physicalDevice:5,token:'retained',receipt:'/owned/lease.json',closed:false}];let calls=0;
 await assert.rejects(continueMicroevalSamples(f,{verify:async()=>{throw Object.assign(Error('Device reservations require reconciliation'),{code:'DEVICE_LEASE_BLOCKED',reservations});},sample:async()=>{calls++;}}),{code:'RSI_EXECUTION_UNRESOLVED'});
 const report=JSON.parse(await readFile(join(f.directory,'results.json')));assert.deepEqual(report.runningSample.unresolvedExecution.reservations,reservations);assert.equal(calls,0);await unchanged(f);
});
test('a closed verifier timeout gets a verification-only retry rather than a paid actor',async t=>{
 const f=await fixture(t,{correct:true});let checks=0,calls=0;
 const report=await continueMicroevalSamples(f,{verify:async args=>{checks++;return checks===1?{correct:null,timedOut:true,exitCode:null}:check(args.python,args.checker,args.cwd,args.seconds,args.signal);},sample:async()=>{calls++;}});
 assert.equal(checks,2);assert.equal(calls,0);assert.equal(report.samples['repair/0/active'].correct,true);
});
test('already exhausted allowances and harness-invalid origin trials do not run actors',async t=>{
 const f=await fixture(t,{spent:3});let calls=0;
 const report=await continueMicroevalSamples(f,{verify:async()=>{calls++;}});assert.equal(report.samples['repair/0/active'].status,'elapsed');assert.equal(calls,0);
 const g=await fixture(t),path=join(g.originDirectory,'results.json'),origin=JSON.parse(await readFile(path));origin.results[0].bucket='harness';await writeFile(path,JSON.stringify(origin));
 await assert.rejects(continueMicroevalSamples(g),/non-harness/);
});
test('changed guidance, frozen task, profile or continuation allowance is rejected before execution',async t=>{
 const f=await fixture(t,{correct:true});await continueMicroevalSamples(f);
 await assert.rejects(continueMicroevalSamples({...f,seconds:3}),/Frozen continuation changed/);
 await assert.rejects(continueMicroevalSamples({...f,profile:{provider:'other',model:'scripted'}}),/model differs/);
 await writeFile(join(f.skills.active,'mapped.md'),'different guidance');await assert.rejects(continueMicroevalSamples(f),/guidance changed/);
 const g=await fixture(t),p=join(g.originDirectory,'frozen/tasks.json'),tasks=JSON.parse(await readFile(p));tasks[0].checker='assert True';await writeFile(p,JSON.stringify(tasks));await assert.rejects(continueMicroevalSamples(g),/frozen task changed/);
});
test('a new actor is admitted only when the origin positively records no actor started',async t=>{
 for(const unstarted of [false,true]){
  const f=await fixture(t),p=join(f.originDirectory,'results.json'),origin=JSON.parse(await readFile(p));delete origin.results[0].session;if(unstarted)origin.results[0].actorStarted=false;await writeFile(p,JSON.stringify(origin));let calls=0;
  const report=await continueMicroevalSamples(f,{sample:async args=>{calls++;assert.equal(args.resumeSession,undefined);await writeFile(join(args.cwd,'kernel.py'),'def answer(): return 42\n');return {};}}),row=report.samples['repair/0/active'];
  assert.equal(calls,unstarted?1:0);assert.equal(row.correct,unstarted?true:null);if(unstarted)assert.equal(row.sessionMode,'created');
 }
});
test('other original samples may append while a completed continuation retains its own identity',async t=>{
 const f=await fixture(t,{correct:true});await continueMicroevalSamples(f);const p=join(f.originDirectory,'results.json'),origin=JSON.parse(await readFile(p));origin.results.push({...origin.results[1],taskId:'new-primary-row'});await writeFile(p,JSON.stringify(origin));
 const report=await continueMicroevalSamples(f,{verify:async()=>{throw Error('must not rerun');}});assert.equal(report.samples['repair/0/active'].correct,true);
 await writeFile(join(f.originDirectory,'repair/0-active/work/kernel.py'),'def answer(): return 43\n');await assert.rejects(continueMicroevalSamples(f),/Frozen continuation changed/);
});
test('protected-input mutation is retained as a harness fault without admitting an actor',async t=>{
 const f=await fixture(t);let calls=0;
 const report=await continueMicroevalSamples(f,{verify:async args=>{await writeFile(join(args.cwd,'case.json'),'{"expected":0}');return {correct:false,exitCode:1};},sample:async()=>{calls++;}});
 const row=report.samples['repair/0/active'];assert.equal(row.status,'harness-invalid');assert.equal(row.correct,null);assert.equal(calls,0);assert.equal(await readFile(join(dirname(f.source),'case.json'),'utf8'),'{"expected":42}');await unchanged(f);
});
