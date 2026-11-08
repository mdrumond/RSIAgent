import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,cp,rm,access} from 'node:fs/promises';
import {join,isAbsolute} from 'node:path';
import {tmpdir} from 'node:os';
import * as sdk from '@earendil-works/pi-coding-agent';
import {createEnvironment} from '../../integrations/environments/index.mjs';
import {validateTargetBaseline} from '../../controller/baseline-preflight.mjs';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {findingsFor} from '../../cli/findings.mjs';

// Only transport is a local filesystem fixture: production synchronization and
// supervisor scripts actually execute through Pi's native Bash operations.
async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'execution-verdict-')),source=join(root,'source'),workspace=join(root,'target');
  await mkdir(source);await mkdir(workspace);t.after(()=>rm(root,{recursive:true,force:true}));
  await writeFile(join(source,'answer.py'),'answer=42\n');
  const verifier=join(root,'verify.py');await writeFile(verifier,'assert True\n');
  const project={id:'p',goal:'fixed baseline',source,actor_seconds:60,verifier:{files:[verifier],command:'true',baseline_command:'printf ran > payload'}};
  const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());state.add(project);
  const queue=new RecoveryQueue(state),{attempt}=queue.claim('p',1,{}),attemptPath=join(root,'attempt');await mkdir(attemptPath);
  const jobs=new Map(),native=sdk.createLocalBashOperations(),events=[];let sequence=0;
  const target=path=>isAbsolute(path)?path:join(workspace,path);
  const client=async args=>{
    if(args[0]==='capabilities')return {state:'available'};
    if(args[0]==='preflight')return {state:'completed'};
    if(args[0]==='upload'){
      await mkdir(join(target(args[3]),'..'),{recursive:true});await cp(args[2],target(args[3]),{recursive:true});return {state:'completed'};
    }
    if(args[0]==='download'){await cp(target(args[2]),args[3]);return {state:'completed'};}
    if(args[0]==='run'){
      const handle='remote:fixture:job:'+ ++sequence,job={state:'running'};jobs.set(handle,job);
      await writeFile(args[args.indexOf('--receipt-file')+1],JSON.stringify({handle}));
      job.done=native.exec(await readFile(args[args.indexOf('--file')+1],'utf8'),workspace,{timeout:10,onData:()=>{}})
        .then(result=>Object.assign(job,{state:'completed',exit:result.exitCode}));return {handle,state:'running'};
    }
    if(args[0]==='observe')return jobs.get(args[1]);
    throw Error('Unexpected transport operation '+args[0]);
  };
  t.after(async()=>{await Promise.all([...jobs.values()].map(job=>job.done));});
  const options={mode:'cpl',target:'fixture',workspacePath:workspace,remoteDirectory:'owned',pollMs:1};
  const context={instancePath:attemptPath,client,event:async(type,data)=>{events.push({type,data});state.event(attempt.id,type,data);},exclude:async(_label,fn)=>fn()};
  const environment=await createEnvironment(options,context);await environment.prepare({project,attempt});
  const baseline=()=>validateTargetBaseline({project,state,attempt,attemptPath,remaining:()=>10,checkVerifier:async()=>{},register:async()=>{},
    execute:(command,cwd,opts)=>environment.execute(command,cwd,opts)});
  return {root,source,workspace,project,state,attempt,attemptPath,queue,environment,events,baseline,options,context};
}

test('real frozen-file guard exit125 remains infrastructure across target observation and baseline gate',async t=>{
  const f=await fixture(t),remoteVerifier=join(f.workspace,'owned',f.attempt.id,'tree','verify.py');
  await writeFile(remoteVerifier,'changed trusted file\n');
  await assert.rejects(f.baseline(),{code:'EXECUTION_INFRASTRUCTURE'});
  const infra=f.state.queryEvents(f.attempt.id,{types:['preactor-infrastructure']})[0].data;
  assert.equal(infra.exitCode,125);assert.equal(infra.outcome.reason,'frozen-verifier-before');
  assert.equal(f.state.queryEvents(f.attempt.id,{types:['project-invalid']}).length,0);
  const response=f.events.filter(e=>e.type==='execution-state'&&e.data.outcome).at(-1).data;
  assert.equal(response.outcome.kind,'infrastructure');
  const observer=await createEnvironment(f.options,f.context);
  assert.equal((await observer.observe(response.handle)).outcome.reason,'frozen-verifier-before');
  await assert.rejects(access(join(f.workspace,'owned',f.attempt.id,'tree','source','payload')),{code:'ENOENT'});
  const log=await readFile(join(f.attemptPath,'baseline-preflight.log'),'utf8');assert.ok(log.length>0);
  f.state.finish(f.attempt.id,'failed');f.state.event(f.attempt.id,'failure-settlement',{closed:true});
  const findings=await findingsFor({state:f.state,job:{attempt:f.attempt,kind:'original'},path:f.attemptPath,passed:false,
    config:{actor:{provider:'must-not-run'}}});
  const result=f.queue.recordOriginal(f.attempt.id,{passed:false,...findings},1);
  assert.equal(result.frozen,false);assert.equal(result.next.kind,'retry');assert.ok(result.next.retry_after>Date.now());
  assert.deepEqual(f.queue.ready(2),[]);
  for(const lesson of f.queue.lessons())f.queue.acknowledgeLesson(lesson.id);
  f.state.db.prepare('UPDATE recovery_queue SET retry_after=?').run(Date.now()-1);
  assert.equal(f.queue.ready(2)[0].kind,'retry');
  const retry=f.queue.claim('p',2,{}).attempt;
  f.state.event(retry.id,'preactor-infrastructure',{reason:'device-health-refused'});
  f.state.finish(retry.id,'failed');f.state.event(retry.id,'failure-settlement',{closed:true});
  const next=f.queue.complete(retry.id,{passed:false,...findings},2);
  assert.equal(next.frozen,false);assert.equal(next.failed_rounds,0);assert.equal(next.next.kind,'retry');
  assert.ok(next.next.retry_after>Date.now());
});

test('real completed baseline exit1 remains a verifier verdict',async t=>{
  const f=await fixture(t);f.project.verifier.baseline_command='exit 1';
  await assert.rejects(f.baseline(),{code:'PROJECT_BASELINE_INVALID'});
  assert.equal(f.state.queryEvents(f.attempt.id,{types:['project-invalid']})[0].data.exitCode,1);
  assert.equal(f.state.queryEvents(f.attempt.id,{types:['preactor-infrastructure']}).length,0);
});

test('a verifier changed during the real target command is not a failed baseline verdict',async t=>{
  const f=await fixture(t),remoteVerifier=join(f.workspace,'owned',f.attempt.id,'tree','verify.py');
  f.project.verifier.baseline_command=`printf changed > '${remoteVerifier}'`;
  await assert.rejects(f.baseline(),{code:'EXECUTION_INFRASTRUCTURE'});
  const infra=f.state.queryEvents(f.attempt.id,{types:['preactor-infrastructure']})[0].data;
  assert.equal(infra.outcome.reason,'frozen-verifier-after');
  assert.equal(f.state.queryEvents(f.attempt.id,{types:['project-invalid']}).length,0);
});

test('positive closure remains mandatory before infrastructure retry admission',async t=>{
  const f=await fixture(t);f.state.event(f.attempt.id,'preactor-infrastructure',{reason:'execution-unresolved'});
  f.state.finish(f.attempt.id,'failed');f.state.event(f.attempt.id,'failure-settlement',{closed:false});
  assert.throws(()=>f.queue.enqueue(f.attempt.id,1),/settlement/);
  assert.equal(f.queue.entry('p'),undefined);
});

test('existing recovery queues migrate without changing ordinary investigation admission',async t=>{
  const f=await fixture(t);f.state.db.exec('ALTER TABLE recovery_queue DROP COLUMN retry_after');
  const queue=new RecoveryQueue(f.state);f.state.finish(f.attempt.id,'failed');
  f.state.event(f.attempt.id,'failure-settlement',{closed:true});
  queue.enqueue(f.attempt.id,1);
  assert.equal(queue.claim('p',2,{}).kind,'investigation');
});
