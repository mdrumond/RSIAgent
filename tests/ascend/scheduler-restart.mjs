import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
import {once} from 'node:events';
import {spawn,execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {createHash} from 'node:crypto';
import {setup,rows,until} from './driver_setup.mjs';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue,recordOutcome} from '../../controller/recovery.mjs';
import {processIdentity,ownerStopped} from '../../controller/lifecycle.mjs';

async function configure(f,patch){const file=join(f.root,'restart-config.json');
  await writeFile(file,JSON.stringify(patch));await f.run('config','--config',file);}
async function deadOwner(t){
  const child=spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{stdio:'ignore'}),exit=once(child,'exit');
  await once(child,'spawn');const owner=await processIdentity(child.pid);
  t.after(async()=>{if(child.exitCode===null&&child.signalCode===null)child.kill('SIGKILL');await exit;});
  child.kill('SIGTERM');await exit;assert.equal(await ownerStopped(owner),true);return owner;
}
async function seededLease(f,owner,{terminal=true,unknown=false,held=false,recorded=true}={}){
  const state=new State(join(f.root,'state.sqlite'));
  try{
    const queue=new RecoveryQueue(state),job=queue.claim('holder',0,owner,'local'),id=job.attempt.id;
    const path=join(f.root,'attempts',id);await mkdir(path,{recursive:true});
    await writeFile(join(path,'job.json'),JSON.stringify({...job,resource:{id:'local'},options:{},mode:'phase1',skillVersion:'baseline'}));
    state.checkpoint(id,2,'retained-session');
    if(unknown)state.event(id,'execution-handle',{handle:'local:exec-missing'});
    if(held)state.event(id,'execution-starting',{operation:'missing-receipt',startElapsed:2});
    if(terminal){
      const log=join(path,'evidence.log');await writeFile(log,'Seeded settled outcome for restart admission fixture.\n');
      const sha256=createHash('sha256').update(await readFile(log)).digest('hex');
      const findings={findings:'Retained outcome',useful:false,evidence:[{path:log,sha256}]};
      await writeFile(join(path,'findings.json'),JSON.stringify(findings));
      state.finish(id,'failed');
      if(recorded&&!unknown&&!held){recordOutcome(state,{attemptId:id,wave:0,passed:false,...findings});
        for(const lesson of queue.lessons())queue.acknowledgeLesson(lesson.id);}
    }
    return state.attempt(id);
  }finally{state.close();}
}

test('public CLI scheduler loss leaves worker running; restart reclaims only its settled lease without replay',
  {timeout:120000},async t=>{
  let f,release;
  t.after(async()=>{
    if(!f)return;
    if(release)await writeFile(release,'release');
    await f.run('stop');
    await until(async()=>{for(const row of rows(f.root,'SELECT owner FROM attempts'))if(!await ownerStopped(row.owner))return false;return true;},15000);
  });
  f=await setup(t,'interrupt',['first','unrelated'],{install:false});
  release=join(f.root,'release');
  const entered=join(f.root,'entered'),module=join(f.root,'machinery','restart-environment.mjs');
  await writeFile(module,`import {createEnvironment as base} from './integrations/environments/index.mjs';
export async function createEnvironment(options,context){
 const env=await base(options,context);
 return {...env,execute:async(command,cwd,opts)=>env.execute(command.includes('holding execution')?
 ${JSON.stringify(`python -c "from pathlib import Path; import time; Path('${entered}').touch();\nwhile not Path('${release}').exists(): time.sleep(.01)"`)}:command,cwd,opts)};
}\n`);
  await configure(f,{curriculum:{phase1:['first']},environment:{module:'machinery/restart-environment.mjs',options:{mode:'local'}}});
  const wave=f.start('wave'),killed=assert.rejects(wave.output);
  await until(async()=>{try{await readFile(entered);return true;}catch(error){if(error.code!=='ENOENT')throw error;}});
  const [original]=rows(f.root,'SELECT * FROM attempts');
  wave.child.kill('SIGKILL');await killed;
  assert.equal(await ownerStopped(original.owner),false);
  await writeFile(release,'release');
  await until(async()=>{
    const row=rows(f.root,'SELECT * FROM attempts')[0];
    return row.outcome==='passed'&&await ownerStopped(row.owner)&&
      rows(f.root,"SELECT 1 FROM events WHERE type='postprocess-completed'").length;
  },30000);
  const settled=rows(f.root,'SELECT * FROM attempts')[0],session=await readFile(settled.session);
  assert.equal(rows(f.root,'SELECT * FROM resource_leases').length,1);
  const before=f.server.requests.length;
  await configure(f,{curriculum:{phase1:['first','unrelated']}});
  await f.run('wave');
  const attempts=rows(f.root,'SELECT * FROM attempts');assert.equal(attempts.length,2);
  assert.deepEqual(attempts.find(a=>a.id===settled.id),settled);
  assert.equal(attempts.find(a=>a.project==='unrelated').outcome,'passed');
  assert.deepEqual(await readFile(settled.session),session);
  assert.equal(rows(f.root,'SELECT * FROM resource_leases').length,0);
  assert.equal(rows(f.root,`SELECT count(*) AS n FROM events WHERE attempt='${settled.id}' AND type='resource-lease-reclaimed'`)[0].n,1);
  assert.ok(f.server.requests.slice(before).every(r=>!r.messages.some(m=>m.role==='user'&&typeof m.content==='string'&&m.content.startsWith('Goal: first\n'))));
  f.pass();
});

for(const scenario of ['active','live-owner','unknown-native','held-receipt'])test(`public wave retains ${scenario} lease`,
  {timeout:60000},async t=>{
  const f=await setup(t,'interrupt',['holder','unrelated'],{install:false});
  await configure(f,{curriculum:{phase1:['unrelated']}});
  const owner=scenario==='live-owner'?await processIdentity():await deadOwner(t);
  const before=await seededLease(f,owner,{terminal:scenario!=='active',unknown:scenario==='unknown-native',held:scenario==='held-receipt'});
  await f.run('wave');
  assert.equal(rows(f.root,'SELECT attempt FROM resource_leases')[0].attempt,before.id);
  assert.deepEqual(rows(f.root,`SELECT * FROM attempts WHERE id='${before.id}'`)[0],before);
  assert.equal(rows(f.root,'SELECT * FROM attempts').length,1);
  assert.equal(f.server.requests.length,0);
  f.pass();
});

test('pending finalization retains its lease until worker exit before concurrent unrelated admission',
  {timeout:120000},async t=>{
  let f,release,wave;
  t.after(async()=>{if(!f)return;if(release)await writeFile(release,'release');
    await f.run('stop');if(wave)await wave.output;
  });
  f=await setup(t,'phase2',['holder','unrelated'],{install:false});
  const original=await seededLease(f,await deadOwner(t),{recorded:false});
  const config=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));
  const python=config.python,entered=join(f.root,'finalizing.pid'),wrapper=join(f.root,'finalizing-python');
  release=join(f.root,'release-finalization');
  const {stdout}=await promisify(execFile)(python,['-c','import sys;print(sys.executable)']);
  await writeFile(wrapper,`#!${stdout.trim()}
import os,sys,time
from pathlib import Path
if sys.argv[1:3]==['-m','ascend_tools.kdb_client'] and '--attempt' in sys.argv and sys.argv[sys.argv.index('--attempt')+1]==${JSON.stringify(original.id)}:
 Path(${JSON.stringify(entered)}).write_text(str(os.getpid()))
 while not Path(${JSON.stringify(release)}).exists(): time.sleep(.01)
os.execv(${JSON.stringify(stdout.trim())},[${JSON.stringify(stdout.trim())},*sys.argv[1:]])
`,{mode:0o700});
  await configure(f,{python:wrapper,concurrency:2,curriculum:{phase1:['unrelated']}});
  wave=f.start('wave');void wave.output.catch(()=>{});
  await until(async()=>{try{await readFile(entered);return true;}catch(error){if(error.code!=='ENOENT')throw error;}});
  assert.equal(rows(f.root,'SELECT attempt FROM resource_leases')[0].attempt,original.id);
  assert.equal(rows(f.root,'SELECT * FROM attempts').length,1);
  assert.equal(rows(f.root,'SELECT recorded FROM recovery_lessons')[0].recorded,0);
  const finalizer=rows(f.root,'SELECT * FROM attempts')[0];assert.equal(await ownerStopped(finalizer.owner),false);
  await writeFile(release,'release');await wave.output;
  const attempts=rows(f.root,'SELECT * FROM attempts');assert.equal(attempts.length,2);
  assert.equal(attempts.find(a=>a.project==='unrelated').outcome,'passed');
  assert.equal(await ownerStopped(finalizer.owner),true);
  assert.equal(rows(f.root,'SELECT * FROM resource_leases').length,0);
  assert.equal(rows(f.root,`SELECT count(*) AS n FROM events WHERE attempt='${original.id}' AND type='resource-lease-reclaimed'`)[0].n,0);
  assert.ok(rows(f.root,'SELECT recorded FROM recovery_lessons').every(row=>row.recorded===1));
  f.pass();
});

test('public STOP aborts pending outbox delivery; later resume ACKs once without actor replay',
  {timeout:120000},async t=>{
  const f=await setup(t,'phase2',['unrelated'],{install:false});await f.run('wave');
  const settled=rows(f.root,'SELECT * FROM attempts')[0],requestCount=f.server.requests.length;
  const state=new State(join(f.root,'state.sqlite'));
  const lesson=state.db.prepare('SELECT id FROM recovery_lessons').get();
  state.db.prepare('UPDATE recovery_lessons SET recorded=0 WHERE id=?').run(lesson.id);state.close();
  const current=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));
  const python=current.python,identity=join(f.root,'waiting-client.pid'),wrapper=join(f.root,'pending-python');
  const {stdout}=await promisify(execFile)(python,['-c','import sys;print(sys.executable)']);
  await writeFile(wrapper,`#!${stdout.trim()}
import os,sys,time
from pathlib import Path
if sys.argv[1:3]==['-m','ascend_tools.kdb_client']:
 Path(${JSON.stringify(identity)}).write_text(str(os.getpid()))
 time.sleep(30)
else:
 os.execv(${JSON.stringify(stdout.trim())},[${JSON.stringify(stdout.trim())},*sys.argv[1:]])
`,{mode:0o700});
  await configure(f,{python:wrapper});
  const wave=f.start('wave');
  await until(async()=>{try{return Number(await readFile(identity,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}});
  const owner=await processIdentity(Number(await readFile(identity,'utf8')));
  t.after(async()=>{if(!await ownerStopped(owner))process.kill(owner.pid,'SIGKILL');});
  const stopped=performance.now();await f.run('stop');
  await Promise.race([wave.output,new Promise((_,reject)=>{const timer=setTimeout(()=>reject(Error('STOP did not cancel pending outbox')),5000);timer.unref();})]);
  assert.ok(performance.now()-stopped<5000);assert.equal(await ownerStopped(owner),true);
  assert.equal(rows(f.root,'SELECT recorded FROM recovery_lessons')[0].recorded,0);
  await configure(f,{python});await f.run('resume');await f.run('resume');
  assert.equal(rows(f.root,'SELECT recorded FROM recovery_lessons')[0].recorded,1);
  assert.deepEqual(rows(f.root,'SELECT * FROM attempts'),[settled]);assert.equal(f.server.requests.length,requestCount);
  f.pass();
});
