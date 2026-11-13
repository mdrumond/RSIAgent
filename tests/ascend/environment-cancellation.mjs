import test from 'node:test';
import fs from 'node:fs/promises';
import {syncBuiltinESMExports} from 'node:module';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import {createEnvironment} from '../../integrations/environments/index.mjs';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {runAttempt} from '../../controller/pi.mjs';
import {State} from '../../controller/state.mjs';

async function fixture(t,handler) {
  const root=await mkdtemp(join(tmpdir(),'environment-abort-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const source=join(root,'source');await mkdir(source);await writeFile(join(source,'answer.py'),'answer=42\n');
  const verifier=join(root,'verify.py');await writeFile(verifier,'assert True\n');
  const abort=new AbortController(),calls=[],events=[];
  const client=async(args,{signal}={})=>{
    calls.push({args,signal});
    const result=await handler?.(args,signal,{abort,root,events});
    if(result)return result;
    if(args[0]==='capabilities')return {state:'available'};
    if(args[0]==='observe')return {state:'completed',exit:0};
    if(args[0]==='download')await writeFile(args[3],'{}');
    return {state:'completed'};
  };
  const options={mode:'cpl',target:'fixture',runtime:'cpu',workspacePath:'/fixture',remoteDirectory:'owned',pollMs:1};
  const context={instancePath:root,signal:abort.signal,client,event:async(type,data)=>events.push({type,data}),exclude:async(_,fn)=>fn()};
  const environment=await createEnvironment(options,context);
  const project={id:'p',goal:'numeric',source,actor_seconds:20,verifier:{files:[verifier],command:'python ../verify.py'}};
  return {root,source,abort,calls,events,environment,project,options,context,
    prepare:()=>environment.prepare({project,attempt:{id:'a'}})};
}
for(const phase of ['preflight','upload'])test(`body cancellation reaches delayed native ${phase} before workload dispatch`,async t=>{
  const f=await fixture(t,async(args,signal,{abort})=>{
    if(args[0]===phase){assert.equal(signal,abort.signal);queueMicrotask(()=>abort.abort());await delay(10000,null,{signal});}
  });
  await assert.rejects(f.prepare());
  assert.ok(f.calls.some(c=>c.args[0]===phase));
  assert.equal(f.calls.some(c=>c.args[0]==='run'),false);
  if(phase==='upload')assert.ok(f.events.some(e=>e.type==='execution-unresolved'&&!e.data.handle));
});
for(const closure of ['completed','unknown'])test(`aborted body recovers run receipt and uses independent cleanup; ${closure}`,async t=>{
  let sync=0,workload=0;
  const f=await fixture(t,async(args,signal,{abort})=>{
    if(args[0]==='run') {
      const isSync=args[args.indexOf('--file')+1].endsWith('sync.sh');
      const handle=`remote:fixture:job:${isSync?'sync':'work'}`;
      await writeFile(args[args.indexOf('--receipt-file')+1],JSON.stringify({handle}));
      if(isSync){sync++;return {state:'queued',handle};}
      workload++;abort.abort();signal.throwIfAborted();
    }
    if(['cancel','observe'].includes(args[0])&&args[1]==='remote:fixture:job:work') {
      assert.notEqual(signal,abort.signal);assert.equal(signal.aborted,false);
      if(args[0]==='cancel')return {state:'cancelled'};
      return {state:closure,exit:0};
    }
  });
  await f.prepare();const result=await f.environment.execute('python answer.py',f.source,{signal:f.abort.signal});
  assert.equal(result.settled,closure==='completed');assert.equal(workload,1);assert.equal(sync,1);
  const cleanup=f.calls.filter(c=>c.args[1]==='remote:fixture:job:work');
  assert.deepEqual(cleanup.map(c=>c.args[0]),closure==='completed'?['cancel','observe','logs']:['cancel','observe']);
  assert.equal(cleanup[0].signal,cleanup[1].signal);
  assert.ok(f.events.some(e=>e.type==='execution-submitted'&&e.data.handle==='remote:fixture:job:work'));
  if(closure==='unknown') {
    const unresolved=await f.environment.reconcileOperation({operation:'unrecorded',events:f.events});
    assert.equal(unresolved.closed,false);
  }
});
test('aborted upload with retained receipt is cancelled without replay',async t=>{
  const handle='remote:fixture:transfer:upload';let uploads=0;
  const f=await fixture(t,async(args,signal,{abort})=>{
    if(args[0]==='upload'){
      uploads++;await writeFile(args[args.indexOf('--receipt-file')+1],JSON.stringify({handle}));
      abort.abort();signal.throwIfAborted();
    }
    if(args[0]==='cancel'){assert.equal(signal.aborted,false);return {state:'cancelled'};}
    if(args[0]==='observe')return {state:'cancelled'};
  });
  await assert.rejects(f.prepare());assert.equal(uploads,1);
  assert.deepEqual(f.calls.filter(c=>c.args[1]===handle).map(c=>c.args[0]),['cancel','observe']);
});
test('runAttempt passes the actual caller signal into its environment factory',async t=>{
  const f=await fixture(t),state=new State(join(f.root,'state.sqlite'));t.after(()=>state.close());
  state.add(f.project);const attempt=state.admit('p',{});const boundary=new Error('factory boundary');
  await assert.rejects(runAttempt({state,attempt,project:f.project,attemptPath:join(f.root,'attempt'),signal:f.abort.signal,
    environmentFactory:async context=>{assert.equal(context.signal,f.abort.signal);throw boundary;}}),error=>error===boundary);
  assert.equal(state.attempt(attempt.id).outcome,null);
});
test('public stop aborts pending scheduler discovery and releases owner without admission',{timeout:15000},async t=>{
  const f=await fixture(t),module=join(f.root,'environment.mjs'),marker=join(f.root,'discovering');
  await writeFile(module,`
import {writeFile} from 'node:fs/promises';
import {setTimeout as delay} from 'node:timers/promises';
export const createEnvironment=async(_options,context)=>({resources:async()=>{
  await writeFile(${JSON.stringify(marker)},'started');
  await delay(10000,null,{signal:context.signal});
  throw Error('Discovery was not cancelled');
}});
`);
  const config={format:1,root:f.root,mode:'phase1',projects:[f.project],environment:{module,options:{}},skills:{activeVersion:'baseline'}};
  await writeFile(join(f.root,'instance.json'),JSON.stringify(config));
  const cli=new URL('../../cli/main.mjs',import.meta.url).pathname,exec=promisify(execFile);
  const wave=exec(process.execPath,[cli,'wave','--instance',f.root],{timeout:12000});
  const deadline=performance.now()+5000;
  for(;;){try{await readFile(marker);break;}catch(error){if(error.code!=='ENOENT')throw error;}
    assert.ok(performance.now()<deadline);await delay(10);}
  await exec(process.execPath,[cli,'stop','--instance',f.root]);
  assert.deepEqual(JSON.parse((await wave).stdout.trim()).results,[]);
  const state=new State(join(f.root,'state.sqlite'));t.after(()=>state.close());
  assert.equal(JSON.parse(state.db.prepare("SELECT value FROM control WHERE key='owner'").get().value),null);
  assert.equal(state.db.prepare('SELECT count(*) AS n FROM attempts').get().n,0);
  assert.equal(state.events(null).some(e=>e.type==='execution-submitting'),false);
});
test('cancellation before sync dispatch clears only positively undispatched cache and fresh attempt continues',async t=>{
  let sync=0,commands=0;
  const f=await fixture(t,async(args)=>{
    if(args[0]==='run'){
      const setup=args[args.indexOf('--file')+1].endsWith('sync.sh');
      if(setup)sync++;else commands++;
      const handle=`remote:fixture:job:${setup?'sync':'command'}`;
      await writeFile(args[args.indexOf('--receipt-file')+1],JSON.stringify({handle}));
      return {state:'queued',handle};
    }
  });
  await f.prepare();const cache=join(f.root,'.executions','source-staging.json'),rename=fs.rename;
  const mocked=t.mock.method(fs,'rename',async(from,to)=>{
    await rename(from,to);
    if(to===cache&&!JSON.parse(await readFile(cache,'utf8')).confirmed)f.abort.abort();
  });syncBuiltinESMExports();
  try{await assert.rejects(f.environment.execute('python answer.py',f.source),error=>error.code==='CPL_NOT_ADMITTED');}
  finally{mocked.mock.restore();syncBuiltinESMExports();}
  assert.equal(sync,0);assert.equal(commands,0);
  assert.equal(f.events.some(e=>e.type==='execution-submitting'&&e.data.kind==='run'),false);
  await assert.rejects(readFile(cache),{code:'ENOENT'});
  const fresh=await createEnvironment(f.options,{...f.context,signal:new AbortController().signal});
  await fresh.prepare({project:f.project,attempt:{id:'a'}});
  assert.equal((await fresh.execute('python answer.py',f.source)).settled,true);
  assert.equal(sync,1);assert.equal(commands,1);
});
for(const state of ['completed','unknown'])test(`initial retained download timeout is observed once; observed ${state}`,async t=>{
  const handle='remote:fixture:transfer:download';
  const f=await fixture(t,async(args)=>{
    if(args[0]==='download')return {state:'unknown',handle};
    if(args[0]==='observe')return {state,handle};
    if(args[0]==='fetch'){await writeFile(args[2],'retained output');return {state:'completed'};}
  });
  if(state==='completed'){
    assert.equal((await f.environment.browse('report.log')).state,'completed');
    const fetch=f.calls.find(c=>c.args[0]==='fetch');assert.equal(await readFile(fetch.args[2],'utf8'),'retained output');
  }else{await assert.rejects(f.environment.browse('report.log'),/Transfer unresolved/);assert.equal(f.calls.some(c=>c.args[0]==='fetch'),false);}
  assert.equal(f.calls.filter(c=>c.args[0]==='download').length,1);
  assert.equal(f.calls.filter(c=>c.args[0]==='observe'&&c.args[1]===handle).length,1);
});
