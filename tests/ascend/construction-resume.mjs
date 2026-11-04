import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {mkdtemp,mkdir,writeFile,readFile,rm,readdir,stat} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import {State} from '../../controller/state.mjs';
import {investigate} from '../../cli/curriculum.mjs';
import {setup,rows,until} from './driver_setup.mjs';

const proposal={goal:'SMALL arithmetic observation',question:'Does the numeric implementation return42?',verifier:'verify.py'};
const verifier="import runpy\nfrom pathlib import Path\nassert runpy.run_path(str(Path(__file__).parent/'source/answer.py'))['answer']==42\n";
async function fixture(t,plan) {
  const root=await mkdtemp(join(tmpdir(),'construction-resume-')),source=join(root,'original'),path=join(root,'attempt');
  await mkdir(source);await mkdir(path);await writeFile(join(source,'answer.py'),'answer=0\n');
  const state=new State(join(root,'state.sqlite'));state.add({id:'p',source,goal:'numeric',actor_seconds:30,verifier:{files:[],command:'true'}});
  const attempt=state.admit('p',{}),requests=[];
  const server=createServer(async(req,res)=>{
    let raw='';for await(const part of req)raw+=part;const body=JSON.parse(raw);requests.push(body);
    const index=body.messages.filter(m=>m.role==='tool').length,call=plan[index];
    if(call==='wait')return;
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    if(call){const [name,args]=call;frame({role:'assistant',tool_calls:[{index:0,id:'call-'+index,type:'function',function:{name,arguments:JSON.stringify(args)}}]});frame({},'tool_calls');}
    else{frame({role:'assistant',content:'Done.'});frame({},'stop');}res.end('data: [DONE]\n\n');
  });server.listen(0,'127.0.0.1');await once(server,'listening');
  const key='CONSTRUCTION_FIXTURE_KEY',old=process.env[key];process.env[key]='fixture-only';
  const options={config:{curriculum_seconds:5,actor:{provider:'construction-fixture',model:'scripted',credential_env:key,
    base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}}},
    path,state,base:state.project('p').spec,job:{attempt,origin_attempt:attempt.id,actor_seconds:30},bridge:{extension:()=>{}}};
  t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));state.close();if(old===undefined)delete process.env[key];else process.env[key]=old;await rm(root,{recursive:true,force:true});});
  return {options,requests,events:()=>state.events(attempt.id)};
}
async function interrupt(f) {
  const abort=new AbortController(),pending=investigate({...f.options,signal:abort.signal});
  const rejected=assert.rejects(pending,/external signal/);
  await until(()=>f.requests.length===2);abort.abort();await rejected;
  f.options.job.resume=true;
}

test('constructor resumes the same native session and source; no manifest or actor allocation before submission',{timeout:15000},async t=>{
  const plan=[['write',{path:'source/answer.py',content:'answer=42\n'}],'wait'],f=await fixture(t,plan);
  await interrupt(f);
  const before=f.events().find(e=>e.type==='curriculum-session').data;
  await assert.rejects(stat(join(f.options.path,'project.json')),{code:'ENOENT'});
  const stopped=f.options.state.attempt(f.options.job.attempt.id);assert.equal(stopped.session,null);assert.equal(stopped.elapsed,0);
  plan.splice(1,1,['read',{path:'source/answer.py'}],['write',{path:'verify.py',content:verifier}],['submit_investigation',proposal]);
  const project=await investigate(f.options),sessions=f.events().filter(e=>e.type==='curriculum-session');
  assert.equal(sessions.length,2);assert.equal(sessions[1].data.session,before.session);
  assert.ok(f.requests.at(-1).messages.some(m=>m.role==='tool'&&m.content==='answer=42\n'));
  assert.equal((await readdir(join(f.options.path,'investigation','sessions'))).filter(p=>p.endsWith('.jsonl')).length,1);
  assert.equal(await readFile(join(project.source,'answer.py'),'utf8'),'answer=42\n');
  assert.equal(await readFile(join(f.options.base.source,'answer.py'),'utf8'),'answer=0\n');
  assert.deepEqual(f.options.state.attempt(stopped.id),stopped);
});

test('resumed constructor uses original elapsed allocation even when config increases; exhausted resume never calls provider',{timeout:15000},async t=>{
  const f=await fixture(t,[['write',{path:'source/answer.py',content:'answer=42\n'}],'wait']);
  f.options.config.curriculum_seconds=.6;await interrupt(f);
  const first=f.events().find(e=>e.type==='curriculum-interrupted').data;
  assert.ok(first.elapsed_seconds>0);f.options.config.curriculum_seconds=480;
  await assert.rejects(investigate(f.options),/elapsed allocation exhausted/);
  const last=f.events().filter(e=>e.type==='curriculum-interrupted').at(-1).data;
  assert.equal(last.allocation_seconds,.6);assert.ok(last.elapsed_seconds>=.6);assert.equal(last.session,first.session);
  const requests=f.requests.length;await assert.rejects(investigate(f.options),/elapsed allocation exhausted/);
  assert.equal(f.requests.length,requests);assert.equal(f.options.state.attempt(f.options.job.attempt.id).elapsed,0);
});

for(const cause of ['elapsed','external'])test(`resumed constructor ${cause} cancellation interrupts readiness before provider work`,{timeout:15000},async t=>{
  const f=await fixture(t,[['write',{path:'source/answer.py',content:'answer=42\n'}],'wait']);
  f.options.config.curriculum_seconds=1.6;await interrupt(f);
  const first=f.events().find(e=>e.type==='curriculum-interrupted').data;
  const requestCount=f.requests.length,eventCount=f.events().length,abort=new AbortController();
  let ready=false,setupSignal,shutdown=0,binding;
  const setup=new Promise(ok=>binding=ok);
  f.options.bridge={extension:pi=>pi.on('session_shutdown',()=>{shutdown++;}),
    prepareSession:async(session,{signal})=>{
      setupSignal=signal;await session.bindExtensions({});binding();
      await delay(4000,undefined,{signal});ready=true;
    }};
  f.options.config.curriculum_seconds=480; // A resumed allocation remains the original1.6s.
  const started=performance.now();
  const pending=investigate({...f.options,signal:abort.signal});
  const rejected=assert.rejects(pending,cause==='elapsed'?/elapsed allocation exhausted/:/external signal/);
  await setup;if(cause==='external')abort.abort();await rejected;
  assert.equal(setupSignal.aborted,true);assert.equal(ready,false);assert.equal(shutdown,1);
  assert.ok(performance.now()-started<3000);assert.equal(f.requests.length,requestCount);
  const last=f.events().filter(e=>e.type==='curriculum-interrupted').at(-1).data;
  assert.equal(last.allocation_seconds,1.6);assert.equal(last.session,first.session);
  assert.equal(last.reason,cause==='elapsed'?'elapsed-allocation':'external-signal');
  assert.ok(last.elapsed_seconds>=first.elapsed_seconds);
  if(cause==='elapsed'){
    assert.ok(last.elapsed_seconds>=1.6);
    const checkpoints=f.events().slice(eventCount).filter(e=>e.type==='curriculum-checkpoint');
    assert.ok(checkpoints.some(e=>e.data.elapsed_seconds<last.elapsed_seconds && e.data.elapsed_seconds>=first.elapsed_seconds+1));
    await assert.rejects(investigate(f.options),/elapsed allocation exhausted/);
    assert.equal(f.requests.length,requestCount);
  }
  assert.equal(f.options.state.attempt(f.options.job.attempt.id).elapsed,0);
  assert.equal(f.options.state.attempt(f.options.job.attempt.id).session,null);
  assert.equal(await readFile(join(f.options.path,'investigation/source/answer.py'),'utf8'),'answer=42\n');
});

test('validated proposal survives interruption before manifest; resume rechecks artifacts without another provider call',{timeout:15000},async t=>{
  const plan=[['write',{path:'verify.py',content:verifier}],['submit_investigation',proposal],'wait'],f=await fixture(t,plan);
  const abort=new AbortController(),pending=investigate({...f.options,signal:abort.signal});
  await until(()=>f.requests.length===3);abort.abort();await pending;
  f.options.job.resume=true;const count=f.requests.length;
  const resumed=await investigate(f.options);assert.equal(f.requests.length,count);assert.equal(resumed.goal,proposal.goal+'\nInvestigate: '+proposal.question);
  await rm(resumed.verifier.files[0]);await assert.rejects(investigate(f.options),/Artifact is missing/);assert.equal(f.requests.length,count);
});

test('public CLI stop before investigation manifest then resume completes same child canonical workflow',{timeout:120000},async t=>{
  const f=await setup(t,'recover',['recover'],{install:false});await f.run('wave');
  const original=rows(f.root,'SELECT * FROM attempts')[0],frozen=rows(f.root,'SELECT spec FROM projects')[0].spec;
  let resume=false;const wires=[];
  const server=createServer(async(req,res)=>{
    let raw='';for await(const part of req)raw+=part;const body=JSON.parse(raw);
    if(!body.tools.some(t=>t.function.name==='submit_investigation')){
      const reply=await fetch(f.server.url+'/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:raw});
      res.writeHead(reply.status,{'Content-Type':'text/event-stream'});res.end(await reply.text());return;
    }
    wires.push(body);const n=body.messages.filter(m=>m.role==='tool').length;
    if(n===1&&!resume)return;
    const calls=[['write',{path:'source/answer.py',content:'def answer(): return 41\n'}],['read',{path:'source/answer.py'}],
      ['write',{path:'verify.py',content:"import runpy\nfrom pathlib import Path\nassert runpy.run_path(str(Path(__file__).parent/'source/answer.py'))['answer']()==42\n"}],['submit_investigation',proposal]];
    const call=calls[n];res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'constructor',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    if(call){frame({role:'assistant',tool_calls:[{index:0,id:'construction-'+n,type:'function',function:{name:call[0],arguments:JSON.stringify(call[1])}}]});frame({},'tool_calls');}
    else{frame({role:'assistant',content:'Submitted.'});frame({},'stop');}res.end('data: [DONE]\n\n');
  });server.listen(0,'127.0.0.1');await once(server,'listening');
  t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));});
  const settings=join(f.root,'constructor-settings.json');
  const config=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));
  await writeFile(settings,JSON.stringify({curriculum_actor:{...config.actor,provider:'constructor-fixture',base_url:`http://127.0.0.1:${server.address().port}/v1`}}));
  await f.run('config','--config',settings);
  const wave=f.start('wave');await until(()=>wires.length===2);await f.run('stop');await wave.output;
  const child=rows(f.root,'SELECT * FROM attempts').find(a=>a.id!==original.id),path=join(f.root,'attempts',child.id);
  assert.equal(child.outcome,null);assert.equal(child.elapsed,0);assert.equal(child.session,null);await assert.rejects(stat(join(path,'project.json')),{code:'ENOENT'});
  const sessions=()=>rows(f.root,`SELECT data FROM events WHERE attempt='${child.id}' AND type='curriculum-session'`).map(e=>JSON.parse(e.data)).filter(e=>e.role==='investigation');
  const session=sessions()[0].session;resume=true;await f.run('resume');
  const after=rows(f.root,'SELECT * FROM attempts');assert.equal(after.length,2);assert.equal(after.find(a=>a.id===child.id).outcome,'passed');
  assert.deepEqual(after.find(a=>a.id===original.id),original);assert.equal(rows(f.root,'SELECT spec FROM projects')[0].spec,frozen);
  assert.equal(sessions().length,2);assert.ok(sessions().every(s=>s.session===session));
  assert.ok(wires.at(-1).messages.some(m=>m.role==='tool'&&m.content.includes('return 41')));
  const types=rows(f.root,`SELECT type FROM events WHERE attempt='${child.id}'`).map(e=>e.type);
  assert.equal(types.filter(t=>t==='accepted').length,1);assert.equal(types.filter(t=>t==='stage-completed').length,8);
  f.pass();
});
