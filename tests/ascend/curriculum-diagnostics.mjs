import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {mkdtemp,mkdir,writeFile,readFile,rm,stat,readdir} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {investigate,summarize} from '../../cli/curriculum.mjs';
import {findingsFor} from '../../cli/findings.mjs';
import {State} from '../../controller/state.mjs';
import {AgentSession} from '@earendil-works/pi-coding-agent';

async function fixture(t,plan) {
  const root=await mkdtemp(join(tmpdir(),'investigation-admission-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const path=join(root,'attempt'),source=join(root,'original');
  await mkdir(path);await mkdir(source);
  await writeFile(join(source,'answer.py'),'answer=42\n');
  const received=[],prompts=[];
  let waiting;const waitStarted=new Promise(ok=>waiting=ok);
  const server=createServer(async(req,res)=>{
    let raw='';for await(const chunk of req)raw+=chunk;
    const body=JSON.parse(raw),results=body.messages.filter(m=>m.role==='tool');
    received.push(results);
    prompts.push(body.messages.filter(m=>m.role==='user').map(m=>typeof m.content==='string'?m.content:m.content.map(b=>b.text??'').join('\n')).join('\n'));
    const call=plan[results.length];
    if(call==='wait'){waiting();await new Promise(ok=>res.once('close',ok));return;}
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'admission',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    if(call){const [name,args]=call;frame({role:'assistant',tool_calls:[{index:0,id:'call-'+results.length,type:'function',function:{name,arguments:JSON.stringify(args)}}]});frame({},'tool_calls');}
    else{frame({role:'assistant',content:'Done.'});frame({},'stop');}
    res.end('data: [DONE]\n\n');
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');
  t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));});
  const key='ADMISSION_FIXTURE_KEY',previous=process.env[key];process.env[key]='fixture-only';
  t.after(()=>{if(previous===undefined)delete process.env[key];else process.env[key]=previous;});
  const config={root,curriculum_seconds:15,actor:{provider:'admission-fixture',model:'scripted',credential_env:key,
    base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}}};
  const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  state.add({id:'p',source,goal:'numeric',actor_seconds:120,verifier:{files:[],command:'true'}});
  const attempt=state.admit('p',{}),bridge={extension:()=>{}};
  const options={config,path,state,bridge,job:{origin_attempt:attempt.id,attempt,actor_seconds:120},base:{source,goal:'numeric'}};
  config.curriculum_seconds=1;
  return {root,path,source,options,get events(){return state.events(attempt.id);},prompts,waitStarted,results:()=>received.at(-1)};

}

const proposal={goal:'Check numeric fact',question:'Is answer42?',verifier:'verify.py'};
const acceptedPlan=[['write',{path:'verify.py',content:'print(42)\n'}],['submit_investigation',proposal],'wait'];

test('elapsed abort retains real session and summary reads it when actor session is absent',{timeout:15000},async t=>{
  const plan=['wait'],f=await fixture(t,plan);
  await assert.rejects(investigate(f.options),/elapsed allocation exhausted/);
  const stopped=f.events.find(e=>e.type==='curriculum-interrupted').data;
  assert.equal(stopped.reason,'elapsed-allocation');assert.equal(stopped.accepted,false);
  assert.ok(stopped.elapsed_seconds>=stopped.allocation_seconds);
  const session=f.events.find(e=>e.type==='curriculum-session').data;
  assert.equal(session.role,'investigation');assert.equal(stopped.session,session.session);
  const rows=(await readFile(session.session,'utf8')).split('\n').filter(Boolean).map(JSON.parse);
  assert.ok(rows.some(e=>e.message?.stopReason==='aborted'));
  plan.splice(0,plan.length,['read',{path:session.session}],['read',{path:'diagnostics.json'}],
    ['record_findings',{findings:'The curriculum elapsed allocation expired without a proposal; no actor ran.',useful:false,paths:['diagnostics.json']}]);
  const result=await findingsFor({...f.options,passed:false});
  const wire=JSON.parse(f.prompts.at(-1).split('Attempt diagnostics: ')[1].split('\n')[0]);
  assert.equal(wire.session,session.session);
  assert.ok(wire.curriculum_sessions.some(x=>x.role==='investigation'&&x.session===session.session));
  const snapshot=JSON.parse(await readFile(result.evidence[0].path,'utf8'));
  assert.ok(snapshot.some(e=>e.type==='curriculum-interrupted'&&e.data.reason==='elapsed-allocation'));
  assert.ok(snapshot.some(e=>e.type==='curriculum-session'&&e.data.session===session.session));
});

test('valid proposal survives elapsed abort after native submission',{timeout:15000},async t=>{
  const f=await fixture(t,acceptedPlan);
  const spec=await investigate(f.options);
  assert.equal(spec.verifier.files[0],join(f.path,'investigation','verify.py'));
  assert.equal(await readFile(spec.verifier.files[0],'utf8'),'print(42)\n');
  const stop=f.events.find(e=>e.type==='curriculum-interrupted').data;
  assert.equal(stop.reason,'elapsed-allocation');assert.equal(stop.accepted,true);
});

test('valid findings survive elapsed abort and retain final evidence hash',{timeout:15000},async t=>{
  const f=await fixture(t,[['record_findings',{findings:'Observed diagnostic fact.',useful:false,paths:['evidence.log']}],'wait']);
  await writeFile(join(f.path,'evidence.log'),'fact\n');
  const result=await summarize({...f.options,passed:false});
  assert.equal(result.evidence[0].sha256,createHash('sha256').update('fact\n').digest('hex'));
  assert.equal(f.events.find(e=>e.type==='curriculum-interrupted').data.accepted,true);
  assert.equal(f.events.find(e=>e.type==='curriculum-session').data.role,'summary');
});

test('external cancellation is distinct from elapsed exhaustion',{timeout:15000},async t=>{
  const f=await fixture(t,['wait']),abort=new AbortController();f.options.config.curriculum_seconds=10;
  const timer=setTimeout(()=>abort.abort(),100);t.after(()=>clearTimeout(timer));
  await assert.rejects(investigate({...f.options,signal:abort.signal}),/external signal/);
  const stopped=f.events.find(e=>e.type==='curriculum-interrupted').data;
  assert.equal(stopped.reason,'external-signal');assert.equal(stopped.accepted,false);
  assert.ok(stopped.elapsed_seconds<stopped.allocation_seconds);
});


test('external interruption after accepted proposal preserves its valid artifacts',{timeout:15000},async t=>{
  const f=await fixture(t,acceptedPlan),abort=new AbortController();f.options.config.curriculum_seconds=10;
  const pending=investigate({...f.options,signal:abort.signal});
  await f.waitStarted;abort.abort();
  const spec=await pending;
  assert.equal(await readFile(spec.verifier.files[0],'utf8'),'print(42)\n');
  const stopped=f.events.find(e=>e.type==='curriculum-interrupted').data;
  assert.equal(stopped.reason,'external-signal');assert.equal(stopped.accepted,true);
});


for(const type of ['curriculum-session','curriculum-interrupted'])
  test(`native session shuts down and disposes when ${type} telemetry fails`,{timeout:15000},async t=>{
    const f=await fixture(t,['wait']);let shutdown=0;
    f.options.bridge={extension:pi=>pi.on('session_shutdown',async()=>{shutdown++;})};
    const original=AgentSession.prototype.dispose;
    const dispose=t.mock.method(AgentSession.prototype,'dispose',function(){return original.call(this);});
    const event=f.options.state.event.bind(f.options.state);
    f.options.state.event=(id,name,data)=>{if(name===type)throw new Error('telemetry disk failure');return event(id,name,data);};
    await assert.rejects(investigate(f.options),/telemetry disk failure/);
    assert.equal(shutdown,1);assert.equal(dispose.mock.callCount(),1);
    assert.equal(f.options.state.attempt(f.options.job.attempt.id).session,null);
  });
