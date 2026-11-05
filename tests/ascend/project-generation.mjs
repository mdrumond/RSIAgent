import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {mkdtemp,writeFile,mkdir,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {DatabaseSync} from 'node:sqlite';
import {curriculumRoleHistory} from '../../cli/curriculum.mjs';
import {State} from '../../controller/state.mjs';
import {refillPhase1,generationEventState} from '../../cli/project-generation.mjs';

async function fixture(t,{badVerifier=false,negative=0,positive=42,pause=false}={}) {
  const root=await mkdtemp(join(tmpdir(),'generation-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  let waiting;const waitStarted=new Promise(ok=>waiting=ok);
  const server=createServer(async(req,res)=>{
    let raw='';for await(const b of req)raw+=b;
    const body=JSON.parse(raw),n=body.messages.filter(m=>m.role==='tool').length;
    if(pause){waiting();await new Promise(ok=>res.once('close',ok));return;}
    const plan=[
      ['write',{path:'source/answer.py',content:'answer=0\n'}],
      ['write',{path:'reference/answer.py',content:`answer=${positive}\n`}],
      ['write',{path:'negative/answer.py',content:`answer=${negative}\n`}],
      ['write',{path:'verify.py',content:badVerifier?'pass\n':'from answer import answer\nimport sys\nsys.exit(0 if answer == 42 else 1)\n'}],
      ['submit_project',{goal:'Compute answer',source:'source',reference:'reference',negative_controls:['negative'],
        verifier:{files:['verify.py'],command:"PYTHONPATH=. python3 ../verify.py"}}]];
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'generation',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    const call=plan[n];
    if(call){frame({role:'assistant',tool_calls:[{index:0,id:'call-'+n,type:'function',function:{name:call[0],arguments:JSON.stringify(call[1])}}]});frame({},'tool_calls');}
    else{frame({role:'assistant',content:'Done'});frame({},'stop');}
    res.end('data: [DONE]\n\n');
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');
  t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));});
  const key='GENERATION_FIXTURE_KEY';process.env[key]='fixture';t.after(()=>delete process.env[key]);
  const config={root,mode:'phase1',actor_seconds:60,curriculum_seconds:20,
    curriculum:{phase1:['supplied'],generation:{enabled:true,topics:['arithmetic'],preprompt:'Construct CPU fixtures'}},
    actor:{provider:'generation-fixture',model:'scripted',credential_env:key,base_url:`http://127.0.0.1:${server.address().port}/v1`,
      metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}}};
  const options={config,state,path:join(root,'drafts'),bridge:{extension:()=>{}},concurrency:1};
  return {options,state,config,waitStarted,unpause:()=>{pause=false;}};
}

test('native Pi constructs projects; real Python controls gate staged specs without original quota',async t=>{
  const f=await fixture(t),specs=await refillPhase1(f.options);
  assert.equal(specs.length,2);
  assert.equal(f.state.db.prepare('SELECT COUNT(*) AS n FROM projects').get().n,0);
  assert.equal(f.state.db.prepare('SELECT COUNT(*) AS n FROM attempts').get().n,0);
  assert.deepEqual(specs[0].generation.checks.map(c=>c.exitCode),[0,1]);
  assert.ok(Object.keys(specs[0].verifier.hashes).length);
  const again=await refillPhase1(f.options);assert.deepEqual(again.map(s=>s.id),specs.map(s=>s.id));
  for(const s of specs)f.state.add(s);
  assert.deepEqual(await refillPhase1(f.options),[]);
  const imported=f.state.db.prepare('SELECT data FROM generation_drafts').all().map(r=>JSON.parse(r.data));
  assert.deepEqual(imported.map(d=>d.status),['imported','imported']);
  assert.ok(imported.every(d=>Object.keys(d).length===2));
});

test('a verifier accepting negative controls never enters original ledger',async t=>{
  const f=await fixture(t,{badVerifier:true});
  assert.deepEqual(await refillPhase1(f.options),[]);
  const drafts=f.state.db.prepare('SELECT data FROM generation_drafts').all().map(r=>JSON.parse(r.data));
  assert.equal(drafts.length,2);assert.ok(drafts.every(d=>d.status==='invalid'&&d.elapsed>0));
  assert.equal(f.state.db.prepare('SELECT COUNT(*) AS n FROM projects').get().n,0);
});

test('an incorrect control accepted by the verifier is rejected and phase2/inference never generate',async t=>{
  const f=await fixture(t,{negative:42});
  assert.deepEqual(await refillPhase1(f.options),[]);
  for(const mode of ['phase2','inference'])assert.deepEqual(await refillPhase1({...f.options,config:{...f.config,mode}}),[]);
});

test('batch cap and exhausted resume retain the original elapsed allocation',async t=>{
  const f=await fixture(t);
  // Initialize durable schema without provider work by filling the backlog.
  for(let i=0;i<2;i++)f.state.add({id:'p'+i,goal:'supplied',actor_seconds:60,generation:{},source:'.'});
  await refillPhase1(f.options);
  f.state.db.prepare("UPDATE projects SET status='passed'").run();
  const d={id:'generated-exhausted',status:'draft',seconds:1,elapsed:1,topic:'arithmetic'};
  f.state.db.prepare('INSERT INTO generation_drafts VALUES (?,?)').run(d.id,JSON.stringify(d));
  f.config.curriculum_seconds=.001;
  assert.deepEqual(await refillPhase1({...f.options,concurrency:10}),[]);
  const rows=f.state.db.prepare('SELECT data FROM generation_drafts').all().map(r=>JSON.parse(r.data));
  assert.equal(rows.length,8);assert.equal(rows[0].status,'exhausted');assert.equal(rows[0].seconds,1);
});


test('interruption resumes the same native draft and retains elapsed allocation',async t=>{
  const f=await fixture(t,{pause:true}),abort=new AbortController();
  const pending=refillPhase1({...f.options,signal:abort.signal});
  await f.waitStarted;abort.abort(new Error('stop'));
  await assert.rejects(pending);
  const original=JSON.parse(f.state.db.prepare('SELECT data FROM generation_drafts').get().data);
  assert.ok(original.elapsed>0);assert.equal(original.status,'draft');
  f.unpause();
  const specs=await refillPhase1(f.options);
  assert.equal(specs[0].id,original.id);
  const resumed=JSON.parse(f.state.db.prepare('SELECT data FROM generation_drafts WHERE id=?').get(original.id).data);
  assert.ok(resumed.elapsed>=original.elapsed);assert.equal(resumed.seconds,original.seconds);
  const sessions=f.state.db.prepare("SELECT data FROM generation_events WHERE draft=? AND type='curriculum-session'").all(original.id).map(r=>JSON.parse(r.data));
  assert.equal(sessions.length,2);assert.equal(sessions[0].session,sessions[1].session);
  assert.ok(sessions[1].elapsed_seconds>0);
});


test('the trusted positive must pass the independent verifier',async t=>{
  const f=await fixture(t,{positive:41});
  assert.deepEqual(await refillPhase1(f.options),[]);
  const rows=f.state.db.prepare('SELECT data FROM generation_drafts').all().map(r=>JSON.parse(r.data));
  assert.ok(rows.every(d=>d.status==='invalid'&&d.error==='CPU construction check failed: reference'));
});


test('generation telemetry pages exclusive cursors without original event tables',t=>{
  const db=new DatabaseSync(':memory:');t.after(()=>db.close());
  db.exec('CREATE TABLE generation_events(sequence INTEGER PRIMARY KEY,draft TEXT,type TEXT,data TEXT)');
  const state=generationEventState(db);
  state.event('draft','curriculum-session',{role:'generation',session:'first',allocation_seconds:30});
  for(let i=1;i<=205;i++)state.event('draft','curriculum-checkpoint',{role:'generation',elapsed_seconds:i/10});
  state.event('other','curriculum-checkpoint',{role:'generation',elapsed_seconds:999});
  state.event('draft','curriculum-session',{role:'generation',session:'latest',allocation_seconds:300});
  const page=state.queryEvents('draft',{types:['curriculum-checkpoint'],order:'desc',limit:100});
  assert.equal(page.length,100);assert.equal(page[0].data.elapsed_seconds,20.5);
  const next=state.queryEvents('draft',{types:['curriculum-checkpoint'],order:'desc',limit:100,before:page.at(-1).sequence});
  assert.equal(next.length,100);assert.ok(next[0].sequence<page.at(-1).sequence);
  const forward=state.queryEvents('draft',{order:'asc',limit:2,after:page.at(-1).sequence,before:page[0].sequence});
  assert.equal(forward.length,2);assert.equal(forward[0].sequence,page.at(-1).sequence+1);
  assert.deepEqual(state.queryEvents('draft',{types:[]}),[]);
  assert.deepEqual(curriculumRoleHistory(state,'draft','generation'),{
    prior:{role:'generation',session:'latest',allocation_seconds:300},seconds:30,elapsed:20.5});
  for(const options of [{limit:0},{limit:1001},{order:'bad'},{before:-1},{after:1.2},{types:'x'}])
    assert.throws(()=>state.queryEvents('draft',options));
});
