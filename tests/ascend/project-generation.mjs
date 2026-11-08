import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {mkdtemp,readFile,writeFile,rm,access,mkdir} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import fs from 'node:fs';
import {syncBuiltinESMExports} from 'node:module';
import {setTimeout as delay} from 'node:timers/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {DatabaseSync} from 'node:sqlite';
import {curriculumRoleHistory} from '../../cli/curriculum.mjs';
import {smokeBinding} from '../../examples/ascend-smoke/run.mjs';
import {validateTargetBaseline} from '../../controller/baseline-preflight.mjs';
import {freezeFiles} from '../../controller/evidence.mjs';
import {State} from '../../controller/state.mjs';
import {campaignProgress} from '../../controller/progress.mjs';
import {RecoveryQueue,admitJob} from '../../controller/recovery.mjs';
import {importProject} from '../../cli/instance.mjs';
import {validateBundle} from '../../cli/project-bundle.mjs';
import {createEnvironment} from '../../integrations/environments/index.mjs';
import {target} from './resilience/fixtures.mjs';
import {refillPhase1,refillProjects,generationEventState,rejectGeneratedProject,checkpointGeneratedBundle} from '../../cli/project-generation.mjs';

async function fixture(t,{badVerifier=false,negative=0,positive=42,pause=false,repair=false,afterAccepted=false,separateCommand=false,pauseAfterRejection=false,verbose=false,mutateVerifier=false,goal='Compute answer',performanceGoal,performanceCommand,baselineCommand,pytorch=false,nestedBundle=false,duplicateControl=false,unequalDepth=false,omitConstruction=false,rawConstruction=false,starterLayout,baselineDevice='cpu',brokenBaseline=false,omitBaselineConstruction=false}={}) {
  const root=await mkdtemp(join(process.env.DRIVER_ARTIFACTS??tmpdir(),'generation-'));
  t.after(()=>process.env.RSI_GENERATED_SMOKE_PROFILE?undefined:rm(root,{recursive:true,force:true}));
  const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  let verifierContent="import sys\nsys.path.insert(0, '.')\n"+(pytorch?`import sys, pathlib, torch, answer\nif '--target' in sys.argv:\n    pathlib.Path('target-requested').touch()\n    # Actor must replace the CPU-only starter with the target entry point.\n    sys.exit(1 if not hasattr(answer, 'target_entry') else 0)\nx=torch.tensor([[1.,-2.,3.],[0.,4.,-5.]])\nexpected=torch.tensor([[2.,-4.,6.],[0.,8.,-10.]])\nprint('cpu-specification-check')\nsys.exit(0 if torch.equal(answer.run(x), expected) else 1)\n`:
    badVerifier?'pass\n':`${nestedBundle?"from helper import factor\nfrom pathlib import Path\nassert (Path(__file__).parent/'reference'/'metadata.json').read_text() == 'trusted'\nassert (Path(__file__).parent/'negative'/'metadata.json').read_text() == 'wrong'\nassert factor == 2\n":''}from answer import answer\nimport sys\n${mutateVerifier?"with open(__file__, 'a') as f: f.write('# changed\\n')\n":''}${verbose?"print('x'*6000)\n":''}print('cpu-check: '+str(answer))\nsys.exit(0 if answer == 42 else 1)\n`);
  if(omitConstruction||unequalDepth)verifierContent=verifierContent.replace("parent/'reference'","parent/'controls'/'reference'/'deep'");
  const requests=[];
  let waiting;const waitStarted=new Promise(ok=>waiting=ok);
  const server=createServer(async(req,res)=>{
    let raw='';for await(const b of req)raw+=b;
    const body=JSON.parse(raw),n=body.messages.filter(m=>m.role==='tool').length;requests.push(body);
    if(pause || (pauseAfterRejection&&n===5)){waiting();await new Promise(ok=>res.once('close',ok));return;}
    const plan=[
      ['write',{path:'source/answer.py',content:pytorch?'def run(x):\n    return x + x\n':'answer=42\n'}],
      ['write',{path:'reference/answer.py',content:pytorch?`def run(x):\n    return x * ${positive===42?2:3}\n`:`answer=${positive}\n`}],
      ['write',{path:'negative/answer.py',content:pytorch?`def run(x):\n    return x * ${negative===42?2:0}\n`:`answer=${negative}\n`}],
      ['write',{path:'verify.py',content:verifierContent}],
      ['submit_project',{goal,...(performanceGoal!==undefined?{performance_goal:performanceGoal}:{}),source:'source',reference:'reference',negative_controls:['negative'],
        verifier:{files:['verify.py'],...(performanceCommand!==undefined?{performance_command:{entrypoint:'verify.py',args:[performanceCommand]}}:{}),
          ...(performanceCommand!==undefined&&baselineCommand!==null?{baseline_command:{entrypoint:'verify.py',args:[baselineCommand??'--baseline']}}:{}),
          command:{entrypoint:'verify.py',args:pytorch?['--target']:separateCommand?['requires-device-runtime']:[]}},
        construction_command:rawConstruction?'verify.py':{entrypoint:'verify.py',args:[]}}]];
    if(performanceCommand!==undefined&&!omitBaselineConstruction)plan.at(-1)[1].baseline_construction_command={entrypoint:'verify.py',args:['--baseline-cpu']};
    if(brokenBaseline) { const verifier=plan.find(call=>call[0]==='write'&&call[1].path==='verify.py')[1];
      verifier.content="import sys\nif '--baseline-cpu' in sys.argv or '--baseline' in sys.argv:\n    open('missing_baseline_helper.py').read()\n"+verifier.content; }
    if(nestedBundle) {
      plan.splice(4,0,['write',{path:'helper.py',content:'factor=2\n'}],
        ['write',{path:'reference/metadata.json',content:'trusted'}],['write',{path:'negative/metadata.json',content:'wrong'}]);
      for(let i=1;i<8;i++)plan.splice(plan.length-1,0,['write',{path:`negative-${i}/answer.py`,content:'answer=0\n'}],
        ['write',{path:`negative-${i}/metadata.json`,content:'wrong'}]);
      for(const call of plan)if(call[0]==='write')call[1].path='project/task/'+call[1].path;
      const proposal=plan.at(-1)[1];proposal.source='project/task/source';proposal.reference='project/task/reference';
      proposal.negative_controls=['project/task/negative',...Array.from({length:7},(_,i)=>`project/task/negative-${i+1}`)];proposal.verifier.files=['project/task/verify.py'];
      proposal.construction_command={entrypoint:'project/task/verify.py',args:[]};
      if(proposal.baseline_construction_command)proposal.baseline_construction_command.entrypoint='project/task/verify.py';
      for(const key of ['command','baseline_command','performance_command'])if(proposal.verifier[key])proposal.verifier[key].entrypoint='project/task/verify.py';
      if(unequalDepth) {
        const starter=plan.find(call=>call[0]==='write'&&call[1].path==='project/task/source/answer.py');
        starter[1].path='project/task/src/impl/answer.py';proposal.source='project/task/src/impl';

      }
      if(omitConstruction||unequalDepth) {
        proposal.reference='project/task/controls/reference/deep';
        for(const call of plan)if(call[0]==='write'&&call[1].path.startsWith('project/task/reference/'))
          call[1].path=call[1].path.replace('project/task/reference/',proposal.reference+'/');
      }
    }
    if(pytorch&&baselineDevice!=='cpu') {
      const file=plan.find(call=>call[0]==='write'&&call[1].path==='verify.py')[1];
      file.content=file.content.replace('expected=torch.tensor',"if '--baseline' in sys.argv:\n    import torch_npu\n    x=x.to('npu:0')\nexpected=torch.tensor").replace('torch.equal(answer.run(x), expected)', 'torch.equal(answer.run(x).cpu(), expected)');
    }
    if(starterLayout) {
      for(const call of plan)if(call[0]==='write') {
        if(call[1].path.endsWith('/answer.py'))call[1].path=call[1].path.replace('/answer.py','/src/answer.py');
        if(call[1].path==='verify.py')call[1].content=call[1].content.replace("sys.path.insert(0, '.')","sys.path.insert(0, 'src')");
      }
      if(starterLayout==='mismatch')plan.at(-1)[1].source='source/src';
    }
    if(omitConstruction)delete plan.at(-1)[1].construction_command;
    if(duplicateControl)plan.at(-1)[1].negative_controls=['negative','negative'];
    if(repair)plan.push(['write',{path:'negative/answer.py',content:'answer=-12345\n'}],plan[4]);
    if(afterAccepted)plan.push(['write',{path:'verify.py',content:'raise Exception("modified")\n'}],
      ['write',{path:'source/answer.py',content:'answer=999\n'}],plan[4]);
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
  // ARM Torch startup is paid by each independent construction process.
  const config={root,mode:'phase1',actor_seconds:60,curriculum_seconds:pytorch||nestedBundle?120:20,
    curriculum:{phase1:['supplied'],generation:{enabled:true,topics:['arithmetic'],preprompt:'Construct CPU fixtures'}},
    actor:{provider:'generation-fixture',model:'scripted',credential_env:key,base_url:`http://127.0.0.1:${server.address().port}/v1`,
      metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}}};
  const options={config,state,path:join(root,'drafts'),bridge:{extension:()=>{}},concurrency:1};
  return {options,state,config,requests,verifierContent,waitStarted,unpause:()=>{pause=false;pauseAfterRejection=false;},
    setPerformance:(goal,command)=>{performanceGoal=goal;performanceCommand=command;separateCommand=true;}};
}

function topicQuota(f,topics) {
  f.config.campaign_plan=[{id:'batch',phase:'phase1',new_projects:topics.length,actor_seconds:7200,
    generation:{topics,requirements:{SIMT:{performance:true,capabilityEvidence:true,comparison:'non-simt-or-vector'}}}}];
  f.config.kernel_language='triton';
}

test('PyTorch-only SIMT specification is admitted on CPU while target implementation remains deferred',async t=>{
  const f=await fixture(t,{pytorch:true,performanceGoal:{comparison:'non-SIMT baseline'},performanceCommand:'actor-target-benchmark'});
  topicQuota(f,['SIMT']);
  const [spec]=await refillProjects(f.options);
  assert.ok(spec,JSON.stringify(f.state.db.prepare('SELECT data FROM generation_drafts').all()));
  assert.deepEqual(spec.generation.checks.filter(c=>c.kind!=='baseline').map(c=>c.exitCode),[0,1,0]);
  assert.equal(spec.generation.requirements.capabilityEvidence,true);
  assert.deepEqual(spec.performance_goal,{comparison:'non-SIMT baseline'});
  assert.deepEqual(spec.verifier.entrypoints.performance_command.args,['actor-target-benchmark']);
  assert.ok(spec.verifier.baseline_command);
  const imported=await importProject(f.config.root,spec);
  assert.equal(imported.verifier.baseline_command,spec.verifier.baseline_command);
  await assert.rejects(access(join(spec.source,'target-requested')),{code:'ENOENT'});
  assert.throws(()=>execFileSync(process.env.RSI_TEST_PYTHON??process.env.PYTHON??'python3',
    [spec.verifier.files[0],'--target'],{cwd:spec.source,env:{...process.env,PYTHONPATH:spec.source},stdio:'pipe'}),
    error=>error.status===1);
  await access(join(spec.source,'target-requested'));
  assert.equal(f.state.db.prepare('SELECT COUNT(*) n FROM attempts').get().n,0);
});

test('generated performance acceptance without an independent target baseline is rejected',async t=>{
  const f=await fixture(t,{performanceGoal:'speedup',performanceCommand:'target-performance',baselineCommand:null,separateCommand:true});
  assert.deepEqual(await refillProjects(f.options),[]);
  assert.equal(f.state.db.prepare('SELECT COUNT(*) n FROM projects').get().n,0);
  assert.equal(f.state.db.prepare('SELECT COUNT(*) n FROM attempts').get().n,0);
});

test('PyTorch specifications still require a correct reference and discriminating negative controls',async t=>{
  for(const scenario of [{positive:41},{negative:42}])await t.test(JSON.stringify(scenario),async t=>{
    const f=await fixture(t,{pytorch:true,...scenario});topicQuota(f,['layout']);
    assert.deepEqual(await refillProjects(f.options),[]);
    assert.equal(f.state.db.prepare('SELECT COUNT(*) n FROM projects').get().n,0);
    const rejection=generationEventState(f.state.db).queryEvents(
      f.state.db.prepare('SELECT id FROM generation_drafts').get().id,{types:['generation-rejected']})[0];
    assert.equal(rejection.data.checks.at(-1).kind,scenario.positive?'reference':'negative');
    assert.equal(rejection.data.checks.at(-1).exitCode,scenario.positive?1:0);
  });
});

test('entry topic quotas stage one SIMT benchmark and seven distinct projects without running device timing',async t=>{
  const performanceGoal={comparison:'vector baseline'},performanceCommand='isolated-target-timing';
  const f=await fixture(t,{performanceGoal,performanceCommand,separateCommand:true});
  const topics=['SIMT','layout','fusion','reduction','softmax','matmul','pipeline','sparse'];topicQuota(f,topics);
  f.config.curriculum.generation.by_phase={phase1:{topics:['stale'],requirements:{SIMT:{performance:false}}}};
  const specs=await refillProjects({...f.options,concurrency:8});
  assert.deepEqual(specs.map(spec=>spec.generation.topic),topics);
  assert.ok(specs.every(spec=>spec.actor_seconds===7200&&spec.campaign_entry==='batch'));
  assert.deepEqual(specs[0].performance_goal,performanceGoal);
  assert.deepEqual(specs[0].verifier.entrypoints.performance_command.args,[performanceCommand]);
  assert.equal(specs[0].generation.requirements.capabilityEvidence,true);
  assert.ok(specs[0].generation.checks.every(check=>check.command!==performanceCommand));
  const before=f.requests.length;
  assert.deepEqual((await refillProjects({...f.options,concurrency:8})).map(spec=>spec.id),specs.map(spec=>spec.id));
  assert.equal(f.requests.length,before);
  for(const spec of specs)f.state.add(spec);
  assert.deepEqual(await refillProjects({...f.options,concurrency:8}),[]);
});

test('missing SIMT acceptance is rejected; invalid and exhausted drafts do not consume its topic',async t=>{
  const f=await fixture(t);topicQuota(f,['SIMT','reduction']);
  assert.deepEqual(await refillProjects(f.options),[]);
  const failed=f.state.db.prepare('SELECT data FROM generation_drafts').all().map(row=>JSON.parse(row.data));
  assert.equal(failed.length,2);assert.ok(failed.every(draft=>draft.status==='invalid'&&draft.topic==='SIMT'));
  assert.equal(f.state.db.prepare('SELECT COUNT(*) n FROM projects').get().n,0);
  const prior={...failed[0],status:'draft',seconds:1,elapsed:1};
  f.state.db.prepare('UPDATE generation_drafts SET data=? WHERE id=?').run(JSON.stringify(prior),prior.id);
  f.setPerformance('compare supported SIMT mapping','target-benchmark');
  const [replacement]=await refillProjects(f.options);
  assert.equal(replacement.generation.topic,'SIMT');assert.notEqual(replacement.id,prior.id);
  f.state.add(replacement);
  const [last]=await refillProjects(f.options);assert.equal(last.generation.topic,'reduction');
  const retained=JSON.parse(f.state.db.prepare('SELECT data FROM generation_drafts WHERE id=?').get(prior.id).data);
  assert.equal(retained.status,'exhausted');assert.equal(retained.seconds,1);assert.equal(retained.elapsed,1);
});

test('native Pi constructs projects; real Python controls gate staged specs without original quota',async t=>{
  const f=await fixture(t),specs=await refillPhase1(f.options);
  assert.equal(specs.length,2);
  assert.equal(f.state.db.prepare('SELECT COUNT(*) AS n FROM projects').get().n,0);
  assert.equal(f.state.db.prepare('SELECT COUNT(*) AS n FROM attempts').get().n,0);
  assert.deepEqual(specs[0].generation.checks.filter(c=>c.kind!=='baseline').map(c=>c.exitCode),[0,1,0]);
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

test('an incorrect control accepted by the verifier is rejected and inference never generates',async t=>{
  const f=await fixture(t,{negative:42});
  assert.deepEqual(await refillPhase1(f.options),[]);
  for(const mode of ['inference'])assert.deepEqual(await refillPhase1({...f.options,config:{...f.config,mode}}),[]);
});

test('batch cap and exhausted resume retain the original elapsed allocation',async t=>{
  const f=await fixture(t);
  // Initialize durable schema without provider work by filling the backlog.
  for(let i=0;i<2;i++)f.state.add({id:'p'+i,goal:'supplied',actor_seconds:60,generation:{},source:'.'});
  await refillPhase1(f.options);
  f.state.db.prepare("UPDATE projects SET status='passed'").run();
  const d={id:'generated-exhausted',status:'draft',phase:'phase1',campaign_entry:null,seconds:1,elapsed:1,topic:'arithmetic'};
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
  assert.ok(rows.every(d=>d.status==='invalid'));
  const rejections=f.state.db.prepare("SELECT data FROM generation_events WHERE type='generation-rejected'").all().map(r=>JSON.parse(r.data));
  assert.ok(rejections.every(e=>e.checks.length===1&&e.checks[0].kind==='reference'&&e.checks[0].exitCode===1));
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


test('native Pi repairs a rejected negative and resubmits in the same draft session',async t=>{
  const f=await fixture(t,{negative:42,repair:true,afterAccepted:true,verbose:true});
  const specs=await refillPhase1(f.options);assert.equal(specs.length,2);
  assert.equal(f.state.db.prepare('SELECT COUNT(*) AS n FROM generation_drafts').get().n,2);
  for(const spec of specs) {
    const events=generationEventState(f.state.db).queryEvents(spec.id,{limit:1000});
    const sessions=events.filter(e=>e.type==='curriculum-session');assert.equal(sessions.length,1);
    const [failure]=events.filter(e=>e.type==='generation-rejected');assert.ok(failure);
    const check=failure.data.checks.at(-1);
    assert.equal(check.kind,'negative');assert.equal(check.exitCode,0);assert.equal(check.expected_exit,1);
    assert.equal(check.cwd,join(f.options.path,spec.id,'negative'));
    assert.ok((await readFile(check.log,'utf8')).length>6000);assert.equal(check.output.length,4000);
    assert.deepEqual(spec.generation.checks.filter(c=>c.kind!=='baseline').map(c=>c.exitCode),[0,1,0]);
    assert.equal(await readFile(join(spec.source,'answer.py'),'utf8'),'answer=42\n');
    assert.equal(await readFile(spec.verifier.files[0],'utf8'),f.verifierContent);
  }
  const replies=f.requests.flatMap(r=>r.messages.filter(m=>m.role==='tool'));
  const error=replies.find(m=>m.tool_call_id==='call-4');
  assert.ok(error); // The model receives the controller's actual negative-check observation.
  const observed=JSON.parse(error.content.slice(error.content.indexOf('{')));
  assert.equal(observed.expected_exit,1);assert.equal(observed.exitCode,0);
  assert.ok(observed.log.endsWith('.log'));
});

test('separate CPU construction command preserves the actual target verifier',async t=>{
  const f=await fixture(t,{separateCommand:true}),specs=await refillPhase1(f.options);
  assert.equal(specs.length,2);
  assert.ok(specs.every(s=>s.verifier.entrypoints.command.args[0]==='requires-device-runtime'));
  assert.deepEqual(specs[0].generation.checks.filter(c=>c.kind!=='baseline').map(c=>c.exitCode),[0,1,0]);
});


test('a rejected submission resumes with its evidence and remaining budget in the same session',async t=>{
  const f=await fixture(t,{negative:42,repair:true,pauseAfterRejection:true}),abort=new AbortController();
  const pending=refillPhase1({...f.options,signal:abort.signal});
  await f.waitStarted;
  const before=JSON.parse(f.state.db.prepare('SELECT data FROM generation_drafts').get().data);
  assert.equal(before.status,'draft');assert.equal(before.proposal,undefined);assert.equal(before.spec,undefined);
  const failure=generationEventState(f.state.db).queryEvents(before.id,{types:['generation-rejected']})[0];
  assert.equal(failure.data.checks.at(-1).exitCode,0);
  abort.abort(new Error('stop after rejection'));await assert.rejects(pending);
  const stopped=JSON.parse(f.state.db.prepare('SELECT data FROM generation_drafts').get().data);
  assert.ok(stopped.elapsed>0);assert.equal(stopped.status,'draft');
  f.unpause();const specs=await refillPhase1(f.options);
  assert.equal(specs[0].id,before.id);
  const events=generationEventState(f.state.db).queryEvents(before.id,{limit:1000});
  const sessions=events.filter(e=>e.type==='curriculum-session');
  assert.equal(sessions.length,2);assert.equal(sessions[0].data.session,sessions[1].data.session);
  assert.equal(sessions[0].data.allocation_seconds,sessions[1].data.allocation_seconds);
  assert.ok(sessions[1].data.elapsed_seconds>0);
  assert.equal(events.filter(e=>e.type==='generation-rejected').length,1);
  assert.equal(await readFile(failure.data.checks.at(-1).log,'utf8'),'cpu-check: 42\n');
});


function campaign(f,phase='phase2',count=2) {
  f.config.mode=phase;
  f.config.campaign_plan=[{id:'old',phase,new_projects:2},{id:'fresh',phase,new_projects:count}];
  // Persist the completed entry's cursor without running an actor campaign.
  f.state.db.exec('CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
  f.state.db.prepare("INSERT OR REPLACE INTO control VALUES ('campaign-plan',?)")
    .run(JSON.stringify({plan:f.config.campaign_plan,index:1}));
}

test('phase2 generates a fresh batch ignoring older projects and ready drafts',async t=>{
  const goal='End-to-end throughput >= 100 samples/s';
  const f=await fixture(t,{goal});
  campaign(f);
  f.config.curriculum.generation.by_phase={phase2:{topics:['end-to-end throughput']}};
  // An old ready draft must not be emitted or included in the current backlog.
  f.state.add({id:'old-project',phase:'phase2',campaign_entry:'old',goal:'old',source:'.',actor_seconds:60});
  const first=await refillProjects(f.options);
  assert.equal(first.length,2);
  for(const spec of first) {
    assert.equal(spec.phase,'phase2');assert.equal(spec.campaign_entry,'fresh');
    assert.equal(spec.goal,goal);
    assert.equal(spec.generation.topic,'end-to-end throughput');
    assert.deepEqual(spec.generation.checks.filter(c=>c.kind!=='baseline').map(c=>c.exitCode),[0,1,0]);
    assert.notEqual(spec.id,'old-project');assert.equal(spec.parent,undefined);
  }
  const plan=[...f.config.campaign_plan,{id:'next',phase:'phase2',new_projects:1}];
  f.config.campaign_plan=plan;
  f.state.db.prepare("UPDATE control SET value=? WHERE key='campaign-plan'").run(JSON.stringify({plan,index:2}));
  const next=await refillProjects(f.options);
  assert.equal(next.length,1);assert.equal(next[0].campaign_entry,'next');
  assert.ok(!first.some(s=>s.id===next[0].id));
  const requests=f.requests.length;
  assert.deepEqual((await refillProjects(f.options)).map(s=>s.id),next.map(s=>s.id));
  assert.equal(f.requests.length,requests);
});

test('a phase2 interrupted draft resumes within admission quota without duplicates',async t=>{
  const f=await fixture(t,{pause:true});campaign(f,'phase2',1);
  const abort=new AbortController(),pending=refillProjects({...f.options,signal:abort.signal});
  await f.waitStarted;abort.abort(new Error('stop'));await assert.rejects(pending);
  const before=JSON.parse(f.state.db.prepare('SELECT data FROM generation_drafts').get().data);
  assert.equal(before.phase,'phase2');assert.equal(before.campaign_entry,'fresh');
  f.unpause();
  const specs=await refillProjects({...f.options,concurrency:8});
  assert.equal(specs.length,1);assert.equal(specs[0].id,before.id);
  assert.deepEqual((await refillProjects({...f.options,concurrency:8})).map(s=>s.id),specs.map(s=>s.id));
  assert.equal(f.state.db.prepare('SELECT COUNT(*) n FROM generation_drafts').get().n,1);
  f.state.add(specs[0]);
  assert.deepEqual(await refillProjects({...f.options,concurrency:8}),[]);
});

test('unattempted current-entry projects consume admission space but old projects do not',async t=>{
  const f=await fixture(t);campaign(f,'phase1',2);
  for(const [id,entry] of [['queued','fresh'],['old','old']])
    f.state.add({id,phase:'phase1',campaign_entry:entry,goal:'supplied',source:'.',actor_seconds:60});
  const specs=await refillProjects({...f.options,concurrency:8});
  assert.equal(specs.length,1);assert.equal(specs[0].campaign_entry,'fresh');
  f.state.add(specs[0]);assert.deepEqual(await refillProjects({...f.options,concurrency:8}),[]);
});

test('phase2 rejects CPU-negative acceptance and verifier mutation',async t=>{
  for(const options of [{negative:42},{mutateVerifier:true}]) {
    const f=await fixture(t,options);campaign(f,'phase2',1);
    assert.deepEqual(await refillProjects(f.options),[]);
    const drafts=f.state.db.prepare('SELECT data FROM generation_drafts').all().map(r=>JSON.parse(r.data));
    assert.equal(drafts.length,1);assert.equal(drafts[0].status,'invalid');
    assert.equal(f.state.db.prepare('SELECT COUNT(*) n FROM projects').get().n,0);
    const [failure]=generationEventState(f.state.db).queryEvents(drafts[0].id,{types:['generation-rejected']});
    if(options.negative)assert.equal(failure.data.checks.at(-1).exitCode,0);
    else assert.equal(failure.data.error,'Verifier changed during CPU checks');
  }
});


test('admitted originals consume remainingAdmissions exactly once',async t=>{
  const f=await fixture(t);campaign(f,'phase2',2);
  f.state.add({id:'active',phase:'phase2',campaign_entry:'fresh',goal:'active',source:'.',actor_seconds:60});
  admitJob(f.state,{projectId:'active',wave:1,mode:'phase2'});
  const specs=await refillProjects({...f.options,concurrency:8});
  assert.equal(specs.length,1);
  f.state.add(specs[0]);admitJob(f.state,{projectId:specs[0].id,wave:1,mode:'phase2'});
  assert.deepEqual(await refillProjects({...f.options,concurrency:8}),[]);
});


test('phase2 retains declared performance contracts without executing timing on CPU',async t=>{
  const performanceGoal={minimum_samples_per_second:100};
  const performanceCommand='requires-isolated-target benchmark';
  const f=await fixture(t,{performanceGoal,performanceCommand});campaign(f,'phase2',1);
  const [spec]=await refillProjects(f.options);
  assert.deepEqual(spec.performance_goal,performanceGoal);
  assert.deepEqual(spec.verifier.entrypoints.performance_command.args,[performanceCommand]);
  assert.deepEqual(spec.generation.checks.filter(c=>c.kind!=='baseline').map(c=>c.exitCode),[0,1,0]);
  assert.ok(spec.generation.checks.every(c=>c.command!==performanceCommand));
  f.state.add(spec);assert.deepEqual(f.state.project(spec.id).spec.performance_goal,performanceGoal);
});

test('missing performance commands block admission and phase1 does not inherit phase2 goals',async t=>{
  const missing=await fixture(t,{performanceGoal:'declared throughput'});campaign(missing,'phase2',1);
  assert.deepEqual(await refillProjects(missing.options),[]);
  assert.equal(missing.state.db.prepare('SELECT COUNT(*) n FROM projects').get().n,0);
  const ordinary=await fixture(t);
  ordinary.config.curriculum.generation.by_phase={phase2:{preprompt:'Require target throughput'}};
  const specs=await refillProjects(ordinary.options);
  assert.ok(specs.every(s=>s.phase==='phase1'&&s.performance_goal===undefined&&s.verifier.performance_command===undefined));
  const explicit=await fixture(t,{performanceGoal:'declared microbenchmark target',performanceCommand:'benchmark'});
  const microbenchmarks=await refillProjects(explicit.options);
  assert.equal(microbenchmarks.length,2);
  for(const spec of microbenchmarks) {
    assert.equal(spec.phase,'phase1');assert.equal(spec.performance_goal,'declared microbenchmark target');
    assert.deepEqual(spec.verifier.entrypoints.performance_command.args,['benchmark']);explicit.state.add(spec);
  }
});


test('phase overrides agree on non-plan completion and refilling in both phases',async t=>{
  for(const phase of ['phase1','phase2'])for(const enabled of [true,false]) {
    const f=await fixture(t);f.config.mode=phase;
    f.config.curriculum[phase]=['supplied'];
    f.config.curriculum.generation={enabled:!enabled,topics:['default'],
      by_phase:{[phase]:{enabled,topics:['override-topic']}}};
    f.state.add({id:'supplied',phase,goal:'supplied',actor_seconds:60});
    const queue=new RecoveryQueue(f.state,{mode:phase}),job=queue.claim('supplied',0);
    queue.recordOriginal(job.attempt.id,{passed:true,findings:'correct',evidence:[{collection:'memory',entry_id:'proof'}]},0);
    queue.acknowledgeLesson(queue.lessons()[0].id);
    f.state.event(job.attempt.id,'postprocess-completed',{});
    assert.equal(campaignProgress(f.state,f.config).phaseComplete,!enabled);
    const specs=await refillProjects(f.options);
    assert.equal(specs.length,enabled?2:0);
    assert.ok(specs.every(s=>s.phase===phase&&s.generation.topic==='override-topic'));
    if(!enabled)assert.equal(f.requests.length,0);
    assert.equal(campaignProgress(f.state,{...f.config,completion_target:1}).phaseComplete,true);
  }
});

test('old non-plan interrupted phase1 drafts retain ID, native session and elapsed budget',async t=>{
  const f=await fixture(t,{pause:true}),abort=new AbortController();
  const pending=refillProjects({...f.options,signal:abort.signal});
  await f.waitStarted;abort.abort(new Error('old stop'));await assert.rejects(pending);
  f.state.db.prepare("UPDATE generation_drafts SET data=json_remove(data,'$.phase','$.campaign_entry')").run();
  const before=JSON.parse(f.state.db.prepare('SELECT data FROM generation_drafts').get().data);
  const priorSession=generationEventState(f.state.db).queryEvents(before.id,{types:['curriculum-session']})[0].data.session;
  f.unpause();const specs=await refillProjects(f.options);
  assert.equal(specs[0].id,before.id);assert.equal(specs[0].phase,'phase1');
  const after=JSON.parse(f.state.db.prepare('SELECT data FROM generation_drafts WHERE id=?').get(before.id).data);
  assert.equal(after.campaign_entry,null);assert.equal(after.seconds,before.seconds);
  assert.ok(after.elapsed>=before.elapsed);
  const sessions=generationEventState(f.state.db).queryEvents(before.id,{types:['curriculum-session']});
  assert.equal(sessions.length,2);assert.ok(sessions.every(s=>s.data.session===priorSession));
  assert.ok(sessions[1].data.elapsed_seconds>0);
});

test('old ready phase1 drafts are returned without replacement or budget reset, never adopted by a plan',async t=>{
  const f=await fixture(t),specs=await refillProjects(f.options);
  f.state.db.prepare("UPDATE generation_drafts SET data=json_remove(data,'$.phase','$.campaign_entry')").run();
  const before=f.state.db.prepare('SELECT data FROM generation_drafts ORDER BY rowid').all().map(r=>JSON.parse(r.data));
  const requests=f.requests.length;
  const resumed=await refillProjects(f.options);
  assert.deepEqual(resumed.map(s=>s.id),specs.map(s=>s.id));assert.equal(f.requests.length,requests);
  const after=f.state.db.prepare('SELECT data FROM generation_drafts ORDER BY rowid').all().map(r=>JSON.parse(r.data));
  assert.deepEqual(after,before.map(d=>({...d,phase:'phase1',campaign_entry:null})));
  // The same unscoped stored shape is ambiguous once a campaign plan exists.
  f.state.db.prepare("UPDATE generation_drafts SET data=json_remove(data,'$.phase','$.campaign_entry')").run();
  campaign(f,'phase1',1);
  const planned=await refillProjects(f.options);
  assert.equal(planned.length,1);assert.ok(!specs.some(s=>s.id===planned[0].id));
  for(const spec of specs)assert.equal(JSON.parse(f.state.db.prepare('SELECT data FROM generation_drafts WHERE id=?').get(spec.id).data).phase,undefined);
});


test('generated immutable bundle survives JSON export, import and actual CPU staging without its original tree',async t=>{
  const f=await fixture(t,{nestedBundle:true});
  const [generated]=await refillProjects(f.options);assert.ok(generated);
  const draft=JSON.parse(f.state.db.prepare('SELECT data FROM generation_drafts WHERE id=?').get(generated.id).data);
  assert.equal(generated.bundle.seconds,draft.seconds-draft.elapsed);
  const exported=join(f.config.root,'export.json');await writeFile(exported,JSON.stringify(generated));
  const imported=await importProject(join(f.config.root,'instance'),JSON.parse(await readFile(exported,'utf8')));
  assert.ok(imported.bundle.seconds<generated.bundle.seconds);
  assert.ok(imported.verifier.files.some(path=>path.endsWith('/helper.py')));
  assert.ok(imported.verifier.files.some(path=>path.endsWith('/reference/metadata.json')));
  assert.ok(imported.verifier.files.some(path=>path.endsWith('/negative/metadata.json')));
  await rm(f.options.path,{recursive:true});
  await validateBundle(imported,join(f.config.root,'roundtrip-proof'));
  const backendRoot=await mkdtemp(join(tmpdir(),'bundle-target-')),backend=await target({root:backendRoot});
  t.after(async()=>{try{await backend.close();}finally{await rm(backendRoot,{recursive:true,force:true});}});
  const environment=await createEnvironment({mode:'cpl',target:'fixture',runtime:'cpu',workspacePath:backend.workspace,
    remoteDirectory:'owned',pollMs:1},{instancePath:join(f.config.root,'staging'),client:backend.client,
    exclude:async(_label,fn)=>fn(),event:()=>{}});
  await environment.prepare({project:imported,attempt:{id:'roundtrip'}});
  for(const [directory,exitCode] of [['reference',0],['negative',1],...Array.from({length:7},(_,i)=>[`negative-${i+1}`,1])]) {
    const result=await environment.execute(`cd ../${directory} && ${imported.bundle.construction_command}`,imported.source,{timeout:20});
    assert.equal(result.settled,true);assert.equal(result.exitCode,exitCode);
  }
  assert.ok([...backend.jobs.values()].every(job=>job.state==='completed'));
});

test('import rejects missing manifested dependencies and CPU checks reject an omitted helper before admission',async t=>{
  const f=await fixture(t,{nestedBundle:true});const [spec]=await refillProjects(f.options);assert.ok(spec);
  const helper=spec.verifier.files.find(path=>path.endsWith('/helper.py'));
  const omitted={...spec,verifier:{...spec.verifier,files:spec.verifier.files.filter(path=>path!==helper)}};
  await assert.rejects(importProject(join(f.config.root,'omitted'),omitted),/Imported bundle CPU reference failed/);
  assert.equal(f.state.db.prepare('SELECT COUNT(*) n FROM attempts').get().n,0);
  await rm(helper);
  await assert.rejects(importProject(join(f.config.root,'missing'),spec),{code:'ENOENT'});
});


test('duplicate negative fixture identities reject generation instead of counting as distinct controls',async t=>{
  const f=await fixture(t,{duplicateControl:true});
  assert.deepEqual(await refillProjects(f.options),[]);
  assert.equal(f.state.db.prepare('SELECT COUNT(*) n FROM projects').get().n,0);
  const draft=JSON.parse(f.state.db.prepare('SELECT data FROM generation_drafts').get().data);
  assert.equal(draft.status,'invalid');
});

test('rejected imported closure is not reoffered and interrupted import retains its elapsed allocation',async t=>{
  const f=await fixture(t,{nestedBundle:true});const [spec]=await refillProjects(f.options);
  const remaining=spec.bundle.seconds;
  spec.bundle.construction_command='sleep 1; '+spec.bundle.construction_command;
  const interrupted=new AbortController();let saved;
  await assert.rejects(importProject(join(f.config.root,'interrupted'),spec,{signal:interrupted.signal,
    checkpoint:seconds=>{saved=seconds;checkpointGeneratedBundle(f.state,spec,seconds);interrupted.abort();}}));
  assert.ok(saved<remaining);
  const savedDraft=JSON.parse(f.state.db.prepare('SELECT data FROM generation_drafts WHERE id=?').get(spec.id).data);
  assert.equal(savedDraft.spec.bundle.seconds,saved);
  const resumed=(await refillProjects(f.options)).find(candidate=>candidate.id===spec.id);
  assert.equal(resumed.bundle.seconds,saved);
  assert.ok(savedDraft.elapsed>=savedDraft.seconds-saved);
  rejectGeneratedProject(f.state,spec,new Error('Imported bundle CPU reference failed'));
  const rejected=JSON.parse(f.state.db.prepare('SELECT data FROM generation_drafts WHERE id=?').get(spec.id).data);
  assert.equal(rejected.status,'invalid');
  assert.equal(f.state.db.prepare("SELECT COUNT(*) n FROM generation_drafts WHERE id=? AND json_extract(data,'$.status') IN ('draft','ready')").get(spec.id).n,0);
});


test('structured construction entrypoint validates reference and controls at unequal cwd depths',async t=>{
  const f=await fixture(t,{nestedBundle:true,unequalDepth:true});
  const [spec]=await refillProjects(f.options);assert.ok(spec,JSON.stringify(f.state.db.prepare('SELECT data FROM generation_drafts').all()));
  assert.deepEqual(spec.generation.checks.filter(check=>check.kind!=='baseline').map(check=>check.exitCode),[0,1,1,1,1,1,1,1,1,0]);
  const imported=await importProject(join(f.config.root,'instance'),JSON.parse(JSON.stringify(spec)));
  await rm(f.options.path,{recursive:true});
  await validateBundle(imported,join(f.config.root,'unequal-proof'));
  await writeFile(join(imported.source,'answer.py'),'answer=0\n');
  assert.throws(()=>execFileSync('bash',['-c',imported.verifier.command],{cwd:imported.source}),
    error=>error.status===1&&error.stdout.toString().trim()==='cpu-check: 0');
});

test('omitted construction command uses one frozen entrypoint across unequal fixture depths',async t=>{
  const f=await fixture(t,{nestedBundle:true,unequalDepth:true,omitConstruction:true});
  const [spec]=await refillProjects(f.options);
  assert.ok(spec,JSON.stringify(f.state.db.prepare('SELECT data FROM generation_drafts').all()));
  assert.deepEqual(spec.generation.checks.filter(check=>check.kind!=='baseline').map(check=>check.exitCode),[0,1,1,1,1,1,1,1,1,0]);
  const imported=await importProject(join(f.config.root,'instance'),JSON.parse(JSON.stringify(spec)));
  await rm(f.options.path,{recursive:true});
  await validateBundle(imported,join(f.config.root,'fallback-proof'));
  await writeFile(join(imported.source,'answer.py'),'answer=0\n');
  assert.throws(()=>execFileSync('bash',['-c',imported.verifier.command],{cwd:imported.source}),
    error=>error.status===1&&error.stdout.toString().trim()==='cpu-check: 0');
});

test('slow pre-CPU import hashing consumes and checkpoints the same elapsed allocation',async t=>{
  const f=await fixture(t,{nestedBundle:true});const [spec]=await refillProjects(f.options);assert.ok(spec);
  spec.bundle.seconds=.5;
  const marker=join(f.config.root,'cpu-must-not-start');
  spec.bundle.construction_command=`touch '${marker}'; `+spec.bundle.construction_command;
  const original=fs.createReadStream,slowFile=spec.verifier.files[0];let delayed=false,hashing=false,checkpointDuringHash=false,saved;
  t.mock.method(fs,'createReadStream',function(path,...args) {
    if(path!==slowFile||delayed)return original(path,...args);
    delayed=true;
    return (async function*(){hashing=true;try{await delay(1000);for await(const chunk of original(path,...args))yield chunk;}finally{hashing=false;}})();
  });
  syncBuiltinESMExports();t.after(()=>{t.mock.restoreAll();syncBuiltinESMExports();});
  await assert.rejects(importProject(join(f.config.root,'slow'),spec,{checkpoint:seconds=>{
    saved=seconds;checkpointDuringHash ||= hashing;checkpointGeneratedBundle(f.state,spec,seconds);
  }}));
  assert.equal(delayed,true);assert.equal(checkpointDuringHash,true);assert.equal(saved,0);
  assert.equal(JSON.parse(f.state.db.prepare('SELECT data FROM generation_drafts WHERE id=?').get(spec.id).data).spec.bundle.seconds,0);
  await assert.rejects(access(marker),{code:'ENOENT'});
});

test('raw construction shell path is rejected before any CPU execution',async t=>{
  const f=await fixture(t,{rawConstruction:true});
  assert.deepEqual(await refillProjects(f.options),[]);
  const events=f.state.db.prepare("SELECT type FROM generation_events WHERE type='generation-construction-check'").all();
  assert.deepEqual(events,[]);
  assert.equal(f.state.db.prepare('SELECT COUNT(*) AS n FROM projects').get().n,0);
});


test('generated starter with mismatched candidate tree is rejected before import or actor admission',async t=>{
  const f=await fixture(t,{starterLayout:'mismatch'});
  const specs=await refillProjects(f.options);
  assert.deepEqual(specs,[]);
  const rows=f.state.db.prepare("SELECT data FROM generation_events WHERE type='generation-rejected'").all();
  assert.ok(rows.length);for(const row of rows) {
    const check=JSON.parse(row.data).checks.at(-1);
    assert.equal(check.kind,'starter');assert.equal(check.exitCode,1);
  }
  assert.equal(f.state.db.prepare('SELECT COUNT(*) AS n FROM attempts').get().n,0);
});

test('generated nested starter survives import staging and real baseline preflight',{timeout:900000},async t=>{
  const liveFile=process.env.RSI_GENERATED_SMOKE_PROFILE;
  let live,binding;if(liveFile) {
    assert.equal(process.env.RSI_ASCEND_SMOKE_LIVE,'1','Live generated smoke requires explicit authorization');
    live=JSON.parse(await readFile(liveFile));binding=smokeBinding(live,{kind:'controller'});
    assert.ok(['a3','a5'].includes(binding.target));
  }
  const f=await fixture(t,{pytorch:true,starterLayout:'compatible',baselineDevice:live?'npu:0':'cpu',
    performanceGoal:'Tiny baseline execution analog',performanceCommand:'--performance'});
  if(live) {f.config.local_execution=binding.localExecution;f.config.target=binding.target;}
  const [spec]=await refillProjects(f.options);assert.ok(spec);
  const imported=await importProject(join(f.config.root,'instance'),spec,{localExecution:f.config.local_execution});
  await rm(f.options.path,{recursive:true});
  let options,client;
  if(live)options={...live.environment.options,profile:binding.target,localExecution:binding.localExecution};
  else {
    const backendRoot=await mkdtemp(join(tmpdir(),'generated-starter-target-')),backend=await target({root:backendRoot});
    t.after(async()=>{try{await backend.close();}finally{await rm(backendRoot,{recursive:true,force:true});}});
    options={mode:'cpl',target:'fixture',runtime:'cpu',workspacePath:backend.workspace,remoteDirectory:'owned',pollMs:1};client=backend.client;
  }
  const factory=live?(await import('../../integrations/environments/ascend.mjs')).createEnvironment:createEnvironment;
  const environment=await factory(options,{instancePath:join(f.config.root,'execution'),...(client?{client}:{}),
    exclude:async(_label,fn)=>fn(),event:()=>{}});
  f.state.add(imported);const {attempt}=new RecoveryQueue(f.state).claim(imported.id,1,{id:'test'});
  await environment.prepare({project:imported,attempt});
  const attemptPath=join(f.config.root,'preflight');await mkdir(attemptPath);
  await validateTargetBaseline({project:imported,state:f.state,attempt,attemptPath,
    execute:(...args)=>environment.execute(...args),checkVerifier:await freezeFiles(imported.verifier.files,imported.source,imported.verifier.hashes),
    remaining:()=>120,register:async()=>{}});
  assert.equal(f.state.queryEvents(attempt.id,{types:['baseline-preflight-completed']})[0].data.passed,true);
  if(environment.reconcileLease)assert.equal(await environment.reconcileLease(),true);
  t.diagnostic('Generated smoke target: '+(binding?.target??'CPU fixture')+'; artifacts: '+f.config.root);
  assert.ok(spec.generation.checks.some(check=>check.kind==='starter'&&check.exitCode===0));
  assert.ok(spec.generation.checks.some(check=>check.kind==='baseline'&&check.exitCode===0));
  assert.ok(imported.bundle.baseline_construction_command.includes(imported.bundle.root));
  await rm(join(imported.source,'src'),{recursive:true});
  await assert.rejects(validateBundle(imported,join(f.config.root,'broken-starter-proof')),/CPU starter failed/);
});


test('generated baseline CPU selftest rejects missing baseline-only dependencies before admission',async t=>{
  const f=await fixture(t,{brokenBaseline:true,performanceGoal:'timing',performanceCommand:'--performance'});
  assert.deepEqual(await refillProjects(f.options),[]);
  const rows=f.state.db.prepare("SELECT data FROM generation_events WHERE type='generation-rejected'").all();
  assert.ok(rows.length);for(const row of rows) {
    const checks=JSON.parse(row.data).checks;
    assert.ok(checks.some(check=>check.kind==='starter'&&check.exitCode===0));
    assert.equal(checks.at(-1).kind,'baseline');assert.equal(checks.at(-1).exitCode,1);
  }
});

test('new generated performance proposals require a baseline CPU selftest before execution',async t=>{
  const f=await fixture(t,{omitBaselineConstruction:true,performanceGoal:'timing',performanceCommand:'--performance'});
  assert.deepEqual(await refillProjects(f.options),[]);
  assert.equal(f.state.db.prepare("SELECT COUNT(*) AS n FROM generation_events WHERE type='generation-construction-check'").get().n,0);
});
