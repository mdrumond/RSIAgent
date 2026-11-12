import {controls,numericControls} from './investigation-fixtures.mjs';
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
import {State} from '../../controller/state.mjs';
import {findingsFor} from '../../cli/findings.mjs';

async function fixture(t,plan) {
  const root=await mkdtemp(join(tmpdir(),'investigation-admission-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const path=join(root,'attempt'),source=join(root,'original');
  await mkdir(path);await mkdir(source);await numericControls(join(path,'investigation'));
  await writeFile(join(source,'answer.py'),'answer=42\n');
  const received=[];
  const server=createServer(async(req,res)=>{
    let raw='';for await(const chunk of req)raw+=chunk;
    const body=JSON.parse(raw),results=body.messages.filter(m=>m.role==='tool');
    received.push(results);
    const call=plan[results.length];
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
  state.add({id:'p',source,goal:'numeric',actor_seconds:120});
  const attempt=state.admit('p',{}),bridge={extension:()=>{},knowledge:{search:async({kind})=>({content:[{kind}],evidence:[]})}};
  const options={config,path,state,bridge,job:{origin_attempt:attempt.id,attempt,actor_seconds:120},base:{source,goal:'numeric'}};
  return {root,path,source,options,results:()=>received.at(-1)};
}
const proposal={goal:'Check exact numeric answer',question:'Does the implementation return42?',verifier:'verify.py',controls};

test('native submit rejects missing verifier then accepts write and resubmit in the same session',{timeout:15000},async t=>{
  const verifier="import runpy\nassert runpy.run_path('answer.py')['answer']==42\nprint(42)\n";
  const f=await fixture(t,[['submit_investigation',proposal],['write',{path:'verify.py',content:verifier}],['investigation_knowledge',{query:'numeric control',kind:'good'}],['investigation_knowledge',{query:'numeric mismatch',kind:'negative'}],['submit_investigation',proposal]]);
  const result=await investigate(f.options),responses=f.results();
  assert.equal(responses.length,5);
  assert.match(responses[0].content,/Artifact is missing/);
  assert.match(responses[0].content,/investigation\/verify.py/);
  assert.equal(responses[4].content,'Recorded');
  assert.equal(result.verifier.files[0],join(f.path,'investigation','verify.py'));
  assert.equal(await readFile(result.verifier.files[0],'utf8'),verifier);
  assert.equal(await readFile(join(f.source,'answer.py'),'utf8'),'answer=42\n');
  const run=await promisify(execFile)(process.env.PYTHON??'python3',['-c',"import runpy; runpy.run_path('../verify.py')"],{cwd:result.source});
  assert.equal(Number(run.stdout.trim()),42);
  assert.equal((await readdir(join(f.path,'investigation','sessions'))).filter(x=>x.endsWith('.jsonl')).length,1);
});

test('construction from a bundled parent owns only the new investigation source and verifier',{timeout:15000},async t=>{
  const verifier="import runpy\nassert runpy.run_path('answer.py')['answer']==42\nprint(42)\n";
  const f=await fixture(t,[['write',{path:'verify.py',content:verifier}],
    ['investigation_knowledge',{query:'numeric control',kind:'good'}],
    ['investigation_knowledge',{query:'numeric mismatch',kind:'negative'}],['submit_investigation',proposal]]);
  Object.assign(f.options.base,{id:'parent',phase:'phase1',campaign_entry:'batch1',sourceDirectory:f.source,
    performance_goal:{kind:'speedup'},bundle:{version:1,root:f.source,reference:f.source,
      negative_controls:[],construction_command:'parent-construction',seconds:30}});
  const before=structuredClone(f.options.base),result=await investigate(f.options);
  assert.equal(result.source,join(f.path,'investigation','source'));
  assert.equal(result.bundle,undefined);assert.equal(result.sourceDirectory,undefined);
  assert.equal(result.performance_goal,undefined);
  assert.equal(result.id,'parent');assert.equal(result.phase,'phase1');assert.equal(result.campaign_entry,'batch1');
  assert.deepEqual(f.options.base,before);
  assert.deepEqual(result.verifier.files,[join(f.path,'investigation','verify.py')]);
  const run=await promisify(execFile)(process.env.PYTHON??'python3',
    ['-c',"import runpy; runpy.run_path('../verify.py')"],{cwd:result.source});
  assert.equal(Number(run.stdout.trim()),42);
});

for(const verifier of ['sessions','source','source/answer.py','../outside.py'])
  test(`invalid native verifier ${verifier} cannot admit an actor`,{timeout:15000},async t=>{
    const f=await fixture(t,[['submit_investigation',{...proposal,verifier}]]);
    await assert.rejects(investigate(f.options),/did not submit an investigation/);
    assert.notEqual(f.results()[0].content,'Recorded');
    await assert.rejects(stat(join(f.path,'project.json')),{code:'ENOENT'});
    assert.equal(await readFile(join(f.source,'answer.py'),'utf8'),'answer=42\n');
  });

test('native summary rejects directory evidence then accepts a real file without changing it',{timeout:15000},async t=>{
  const record={findings:'The proposal failed before execution; no numeric conclusion.',useful:false,paths:['.']};
  const text='missing independent verifier observed\n';
  const f=await fixture(t,[['record_findings',record],['write',{path:'evidence.log',content:text}],['record_findings',{...record,paths:['evidence.log']}]]);
  const result=await summarize({...f.options,passed:false});
  assert.match(f.results()[0].content,/regular file/);
  assert.equal(f.results()[2].content,'Recorded');
  assert.equal(result.useful,false);
  assert.deepEqual(result.evidence,[{path:join(f.path,'evidence.log'),sha256:createHash('sha256').update(text).digest('hex')}]);
  assert.equal(await readFile(join(f.path,'evidence.log'),'utf8'),text);
});


test('pre-actor failure has readable diagnostics before native summary repairs directory evidence',{timeout:15000},async t=>{
  const record={findings:'Missing verifier prevented actor startup; no experiment ran.',useful:false,paths:['.']};
  const f=await fixture(t,[['record_findings',record],['read',{path:'diagnostics.json'}],['record_findings',{...record,paths:['diagnostics.json']}]]);
  const failure={type:'attempt-error',data:{message:'ENOENT missing verifier',stage:'worker'}};
  f.options.state.event(f.options.job.attempt.id,failure.type,failure.data);
  const result=await findingsFor({...f.options,passed:false});
  assert.match(f.results()[0].content,/regular file/);
  assert.deepEqual(JSON.parse(f.results()[1].content),[failure]);
  assert.equal(f.results()[2].content,'Recorded');
  assert.equal(result.useful,false);
  assert.equal(result.evidence[0].path,join(f.path,'diagnostics.json'));
  assert.equal(result.evidence[0].sha256,createHash('sha256').update(await readFile(result.evidence[0].path)).digest('hex'));
  assert.deepEqual(JSON.parse(await readFile(join(f.path,'findings.json'),'utf8')),result);
});

const knowledge=[['investigation_knowledge',{query:'numeric control',kind:'good'}],
  ['investigation_knowledge',{query:'numeric mismatch',kind:'negative'}]];

test('canonical argparse verifier rejects missing argv and accepts corrected submission in the same Pi session',async t=>{
  const filename="probe 'one'.json";
  const verifier="import argparse,json\np=argparse.ArgumentParser();p.add_argument('--result',required=True)\na=p.parse_args()\nraise SystemExit(0 if json.load(open(a.result))['answer']==42 else 1)\n";
  const corrected={...proposal,verifier_args:['--result',filename]};
  const f=await fixture(t,[['write',{path:'verify.py',content:verifier}],...knowledge,
    ['submit_investigation',proposal],['submit_investigation',corrected]]);
  for(const [name,answer] of [['positive',42],['negative',0]])
    await writeFile(join(f.path,'investigation',name,filename),JSON.stringify({answer}));
  const result=await investigate(f.options),responses=f.results();
  assert.match(responses[3].content,/reference failed/);
  assert.equal(responses[4].content,'Recorded');
  const events=f.options.state.queryEvents(f.options.job.attempt.id,{types:['curriculum-proposal']});
  assert.equal(events.length,1);assert.deepEqual(events[0].data.proposal,{...corrected,verifier_command:result.verifier.command});
  const run=cwd=>promisify(execFile)('bash',['-c',result.verifier.command],{cwd});
  await run(join(f.path,'investigation','positive'));
  await assert.rejects(run(join(f.path,'investigation','negative')),error=>error.code===1);
  await writeFile(join(result.source,filename),JSON.stringify({answer:42}));
  await run(result.source);
  await writeFile(join(f.path,'investigation','positive',filename),'not JSON');
  f.options.job.resume=true;
  const resumed=await investigate(f.options);
  assert.equal(resumed.verifier.command,result.verifier.command);
  assert.deepEqual(f.options.state.queryEvents(f.options.job.attempt.id,{types:['curriculum-proposal']}),events);
  assert.equal((await readdir(join(f.path,'investigation','sessions'))).filter(x=>x.endsWith('.jsonl')).length,1);
});

test('positive success alone cannot admit an always-pass investigation verifier',async t=>{
  const f=await fixture(t,[['write',{path:'verify.py',content:'raise SystemExit(0)\n'}],...knowledge,
    ['submit_investigation',proposal]]);
  await assert.rejects(investigate(f.options),/did not submit an investigation/);
  assert.match(f.results()[3].content,/negative control failed/);
  assert.equal(f.options.state.queryEvents(f.options.job.attempt.id,{types:['curriculum-proposal']}).length,0);
});

for(const invalid of [{controls:undefined},{controls:{positive:'source',negative:['negative']}},
  {controls:{positive:'positive',negative:['positive']}}])
  test('fresh investigation refuses missing or non-discriminating fixture ownership '+JSON.stringify(invalid),async t=>{
    const f=await fixture(t,[['write',{path:'verify.py',content:'raise SystemExit(0)\n'}],...knowledge,
      ['submit_investigation',{...proposal,...invalid}]]);
    await assert.rejects(investigate(f.options),/did not submit an investigation/);
    assert.equal(f.options.state.queryEvents(f.options.job.attempt.id,{types:['curriculum-proposal']}).length,0);
  });

for(const argv of [undefined,['--result',"probe 'one'.json"]])
  test('legacy persisted investigation resumes without new controls or provider work '+JSON.stringify(argv),async t=>{
    const f=await fixture(t,[]),root=join(f.path,'investigation');
    await writeFile(join(root,'verify.py'),'raise SystemExit(0)\n');
    const old={goal:proposal.goal,question:proposal.question,verifier:'verify.py',...(argv?{verifier_args:argv}:{})};
    const id=f.options.job.attempt.id;
    await f.options.state.event(id,'curriculum-proposal',{proposal:old});
    const before=f.options.state.queryEvents(id,{types:['curriculum-proposal']});
    f.options.job.resume=true;
    const result=await investigate(f.options);
    assert.deepEqual(f.options.state.queryEvents(id,{types:['curriculum-proposal']}),before);
    assert.equal(f.results(),undefined);
    assert.equal(result.verifier.command,"python '../verify.py'");
  });


test('control execution consumes the existing curriculum elapsed allocation before admission',async t=>{
  const verifier="import time\ntime.sleep(30)\nraise SystemExit(0)\n";
  const f=await fixture(t,[['write',{path:'verify.py',content:verifier}],...knowledge,
    ['submit_investigation',proposal]]);
  f.options.config.curriculum_seconds=.8;
  const started=performance.now();
  await assert.rejects(investigate(f.options),/elapsed allocation exhausted/);
  assert.ok((performance.now()-started)/1000<5);
  const id=f.options.job.attempt.id;
  assert.equal(f.options.state.queryEvents(id,{types:['curriculum-proposal']}).length,0);
  const attempt=f.options.state.attempt(id);
  assert.equal(attempt.elapsed,0);assert.equal(attempt.session,null);
  assert.equal(f.options.state.queryEvents(id,{types:['curriculum-interrupted']})[0].data.accepted,false);
});
