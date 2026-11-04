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

async function fixture(t,plan) {
  const root=await mkdtemp(join(tmpdir(),'investigation-admission-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const path=join(root,'attempt'),source=join(root,'original');
  await mkdir(path);await mkdir(source);
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
  const state={events:()=>[],attempt:()=>({session:null}),event:()=>{}},bridge={extension:()=>{}};
  const options={config,path,state,bridge,job:{origin_attempt:'old',attempt:{id:'new'},actor_seconds:120},base:{source,goal:'numeric'}};
  return {root,path,source,options,results:()=>received.at(-1)};
}
const proposal={goal:'Check exact numeric answer',question:'Does the implementation return42?',verifier:'verify.py'};

test('native submit rejects missing verifier then accepts write and resubmit in the same session',{timeout:15000},async t=>{
  const verifier="import runpy\nassert runpy.run_path('answer.py')['answer']==42\nprint(42)\n";
  const f=await fixture(t,[['submit_investigation',proposal],['write',{path:'verify.py',content:verifier}],['submit_investigation',proposal]]);
  const result=await investigate(f.options),responses=f.results();
  assert.equal(responses.length,3);
  assert.match(responses[0].content,/Artifact is missing/);
  assert.match(responses[0].content,/investigation\/verify.py/);
  assert.equal(responses[2].content,'Recorded');
  assert.equal(result.verifier.files[0],join(f.path,'investigation','verify.py'));
  assert.equal(await readFile(result.verifier.files[0],'utf8'),verifier);
  assert.equal(await readFile(join(f.source,'answer.py'),'utf8'),'answer=42\n');
  const run=await promisify(execFile)(process.env.PYTHON??'python3',['-c',"import runpy; runpy.run_path('../verify.py')"],{cwd:result.source});
  assert.equal(Number(run.stdout.trim()),42);
  assert.equal((await readdir(join(f.path,'investigation','sessions'))).filter(x=>x.endsWith('.jsonl')).length,1);
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
  f.options.state={events:()=>[failure],attempt:()=>({session:null}),event:()=>{}};
  const result=await findingsFor({...f.options,passed:false});
  assert.match(f.results()[0].content,/regular file/);
  assert.deepEqual(JSON.parse(f.results()[1].content),[failure]);
  assert.equal(f.results()[2].content,'Recorded');
  assert.equal(result.useful,false);
  assert.equal(result.evidence[0].path,join(f.path,'diagnostics.json'));
  assert.equal(result.evidence[0].sha256,createHash('sha256').update(await readFile(result.evidence[0].path)).digest('hex'));
  assert.deepEqual(JSON.parse(await readFile(join(f.path,'findings.json'),'utf8')),result);
});
