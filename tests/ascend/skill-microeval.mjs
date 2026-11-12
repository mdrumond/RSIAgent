import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {runMicroeval} from '../../cli/skill-microeval.mjs';
import {tasks} from '../../examples/skill-microeval/tasks.mjs';
import {recoveryRepair} from './microtask-fixture.mjs';
const python=process.env.RSI_TEST_PYTHON??'python3';
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'microeval-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const skills={};for(const arm of ['active','candidate']){const path=join(root,arm);await mkdir(path);await writeFile(join(path,'skill-map.json'),JSON.stringify({implementation:'mapped.md'}));await writeFile(join(path,'mapped.md'),arm+' guidance');skills[arm]=path;}
 const checker=join(root,'check.py');await writeFile(checker,"import sys,runpy,pathlib\nm=runpy.run_path(str(pathlib.Path(sys.argv[1])/'kernel.py'))\nassert m['answer']()==42\n");
 return {root,skills,tasks:[{id:'repair',scopes:['implementation'],prompt:'Repair answer()',files:{'kernel.py':'def answer(): return 0\n'},checker}],profile:{provider:'fixture',model:'scripted'},python,runs:2,seconds:20,directory:join(root,'output')};
}
test('paired shadow preserves balanced order and independently checks actual repaired files',async t=>{
 const f=await fixture(t),guidance=[];
 const report=await runMicroeval(f,{sample:async({cwd,guidance:given})=>{guidance.push(given);await writeFile(join(cwd,'kernel.py'),'def answer(): return 42\n');return {tokens:11,cost:0};}});
 assert.deepEqual(report.results.map(x=>x.arm),['active','candidate','candidate','active']);assert.ok(report.results.every(x=>x.correct&&x.elapsedSeconds>0&&x.tokens===11));
 assert.deepEqual(guidance,['## implementation\nactive guidance','## implementation\ncandidate guidance','## implementation\ncandidate guidance','## implementation\nactive guidance']);
 assert.equal(report.status,'complete');assert.equal(JSON.parse(await readFile(join(f.directory,'results.json'))).results.length,4);
 assert.equal(await readFile(join(f.skills.active,'mapped.md'),'utf8'),'active guidance');
 await assert.rejects(runMicroeval(f),{code:'EEXIST'});
});
test('wrong implementations and exhausted samples cannot pass',async t=>{
 const f=await fixture(t);let calls=0;
 const report=await runMicroeval({...f,runs:1},{sample:async()=>({timedOut:calls++===1,tokens:null})});
 assert.deepEqual(report.results.map(x=>x.correct),[false,false]);assert.equal(report.results[1].tokens,null);
});
test('interruption persists partial evidence without promoting or refilling',async t=>{
 const f=await fixture(t),abort=new AbortController();
 await assert.rejects(runMicroeval({...f,signal:abort.signal},{sample:async()=>{abort.abort();return {tokens:null};}}),{name:'AbortError'});
 const report=JSON.parse(await readFile(join(f.directory,'results.json')));assert.equal(report.status,'interrupted');assert.equal(report.results.length,1);
});
test('provider failures remain unknown rather than correctness regressions',async t=>{
 const f=await fixture(t),report=await runMicroeval({...f,runs:1},{sample:async()=>{throw Error('Provider unavailable');}});
 assert.deepEqual(report.results.map(x=>x.correct),[null,null]);assert.ok(report.results.every(x=>x.verification.error==='Provider unavailable'));
});
test('global elapsed deadline stops dispatch and retains partial results',async t=>{
 const f=await fixture(t);let calls=0;
 const report=await runMicroeval({...f,totalSeconds:.03},{sample:async()=>{calls++;await new Promise(r=>setTimeout(r,50));return {tokens:null};}});
 assert.equal(calls,1);assert.equal(report.status,'elapsed');assert.equal(report.results.length,1);assert.equal(report.results[0].correct,false);
 const frozen=JSON.parse(await readFile(join(f.directory,'frozen/tasks.json')));assert.equal(frozen[0].files['kernel.py'],'def answer(): return 0\n');
});
test('preplanned contract checks reject broken source and accept independent repairs',async t=>{
 const f=await fixture(t),fixed={
 'argument-cache':'def make_launcher(compile_kernel):\n cache={}\n def launch(x,scale):\n  n=len(x)\n  if n not in cache: cache[n]=compile_kernel(n)\n  return cache[n](x,scale)\n return launch\n',
 'ragged-store':'def add_rows(storage,rows,columns,pitch,delta):\n out=list(storage)\n for r in range(rows):\n  for c in range(columns):out[r*pitch+c]+=delta\n return out\n',
 'strided-view':'import itertools\ndef logical_flatten(storage,shape,strides,offset=0):\n return [storage[offset+sum(i*s for i,s in zip(index,strides))] for index in itertools.product(*(range(n) for n in shape))]\n',
 'recovery-decisions':recoveryRepair};
 for(const task of tasks){const work=join(f.root,task.id);await mkdir(work);
  for(const [name,text]of Object.entries(task.files))await writeFile(join(work,name),text);
  assert.notEqual(spawnSync(python,[task.checker,work]).status,0);
  await writeFile(join(work,'kernel.py'),fixed[task.id]);const checked=spawnSync(python,['-B',task.checker,work],{encoding:'utf8'});assert.equal(checked.status,0,task.id+': '+checked.stderr);
 }
});
test('native Pi tools edit files, retain sessions and record provider usage', {timeout:60000},async t=>{
 const f=await fixture(t),requests=[];
 const server=createServer(async(req,res)=>{
  let raw='';for await(const x of req)raw+=x;const body=JSON.parse(raw);requests.push(body);
  const used=body.messages.some(x=>x.role==='tool');res.writeHead(200,{'Content-Type':'text/event-stream'});
  const frame=value=>res.write('data: '+JSON.stringify(value)+'\n\n');
  frame({id:'microeval-fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:{role:'assistant',...(used?{content:'Repaired.'}:{tool_calls:[{index:0,id:'write-answer',type:'function',function:{name:'write',arguments:JSON.stringify({path:'kernel.py',content:'def answer(): return 42\n'})}}]})},finish_reason:null}]});
  frame({id:'microeval-fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:{},finish_reason:used?'stop':'tool_calls'}],usage:{prompt_tokens:11,completion_tokens:7,total_tokens:18}});res.end('data: [DONE]\n\n');
 });server.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>{server.closeAllConnections();server.close();});
 const credentials_file=join(f.root,'fixture.env');await writeFile(credentials_file,'MICROEVAL_FIXTURE=fixture\n');
 const profile={provider:'microeval-fixture',model:'scripted',credential_env:'MICROEVAL_FIXTURE',credentials_file,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:32000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
 const report=await runMicroeval({...f,profile,runs:1});assert.equal(requests.length,4);
 assert.ok(report.results.every(x=>x.correct&&x.toolCalls===1&&x.tokens===36));
 for(const result of report.results){assert.ok((await readFile(result.session,'utf8')).length>0);}
});
test('default scopes require matching guidance in both arms before dispatch',async t=>{
 const f=await fixture(t);await writeFile(join(f.skills.candidate,'skill-map.json'),JSON.stringify({debug:'mapped.md'}));
 const task={...f.tasks[0]};delete task.scopes;let calls=0;
 await assert.rejects(runMicroeval({...f,tasks:[task]},{sample:async()=>{calls++;return {};}}),/Unknown skill scope/);
 assert.equal(calls,0);await assert.rejects(readFile(join(f.directory,'manifest.json')),{code:'ENOENT'});
});
test('native provider HTTP authentication failure is inconclusive with diagnostic', {timeout:60000},async t=>{
 const f=await fixture(t);let requests=0;
 const server=createServer(async(req,res)=>{for await(const _ of req){}requests++;res.writeHead(401,{'Content-Type':'application/json'});res.end(JSON.stringify({error:{message:'Fixture authentication rejected',type:'authentication_error'}}));});
 server.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>{server.closeAllConnections();server.close();});
 const credentials_file=join(f.root,'fixture.env');await writeFile(credentials_file,'MICROEVAL_FIXTURE=fixture\n');
 const profile={provider:'microeval-http-error',model:'scripted',credential_env:'MICROEVAL_FIXTURE',credentials_file,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:32000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
 const report=await runMicroeval({...f,profile,runs:1});assert.equal(requests,2);
 assert.deepEqual(report.results.map(x=>x.correct),[null,null]);assert.ok(report.results.every(x=>x.error.includes('Fixture authentication rejected')));
 assert.ok(report.results.every(x=>x.verification.exitCode===undefined));
});
