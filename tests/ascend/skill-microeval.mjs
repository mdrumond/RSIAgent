import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {runMicroeval,nativeSample} from '../../cli/skill-microeval.mjs';
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

async function scriptedMicro(t,root,plan){
 const server=createServer(async(req,res)=>{let raw='';for await(const chunk of req)raw+=chunk;const body=JSON.parse(raw),n=body.messages.filter(m=>m.role==='tool').length,tool=plan(n);
  res.writeHead(200,{'Content-Type':'text/event-stream'});const frame=value=>res.write('data: '+JSON.stringify(value)+'\n\n');
  frame({id:'scoped-micro',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:{role:'assistant',...(tool?{tool_calls:[{index:0,id:'call-'+n,type:'function',function:{name:tool.name,arguments:JSON.stringify(tool.args)}}]}:{content:'Done'})},finish_reason:null}]});
  frame({id:'scoped-micro',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:{},finish_reason:tool?'tool_calls':'stop'}],usage:{prompt_tokens:11,completion_tokens:7,total_tokens:18}});res.end('data: [DONE]\n\n');
 });server.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>{server.closeAllConnections();server.close();});
 const credentials_file=join(root,'scoped.env');await writeFile(credentials_file,'SCOPED_MICRO=fixture\n');
 return {provider:'scoped-micro',model:'scripted',credential_env:'SCOPED_MICRO',credentials_file,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:32000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
}

test('native microtask sample abort bounds an oversized Bash timeout and reaps its command', {timeout:20000,skip:process.platform!=='linux'},async t=>{
 const f=await fixture(t),profile=await scriptedMicro(t,f.root,n=>n===0?{name:'bash',args:{command:'printf "%s" $$ > command.pid; exec sleep 30',timeout:600}}:null),begun=Date.now();
 const report=await runMicroeval({...f,profile,runs:1,seconds:2});assert.ok(Date.now()-begun<10000);
 assert.ok(report.results.every(r=>r.timedOut&&!r.correct));
 for(const arm of ['active','candidate']){const pid=Number(await readFile(join(f.directory,'repair','0-'+arm,'work','command.pid'),'utf8'));assert.ok(pid>0);await assert.rejects(readFile('/proc/'+pid+'/stat'),{code:'ENOENT'});}
});

test('native microtask policy blocks outside files and shell workspace searches before execution',async t=>{
 const f=await fixture(t),outside=join(f.root,'outside.txt'),escaped=join(f.root,'escaped');await writeFile(outside,'unrelated');
 const plan=[{name:'read',args:{path:outside}},{name:'write',args:{path:outside,content:'overwritten'}},{name:'edit',args:{path:outside,oldText:'unrelated',newText:'changed'}},{name:'bash',args:{command:`cd '${f.root}'; grep -rl unrelated . | head; touch '${escaped}'`}},{name:'bash',args:{command:`grep -rl unrelated '${f.root}' | head; touch '${escaped}'`}},{name:'write',args:{path:'kernel.py',content:'def answer(): return 42\n'}}];
 const profile=await scriptedMicro(t,f.root,n=>plan[n]),report=await runMicroeval({...f,profile,runs:1});assert.ok(report.results.every(r=>r.correct));await assert.rejects(readFile(escaped),{code:'ENOENT'});assert.equal(await readFile(outside,'utf8'),'unrelated');
 for(const arm of ['active','candidate']){const blocks=(await readFile(join(f.directory,'repair','0-'+arm,'tool-policy.jsonl'),'utf8')).trim().split('\n').map(line=>JSON.parse(line));assert.deepEqual(blocks.map(b=>b.tool),['read','write','edit','bash','bash']);assert.ok(blocks.every(b=>b.blocked));}
});

test('bounded native Bash still executes the declared legitimate construction oracle',async t=>{
 const {constructionFixture}=await import('../../cli/microtasks/hypothesis-fixtures.mjs'),f=await fixture(t),{files,proposal}=constructionFixture();
 Object.assign(f.tasks[0].files,files,{'proposal.json':JSON.stringify(proposal),'oracle.mjs':`import {oracleMain} from ${JSON.stringify(new URL('../../cli/microtasks/construction-oracle.mjs',import.meta.url).href)}; await oracleMain(['--root',process.cwd(),'--proposal','proposal.json']);\n`});
 const plan=[{name:'bash',args:{command:`'${process.execPath}' oracle.mjs > oracle-result.json`,timeout:60}},{name:'write',args:{path:'kernel.py',content:'def answer(): return 42\n'}}],profile=await scriptedMicro(t,f.root,n=>plan[n]);
 const report=await runMicroeval({...f,profile,runs:1});assert.ok(report.results.every(r=>r.correct));
 for(const arm of ['active','candidate'])assert.equal(JSON.parse(await readFile(join(f.directory,'repair','0-'+arm,'work','oracle-result.json'),'utf8')).admitted,true);
});

test('native sample elapsed cancellation also bounds delayed MCP startup',{timeout:10000},async t=>{
 const f=await fixture(t),profile=await scriptedMicro(t,f.root,()=>null);let requests=0;
 const server=createServer(async(req,res)=>{for await(const _ of req){}requests++;});server.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>{server.closeAllConnections();server.close();});
 const cwd=join(f.root,'work'),sessionDirectory=join(f.root,'session');await mkdir(cwd);await mkdir(sessionDirectory);
 const start=performance.now();const report=await nativeSample({cwd,sessionDirectory,profile,seconds:.3,guidance:'fixture',prompt:'finish',knowledge:{url:`http://127.0.0.1:${server.address().port}/mcp`}});
 assert.equal(report.timedOut,true);assert.ok(requests>0);assert.ok(performance.now()-start<3000);
});

test('stable retrieval identity permits local endpoint reconnect but rejects a different snapshot',async t=>{
 const f=await fixture(t);let calls=0;
 const first={url:'http://127.0.0.1:1234/mcp',identity:'frozen-memory-snapshot'};
 await runMicroeval({...f,knowledge:first},{sample:async({cwd,knowledge})=>{calls++;assert.equal(knowledge.url,first.url);await writeFile(join(cwd,'kernel.py'),'def answer(): return 42\n');return {};}});
 const report=await runMicroeval({...f,knowledge:{...first,url:'http://127.0.0.1:5678/mcp'},resume:true},{sample:()=>{throw Error('must not replay');}});
 assert.equal(report.status,'complete');assert.equal(calls,4);assert.deepEqual(report.knowledge,{readOnly:true,identity:first.identity});
 await assert.rejects(runMicroeval({...f,knowledge:{...first,identity:'different-snapshot'},resume:true},{sample:()=>{throw Error('must not dispatch');}}),/Frozen micro/);
 const legacy=await fixture(t);await runMicroeval({...legacy,knowledge:{url:first.url}},{sample:async()=>({})});
 await assert.rejects(runMicroeval({...legacy,knowledge:{url:'http://127.0.0.1:5678/mcp'},resume:true}),/Frozen micro/);
});
test('native Pi keeps local edits/bash and dispatches device commands through existing rsi_target',async t=>{
 const f=await fixture(t);const profile=await scriptedMicro(t,f.root,n=>[
  {name:'write',args:{path:'kernel.py',content:'def answer(): return 42\n'}},
  {name:'bash',args:{command:'printf local > local-marker'}},
  {name:'rsi_target',args:{command:'python check.py .',timeout:600}}
 ][n]);
 let remote=0;const report=await runMicroeval({...f,profile,runs:1},{sample:args=>nativeSample({...args,targetEnvironment:{execute:async(command,cwd,options)=>{
  remote++;assert.equal(command,'python check.py .');assert.equal(cwd,args.cwd);assert.ok(options.timeout<=args.seconds);assert.ok(options.signal instanceof AbortSignal);options.onData?.(Buffer.from('remote checked\n'));return {handle:'remote:fixture:job:'+remote,settled:true,exitCode:0};
 }}})});
 assert.equal(remote,2);assert.ok(report.results.every(r=>r.correct));for(const arm of ['active','candidate'])assert.equal(await readFile(join(f.directory,'repair','0-'+arm,'work/local-marker'),'utf8'),'local');
});
test('elapsed allowance covers adapter preparation and trusted verification hooks',async t=>{
 const f=await fixture(t);let verified=0;
 const report=await runMicroeval({...f,runs:1,seconds:.05},{sample:async({signal})=>{await new Promise(resolve=>{const timer=setTimeout(resolve,1000);signal.addEventListener('abort',()=>{clearTimeout(timer);resolve();},{once:true});});return {};},verify:async()=>{verified++;return {correct:true};}});
 assert.equal(verified,0);assert.ok(report.results.every(r=>r.correct===false&&r.timedOut));
 const g=await fixture(t);const measured=await runMicroeval({...g,runs:1},{sample:async()=>({}),verify:async({cwd,seconds,signal})=>{assert.ok(seconds>0&&seconds<=g.seconds);assert.ok(signal instanceof AbortSignal);await writeFile(join(cwd,'kernel.py'),'def answer(): return 42\n');return {correct:true,exitCode:0,output:'trusted target check'};}});
 assert.ok(measured.results.every(r=>r.correct&&r.verification.output==='trusted target check'));
});
test('unresolved actor and verifier commands retain identity and stop before the next arm',async t=>{
 for(const phase of ['sample','verify']){
  const f=await fixture(t),handle='remote:fixture:job:unknown-'+phase;let calls=0;
  const dependencies={sample:async()=>{calls++;return phase==='sample'?{unresolvedExecution:{handle},timedOut:true}:{};},verify:async()=>{throw Object.assign(Error('Target command unresolved'),{code:'RSI_EXECUTION_UNRESOLVED',handle});}};
  await assert.rejects(runMicroeval({...f,runs:1,resume:true},dependencies),{code:'RSI_EXECUTION_UNRESOLVED'});
  const retained=JSON.parse(await readFile(join(f.directory,'results.json'),'utf8'));
  assert.equal(calls,1);assert.equal(retained.status,'interrupted');assert.equal(retained.results.length,0);
  assert.equal(retained.runningSample.unresolvedExecution.handle,handle);
  await assert.rejects(runMicroeval({...f,runs:1,resume:true},dependencies),/unresolved/);assert.equal(calls,1);
 }
});
test('native target unknown status survives session abort as structured evidence',async t=>{
 const f=await fixture(t),profile=await scriptedMicro(t,f.root,n=>n===0?{name:'rsi_target',args:{command:'python check.py .'}}:null);
 const cwd=join(f.root,'work'),sessionDirectory=join(f.root,'session');await mkdir(cwd);await mkdir(sessionDirectory);
 const handle='remote:fixture:job:unknown-native';
 const result=await nativeSample({cwd,sessionDirectory,profile,seconds:5,guidance:'fixture',prompt:'check',targetEnvironment:{execute:async()=>({handle,settled:false,exitCode:null})}});
 assert.deepEqual(result.unresolvedExecution,{handle});
});
