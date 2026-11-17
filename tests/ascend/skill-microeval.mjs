import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve,dirname} from 'node:path';
import {spawnSync} from 'node:child_process';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {runMicroeval,nativeSample} from '../../cli/skill-microeval.mjs';
import {tasks} from '../../examples/skill-microeval/tasks.mjs';
import {recoveryRepair} from './microtask-fixture.mjs';
const python=process.env.RSI_TEST_PYTHON??'python3';
function untilAbort(signal){return new Promise(resolve=>{const timer=setTimeout(resolve,2000),done=()=>{clearTimeout(timer);resolve();};if(signal.aborted)done();else signal.addEventListener('abort',done,{once:true});});}
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
 const profile={provider:'microeval-fixture',model:'scripted',credential_env:'MICROEVAL_FIXTURE',credentials_file,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:1000000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
 const report=await runMicroeval({...f,profile,runs:1});assert.equal(requests.length,4);
 assert.ok(report.results.every(x=>x.correct&&x.toolCalls===1&&x.tokens===36));
 assert.equal(report.model.effectiveContextTokens,200000);
 for(const result of report.results){assert.ok((await readFile(result.session,'utf8')).length>0);const startup=JSON.parse(await readFile(join(f.directory,result.taskId,`${result.run}-${result.arm}`,'model-limits.json'),'utf8'));assert.equal(startup.contextWindow,200000);assert.equal(startup.maxTokens,4096);}
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
 for(const phase of ['sample','verify','legacy-prepare','legacy-verify']){
  const f=await fixture(t),handle='remote:fixture:job:unknown-'+phase;let calls=0;
  const dependencies={sample:async()=>{calls++;if(phase==='legacy-prepare')throw Error('Copy unresolved: '+handle);return phase==='sample'?{unresolvedExecution:{handle},timedOut:true}:{};},verify:async()=>{if(phase==='legacy-verify')throw Object.assign(Error('Local execution unresolved'),{handle});throw Object.assign(Error('Target command unresolved'),{code:'RSI_EXECUTION_UNRESOLVED',handle});}};
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
test('native target legacy setup uncertainty stops later tool dispatch',async t=>{
 for(const message of ['Source synchronization unresolved; observe remote:fixture:job:source','Probe submission identity unknown; do not resubmit']){
  const f=await fixture(t),profile=await scriptedMicro(t,f.root,()=>({name:'rsi_target',args:{command:'python check.py .'}}));
  const cwd=join(f.root,'work'),sessionDirectory=join(f.root,'session');await mkdir(cwd);await mkdir(sessionDirectory);let calls=0;
  const result=await nativeSample({cwd,sessionDirectory,profile,seconds:5,guidance:'fixture',prompt:'check',targetEnvironment:{execute:async()=>{calls++;throw Error(message);}}});
  assert.equal(calls,1);assert.deepEqual(result.unresolvedExecution,{handle:message.includes('remote:')?'remote:fixture:job:source':null});
 }
});
test('actor cutoff reserves independent verification within the same elapsed sample',async t=>{
 for(const answer of [42,0]){
  const f=await fixture(t);let calls=0;
  const options={...f,runs:1,seconds:.6,verificationSeconds:.3};
  const report=await runMicroeval(options,{sample:async({cwd,signal,seconds,verificationReserved})=>{
   calls++;assert.ok(seconds<=.3&&verificationReserved>0);await writeFile(join(cwd,'kernel.py'),`def answer(): return ${answer}\n`);
   await untilAbort(signal);
   return {timedOut:true,error:'Actor elapsed allocation exhausted',tokens:7};
  }});
  assert.equal(calls,2);assert.ok(report.results.every(r=>r.correct===(answer===42)&&r.actorTimedOut&&r.actorError&&!r.timedOut&&r.tokens===7));
  assert.ok(report.results.every(r=>r.elapsedSeconds<.6));
  await assert.rejects(runMicroeval({...options,resume:true,verificationSeconds:.2}),/Frozen micro/);
 }
});
test('verification reserve never extends the whole sample deadline or hides cancellation',async t=>{
 const f=await fixture(t);let verifications=0;
 const report=await runMicroeval({...f,runs:1,seconds:.15,verificationSeconds:.07},{sample:async()=>({}),verify:async({signal})=>{
  verifications++;await untilAbort(signal);return {correct:true};
 }});
 assert.equal(verifications,2);assert.ok(report.results.every(r=>!r.correct&&r.timedOut));
 const g=await fixture(t),abort=new AbortController();let checked=0;
 await assert.rejects(runMicroeval({...g,runs:1,verificationSeconds:5,signal:abort.signal,resume:true},{sample:async()=>{abort.abort();return {timedOut:true};},verify:async()=>{checked++;return {correct:true};}}));
 assert.equal(checked,0);
});
test('resuming interrupted reserved verification does not allocate another actor turn',async t=>{
 const f=await fixture(t),abort=new AbortController();let sampled=0,verified=0;
 const options={...f,runs:1,seconds:2,verificationSeconds:1,resume:true};
 const dependencies={sample:async({cwd})=>{sampled++;await writeFile(join(cwd,'kernel.py'),'def answer(): return 42\n');return {};},verify:async()=>{if(++verified===1)abort.abort();return {correct:true};}};
 await assert.rejects(runMicroeval({...options,signal:abort.signal},dependencies));
 const interrupted=JSON.parse(await readFile(join(f.directory,'results.json'),'utf8'));
 assert.equal(interrupted.partials['repair/0/active'].actorCompleted,true);assert.equal(sampled,1);
 const result=await runMicroeval(options,dependencies);
 assert.equal(sampled,2);assert.equal(verified,3);assert.ok(result.results.every(r=>r.correct));
});
test('an inner actor timeout before the outer actor timer still receives reserved verification',async t=>{
 for(const answer of [42,0]){
  const f=await fixture(t);let checked=0;
  const result=await runMicroeval({...f,runs:1,seconds:1,verificationSeconds:.5},{sample:async({cwd,signal})=>{
   assert.equal(signal.aborted,false);await writeFile(join(cwd,'kernel.py'),`def answer(): return ${answer}\n`);
   return {timedOut:true,error:'Inner actor timer elapsed'};
  },verify:async({cwd})=>{checked++;return {correct:spawnSync(f.python,[f.tasks[0].checker,cwd],{timeout:1000}).status===0};}});
  assert.equal(checked,2);assert.ok(result.results.every(r=>r.correct===(answer===42)&&r.actorTimedOut&&!r.timedOut));
 }
});
test('resume preserves remaining actor time instead of splitting the remaining sample again',async t=>{
 const f=await fixture(t),abort=new AbortController(),options={...f,runs:1,seconds:1,verificationSeconds:.5,resume:true};
 await assert.rejects(runMicroeval({...options,signal:abort.signal},{sample:async()=>{await new Promise(resolve=>setTimeout(resolve,40));abort.abort();return {};}}));
 const retained=JSON.parse(await readFile(join(f.directory,'results.json'),'utf8')),spent=retained.partials['repair/0/active'].elapsedSeconds;
 const result=await runMicroeval(options,{sample:async({cwd,seconds})=>{if(cwd.endsWith('0-active/work'))assert.ok(seconds+spent<=.501);await writeFile(join(cwd,'kernel.py'),'def answer(): return 42\n');return {};}});
 assert.ok(result.results.every(r=>r.correct));
});
test('an exhausted interrupted actor proceeds to verification without another actor turn',async t=>{
 const f=await fixture(t),abort=new AbortController(),options={...f,runs:1,seconds:1,verificationSeconds:.5,resume:true};let sampled=0;
 const sample=async({cwd,signal})=>{sampled++;await writeFile(join(cwd,'kernel.py'),'def answer(): return 42\n');if(sampled===1){await untilAbort(signal);abort.abort();return {timedOut:true,error:'Interrupted after actor cutoff'};}return {};};
 await assert.rejects(runMicroeval({...options,signal:abort.signal},{sample}));
 const result=await runMicroeval(options,{sample});assert.equal(sampled,2);assert.ok(result.results.every(r=>r.correct));
});
test('blocked device admission retains reservations and stops before spending another sample',async t=>{
 const reservations=[{physicalDevice:5,token:'retained-token',receipt:'/retained/lease.json',reason:'submission-unknown',closed:false}];
 for(const phase of ['sample','verify','observed']){
  const f=await fixture(t);let calls=0,checks=0;
  const blocked=()=>Object.assign(Error('Device reservations require reconciliation'),{code:'DEVICE_LEASE_BLOCKED',reservations});
  const dependencies={sample:async()=>{calls++;if(phase==='sample')throw blocked();return phase==='observed'?{unresolvedExecution:{handle:null,reason:'device-lease-blocked',reservations}}:{};},verify:async()=>{checks++;throw blocked();}};
  await assert.rejects(runMicroeval({...f,resume:true},dependencies),{code:'RSI_EXECUTION_UNRESOLVED'});
  const report=JSON.parse(await readFile(join(f.directory,'results.json')));
  assert.equal(calls,1);assert.equal(checks,phase==='verify'?1:0);assert.equal(report.results.length,0);
  assert.deepEqual(report.runningSample.unresolvedExecution,{handle:null,reason:'device-lease-blocked',reservations});
  await assert.rejects(runMicroeval({...f,resume:true},dependencies),/unresolved/);assert.equal(calls,1);
 }
});
test('native target admission failure survives Pi tool handling without further target work',async t=>{
 const f=await fixture(t),profile=await scriptedMicro(t,f.root,n=>n===0?{name:'rsi_target',args:{command:'python check.py .'}}:null);
 const cwd=join(f.root,'work'),sessionDirectory=join(f.root,'session');await mkdir(cwd);await mkdir(sessionDirectory);
 const reservations=[{physicalDevice:5,token:'original',closed:false}];let calls=0;
 const result=await nativeSample({cwd,sessionDirectory,profile,seconds:5,guidance:'fixture',prompt:'check',targetEnvironment:{execute:async()=>{calls++;throw Object.assign(Error('Device reservation blocked'),{code:'DEVICE_LEASE_BLOCKED',reservations});}}});
 assert.equal(calls,1);assert.deepEqual(result.unresolvedExecution,{handle:null,reason:'device-lease-blocked',reservations});
});
test('sandboxed native Bash can edit its kernel but cannot quarantine shared harness files',async t=>{
 const f=await fixture(t),outside=join(f.root,'shared-leases'),cwd=join(f.root,'work'),sessionDirectory=join(f.root,'session');
 await mkdir(outside);await mkdir(cwd);await mkdir(sessionDirectory);await writeFile(join(outside,'lease.json'),'original-lease');
 const command=`mv '${outside}' '${outside}.quarantined'; printf repaired > kernel.py`;
 let requests=0;const profile=await scriptedMicro(t,f.root,n=>{requests++;return n===0?{name:'bash',args:{command}}:null;});
 // Operational capability proof is independent of the sampler's own preflight.
 const probe=spawnSync('bwrap',['--die-with-parent','--new-session','--unshare-pid','--ro-bind','/','/','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--bind',cwd,cwd,'--chdir',cwd,'--','bash','-c','true'],{encoding:'utf8',timeout:5000});
 const result=await nativeSample({cwd,sessionDirectory,profile,seconds:5,guidance:'fixture',prompt:'repair',localFilesystemSandbox:true});
 const capability=JSON.parse(await readFile(join(sessionDirectory,'sandbox-preflight.json'),'utf8'));assert.equal(capability.available,probe.status===0,JSON.stringify({probe:{status:probe.status,error:probe.error?.code,stderr:probe.stderr},capability}));
 assert.equal(await readFile(join(outside,'lease.json'),'utf8'),'original-lease');await assert.rejects(readFile(join(outside+'.quarantined','lease.json')),{code:'ENOENT'});
 if(probe.status===0){assert.equal(result.error,null);assert(requests>0);assert.equal(await readFile(join(cwd,'kernel.py'),'utf8'),'repaired');}
 else{assert.equal(result.errorCode,'LOCAL_SANDBOX_UNAVAILABLE');assert.equal(result.bucket,'harness');assert.equal(result.toolCalls,0);assert.equal(requests,0);assert.notEqual(capability.exitCode,0);assert.equal(typeof capability.output,'string');await assert.rejects(readFile(join(cwd,'kernel.py')),{code:'ENOENT'});}

});
for(const mode of ['reserved-cleanup','external-cancel','whole-deadline','legacy'])test(`native target cleanup follows the retained whole-sample clock: ${mode}`,async t=>{
 const f=await fixture(t),profile=await scriptedMicro(t,f.root,n=>n===0?{name:'rsi_target',args:{command:'python check.py .',timeout:1200}}:null);
 const cwd=join(f.root,'work'),sessionDirectory=join(f.root,'session');await mkdir(cwd);await mkdir(sessionDirectory);
 const actor=new AbortController(),external=new AbortController(),allocation=new AbortController(),workSignal=AbortSignal.any([external.signal,allocation.signal]),handle='remote:fixture:job:cleanup-'+mode;
 let captured,commands=0,cleanupFinished=false;
 const result=await nativeSample({cwd,sessionDirectory,profile,seconds:3,signal:AbortSignal.any([actor.signal,workSignal]),sampleSignal:workSignal,...(mode==='legacy'?{}:{sampleStopSignal:allocation.signal}),guidance:'fixture',prompt:'check',targetEnvironment:{execute:async(_command,_cwd,options)=>{
  commands++;captured=options.stopSignal;
  if(mode==='legacy')return {handle,settled:true,exitCode:0};
  if(options.stopSignal!==allocation.signal)return {handle,settled:false,exitCode:null};
  // Preparation consumed the short actor allowance before remote admission.
  await new Promise(resolve=>setTimeout(resolve,25));if(mode==='external-cancel')external.abort(Error('External caller cancelled'));else actor.abort(Error('Actor elapsed cutoff'));
  assert.equal(options.signal.aborted,true);assert.equal(options.stopSignal.aborted,false);
  if(mode==='whole-deadline'){
   setTimeout(()=>allocation.abort(Error('Whole sample elapsed cutoff')),25);
   await untilAbort(options.stopSignal);return {handle,settled:false,exitCode:null};
  }
  await new Promise(resolve=>setTimeout(resolve,60));assert.equal(options.stopSignal.aborted,false);cleanupFinished=true;return {handle,settled:true,exitCode:124};
 }}});
 assert.equal(commands,1);assert.equal(captured,mode==='legacy'?undefined:allocation.signal);
 if(mode==='whole-deadline')assert.deepEqual(result.unresolvedExecution,{handle});else assert.equal(result.unresolvedExecution,undefined);
 if(['reserved-cleanup','external-cancel'].includes(mode))assert.equal(cleanupFinished,true);
});

test('microeval hands cleanup the existing allocation deadline independently of external work cancellation',async t=>{
 const f=await fixture(t),external=new AbortController();let calls=0,captured;
 await assert.rejects(runMicroeval({...f,runs:1,signal:external.signal,resume:true},{sample:async(args)=>{
  calls++;captured=args;external.abort(Error('User stopped experiment'));return {};
 }}),/User stopped experiment/);assert.equal(calls,1);
 assert.ok(captured.sampleStopSignal instanceof AbortSignal);assert.notEqual(captured.sampleStopSignal,captured.sampleSignal);
 assert.equal(captured.signal.aborted,true);assert.equal(captured.sampleSignal.aborted,true);assert.equal(captured.sampleStopSignal.aborted,false);
});

test('whole work expiry allows exact-handle cleanup but rejects late verification success',async t=>{
 const f=await fixture(t),handle='remote:fixture:job:late-verifier';let actors=0,verifiers=0,closed=0;
 const report=await runMicroeval({...f,runs:1,seconds:.08,totalSeconds:.08,verificationSeconds:.04},{sample:async()=>{actors++;return {};},verify:async args=>{
  verifiers++;await untilAbort(args.signal);assert.equal(args.sampleStopSignal.aborted,false);
  await new Promise(resolve=>setTimeout(resolve,20));assert.equal(args.sampleStopSignal.aborted,false);closed++;
  return {correct:true,exitCode:0,handle};
 }});
 assert.equal(actors,1);assert.equal(verifiers,1);assert.equal(closed,1);assert.equal(report.results.length,1);
 assert.equal(report.results[0].correct,false);assert.equal(report.results[0].timedOut,true);assert.ok(report.results[0].elapsedSeconds>=.08);assert.equal(report.status,'elapsed');
});
test('true cleanup grace exhaustion retains the original unknown handle and blocks further admissions',{timeout:40000},async t=>{
 const f=await fixture(t),handle='remote:fixture:job:grace-exhausted';let actors=0,verifiers=0;
 await assert.rejects(runMicroeval({...f,runs:1,seconds:.04,totalSeconds:.04,verificationSeconds:.02},{sample:async()=>{actors++;return {};},verify:async args=>{
  verifiers++;await untilAbort(args.signal);assert.equal(args.sampleStopSignal.aborted,false);
  await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('Cleanup deadline did not expire')),35000);args.sampleStopSignal.addEventListener('abort',()=>{clearTimeout(timer);resolve();},{once:true});});assert.equal(args.sampleStopSignal.aborted,true);throw Object.assign(Error('Exact native execution remains unresolved'),{code:'RSI_EXECUTION_UNRESOLVED',handle});
 }}),{code:'RSI_EXECUTION_UNRESOLVED'});
 const report=JSON.parse(await readFile(join(f.directory,'results.json')));assert.equal(actors,1);assert.equal(verifiers,1);assert.equal(report.results.length,0);assert.equal(report.runningSample.unresolvedExecution.handle,handle);
 assert.ok(report.elapsedSeconds>=30);assert.equal(report.status,'interrupted');
});

test('native Pi target closes its same handle after whole work cutoff without a second command',async t=>{
 const f=await fixture(t),profile=await scriptedMicro(t,f.root,n=>n===0?{name:'rsi_target',args:{command:'python check.py .',timeout:1200}}:null);
 const handle='remote:fixture:job:whole-expiry-native',requests=[];let verifies=0;
 const report=await runMicroeval({...f,profile,runs:1,seconds:2,totalSeconds:2,verificationSeconds:1},{sample:args=>nativeSample({...args,targetEnvironment:{execute:async(_command,_cwd,options)=>{
  requests.push(['execute',handle]);await untilAbort(args.sampleSignal);assert.equal(options.signal.aborted,true);assert.equal(options.stopSignal.aborted,false);
  requests.push(['cancel',handle]);await new Promise(resolve=>setTimeout(resolve,25));assert.equal(options.stopSignal.aborted,false);requests.push(['closed',handle]);
  return {handle,settled:true,exitCode:124};
 }}}),verify:async()=>{verifies++;return {correct:true};}});
 assert.deepEqual(requests,[['execute',handle],['cancel',handle],['closed',handle]]);assert.equal(verifies,0);assert.equal(report.results.length,1);assert.equal(report.results[0].timedOut,true);assert.equal(report.results[0].correct,false);assert.equal(report.results[0].unresolvedExecution,undefined);
});

test('resumed elapsed cleanup spend cannot create another work or cleanup turn',async t=>{
 const f=await fixture(t),abort=new AbortController(),options={...f,runs:1,seconds:1,resume:true};
 await assert.rejects(runMicroeval({...options,signal:abort.signal},{sample:async()=>{abort.abort();return {};}}));
 const path=join(f.directory,'results.json'),retained=JSON.parse(await readFile(path));
 // Retained elapsed cost includes cleanup after the original work allocation.
 retained.partials['repair/0/active'].elapsedSeconds=1.2;retained.elapsedSeconds=1.2;await writeFile(path,JSON.stringify(retained));
 const admitted=[];const report=await runMicroeval(options,{sample:async args=>{admitted.push(args.cwd);return {};},verify:async()=>({correct:false})});
 assert.equal(admitted.length,1);assert.ok(admitted[0].endsWith('0-candidate/work'));
 const row=report.results.find(row=>row.arm==='active');assert.equal(row.elapsedSeconds,1.2);assert.equal(row.timedOut,true);assert.equal(row.correct,false);
});

test('native target-local Bash uses admitted environment instead of an unmasked host process',async t=>{
 const f=await fixture(t),profile=await scriptedMicro(t,f.root,n=>n===0?{name:'bash',args:{command:'printf bypass > host-marker'}}:null);
 const cwd=join(f.root,'work'),sessionDirectory=join(f.root,'session'),cleanup=new AbortController();await mkdir(cwd);await mkdir(sessionDirectory);let calls=0;
 const result=await nativeSample({cwd,sessionDirectory,profile,seconds:5,sampleStopSignal:cleanup.signal,guidance:'fixture',prompt:'check',targetBash:true,targetEnvironment:{execute:async(command,path,options)=>{
  calls++;assert.equal(command,'printf bypass > host-marker');assert.equal(path,cwd);assert.equal(options.stopSignal,cleanup.signal);options.onData?.(Buffer.from('admitted target execution\n'));return {handle:'local:admitted',settled:true,exitCode:0};
 }}});
 assert.equal(result.error,null);assert.equal(calls,1);await assert.rejects(readFile(join(cwd,'host-marker')),{code:'ENOENT'});
});

test('a real native target-tool deadline retains its partial marker until SDK lease closure is proved',async t=>{
 const {hypothesisEvaluationSettled}=await import('../../cli/skill-hypothesis-evaluation.mjs'),{processIdentity}=await import('../../controller/lifecycle.mjs');
 const f=await fixture(t),profile=await scriptedMicro(t,f.root,n=>n===0?{name:'rsi_target',args:{command:'python check.py .'}}:null),root=join(f.root,'hypothesis'),directory=join(root,'micro');await mkdir(root);
 const binding={target:'a3',environment:{mode:'local',target:'fixture',runtime:'fixture',health:{product:'a3'}}};await writeFile(join(root,'hypothesis.json'),JSON.stringify({binding}));
 let nativeCalls=0,leaseClosed=false,target;const tasks=[{...f.tasks[0],id:'prior'},{...f.tasks[0],kind:'a3-kernel'}];
 await assert.rejects(runMicroeval({...f,directory,profile,tasks,runs:1,seconds:.6,totalSeconds:2,resume:true,binding},{
  sample:async args=>{
   if(args.kind==='unspecified'){await writeFile(join(args.cwd,'kernel.py'),'def answer(): return 42\n');return {tokens:1};}
   return nativeSample({...args,targetEnvironment:{execute:async(_command,cwd,options)=>{
    nativeCalls++;target=join(dirname(cwd),'target');await mkdir(target,{recursive:true});const handle='remote:fixture:job:deadline-setup',lease={receipt:join(root,'lease.json'),token:'deadline-owned',pool:'fixture',physicalDevice:4,owner:await processIdentity()};
    await writeFile(join(target,'.ascend-command-lease.json'),JSON.stringify(lease));
    const events=[{type:'device-lease-acquired',data:lease},{type:'execution-submitting',data:{operation:'setup',submission:'setup'}},{type:'execution-submitted',data:{submission:'setup',handle}},{type:'execution-handle',data:{handle}}];
    await writeFile(join(target,'environment-events.jsonl'),events.map(JSON.stringify).join('\n')+'\n');await untilAbort(options.signal);
    events.push({type:'execution-state',data:{handle,state:'stopped',nativeState:'cancelled'}});await writeFile(join(target,'environment-events.jsonl'),events.map(JSON.stringify).join('\n')+'\n');return {settled:true,exitCode:124,handle};
   }}});
  },verify:async args=>{const result=spawnSync(f.python,[args.checker,args.cwd],{encoding:'utf8'});return {correct:result.status===0,exitCode:result.status,output:result.stdout,error:result.stderr};},
  sampleSettled:async()=>false
 }),{code:'RSI_EXECUTION_UNRESOLVED'});
 const file=join(directory,'results.json'),before=JSON.parse(await readFile(file,'utf8'));assert.equal(nativeCalls,1);assert.equal(before.results.length,2);assert(before.results.every(result=>result.correct));assert.equal(before.runningSample.key,'repair/0/active');assert.equal(before.runningSample.needsClosure,true);assert(before.partials['repair/0/active'].timedOut);
 const deps={createEnvironment:async(_options,context)=>{assert.equal(context.instancePath,target);return {reconcileLease:async()=>{if(leaseClosed)await rm(join(target,'.ascend-command-lease.json'),{force:true});return leaseClosed;}};}};
 assert.equal(await hypothesisEvaluationSettled({directory:root},deps),false);leaseClosed=true;assert.equal(await hypothesisEvaluationSettled({directory:root},deps),true);
 const after=JSON.parse(await readFile(file,'utf8'));assert.equal(after.runningSample,undefined);assert.deepEqual(after.results,before.results);assert.deepEqual(after.partials,before.partials);assert.equal(after.elapsedSeconds,before.elapsedSeconds);assert.equal(nativeCalls,1);
});
test('frozen split protocol uses independent actor and verification deadlines inside the original total',async t=>{
 const f=await fixture(t),protocol={version:1,sampleSeconds:1,actorSeconds:.5,verificationSeconds:.5};let actors=0,checks=0;
 const report=await runMicroeval({...f,runs:1,seconds:1,verificationSeconds:.5,sampleProtocol:protocol},{sample:async({signal,sampleSignal,seconds,verificationReserved})=>{
  actors++;assert(seconds<=.5&&seconds>.4);assert(verificationReserved>=.5);await untilAbort(signal);assert.equal(signal.aborted,true);assert.equal(sampleSignal.aborted,false);return {timedOut:true};
 },verify:async({signal,seconds})=>{checks++;assert.equal(signal.aborted,false);assert(seconds>0&&seconds<=.5);return {correct:true};}});
 assert.equal(actors,2);assert.equal(checks,2);assert.deepEqual(report.sampleProtocol,protocol);assert(report.results.every(row=>row.allocatedSeconds===1&&row.actorTimedOut&&!row.timedOut&&row.correct));
 await assert.rejects(runMicroeval({...f,runs:1,seconds:1,verificationSeconds:.5,sampleProtocol:{...protocol,actorSeconds:.6}}),/Sample timing protocol differs/);
});
test('new verification phase refuses late correctness while legacy verification may use remaining actor time',async t=>{
 for(const fresh of [true,false]){
  const f=await fixture(t),protocol={version:1,sampleSeconds:1,actorSeconds:.8,verificationSeconds:.2};let checks=0;
  const report=await runMicroeval({...f,runs:1,seconds:1,verificationSeconds:.2,...(fresh?{sampleProtocol:protocol}:{})},{sample:async()=>({}),verify:async args=>{
   checks++;assert.equal(args.signal===args.sampleSignal,!fresh);assert(fresh?args.seconds<=.2:args.seconds>.8);
   if(fresh){await untilAbort(args.signal);assert.equal(args.sampleSignal.aborted,false);}return {correct:true};
  }});
  assert.equal(checks,2);assert(report.results.every(row=>row.correct===!fresh));if(fresh)assert(report.results.every(row=>row.verificationTimedOut&&!row.timedOut));
 }
});
test('a verification-only target cutoff retains its marker and partial until positive SDK custody proof',async t=>{
 const f=await fixture(t),protocol={version:1,sampleSeconds:1,actorSeconds:.8,verificationSeconds:.2};let actors=0,closures=0;
 const tasks=f.tasks.map(task=>({...task,kind:'target-kernel',target:'a3',bucket:'correctness'}));
 await assert.rejects(runMicroeval({...f,tasks,runs:1,seconds:1,verificationSeconds:.2,sampleProtocol:protocol,objective:'correctness',binding:{target:'a3',environment:{health:{product:'a3'}}}},{sample:async()=>{actors++;return {};},verify:async args=>{
  await untilAbort(args.signal);assert.equal(args.sampleSignal.aborted,false);return {correct:false};
 },sampleSettled:async args=>{closures++;assert.equal(args.signal.aborted,false);return false;}}),{code:'RSI_EXECUTION_UNRESOLVED'});
 const report=JSON.parse(await readFile(join(f.directory,'results.json'),'utf8'));assert.equal(actors,1);assert.equal(closures,1);assert.equal(report.results.length,0);assert.equal(report.runningSample.needsClosure,true);assert.equal(report.partials['repair/0/active'].verificationTimedOut,true);assert.equal(report.partials['repair/0/active'].correct,false);
 await assert.rejects(runMicroeval({...f,tasks,runs:1,seconds:1,verificationSeconds:.2,sampleProtocol:protocol,resume:true,objective:'correctness',binding:{target:'a3',environment:{health:{product:'a3'}}}},{sample:()=>assert.fail('Unresolved verifier cannot advance')}),/execution remains unresolved/);
});
test('an explicit zero reserve preserves whole-sample verification compatibility',async t=>{
 const f=await fixture(t),protocol={version:1,sampleSeconds:1,actorSeconds:1,verificationSeconds:0};let checks=0;
 const report=await runMicroeval({...f,runs:1,seconds:1,sampleProtocol:protocol},{sample:async()=>({}),verify:async args=>{checks++;assert(args.seconds>.8);assert.equal(args.signal,args.sampleSignal);return {correct:true};}});
 assert.equal(checks,2);assert(report.results.every(row=>row.correct));
});
test('resuming new-protocol verification preserves its spent phase allowance without another actor turn',async t=>{
 const f=await fixture(t),external=new AbortController(),protocol={version:1,sampleSeconds:1,actorSeconds:.6,verificationSeconds:.4},options={...f,runs:1,seconds:1,verificationSeconds:.4,sampleProtocol:protocol,resume:true};let actors=0,checks=0;
 await assert.rejects(runMicroeval({...options,signal:external.signal},{sample:async()=>{actors++;return {};},verify:async args=>{
  setTimeout(()=>external.abort(Error('Pause during verification')),100);await untilAbort(args.signal);return {correct:false};
 }}),/Pause during verification/);
 const before=JSON.parse(await readFile(join(f.directory,'results.json'),'utf8')),spent=before.partials['repair/0/active'].verificationElapsedSeconds;assert(spent>=.09&&spent<.4);
 const report=await runMicroeval(options,{sample:async()=>{actors++;return {};},verify:async args=>{checks++;if(checks===1)assert(args.seconds<=.4-spent&&args.seconds>0);return {correct:true};}});
 assert.equal(actors,2);assert.equal(checks,2);assert(report.results[0].verificationElapsedSeconds>=spent);assert(report.results.every(row=>row.correct));
});

test('resuming a microevaluation rejects a different frozen model context allowance',async t=>{
 const f=await fixture(t),profile={...f.profile,max_context_tokens:200000};let calls=0;
 await runMicroeval({...f,profile},{sample:async()=>{calls++;return {};}});
 const before=calls;
 await assert.rejects(runMicroeval({...f,profile:{...profile,max_context_tokens:100000},resume:true},{sample:()=>{calls++;throw Error('must not dispatch');}}),/Frozen micro/);
 assert.equal(calls,before);
 const resumed=await runMicroeval({...f,profile,resume:true},{sample:()=>{throw Error('must not replay');}});assert.equal(resumed.status,'complete');
});


test('direct comparisons freeze omitted context caps and pass the same bounded profile to both arms',async t=>{
 const f=await fixture(t),caps=[];
 await runMicroeval(f,{sample:async({profile})=>{caps.push(profile.max_context_tokens);return {};}});
 assert.deepEqual(caps,[200000,200000,200000,200000]);assert.equal(f.profile.max_context_tokens,undefined);
 const manifest=JSON.parse(await readFile(join(f.directory,'manifest.json'),'utf8'));assert.equal(manifest.model.maxContextTokens,200000);
 const resumed=await runMicroeval({...f,resume:true},{sample:()=>{throw Error('must not replay');}});assert.equal(resumed.status,'complete');
 await assert.rejects(runMicroeval({...f,profile:{...f.profile,max_context_tokens:64000},resume:true},{sample:()=>{throw Error('must not dispatch');}}),/Frozen micro/);
});
