import test from 'node:test';import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,stat} from 'node:fs/promises';import {join,resolve} from 'node:path';import {homedir} from 'node:os';import {createHash,randomUUID} from 'node:crypto';import {execFile,spawn} from 'node:child_process';import {promisify} from 'node:util';import {setTimeout as delay} from 'node:timers/promises';
import {State} from '../../controller/state.mjs';import {initializeAttention} from '../../controller/failure-attention.mjs';import {processIdentity} from '../../controller/lifecycle.mjs';
import {prepareFailureMemory,admitWarmSearch} from '../../cli/skill-failure-memory.mjs';import {createFailureRuntime,failureRuntimeSettled,failureChecker,kdbContamination,failureReceipt} from '../../cli/skill-failure-runtime.mjs';
import {hypothesisTarget} from '../../cli/skill-hypothesis-evaluation.mjs';import {nativeSample} from '../../cli/skill-microeval.mjs';import {scriptedReview} from './helpers/scripted-review.mjs';
const execute=promisify(execFile),hash=x=>createHash('sha256').update(x).digest('hex'),python=process.env.DRIVER_PYTHON??'python3',closers=new Map();
async function fixture(t){const parent=join(homedir(),'.cache/rsi-failure-runtime-tests');await mkdir(parent,{recursive:true});const root=await mkdtemp(join(parent,'case-'));closers.set(root,[]);t.after(async()=>{for(const close of closers.get(root))await close();closers.delete(root);await rm(root,{recursive:true,force:true});});return root;}
const receiptCode=(value,extra='')=>`import hashlib,json,pathlib,sys,time,subprocess\np=pathlib.Path(sys.argv[1])/'kernel.py'\nprint(json.dumps({'correct':${value},'kernelSha256':hashlib.sha256(p.read_bytes()).hexdigest(),'evidence':{'checkedCases':[]}}),flush=True)\n${extra}\n`;
async function oracleRun(root,checker,guard={}){const file=join(root,randomUUID()+'.py');await writeFile(file,await failureChecker({checker},null,{seconds:.25,cleanupSeconds:.05,...guard}));try{return JSON.parse((await execute(python,[file,root],{timeout:10000})).stdout.trim());}catch(e){assert(e.stdout,e.message);return JSON.parse(e.stdout.trim());}}

test('trusted process guard preserves recorded failure only with source identity and full child closure',async t=>{
 const root=await fixture(t);await writeFile(join(root,'kernel.py'),'answer=0\n');const checker=join(root,'check.py');
 await writeFile(checker,receiptCode('False',"subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);time.sleep(60)"));const failed=await oracleRun(root,checker,{seconds:2,cleanupSeconds:.25});assert.equal(failed.correct,false,JSON.stringify(failed));assert.equal(failed.evidence.cleanupElapsedAfterFailure,true);assert.equal(failed.evidence.processGuard.allChildrenReaped,true);
 for(const ending of ['sys.exit(1)',"raise RuntimeError('after receipt')"]){await writeFile(checker,receiptCode('True',ending));assert.equal((await oracleRun(root,checker)).correct,null);}
 await writeFile(checker,receiptCode('True','time.sleep(60)'));assert.equal((await oracleRun(root,checker)).correct,null);
 await writeFile(checker,receiptCode('True','time.sleep(1.25)'));
 assert.equal((await oracleRun(root,checker,{seconds:4})).correct,null);
 const delayed=await oracleRun(root,checker,{seconds:4,cleanupSeconds:2});
 assert.equal(delayed.correct,true);assert.equal(delayed.evidence.processGuard.commandExit,0);assert.equal(delayed.evidence.processGuard.allChildrenReaped,true);
 await writeFile(checker,'import time;time.sleep(60)');assert.equal((await oracleRun(root,checker)).correct,null);
 await writeFile(checker,receiptCode('False','time.sleep(60)'));assert.equal((await oracleRun(root,checker,{rssBytes:1})).correct,null);
 await writeFile(checker,receiptCode('False','time.sleep(60)').replace("hashlib.sha256(p.read_bytes()).hexdigest()","'wrong'"));assert.equal((await oracleRun(root,checker)).correct,null);
 await writeFile(checker,receiptCode('True').replace("'kernelSha256':hashlib.sha256(p.read_bytes()).hexdigest(),",''));
 const guardBound=await oracleRun(root,checker,{seconds:2});assert.equal(guardBound.correct,true);assert.equal(guardBound.kernelSha256,hash('answer=0\n'));
 await writeFile(checker,receiptCode('True').replace('hashlib.sha256(p.read_bytes()).hexdigest()','None'));
 assert.equal((await oracleRun(root,checker,{seconds:2})).kernelSha256,guardBound.kernelSha256);
 await writeFile(checker,receiptCode('True',"p.write_text('changed')").replace("'kernelSha256':hashlib.sha256(p.read_bytes()).hexdigest(),",''));
 assert.equal((await oracleRun(root,checker,{seconds:2})).correct,null);
 await writeFile(checker,"import signal,time,sys,pathlib,json,hashlib\np=pathlib.Path(sys.argv[1])/'kernel.py'\ndef stop(*_):\n print(json.dumps({'correct':False,'kernelSha256':hashlib.sha256(p.read_bytes()).hexdigest(),'evidence':{}}),flush=True);sys.exit(1)\nsignal.signal(signal.SIGTERM,stop);time.sleep(60)\n");assert.equal((await oracleRun(root,checker)).correct,null);
 assert.equal(failureReceipt({sourceFilesMatch:false}).correct,null);
});
test('explicit case outcomes distinguish executed failures and reject missing or inconsistent results',async t=>{
 const root=await fixture(t),checker=join(root,'checker.py');await writeFile(checker,'pass');const task={id:'dev',caseIds:['value','control'],files:{'kernel.py':'answer=0'},checker,publicChecker:checker,prompt:'Fixture'};
 const receipt=caseResults=>({sourceFilesMatch:true,sourceFilesSha256:{'kernel.py':'sha'},output:JSON.stringify({correct:false,kernelSha256:'sha',evidence:{checkedCases:['value','control'],caseResults,processGuard:{allChildrenReaped:true}}})});
 const valid=[{id:'value',correct:false},{id:'control',correct:true}];assert.equal(failureReceipt(receipt(valid),task).correct,false);
 for(const cases of [undefined,[],[{id:'value',correct:false}],valid.map(c=>({...c,correct:true})),[valid[0],valid[0]],[valid[0],{id:'other',correct:true}],[valid[0],{id:'control',correct:1}]])assert.equal(failureReceipt(receipt(cases),task).correct,null);
 const config={target:'cpu',actor:{},environment:{options:{mode:'local'}}},directory=join(root,'private');const runtime=await createFailureRuntime({config,directory,slots:[{resource:{id:'slot'},physical:1}]},{target:()=>({sample:async()=>({timedOut:false}),sampleSettled:async()=>true,qualifyBatch:async input=>input.entries.map(e=>({...receipt(undefined),key:e.key}))}),settled:async()=>true});
 const family={id:'missing',mechanism:{phase:'check',type:'Mismatch'},development:task,transfer:[{...task,id:'transfer1'},{...task,id:'transfer2'}]};for(const t of [family.development,...family.transfer])t.controls={reference:{files:t.files},negatives:[{files:t.files}]};
 await assert.rejects(runtime.qualify({family,directory:join(directory,'qualification'),remainingSeconds:10}),/Explicit case outcomes/);
 const result=await runtime.trial({familyId:family.id,task,directory:join(directory,'trial'),guidance:'same',actorSeconds:1,verificationSeconds:1});assert.equal(result.infrastructure,true);assert.equal(result.correct,null);assert.match(result.reason,/Explicit case outcomes/);assert.equal(await runtime.close(),true);
});
test('oracle SIGTERM closes and reaps owned children without scoring interrupted work',async t=>{
 const root=await fixture(t),checker=join(root,'check.py');await writeFile(join(root,'kernel.py'),'answer=0');await writeFile(checker,"import sys,pathlib,time\n(pathlib.Path(sys.argv[1])/'entered').write_text('ready');time.sleep(60)\n");const script=join(root,'guard.py');await writeFile(script,await failureChecker({checker},null,{seconds:60}));
 const child=spawn(python,[script,root],{stdio:['ignore','pipe','pipe']});t.after(()=>child.kill('SIGKILL'));let output='';child.stdout.on('data',part=>{output+=part;});const finished=new Promise((done,reject)=>{child.once('error',reject);child.once('close',done);});
 const deadline=Date.now()+5000;while(!await stat(join(root,'entered')).catch(e=>{if(e.code==='ENOENT')return false;throw e;})){assert(Date.now()<deadline,'Oracle did not start');await delay(10);}child.kill('SIGTERM');await finished;const result=JSON.parse(output.trim());assert.equal(result.correct,null);assert.equal(result.evidence.processGuard.reason,'signal:15');assert.equal(result.evidence.processGuard.allChildrenReaped,true);
});
test('KDB infrastructure contamination excludes even a passing session without treating compiler or argument errors as outages',()=>{
 const line=(tool,text)=>JSON.stringify({message:{role:'toolResult',toolName:tool,isError:true,content:[{type:'text',text}]}});
 assert.equal(kdbContamination(line('mcp__kdb__search','MCP request timed out after 60000ms')).infrastructure,true);
 assert.equal(kdbContamination(line('mcp__kdb__read_entry','transport disconnected')).correct,null);
 assert.equal(kdbContamination(line('mcp__kdb__search','kind must be good/negative/all')),null);assert.equal(kdbContamination(line('rsi_target','Compiler failure')),null);
 assert.throws(()=>admitWarmSearch({seconds:10,structured:true,isError:false}));assert.throws(()=>admitWarmSearch({seconds:1,structured:false}));assert.doesNotThrow(()=>admitWarmSearch({seconds:1,structured:true}));
});

async function memoryFixture(t,root){
 const packageRoot=resolve(new URL('../..',import.meta.url).pathname),original=join(root,'campaign');await mkdir(original);const state=new State(join(original,'state.sqlite'));initializeAttention(state.db);state.db.exec("insert into failure_attention_actions values(1,'fixed','retired','{}');CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT NOT NULL)");state.db.prepare("insert into control values('knowledge-startup',?)").run(JSON.stringify(await processIdentity()));state.close();
 await writeFile(join(original,'memory_fixture.py'),"import hashlib\nclass Embeddings:\n model='deterministic-cpu';revision='1'\n def embed(self,texts):return [[v/255 for v in hashlib.sha256(t.encode()).digest()]for t in texts]\n");
 const wrapper=join(original,'python-wrapper'),resolvedPython=(await execute(python,['-c','import sys;print(sys.executable)'])).stdout.trim();
 await writeFile(wrapper,`#!${resolvedPython}\nimport sys,runpy\nsys.path[:0]=[${JSON.stringify(original)},${JSON.stringify(packageRoot)}]\nif sys.argv[1]=='-c':\n code=sys.argv[2];sys.argv=['-c',*sys.argv[3:]];exec(code)\nelse:\n name=sys.argv[2];sys.argv=[name,*sys.argv[3:]];runpy.run_module(name,run_name='__main__')\n`,{mode:0o700});
 const docs=join(original,'docs');await mkdir(docs);await writeFile(join(docs,'kernel.py'),'def answer(): return 42\n');await execute(wrapper,['-c',`from pathlib import Path\nfrom ascend_tools.kdb import KDB\nfrom memory_fixture import Embeddings\nk=KDB(Path(${JSON.stringify(join(original,'knowledge.sqlite'))}),Embeddings());k.add_repo(${JSON.stringify(docs)},'fixture','python');k.close()`]);
 return {root:original,python:wrapper,embedding_factory:'memory_fixture:Embeddings',target:'cpu',environment:{options:{mode:'local',python:resolvedPython}}};
}
function localTargetFactory(record){return args=>{const target=hypothesisTarget(args,{createEnvironment:async(_options,context)=>({prepare:async()=>{},reconcileLease:async()=>true,instructions:'CPU fixture target; use rsi_target for checks.',execute:async(command,cwd,options)=>{
 record.push({command,cwd});try{const result=await execute('bash',['-c',command],{cwd,signal:options.signal,maxBuffer:1048576});options.onData?.(result.stdout);return {settled:true,exitCode:0,handle:'local:'+randomUUID()};}catch(error){if(!Number.isInteger(error.code))throw error;options.onData?.(error.stdout);return {settled:true,exitCode:error.code,handle:'local:'+randomUUID()};}
 }})});for(const [name,kind]of [['sample','actor'],['qualifyBatch','verification']]){const original=target[name];target[name]=input=>{record.push({kind,seconds:input.seconds,guidance:input.guidance,stopSignal:input.sampleStopSignal});return original(input);};}return target;};}
test('production adapter uses native Pi and real MCP warm memory, isolates oracle, verifies frozen sources, and recovers without redispatch', {timeout:60000},async t=>{
 const root=await fixture(t),config=await memoryFixture(t,root),memory=await prepareFailureMemory({config,directory:join(root,'views'),generation:{revision:'fixed',attention:1}});closers.get(root).push(memory.close);
 assert.equal(memory.snapshot.attentionRevision,1);assert.equal(memory.knowledge.readOnly,true);const before=await stat(join(memory.config.root,'knowledge.sqlite'));const cached=await prepareFailureMemory({config,directory:join(root,'views'),generation:{revision:'fixed',attention:1}});assert.equal(cached.snapshot.sha256,memory.snapshot.sha256);assert.equal((await stat(join(memory.config.root,'knowledge.sqlite'))).ino,before.ino);
 const originalState=new State(join(config.root,'state.sqlite'));assert(originalState.db.prepare("select value from control where key='knowledge-startup'").get());originalState.close();
 const providerRoot=join(root,'provider');await mkdir(providerRoot);const provider=await scriptedReview(t,providerRoot,body=>{const calls=body.messages.flatMap(m=>m.tool_calls??[]),first=body.messages.find(m=>m.role==='user').content,text=typeof first==='string'?first:first.map(p=>p.text??'').join('\n'),probe='if test -r '+JSON.stringify(join(root,'trusted.py'))+'; then exit 17; fi; printf hidden > mask-proof.json';
  if(!calls.some(c=>c.function.name==='bash'&&JSON.parse(c.function.arguments).command===probe))return {name:'bash',args:{command:probe}};
  if(!calls.some(c=>c.function.name==='mcp__kdb__search'))return {name:'mcp__kdb__search',args:{query:'answer kernel',limit:3}};
  if(!calls.some(c=>c.function.name==='write'))return {name:'write',args:{path:'kernel.py',content:'answer='+ (text.includes('complete candidate guidance')?'42':'0')+'\n'}};
  if(!calls.some(c=>c.function.name==='rsi_target'))return {name:'rsi_target',args:{command:'printf target > target-proof.json'}};
  if(!calls.some(c=>c.function.name==='bash'&&JSON.parse(c.function.arguments).command==='python3 public_verify.py .'))return {name:'bash',args:{command:'python3 public_verify.py .'}};return null;
 });config.actor=provider.profile;
 const checker=join(root,'trusted.py');await writeFile(checker,"import pathlib,runpy,json,hashlib,sys\np=pathlib.Path(sys.argv[1])/'kernel.py';value=runpy.run_path(str(p))['answer'];ok=value==sum([6]*7)\nprint(json.dumps({'correct':ok,'kernelSha256':hashlib.sha256(p.read_bytes()).hexdigest(),'evidence':{'checkedCases':['value'],'caseResults':[{'id':'value','correct':ok}],'phase':'numeric','type':'Mismatch'}}));sys.exit(0 if ok else 1)\n");
 const publicChecker=join(root,'public.py');await writeFile(publicChecker,'from kernel import answer\nassert answer==42\n');
 const task=id=>({id,prompt:'Implement answer=42 and use public feedback.',caseIds:['value'],files:{'kernel.py':'answer=0\n'},checker,publicChecker,controls:{reference:{files:{'kernel.py':'answer=42\n'}},negatives:[{files:{'kernel.py':'answer=0\n'}}]}}),family={id:'family',mechanism:{phase:'numeric',type:'Mismatch'},development:task('dev'),transfer:[task('layout'),task('operation')]};
 const directory=join(root,'private'),record=[],slots=[{physical:3,resource:{id:'slot',metadata:{}}}],deps={target:localTargetFactory(record),settled:async()=>true};
 const runtime=await createFailureRuntime({config,directory,slots,knowledge:memory.knowledge},deps);assert.deepEqual(await runtime.identify(),{target:'cpu'});
 const qualified=await runtime.qualify({family,directory:join(directory,'qualify'),remainingSeconds:30});assert.equal(qualified.qualified,true);
 const input={familyId:'family',task:family.development,directory:join(directory,'baseline'),guidance:'complete baseline guidance',actorSeconds:10,verificationSeconds:17};
 const baseline=await runtime.trial(input);assert.equal(baseline.correct,false);assert.equal(baseline.mechanismMatch,true);assert.deepEqual(baseline.caseResults,[{id:'value',correct:false}]);
 const candidate=await runtime.trial({...input,directory:join(directory,'candidate'),guidance:'complete candidate guidance'});assert.equal(candidate.correct,true);assert.equal(candidate.caseResults[0].correct,true);assert.equal(candidate.handles.length,1);assert.equal(record.filter(r=>r.command==='printf target > target-proof.json').length,2);assert.equal(record.some(r=>r.command?.includes('printf hidden')),false);
 assert.deepEqual(record.filter(r=>r.kind==='actor').map(r=>[r.seconds,r.guidance]),[[10,'complete baseline guidance'],[10,'complete candidate guidance']]);assert.equal(record.filter(r=>r.kind==='verification').slice(-2).every(r=>r.seconds>16&&r.seconds<=17),true);
 const count=record.length,reopened=await createFailureRuntime({config,directory,slots,knowledge:memory.knowledge},deps),recovered=await reopened.settle({directory:join(directory,'candidate')});assert.deepEqual(recovered.result,candidate);assert.equal(record.length,count);assert.equal((await failureRuntimeSettled({directory,config},deps)).closed,true);assert.equal(await runtime.close(),true);
 const second={...family,id:'second-family',development:{...family.development,files:{'kernel.py':'answer=-1\n'}}};assert.equal((await runtime.qualify({family:second,directory:join(directory,'second-family'),remainingSeconds:30})).qualified,true);
 const adapter=JSON.parse(await readFile(join(directory,'candidate','adapter.json'),'utf8'));const requests=(await readFile(join(adapter.cwd,'..','pi-request-events.jsonl'),'utf8')).trim().split('\n').map(JSON.parse);assert(requests.some(row=>row.event==='response-end'&&row.outcome!== 'error'));assert.equal(await readFile(join(adapter.cwd,'mask-proof.json'),'utf8'),'hidden');await writeFile(join(adapter.cwd,'kernel.py'),'answer=99\n');await assert.rejects(reopened.settle({directory:join(directory,'candidate')}),/Recovered source changed/);
 await writeFile(checker,'changed oracle');await assert.rejects(runtime.qualify({family,directory:join(directory,'changed'),remainingSeconds:30}),/Frozen task changed/);
 await memory.close();await rm(join(memory.config.root,'knowledge.sqlite'));let starts=0;await assert.rejects(prepareFailureMemory({config,directory:join(root,'views'),generation:{revision:'fixed',attention:1}},{ensure:async()=>{starts++;}}),{code:'ENOENT'});assert.equal(starts,0);
});
test('readiness refuses slow warmed queries and releases only its private service',async t=>{
 const root=await fixture(t),config=await memoryFixture(t,root);let releases=0,time=0,calls=0;
 await assert.rejects(prepareFailureMemory({config,directory:join(root,'views'),generation:'slow'},{ensure:async()=> 'http://127.0.0.1/fixture',run:async()=>{time+=calls++?11000:90000;return {structuredContent:{result:[]}};},now:()=>time,release:async cfg=>{assert.notEqual(cfg.root,config.root);releases++;}}),/exceeds ten seconds/);assert.equal(releases,1);assert.equal(calls,2);
});
test('unknown custody survives adapter restart, blocks its slot and does not require a knowledge service',async t=>{
 const root=await fixture(t),directory=join(root,'private'),task={id:'dev',files:{'kernel.py':'answer=0\n'},caseIds:['value'],prompt:'Fixture'},checker=join(root,'checker.py');await writeFile(checker,'pass');task.checker=task.publicChecker=checker;
 const config={target:'cpu',actor:{},environment:{options:{mode:'local'}}},slot={physical:7,resource:{id:'lease',metadata:{pool:'fixture'}},binding:{python:'qualified-python'}};let dispatches=0;
 const deps={target:()=>({sample:async()=>{dispatches++;return {timedOut:false};},sampleSettled:async()=>false}),settled:async()=>false};
 const runtime=await createFailureRuntime({config,directory,slots:[slot]},deps),input={task,familyId:'family',guidance:'same',directory:join(directory,'op'),actorSeconds:1,verificationSeconds:1};const bad=await runtime.trial(input);assert.equal(bad.infrastructure,true);assert.match(bad.reason,/custody/);assert.equal(dispatches,1);
 await assert.rejects(runtime.trial({...input,directory:join(directory,'later')}),/No admitted trial slot/);assert.equal(dispatches,1);assert.equal(await runtime.close(),false);
 const seen=[];const status=await failureRuntimeSettled({config,directory},{settled:async args=>{seen.push(args);return false;}});assert.equal(status.closed,false);assert.equal(seen.length,1);assert.deepEqual(seen[0].resource,slot.resource);assert.equal(seen[0].binding.python,'qualified-python');assert.deepEqual(seen[0].binding.environment.deviceAffinity,[7]);
 const stopped=await createFailureRuntime({config,directory:join(root,'new'),slots:[slot]},deps);await assert.rejects(stopped.trial({...input,signal:AbortSignal.abort(Error('operator stop'))}),/operator stop/);assert.equal(dispatches,1);
});
test('Request timed out returns infrastructure only after actor custody closes and skips scoring',async t=>{
 const root=await fixture(t),checker=join(root,'checker.py');await writeFile(checker,'pass');const directory=join(root,'private'),config={target:'cpu',actor:{},environment:{options:{mode:'local'}}},slots=[{resource:{id:'slot'},physical:1}];let closures=0,verifications=0;
 const runtime=await createFailureRuntime({config,directory,slots},{target:()=>({sample:async()=>({error:'Request timed out',timedOut:false,targetDispatched:false}),sampleSettled:async()=>{closures++;return true;},qualifyBatch:async()=>{verifications++;}}),settled:async()=>false});
 const result=await runtime.trial({directory:join(directory,'op'),task:{id:'dev',caseIds:['value'],files:{'kernel.py':'answer=0'},checker,publicChecker:checker,prompt:'Fixture'},guidance:'same',actorSeconds:1,verificationSeconds:1});assert.equal(result.infrastructure,true);assert.match(result.reason,/Request timed out/);assert.equal(closures,1);assert.equal(verifications,0);assert.equal(await runtime.close(),true);
 const operation=join(directory,'op'),offline={
  settled:async()=>false,target:()=>assert.fail('Custody reconciliation must not launch an actor'),
 };
 const recovered=await createFailureRuntime({config,directory,slots},offline);
 assert.equal(await recovered.close(),true);
 assert.deepEqual(await recovered.settle({directory:operation}),{closed:true});
 const registry=join(directory,'runtime-contexts.json'),rows=JSON.parse(await readFile(registry,'utf8'));
 rows[0].binding.environment.deviceAffinity=[7];await writeFile(registry,JSON.stringify(rows));
 assert.equal((await failureRuntimeSettled({directory,config},offline)).closed,false);

});
test('unavailable Bubblewrap reports preflight failure and closes without actor or checker dispatch',async t=>{
 const root=await fixture(t),directory=join(root,'runtime'),checker=join(root,'checker.py'),bin=join(root,'bin');
 await mkdir(bin);await writeFile(checker,'pass');
 const diagnostic='sandbox fixture: Bubblewrap unavailable';
 await writeFile(join(bin,'bwrap'),`#!/bin/sh\nprintf '%s\n' '${diagnostic}' >&2\nexit 127\n`,{mode:0o700});
 const provider=await scriptedReview(t,root,()=>null);
 const config={target:'cpu',actor:provider.profile,environment:{options:{mode:'local'}}};
 const slots=[{resource:{id:'slot'},physical:1}];
 let actor,dispatches=0,checks=0;
 const dependencies={
  target:()=>({
   sample:async input=>(actor=await nativeSample({...input,targetEnvironment:{execute:async()=>{dispatches++;}}})),
   sampleSettled:async()=>true,
   qualifyBatch:async()=>{checks++;return [];},
  }),
  settled:async()=>false,
 };
 const runtime=await createFailureRuntime({config,directory,slots},dependencies);
 const operation=join(directory,'op'),task={id:'dev',caseIds:['value'],files:{'kernel.py':'answer=0'},checker,publicChecker:checker,prompt:'Fixture'};
 const path=process.env.PATH;process.env.PATH=path?bin+':'+path:bin;
 let result;
 try{
  result=await runtime.trial({directory:operation,task,guidance:'same',actorSeconds:3,verificationSeconds:1});
 }finally{
  if(path===undefined)delete process.env.PATH;else process.env.PATH=path;
 }
 assert.deepEqual(result,{
  infrastructure:true,correct:null,reason:'Native session failed: Local filesystem sandbox unavailable: '+diagnostic,
  errorCode:'LOCAL_SANDBOX_UNAVAILABLE',bucket:'harness',
 });
 assert.equal(actor.targetDispatched,false);
 assert.equal(actor.session,undefined);
 assert.equal(actor.toolCalls,0);
 assert.equal(provider.requests(),0);
 assert.equal(dispatches,0);
 assert.equal(checks,0);
 assert.equal(await runtime.close(),true);
 assert.deepEqual(await runtime.settle({directory:operation}),{closed:true});
});

test('native Pi exhausting the actor grant during provider timeout retry skips scoring',async t=>{
 const {createServer}=await import('node:http'),{once}=await import('node:events');
 const root=await fixture(t),checker=join(root,'checker.py'),credentials_file=join(root,'profile.env');
 await writeFile(checker,'pass');await writeFile(credentials_file,'RSI_TEST_PROVIDER=fixture\n');
 const server=createServer(async(req,res)=>{
  for await(const _ of req){}
  res.writeHead(500,{'Content-Type':'application/json'});res.end(JSON.stringify({error:{message:'Request timed out'}}));
 });
 server.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>{server.closeAllConnections();server.close();});
 const profile={provider:'timeout-fixture',model:'scripted',credential_env:'RSI_TEST_PROVIDER',credentials_file,
  base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'scripted',name:'Fixture',api:'openai-completions',
   reasoning:false,input:['text'],contextWindow:32000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
 const directory=join(root,'runtime'),config={target:'cpu',actor:profile,environment:{options:{mode:'local'}}};let scored=0,result;
 const runtime=await createFailureRuntime({config,directory,slots:[{resource:{id:'slot'},physical:1}]},{
  target:()=>({sample:async input=>(result=await nativeSample(input)),sampleSettled:async()=>true,qualifyBatch:async()=>{scored++;return [];}}),
  settled:async()=>true,
 });
 const receipt=await runtime.trial({directory:join(directory,'op'),task:{id:'dev',caseIds:['value'],files:{'kernel.py':'answer=0'},checker,publicChecker:checker,prompt:'Fixture'},guidance:'same',actorSeconds:1,verificationSeconds:1});
 assert.equal(result.timedOut,true);assert.equal(receipt.infrastructure,true);assert.match(receipt.reason,/Request timed out/);
 assert.equal(scored,0);assert.equal(await runtime.close(),true);
});

test('late trusted verification is not admitted after its own allowance, independently of the actor grant',async t=>{
 const root=await fixture(t),checker=join(root,'checker.py');await writeFile(checker,'pass');const directory=join(root,'private'),config={target:'cpu',actor:{},environment:{options:{mode:'local'}}};let actorSeconds,verificationSeconds;
 const runtime=await createFailureRuntime({config,directory,slots:[{resource:{id:'slot'},physical:1}]},{target:()=>({sample:async input=>{actorSeconds=input.seconds;return {timedOut:false,targetDispatched:false};},sampleSettled:async()=>true,qualifyBatch:async input=>{verificationSeconds=input.seconds;await delay(150);return [];}}),settled:async({directory})=>directory.endsWith('/verification')});
 const result=await runtime.trial({directory:join(directory,'op'),task:{id:'dev',caseIds:['value'],files:{'kernel.py':'answer=0'},checker,publicChecker:checker,prompt:'Fixture'},guidance:'same',actorSeconds:10,verificationSeconds:.1});assert.equal(result.infrastructure,true);assert.equal(result.stage,'verification');assert.equal(actorSeconds,10);assert(verificationSeconds>0&&verificationSeconds<=.1);assert.equal(await runtime.close(),true);
});
test('submission checks replay the starter first and return incorrect controls for revision',async t=>{
 const root=await fixture(t),checker=join(root,'checker.py');await writeFile(checker,'bad checker');const task={id:'dev',files:{'kernel.py':'answer=42'},caseIds:['value'],checker,publicChecker:checker,controls:{reference:{files:{'kernel.py':'answer=42'}},negatives:[{files:{'kernel.py':'answer=42'}}]}};
 const family={id:'family',mechanism:{phase:'candidate-call',type:'numerics'},development:task,transfer:[{...task,id:'layout'},{...task,id:'operation'}]},config={target:'cpu',environment:{options:{mode:'local'}}},directory=join(root,'runtime');let bad=true,starterPasses=false,unknown=false,guardReason=null,calls=[];
 const row=key=>{const correct=bad?null:key!=='starter'||starterPasses;return {key,sourceFilesMatch:true,sourceFilesSha256:{'kernel.py':'sha'},error:bad?'Traceback: ImportError: missing fixture dependency':'',output:JSON.stringify({correct,kernelSha256:'sha',evidence:{phase:'candidate-call',type:'numerics',checkedCases:['value'],caseResults:[{id:'value',correct}],processGuard:{reason:guardReason,allChildrenReaped:true,commandExit:correct?0:1}}})};};
 const runtime=await createFailureRuntime({config,directory,slots:[{resource:{id:'slot'},physical:1}]},{target:()=>({qualifyBatch:async input=>{calls.push(input);if(unknown)throw Object.assign(Error('transport disconnected'),{handle:'remote:fixture:job:unknown'});return [row(input.entries[0].key)];}}),settled:async()=>!unknown});
 await assert.rejects(runtime.checkReferences({family,directory:join(root,'reject'),seconds:30}),error=>error.code==='FAILURE_REFERENCE_REJECTED'&&error.message.includes('ImportError: missing fixture dependency'));assert.equal(calls.length,1);assert.equal(calls[0].entries.length,1);assert.equal(calls[0].entries[0].key,'starter');assert(calls[0].seconds<=30);
 await writeFile(checker,'corrected checker');bad=false;starterPasses=true;await assert.rejects(runtime.checkReferences({family,directory:join(root,'non-reproduction'),seconds:30}),error=>error.code==='FAILURE_REFERENCE_REJECTED'&&error.message.includes('Starter must reproduce'));assert.equal(calls.length,2);starterPasses=false;await writeFile(checker,'repaired starter checker');const result=await runtime.checkReferences({family,directory:join(root,'accept'),seconds:900});assert.equal(result.checked,true);assert.equal(result.references.length,3);assert.equal(new Set(calls.map(c=>c.directory)).size,6);assert(calls.slice(1).every(c=>c.entries.length===1));
 assert(calls[2].seconds>119&&calls[2].seconds<=120);assert(calls.slice(3).every(c=>c.seconds>299&&c.seconds<=300));assert(result.references.every(row=>Number.isFinite(row.timing.elapsedSeconds)));
 bad=true;guardReason='rss-limit';await writeFile(checker,'checker triggering RSS guard');await assert.rejects(runtime.checkReferences({family,directory:join(root,'memory'),seconds:30}),error=>error.code!=='FAILURE_REFERENCE_REJECTED');const count=calls.length;await assert.rejects(runtime.checkReferences({family,directory:join(root,'aborted'),seconds:30,signal:AbortSignal.abort(Error('operator stop'))}),/operator stop/);assert.equal(calls.length,count);
 unknown=true;await assert.rejects(runtime.checkReferences({family,directory:join(root,'unknown'),seconds:30}),error=>error.message==='transport disconnected'&&error.code!=='FAILURE_REFERENCE_REJECTED');assert.equal(await runtime.close(),false);await assert.rejects(runtime.checkReferences({family,directory:join(root,'blocked'),seconds:30}),/No reference slot/);
});

test('A5 checker feedback identifies missing evidence and replays only changed or unchecked controls',async t=>{
 const root=await fixture(t),checker=join(root,'checker.py');
 const control=text=>({files:{'kernel.py':text}}),mechanism={phase:'candidate-call',type:'numerics'};
 const task=id=>({id,caseIds:['value','edge'],files:control('fails').files,checker,publicChecker:checker,
  controls:{reference:control('reference'),negatives:[control('fails')],calibration:control('calibration')}});
 const development=task('dev');development.controls.negatives.push(control('unexpected-pass'));
 development.controls.calibration=control('bad-calibration');
 const family={id:'family',mechanism,development,transfer:[task('layout'),task('operation')]};
 const config={target:'a5',environment:{options:{mode:'local'}}},calls=[];let settled=true;
 const mismatch={edge:{relativeError:1.25,wrongElements:18,totalElements:32}},longDetail={value:{message:'x'.repeat(3000)}};
 await writeFile(checker,'initial checker');
 const runtime=await createFailureRuntime({config,directory:join(root,'runtime'),slots:[{resource:{id:'slot'},physical:1}]},{
  settled:async()=>settled,
  target:()=>({runtimeIdentity:async()=>({target:'a5'}),qualifyBatch:async({entries})=>{
   const [entry]=entries,source=entry.files['kernel.py'];calls.push(entry.key);
   const correct=!['fails','bad-calibration'].includes(source),initial=(await readFile(checker,'utf8'))==='initial checker';
   const evidence={...(correct||initial?{}:mechanism),checkedCases:initial?['value']:['value','edge'],
    detail:initial?longDetail:source==='bad-calibration'?mismatch:undefined,
    caseResults:[{id:'value',correct:correct||source==='bad-calibration'},{id:'edge',correct}],processGuard:{allChildrenReaped:true}};
   return [{key:entry.key,sourceFilesMatch:true,sourceFilesSha256:{'kernel.py':'sha'},output:JSON.stringify({correct,kernelSha256:'sha',evidence})}];
  }}),
 });
 await runtime.identify();
 const submit=()=>runtime.checkReferences({family,directory:join(root,'submission'),seconds:30});
 const reject=async verify=>assert.rejects(submit(),error=>{assert.equal(error.code,'FAILURE_REFERENCE_REJECTED');verify(error.diagnostics.at(-1),error.message);return true;});
 await reject(row=>{
  assert.equal(row.variant,'starter');assert.deepEqual(row.missingFields,['evidence.phase','evidence.type']);
  assert.deepEqual(row.declaredMechanism,mechanism);assert.deepEqual(row.observedMechanism,{phase:null,type:null});
  assert.deepEqual(row.failedCases,['value']);assert.deepEqual(row.unverifiedCases,['edge']);
  assert.equal(row.detail.truncated,true);assert(JSON.stringify(longDetail).startsWith(row.detail.preview));
  assert(row.detail.preview.length<JSON.stringify(longDetail).length);
 });
 await reject(row=>assert.equal(row.reused,true));assert.deepEqual(calls.splice(0),['starter']);
 await writeFile(checker,'checker emits mechanism');
 await reject(row=>{
  assert.equal(row.variant,'negative-1');assert.equal(row.expectedCorrect,false);assert.equal(row.observedCorrect,true);
 });
 assert.deepEqual(calls.splice(0),['starter','reference','reference','reference','negative-1']);
 development.controls.negatives[1]=control('fails');
 await reject((row,message)=>{
  assert.equal(row.variant,'calibration');assert.equal(row.expectedCorrect,true);assert.equal(row.observedCorrect,false);
  assert.deepEqual(row.failedCases,['edge']);assert.deepEqual(row.observedMechanism,mechanism);
  assert.deepEqual(row.detail,mismatch);assert(message.includes(JSON.stringify(mismatch)));
 });
 assert.deepEqual(calls.splice(0),['negative-1','calibration']);
 development.controls.calibration=control('calibration');
 const accepted=await submit();assert.equal(accepted.checked,true);assert.equal(accepted.references.length,3);
 assert(accepted.references.every(row=>row.reused));assert.deepEqual(calls.splice(0),['calibration','negative-0','calibration','negative-0','calibration']);
 settled=false;
 await assert.rejects(submit(),error=>error.code!=='FAILURE_REFERENCE_REJECTED'&&calls.length===0);
});

test('focused qualification runs one task and requires matching negative and calibration',async t=>{
 const root=await fixture(t),checker=join(root,'checker.py');await writeFile(checker,'pass');
 const control=value=>({files:{'kernel.py':value}}),mechanism={phase:'compile',type:'MixedScalar'};
 const task={id:'transfer',caseIds:['value'],files:control('negative').files,checker,publicChecker:checker,
  controls:{reference:control('reference'),calibration:control('calibration'),negatives:[control('negative')]}};
 const family={id:'family',mechanism,development:{id:'development'},transfer:[task,{id:'other'}]};
 const config={target:'cpu',actor:{},environment:{options:{mode:'local'}}},calls=[];
 let calibration=true,negativeType=mechanism.type;
 const runtime=await createFailureRuntime({config,directory:join(root,'private'),slots:[{resource:{id:'slot'},physical:1}]},{target:()=>({
  qualifyBatch:async input=>input.entries.map(entry=>{
   calls.push(entry.key);
   const correct=entry.key==='reference'||entry.key==='calibration'&&calibration;
   const sha='a'.repeat(64);
   return {key:entry.key,sourceFilesMatch:true,sourceFilesSha256:{'kernel.py':sha},
    output:JSON.stringify({correct,kernelSha256:sha,evidence:{phase:mechanism.phase,type:negativeType,
     checkedCases:['value'],caseResults:[{id:'value',correct}],processGuard:{allChildrenReaped:true}}})};
  })
 }),settled:async()=>true});
 const run=id=>runtime.qualify({family,task,directory:join(root,id),remainingSeconds:30});
 assert.equal((await run('pass')).qualified,true);
 assert.deepEqual(calls.splice(0),['reference','negative-0','calibration']);
 calibration=false;const rejected=await run('calibration');assert.equal(rejected.qualified,false);assert.equal(rejected.calibrationPassed,false);
 calibration=true;negativeType='WrongFailure';const unrelated=await run('unrelated');assert.equal(unrelated.qualified,false);assert.equal(unrelated.negativeFailed,false);
});

test('legacy A5 recovery requires matched terminal Pi evidence and no dispatch',async t=>{
 const root=await fixture(t),id=randomUUID(),run=join(root,'skills','priority','failure-first',id),directory=join(run,'runtime');
 const publicRoot=join(run,'public',randomUUID()),trial=join(publicRoot,'micro','development','0-trial'),target=join(trial,'target'),cwd=join(trial,'work');
 const opId='screen--example--development--baseline--0',operation=join(run,'pilot','operations',opId),sessionId=randomUUID();
 const events=join(target,'environment-events.jsonl'),native=join(trial,'session.native.jsonl'),original=join(trial,'2026-11-19T10-04-21Z_'+sessionId+'.jsonl');
 const json=async(path,data)=>{await mkdir(resolve(path,'..'),{recursive:true});await writeFile(path,JSON.stringify(data));};
 const lines=async(path,data)=>writeFile(path,data.map(x=>JSON.stringify(x)).join('\n')+'\n');
 const manifest={identities:{source:'863159a786452c9365e8b35de1962996b0b92e29'}};
 await json(join(run,'experiment.json'),manifest);await json(join(run,'pilot','manifest.json'),manifest);
 await json(join(run,'pilot','pilot.json'),{status:'inconclusive',reason:'Actor custody remains unresolved',operations:{[opId]:{kind:'trial',status:'complete',directory:operation,result:{infrastructure:true,correct:null,reason:'Actor custody remains unresolved',handle:null}}}});
 await json(join(operation,'adapter.json'),{publicRoot,privateRoot:join(operation,'verification'),cwd});
 await json(join(directory,'runtime-contexts.json'),[{root:publicRoot,binding:{target:'a5',environment:{mode:'local',target:'bz-a5-2',deviceAffinity:[5]}}}]);
 await json(join(trial,'sandbox-preflight.json'),{kind:'local-filesystem-sandbox',available:true,exitCode:0,output:''});
 await mkdir(join(target,'.ascend-resources','a'.repeat(64)),{recursive:true});await mkdir(join(target,'.executions'));await mkdir(cwd,{recursive:true});
 const attempt='micro-'+hash(target).slice(0,24),journal=[{type:'resource-policy'},{type:'environment-prepared',data:{project:attempt,attempt}}];
 await lines(events,journal);
 const record=[{type:'session',id:sessionId,cwd},{type:'message',message:{role:'user',content:[{type:'text',text:'The model, files and bash are local. Use rsi_target'}]}},{type:'message',message:{role:'assistant',stopReason:'error',errorMessage:'Request timed out',content:[]}}];
 await lines(original,record);await lines(native,record);
 const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
 state.db.exec('CREATE TABLE IF NOT EXISTS skill_backlog_runs(id TEXT PRIMARY KEY,status TEXT,owner TEXT,checkpoint TEXT)');
 const realOwner=await processIdentity(),child=spawn(process.execPath,['-e','setInterval(()=>{},1000)']);
 t.after(()=>child.kill());const exited=new Promise((resolve,reject)=>{child.once('exit',resolve);child.once('error',reject);});
 const stoppedOwner=await processIdentity(child.pid);child.kill();await exited;
 const checkpoint={directory:run,runtimeDirectory:directory,workflow:'failure-first-v1',phase:'evaluate'};
 state.db.prepare('INSERT INTO skill_backlog_runs(id,status,owner,checkpoint) VALUES(?,?,?,?)').run(id,'running',JSON.stringify(stoppedOwner),JSON.stringify(checkpoint));
 const reconcile=async()=> (await failureRuntimeSettled({directory,config:{root,target:'a5'}},{settled:async()=>false})).closed;
 assert.equal(await reconcile(),true);
 await rm(native);assert.equal(await reconcile(),false);await writeFile(native,'{broken\n');assert.equal(await reconcile(),false);await lines(native,record);
 await lines(events,[...journal,{type:'execution-starting',data:{operation:'unexpected'}}]);assert.equal(await reconcile(),false);await lines(events,journal);
 await json(join(target,'.ascend-command-lease.json'),{token:'unexpected'});assert.equal(await reconcile(),false);await rm(join(target,'.ascend-command-lease.json'));
 const dispatched=[...record.slice(0,2),{type:'message',message:{role:'assistant',content:[{type:'toolCall',id:'target',name:'rsi_target',arguments:{cmd:'true'}}]}},{type:'message',message:{role:'toolResult',toolCallId:'target',toolName:'rsi_target'}},record.at(-1)];
 await lines(native,dispatched);await lines(original,dispatched);assert.equal(await reconcile(),false);await lines(native,record);await lines(original,record);
 await json(join(run,'pilot','manifest.json'),{identities:{source:'different'}});assert.equal(await reconcile(),false);await json(join(run,'pilot','manifest.json'),manifest);
 state.db.prepare('UPDATE skill_backlog_runs SET owner=? WHERE id=?').run(JSON.stringify(realOwner),id);
 assert.equal(await reconcile(),false);
});
