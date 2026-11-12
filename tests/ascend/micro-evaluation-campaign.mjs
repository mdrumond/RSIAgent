import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {State} from '../../controller/state.mjs';
import {stages} from '../../controller/workflow.mjs';
import {tune} from '../../controller/skill-evolution.mjs';
import {readConfig,updateConfig,upgrade} from '../../cli/instance.mjs';
import {runMicroeval} from '../../cli/skill-microeval.mjs';
import {compare,restartMicroEvaluation,settled} from '../../cli/micro-evaluation.mjs';
import {tasks} from '../../cli/microtasks/tasks.mjs';
import {recoveryRepair} from './microtask-fixture.mjs';
import {evaluationArchiveStatus,enqueueCompletedEvaluation} from '../../cli/evaluation-archives.mjs';
const json=async p=>JSON.parse(await readFile(p,'utf8'));
const python=process.env.RSI_TEST_PYTHON??'python3';
const fixed=`import itertools
def make_launcher(compile_kernel):
 cache={}
 def launch(x,scale):
  n=len(x)
  if n not in cache:cache[n]=compile_kernel(n)
  return cache[n](x,scale)
 return launch
def add_rows(storage,rows,columns,pitch,delta):
 out=list(storage)
 for r in range(rows):
  for c in range(columns):out[r*pitch+c]+=delta
 return out
def logical_flatten(storage,shape,strides,offset=0):
 return [storage[offset+sum(i*s for i,s in zip(index,strides))] for index in itertools.product(*(range(n) for n in shape))]
`+recoveryRepair;
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'micro-campaign-'));t.after(()=>rm(root,{recursive:true,force:true}));
 for(const version of ['baseline','candidate']){
  const directory=join(root,'skills','versions',version);await mkdir(directory,{recursive:true});
  const map=Object.fromEntries(['global-kdb',...stages].map(scope=>[scope,scope+'.md']));
  for(const name of Object.values(map))await writeFile(join(directory,name),'Keep the source and verifier contract.');
  await writeFile(join(directory,'skill-map.json'),JSON.stringify(map));
 }
 const config={format:1,root,mode:'phase1',python,projects:[],skills:{activeVersion:'baseline',background:true,tuningProjectIds:['fa','gdn'],archiveEvaluations:true},environment:{module:'environment.mjs',options:{}}};
 await writeFile(join(root,'instance.json'),JSON.stringify(config));
 await mkdir(join(root,'machinery'));await writeFile(join(root,'environment.mjs'),'export function createEnvironment(){return {resources:async()=>[]}}');
 const db=new State(join(root,'state.sqlite'));t.after(()=>db.close());db.db.exec('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
 const set=(key,value)=>db.db.prepare('INSERT OR REPLACE INTO control VALUES (?,?)').run(key,JSON.stringify(value));
 set('pause',true);set('campaign-budget',{cycle:'original-cycle',seconds:10000,elapsed:123,running:false});
 db.add({id:'retained',goal:'Original project',actor_seconds:100});const a=db.admit('retained',{pid:2147483647,start:'0',boot:'old'});db.checkpoint(a.id,23,'original-session');
 const oldRoot=join(root,'skills','evaluations','previous');await mkdir(oldRoot,{recursive:true});await writeFile(join(oldRoot,'round.json'),JSON.stringify({status:'interrupted',samples:[],elapsedSeconds:77,allocatedSeconds:500}));
 const previous={history:[{id:'evidence',projectId:'fa',kind:'original',phase:'phase2',completed:true,events:[]}],lastEvaluated:0,candidate:'candidate',candidateBase:'baseline',allocations:{retained:{seconds:500}},pending:{active:'baseline',candidate:'candidate',evaluationRoot:oldRoot,config,projectIds:['fa','gdn'],runs:2,evaluationSeconds:500}};
 await writeFile(join(root,'skills','evolution.json'),JSON.stringify(previous));
 return {root,config,db,set,previous,oldRoot,attempt:a.id};
}
test('restart archives previous A/B and preserves projects, sessions, allocations and campaign cycle',async t=>{
 const f=await fixture(t),before=f.db.attempt(f.attempt),budget=f.db.db.prepare("SELECT value FROM control WHERE key='campaign-budget'").get().value;
 const result=await restartMicroEvaluation(f.config);
 const after=await json(join(f.root,'skills','evolution.json')),config=await readConfig(f.root);
 assert.deepEqual((await json(result.archived)).evolution,f.previous);assert.equal(await readFile(join(f.oldRoot,'round.json'),'utf8'),JSON.stringify({status:'interrupted',samples:[],elapsedSeconds:77,allocatedSeconds:500}));
 assert.equal(after.pending.candidate,'candidate');assert.deepEqual(after.history,f.previous.history);assert.deepEqual(after.allocations,f.previous.allocations);
 assert.deepEqual(after.pending.projectIds,tasks.map(t=>t.id));assert.equal(after.pending.evaluationSeconds,900);
 assert.equal(config.skills.activeVersion,'baseline');assert.equal(config.skills.microEvaluation.margin,.05);assert.equal(config.skills.runtimeModule,'machinery/cli/micro-evaluation.mjs');
 assert.deepEqual(f.db.attempt(f.attempt),before);assert.equal(f.db.db.prepare("SELECT value FROM control WHERE key='campaign-budget'").get().value,budget);
});
test('restart refuses an unpaused campaign and a live writer without changing evolution',async t=>{
 const f=await fixture(t);f.set('pause',false);await assert.rejects(restartMicroEvaluation(f.config),/Pause campaign/);
 assert.deepEqual(await json(join(f.root,'skills','evolution.json')),f.previous);
 f.set('pause',true);const {processIdentity}=await import('../../controller/lifecycle.mjs');f.set('owner',await processIdentity());
 await assert.rejects(restartMicroEvaluation(f.config),/Stop campaign/);assert.deepEqual(await json(join(f.root,'skills','evolution.json')),f.previous);
});
test('resumable samples retain completed results, session identity and remaining elapsed after normal abort',async t=>{
 const f=await fixture(t),checker=join(f.root,'check.py');await writeFile(checker,"import runpy,sys,pathlib\nassert runpy.run_path(str(pathlib.Path(sys.argv[1])/'kernel.py'))['value']==42\n");
 const options={directory:join(f.root,'micro'),skills:{active:join(f.root,'skills/versions/baseline'),candidate:join(f.root,'skills/versions/candidate')},tasks:[{id:'repair',prompt:'Repair value',scopes:['implementation'],files:{'kernel.py':'value=0\n'},checker}],profile:{provider:'fixture',model:'fixture'},runs:2,seconds:10,totalSeconds:100,resume:true};
 const abort=new AbortController();let calls=0;
 await assert.rejects(runMicroeval({...options,signal:abort.signal},{sample:async({cwd})=>{calls++;await writeFile(join(cwd,'kernel.py'),'value=42\n');if(calls===2)abort.abort();return {tokens:10,session:'retained-session'};}}),{name:'AbortError'});
 const partial=await json(join(options.directory,'results.json'));assert.equal(partial.results.length,1);assert.equal(Object.keys(partial.partials).length,1);assert.equal(await settled({instanceRoot:f.root}),true);
 const observed=[];const result=await runMicroeval(options,{sample:async request=>{observed.push(request);await writeFile(join(request.cwd,'kernel.py'),'value=42\n');return {tokens:10};}});
 assert.equal(observed.length,3);assert.equal(observed[0].resumeSession,'retained-session');assert.ok(observed[0].seconds<10);assert.equal(result.results.length,4);assert.ok(result.results.every(x=>x.correct));assert.ok(result.elapsedSeconds>=partial.elapsedSeconds);
 result.runningSample={key:'unknown'};await writeFile(join(options.directory,'results.json'),JSON.stringify(result));await assert.rejects(runMicroeval(options),/unresolved/);
});
test('actual native Pi comparison covers recovery decisions and resumes without provider replay',async t=>{
 const f=await fixture(t);let requests=0;
 const server=createServer(async(req,res)=>{let raw='';for await(const x of req)raw+=x;const body=JSON.parse(raw);requests++;const used=body.messages.some(m=>m.role==='tool');
  res.writeHead(200,{'Content-Type':'text/event-stream'});const frame=x=>res.write('data: '+JSON.stringify(x)+'\n\n');
  frame({id:'fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:{role:'assistant',...(used?{content:'done'}:{tool_calls:[{index:0,id:'repair',type:'function',function:{name:'write',arguments:JSON.stringify({path:'kernel.py',content:fixed})}}]})},finish_reason:null}]});
  frame({id:'fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:{},finish_reason:used?'stop':'tool_calls'}],usage:{prompt_tokens:11,completion_tokens:7,total_tokens:18}});res.end('data: [DONE]\n\n');
 });server.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>{server.closeAllConnections();server.close();});
 const credentials_file=join(f.root,'fixture.env');await writeFile(credentials_file,'MICRO_FIXTURE=fixture\n');
 f.config.actor={provider:'micro-campaign-fixture',model:'scripted',credential_env:'MICRO_FIXTURE',credentials_file,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:32000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
 await writeFile(join(f.root,'instance.json'),JSON.stringify(f.config));
 await restartMicroEvaluation(f.config,{settings:{margin:1}});const pending=(await json(join(f.root,'skills/evolution.json'))).pending;
 const round=await compare({pending});assert.equal(round.status,'completed');assert.equal(round.samples.length,16);assert.ok(round.samples.every(x=>x.result.correct&&x.result.tokens===36));assert.equal(requests,32);
 assert.equal(await settled({instanceRoot:pending.evaluationRoot}),true);await compare({pending});assert.equal(requests,32);
 const config=await readConfig(f.root);assert.equal(await enqueueCompletedEvaluation(config,{lastEvaluationRoot:pending.evaluationRoot}),false);assert.equal((await evaluationArchiveStatus(config,{lastEvaluationRoot:pending.evaluationRoot})).hasWork,false);
 // Pre-completed behavioral results enter the normal promotion/CAS path, never the full-project evaluator.
 const reportPath=join(pending.evaluationRoot,'micro/results.json'),report=await json(reportPath);for(const x of report.results){x.elapsedSeconds=1;x.tokens=10;}await writeFile(reportPath,JSON.stringify(report));
 const decision=await tune({instanceRoot:f.root,config:{...config,mode:'phase2'},background:true,backgroundSeconds:100,backgroundResource:{id:'slot'},evaluate:()=>{throw Error('Heavy evaluator must not run');}});
 assert.equal(decision.promote,true);assert.equal((await readConfig(f.root)).skills.activeVersion,'candidate');assert.equal((await json(join(f.root,'skills/evolution.json'))).pending,null);
 const nextConfig=await updateConfig(f.root,c=>({...c,tune_every:1}));
 const next=await tune({instanceRoot:f.root,config:{...nextConfig,mode:'phase2'},background:true,backgroundSeconds:100,backgroundResource:{id:'slot'},
  task:{id:'next-evidence',projectId:'gdn',kind:'original',phase:'phase2',completed:true,events:[]},generate:async()=>[],evaluate:()=>{throw Error('Heavy evaluator must not run');}});
 assert.notEqual(next.candidate,'candidate');assert.equal(next.evaluationStatus,'completed');assert.equal(requests,64);
 assert.equal((await json(join(f.root,'skills/evolution.json'))).history.length,2);
});
test('retained three-task comparisons keep their original manifest and never gain new samples',async t=>{
 const f=await fixture(t);await restartMicroEvaluation(f.config);
 const pending=(await json(join(f.root,'skills/evolution.json'))).pending;
 pending.projectIds=tasks.slice(0,3).map(task=>task.id);pending.config.actor={provider:'fixture',model:'fixture'};
 let calls=0;const sample=async({cwd})=>{calls++;await writeFile(join(cwd,'kernel.py'),fixed);return {tokens:10};};
 const first=await compare({pending},{sample});assert.equal(first.samples.length,12);assert.equal(calls,12);
 const manifest=join(pending.evaluationRoot,'micro/manifest.json'),before=await readFile(manifest,'utf8');
 assert.equal(JSON.parse(before).seconds,90);assert.equal(first.allocatedSeconds,900);
 const next=await compare({pending},{sample:async()=>{throw Error('Completed native samples must not replay');}});
 assert.deepEqual(next.samples,first.samples);assert.ok(next.elapsedSeconds>=first.elapsedSeconds);
 assert.equal(await readFile(manifest,'utf8'),before);
 pending.projectIds=tasks.map(task=>task.id);
 await assert.rejects(compare({pending},{sample}),/Frozen micro evaluation changed/);
 assert.equal(calls,12);assert.equal(await readFile(manifest,'utf8'),before);
});
test('official machinery upgrade carries runnable microtask fixtures',async t=>{
 const f=await fixture(t);await upgrade(f.root,resolve('.'));

 const module=await import('file://'+join(f.root,'machinery/cli/micro-evaluation.mjs'));
 assert.deepEqual(module.projectIds(),tasks.map(x=>x.id));
 const packaged=await import('file://'+join(f.root,'machinery/cli/microtasks/tasks.mjs'));
 for(const task of packaged.tasks)assert.ok((await readFile(task.checker,'utf8')).length>0);
});
test('normal abort resumes the actual native Pi session and its edited source',async t=>{
 const f=await fixture(t),abort=new AbortController();let requests=0;
 const server=createServer(async(req,res)=>{let raw='';for await(const x of req)raw+=x;const body=JSON.parse(raw);requests++;const used=body.messages.some(m=>m.role==='tool');
  if(requests===2)abort.abort();
  res.writeHead(200,{'Content-Type':'text/event-stream'});const frame=x=>res.write('data: '+JSON.stringify(x)+'\n\n');
  frame({id:'fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:{role:'assistant',...(used?{content:'done'}:{tool_calls:[{index:0,id:'repair',type:'function',function:{name:'write',arguments:JSON.stringify({path:'kernel.py',content:fixed})}}]})},finish_reason:null}]});
  frame({id:'fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:{},finish_reason:used?'stop':'tool_calls'}],usage:{prompt_tokens:11,completion_tokens:7,total_tokens:18}});res.end('data: [DONE]\n\n');
 });server.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>{server.closeAllConnections();server.close();});
 const credentials_file=join(f.root,'fixture.env');await writeFile(credentials_file,'MICRO_RESUME=fixture\n');
 const profile={provider:'micro-native-resume',model:'scripted',credential_env:'MICRO_RESUME',credentials_file,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:32000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}};
 const options={directory:join(f.root,'native-resume'),skills:{active:join(f.root,'skills/versions/baseline'),candidate:join(f.root,'skills/versions/candidate')},tasks:[tasks[0]],profile,python,runs:1,seconds:20,totalSeconds:100,resume:true};
 await assert.rejects(runMicroeval({...options,signal:abort.signal}),{name:'AbortError'});
 const before=await json(join(options.directory,'results.json')),partial=Object.values(before.partials)[0];assert.equal(before.results.length,0);assert.ok(partial.session);
 const result=await runMicroeval(options);assert.equal(requests,5);assert.ok(result.results.every(r=>r.correct));assert.equal(result.results[0].session,partial.session);
 assert.equal(result.results[0].tokens,36);assert.ok(result.elapsedSeconds>=before.elapsedSeconds);
});
test('elapsed micro comparison terminates inconclusively without renewing its allocation',async t=>{
 const f=await fixture(t);await restartMicroEvaluation(f.config,{settings:{totalSeconds:.01}});
 const pending=(await json(join(f.root,'skills/evolution.json'))).pending;pending.config.actor={provider:'fixture',model:'fixture'};
 let calls=0;const round=await compare({pending},{sample:async()=>{calls++;await new Promise(r=>setTimeout(r,20));return {tokens:10};}});
 assert.equal(round.status,'completed');assert.equal(round.termination,'elapsed');assert.equal(calls,1);assert.equal(round.allocatedSeconds,.01);
 const {decide}=await import('../../cli/micro-evaluation.mjs');
 assert.equal(decide(round.samples.map(x=>({...x.result,...x})),pending.projectIds,pending.runs,pending.config).promote,false);
 await compare({pending},{sample:async()=>{throw Error('Expired allocation must not dispatch');}});assert.equal(calls,1);
});
test('unknown micro execution stays fenced through actual background settlement and restart',async t=>{
 const f=await fixture(t);await restartMicroEvaluation(f.config);
 const file=join(f.root,'skills/evolution.json'),evolution=await json(file),pending=evolution.pending;
 pending.config.actor={provider:'fixture',model:'fixture'};pending.config.skills.runtimeModule=resolve('cli/micro-evaluation.mjs');await writeFile(file,JSON.stringify(evolution));
 await assert.rejects(compare({pending},{sample:async()=>{throw Error('Unknown native cleanup');}}),/Unknown native cleanup/);
 const round=await json(join(pending.evaluationRoot,'round.json'));assert.ok(round.runningAt>0);assert.equal(await settled({instanceRoot:pending.evaluationRoot}),false);
 const {backgroundSkillsSettled}=await import('../../cli/skill-background.mjs');
 assert.equal(await backgroundSkillsSettled(await readConfig(f.root)),false);
 const before=await readFile(file,'utf8');await assert.rejects(restartMicroEvaluation(await readConfig(f.root)),/Stop campaign/);assert.equal(await readFile(file,'utf8'),before);
});
test('micro allocation grants reject before changing pending, allocations or configuration',async t=>{
 const f=await fixture(t);await restartMicroEvaluation(f.config);
 const file=join(f.root,'skills/evolution.json'),before=await readFile(file,'utf8'),config=await readConfig(f.root),beforeConfig=await readFile(join(f.root,'instance.json'),'utf8');
 await assert.rejects(tune({instanceRoot:f.root,config,background:true,backgroundSeconds:100,evaluationSeconds:500,allocationId:'extra'}),/allocations are fixed/);
 assert.equal(await readFile(file,'utf8'),before);assert.equal(await readFile(join(f.root,'instance.json'),'utf8'),beforeConfig);
});
