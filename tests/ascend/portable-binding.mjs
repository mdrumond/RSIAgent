import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,copyFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {State} from '../../controller/state.mjs';
import {processIdentity,operationsStopped,ownerStopped,attemptStopped} from '../../controller/lifecycle.mjs';
import {bindPortable} from '../../cli/portable-binding.mjs';
import {main} from '../../cli/main.mjs';
import {operate} from '../../cli/runtime.mjs';
const execute=promisify(execFile);
const save=(path,value)=>writeFile(path,JSON.stringify(value));
const hash=async path=>createHash('sha256').update(await readFile(path)).digest('hex');
const plan=[{id:'entry',phase:'phase1',new_projects:1}];
async function fixture(t,{payload=false}={}) {
 const root=await mkdtemp(join(tmpdir(),'portable-binding-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const old='/source-campaign',source=join(root,'original.sqlite'),file=join(root,'state.sqlite');
 const state=new State(source);state.db.exec(`CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT NOT NULL);
 CREATE TABLE storage_campaign(id TEXT PRIMARY KEY);INSERT INTO storage_campaign VALUES('campaign');
 CREATE TABLE storage_artifacts(path TEXT,encoding TEXT,descriptor TEXT);INSERT INTO storage_artifacts VALUES('/source-campaign/attempts/log','restic','archive');`);
 state.add({id:'p',goal:'retained pending numerical project',actor_seconds:600,source:old+'/projects/p/source'});
 state.add({id:'q',goal:'queued later project',actor_seconds:600,source:old+'/projects/q/source'});
 state.add({id:'done',goal:'historical completed project',actor_seconds:600});
 const completed=state.admit('done',{pid:2147483646,start:'1',boot:'source'});state.checkpoint(completed.id,23.5,null);state.finish(completed.id,'passed');
 state.add({id:'invalid',goal:'frozen invalid project',actor_seconds:600});state.db.prepare('UPDATE projects SET frozen=1 WHERE id=?').run('invalid');
 const attempt=state.admit('p',{pid:2147483647,start:'0',boot:'source'});state.checkpoint(attempt.id,17.25,old+'/attempts/session.jsonl');
 const handle='local:exec-health';
 state.event(attempt.id,'execution-starting',{operation:'op',startElapsed:17});
 state.event(attempt.id,'execution-submitting',{operation:'op',submission:'s',kind:'run',accountingExcluded:!payload});
 state.event(attempt.id,'execution-submitted',{operation:'op',submission:'s',handle});
 state.event(attempt.id,'execution-handle',{operation:'op',handle,purpose:payload?'payload':'resource-health'});
 state.db.prepare('INSERT INTO control VALUES (?,?)').run('campaign-budget',JSON.stringify({cycle:'cycle',elapsed:31,seconds:900,running:true,boot:'source',monotonic:10}));
 state.db.prepare('INSERT INTO control VALUES (?,?)').run('campaign-plan',JSON.stringify({plan,index:0,started:0}));
 state.close();await copyFile(source,file);
 const portable=new State(file);portable.db.exec('UPDATE attempts SET owner=NULL;UPDATE projects SET frozen=1');portable.close();
 await mkdir(join(root,'attempts',attempt.id),{recursive:true});
 const resource={id:'old-slot',available:true,metadata:{kind:'host-slot',hostSlot:2,pool:'source'}};
 await save(join(root,'attempts',attempt.id,'job.json'),{resource,actor_seconds:600,project:{source:old+'/projects/p/source'}});
 await mkdir(join(root,'source-history/repo'),{recursive:true});await save(join(root,'source-history/repo/rsi-source.json'),{source:old+'/projects/p/source'});
 const config={format:1,revision:1,paused:true,binding_required:true,environment:null,python:'/old/python',local_execution:null,concurrency:4,
  campaign_seconds:900,campaign_plan:plan,storage:{campaign_id:'campaign'},skills:{activeVersion:'baseline'},projects:[]};
 await save(join(root,'instance.json'),config);
 await save(join(root,'restore.json'),{complete:true,paused:true,binding_required:true,paths_need_rebase:true,source_package:'original-package.json'});
 const packagePath=join(root,'original-package.json');await save(packagePath,{format:2,engine:'restic',scope:'campaign',complete:true,missing:[],campaign:[{id:'campaign'}],
  paths:{[old+'/projects/p/source']:'projects/p/source',[old+'/projects/q/source']:'projects/q/source'},repositories:[{kind:'source-git',path:'source-history/repo'}]});
 const audit=join(root,'shutdown.json');await save(audit,{stopped:true,recordedScopes:1,remainingScopesStopped:0});
 const destination={id:'new-slot',available:true,metadata:{kind:'host-slot',hostSlot:2,pool:'destination'}};
 const input={python:'/destination/python',environment:{module:'machinery/environment.mjs',options:{target:'destination'}},resources:[destination],
  packageSha256:await hash(packagePath),sourceShutdown:{path:audit,sha256:await hash(audit)},sourceState:{path:source,sha256:await hash(source)}};
 return {root,input,id:attempt.id,handle,source,destination};
}

test('portable bind relocates paths and resumes only the retired health prefix with retained identities and allocations',async t=>{
 const {root,input,id,handle,destination}=await fixture(t);
 const config=await bindPortable(root,input),state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
 assert.equal(config.binding_required,false);assert.equal(config.local_execution,null);assert.equal(config.concurrency,4);
 assert.equal(state.attempt(id).elapsed,17.25);assert.equal(state.attempt(id).outcome,null);
 assert.equal(state.attempt(id).session,join(root,'attempts/session.jsonl'));
 assert.deepEqual(JSON.parse(state.attempt(id).owner),await processIdentity());
 assert.equal(await ownerStopped(state.attempt(id).owner),false,'binder retains a real live administrative claim until it exits');
 assert.equal(state.project('q').frozen,false);assert.equal(state.project('invalid').frozen,true);
 const completed=state.db.prepare("SELECT * FROM attempts WHERE project='done'").get();
 assert.equal(completed.elapsed,23.5);assert.equal(completed.outcome,'passed');assert.deepEqual(JSON.parse(completed.owner),{pid:2147483646,start:'1',boot:'source'});
 assert.equal(state.project('p').spec.source,join(root,'projects/p/source'));
 const budget=JSON.parse(state.db.prepare("SELECT value FROM control WHERE key='campaign-budget'").get().value);
 assert.deepEqual(budget,{cycle:'cycle',elapsed:31,seconds:900,running:false,boot:'source',monotonic:10});
 assert.deepEqual(JSON.parse(state.db.prepare("SELECT value FROM control WHERE key='campaign-plan'").get().value),{plan,index:0,started:0});
 assert.equal(state.db.prepare('SELECT encoding FROM storage_artifacts').get().encoding,'plain');
 const job=JSON.parse(await readFile(join(root,'attempts',id,'job.json')));assert.deepEqual(job.resource,destination);
 assert.equal(job.actor_seconds,600);assert.equal(JSON.parse(await readFile(join(root,'source-history/repo/rsi-source.json'))).source,join(root,'projects/p/source'));
 assert.equal(await operationsStopped(state,state.attempt(id),{observe:()=>assert.fail('source-side health handle observed on destination')}),true);
 const binding=state.events(id).find(e=>e.type==='portable-bound').data;assert.deepEqual(binding.healthHandles,[handle]);
 state.event(id,'execution-handle',{handle:'local:exec-new'});let called=0;
 assert.equal(await operationsStopped(state,state.attempt(id),{observe:async()=>{called++;return {state:'running'};}}),false);assert.equal(called,1);
 await assert.rejects(bindPortable(root,input),/unbound paused/);
});

for(const kind of ['payload','native-bash','unidentified','unknown-start','unknown-unresolved','missing-operation','wrong-audit','incomplete-audit','foreign-state','wrong-slot','active-request'])test('binding rejects '+kind+' without changing portable checkpoint',async t=>{
 const f=await fixture(t,{payload:kind==='payload'}),state=new State(join(f.root,'state.sqlite'));
 if(kind==='native-bash')state.event(f.id,'native-bash-started',{operation:'shell'});
 if(kind==='unidentified')state.event(f.id,'execution-submitting',{operation:'x',submission:'x',kind:'run',accountingExcluded:true});
 if(kind==='unknown-start')state.event(f.id,'execution-starting',{operation:'unknown'});
 if(kind==='unknown-unresolved')state.event(f.id,'execution-unresolved',{operation:'unknown'});
 if(kind==='missing-operation')state.event(f.id,'execution-unresolved',{});
 if(kind==='active-request')state.db.exec("CREATE TABLE requests(status TEXT);INSERT INTO requests VALUES('queued')");
 if(kind==='wrong-audit')f.input.sourceShutdown.sha256='0'.repeat(64);
 if(kind==='incomplete-audit') {await save(f.input.sourceShutdown.path,{stopped:false,recordedScopes:1,remainingScopesStopped:1});f.input.sourceShutdown.sha256=await hash(f.input.sourceShutdown.path);}
 if(kind==='foreign-state') {
  const s=new State(f.source);s.db.prepare('UPDATE storage_campaign SET id=?').run('foreign');s.close();f.input.sourceState.sha256=await hash(f.source);
 }
 if(kind==='wrong-slot')f.input.resources[0].metadata.hostSlot=3;
 const before=state.attempt(f.id),events=state.events(f.id);state.close();
 const config=await readFile(join(f.root,'instance.json')),job=await readFile(join(f.root,'attempts',f.id,'job.json'));
 await assert.rejects(bindPortable(f.root,f.input));
 const next=new State(join(f.root,'state.sqlite'));t.after(()=>next.close());assert.deepEqual(next.attempt(f.id),before);assert.deepEqual(next.events(f.id),events);
 assert.deepEqual(await readFile(join(f.root,'instance.json')),config);assert.deepEqual(await readFile(join(f.root,'attempts',f.id,'job.json')),job);
});

test('ordinary CLI cannot execute an unbound portable campaign',async t=>{
 const {root}=await fixture(t);
 await assert.rejects(main(['resume','--instance',root,'--detach']),/Bind the portable campaign/);
 await assert.rejects(operate('wave',{binding_required:true}),/Bind the portable campaign/);
});

test('already relocated standard restore still relocates job paths; altered binding history fails closed',async t=>{
 const {root,input,id}=await fixture(t),state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
 await save(join(root,'restore.json'),{complete:true,paused:true,binding_required:true,source_package:'original-package.json'});
 state.db.prepare('UPDATE attempts SET session=? WHERE id=?').run(join(root,'attempts/session.jsonl'),id);
 state.db.prepare("UPDATE projects SET spec=replace(spec,?,?)").run('/source-campaign',root);
 await bindPortable(root,input);
 const job=JSON.parse(await readFile(join(root,'attempts',id,'job.json')));assert.equal(job.project.source,join(root,'projects/p/source'));
 const row=state.db.prepare("SELECT sequence,data FROM events WHERE attempt=? AND type='execution-submitting'").get(id);
 state.db.prepare('UPDATE events SET data=? WHERE sequence=?').run(JSON.stringify({...JSON.parse(row.data),accountingExcluded:false}),row.sequence);
 let reconciled=0;assert.equal(await operationsStopped(state,state.attempt(id),{reconcileOperation:async()=>{reconciled++;return {closed:false};}}),false);assert.equal(reconciled,1);
});

test('concurrent real binder processes have one winner; its exited identity permits ordinary resume',async t=>{
 const {root,input,id}=await fixture(t),file=join(root,'binding.json');await save(file,input);
 const entry=new URL('../../cli/main.mjs',import.meta.url).href;
 const code=`import {main} from ${JSON.stringify(entry)};await main(['bind','--instance',process.argv[1],'--config',process.argv[2]]);`;
 const results=await Promise.allSettled([1,2].map(()=>execute(process.execPath,['--input-type=module','-e',code,root,file])));
 assert.equal(results.filter(x=>x.status==='fulfilled').length,1);
 const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
 assert.equal(state.events(id).filter(e=>e.type==='portable-bound').length,1);
 assert.equal(await ownerStopped(state.attempt(id).owner),true);
 const resumed=await state.resume(id,attempt=>attemptStopped(state,attempt,{observe:()=>assert.fail('old source handle observed')}),await processIdentity());
 assert.equal(resumed.id,id);assert.equal(resumed.elapsed,17.25);assert.equal(resumed.session,join(root,'attempts/session.jsonl'));assert.equal(resumed.outcome,null);
});
