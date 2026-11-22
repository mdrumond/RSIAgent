import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,cp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {createEnvironment} from '../../integrations/environments/index.mjs';
const exec=promisify(execFile);

// Execute the actual generated Python supervisor. Only native CPL transport is
// replaced with copies and local processes; these are not NPU measurements.
async function fixture(t,{logs='valid'}={}){
 const root=await mkdtemp(join(tmpdir(),'terminal-results-')),project=join(root,'project'),workspace=join(root,'remote');
 await mkdir(project);await mkdir(workspace);await writeFile(join(project,'input'),'kept');
 const jobs=new Map(),calls=[],events=[];let serial=0,unknown=false;
 t.after(async()=>{await Promise.all([...jobs.values()].map(j=>j.done));await rm(root,{recursive:true,force:true});});
 const context={instancePath:root,sdk:{createLocalBashOperations:()=>({})},exclude:async(_,fn)=>fn(),event:async(type,data)=>events.push({type,data}),client:async args=>{
  calls.push(args);
  if(args[0]==='capabilities')return {state:'available',backend:'remote-agent'};
  if(args[0]==='preflight')return {state:'completed'};
  if(args[0]==='upload'){await mkdir(dirname(join(workspace,args[3])),{recursive:true});await cp(args[2],join(workspace,args[3]),{recursive:true});return {state:'completed'};}
  if(args[0]==='download'){await cp(join(workspace,args[2]),args[3]);return {state:'completed'};}
  if(args[0]==='run'){
   const handle='remote:fixture:job:'+ ++serial,job={state:'running',stdout:'',exit:null,setup:args[args.indexOf('--file')+1].endsWith('sync.sh')};jobs.set(handle,job);
   job.done=exec('bash',[args[args.indexOf('--file')+1]],{cwd:workspace,maxBuffer:1024*1024}).then(r=>{job.stdout=r.stdout;job.exit=0;},e=>{job.stdout=e.stdout;job.exit=e.code;}).finally(()=>{job.state=job.exit===0?'completed':'failed';});
   return {state:'queued',handle};
  }
  if(args[0]==='observe'){const job=jobs.get(args[1]);return {state:unknown&&!job.setup?'observation-unavailable':job.state,exit:job.exit};}
  if(args[0]==='logs'){
   const content=jobs.get(args[1]).stdout;
   if(logs==='malformed')return {state:'completed',content:'not JSON'};
   if(logs==='missing')return {state:'completed',content:''};
   if(logs==='wrong-key'){const data=JSON.parse(content);return {state:'completed',content:JSON.stringify({...data,key:'other'})};}
   return {state:'completed',content};
  }
  throw Error('Unexpected CPL operation '+args[0]);
 }};
 const options={mode:'cpl',target:'fixture',workspacePath:workspace,remoteDirectory:'owned',pollMs:1};
 const environment=await createEnvironment(options,context);
 await environment.prepare({project:{source:project},attempt:{id:'a'}});
 return {root,project,workspace,environment,options,context,calls,events,jobs,unknown:value=>{unknown=value;}};
}

test('completed native result supplies both streams, timing and admission without file transfers, including resume',async t=>{
 const f=await fixture(t);let output='';
 const result=await f.environment.execute(`python3 -c 'import os,json; json.dump({"token":"owned","decision":"admitted","released":True},open(os.environ["RSI_LEASE_RECEIPT"],"w")); print("answer"); print("diagnostic",file=__import__("sys").stderr)'`,f.project,{leaseReceipt:true,admissionToken:'owned',onData:data=>{output+=data;}});
 assert.equal(result.settled,true);assert.equal(result.exitCode,0);assert.equal(result.timingSource,'wrapper');assert.ok(result.executionSeconds>=0);
 assert.equal(output,'answer\ndiagnostic\n');
 assert.deepEqual(await f.environment.readAdmission(result.handle),{token:'owned',decision:'admitted',released:true});
 assert.equal(f.calls.filter(c=>c[0]==='download').length,0);
 const logCalls=f.calls.filter(c=>c[0]==='logs').length;
 const resumed=await createEnvironment(f.options,f.context);
 assert.equal((await resumed.observe(result.handle)).executionSeconds,result.executionSeconds);
 assert.equal((await resumed.readAdmission(result.handle)).released,true);
 assert.equal(f.calls.filter(c=>c[0]==='logs').length,logCalls);
 assert.equal(f.calls.filter(c=>c[0]==='download').length,0);
 const artifact=f.events.find(e=>e.type==='execution-output').data;
 assert.equal(await readFile(artifact.stdout,'utf8'),'answer\n');assert.equal(await readFile(artifact.stderr,'utf8'),'diagnostic\n');
});

for(const logs of ['missing','wrong-key','malformed'])test('legacy file collection remains authoritative for '+logs+' native metadata',async t=>{
 const f=await fixture(t,{logs});let output='';const result=await f.environment.execute('printf legacy',f.project,{onData:d=>{output+=d;}});
 assert.equal(result.exitCode,0);assert.equal(result.timingSource,'wrapper');assert.equal(output,'legacy');
 assert.equal(f.calls.filter(c=>c[0]==='download').length,3);
});

test('oversized output keeps inline timing but collects complete streams through the existing archive path',async t=>{
 const f=await fixture(t);let output='';const result=await f.environment.execute("python3 -c 'print(\"x\"*70000)'",f.project,{onData:d=>{output+=d;}});
 assert.equal(result.exitCode,0);assert.equal(result.timingSource,'wrapper');assert.equal(output,'x'.repeat(70000)+'\n');
 const downloads=f.calls.filter(c=>c[0]==='download');assert.equal(downloads.length,2);assert.ok(downloads.every(c=>/\/(stdout|stderr)$/.test(c[2])));
});

for(const phase of ['before','after'])test('terminal metadata preserves frozen-verifier rejection '+phase+' execution',async t=>{
 const f=await fixture(t),verifier=join(f.root,'verify.py');await writeFile(verifier,'frozen');
 await f.environment.prepare({project:{source:f.project,bundle:{root:f.root},verifier:{files:[verifier]}},attempt:{id:'verified'}});
 const remote=join(f.workspace,'owned/verified/tree/verify.py');if(phase==='before')await writeFile(remote,'changed');
 const command=phase==='after'?`printf changed > '${remote}'; printf ran > marker`:'printf ran > marker';
 const result=await f.environment.execute(command,f.project,{purpose:'verification'});
 assert.equal(result.exitCode,125);assert.deepEqual(result.outcome,{kind:'infrastructure',reason:'frozen-verifier-'+phase});
 assert.equal(f.calls.filter(c=>c[0]==='download').length,0);
});

test('unknown native status does not use terminal logs or replay, and a later terminal observation can recover',async t=>{
 const f=await fixture(t);f.unknown(true);const result=await f.environment.execute('printf answer',f.project);
 assert.equal(result.settled,false);assert.equal(f.calls.some(c=>c[0]==='logs'),false);assert.equal(f.calls.some(c=>c[0]==='download'),false);
 await Promise.all([...f.jobs.values()].map(j=>j.done));f.unknown(false);
 const submissions=f.calls.filter(c=>c[0]==='run').length;
 assert.equal((await f.environment.observe(result.handle)).state,'stopped');assert.equal(f.calls.filter(c=>c[0]==='run').length,submissions);
});

test('a timed-out payload cannot manufacture a released lease from a completed supervisor receipt',async t=>{
 const f=await fixture(t);const result=await f.environment.execute(`python3 -c 'import json,os,time; json.dump({"token":"owned","decision":"admitted"},open(os.environ["RSI_LEASE_RECEIPT"],"w")); time.sleep(30)'`,f.project,{timeout:.1,leaseReceipt:true,admissionToken:'owned'});
 await Promise.all([...f.jobs.values()].map(j=>j.done));const state=await f.environment.observe(result.handle);
 assert.equal(state.state,'stopped');assert.deepEqual(state.outcome,{kind:'infrastructure',reason:'elapsed-timeout'});
 assert.notEqual((await f.environment.readAdmission(result.handle))?.released,true);
});

test('binary and escaped output retain exact bytes through the file fallback',async t=>{
 const f=await fixture(t);
 for(const bytes of [[0,255,128],Array(20000).fill(0)]){
  const chunks=[],body=bytes.length===3?'bytes([0,255,128])':'bytes(20000)';
  const before=f.calls.filter(c=>c[0]==='download').length;
  const result=await f.environment.execute(`python3 -c 'import sys; sys.stdout.buffer.write(${body})'`,f.project,{onData:d=>chunks.push(d)});
  assert.equal(result.exitCode,0);assert.deepEqual(Buffer.concat(chunks),Buffer.from(bytes));
  assert.equal(f.calls.filter(c=>c[0]==='download').length-before,2);
 }
});

test('inline output preserves CRLF bytes',async t=>{
 const f=await fixture(t),chunks=[];
 const result=await f.environment.execute("printf 'line\\r\\n'",f.project,{onData:d=>chunks.push(d)});
 assert.equal(result.exitCode,0);assert.equal(Buffer.concat(chunks).toString(),'line\r\n');
 assert.equal(f.calls.filter(c=>c[0]==='download').length,0);
});
