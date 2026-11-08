import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,access} from 'node:fs/promises';
import {join,relative} from 'node:path';
import {tmpdir} from 'node:os';
import {importProject} from '../../cli/instance.mjs';
import {compileVerifierEntries} from '../../controller/project-commands.mjs';
import {createEnvironment} from '../../integrations/environments/index.mjs';
import {target} from './resilience/fixtures.mjs';

async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'command-layout-')),original=join(root,'original');
  t.after(()=>rm(root,{recursive:true,force:true}));
  const source=join(original,'envelope/work/items/source'),reference=join(original,'envelope/trusted/reference');
  const wrong=join(original,'envelope/trusted/wrong'),verifier=join(original,'envelope/trusted/checks/verify.py');
  for(const path of [source,reference,wrong,join(original,'envelope/trusted/checks')])await mkdir(path,{recursive:true});
  for(const [path,value] of [[source,42],[reference,42],[wrong,0]])await writeFile(join(path,'answer.py'),`answer=${value}\n`);
  await writeFile(verifier,`import importlib.util, json, pathlib, sys, time
p=pathlib.Path.cwd()/'answer.py'
s=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
start=time.perf_counter(); correct=m.answer==42; elapsed=time.perf_counter()-start
print(json.dumps({'mode':sys.argv[1],'answer':m.answer,'cwd':str(pathlib.Path.cwd()),'verifier':str(pathlib.Path(__file__).resolve()),'args':sys.argv[2:],'seconds':elapsed}))
sys.exit(0 if correct else 1)
`);
  const files=[verifier,join(reference,'answer.py'),join(wrong,'answer.py')];
  const entry=mode=>({entrypoint:relative(original,verifier),args:[mode,'literal ; touch SHOULD_NOT_EXIST']});
  const spec={id:'fixed',goal:'Return42',actor_seconds:60,source,
    bundle:{version:1,root:original,reference,negative_controls:[wrong],seconds:60,
      construction_command:`python3 '${verifier}' construction`},
    verifier:{files,command:entry('correctness'),baseline_command:entry('baseline'),performance_command:entry('performance')},
    performance_goal:'Report finite timing'};
  return {root,original,spec,verifier};
}

test('entrypoint commands import and execute through production staging after the original bundle is deleted',async t=>{
  const f=await fixture(t),spec=await importProject(join(f.root,'instance'),f.spec);
  await rm(f.original,{recursive:true});
  const backendRoot=await mkdtemp(join(tmpdir(),'command-target-')),backend=await target({root:backendRoot});
  t.after(async()=>{try{await backend.close();}finally{await rm(backendRoot,{recursive:true,force:true});}});
  const environment=await createEnvironment({mode:'cpl',target:'fixture',runtime:'cpu',workspacePath:backend.workspace,
    remoteDirectory:'owned',pollMs:1},{instancePath:join(f.root,'execution'),client:backend.client,
    exclude:async(_label,fn)=>fn(),event:()=>{}});
  await environment.prepare({project:spec,attempt:{id:'roundtrip'}});
  for(const [field,mode] of [['baseline_command','baseline'],['command','correctness'],['performance_command','performance']]) {
    let output='';const result=await environment.execute(spec.verifier[field],spec.source,{timeout:30,onData:chunk=>{output+=chunk;}});
    assert.equal(result.settled,true);assert.equal(result.exitCode,0,output);
    const report=JSON.parse(output.trim());assert.equal(report.mode,mode);assert.equal(report.answer,42);
    assert.deepEqual(report.args,['literal ; touch SHOULD_NOT_EXIST']);assert.ok(report.seconds>0&&Number.isFinite(report.seconds));
    assert.equal(report.cwd,join(backend.workspace,'owned/roundtrip/tree/envelope/work/items/source'));
    assert.equal(report.verifier,join(backend.workspace,'owned/roundtrip/tree/envelope/trusted/checks/verify.py'));
  }
  await assert.rejects(access(join(spec.source,'SHOULD_NOT_EXIST')),{code:'ENOENT'});
  await assert.rejects(access(join(backend.workspace,'owned/roundtrip/tree/envelope/work/items/source/SHOULD_NOT_EXIST')),{code:'ENOENT'});
  await writeFile(join(spec.source,'answer.py'),'answer=0\n');
  const rejected=await environment.execute(spec.verifier.command,spec.source,{timeout:30});
  assert.equal(rejected.settled,true);assert.equal(rejected.exitCode,1);
  assert.equal([...backend.jobs.values()].every(job=>job.state==='completed'),true);
});

test('a missing or undeclared target entrypoint is rejected before import',async t=>{
  const f=await fixture(t);
  for(const entrypoint of ['../wrong/verify.py','envelope/work/items/source/answer.py']) {
    await assert.rejects(importProject(join(f.root,'invalid'),{...f.spec,verifier:{...f.spec.verifier,
      baseline_command:{entrypoint,args:[]}}}),/declared frozen file/);
    await assert.rejects(access(join(f.root,'invalid/projects/fixed')),{code:'ENOENT'});
  }
  assert.throws(()=>compileVerifierEntries({files:[f.verifier],command:'cd missing/source && python3 ../verify.py'},
    {root:f.original,source:f.spec.source,required:true}),/entrypoint and literal args/);
  for(const arg of ['/tmp/candidate.py','--candidate=/tmp/candidate.py',`--config=${f.original}/config.json`])
    assert.throws(()=>compileVerifierEntries({files:[f.verifier],command:{entrypoint:f.verifier,args:[arg]}},
      {root:f.original,source:f.spec.source,required:true}),/source-relative paths/);
});

test('supplied legacy string commands still import and execute',async t=>{
  const f=await fixture(t),command=`python3 '${f.verifier}' correctness`;
  const spec=await importProject(join(f.root,'legacy'),{...f.spec,performance_goal:undefined,verifier:{files:f.spec.verifier.files,command}});
  await rm(f.original,{recursive:true});
  const environment=await createEnvironment({mode:'local'},{instancePath:join(f.root,'local'),event:()=>{},exclude:async(_label,fn)=>fn()});
  const result=await environment.execute(spec.verifier.command,spec.source,{timeout:10});
  assert.equal(result.settled,true);assert.equal(result.exitCode,0);
  assert.equal((await readFile(join(spec.source,'answer.py'),'utf8')).trim(),'answer=42');
});
