import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,readlink,symlink,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createEnvironment} from '../../integrations/environments/index.mjs';
import {importProject} from '../../cli/instance.mjs';
import {caseClient} from './resilience/transport.case.mjs';
import {target} from './resilience/fixtures.mjs';
import {exerciseSymlinks} from './resilience/source-symlinks.case.mjs';

test('native SDK two-tree staging preserves existing self alias, external file link, repeat edit and removal',{timeout:30000},async t=>{
  const root=await mkdtemp(join(tmpdir(),'source-links-'));let backend;
  t.after(async()=>{try{await backend?.close();}finally{await rm(root,{recursive:true,force:true});}});
  backend=await target({root});
  const events=[],environment=await createEnvironment({mode:'cpl',target:'fixture',runtime:'cpu',workspacePath:backend.workspace,remoteDirectory:'owned',pollMs:1},
    {instancePath:root,client:backend.client,event:async(type,data)=>events.push({type,data}),exclude:async(_,fn)=>fn()});
  const evidence=await exerciseSymlinks(environment,join(root,'source'),new AbortController().signal);
  assert.equal(evidence.removedLinks,true);
  assert.equal(events.filter(e=>e.type==='source-staged').length,4);
  assert.ok([...backend.jobs.values()].every(j=>j.state==='completed'));
});
test('project import preserves physical relative link text and external original file',async t=>{
  const root=await mkdtemp(join(tmpdir(),'import-links-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const source=join(root,'input/source');await mkdir(source,{recursive:true});
  const verifier=join(root,'input/verify.py');await writeFile(verifier,'assert True\n');
  const external=join(root,'input/external.txt');await writeFile(external,'unchanged');
  await symlink('.',join(source,'source'));await symlink('../external.txt',join(source,'external'));
  const spec=await importProject(join(root,'instance'),{id:'p',goal:'layout',source,actor_seconds:30,verifier:{files:[verifier],command:'python ../verify.py'}});
  assert.equal(await readlink(join(spec.source,'source')),'.');
  assert.equal(await readlink(join(spec.source,'external')),'../external.txt');
  assert.equal(await readFile(external,'utf8'),'unchanged');
});
test('shared native closure accepts fractional remaining elapsed milliseconds',async t=>{
  const root=await mkdtemp(join(tmpdir(),'fractional-closure-'));t.after(()=>rm(root,{recursive:true,force:true}));
  let observations=0;
  const scoped=caseClient({signal:new AbortController().signal,readInstructions:async()=>{},native:async(args,{signal})=>{
    assert.equal(signal.aborted,false);assert.equal(args[0],'observe');observations++;return {state:'completed'};
  }});
  scoped.handles.add('known');
  await scoped.close({observe:async handle=>{await scoped.client(['observe',handle]);return {state:'stopped'};}},root,{milliseconds:326514.193408});
  assert.equal(observations,1);
  assert.deepEqual(JSON.parse(await readFile(join(root,'closure.json'),'utf8')),[{handle:'known',state:'stopped'}]);
});
test('retained native backend archives actual failed job stderr after close',async t=>{
  const root=await mkdtemp(join(tmpdir(),'retained-error-'));let backend;
  t.after(async()=>{try{await backend?.close();}finally{await rm(root,{recursive:true,force:true});}});
  backend=await target({root});const script=join(root,'fail.sh');
  await writeFile(script,"printf 'native copy error\\n' >&2; exit 17\n");
  const result=await backend.client(['run','fixture','--file',script,'--receipt-file',join(root,'receipt.json')]);
  await backend.close();const rows=JSON.parse(await readFile(join(root,'backend.json'),'utf8'));
  const job=rows.find(row=>row.handle===result.handle);
  assert.equal(job.exit,17);assert.equal(job.stdout,'native copy error\n');
});
