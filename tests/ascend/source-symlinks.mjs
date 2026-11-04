import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,readlink,symlink,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createEnvironment} from '../../integrations/environments/index.mjs';
import {importProject} from '../../cli/instance.mjs';
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
