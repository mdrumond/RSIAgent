import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createEnvironment} from '../../integrations/environments/index.mjs';

for(const backend of ['ssh','remote-agent'])test(`${backend} ambiguous transfer retains its original response without asserting closure`,async t=>{
  const root=await mkdtemp(join(tmpdir(),'submission-cause-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const source=join(root,'source');await mkdir(source);await writeFile(join(source,'a.py'),'a=1\n');
  const failure={target:'fake',backend,operation:'download',state:'failed',message:'synchronous transfer preflight failed'};
  const events=[];
  const env=await createEnvironment({mode:'cpl',target:'fake',workspacePath:root,remoteDirectory:'owned'},
    {instancePath:root,sdk:{createLocalBashOperations:()=>({})},exclude:async(_label,fn)=>fn(),
      event:async(type,data)=>events.push({type,data}),client:async args=>{
        if(args[0]==='capabilities')return {state:'available',backend};
        if(args[0]==='preflight')return {state:'completed'};
        if(args[0]==='download')return failure;
        assert.fail('Ambiguous download must not dispatch '+args[0]);
      }});
  await env.prepare({project:{source},attempt:{id:'a'}});
  await assert.rejects(env.collectArtifact('owned/a/project/result',join(root,'result')),error=>{
    assert.equal(error.closure,undefined);
    assert.equal(error.cause,failure);
    return /identity unknown/.test(error.message);
  });
  assert.equal(events.some(e=>e.type==='execution-unresolved'),true);
  assert.equal(events.some(e=>e.type==='execution-not-admitted'||e.type==='execution-submitted'),false);
});
