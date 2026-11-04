import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {syncBuiltinESMExports} from 'node:module';
import {ownerStopped,processIdentity,attemptStopped} from '../../controller/lifecycle.mjs';

test('exiting owner ESRCH proves absence; permission errors and unknown work remain blocked',async t=>{
  const owner=await processIdentity();
  assert.equal(await ownerStopped(owner),false);
  const read=fs.readFile;
  let code='ESRCH';
  const mocked=t.mock.method(fs,'readFile',async(path,...args)=>{
    if(path===`/proc/${owner.pid}/stat`)throw Object.assign(new Error('Native process lookup'),{code});
    return read(path,...args);
  });
  syncBuiltinESMExports();
  try {
    assert.equal(await ownerStopped(owner),true);
    const state={events:()=>[{type:'execution-handle',data:{handle:'known'}}]};
    assert.equal(await attemptStopped(state,{id:'attempt',owner,elapsed:0},
      {observe:async()=>({state:'unknown'})}),false);
    code='EACCES';await assert.rejects(ownerStopped(owner),{code:'EACCES'});
  }finally{mocked.mock.restore();syncBuiltinESMExports();}
  assert.equal(await ownerStopped(owner),false);
});
