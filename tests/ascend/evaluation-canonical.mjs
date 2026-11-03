import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,cp,readFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {setup,rows} from './driver_setup.mjs';
import {readConfig} from '../../cli/instance.mjs';
import {bootstrap,wave} from '../../cli/evaluation.mjs';
import {ownerStopped} from '../../controller/lifecycle.mjs';

test('isolated evaluation runs its requested canonical project outside the production curriculum',{timeout:120000},async t=>{
  const f=await setup(t,'recover',['unrelated']);
  const config=await readConfig(f.root),round=join(f.root,'round');await mkdir(round);
  const credentials=join(round,'fixture.env');await writeFile(credentials,'DRIVER_FIXTURE_KEY=fixture-only\n');
  const requested={...config,curriculum:{phase1:[]},actor:{...config.actor,credentials_file:credentials}};
  const instanceRoot=join(round,'sample');
  await bootstrap({sourceInstanceRoot:f.root,instanceRoot,projectId:'unrelated',projectIds:['unrelated'],skillVersion:'baseline',config:requested});
  await cp(resolve('tests/ascend/driver_fixture.py'),join(instanceRoot,'machinery','driver_fixture.py'));
  t.after(async()=>{
    try{
      const receipt=JSON.parse(await readFile(join(instanceRoot,'knowledge-service.json'),'utf8'));
      if(!await ownerStopped(receipt.owner))process.kill(receipt.owner.pid,'SIGTERM');
    }catch(error){if(error.code!=='ENOENT'&&error.code!=='ESRCH')throw error;}
  });
  const result=await wave({instanceRoot,config:requested,projectIds:['unrelated']});
  assert.equal(result.correct,true);assert.equal(result.attempts.length,1);
  assert.equal(rows(instanceRoot,"SELECT count(*) AS n FROM events WHERE type='stage-completed'")[0].n,8);
  assert.ok(f.server.requests.some(r=>r.messages.some(m=>m.role==='tool')));
  f.pass();
});
