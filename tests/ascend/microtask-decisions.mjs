import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {tasks} from '../../cli/microtasks/tasks.mjs';
import {recoveryRepair} from './microtask-fixture.mjs';
const python=process.env.RSI_TEST_PYTHON??'python3';

test('recovery checker accepts bounded actor decisions and rejects observed failure modes',async t=>{
 const root=await mkdtemp(join(tmpdir(),'microtask-decisions-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const task=tasks.find(task=>task.id==='recovery-decisions');
 const scenarios=[
  {name:'repair',source:recoveryRepair,passes:true},
  {name:'starter',source:task.files['kernel.py']},
  {name:'cumulative-clock',source:recoveryRepair.replace('return ends[0]-starts[0]',"return events[-1]['actor_elapsed']")},
  {name:'actor-closes-lease',source:recoveryRepair.replace("actions.report('harness-lease')","actions.close_lease()")},
  {name:'ignores-runtime',source:recoveryRepair.replace("actions.report('runtime-identity')","actions.repair()")},
  {name:'optimises-before-verification',source:recoveryRepair.replace('actions.verify()','actions.optimise()')},
  {name:'changes-owned-record',source:recoveryRepair+"\nopen('lease.json','w').write('{}')\n"},
 ];
 for(const scenario of scenarios){
  const directory=join(root,scenario.name);await mkdir(directory);
  for(const [name,text]of Object.entries(task.files))await writeFile(join(directory,name),text);
  await writeFile(join(directory,'kernel.py'),scenario.source);
  const result=spawnSync(python,['-B',task.checker,directory],{cwd:directory,encoding:'utf8'});
  if(scenario.passes)assert.equal(result.status,0,result.stderr);
  else assert.notEqual(result.status,0,scenario.name+' must not pass');
 }
});
