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
  {name:'inferred-runtime-aliases',source:recoveryRepair.replace("case['expected']!=case['observed']", "case.get('expected_runtime')!=case.get('observed_runtime')")},
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

test('the public starter case runs through the same decision and event API as the checker',async t=>{
 const root=await mkdtemp(join(tmpdir(),'microtask-public-api-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const task=tasks.find(task=>task.id==='recovery-decisions');
 await writeFile(join(root,'kernel.py'),task.files['kernel.py']+'\n'+recoveryRepair);
 const result=spawnSync(python,['-B','-c',`
import copy, runpy
module=runpy.run_path('kernel.py')
case=copy.deepcopy(module['EXAMPLE_CASE'])
before=copy.deepcopy(case)
class Actions:
 def __init__(self): self.calls=[]
 def verify(self): self.calls.append(('verify',))
 def report(self,reason): self.calls.append(('report',reason))
actions=Actions()
assert module['next_action'](case,actions)=={'action':'validate','tool_seconds':2.0}
assert actions.calls==[('verify',)]
assert case==before
case['observed']['torch']='another-demo'
actions=Actions()
assert module['next_action'](case,actions)=={'action':'blocked','tool_seconds':2.0}
assert actions.calls==[('report','runtime-identity')]
assert module['EXAMPLE_CASE']==before
`],{cwd:root,encoding:'utf8'});
 assert.equal(result.status,0,result.stderr);
});
