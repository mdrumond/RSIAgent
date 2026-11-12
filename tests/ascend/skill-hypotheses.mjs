import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {spawnSync} from 'node:child_process';
import {prepareHypothesisTasks,hypotheses,hypothesisForProposal,recoveryDecisionsV2} from '../../cli/microtasks/hypotheses.mjs';
import {tasks as legacy} from '../../cli/microtasks/tasks.mjs';
import {recoveryRepair} from './microtask-fixture.mjs';
const python=process.env.RSI_TEST_PYTHON??'python3';
const answers={
 'fault-ownership':`def route(r):
 if r['runtime']['expected']!=r['runtime']['observed'] or r['lease']!='clear':return {'classification':'operator-blocked','action':'report-operator'}
 if r['trace']:
  path=r['trace'][-1]['path']
  if path.startswith(r['candidatePrefix']):return {'classification':'candidate-defect','action':'repair-source'}
  if path.startswith(r['protectedPrefix']):return {'classification':'operator-blocked','action':'report-operator'}
 return {'classification':'unproven','action':'collect-evidence'}
`,
 'dtype-triage':`def triage(r):
 if r['required'] not in r['supported']:return {'classification':'specification-defect','action':'report-operator'}
 v=r['verification']
 if v and v['sourceHash']==r['sourceHash']:
  return {'classification':'verified','action':'accept'} if v['passed'] else {'classification':'candidate-defect','action':'repair-source'}
 if r['error'] and r['requested'] not in r['supported']:return {'classification':'candidate-defect','action':'repair-source'}
 return {'classification':'unproven','action':'run-verifier'}
`,
 'construction-admissibility':`import copy,posixpath
def repair_proposal(p):
 p=copy.deepcopy(p)
 for field in ('source','reference'):
  if p[field].endswith('.py'):p[field]=posixpath.dirname(p[field])
 trees=[p['source'],p['reference']]+p['negative_controls']
 p['verifier']['files']=[f for f in p['verifier']['files'] if not any(f==t or f.startswith(t+'/') for t in trees)]
 p['baseline_construction_command']=p.get('baseline_construction_command',p['verifier'].get('baseline_construction_command'))
 p['baseline_construction_command']['entrypoint']=p['verifier']['baseline_command']['entrypoint']
 p['baseline_construction_command']['args']=['--mode','cpu']
 return p
`
};
async function fixture(t){const root=await mkdtemp(join(tmpdir(),'skill-hypotheses-'));t.after(()=>rm(root,{recursive:true,force:true}));return {root,...await prepareHypothesisTasks(join(root,'checkers'))};}
async function work(root,task,suffix,code){const path=join(root,task.id+'-'+suffix);await mkdir(path);for(const [name,text]of Object.entries(task.files)){await mkdir(dirname(join(path,name)),{recursive:true});await writeFile(join(path,name),text);}await writeFile(join(path,'kernel.py'),code);return path;}
const check=(task,cwd)=>spawnSync(python,['-B',task.checker,cwd],{cwd,encoding:'utf8',timeout:20000});

test('catalogue freezes five distinct cases per hypothesis and deduplicates identical fault guidance provenance',async t=>{
 const f=await fixture(t);assert.equal(f.tasks.length,15);assert.equal(new Set(f.tasks.map(t=>t.id)).size,15);
 for(const h of hypotheses){assert.equal(f.byHypothesis[h.id].length,5);assert.equal(new Set(f.byHypothesis[h.id].map(t=>t.checker)).size,5);}
 assert.equal(hypothesisForProposal([{id:'preflight-fault-ownership'}]).id,hypothesisForProposal([{id:'preflight-and-budget-triage'}]).id);
 assert.equal(f.manifest.cases.every(c=>c.heldoutVariants===2),true);assert.equal(new Set(f.manifest.cases.map(c=>c.checkerSha256)).size,15);
 assert.deepEqual(JSON.parse(await readFile(join(f.root,'checkers/catalogue.json'),'utf8')),f.manifest);
});

test('all fifteen independent checkers accept valid implementations and reject broken behavior', {timeout:180000},async t=>{
 const f=await fixture(t);
 for(const task of f.tasks){
  const hypothesis=f.manifest.cases.find(c=>c.id===task.id).hypothesis;
  const good=await work(f.root,task,'good',answers[hypothesis]),accepted=check(task,good);assert.equal(accepted.status,0,task.id+'\n'+accepted.stderr);
  const bad=await work(f.root,task,'bad',hypothesis==='construction-admissibility'?'def repair_proposal(p): return p\n':`def ${hypothesis==='fault-ownership'?'route':'triage'}(r): return {'classification':'verified','action':'accept'}\n`);
  assert.notEqual(check(task,bad).status,0,task.id+' must reject the negative control');
 }
});

test('held-out path and proof variations and protected-state checks prevent public-case-only repairs',async t=>{
 const f=await fixture(t);
 const cases=[
  ['fault-ownership-candidate-traceback',answers['fault-ownership'].replace("path.startswith(r['candidatePrefix'])","path.startswith('variant-0/candidate/')")],
  ['dtype-triage-cpu-reference',answers['dtype-triage'].replace("v and v['sourceHash']==r['sourceHash']",'v')],
  ['fault-ownership-lease-ownership',"open('harness-state.json','w').write('{}')\n"+answers['fault-ownership']]
 ];
 for(const [id,code]of cases){const task=f.tasks.find(t=>t.id===id),cwd=await work(f.root,task,'mutant',code);assert.notEqual(check(task,cwd).status,0,id);}
});

test('new recovery event-order contract accepts timestamp pairs independent of list order; old task remains usable',async t=>{
 const root=await mkdtemp(join(tmpdir(),'recovery-order-v2-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const old=legacy.find(t=>t.id==='recovery-decisions');assert.notEqual(recoveryDecisionsV2.id,old.id);
 for(const [name,text]of Object.entries(recoveryDecisionsV2.files))await writeFile(join(root,name),text);
 await writeFile(join(root,'kernel.py'),recoveryDecisionsV2.files['kernel.py']+'\n'+recoveryRepair);
 const good=spawnSync(python,['-B',recoveryDecisionsV2.checker,root],{cwd:root,encoding:'utf8'});assert.equal(good.status,0,good.stderr);
 const backwardsList=recoveryRepair.replace("return ends[0]-starts[0]","return None if events[0]['phase']=='end' else ends[0]-starts[0]");
 await writeFile(join(root,'kernel.py'),recoveryDecisionsV2.files['kernel.py']+'\n'+backwardsList);assert.notEqual(spawnSync(python,['-B',old.checker,root],{cwd:root}).status,0);
});
