import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {curatedPacks,prepareCuratedSuites,applicablePack} from '../../cli/skill-curated-suites.mjs';
import {resolveHypothesisCases,hypothesisCaseReadiness} from '../../cli/skill-priority-cases.mjs';
import {hypothesisTarget,hypothesisEvaluationSettled,qualifyHypothesisTasks} from '../../cli/skill-hypothesis-evaluation.mjs';
import {validateKernelReceipt} from '../../cli/skill-microeval.mjs';
import {State} from '../../controller/state.mjs';
import {SkillBacklog} from '../../controller/skill-backlog.mjs';
const runtime={setup:{packages:{torch:['fixture'],triton:['fixture'],'torch-npu':['fixture']},compilers:{hivmc:['revision-a']},cann:{}},cannVersions:{manifest:'a'},product:'fixture-a3'};
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'curated-suites-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const config={root,target:'a3',environment:{options:{mode:'local',target:'registered',runtime:'profile',health:{product:'a3'}}},skills:{priority:{}}};
 const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());const backlog=new SkillBacklog(state,{target:'a3'});
 return {root,config,state,backlog};
}
test('retained five-pack references and independent numerical corruptions exercise actual CPU comparators',()=>{
 const result=spawnSync(process.env.RSI_TEST_PYTHON??'python3',[new URL('./test_skill_curated_numeric.py',import.meta.url).pathname],{encoding:'utf8',env:{...process.env,OMP_NUM_THREADS:'1',MKL_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1'},timeout:60000});
 assert.equal(result.status,0,result.stderr);assert.equal(result.signal,null);
});

test('five curated families preserve independent controls and private boundary variants',async t=>{
 const f=await fixture(t),packs=await curatedPacks();assert.equal(packs.length,5);
 const suites=await prepareCuratedSuites({...f,directory:join(f.root,'packs'),runtimeIdentity:runtime});
 for(const suite of suites){
  assert.equal(suite.tasks.length,5);assert.equal(suite.binding.runtimeIdentity,runtime);
  for(const task of suite.tasks){
   assert.notEqual(task.controls.reference.files['kernel.py'],task.controls.negatives[0].files['kernel.py']);
   const checker=await readFile(task.checker,'utf8'),publicChecker=task.files['check.py'];assert.notEqual(checker,publicChecker);
   const compile=spawnSync(process.env.RSI_TEST_PYTHON??'python3',['-c','import ast,sys; ast.parse(sys.stdin.read())'],{input:checker,encoding:'utf8'});assert.equal(compile.status,0,compile.stderr);
   if(suite.objective==='optimization'){assert.notEqual(task.files['kernel.py'],task.controls.negatives[0].files['kernel.py']);assert(task.pairedChecker);}
  }
 }
 assert.equal(applicablePack(packs.find(p=>p.id==='first-call-launch'),{objective:'correctness',proposal:{changes:[{text:'After compilation, execute the first launch using current arguments'}]}}),true);
 assert.equal(applicablePack(packs[0],{objective:'correctness',proposal:{changes:[{text:'Report a runtime lease failure to the operator'}]}}),false);
 assert.deepEqual(await prepareCuratedSuites({config:{...f.config,target:'a5'},directory:f.root,runtimeIdentity:runtime}),[]);
});
test('fresh runtime probes reuse qualified controls across hypotheses and invalidate changed toolchains',async t=>{
 const f=await fixture(t);let probes=0,qualifications=0,identity=runtime;const selected=[];
 const item={objective:'correctness',proposal:{changes:[{text:'Restore the reduced row axis for broadcast with column tiles'}]}};
 const args={...f,item,seconds:120,resource:{id:'reserved'}};
 const deps={runtimeIdentity:async()=>{probes++;return structuredClone(identity);},prepareTasks:async()=>({manifest:{hypotheses:[]},byHypothesis:{}}),
  select:async({prompt})=>({value:{catalogueId:prompt.catalogues.find(c=>c.id==='curated-reduction-broadcast').id,reason:'Direct broadcast rank boundary'}}),
  qualify:async input=>{
   selected.push(input.directory);return qualifyHypothesisTasks(input,{verify:async({cwd,task})=>{
    qualifications++;const source=await readFile(join(cwd,'kernel.py'),'utf8'),correct=source===task.controls.reference.files['kernel.py'];
    return {correct,exitCode:correct?0:1,output:JSON.stringify({correct,kernelSha256:createHash('sha256').update(source).digest('hex'),evidence:{device:'npu:0',fixture:true}})};
   }});
  }};
 const first=await resolveHypothesisCases({...args,directory:join(f.root,'h1')},deps);assert.equal(first.status,'ready');
 const second=await resolveHypothesisCases({...args,directory:join(f.root,'h2')},deps);assert.equal(second.status,'ready');assert.equal(probes,2);assert.equal(qualifications,10);assert.deepEqual(second.binding,first.binding);
 identity={...runtime,cannVersions:{manifest:'changed'}};
 const changed=await resolveHypothesisCases({...args,directory:join(f.root,'h1')},deps);assert.equal(changed.status,'ready');assert.equal(probes,3);assert.equal(qualifications,20);assert.equal(selected[0],selected[1]);assert.notEqual(selected[0],selected[2]);
 const ready=await hypothesisCaseReadiness({config:f.config,item});assert.equal(ready.ready,true);assert.equal(ready.supported,true);
});
test('semantic rejection never qualifies a broadly related suite or constructs a project',async t=>{
 const f=await fixture(t),item={objective:'correctness',proposal:{changes:[{text:'Fix a reduction broadcast axis'}]}};
 const result=await resolveHypothesisCases({...f,item,directory:join(f.root,'h'),seconds:30},{runtimeIdentity:async()=>runtime,
  prepareTasks:async()=>({manifest:{hypotheses:[]},byHypothesis:{}}),select:async()=>({value:{catalogueId:null,reason:'Specific mechanism differs'}}),qualify:()=>{throw Error('Irrelevant control dispatch');}});
 assert.equal(result.status,'unsupported');assert.equal(result.missingCapability,true);assert.equal(f.state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='skill_learning_requests'").get()?f.state.db.prepare('SELECT COUNT(*) n FROM skill_learning_requests').get().n:0,0);
});
test('runtime probe dispatch uses admitted environment and retains unknown execution custody',async t=>{
 const f=await fixture(t),directory=join(f.root,'probe'),resource={id:'reserved'};let prepared;
 const binding={target:'a3',environment:f.config.environment.options};
 const env=async(_options,context)=>{assert.equal(context.resource,resource);return {prepare:async p=>{prepared=p;},execute:async()=>({settled:false,handle:'remote:fixture:job:probe'})};};
 await assert.rejects(hypothesisTarget({root:directory,binding,resource},{createEnvironment:env}).runtimeIdentity({directory,seconds:30}),{code:'RSI_EXECUTION_UNRESOLVED'});
 assert.equal(prepared.project.bundle.root,directory);const report=JSON.parse(await readFile(join(directory,'micro/results.json'),'utf8'));assert.equal(report.runningSample.unresolvedExecution.handle,'remote:fixture:job:probe');
 assert.equal(await hypothesisEvaluationSettled({directory}),false);
});


test('new CPU diagnostic proposals reuse premade report cases through semantic selection',async t=>{
 const f=await fixture(t),item={objective:'correctness',proposal:{changes:[{id:'new-origin-rule',text:'Distinguish mutable candidate traceback from protected runtime failure before source repair'}]}};
 const result=await resolveHypothesisCases({...f,item,directory:join(f.root,'cpu-hypothesis'),seconds:30},{
  select:async({prompt})=>{assert(prompt.catalogues.some(c=>c.id==='cpu-fault-ownership'));return {value:{catalogueId:'cpu-fault-ownership',reason:'Directly tests report attribution and ownership'}};},
  runtimeIdentity:()=>{throw Error('CPU question dispatched a device command');}});
 assert.equal(result.status,'ready');assert.equal(result.target,'cpu');assert.equal(result.tasks.length,5);
 assert.equal((await hypothesisCaseReadiness({config:f.config,item})).ready,true);
});

test('curated runtime device selection reaches the public receipt validator only after initialization',async t=>{
 const f=await fixture(t),kernel=join(f.root,'kernel.py');await writeFile(kernel,'fixture_source=True\n');
 const script=`import importlib.util,json,sys,types
s=importlib.util.spec_from_file_location('core',sys.argv[2]);core=importlib.util.module_from_spec(s);s.loader.exec_module(core)
calls=[];fail=sys.argv[3]=='fail'
class Npu:
 def set_device(self,device):
  calls.append(['set',device])
  if fail: raise RuntimeError('initialization fixture failed')
 def current_device(self):
  calls.append(['current']);return 0
sys.modules['torch']=types.SimpleNamespace(__version__='fixture',npu=Npu())
sys.modules['torch_npu']=types.SimpleNamespace(__version__='fixture')
sys.modules['triton']=types.SimpleNamespace(__version__='fixture')
expected={'fixture':True};core.load=lambda *args:types.SimpleNamespace(identity=lambda:expected)
core.health=lambda:{'physicalDevice':4,'logicalDevice':0,'queries':{'health':'unit-fixture'}}
core.check_module=lambda *args:{'latencySeconds':.1,'rawSeconds':[.1],'checkedCases':['unit-fixture']}
try:core.main(sys.argv[1],{'runtimeIdentity':expected,'family':'fixture','cases':[]})
finally:assert calls==([['set',0]] if fail else [['set',0],['current']]),calls
`;
 const core=new URL('../../cli/skill-suites/kernel-check-core.py',import.meta.url).pathname;
 for(const mode of ['pass','fail']){
  const result=spawnSync(process.env.RSI_TEST_PYTHON??'python3',['-c',script,f.root,core,mode],{encoding:'utf8'});
  assert.equal(result.status,mode==='pass'?0:125,result.stderr);
  const checked=await validateKernelReceipt({exitCode:result.status,output:result.stdout},f.root,{requireDevice:true});
  if(mode==='pass'){assert.equal(checked.receipt.evidence.device,'npu:0');assert.equal(checked.receipt.evidence.physicalDevice,4);assert.equal(checked.receipt.evidence.queries.health,'unit-fixture');}
  else{assert.equal(checked.correct,null);assert.equal(checked.receipt.evidence.device,undefined);}
 }
});
