import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {importProject} from '../../cli/instance.mjs';
import {compileGenerationProposal} from '../../cli/project-generation.mjs';
import {loadSkillEvaluation} from '../../cli/skill-evaluation-contract.mjs';
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'suite-contract-'));t.after(()=>rm(root,{recursive:true,force:true}));
 for(const dir of ['source','reference','negative','verifier','instance'])await mkdir(join(root,dir));
 for(const dir of ['source','reference','negative'])await writeFile(join(root,dir,'kernel.py'),'answer=42\n');
 const trusted=join(root,'verifier'),files=['skill-evaluation.json','checker.py','main.py','good.py','bad.py'];
 await writeFile(join(trusted,'checker.py'),"import runpy,sys\nfrom pathlib import Path\nraise SystemExit(0 if runpy.run_path(str(Path(sys.argv[1])/'kernel.py')).get('answer')==42 else 1)\n");
 await writeFile(join(trusted,'main.py'),'raise SystemExit(0)\n');await writeFile(join(trusted,'good.py'),'answer=42\n');await writeFile(join(trusted,'bad.py'),'answer=0\n');
 const contract={target:'cpu',objective:'correctness',tasks:Array.from({length:5},(_,i)=>({id:'case-'+i,kind:'hypothesis-case',target:'cpu',bucket:'correctness',prompt:'Check answer',checker:'checker.py',files:{'kernel.py':'bad.py'},controls:{reference:{files:{'kernel.py':'good.py'}},negatives:[{files:{'kernel.py':'bad.py'}}]}}))};
 const save=()=>writeFile(join(trusted,'skill-evaluation.json'),JSON.stringify(contract));await save();
 const proposal={goal:'Suite question',source:'source',reference:'reference',negative_controls:['negative'],verifier:{files:files.map(f=>'verifier/'+f),command:{entrypoint:'verifier/main.py',args:[]}}};
 const spec=()=>({id:'suite',goal:proposal.goal,source:join(root,'source'),actor_seconds:30,verifier:{files:proposal.verifier.files.map(p=>join(root,p)),command:{entrypoint:join(trusted,'main.py'),args:[]}},generation:{learningRequest:{workKind:'suite'}}});
 return {root,proposal,spec,contract,save};
}
test('public curriculum admission and import share manifest-relative protected suite interpretation',async t=>{
 const f=await fixture(t);await compileGenerationProposal(f.root,f.proposal,{skillEvaluation:true});
 const imported=await importProject(join(f.root,'instance'),f.spec());const loaded=await loadSkillEvaluation(imported,{required:true});
 assert.equal(loaded.tasks.length,5);assert.equal(loaded.tasks[0].files['kernel.py'],'answer=0\n');assert(loaded.tasks[0].checker.startsWith(join(f.root,'instance','projects')));
});
for(const defect of ['project-path','undeclared','binary','four-cases','no-negatives'])test('public admission and import reject '+defect+' before actor work',async t=>{
 const f=await fixture(t);
 if(defect==='project-path')f.contract.tasks[0].checker='verifier/checker.py';
 if(defect==='undeclared')f.proposal.verifier.files=f.proposal.verifier.files.filter(p=>!p.endsWith('checker.py'));
 if(defect==='binary'){await writeFile(join(f.root,'verifier','data.pt'),Buffer.from([0x80,0xff,0]));f.proposal.verifier.files.push('verifier/data.pt');f.contract.tasks[0].files['data.pt']='data.pt';}
 if(defect==='four-cases')f.contract.tasks.pop();if(defect==='no-negatives')f.contract.tasks[0].controls.negatives=[];await f.save();
 const expected=defect==='project-path'?/manifest-relative, not project-relative/:defect==='undeclared'?/declared in verifier.files/:defect==='binary'?/UTF-8 text, not binary/:defect==='four-cases'?/exactly five/:/negative control/;
 await assert.rejects(compileGenerationProposal(f.root,f.proposal,{skillEvaluation:true}),expected);await assert.rejects(importProject(join(f.root,'instance'),f.spec()),expected);
});
test('suite requests require export but ordinary projects without a sidecar remain compatible',async t=>{
 const f=await fixture(t);f.proposal.verifier.files=['verifier/main.py'];
 await assert.rejects(compileGenerationProposal(f.root,f.proposal,{skillEvaluation:true}),/requires a protected/);
 await assert.rejects(importProject(join(f.root,'instance'),f.spec()),/requires a protected/);
 await compileGenerationProposal(f.root,f.proposal);const ordinary=f.spec();delete ordinary.generation;assert(await importProject(join(f.root,'instance'),ordinary));
});
