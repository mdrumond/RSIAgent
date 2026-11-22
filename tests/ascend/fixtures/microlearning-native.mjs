/** CPU-only provider plans; every tool, verifier and A/B result is production. */
export const correct='def answer(): return 6 * 7\n';
export const incorrect='def answer(): return 0\n';
const call=(name,args)=>({name,args});
export function generationPlan({badControl=false,malformed=false}={}){
 const checker="import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(sys.argv[1]).resolve()))\nfrom kernel import answer\nprint({'actual':answer(),'expected':sum(6 for _ in range(7))})\nsys.exit(0 if answer()==sum(6 for _ in range(7)) else 1)\n";
 const files={'kernel.py':'evaluation/reference.py'};
 const manifest=malformed?{schema:1,project:'captured-shape',cases:[]}:{target:'cpu',objective:'correctness',tasks:Array.from({length:5},(_,i)=>({id:'case-'+i,kind:'hypothesis-case',bucket:'correctness',scopes:['implementation'],prompt:'Implement answer() returning six times seven. Case '+i,checker:'checker.py',files,controls:{reference:{files},negatives:[{files:{'kernel.py':badControl?'evaluation/reference.py':'evaluation/negative.py'}}]}}))};
 return [
  call('write',{path:'project/source/answer.py',content:correct}),
  call('write',{path:'project/reference/answer.py',content:correct}),
  call('write',{path:'project/negative/answer.py',content:incorrect}),
  call('write',{path:'project/verify.py',content:"import sys\nsys.path.insert(0,'.')\nfrom answer import answer\nsys.exit(0 if answer()==sum(6 for _ in range(7)) else 1)\n"}),
  call('write',{path:'project/checker.py',content:checker}),
  call('write',{path:'project/evaluation/reference.py',content:correct}),
  call('write',{path:'project/evaluation/negative.py',content:incorrect}),
  call('write',{path:'project/skill-evaluation.json',content:JSON.stringify(manifest)}),
  call('submit_project',{goal:'unrelated generated arithmetic',source:'project/source',reference:'project/reference',negative_controls:['project/negative'],verifier:{files:['project/verify.py','project/checker.py','project/evaluation/reference.py','project/evaluation/negative.py','project/skill-evaluation.json'],command:{entrypoint:'project/verify.py',args:[]}},construction_command:{entrypoint:'project/verify.py',args:[]}})
 ];
}
export function samplePlan(body,{regression=false}={}){
 const content=body.messages.find(message=>message.role==='user').content;
 const prompt=typeof content==='string'?content:content.map(block=>block.text??'').join('');
 const candidate=prompt.includes('USE_INDEPENDENT_ORACLE');
 return [call('write',{path:'kernel.py',content:candidate!==regression?correct:incorrect})];
}
