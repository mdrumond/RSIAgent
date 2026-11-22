/** Shared admission and evaluation loader. Manifest paths are relative to its directory. */
import {readFile,realpath} from 'node:fs/promises';
import {dirname,resolve,relative,isAbsolute} from 'node:path';
// A complete, manifest-relative template for curriculum authors. Files remain
// protected dependencies, not an invitation to weaken admission or controls.
export const skillEvaluationExample={target:'cpu',objective:'correctness',tasks:Array.from({length:5},(_,i)=>({
 id:'case-'+i,kind:'hypothesis-case',bucket:'correctness',scopes:['implementation'],prompt:'Implement the isolated correctness behavior',
 checker:'evaluation/check.py',files:{'kernel.py':'evaluation/starter.py'},
 controls:{reference:{files:{'kernel.py':'evaluation/reference.py'}},negatives:[{files:{'kernel.py':'evaluation/negative.py'}}]}
}))};
const within=(root,path)=>{const rel=relative(root,path);return rel===''||!isAbsolute(rel)&&rel!=='..'&&!rel.startsWith('../');};
export class SkillEvaluationContractError extends Error {
 constructor(message,options){super(message,options);this.name='SkillEvaluationContractError';this.code='INVALID_SKILL_EVALUATION';}
}
export async function loadSkillEvaluation(spec,options={}){
 try{return await readSkillEvaluation(spec,options);}catch(error){
  // Only a malformed protected artifact is terminal. I/O and interrupted work
  // must retain their original retry/settlement semantics.
  if(error instanceof SkillEvaluationContractError)throw error;
  if(error instanceof SyntaxError||['ENOENT','ENOTDIR'].includes(error.code))throw new SkillEvaluationContractError(error.message,{cause:error});
  throw error;
 }
}
async function readSkillEvaluation(spec,{required=false}={}){
 const declarations=spec.verifier?.files??[],manifests=declarations.filter(file=>file.endsWith('/skill-evaluation.json'));
 if(!manifests.length){if(required)throw new SkillEvaluationContractError('Suite learning requires a protected verifier/skill-evaluation.json declared in verifier.files before actor admission');return null;}
 if(manifests.length!==1)throw new SkillEvaluationContractError('Declare exactly one skill-evaluation.json');
 const manifest=await realpath(manifests[0]),root=dirname(manifest),source=await realpath(spec.source),declared=new Set(await Promise.all(declarations.map(file=>realpath(file))));
 const decode=async path=>{const bytes=await readFile(path);try{const text=new TextDecoder('utf-8',{fatal:true}).decode(bytes);if(text.includes('\0'))throw new SkillEvaluationContractError('Binary NUL');return text;}catch{throw new SkillEvaluationContractError('Evaluation files must be UTF-8 text, not binary fixtures: '+path+'. Use JSON or deterministic seed-generated fixtures.');}};
 const contract=JSON.parse(await decode(manifest));
 if(!contract||typeof contract!=='object'||!Array.isArray(contract.tasks)||!['correctness','optimization'].includes(contract.objective)||!['cpu','a3','a5'].includes(contract.target)||contract.target==='cpu'&&contract.objective!=='correctness'||contract.tasks?.length!==5)throw new SkillEvaluationContractError('skill-evaluation.json requires target, correctness|optimization objective and exactly five cases');
 const path=async name=>{
  if(typeof name!=='string'||!name||isAbsolute(name))throw new SkillEvaluationContractError('Evaluation paths must be relative to the skill-evaluation.json directory');
  let file;try{file=await realpath(resolve(root,name));}catch(error){if(error.code==='ENOENT')throw new SkillEvaluationContractError('Evaluation path '+JSON.stringify(name)+' does not exist relative to '+root+'. Paths are manifest-relative, not project-relative.');throw error;}
  if(!within(root,file)||within(source,file)||!declared.has(file))throw new SkillEvaluationContractError('Evaluation dependency must be protected, manifest-relative and declared in verifier.files: '+name);
  await decode(file);return file;
 };
 const files=async mapping=>{
  if(!mapping||typeof mapping!=='object'||Array.isArray(mapping)||typeof mapping['kernel.py']!=='string')throw new SkillEvaluationContractError('Each case and control requires a files map containing kernel.py');
  const result={};for(const [name,file]of Object.entries(mapping)){if(!name||isAbsolute(name)||!within('/sample',resolve('/sample',name))||name==='.')throw new SkillEvaluationContractError('Sample file names must stay inside the sample directory');result[name]=await decode(await path(file));}return result;
 };
 const ids=new Set(),tasks=[];
 for(const task of contract.tasks){
  if(!task||typeof task!=='object')throw new SkillEvaluationContractError('Evaluation cases must be objects');
  const target=task.target??(task.kind==='a3-kernel'?'a3':task.kind==='hypothesis-case'?'cpu':null),kindMatches=contract.target==='cpu'?task.kind==='hypothesis-case':['a3-kernel','target-kernel'].includes(task.kind);
  if(!/^[a-z0-9-]+$/.test(task.id)||ids.has(task.id)||!kindMatches||target!==contract.target||task.bucket!==contract.objective||typeof task.prompt!=='string'||!task.prompt.trim())throw new SkillEvaluationContractError('Evaluation cases require unique lowercase IDs, CPU hypothesis-case or device target-kernel/a3-kernel kind, matching target/bucket and a prompt');ids.add(task.id);
  if(!Array.isArray(task.controls?.negatives)||!task.controls.negatives.length)throw new SkillEvaluationContractError('Every evaluation case requires a reference and at least one negative control');
  if(contract.objective==='optimization'&&!task.pairedChecker)throw new SkillEvaluationContractError('Optimization cases require a pairedChecker');
  tasks.push({...task,checker:await path(task.checker),...(task.pairedChecker?{pairedChecker:await path(task.pairedChecker)}:{}),files:await files(task.files),controls:{reference:{files:await files(task.controls?.reference?.files)},negatives:await Promise.all(task.controls.negatives.map(async negative=>({files:await files(negative?.files)})))}});
 }
 return {manifest,contract,tasks};
}
