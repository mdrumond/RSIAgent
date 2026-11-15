/** Shared admission and evaluation loader. Manifest paths are relative to its directory. */
import {readFile,realpath} from 'node:fs/promises';
import {dirname,resolve,relative,isAbsolute} from 'node:path';
const within=(root,path)=>{const rel=relative(root,path);return rel===''||!isAbsolute(rel)&&rel!=='..'&&!rel.startsWith('../');};
export async function loadSkillEvaluation(spec,{required=false}={}){
 const declarations=spec.verifier?.files??[],manifests=declarations.filter(file=>file.endsWith('/skill-evaluation.json'));
 if(!manifests.length){if(required)throw Error('Suite learning requires a protected verifier/skill-evaluation.json declared in verifier.files before actor admission');return null;}
 if(manifests.length!==1)throw Error('Declare exactly one skill-evaluation.json');
 const manifest=await realpath(manifests[0]),root=dirname(manifest),source=await realpath(spec.source),declared=new Set(await Promise.all(declarations.map(file=>realpath(file))));
 const decode=async path=>{const bytes=await readFile(path);try{const text=new TextDecoder('utf-8',{fatal:true}).decode(bytes);if(text.includes('\0'))throw Error('Binary NUL');return text;}catch{throw Error('Evaluation files must be UTF-8 text, not binary fixtures: '+path+'. Use JSON or deterministic seed-generated fixtures.');}};
 const contract=JSON.parse(await decode(manifest));
 if(!['correctness','optimization'].includes(contract.objective)||typeof contract.target!=='string'||!contract.target||contract.tasks?.length!==5)throw Error('skill-evaluation.json requires target, correctness|optimization objective and exactly five cases');
 const path=async name=>{
  if(typeof name!=='string'||!name||isAbsolute(name))throw Error('Evaluation paths must be relative to the skill-evaluation.json directory');
  let file;try{file=await realpath(resolve(root,name));}catch(error){if(error.code==='ENOENT')throw Error('Evaluation path '+JSON.stringify(name)+' does not exist relative to '+root+'. Paths are manifest-relative, not project-relative.');throw error;}
  if(!within(root,file)||within(source,file)||!declared.has(file))throw Error('Evaluation dependency must be protected, manifest-relative and declared in verifier.files: '+name);
  await decode(file);return file;
 };
 const files=async mapping=>{
  if(!mapping||typeof mapping!=='object'||Array.isArray(mapping)||typeof mapping['kernel.py']!=='string')throw Error('Each case and control requires a files map containing kernel.py');
  const result={};for(const [name,file]of Object.entries(mapping)){if(!name||isAbsolute(name)||!within('/sample',resolve('/sample',name))||name==='.')throw Error('Sample file names must stay inside the sample directory');result[name]=await decode(await path(file));}return result;
 };
 const ids=new Set(),tasks=[];
 for(const task of contract.tasks){
  if(!/^[a-z0-9-]+$/.test(task.id)||ids.has(task.id)||task.kind!=='hypothesis-case'||task.target!==contract.target||task.bucket!==contract.objective||!task.prompt?.trim())throw Error('Evaluation cases require unique lowercase IDs, hypothesis-case kind, matching target/bucket and a prompt');ids.add(task.id);
  if(!Array.isArray(task.controls?.negatives)||!task.controls.negatives.length)throw Error('Every evaluation case requires a reference and at least one negative control');
  if(contract.objective==='optimization'&&!task.pairedChecker)throw Error('Optimization cases require a pairedChecker');
  tasks.push({...task,checker:await path(task.checker),...(task.pairedChecker?{pairedChecker:await path(task.pairedChecker)}:{}),files:await files(task.files),controls:{reference:{files:await files(task.controls?.reference?.files)},negatives:await Promise.all(task.controls.negatives.map(async negative=>({files:await files(negative.files)})))}});
 }
 return {manifest,contract,tasks};
}
