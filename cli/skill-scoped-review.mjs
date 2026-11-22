import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {Type} from 'typebox';
import {reviewSession} from './skill-review-native.mjs';

export const scopedPolicy='scoped-v1';
export const artifactHash=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const nonempty=Type.String({minLength:1});
const verdictSchema=Type.Union([
 Type.Object({verdict:Type.Literal('approved')},{additionalProperties:false}),
 Type.Object({verdict:Type.Literal('revise'),field:nonempty,quote:nonempty,defect:nonempty},{additionalProperties:false}),
]);

export function scopedReviewIdentity({scope,brief,fields,sources=[]}){
 return artifactHash({scope,brief,fields,sources:sources.map(({reference,text})=>[reference,artifactHash(text)])});
}

export async function reviewScopedArtifact({scope,fields,brief,directory,config,seconds,signal,allowedPaths=[],evidenceGroups=[],sources=[]},session=reviewSession){
 assert(['writing','test'].includes(scope)&&Object.keys(fields).length,'Scoped review needs a concrete artifact');
 const validate=value=>{
  if(value.verdict==='approved'){assert.deepEqual(value,{verdict:'approved'});return value;}
  assert(value.verdict==='revise'&&typeof value.defect==='string'&&value.defect.trim(),'Explain one actionable defect');
  assert(Object.hasOwn(fields,value.field)&&typeof value.quote==='string'&&value.quote.trim()&&fields[value.field].includes(value.quote),'Select a field and copy the exact defective text from that field');
  return value;
 };
 // Keep the prompt small and retain exact source bytes for native paged reads.
 const sourceDirectory=join(directory,'review-sources');
 await mkdir(sourceDirectory,{recursive:true});
 async function page(name,text){
  const path=join(sourceDirectory,artifactHash([name,text])+'.txt');await writeFile(path,text);
  return {name,path,sha256:artifactHash(text),preview:text.slice(0,1200)};
 }
 const displayed={};
 for(const [key,text] of Object.entries(fields))displayed[key]=text.length<=2000?text:await page(key,text);
 const evidence=[];
 for(const {reference,text} of sources)evidence.push(await page(reference,text));
 const {value}=await session({directory,config,seconds,signal,knowledge:false,allowedPaths:[...allowedPaths,sourceDirectory],evidenceGroups,resultSchema:verdictSchema,
  prompt:{scope,brief,fields:displayed,evidence,task:scope==='writing'?'Review only the proposed instruction for factual accuracy, applicability to the ranked issue, and clear reusable guidance. Benchmark design and empirical effectiveness are evaluated later. Approve the submitted text or quote one material defect for the writer.':'Review whether this test discriminates the ranked failure, whether the oracle and controls test the stated mechanism, and whether transfers preserve it. Inspect the referenced checker files. Approve the submitted test or quote one material defect for the test author.'},validate});
 return {...validate(value),identity:scopedReviewIdentity({scope,brief,fields,sources})};
}


export async function reviewScopedTest({writing,family,directory,config,seconds,signal},review=reviewScopedArtifact){
 const fields={mechanism:JSON.stringify(family.mechanism)};
 const allowedPaths=[];
 for(const task of [family.development,...family.transfer]){
  for(const [name,value] of Object.entries({prompt:task.prompt,caseIds:task.caseIds,files:task.files,controls:task.controls,
    checker:await readFile(task.checker,'utf8'),publicChecker:await readFile(task.publicChecker,'utf8'),relation:task.relation,checkerFiles:task.checkerFiles})){
   if(value!==undefined)fields[task.id+'.'+name]=typeof value==='string'?value:JSON.stringify(value);
  }
  allowedPaths.push(task.checker,task.publicChecker);
 }
 const brief={...writing.brief,approvedInstruction:writing.fields,checkedFamily:family.identity};
 return review({scope:'test',fields,brief,directory,config,seconds,signal,allowedPaths,sources:writing.sources,evidenceGroups:writing.groups});
}
