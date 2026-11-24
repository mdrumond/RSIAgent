import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {setTimeout as delay} from 'node:timers/promises';
import {join} from 'node:path';
import {Type} from 'typebox';
import {reviewSession} from './skill-review-native.mjs';

export const scopedPolicy='scoped-v1';
export function experimentPolicy(requested){
 const policy=requested.experimentPolicy??scopedPolicy;
 assert([scopedPolicy,'legacy'].includes(policy),'Unknown failure-first experiment policy');return policy;
}
const textHash=value=>createHash('sha256').update(value).digest('hex');
const artifactHash=value=>textHash(JSON.stringify(value));
const nonempty=Type.String({minLength:1});
const verdictSchema=Type.Union([
 Type.Object({verdict:Type.Literal('approved')},{additionalProperties:false}),
 Type.Object({verdict:Type.Literal('revise'),field:nonempty,quote:nonempty,defect:nonempty},{additionalProperties:false}),
]);

export function scopedReviewIdentity({scope,brief,fields,sources=[]}){
 return artifactHash({scope,brief,fields,sources:sources.map(({reference,text})=>[reference,textHash(text)])});
}

// Pi retries individual requests. Reopen a closed timed-out turn under its original elapsed grant.
export async function scopedSession(session,request){
 const started=performance.now();let pause=1;
 for(;;){
  request.signal?.throwIfAborted();
  const left=()=>request.seconds-(performance.now()-started)/1000;
  try{return await session({...request,seconds:Math.max(.001,left())});}
  catch(error){
   request.signal?.throwIfAborted();
   if(!error.reviewSession?.retryableTimeout||error.reviewSession.kind!=='provider-error'||left()<=pause)throw error;
   await delay(pause*1000,undefined,{signal:request.signal});pause=Math.min(30,pause*2);
  }
 }
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
 const identity=scopedReviewIdentity({scope,brief,fields,sources});
 const sessionDirectory=join(directory,identity);
 const sourceDirectory=join(sessionDirectory,'review-sources');
 await mkdir(sourceDirectory,{recursive:true});
 async function page(name,text){
  const path=join(sourceDirectory,artifactHash([name,text])+'.txt');await writeFile(path,text);
  return {name,path,sha256:textHash(text),preview:text.slice(0,1200)};
 }
 const displayed={};
 for(const [key,text] of Object.entries(fields))displayed[key]=text.length<=2000?text:await page(key,text);
 const evidence=[];
 for(const {reference,text} of sources)evidence.push(await page(reference,text));
 const {value}=await scopedSession(session,{directory:sessionDirectory,config,seconds,signal,knowledge:false,allowedPaths:[...allowedPaths,sourceDirectory],evidenceGroups,resultSchema:verdictSchema,
  prompt:{scope,brief,fields:displayed,evidence,task:scope==='writing'?'Review only the proposed instruction for factual accuracy, applicability to the ranked issue, and clear reusable guidance. Benchmark design and empirical effectiveness are evaluated later. Approve the submitted text or quote one material defect for the writer.':'Review whether this test discriminates the ranked failure, whether the oracle and controls test the stated mechanism, and whether transfers preserve it. Inspect the referenced checker files. Approve the submitted test or quote one material defect for the test author.'},validate});
 return {...validate(value),identity};
}

export async function reviewScopedTest({writing,family,measurements,directory,config,seconds,signal},review=reviewScopedArtifact){
 const fields={mechanism:JSON.stringify(family.mechanism),...(measurements?{measuredControls:JSON.stringify(measurements)}:{})};
 const allowedPaths=[];
 for(const task of [family.development,...family.transfer]){
  for(const [name,value] of Object.entries({prompt:task.prompt,caseIds:task.caseIds,controlCaseIds:task.controlCaseIds??task.caseIds,changeType:task.changeType,files:task.files,controls:task.controls,
    checker:await readFile(task.checker,'utf8'),publicChecker:await readFile(task.publicChecker,'utf8'),relation:task.relation,checkerFiles:task.checkerFiles})){
   if(value!==undefined)fields[task.id+'.'+name]=typeof value==='string'?value:JSON.stringify(value);
  }
  allowedPaths.push(task.checker,task.publicChecker);
 }
 const brief={...writing.brief,approvedInstruction:writing.fields,checkedFamily:family.identity};
 return review({scope:'test',fields,brief,directory,config,seconds,signal,allowedPaths,sources:writing.sources,evidenceGroups:writing.groups});
}
