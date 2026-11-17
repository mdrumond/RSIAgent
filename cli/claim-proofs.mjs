/** Resolve small source selectors to checked, reproducible proof excerpts. */
import {createHash} from 'node:crypto';
const hash=x=>createHash('sha256').update(x).digest('hex');
const allowed=['reference','quote','factId','value','startLine','endLine','segment','field','sourceSha256','recordSha256','sourceRefs','sourcePaths'];
const own=(x,k)=>Object.hasOwn(x,k),same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
function fieldValue(text,pointer){
 if(typeof pointer!=='string'||pointer!==''&&!pointer.startsWith('/'))throw Error('field must be an RFC6901 JSON pointer');
 let value;try{value=JSON.parse(text);}catch{throw Error('JSON field proof requires a complete serialized input record');}
 for(const part of pointer===''?[]:pointer.slice(1).split('/')){
  if(/~(?:[^01]|$)/.test(part))throw Error('Invalid JSON pointer escape');
  const key=part.replace(/~1/g,'/').replace(/~0/g,'~');
  if(!value||typeof value!=='object'||Array.isArray(value)&&!/^(0|[1-9]\d*)$/.test(key)||!own(value,key))throw Error('Unknown JSON field: '+pointer);
  value=value[key];
 }
 return value;
}
export function materializeProof(proof,records){
 if(!proof||typeof proof!=='object'||Object.keys(proof).some(k=>!allowed.includes(k)))throw Error('Unknown proof selector field');
 const source=records.find(r=>r.reference===proof.reference);if(!source||source.available===false||typeof source.text!=='string')throw Error('Unknown proof reference: '+proof.reference);
 const segments=source.segments??[source.text],sourceRefs=source.sourceRefs??[],sourcePaths=source.paths??[];
 if(!Array.isArray(segments)||!segments.length||segments.some(s=>typeof s!=='string'))throw Error('Proof source segments unavailable');
 const recordSha256=hash(JSON.stringify({text:source.text,segments,factId:source.factId??null,value:source.value??null,sourceRefs,sourcePaths})),sourceSha256=source.sha256??hash(source.text);
 for(const [key,value]of Object.entries({recordSha256,sourceSha256,sourceRefs,sourcePaths}))if(own(proof,key)&&!same(proof[key],value))throw Error('Proof source changed: '+proof.reference+' ('+key+')');
 const lines=own(proof,'startLine')||own(proof,'endLine'),field=own(proof,'field');
 if(lines&&field)throw Error('Use one source selector: lines or field');
 if(!source.factId&&!field&&(own(proof,'factId')||own(proof,'value')))throw Error('Structured assertions require a typed fact reference or JSON field selector');
 let quote,selectedValue,segment=proof.segment;
 if(segment!==undefined&&(!Number.isSafeInteger(segment)||segment<1||segment>segments.length))throw Error('Invalid proof segment');
 if(source.factId){
  if(lines||field||segment!==undefined)throw Error('Typed facts use their whole reference, not a range/field selector');
  if(own(proof,'factId')&&proof.factId!==source.factId||own(proof,'value')&&!same(proof.value,source.value))throw Error('Structured fact assertion does not match '+source.factId);
  quote=proof.quote??source.text;
 }else if(lines){
  if(!Number.isSafeInteger(proof.startLine)||!Number.isSafeInteger(proof.endLine)||proof.startLine<1||proof.endLine<proof.startLine)throw Error('Invalid inclusive proof line range');
  if(segments.length>1&&segment===undefined)throw Error('Excerpt line proof requires an explicit segment; omitted gaps cannot be crossed');
  segment??=1;const parts=segments[segment-1].split('\n');if(parts.at(-1)===''&&segments[segment-1].endsWith('\n'))parts.pop();
  if(proof.endLine>parts.length)throw Error('Proof line range exceeds selected source segment');
  quote=parts.slice(proof.startLine-1,proof.endLine).join('\n');
 }else if(field){
  if(!['input:proposal','input:answers','input:answerClaims'].includes(proof.reference)||segments.length!==1||segment!==undefined)throw Error('JSON field selectors require a complete input:proposal/answers/answerClaims record');
  selectedValue=fieldValue(source.text,proof.field);quote=JSON.stringify(selectedValue);
  if(own(proof,'factId')||own(proof,'value')&&!same(proof.value,selectedValue))throw Error('Structured field assertion does not match '+proof.reference+proof.field);
 }else quote=proof.quote;
 if(typeof quote!=='string'||!quote.trim()||!(segment===undefined?segments:[segments[segment-1]]).some(s=>s.includes(quote)))throw Error('Quote does not match '+proof.reference+'; select one contiguous source span');
 if(own(proof,'quote')&&proof.quote!==quote)throw Error('Supplied quote does not match selected source span: '+proof.reference);
 return {...proof,quote,...(segment===undefined?{}:{segment}),...(source.factId?{factId:source.factId,value:source.value}:field?{value:selectedValue}:{}),sourceSha256,recordSha256,sourceRefs,sourcePaths};
}
export function validateClaims(claims,records){
 if(!Array.isArray(claims)||!claims.length)throw Error('Provide proof-bearing claims; factual prose without claims is not accepted');
 for(const claim of claims){
  if(typeof claim.text!=='string'||!claim.text.trim()||!['observation','hypothesis'].includes(claim.kind)||!Array.isArray(claim.proofs)||!claim.proofs.length)throw Error('Each claim requires text, observation|hypothesis kind, and source proofs');
  claim.proofs=claim.proofs.map(p=>materializeProof(p,records));
 }
 return claims;
}
export function validateClaimedResult(value,records){
 const targets=new Map(),inline=[];
 const visit=(x,path='')=>{if(!x||typeof x!=='object')return;for(const [key,item]of Object.entries(x)){if(key==='claims')continue;const child=path?path+'.'+key:key;
  if(typeof item==='string'&&['reason','text','findings','problem','description'].includes(key))targets.set(child,item);
  else if(key==='issues'&&Array.isArray(item))item.forEach((issue,i)=>{if(typeof issue==='string')targets.set(child+'.'+i,issue);else{visit(issue,child+'.'+i);if(issue&&allowed.some(k=>k!=='field'&&own(issue,k)))inline.push({issue,path:child+'.'+i});}});
  else visit(item,child);
 }};visit(value);
 const hint=' Required prose claim paths: '+JSON.stringify([...targets.keys()])+'. Categorical fields such as verdict are not prose claim paths.';
 try{
  for(const [i,c]of (value?.claims??[]).entries()){
   if(c&&c.text===undefined&&targets.has(c.path))c.text=targets.get(c.path);
   if(!targets.has(c?.path))throw Error('claims.'+i+'.path does not identify submitted prose: '+c?.path);
   try{validateClaims([c],records);}catch(error){throw Error('claims.'+i+' ('+c.path+'): '+error.message);}
  }
  if(!value?.claims?.length)validateClaims(value?.claims,records);
  for(const [path,text]of targets)if(!value.claims.some(c=>c.path===path&&c.text===text))throw Error('Missing exact proof-bearing claim for '+path);
  for(const c of value.claims)if(!targets.has(c.path)||targets.get(c.path)!==c.text)throw Error('Claim path/text does not match submitted prose: '+c.path);
  for(const {issue,path}of inline){
   if(typeof issue.reference!=='string'||!issue.reference.trim())throw Error(path+'.reference is required for an inline issue proof; put selectors in top-level claims instead');
   try{Object.assign(issue,materializeProof(Object.fromEntries(allowed.filter(k=>own(issue,k)).map(k=>[k,issue[k]])),records));}catch(error){throw Error(path+': '+error.message);}
  }
 }catch(error){throw Error(error.message+hint);}
 return value;
}
export const proofInstructions='Every factual reason/text/findings/problem/description and every issues string requires a TOP-LEVEL claims array of {path,kind:observation|hypothesis,proofs:[{reference,...selector}]}; claim.text may be omitted and is copied from its actual output path. Categorical verdict is not a prose claim path. Prefer whole typed fact {reference}; code supplies exact factId/value. For source lines use 1-based inclusive startLine/endLine and, for excerpts, explicit 1-based segment. For serialized input:proposal/answers/answerClaims use field as an RFC6901 pointer, e.g /changes/0/text. Code resolves the exact quote and source identity; do not retype prose or quotes. Legacy reference+quote remains valid; supplied quote/factId/value/source hashes must match. QA must still check entailment: source selection and copying do not prove the claim follows.';
