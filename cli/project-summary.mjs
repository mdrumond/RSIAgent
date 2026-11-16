/** Deterministic project facts and scoped claim evidence; no model arithmetic. */
import {DatabaseSync} from 'node:sqlite';
import {open,stat,readdir,realpath} from 'node:fs/promises';
import {join,relative,sep} from 'node:path';
import {createHash} from 'node:crypto';
const digest=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
function jsonObject(text){
 const start=text.indexOf('{');if(start<0)return null;let depth=0,string=false,escape=false;
 for(let i=start;i<text.length;i++){const c=text[i];if(string){if(escape)escape=false;else if(c==='\\')escape=true;else if(c==='"')string=false;}else if(c==='"')string=true;else if(c==='{')depth++;else if(c==='}'&&!--depth){try{return JSON.parse(text.slice(start,i+1));}catch{return null;}}}
 return null;
}
export async function projectSummary({state,projectId,artifactResolver=path=>path}){
 const row=state.db.prepare('SELECT id,status FROM projects WHERE id=?').get(projectId);
 if(!row)return {format:1,projectId,available:false,facts:[],references:[],unknowns:['Project not present in this evidence database']};
 const attempts=state.db.prepare('SELECT id,outcome,created FROM attempts WHERE project=? ORDER BY created,id').all(projectId).map(a=>({...a}));
 const summary={format:1,projectId,available:true,projectStatus:row.status,eventCutoff:0,attempts,verifications:[],measurements:[],facts:[],references:[],unknowns:[]};
 const fact=(name,value,sourceRefs=[])=>{const reference=`project:${projectId}:${name}`;summary.facts.push({reference,factId:reference,value,text:JSON.stringify(value),sourceRefs});};
 for(const event of state.db.prepare("SELECT e.sequence,e.attempt,e.type,e.data,e.created FROM events e JOIN attempts a ON a.id=e.attempt WHERE a.project=? AND e.type IN ('verification','failure-recorded','attempt-error','project-invalid','preactor-infrastructure','rejected') ORDER BY e.sequence").iterate(projectId)){
  const data=JSON.parse(event.data),reference=`project:${projectId}:events:${event.sequence}`;
  summary.eventCutoff=event.sequence;
  summary.references.push({reference,text:JSON.stringify({sequence:event.sequence,attemptId:event.attempt,created:event.created,type:event.type,data})});
  if(event.type!=='verification')continue;
  const record={reference,attemptId:event.attempt,created:event.created,passed:data.passed??null,correctnessPassed:data.correctness?.passed??null,performancePassed:data.performance?.passed??null,fingerprint:data.fingerprint??null};summary.verifications.push(record);
  const evidence=data.performance?.evidence;if(typeof evidence!=='string')continue;
  let file;try{
   file=await open(artifactResolver(evidence),'r');const info=await file.stat();if(!info.isFile()||info.size>1048576){summary.unknowns.push({reference,artifact:evidence,reason:'Performance artifact unavailable as bounded complete evidence'});continue;}
   const bytes=await file.readFile(),sha256=createHash('sha256').update(bytes).digest('hex');
   if(data.performance.evidence_sha256&&sha256!==data.performance.evidence_sha256){summary.unknowns.push({reference,artifact:evidence,reason:'Performance artifact hash mismatch'});continue;}
   const result=jsonObject(bytes.toString('utf8')),gain=result?.geometricMeanSpeedup??result?.geomean_speedup;
   if(!Number.isFinite(gain)||gain<=0){summary.unknowns.push({reference,artifact:evidence,reason:'No aggregate performance gain in artifact'});continue;}
   const measurement={reference,artifact:evidence,sha256,verified:record.correctnessPassed===true&&record.performancePassed===true&&result.passed!==false&&result.ok!==false,geometricMeanSpeedup:gain,
    candidateHash:result.candidateHash??null,sourceHash:result.sourceHash??null,caseHash:result.caseHash??null,verifierHash:result.verifierHash??null,runtimeFingerprint:result.runtimeFingerprint??null,caseIds:Array.isArray(result.cases)&&result.cases.every(c=>c&&typeof c==='object'&&!Array.isArray(c))?result.cases.map(c=>c.id??null):null};summary.measurements.push(measurement);
   if(result.cases!==undefined&&measurement.caseIds===null)summary.unknowns.push({reference,reason:'Optional cases metadata is not an array of case records'});
   if((result.passed===false||result.ok===false)&&record.performancePassed===true)summary.unknowns.push({reference,reason:'Performance artifact contradicts controller pass'});
   for(const key of ['geometricMeanSpeedup','candidateHash','sourceHash','caseHash','verifierHash','runtimeFingerprint'])fact(`measurement:${event.sequence}:${key}`,measurement[key],[reference]);
  }catch(error){if(!['ENOENT','EACCES','EISDIR'].includes(error.code))throw error;summary.unknowns.push({reference,artifact:evidence,reason:error.code});}finally{await file?.close();}
 }
 const refs=summary.verifications.map(v=>v.reference);
 fact('projectStatus',row.status);fact('eventCutoff',summary.eventCutoff);fact('unknownVerificationCount',summary.verifications.filter(v=>v.passed===null).length,refs);fact('attemptCount',attempts.length);fact('verificationCount',summary.verifications.length,refs);fact('passingVerificationCount',summary.verifications.filter(v=>v.passed).length,refs);fact('performancePassCount',summary.verifications.filter(v=>v.performancePassed===true).length,refs);
 summary.identity=digest(summary);return summary;
}
export async function projectSummaries(config,groups){
 const projectIds=new Set(groups.flatMap(g=>[...(g.occurrences??[]),...(g.linkedDiagnostics??[])].map(o=>o.projectId).filter(Boolean)));
 if(!config?.root)return [...projectIds].map(projectId=>({projectId,available:false,facts:[],references:[],unknowns:['No evidence database configured']}));
 const path=join(config.project_evidence_root??config.root,'state.sqlite');try{await stat(path);}catch(error){if(error.code==='ENOENT')return [...projectIds].map(projectId=>({projectId,available:false,facts:[],references:[],unknowns:['Evidence database unavailable']}));throw error;}
 const db=new DatabaseSync(path,{readOnly:true});try{
  for(const group of groups)for(const o of [...(group.occurrences??[]),...(group.linkedDiagnostics??[])])if(o.attemptId){const a=db.prepare('SELECT project FROM attempts WHERE id=?').get(o.attemptId);if(a)projectIds.add(a.project);}
  const summaries=[];for(const projectId of projectIds)summaries.push(await projectSummary({state:{db},projectId}));return summaries;
 }finally{db.close();}
}
/** Only copied/verified learning is a source; source code does not itself prove device execution. */
export async function learningProofEvidence(learning=[],learningEvidence=[]){
 const records=[],seen=new Set();let totalBytes=0;
 for(const [id,record]of learning){const result=record.result;if(!['verified-cpu','verified-device','verified-project'].includes(result?.status))continue;
  const metadata={learningId:id,status:result.status,binding:result.binding??null},declared=[result.project?.source,...(result.project?.verifier?.files??[]),...(result.evidence??[]).flatMap(e=>[e.log,e.path])].filter(p=>typeof p==='string');
  records.push({reference:'learning:'+id+':result',text:JSON.stringify(result),...metadata});for(const e of result.evidence??[])if(e.reference&&!e.log&&!e.path)records.push({reference:e.reference,text:JSON.stringify(e),...metadata});let visited=0;
  const read=async(path)=>{if(seen.has(path))return;seen.add(path);let file;
   try{const actual=await realpath(path),info=await stat(actual);
    if(info.isDirectory()){for(const entry of await readdir(actual,{withFileTypes:true})){if(++visited>128){records.push({reference:'file:'+path,available:false,reason:'Learning source inventory truncated',...metadata});break;}if(!entry.name.startsWith('.')&&!['node_modules','__pycache__'].includes(entry.name)&&!entry.isSymbolicLink())await read(join(actual,entry.name));}return;}
    if(!info.isFile()||info.size>65536||totalBytes+info.size>262144){records.push({reference:'file:'+path,available:false,reason:'Learning source unavailable as bounded complete evidence',...metadata});return;}
    totalBytes+=info.size;file=await open(actual,'r');const bytes=await file.readFile(),sha256=createHash('sha256').update(bytes).digest('hex'),expected=[...(result.evidence??[]),...learningEvidence].find(e=>(e.log??e.path)===path)?.sha256;
    if(expected&&sha256!==expected){records.push({reference:'file:'+path,available:false,reason:'Learning source hash mismatch',...metadata});return;}
    const text=bytes.toString('utf8');for(const reference of new Set(['file:'+path,'file:'+actual,...(result.evidence??[]).filter(e=>(e.log??e.path)===path).map(e=>e.reference).filter(Boolean)]))records.push({reference,text,sha256,...metadata});
   }catch(error){if(!['ENOENT','EACCES','EISDIR'].includes(error.code))throw error;records.push({reference:'file:'+path,available:false,reason:error.code,...metadata});}finally{await file?.close();}
  };
  for(const path of declared)await read(path);
  // Additional evidence may alias a verified declared file/tree, never expand its scope.
  for(const e of learningEvidence)if(typeof e.path==='string'&&declared.some(p=>e.path===p||relative(p,e.path)&&relative(p,e.path)!=='..'&&!relative(p,e.path).startsWith('..'+sep)&&!relative(p,e.path).startsWith(sep)))await read(e.path);
 }
 return records;
}
export function claimEvidence({prompt={},packet={},retrievals=[]}){
 const records=[];
 const add=(reference,text,extra={})=>{if(reference&&typeof text==='string')records.push({reference,text,...extra});};
 for(const s of prompt.projectSummaries??[])for(const r of [...(s.facts??[]),...(s.references??[])])records.push(r);
 for(const o of packet.observations??[])add(o.reference,[o.message,JSON.stringify(o.data)].filter(Boolean).join('\n'));
 for(const a of packet.artifacts??[])if(a.available)add(a.id,a.text,{segments:a.segments??[a.text]});
 for(const e of [...(prompt.entries??[]),...(prompt.baseline??[])])add(e.id,e.text);
 for(const e of [...(prompt.memory??[]),...retrievals.flatMap(r=>r.entries??[])])add(e.ref??e.citationId??`${e.collection}:${e.entry_id??e.chunk_id??e.id}`,e.text??e.content??JSON.stringify(e));
 for(const record of prompt.learningProofEvidence??[])if(record.available!==false&&typeof record.text==='string')records.push(record);
 for(const key of ['proposal','answers','answerClaims'])if(prompt[key]!==undefined)add('input:'+key,JSON.stringify(prompt[key]));
 return [...new Map(records.map(r=>[r.reference,r])).values()];
}
export function validateClaims(claims,records){
 if(!Array.isArray(claims)||!claims.length)throw Error('Provide proof-bearing claims; factual prose without claims is not accepted');
 const sources=new Map(records.map(r=>[r.reference,r]));
 for(const claim of claims){
  if(!claim.text?.trim()||!['observation','hypothesis'].includes(claim.kind)||!Array.isArray(claim.proofs)||!claim.proofs.length)throw Error('Each claim requires text, observation|hypothesis kind, and copied proofs');
  for(const proof of claim.proofs){const source=sources.get(proof.reference);if(!source)throw Error('Unknown proof reference: '+proof.reference);
   if(typeof proof.quote!=='string'||!proof.quote.trim()||!(source.segments??[source.text]).some(s=>s.includes(proof.quote)))throw Error('Quote does not match '+proof.reference+'; copy one contiguous source span from the cited record');
   if(source.factId&&(proof.factId!==source.factId||!Object.hasOwn(proof,'value')||JSON.stringify(proof.value)!==JSON.stringify(source.value)))throw Error('Structured fact assertion does not match '+source.factId+'; use its exact factId and value');
  }
 }
 return claims;
}
export function validateClaimedResult(value,records){
 const targets=new Map(),inline=[];
 const visit=(x,path='')=>{if(!x||typeof x!=='object')return;for(const [key,item]of Object.entries(x)){if(key==='claims')continue;const child=path?path+'.'+key:key;
  if(typeof item==='string'&&['reason','text','findings','problem','description'].includes(key))targets.set(child,item);
  else if(key==='issues'&&Array.isArray(item))item.forEach((issue,i)=>{if(typeof issue==='string')targets.set(child+'.'+i,issue);else{visit(issue,child+'.'+i);if(issue?.reference!==undefined||issue?.quote!==undefined)inline.push(issue);}});
  else visit(item,child);
 }};visit(value);
 const paths=[...targets.keys()],hint=' Required prose claim paths: '+JSON.stringify(paths)+'. Categorical fields such as verdict are not prose claim paths.';
 try{
  validateClaims(value?.claims,records);
  for(const [path,text]of targets)if(!value.claims.some(c=>c.path===path&&c.text===text))throw Error('Missing exact proof-bearing claim for '+path);
  for(const c of value.claims)if(!targets.has(c.path)||targets.get(c.path)!==c.text)throw Error('Claim path/text does not match submitted prose: '+c.path);
  for(const issue of inline)validateClaims([{text:issue.problem??issue.description??'Issue citation',kind:'observation',proofs:[issue]}],records);
 }catch(error){throw Error(error.message+hint);}
 return value;
}
export const proofInstructions='Every factual reason/text/findings/problem/description and every issues string requires claims:[{path:exact dotted output field,text:exact field text,kind:observation|hypothesis,proofs:[{reference,quote}]}]. For QA review include reason and each issues.N.problem (or issues.N for string issues); do not create claims for categorical verdict. Copy one contiguous exact quote from the scoped proofEvidence record. For a typed fact also copy its exact factId and value into the proof. Structured issues may use problem/description, but any reference/quote fields they include must also match and carry factId/value for typed facts. Unknown is valid; mark unverified premises as hypothesis, never invent a fact. Quote matching checks copying; the independent QA must also check whether the claim follows from the evidence.';
