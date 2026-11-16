/** Immutable primary records live on disk; summaries never become proof sources. */
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {claimEvidence,validateClaimedResult} from './project-summary.mjs';
import {frozenEvidencePacket} from './skill-review-evidence.mjs';
export const summaryProtocol='primary-summary-v1';
export const sourceIdentity=records=>createHash('sha256').update(JSON.stringify(records)).digest('hex');
const memoryRef=m=>m.ref??m.citationId??`${m.collection}:${m.entry_id??m.chunk_id??m.id}`;
export async function prepareQaSources(input){
 let memory=input.memoryEvidence;
 if(!memory){const text=await readFile(join(input.proposalDirectory??input.directory,'retrievals.jsonl'),'utf8').catch(e=>{if(e.code!=='ENOENT')throw e;return '';});memory=text.split('\n').filter(Boolean).flatMap(line=>JSON.parse(line).entries??[]);}
 const cited=new Set(input.proposal.memoryRefs??[]);memory=memory.filter(m=>cited.has(memoryRef(m)));
 const packet=await frozenEvidencePacket(input.groups),prompt={projectSummaries:input.projectSummaries,memory,learningProofEvidence:input.learningProofEvidence};
 const records=claimEvidence({prompt,packet}).map(r=>({...r})),observations=new Map(packet.observations.map(o=>[o.reference,o]));
 for(const r of records){const o=observations.get(r.reference),tool=o?.data?.sourceCall?.tool;
  r.provenance=r.factId?'typed-controller-fact':memory.some(m=>memoryRef(m)===r.reference)?'actor-memory-report':/source|code/.test(o?.type??'')?'code-text':tool==='rsi_target'?'target-tool-record':tool==='bash'?'host-tool-record':r.learningId?'learning-artifact':'unknown';
  r.observationType=o?.type??null;
 }
 // Literal aliases retain parent identity and byte offsets; originals remain untouched for legacy proofs.
 const originals=[...records],spanBytes=16*1024;
 for(const record of originals)for(const [segment,text]of (record.segments??[record.text]).entries()){
  if(Buffer.byteLength(text)<=spanBytes)continue;
  let offset=0,byteOffset=0,part=0;
  while(offset<text.length){let end=Math.min(offset+spanBytes,text.length);while(Buffer.byteLength(text.slice(offset,end))>spanBytes)end--;if(end<text.length&&/[\uD800-\uDBFF]/.test(text[end-1]))end--;
   const prefix=byteOffset,literal=text.slice(offset,end);byteOffset+=Buffer.byteLength(literal);
   records.push({reference:`${record.reference}:span:${segment+1}:${++part}`,text:literal,sourceRefs:[record.reference],sourcePaths:record.sourcePaths??[],provenance:record.provenance,observationType:record.observationType,parent:{reference:record.reference,sha256:record.sha256??createHash('sha256').update(record.text).digest('hex'),segment:segment+1,startByte:prefix,endByte:prefix+Buffer.byteLength(literal)}});offset=end;
  }
 }

 const identity=sourceIdentity(records),directory=join(input.directory,'primary-sources',identity);await mkdir(directory,{recursive:true});
 const sources=[];
 for(const record of records){
  const stem=createHash('sha256').update(record.reference).digest('hex').slice(0,24),segments=[];
  for(const [i,text]of (record.segments??[record.text]).entries()){
   const path=join(directory,`${stem}-${i+1}.txt`);await writeFile(path,text);segments.push({segment:i+1,path,lines:text.split('\n').length,bytes:Buffer.byteLength(text)});
  }
  sources.push({reference:record.reference,provenance:record.provenance,sha256:record.sha256??createHash('sha256').update(record.text).digest('hex'),segments,observationType:record.observationType,...(record.parent?{parent:record.parent}:{}),...(record.factId?{factId:record.factId,value:record.value}:{})});
 }
 const indexFile=join(directory,'catalogue.json');await writeFile(indexFile,JSON.stringify(sources,null,2));
 const citedRefs=new Set((input.proposal.changes??[]).flatMap(c=>c.evidenceRefs??[])),scope=new Set();
 for(const g of input.groups)if([...g.occurrences,...(g.linkedDiagnostics??[])].some(o=>citedRefs.has(o.reference)))for(const o of [...g.occurrences.filter(o=>citedRefs.has(o.reference)),...(g.linkedDiagnostics??[])]){scope.add(o.reference);for(const a of observations.get(o.reference)?.artifacts??[])scope.add(a);}
 const pages=[],pending=sources.filter(s=>(scope.has(s.reference)||scope.has(s.parent?.reference)||memory.some(m=>memoryRef(m)===s.reference)||s.reference.startsWith('learning:'))&&!s.factId&&!s.reference.startsWith('project:')&&s.segments.reduce((n,p)=>n+p.bytes,0)<=spanBytes);
 let rows=[],bytes=0;for(const source of pending){const size=source.segments.reduce((n,p)=>n+p.bytes,0);if(rows.length&&(rows.length===4||bytes+size>spanBytes)){pages.push({id:sourceIdentity(rows),sources:rows});rows=[];bytes=0;}rows.push(source);bytes+=size;}if(rows.length)pages.push({id:sourceIdentity(rows),sources:rows});
 return {identity,directory,indexFile,records,sources,pages,facts:sources.filter(s=>s.factId&&!s.reference.includes(':measurement:'))};
}
export function validateSummary(value,catalogue){
 if(Buffer.byteLength(JSON.stringify({items:value?.items,claims:value?.claims?.map(({path,kind,proofs})=>({path,kind,proofs:proofs?.map(({reference,startLine,endLine,segment,field})=>({reference,startLine,endLine,segment,field}))}))}))>8192)throw Error('Summary page exceeds the compact output byte limit');
 if(!Array.isArray(value?.items)||!value.items.length)throw Error('Evidence summary requires source-bound items');
 validateClaimedResult(value,catalogue.records);
 for(const claim of value.claims)if(claim.proofs.reduce((n,p)=>n+Buffer.byteLength(p.quote??''),0)>16384)throw Error('Selected primary excerpts exceed 16 KiB; use narrower literal span aliases');
 for(const [i,item]of value.items.entries()){
  const claim=value.claims.find(c=>c.path===`items.${i}.text`);
  if(!['observation','hypothesis'].includes(item.kind)||claim.kind!==item.kind)throw Error('Summary item kind must match its source-bound claim');
 }
 return value;
}
export function summaryItems(value,pageId){
 return value.items.map((item,i)=>{const claim=value.claims.find(c=>c.path===`items.${i}.text`);return {...item,pageId,proofs:claim.proofs,id:sourceIdentity([pageId,item,claim.proofs])};});
}
export function sourcePointers(catalogue,refs){return catalogue.sources.filter(s=>refs.includes(s.reference));}
export function compactSummary(summary,catalogue){
 return {sourceIdentity:catalogue.identity,items:summary.items.map(({id,text,kind,proofs})=>({id,text,kind,sources:proofs.map(({reference,startLine,endLine,segment,field})=>({reference,...(startLine?{startLine,endLine}:{}),...(segment?{segment}:{}),...(field!==undefined?{field}:{})}))})),authoritativeFacts:catalogue.facts,sourceCatalogue:catalogue.indexFile};
}
