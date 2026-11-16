/** On-demand primary proofs for oversized discovery records; never inline source text. */
import {createHash} from 'node:crypto';
import {readFile,writeFile} from 'node:fs/promises';
import {join,dirname,resolve} from 'node:path';
const hash=value=>createHash('sha256').update(value).digest('hex');
const pointer=(value,path)=>path.slice(1).split('/').reduce((v,key)=>v?.[key.replaceAll('~1','/').replaceAll('~0','~')],value);
async function checked(file){
 const data=await readFile(file.path);
 if(hash(data)!==file.sha256||file.bytes!==undefined&&data.length!==file.bytes)throw Error('Discovery proof source changed: '+file.path);
 return data;
}
export async function discoveryProofSources(groups){
 const records=[],indexes=[];
 for(const group of groups??[])for(const row of [...(group.occurrences??[]),...(group.linkedDiagnostics??[])]){
  const parent=row.data?.evidencePointer,readable=parent?.readableText;if(!readable)continue;
  const raw=await checked(parent),original=JSON.parse(raw);
  if(original.reference!==row.reference||parent.reference!==row.reference)throw Error('Discovery proof parent identity changed');
  const index=(await checked(readable)).toString('utf8').trim().split('\n').filter(Boolean).map(JSON.parse),descriptors=[];
  const fields=new Map();
  for(const part of index){
   if(dirname(resolve(part.path))!==resolve(readable.directory))throw Error('Discovery proof part outside retained directory');
   const source=pointer(original,part.field);
   if(typeof source!=='string')throw Error('Discovery proof field unavailable');
   const full=Buffer.from(source),text=await checked(part);
   if(!Number.isSafeInteger(part.startByte)||!Number.isSafeInteger(part.endByte)||part.startByte<0||part.endByte<=part.startByte||part.endByte>full.length||text.length>16384||hash(full)!==part.fieldSha256||!text.equals(full.subarray(part.startByte,part.endByte)))throw Error('Discovery proof field binding changed');
   const end=fields.get(part.field)??0;if(part.startByte!==end)throw Error('Discovery proof field coverage changed');fields.set(part.field,part.endByte);
   const reference='discovery-part:'+hash(JSON.stringify([row.reference,parent.sha256,part.field,part.startByte,part.endByte,part.sha256]));
   const descriptor={reference,parentReference:row.reference,parentSha256:parent.sha256,...part};descriptors.push(descriptor);
   records.push({reference,text:text.toString('utf8'),sha256:part.sha256,sourceRefs:[row.reference],paths:[parent.path,part.path]});
  }
  if(index.length!==readable.parts||fields.size!==readable.fields||[...fields].some(([field,end])=>end!==Buffer.byteLength(pointer(original,field))))throw Error('Discovery proof index coverage changed');
  const path=join(readable.directory,'proof-index-v2.jsonl'),text=descriptors.map(d=>JSON.stringify(d)).join('\n')+'\n';
  try{await writeFile(path,text,{flag:'wx',mode:0o600});}catch(error){if(error.code!=='EEXIST')throw error;if(await readFile(path,'utf8')!==text)throw Error('Discovery proof catalogue changed');}
  indexes.push({parentReference:row.reference,path,sha256:hash(text),parts:descriptors.length});
 }
 return {records,indexes};
}
