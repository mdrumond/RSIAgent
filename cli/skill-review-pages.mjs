/** Source-bound discovery pages; bytes/rows bound context, never elapsed work. */
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';

const digest=text=>createHash('sha256').update(text).digest('hex');
const bytes=value=>Buffer.byteLength(JSON.stringify(value));
async function immutableFile(path,text,reference){
 const sha256=digest(text);
 try{await writeFile(path,text,{flag:'wx',mode:0o600});}catch(error){
  if(error.code!=='EEXIST')throw error;
  if(digest(await readFile(path))!==sha256)throw Error('Retained discovery record changed: '+reference);
 }
 return {path,sha256,bytes:Buffer.byteLength(text)};
}
async function readableFields(record,directory,reference){
 const fields=[];
 const visit=(value,pointer='')=>{
  if(typeof value==='string'&&Buffer.byteLength(value)>1024)fields.push({pointer,bytes:Buffer.from(value)});
  else if(value&&typeof value==='object')for(const [key,item]of Object.entries(value))visit(item,pointer+'/'+key.replaceAll('~','~0').replaceAll('/','~1'));
 };visit(record);if(!fields.length)return null;
 await mkdir(directory,{recursive:true});const index=[];
 for(const field of fields){
  for(let start=0,part=0;start<field.bytes.length;part++){
   let end=Math.min(start+16384,field.bytes.length);
   while(end<field.bytes.length&&(field.bytes[end]&0xc0)===0x80)end--;
   const path=join(directory,digest(field.pointer).slice(0,16)+'-'+part+'.txt');
   const saved=await immutableFile(path,field.bytes.subarray(start,end),reference);
   index.push({field:field.pointer,startByte:start,endByte:end,fieldSha256:digest(field.bytes),...saved});start=end;
  }
 }
 const saved=await immutableFile(join(directory,'index.jsonl'),index.map(row=>JSON.stringify(row)).join('\n')+'\n',reference);
 return {...saved,directory,fields:fields.length,parts:index.length};
}
export function reviewPageLimits(value={}){
 const limits={maxBytes:24576,maxGroups:4,maxRecords:8,...value};
 if(Object.keys(limits).some(k=>!['maxBytes','maxGroups','maxRecords'].includes(k))||
  Object.values(limits).some(n=>!Number.isSafeInteger(n)||n<=0)||limits.maxBytes<1024)
  throw Error('Discovery page limits require positive integers and maxBytes >= 1024');
 return limits;
}
const header=group=>{const {occurrences,linkedDiagnostics,...rest}=group;return rest;};
const rows=groups=>groups.flatMap(group=>[
 ...(group.occurrences??[]).map(record=>({group,kind:'occurrences',record})),
 ...(group.linkedDiagnostics??[]).map(record=>{
  const origins=(group.occurrences??[]).filter(o=>o.relatedDiagnosticRefs?.includes(record.reference));
  const projects=[...new Set(origins.map(o=>o.projectId).filter(Boolean))],attempts=[...new Set(origins.map(o=>o.attemptId).filter(Boolean))];
  const witnesses=[...new Map(origins.map(o=>[JSON.stringify([o.projectId,o.attemptId]),o.reference])).values()];
  return {group,kind:'linkedDiagnostics',record:{...record,
   ...(!record.projectId&&projects.length===1?{projectId:projects[0]}:{}),
   ...(!record.attemptId&&attempts.length===1?{attemptId:attempts[0]}:{}),
   ...(witnesses.length?{pageOriginRefs:witnesses}:{})}};
 })]);

export async function reviewPages(groups,{directory,limits:configured}={}){
 const limits=reviewPageLimits(configured),input=rows(groups),pages=[],all=new Set();
 for(const row of input){
  if(!row.record.reference)throw Error('Discovery records require original references');
  const key=JSON.stringify([row.group.id,row.kind,row.record.reference]);
  if(all.has(key))throw Error('Duplicate discovery reference within a group');
  all.add(key);
 }
 // Empty groups still need a disposition; no record is fabricated for them.
 for(const group of groups)if(!input.some(row=>row.group===group))input.push({group,kind:null,record:null});
 const recordsDirectory=directory&&join(directory,'records');let page={groups:[],bindings:[]};
 const seal=()=>{if(page.groups.length){pages.push(page);page={groups:[],bindings:[]};}};
 for(const {group,kind,record}of input){
  let selected=record,recordHash=record&&digest(JSON.stringify(record));
  if(record&&bytes(record)>Math.min(4096,Math.floor(limits.maxBytes/3))){
   if(!recordsDirectory)throw Error('Oversized discovery evidence requires a records directory');
   await mkdir(recordsDirectory,{recursive:true});const text=JSON.stringify(record,null,2)+'\n';
   const saved=await immutableFile(join(recordsDirectory,recordHash+'.json'),text,record.reference);
   const readable=await readableFields(record,join(recordsDirectory,recordHash+'-text'),record.reference);
   const {data,message,reflection,skillReads,...metadata}=record;
   selected={...metadata,data:{evidencePointer:{...saved,reference:record.reference,...(readable?{readableText:readable}:{})}},
    artifacts:[...(record.artifacts??[]),{...saved,reference:record.reference,kind:'complete-discovery-record'},
     ...(readable?[{path:readable.path,sha256:readable.sha256,kind:'readable-text-index'},
      {path:readable.directory,kind:'readable-text-directory'}]:[])]};
  }
  const add=current=>{
   let target=current.find(g=>g.id===group.id);
   if(!target){target={...header(group),discoveryContext:{originalOccurrences:group.occurrences?.length??0,
    originalLinkedDiagnostics:group.linkedDiagnostics?.length??0,view:'page-slice-original-references'},occurrences:[],linkedDiagnostics:[]};current.push(target);}
   if(kind)target[kind].push(selected);
   return current;
  };
  let next=add(structuredClone(page.groups));
  const count=next.reduce((n,g)=>n+g.occurrences.length+g.linkedDiagnostics.length,0);
  if(page.groups.length&&(bytes(next)>limits.maxBytes||next.length>limits.maxGroups||count>limits.maxRecords)){
   seal();next=add([]);
  }
  if(bytes(next)>limits.maxBytes)throw Error('Discovery record metadata exceeds page byte limit: '+(record?.reference??group.id));
  page.groups=next;page.bindings.push({groupId:group.id,kind,reference:record?.reference??null,recordIdentity:recordHash});
 }
 seal();
 const totals=new Map(groups.map(group=>[group.id,{occurrences:group.occurrences?.length??0,linkedDiagnostics:group.linkedDiagnostics?.length??0}]));
 const result=pages.map((page,index)=>{
  const id=digest(JSON.stringify({headers:page.groups.map(header),bindings:page.bindings}));
  return {id,groups:page.groups,metadata:{format:1,id,index,totalPages:pages.length,bytes:bytes(page.groups),
   groupIds:page.groups.map(g=>g.id),projects:[...new Set(page.groups.flatMap(g=>[...g.occurrences,...g.linkedDiagnostics].map(r=>r.projectId).filter(Boolean)))],
   records:page.bindings.filter(b=>b.kind),continuation:page.groups.map(g=>({groupId:g.id,total:totals.get(g.id),
    page:{occurrences:g.occurrences.length,linkedDiagnostics:g.linkedDiagnostics.length}}))}};
 });
 const covered=result.flatMap(p=>p.metadata.records);
 const references=new Set(covered.map(row=>JSON.stringify([row.groupId,row.kind,row.reference])));
 if(covered.length!==all.size||references.size!==all.size||[...all].some(ref=>!references.has(ref)))throw Error('Discovery paging lost evidence');
 return result;
}
