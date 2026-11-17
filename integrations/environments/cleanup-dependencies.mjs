/** A conservative inventory: missing dependency paths remain protected. */
import {readFile,readdir,stat} from 'node:fs/promises';
import {resolve,relative,isAbsolute,join,dirname} from 'node:path';
import {DatabaseSync} from 'node:sqlite';
const intersects=(a,b)=>{const within=(root,path)=>{const rel=relative(root,path);return !rel||!isAbsolute(rel)&&rel!=='..'&&!rel.startsWith('../');};return within(a,b)||within(b,a);};
export async function cleanupDependencies({leaseDirectory,checkpoints=[]}){
 const result=[],add=(path,source,kind)=>{if(typeof path==='string'&&isAbsolute(path))result.push({path:resolve(path),source,kind});};
 let entries;try{entries=await readdir(leaseDirectory,{withFileTypes:true});}catch(e){if(e.code==='ENOENT')entries=[];else throw e;}
 for(const entry of entries.filter(e=>e.isDirectory()&&!e.name.startsWith('.'))){
  const receipt=join(leaseDirectory,entry.name,'lease.json');add(dirname(receipt),receipt,'lease');
  let lease;try{lease=JSON.parse(await readFile(receipt,'utf8'));}catch(cause){throw Object.assign(Error('Cleanup dependency inventory cannot read lease: '+receipt,{cause}),{code:'CLEANUP_DEPENDENCY_UNKNOWN'});}
  if(!lease||typeof lease.instancePath!=='string'||!isAbsolute(lease.instancePath))throw Object.assign(Error('Cleanup dependency inventory requires the lease execution path: '+receipt),{code:'CLEANUP_DEPENDENCY_UNKNOWN'});
  add(lease.instancePath,receipt,'execution');add(lease.controlDirectory,receipt,'control-journal');
  for(const submission of lease.submissions??[])add(submission.receiptFile,receipt,'submission');
  for(const path of [lease.cleanup?.directory,lease.cleanup?.pointer])add(path,receipt,'cleanup');
 }
 const queue=join(leaseDirectory,'.cleanup','queue.sqlite');
 try{await stat(queue);const db=new DatabaseSync(queue,{readOnly:true});try{
  if(db.prepare("SELECT 1 FROM sqlite_master WHERE name='jobs'").get())for(const row of db.prepare("SELECT request FROM jobs WHERE status!='closed'").all()){
   const request=JSON.parse(row.request);
   for(const name of ['instancePath','journal','receipt','pointer','activePointer','campaignRoot'])add(request[name],queue,'cleanup');
  }
 }finally{db.close();}}catch(e){if(e.code!=='ENOENT')throw e;}
 for(const checkpoint of checkpoints){add(checkpoint.directory,'checkpoint','checkpoint');for(const path of checkpoint.paths??[])add(path,'checkpoint','artifact');}
 return [...new Map(result.map(row=>[JSON.stringify(row),row])).values()];
}
export function cleanupCandidates(paths,dependencies){
 return paths.map(path=>({path:resolve(path),protectedBy:dependencies.filter(row=>intersects(resolve(path),row.path))}));
}
