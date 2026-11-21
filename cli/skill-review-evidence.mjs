/** Bounded factual packets; paths remain available for deeper native reads. */
import {open} from 'node:fs/promises';
import {createHash} from 'node:crypto';
export async function frozenEvidencePacket(groups,{limit=4096}={}){
 const artifacts=[],byHash=new Map(),observations=[];
 for(const group of groups)for(const observation of [...group.occurrences,...(group.linkedDiagnostics??[])]){
  const refs=[];
  for(const artifact of observation.artifacts??[]){
   let handle;
   try{
    handle=await open(artifact.path,'r');const info=await handle.stat();if(!info.isFile())throw Error('not a regular file');
    const bytes=Buffer.alloc(Math.min(info.size,limit));
    if(info.size>limit){const head=Math.floor(limit/2);await handle.read(bytes,0,head,0);await handle.read(bytes,head,bytes.length-head,info.size-(bytes.length-head));}
    else await handle.read(bytes,0,bytes.length,0);
    // Verify the copied bytes before exposing a supplied identity. A large
    // file can keep its full hash while only excerpts enter the proof packet.
    const hash=createHash('sha256');
    if(artifact.sha256!=null&&info.size>limit){
     for await(const chunk of handle.createReadStream({autoClose:false,start:0}))hash.update(chunk);
    }else hash.update(bytes);
    const sha256=hash.digest('hex');
    if(artifact.sha256!=null&&artifact.sha256!==sha256)throw Object.assign(Error('Artifact content hash mismatch'),{code:'ARTIFACT_HASH_MISMATCH'});
    let entry=byHash.get(sha256);
    if(!entry){entry={id:'artifact:'+sha256,sha256,paths:[],available:true,text:bytes.toString('utf8'),truncated:!!artifact.truncated||info.size>limit,layout:info.size>limit?'head-and-tail':artifact.layout??'complete',segments:(info.size>limit||artifact.layout==='head-and-tail')?[bytes.subarray(0,Math.floor(bytes.length/2)).toString('utf8'),bytes.subarray(Math.floor(bytes.length/2)).toString('utf8')]:[bytes.toString('utf8')],originalBytes:artifact.originalBytes??info.size};byHash.set(sha256,entry);artifacts.push(entry);}
    if(!entry.paths.includes(artifact.path))entry.paths.push(artifact.path);refs.push(entry.id);
   }catch(error){if(!['ENOENT','EACCES','EISDIR','ARTIFACT_HASH_MISMATCH'].includes(error.code)&&error.message!=='not a regular file')throw error;
    const id='unavailable:'+artifact.path;artifacts.push({id,paths:[artifact.path],available:false,reason:error.code??error.message});refs.push(id);
   }finally{await handle?.close();}
  }
  observations.push({reference:observation.reference,type:observation.type,message:observation.message,data:observation.data,artifacts:refs,unavailable:refs.length?undefined:'No frozen artifact supplied'});
 }
 return {observations,artifacts};
}
export function scopedBaseline(entries,proposal){
 const scopes=new Set((proposal?.changes??[]).map(change=>change.scope)),ids=new Set((proposal?.changes??[]).map(change=>change.id));
 return entries.filter(entry=>entry.scope==='global-kdb'||entry.critical||scopes.has(entry.scope)||ids.has(entry.id)||!entry.scope);
}

export function scopedQaGroups(groups,proposal){
 const refs=new Set((proposal.changes??[]).flatMap(change=>change.evidenceRefs??[]));
 return groups.filter(group=>[...group.occurrences,...(group.linkedDiagnostics??[])].some(item=>refs.has(item.reference))).map(group=>({
  ...group,occurrences:group.occurrences.filter(item=>refs.has(item.reference)),linkedDiagnostics:group.linkedDiagnostics??[],
 }));
}
