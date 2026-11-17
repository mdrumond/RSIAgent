/** Controller-owned job metadata; the attempt directory is only its file mirror. */
import {mkdir,readFile,writeFile,rename,rm} from 'node:fs/promises';
import {join,dirname} from 'node:path';
import {randomUUID} from 'node:crypto';

export class AttemptJobUnavailable extends Error {
  constructor(attempt,reason,cause) {
    super(`Attempt ${attempt} job metadata unavailable (${reason}); preserve its custody and restore the original metadata`,{cause});
    this.code='ATTEMPT_JOB_UNAVAILABLE';this.attempt=attempt;this.reason=reason;
  }
}
const exists=state=>Boolean(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='attempt_job_manifests'").get());
const latest= (state,id)=>state.queryEvents(id,{types:['attempt-metadata-unavailable','attempt-metadata-restored'],order:'desc',limit:1})[0];
async function publish(path,content) {
  await mkdir(dirname(path),{recursive:true});
  const temporary=path+'.'+randomUUID()+'.tmp';
  await writeFile(temporary,content,{mode:0o600,flag:'wx'});
  try{await rename(temporary,path);}finally{await rm(temporary,{force:true});}
}
export function publishAttemptJob(state,id,value) {
  if(!state.attempt(id))throw Error('Unknown attempt for job metadata');
  const content=JSON.stringify(value);
  state.db.exec('CREATE TABLE IF NOT EXISTS attempt_job_manifests(attempt TEXT PRIMARY KEY,data TEXT NOT NULL,event_sequence INTEGER NOT NULL)');
  const sequence=state.db.prepare('SELECT COALESCE(MAX(sequence),0) n FROM events').get().n;
  // Publish durable trusted bytes before the replaceable file, including when
  // interruption or an I/O failure prevents its atomic rename.
  state.db.prepare('INSERT OR REPLACE INTO attempt_job_manifests VALUES (?,?,?)').run(id,content,sequence);
}
export async function materializeAttemptJob(root,id,value) {
  await publish(join(root,'attempts',id,'job.json'),JSON.stringify(value));
}
export async function writeAttemptJob(state,root,id,value) {
  publishAttemptJob(state,id,value);
  await materializeAttemptJob(root,id,value);
}
export async function readAttemptJob(state,root,id) {
  const file=join(root,'attempts',id,'job.json');let content,source='file',missing;
  try{content=await readFile(file,'utf8');}catch(error){if(error.code!=='ENOENT')throw new AttemptJobUnavailable(id,'unreadable-file',error);missing=error;}
  const saved=exists(state)?state.db.prepare('SELECT * FROM attempt_job_manifests WHERE attempt=?').get(id):null;
  const boundary=saved&&state.queryEvents(id,{types:['portable-bound'],order:'desc',limit:1})[0];
  const stale=boundary&&boundary.sequence>saved.event_sequence;
  if(saved&&!stale&&content!==saved.data){content=saved.data;source='controller-checkpoint';await publish(file,content);}
  if(content===undefined)throw new AttemptJobUnavailable(id,stale?'checkpoint-precedes-binding':'no-trusted-checkpoint',missing);
  let value;try{value=JSON.parse(content);}catch(error){throw new AttemptJobUnavailable(id,'invalid-metadata',error);}
  if(!value||typeof value!=='object'||Array.isArray(value))throw new AttemptJobUnavailable(id,'invalid-metadata');
  if(value.attempt?.id!==undefined&&value.attempt.id!==id)throw new AttemptJobUnavailable(id,'attempt-identity-mismatch');
  if(source==='controller-checkpoint'||latest(state,id)?.type==='attempt-metadata-unavailable')state.event(id,'attempt-metadata-restored',{source});
  return value;
}
// Missing legacy metadata blocks this attempt, never unrelated scheduling. No
// execution lease, elapsed checkpoint or scientific outcome is changed here.
export async function availableAttemptJob(state,root,id) {
  try{return await readAttemptJob(state,root,id);}
  catch(error){
    if(error.code!=='ATTEMPT_JOB_UNAVAILABLE')throw error;
    const last=latest(state,id);
    if(last?.type!=='attempt-metadata-unavailable'||last.data.reason!==error.reason){
      state.event(id,'attempt-metadata-unavailable',{reason:error.reason,message:error.message});
      state.recordFailure(id,{phase:'controller',kind:'job-metadata-unavailable',message:error.message});
    }
    return null;
  }
}
