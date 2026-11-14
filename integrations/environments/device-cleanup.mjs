/** Durable local ownership of token-scoped cleanup; all remote I/O stays in Ascend/CPL. */
import {mkdir,readFile,appendFile,open,writeFile,unlink} from 'node:fs/promises';
import {join,dirname} from 'node:path';
import {homedir} from 'node:os';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {randomUUID} from 'node:crypto';
import {DatabaseSync} from 'node:sqlite';
import {setTimeout as delay} from 'node:timers/promises';
import {closedDeviceLease,claimCleanupLease,retainCleanupLease} from './device-leases.mjs';
import {processIdentity,ownerStopped} from '../../controller/lifecycle.mjs';
const entry=fileURLToPath(new URL('./device-cleanup-worker.mjs',import.meta.url));
export const cleanupDirectory=context=>join(context.leaseDirectory??join(homedir(),'.local/state/rsi/device-leases'),'.cleanup');
async function database(directory){await mkdir(directory,{recursive:true});const db=new DatabaseSync(join(directory,'queue.sqlite'));db.exec(`PRAGMA busy_timeout=5000;
 CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,request TEXT NOT NULL,status TEXT NOT NULL,owner TEXT,attempts INTEGER NOT NULL DEFAULT 0,error TEXT,next_at INTEGER NOT NULL DEFAULT 0);
 CREATE TABLE IF NOT EXISTS worker(singleton INTEGER PRIMARY KEY CHECK(singleton=1),ticket TEXT NOT NULL,owner TEXT NOT NULL);`);return db;}
const decode=row=>row?{...row,request:JSON.parse(row.request),owner:row.owner?JSON.parse(row.owner):null}:null;
export async function cleanupStatus(directory){const db=await database(directory);try{return db.prepare('SELECT * FROM jobs ORDER BY rowid').all().map(decode);}finally{db.close();}}
export async function campaignCleanupPending(root){
 let service;try{service=JSON.parse(await readFile(join(root,'device-cleanup-service.json'),'utf8'));}catch(error){if(error.code==='ENOENT')return false;throw error;}
 // An expired fence is durable state, not a live writer. It must remain safe
 // to checkpoint/upgrade after ownership stops without declaring that fence free.
 const rows=(await cleanupStatus(service.directory)).filter(row=>row.request.campaignRoot===root&&['queued','running','handoff'].includes(row.status));
 if(!rows.length)return false;
 for(const row of rows)if(row.status==='handoff'&&!await ownerStopped(row.request.handoffOwner))return true;
 const db=await database(service.directory);try{const worker=db.prepare('SELECT owner FROM worker WHERE singleton=1').get();return Boolean(worker&&!await ownerStopped(JSON.parse(worker.owner)));}finally{db.close();}
}
export async function queueDeviceCleanup(directory,request,dependencies={}){
 const db=await database(directory);let retained;try{
  db.prepare('INSERT OR IGNORE INTO jobs(id,request,status) VALUES (?,?,?)').run(request.id,JSON.stringify(request),request.handoffOwner?'handoff':'queued');
  retained=decode(db.prepare('SELECT * FROM jobs WHERE id=?').get(request.id));
  if(retained.request.receipt!==request.receipt||retained.request.token!==request.token||retained.request.pointer!==request.pointer)throw Error('Cleanup identity changed');
 }finally{db.close();}
 let worker,launchError;if(!request.handoffOwner)try{worker=await resumeDeviceCleanup(directory,dependencies);}catch(error){launchError=error.message;}
 if(request.campaignRoot)await publishCampaignOwner(request.campaignRoot,{directory,owner:worker?.owner??null});
 return {id:request.id,directory,deadline:retained.request.deadline,owner:worker?.owner??null,...(launchError?{launchError}:{})};
}
export async function activateDeviceCleanup(directory,id,dependencies={}){
 const db=await database(directory);try{db.prepare("UPDATE jobs SET status='queued' WHERE id=? AND status='handoff'").run(id);}finally{db.close();}
 return resumeDeviceCleanup(directory,dependencies);
}
async function recoverHandoffs(db,directory){
 const owner=JSON.stringify(await processIdentity());
 for(const row of db.prepare("SELECT * FROM jobs WHERE status='handoff'").all().map(decode)){
  if(row.owner&&!await ownerStopped(row.owner))continue;
  if(!db.prepare("UPDATE jobs SET owner=? WHERE id=? AND status='handoff' AND owner IS ?").run(owner,row.id,row.owner?JSON.stringify(row.owner):null).changes)continue;
  try{
   let pointer;try{pointer=JSON.parse(await readFile(row.request.activePointer,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
   if(pointer?.token===row.request.token){
    if(!await ownerStopped(row.request.handoffOwner))continue;
    if(pointer.receipt!==row.request.receipt)throw Error('Cleanup handoff receipt changed');
    if(!await closedDeviceLease(row.request.receipt,row.request.token)&&!await retainCleanupLease(row.request.receipt,row.request.token,
      {id:row.id,directory,deadline:row.request.deadline,pointer:row.request.pointer}))continue;
    let current;try{current=JSON.parse(await readFile(row.request.activePointer,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
    if(current?.token===row.request.token)await unlink(row.request.activePointer);
   }
   db.prepare("UPDATE jobs SET status='queued',owner=NULL WHERE id=? AND status='handoff' AND owner=?").run(row.id,owner);
  }finally{db.prepare("UPDATE jobs SET owner=NULL WHERE id=? AND status='handoff' AND owner=?").run(row.id,owner);}
 }
}
export async function resumeDeviceCleanup(directory,{launch=launchWorker}={}){
 const db=await database(directory);let prior;
 try{
  await recoverHandoffs(db,directory);
  if(!db.prepare("SELECT 1 FROM jobs WHERE status IN ('queued','running')").get())return null;
  prior=db.prepare('SELECT * FROM worker WHERE singleton=1').get();
  if(prior&&!await ownerStopped(JSON.parse(prior.owner)))return await publishOwner(db,directory,{ticket:prior.ticket,owner:JSON.parse(prior.owner)});
  const owner=await processIdentity(),ticket=randomUUID();db.exec('BEGIN IMMEDIATE');
  const current=db.prepare('SELECT * FROM worker WHERE singleton=1').get();
  if(JSON.stringify(current)!==JSON.stringify(prior)){db.exec('ROLLBACK');return current?{ticket:current.ticket,owner:JSON.parse(current.owner)}:null;}
  db.prepare('INSERT OR REPLACE INTO worker VALUES (1,?,?)').run(ticket,JSON.stringify(owner));db.exec('COMMIT');
  try{
   const child=await launch(directory,ticket);
   db.prepare('UPDATE worker SET owner=? WHERE ticket=?').run(JSON.stringify(child),ticket);
   return await publishOwner(db,directory,{ticket,owner:child});
  }catch(error){db.prepare('DELETE FROM worker WHERE ticket=?').run(ticket);throw error;}
 }finally{db.close();}
}
async function publishCampaignOwner(root,service){
 try{await writeFile(join(root,'device-cleanup-service.json'),JSON.stringify(service),{mode:0o600});}
 catch(error){if(error.code!=='ENOENT')throw error;} // Retired campaigns retain fences without poisoning the shared queue.
}
async function publishOwner(db,directory,worker){
 for(const root of new Set(db.prepare("SELECT request FROM jobs WHERE status!='closed'").all().map(row=>JSON.parse(row.request).campaignRoot).filter(Boolean)))
  await publishCampaignOwner(root,{directory,owner:worker.owner});
 return worker;
}
async function launchWorker(directory,ticket){
 const log=await open(join(directory,'worker.log'),'a',0o600);let child;
 try{
  child=spawn(process.execPath,[entry,directory,ticket],{detached:true,stdio:['ignore',log.fd,log.fd],env:process.env});
  await new Promise((resolve,reject)=>{child.once('spawn',resolve);child.once('error',reject);});
  const owner=await processIdentity(child.pid);child.unref();return owner;
 }finally{await log.close();}
}
async function runJob(request,signal){
 if(await closedDeviceLease(request.receipt,request.token))return true;
 if(!await claimCleanupLease(request.receipt,request.token,await processIdentity()))return false;
 const {createEnvironment}=await import('./ascend.mjs');
 const environment=await createEnvironment({...request.options,asyncCleanup:false,probeObservationSeconds:Math.max(.001,(request.deadline-Date.now())/1000)},
  {instancePath:request.instancePath,leaseDirectory:request.leaseDirectory,cleanupLeasePath:request.pointer,signal,stopSignal:signal,
   exclude:async(_label,action)=>action(),event:async(type,data)=>appendFile(request.journal,JSON.stringify({at:new Date().toISOString(),type,data})+'\n')});
 return environment.reconcileLease();
}
export async function drainDeviceCleanup(directory,ticket,{run=runJob,parallelism=2,now=Date.now,beforeIdleRetirement=async()=>{}}={}){
 if(!Number.isInteger(parallelism)||parallelism<1||parallelism>2)throw Error('Cleanup concurrency must be one or two');
 const db=await database(directory),owner=await processIdentity(),active=new Map();
 try{
  if(db.prepare('SELECT ticket FROM worker WHERE singleton=1').get()?.ticket!==ticket)return;
  db.prepare('UPDATE worker SET owner=? WHERE ticket=?').run(JSON.stringify(owner),ticket);
  for(const row of db.prepare("SELECT * FROM jobs WHERE status='running'").all())if(await ownerStopped(JSON.parse(row.owner)))
   db.prepare("UPDATE jobs SET status='queued',owner=NULL WHERE id=? AND owner=?").run(row.id,row.owner);
  for(;;){
   const rows=db.prepare("SELECT * FROM jobs WHERE status='queued' ORDER BY rowid").all().map(decode);
   for(const row of rows){
    if(row.request.deadline<=now()){db.prepare("UPDATE jobs SET status='expired',error='Original cleanup elapsed allocation exhausted' WHERE id=? AND status='queued'").run(row.id);continue;}
    if(active.size>=parallelism||row.next_at>now())continue;
    if(!db.prepare("UPDATE jobs SET status='running',owner=?,attempts=attempts+1 WHERE id=? AND status='queued'").run(JSON.stringify(owner),row.id).changes)continue;
    const signal=AbortSignal.timeout(Math.max(1,Math.ceil(row.request.deadline-now())));
    const task=(async()=>{
     let closed=false,error;
     try{closed=await run(row.request,signal);}catch(failure){error=failure.message;}
     const status=closed?'closed':row.request.deadline<=now()?'expired':'queued';
     db.prepare('UPDATE jobs SET status=?,owner=NULL,error=?,next_at=? WHERE id=? AND owner=?').run(status,error??(closed?null:'Cleanup remains fenced'),now()+Math.min(5000,1000*2**Math.min(row.attempts,3)),row.id,JSON.stringify(owner));
    })().finally(()=>active.delete(row.id));active.set(row.id,task);
   }
   const pending=db.prepare("SELECT MIN(next_at) AS next FROM jobs WHERE status='queued'").get().next;
   if(!active.size&&pending===null){
    await beforeIdleRetirement();
    // Enqueue and retirement serialize on the same SQLite write lock. A producer
    // either sees this live claim and its row is drained, or sees no claim and
    // starts a successor. The final old-ticket delete cannot remove that owner.
    db.exec('BEGIN IMMEDIATE');
    try{
     const pending=db.prepare("SELECT 1 FROM jobs WHERE status IN ('queued','running')").get();
     if(!pending)db.prepare('DELETE FROM worker WHERE ticket=?').run(ticket);
     db.exec('COMMIT');if(!pending)break;
    }catch(error){db.exec('ROLLBACK');throw error;}
    continue;
   }
   const waiting=new AbortController();
   try{await Promise.race([...active.values(),delay(Math.max(10,Math.min(1000,pending===null?1000:pending-now())),undefined,{signal:waiting.signal})]);}
   finally{waiting.abort();}
  }
 }finally{await Promise.allSettled(active.values());db.prepare('DELETE FROM worker WHERE ticket=?').run(ticket);db.close();}
}
