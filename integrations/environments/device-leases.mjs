import { mkdir, readFile, writeFile, rm, rename } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, dirname, relative } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import {readFileSync,writeFileSync,renameSync,mkdirSync} from 'node:fs';
import {DatabaseSync} from 'node:sqlite';
import {processIdentity,ownerStopped} from '../../controller/lifecycle.mjs';

// Output downloads cannot launch device work. Their unknown transfer identity
// still belongs to the original operation, but need not hold a physical device.
export const artifactDownload=row=>row.kind==='download'&&row.accountingExcluded===true;
const unresolvedArtifacts=record=>record.submissions.filter(row=>artifactDownload(row)&&row.closed!==true&&!row.handle);

export async function openDeviceLease(receipt, token) {
  const record=JSON.parse(await readFile(receipt,'utf8'));
  if(record.token!==token) throw new Error('Lease ownership changed');
  const save=async()=>{
    if(JSON.parse(await readFile(receipt,'utf8')).token!==token) throw new Error('Lease ownership changed');
    await writeFile(receipt+'.tmp',JSON.stringify(record),{mode:0o600});
    await rename(receipt+'.tmp',receipt);
  };
  return {token,receipt,record,
    async retain(handle){if(handle&&!record.handles.includes(handle)){record.handles.push(handle);await save();}},
    async event(type,data){
      if(type==='execution-submitting') record.submissions.push({...data});
      if(['execution-submitted','execution-not-admitted'].includes(type)){
        const entry=record.submissions.find(row=>row.submission===data.submission);
        if(entry)Object.assign(entry,data);
      }
      if(type==='execution-handle'&&data.handle&&!record.handles.includes(data.handle))record.handles.push(data.handle);
      await save();
    },
    async confirmSubmissionClosed(proof){
      const row=record.submissions.find(item=>item.submission===proof.submission);
      if(!row||row.kind!=='run'||row.accountingExcluded===true||row.handle||row.receiptFile!==proof.receiptFile||
          proof.closed!==true||!proof.timingKey||proof.admissionToken!==token||proof.admission?.token!==token||
          proof.admission.decision!=='admitted'||proof.admission.released!==true||proof.nativeIdentityMissing!==true)return false;
      Object.assign(row,{closed:true,wrapperClosure:proof});await save();return true;
    },
    async update(data){Object.assign(record,data);await save();},
    async release(){
      if(JSON.parse(await readFile(receipt,'utf8')).token!==token)throw new Error('Lease ownership changed');
      await rm(dirname(receipt),{recursive:true});
    }};
}
// Positive closure is token-scoped and outlives the active device directory.
// Retain native journals so the original checkpoint can finish observing them.
const closedPath=(receipt,token)=>join(dirname(dirname(receipt)),'.closed',
  createHash('sha256').update(JSON.stringify([receipt,token])).digest('hex'));
export async function closedDeviceLease(receipt,token) {
  const path=closedPath(receipt,token);
  try {
    const proof=JSON.parse(await readFile(join(path,'closure.json'),'utf8'));
    return proof.closed===true&&proof.receipt===receipt&&proof.token===token?{...proof,path}:null;
  }catch(error){if(error.code==='ENOENT')return null;throw error;}
}
export async function closeDeviceLease(lease) {
  const path=closedPath(lease.receipt,lease.token);
  await mkdir(dirname(path),{recursive:true,mode:0o700});
  return leaseLock(dirname(dirname(lease.receipt)),()=>{
    const record=JSON.parse(readFileSync(lease.receipt,'utf8'));
    if(record.token!==lease.token)throw new Error('Lease ownership changed');
    const artifacts=unresolvedArtifacts(record);
    const proof={closed:true,receipt:lease.receipt,token:lease.token,pool:record.pool,
      physicalDevice:record.physicalDevice,handles:record.handles,controlDirectory:record.controlDirectory,
      ...(artifacts.length?{unresolvedArtifacts:artifacts}: {})};
    writeFileSync(join(dirname(lease.receipt),'closure.json'),JSON.stringify(proof),{mode:0o600});
    renameSync(dirname(lease.receipt),path);return proof;
  });
}

// Explicit permanent abandonment, not transfer closure. Legacy execute uploads
// first stage into an exec-*/incoming directory before any sync or run dispatch.
export async function retirePrePayloadLease({receipt,token,instancePath,fence}, {observe,stopped=ownerStopped}={}) {
 const blocked=reason=>({closed:false,reason});
 const old=await closedDeviceLease(receipt,token);if(old)return old.retiredPrePayload&&old.instancePath===instancePath&&old.fence===fence?old:blocked('different-closure');
 let bytes;try{bytes=await readFile(receipt,'utf8');}catch(error){if(error.code==='ENOENT')return blocked('missing-receipt');throw error;}
 const record=JSON.parse(bytes),owner=record.owner;
 if(record.version!==2||record.token!==token||record.instancePath!==instancePath)return blocked('ownership-mismatch');
 if(!owner?.pid||!owner.start||!owner.boot||!await stopped(owner))return blocked('owner-not-stopped');
 if(record.reconciliation&&!await ownerStopped(record.reconciliation.owner))return blocked('reconciliation-active');
 if(record.payloadHandle||record.controlHandle||record.controlStarted||record.cleanup)return blocked('payload-intent');
 const rows=record.submissions;if(rows?.length!==1||rows.some(row=>row.kind!=='upload'||row.accountingExcluded!==true||row.controlPayload||row.wrapperClosure))return blocked('non-upload-intent');
 const journalPath=join(instancePath,'environment-events.jsonl'),journal=await readFile(journalPath,'utf8');
 if(!journal.endsWith('\n'))return blocked('incomplete-journal');
 const events=journal.trim().split('\n').filter(Boolean).map(line=>JSON.parse(line)),acquired=events.findIndex(e=>e.type==='device-lease-acquired'&&e.data.receipt===receipt&&e.data.token===token);
 if(acquired<0)return blocked('missing-admission-provenance');
 if(events.some(e=>e.type==='execution-starting'))return blocked('unclassified-intent');
 const after=events.slice(acquired+1),intents=after.filter(e=>['execution-submitting','execution-starting'].includes(e.type));
 if(intents.length!==rows.length||intents.some(e=>e.type!=='execution-submitting'||!rows.some(row=>row.submission===e.data.submission&&row.kind===e.data.kind&&row.receiptFile===e.data.receiptFile)))return blocked('unclassified-intent');
 const handles=new Set();
 for(const row of rows){
  const name=relative(join(instancePath,'.executions'),row.receiptFile??'');
  if(!/^exec-[A-Za-z0-9]+\/transfer\.json$/.test(name))return blocked('noncanonical-staging');
  // A later synchronization manifest or command invalidates the first-upload proof.
  for(const file of ['execution.json','sync.json'])try{await readFile(join(dirname(row.receiptFile),file));return blocked('later-stage-artifact');}catch(error){if(error.code!=='ENOENT')throw error;}
  let retained;try{retained=JSON.parse(await readFile(row.receiptFile,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
  const handle=row.handle??retained?.handle;
  if(handle){if(!/^remote:[^:]+:transfer:[^:]+$/.test(handle))return blocked('non-transfer-handle');handles.add(handle);}
 }
 if(after.some(e=>e.type==='execution-handle'&&!handles.has(e.data.handle)))return blocked('unclassified-handle');
 if((record.handles??[]).some(handle=>!handles.has(handle)))return blocked('unclassified-handle');
 // Earlier health/setup jobs must already have positive terminal journal proof.
 for(const e of events.slice(0,acquired))if(e.type==='execution-submitting'&&e.data.kind!=='upload'){
  const response=events.find(x=>x.type==='execution-submitted'&&x.data.submission===e.data.submission);
  if(!response?.data.handle||!events.some(x=>x.type==='execution-state'&&x.data.handle===response.data.handle&&x.data.state==='stopped'))return blocked('earlier-execution-unresolved');
 }
 const fenceBytes=await readFile(fence,'utf8'),guard=JSON.parse(fenceBytes);
 if(guard.permanent!==true||guard.instancePath!==instancePath||guard.receipt!==receipt||guard.token!==token)return blocked('missing-permanent-fence');
 const observations=[];for(const handle of handles){let state;try{state=await observe(handle);}catch{state={state:'unknown'};}observations.push({handle,state});}
 const hash=value=>createHash('sha256').update(value).digest('hex'),path=closedPath(receipt,token);
 const proof={closed:true,retiredPrePayload:true,payloadStarted:false,receipt,token,pool:record.pool,physicalDevice:record.physicalDevice,handles:record.handles,controlDirectory:record.controlDirectory,
  unresolvedArtifacts:rows.map(row=>({...row,fenced:true})),observations,owner,fence,fenceSha256:hash(fenceBytes),leaseSha256:hash(bytes),journalSha256:hash(journal),sourceSha256:hash(await readFile(new URL(import.meta.url))),executorSourceSha256:hash(await readFile(new URL('./index.mjs',import.meta.url))),protocol:'legacy-first-source-upload-v1',instancePath};
 await mkdir(dirname(path),{recursive:true,mode:0o700});
 return leaseLock(dirname(dirname(receipt)),()=>{
  if(readFileSync(receipt,'utf8')!==bytes||readFileSync(journalPath,'utf8')!==journal||readFileSync(fence,'utf8')!==fenceBytes)return blocked('ownership-changed');
  writeFileSync(join(dirname(receipt),'closure.json'),JSON.stringify(proof),{mode:0o600});renameSync(dirname(receipt),path);return {...proof,path};
 });
}

// Atomic mkdir spans instances/runtimes; owner death/expiry never proves closure.
export function deviceLeaseReceipt(pool,physicalDevice,directory=join(homedir(),'.local/state/rsi/device-leases')) {
  return join(directory,createHash('sha256').update(JSON.stringify([pool,physicalDevice])).digest('hex'),'lease.json');
}
export async function reserveDevice(pool,physicalDevice,directory=join(homedir(),'.local/state/rsi/device-leases'),metadata={}) {
  const key=createHash('sha256').update(JSON.stringify([pool,physicalDevice])).digest('hex');
  await mkdir(directory,{recursive:true,mode:0o700});
  const path=join(directory,key),token=randomUUID(),receipt=join(path,'lease.json'),owner=await processIdentity();
  const created=leaseLock(directory,()=>{
    try{mkdirSync(path,{mode:0o700});}catch(error){if(error.code==='EEXIST')return false;throw error;}
    writeFileSync(receipt,JSON.stringify({version:2,token,pool,physicalDevice,owner,instancePath:metadata.instancePath,controlDirectory:join(directory,'.journals',token,'control'),handles:[],submissions:[]}),{mode:0o600});
    return true;
  });
  return created?openDeviceLease(receipt,token):null;
}
// Serialize only local claim metadata; never hold a SQLite lock across remote I/O.
function leaseLock(directory,action) {
  const db=new DatabaseSync(join(directory,'.reconciliation.sqlite'));
  try {db.exec('PRAGMA busy_timeout=5000; BEGIN IMMEDIATE');const result=action();db.exec('COMMIT');return result;}
  finally{db.close();}
}
function changeLease(receipt,change) {
  return leaseLock(dirname(dirname(receipt)),()=>{
    const record=JSON.parse(readFileSync(receipt,'utf8'));
    if(!change(record))return false;
    const temporary=receipt+'.'+randomUUID();
    writeFileSync(temporary,JSON.stringify(record),{mode:0o600});renameSync(temporary,receipt);return true;
  });
}

export async function retainCleanupLease(receipt,token,cleanup){
 let before;try{before=JSON.parse(await readFile(receipt,'utf8'));}catch(error){if(error.code==='ENOENT')return false;throw error;}
 if(before.token!==token||before.reconciliation&&!await ownerStopped(before.reconciliation.owner))return false;
 return changeLease(receipt,record=>{
  if(JSON.stringify(record)!==JSON.stringify(before))return false;
  record.cleanup=cleanup;return true;
 });
}
// A durable cleanup job may take custody of its own known-closed payload lease.
// The atomic comparison excludes a concurrent public recovery claim.
export async function claimCleanupLease(receipt,token,owner){
 let before;try{before=JSON.parse(await readFile(receipt,'utf8'));}catch(error){if(error.code==='ENOENT')return false;throw error;}
 if(before.token!==token||before.cleanup?.id!==token)return false;
 if(before.reconciliation&&!await ownerStopped(before.reconciliation.owner))return false;
 return changeLease(receipt,record=>{
  if(JSON.stringify(record)!==JSON.stringify(before))return false;
  record.payloadOwner??=record.owner;record.owner=owner;return true;
 });
}
// Closure and fencing are separate proofs. Callers may supply an independently
// retained owner-closure proof for legacy records without owner metadata.
export async function reconcileDeviceLease(receipt,{stopped=ownerStopped,observe,reconcileHost}={}) {
  let before;
  try{before=leaseLock(dirname(dirname(receipt)),()=>JSON.parse(readFileSync(receipt,'utf8')));}
  catch(error){if(error.code==='ENOENT')return {closed:false,reason:'missing-receipt',receipt};throw error;}
  const blocked=(reason,extra={})=>({closed:false,reason,receipt,token:before.token,...extra});
  const closed=proof=>({closed:true,receipt,token:before.token,
    ...(proof.unresolvedArtifacts?{unresolvedArtifacts:proof.unresolvedArtifacts}: {})});
  if(before.version!==2)return blocked('unsupported-receipt');
  if(!await stopped(before.owner,before))return blocked(before.owner?'owner-active':'owner-unknown');
  if(before.reconciliation&&!await ownerStopped(before.reconciliation.owner))return blocked('reconciliation-active');
  const claim={id:randomUUID(),owner:await processIdentity()};
  if(!changeLease(receipt,record=>{
    if(JSON.stringify(record)!==JSON.stringify(before))return false;
    record.reconciliation=claim;return true;
  }))return blocked('ownership-changed');
  try {
    const lease=await openDeviceLease(receipt,before.token),handles=new Set([...lease.record.handles,lease.record.controlHandle,lease.record.payloadHandle].filter(Boolean));
    for(const row of lease.record.submissions){
      if(row.closed===true)continue;
      let handle=row.handle;
      if(!handle&&row.receiptFile)try{handle=JSON.parse(await readFile(row.receiptFile,'utf8')).handle;}catch{}
      if(!handle){
        if(artifactDownload(row))continue;
        return blocked('submission-unknown',{submission:row.submission,receiptFile:row.receiptFile??null});
      }
      handles.add(handle);
    }
    for(const handle of handles){
      let result;try{result=await observe(handle);}catch{return blocked('execution-unresolved',{handle});}
      if(!['stopped','completed','failed','cancelled'].includes(result?.state))return blocked('execution-unresolved',{handle});
    }
    if(!await reconcileHost(lease)){
      const proof=await closedDeviceLease(receipt,before.token);
      if(proof)return closed(proof);
      return blocked('host-fence-unresolved');
    }
    const proof=await closeDeviceLease(lease);
    return closed(proof);
  }catch(error){
    // Another checkpoint can close this exact token while an observer is in I/O.
    // Its positive closure supersedes missing/moved journals, never another token.
    const proof=await closedDeviceLease(receipt,before.token);
    if(proof)return closed(proof);
    throw error;
  }finally{
    try{changeLease(receipt,record=>{
      if(record.token!==before.token||record.reconciliation?.id!==claim.id)return false;
      delete record.reconciliation;return true;
    });}catch(error){if(error.code!=='ENOENT')throw error;}
  }
}

// Used only inside generic supervised execution (CPL or native local Pi).
const hostCommon=`import fcntl,signal,subprocess,ctypes
root=os.path.expanduser('~/.local/state/rsi/device-leases-host')
os.makedirs(root,mode=0o700,exist_ok=True)
key=hashlib.sha256(json.dumps([c['lease']['pool'],c['devices'][0]],separators=(',',':')).encode()).hexdigest()
lock=open(root+'/'+key+'.lock','a'); fence=root+'/'+key+'.held'
def report(decision,**facts):
 path=os.environ['RSI_LEASE_RECEIPT']
 with open(path+'.tmp','w') as output: json.dump({'token':c['lease']['token'],'decision':decision,**facts},output); output.flush(); os.fsync(output.fileno())
 os.replace(path+'.tmp',path)
try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError: report('contended'); raise SystemExit(126)
`;
export const hostReconcile=hostCommon+`try:
 with open(fence+'/lease.json') as receipt: owner=json.load(receipt)
except FileNotFoundError: owner=None
if owner is not None:
 if owner['token']!=c['lease']['token']: report('ownership-mismatch'); raise SystemExit(125)
 os.unlink(fence+'/lease.json'); os.rmdir(fence)
report('reconciled')
`;
export const hostLease=hostCommon+`try: os.mkdir(fence,0o700)
except FileExistsError: report('contended'); raise SystemExit(126)
with open(fence+'/lease.json','w') as receipt: json.dump(c['lease'],receipt)
rows=[ns['local_probe'](physical_device=c['devices'][0],configuration=c['configuration'],remaining=lambda:max(0.,deadline-time.monotonic()))]
if len(rows)!=1 or rows[0]['admitted'] is not True or rows[0]['measurements']['idle'] is not True:
 os.unlink(fence+'/lease.json'); os.rmdir(fence); report('health-refused'); raise SystemExit(125)
# Reparent background grandchildren here so ordinary exit can prove cleanup.
if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0: raise OSError('subreaper unavailable')
report('admitted')
child=subprocess.Popen(['bash','-c',c['command']],pass_fds=(lock.fileno(),))
code=child.wait()
# Same process group as the supervisor: timeout/STOP kills wrapper AND payload.
# On ordinary exit, reap even shell background descendants before releasing.
while True:
 try:
  pid,status=os.waitpid(-1,os.WNOHANG)
  if pid: continue
 except ChildProcessError: break
 for name in os.listdir('/proc'):
  if not name.isdigit(): continue
  try:
   fields=open('/proc/'+name+'/stat').read().rsplit(')',1)[1].split()
   if int(fields[1])==os.getpid(): os.kill(int(name),signal.SIGKILL)
  except (FileNotFoundError,ProcessLookupError): pass
 time.sleep(.01)
os.unlink(fence+'/lease.json'); os.rmdir(fence)
report('admitted',released=True)
raise SystemExit(code if code>=0 else 128-code)
`;
