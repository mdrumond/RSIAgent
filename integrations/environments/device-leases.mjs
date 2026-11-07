import { mkdir, readFile, writeFile, rm, rename } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, dirname } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import {readFileSync,writeFileSync,renameSync,mkdirSync} from 'node:fs';
import {DatabaseSync} from 'node:sqlite';
import {processIdentity,ownerStopped} from '../../controller/lifecycle.mjs';

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
    async update(data){Object.assign(record,data);await save();},
    async release(){
      if(JSON.parse(await readFile(receipt,'utf8')).token!==token)throw new Error('Lease ownership changed');
      await rm(dirname(receipt),{recursive:true});
    }};
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
    writeFileSync(receipt,JSON.stringify({version:2,token,pool,physicalDevice,owner,instancePath:metadata.instancePath,handles:[],submissions:[]}),{mode:0o600});
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

// Closure and fencing are separate proofs. Callers may supply an independently
// retained owner-closure proof for legacy records without owner metadata.
export async function reconcileDeviceLease(receipt,{stopped=ownerStopped,observe,reconcileHost}={}) {
  let before;
  try{before=leaseLock(dirname(dirname(receipt)),()=>JSON.parse(readFileSync(receipt,'utf8')));}
  catch(error){if(error.code==='ENOENT')return {closed:false,reason:'missing-receipt',receipt};throw error;}
  const blocked=(reason,extra={})=>({closed:false,reason,receipt,token:before.token,...extra});
  if(before.version!==2)return blocked('unsupported-receipt');
  if(!await stopped(before.owner,before))return blocked(before.owner?'owner-active':'owner-unknown');
  if(before.reconciliation&&!await ownerStopped(before.reconciliation.owner))return blocked('reconciliation-active');
  const claim={id:randomUUID(),owner:await processIdentity()};
  if(!changeLease(receipt,record=>{
    if(JSON.stringify(record)!==JSON.stringify(before))return false;
    record.reconciliation=claim;return true;
  }))return blocked('ownership-changed');
  try {
    const lease=await openDeviceLease(receipt,before.token),handles=new Set(lease.record.handles);
    for(const row of lease.record.submissions){
      if(row.closed===true)continue;
      let handle=row.handle;
      if(!handle&&row.receiptFile)try{handle=JSON.parse(await readFile(row.receiptFile,'utf8')).handle;}catch{}
      if(!handle)return blocked('submission-unknown');
      handles.add(handle);
    }
    for(const handle of handles){
      let result;try{result=await observe(handle);}catch{return blocked('execution-unresolved',{handle});}
      if(!['stopped','completed','failed','cancelled'].includes(result?.state))return blocked('execution-unresolved',{handle});
    }
    if(!await reconcileHost(lease))return blocked('host-fence-unresolved');
    await lease.release();return {closed:true,receipt,token:before.token};
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
def report(decision):
 path=os.environ['RSI_LEASE_RECEIPT']
 with open(path+'.tmp','w') as output: json.dump({'token':c['lease']['token'],'decision':decision},output); output.flush(); os.fsync(output.fileno())
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
raise SystemExit(code if code>=0 else 128-code)
`;
