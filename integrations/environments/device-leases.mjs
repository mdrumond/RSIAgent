import { mkdir, readFile, writeFile, rm, rename } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, dirname } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';

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
export async function reserveDevice(pool,physicalDevice,directory=join(homedir(),'.local/state/rsi/device-leases')) {
  const key=createHash('sha256').update(JSON.stringify([pool,physicalDevice])).digest('hex');
  await mkdir(directory,{recursive:true,mode:0o700});
  const path=join(directory,key),token=randomUUID(),receipt=join(path,'lease.json');
  try{await mkdir(path,{mode:0o700});}catch(error){if(error.code==='EEXIST')return null;throw error;}
  await writeFile(receipt,JSON.stringify({version:2,token,pool,physicalDevice,handles:[],submissions:[]}),{mode:0o600});
  return openDeviceLease(receipt,token);
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
