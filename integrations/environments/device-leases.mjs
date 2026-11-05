import { mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';

// Atomic mkdir is shared across instances/runtimes. A dead controller is never
// evidence of payload closure: only the owner with positive closure releases.
export async function reserveDevice(pool, physicalDevice, directory = join(homedir(), '.local/state/rsi/device-leases')) {
  const key = createHash('sha256').update(JSON.stringify([pool, physicalDevice])).digest('hex');
  await mkdir(directory, { recursive: true, mode: 0o700 });
  const path = join(directory, key), token = randomUUID();
  try { await mkdir(path, { mode: 0o700 }); }
  catch (error) { if (error.code === 'EEXIST') return null; throw error; }
  const record = { token, pool, physicalDevice, createdAt: new Date().toISOString(), handles: [] };
  const receipt = join(path, 'lease.json');
  await writeFile(receipt, JSON.stringify(record), { mode: 0o600 });
  return {
    token, receipt,
    async retain(handle) {
      if (handle && !record.handles.includes(handle)) {
        record.handles.push(handle);
        await writeFile(receipt, JSON.stringify(record), { mode: 0o600 });
      }
    },
    async release() {
      if (JSON.parse(await readFile(receipt, 'utf8')).token !== token) throw new Error('Lease ownership changed');
      await rm(path, { recursive: true });
    },
  };
}

// Runs INSIDE the admitted host command through CPL/Pi, never a second transport.
// flock spans descendants; the directory fences a killed wrapper after flock
// disappears. No expiry, PID-based reclaim, or automatic removal of old fences.
export const hostLease = `import fcntl,signal,subprocess
root=os.path.expanduser('~/.local/state/rsi/device-leases-host')
os.makedirs(root,mode=0o700,exist_ok=True)
key=hashlib.sha256(json.dumps([c['lease']['pool'],c['devices'][0]],separators=(',',':')).encode()).hexdigest()
lock=open(root+'/'+key+'.lock','a')
try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError: raise SystemExit(126)
fence=root+'/'+key+'.held'
try: os.mkdir(fence,0o700)
except FileExistsError: raise SystemExit(126)
with open(fence+'/lease.json','w') as receipt: json.dump(c['lease'],receipt)
try:
 rows=[ns['local_probe'](physical_device=c['devices'][0],configuration=c['configuration'],remaining=lambda:max(0.,deadline-time.monotonic()))]
 if len(rows)!=1 or rows[0]['admitted'] is not True or rows[0]['measurements']['idle'] is not True: raise SystemExit(125)
 child=subprocess.Popen(['bash','-c',c['command']],start_new_session=True,pass_fds=(lock.fileno(),))
 def stop(*args):
  try: os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError: pass
 signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
 try: code=child.wait()
 finally: stop(); child.wait()
finally:
 os.unlink(fence+'/lease.json'); os.rmdir(fence)
raise SystemExit(code if code>=0 else 128-code)
`;
