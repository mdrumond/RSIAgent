import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {randomUUID} from 'node:crypto';
import {State} from '../controller/state.mjs';
import {processIdentity,ownerStopped} from '../controller/lifecycle.mjs';
import {readConfig} from './instance.mjs';
import * as managed from './managed-runtime.mjs';

export function requests(state) {
  state.db.exec(`CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,kind TEXT NOT NULL,status TEXT NOT NULL,owner TEXT,result TEXT,error TEXT)`);
  return state.db.prepare('SELECT id,kind,status,owner,result,error FROM requests ORDER BY rowid DESC LIMIT 100').all().map(row=>({
    ...row,owner:row.owner?JSON.parse(row.owner):null,result:row.result?JSON.parse(row.result):null}));
}

/** Read-only observation plus factual terminal repair; no operation is replayed. */
export async function reconcileRequests(root,{state,manager=managed}={}) {
  const own=!state;state??=new State(join(root,'state.sqlite'));
  try {
    for(const request of requests(state)) {
      if(!['queued','running'].includes(request.status)||!request.owner?.unit||!request.owner?.invocation||
          !await ownerStopped(request.owner))continue;
      const handle=await manager.observeManagedMaintenance(request.owner);
      if(await handle.closed())state.db.prepare(`UPDATE requests SET status='failed',error=?
        WHERE id=? AND status IN ('queued','running') AND owner=?`).run(
        'Managed request owner exited before recording completion',request.id,JSON.stringify(request.owner));
    }
    return requests(state);
  } finally {if(own)state.close();}
}

export async function startBackground(root,command,action,{detach=true,signal,manager=managed}={}) {
  if(!['kdb','tune','storage'].includes(command))throw new Error('Managed maintenance command required');
  const config=await readConfig(root);root=config.root;
  const profile=managed.managedOptions(config);
  const {instanceStopped}=await import('./runtime.mjs');
  await manager.configureHostMemory(config.host_memory,{...profile,verifyInstanceStopped:()=>instanceStopped(config)});
  const id=randomUUID(),unit='rsi-maintenance-'+id+'.service',directory=join(root,'requests',id);
  await mkdir(directory,{recursive:true});
  const file=join(directory,'input.json');await writeFile(file,JSON.stringify(action),{mode:0o600});
  const state=new State(join(root,'state.sqlite'));requests(state);
  state.db.prepare("INSERT INTO requests(id,kind,status) VALUES (?,?,'queued')").run(id,command);
  let handle;
  const stop=()=>{if(handle)void handle.stop().catch(()=>{});};
  signal?.addEventListener('abort',stop,{once:true});
  try {
    signal?.throwIfAborted();
    handle=await manager.startManagedMaintenance({argv:[process.execPath,...(command==='storage'?['--max-old-space-size=256']:[]),fileURLToPath(new URL('./main.mjs',import.meta.url)),
      command,'--instance',root,'--config',file,'--request',id,'--managed'],cwd:root,
      env:{...process.env,RSI_MANAGED_ROOT:root,RSI_MANAGED_ROLE:'maintenance',RSI_MANAGED_UNIT:unit},
      stdout:join(directory,'output.log'),stderr:join(directory,'output.log')},
      {...profile,unit,backgroundCompression:command==='storage',memoryBytes:command==='storage'?Math.min(512*1024*1024,config.host_memory.max_bytes-profile.controlMemoryBytes):config.host_memory.max_bytes-profile.controlMemoryBytes});
    state.db.prepare("UPDATE requests SET owner=? WHERE id=? AND status='queued' AND owner IS NULL")
      .run(JSON.stringify(handle.identity),id);
    if(signal?.aborted)await handle.stop();
    if(detach)return {request:id,...handle.identity,submitted:true};
    const exit=await handle.wait({timeoutMs:86400000});
    if(!exit.closed)throw new Error('Managed maintenance closure unresolved');
    await reconcileRequests(root,{state,manager});
    const row=state.db.prepare('SELECT status,result,error FROM requests WHERE id=?').get(id);
    if(row.status!=='passed')throw new Error(row.error??'Managed request did not complete');
    return JSON.parse(row.result);
  } catch(error) {
    // A launch acknowledgement can race a fast, successful transient service's exit.
    const row=state.db.prepare('SELECT status,owner,result FROM requests WHERE id=?').get(id);
    if(row.status==='passed') {
      const retained=await manager.observeManagedMaintenance(JSON.parse(row.owner),profile);
      if(!(await retained.stop()).closed)throw new Error('Completed maintenance closure unresolved');
      return detach?{request:id,...JSON.parse(row.owner),submitted:true}:JSON.parse(row.result);
    }
    if(!handle) {
      const beforeAdmission=['RSI_MEMORY_ENFORCEMENT','RSI_MEMORY_CAPACITY'].includes(error.code);
      state.db.prepare("UPDATE requests SET error=?,status=CASE WHEN ? THEN 'failed' ELSE status END WHERE id=? AND status='queued'")
        .run(error.message,beforeAdmission?1:0,id);
    }
    throw error;
  } finally {signal?.removeEventListener('abort',stop);state.close();}
}

export async function trackRequest(root,id,operation) {
  const state=new State(join(root,'state.sqlite'));requests(state);
  let owner=await processIdentity();
  if(process.env.RSI_MANAGED_ROOT===root) {
    const config=await readConfig(root);
    owner=await managed.managedMaintenanceIdentity(process.env.RSI_MANAGED_UNIT,managed.managedOptions(config));
    if(owner.pid!==process.pid||!owner.invocation){state.close();throw new Error('Managed maintenance service identity required');}
  }
  const updated=state.db.prepare("UPDATE requests SET status='running',owner=? WHERE id=? AND status='queued'")
    .run(JSON.stringify(owner),id);
  if(!updated.changes){state.close();throw new Error('Request is not queued');}
  try {
    const result=await operation();
    state.db.prepare("UPDATE requests SET status='passed',result=? WHERE id=?").run(JSON.stringify(result??null),id);
    return result;
  }catch(error){state.db.prepare("UPDATE requests SET status='failed',error=? WHERE id=?").run(error.message,id);throw error;}
  finally{state.close();}
}
