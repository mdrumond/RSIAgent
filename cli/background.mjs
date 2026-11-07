import {mkdir,writeFile,open} from 'node:fs/promises';
import {spawn} from 'node:child_process';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {randomUUID} from 'node:crypto';
import {State} from '../controller/state.mjs';
import {processIdentity,ownerStopped} from '../controller/lifecycle.mjs';
import {readConfig} from './instance.mjs';
import {assertPackAvailable} from './storage.mjs';

export function requests(state) {
  state.db.exec(`CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,kind TEXT NOT NULL,status TEXT NOT NULL,owner TEXT,result TEXT,error TEXT)`);
  return state.db.prepare('SELECT id,kind,status,owner,result,error FROM requests ORDER BY rowid DESC LIMIT 100').all().map(row=>({
    ...row,owner:row.owner?JSON.parse(row.owner):null,result:row.result?JSON.parse(row.result):null}));
}

// Observe recorded process identity; never replay a request automatically.
export async function reconcileRequests(root,{state}={}) {
  const own=!state;state??=new State(join(root,'state.sqlite'));
  try {
    for(const request of requests(state)) {
      if(!['queued','running'].includes(request.status)||!request.owner||!await ownerStopped(request.owner))continue;
      state.db.prepare(`UPDATE requests SET status='failed',error=? WHERE id=?
        AND status IN ('queued','running') AND owner=?`).run(
        'Request process exited before recording completion',request.id,JSON.stringify(request.owner));
    }
    return requests(state);
  }finally{if(own)state.close();}
}

function terminateGroup(pid) {
  try{process.kill(-pid,'SIGTERM');}
  catch(error){if(error.code!=='ESRCH')throw error;try{process.kill(pid,'SIGTERM');}catch(e){if(e.code!=='ESRCH')throw e;}}
}
export async function stopRequest(owner) {
  if(owner?.pid&&owner.start&&owner.boot&&!await ownerStopped(owner))terminateGroup(owner.pid);
}

export async function startBackground(root,command,action,{detach=true,signal}={}) {
  if(!['kdb','tune','storage','wave','campaign','resume'].includes(command))throw new Error('Unknown background command');
  signal?.throwIfAborted();
  const config=await readConfig(root);root=config.root;
  const id=randomUUID(),directory=join(root,'requests',id);
  await mkdir(directory,{recursive:true});
  const file=join(directory,'input.json');await writeFile(file,JSON.stringify(action),{mode:0o600});
  const state=new State(join(root,'state.sqlite'));requests(state);
  state.db.exec('BEGIN IMMEDIATE');
  try {
    await assertPackAvailable(state);
    state.db.prepare("INSERT INTO requests(id,kind,status) VALUES (?,?,'queued')").run(id,command);
    state.db.exec('COMMIT');
  }catch(error){state.db.exec('ROLLBACK');state.close();throw error;}
  const log=await open(join(directory,'output.log'),'a',0o600);
  let child,exited;
  const cancel=()=>{if(child?.pid&&child.exitCode===null&&child.signalCode===null)terminateGroup(child.pid);};
  signal?.addEventListener('abort',cancel,{once:true});
  try {
    const argv=[fileURLToPath(new URL('./main.mjs',import.meta.url)),command,'--instance',root,'--config',file,'--request',id];
    for(const key of ['mode','concurrency','memory-pause-token'])if(action[key]!==undefined)argv.push('--'+key,String(action[key]));
    if(action['no-recovery'])argv.push('--no-recovery');
    child=spawn(process.execPath,argv,{cwd:root,detached:true,env:process.env,stdio:['ignore',log.fd,log.fd]});
    exited=new Promise(resolve=>child.once('exit',resolve));
    await new Promise((ok,no)=>{child.once('spawn',ok);child.once('error',no);});
    if(signal?.aborted)cancel();
    const owner=await processIdentity(child.pid);
    state.db.prepare("UPDATE requests SET owner=? WHERE id=? AND status='queued' AND owner IS NULL")
      .run(JSON.stringify(owner),id);
    if(detach){child.unref();return {request:id,...owner,submitted:true};}
    await exited;
    await reconcileRequests(root,{state});
    const row=state.db.prepare('SELECT status,result,error FROM requests WHERE id=?').get(id);
    if(row.status!=='passed')throw new Error(row.error??'Request did not complete');
    return JSON.parse(row.result);
  }catch(error){
    if(child?.pid&&['ENOENT','ESRCH'].includes(error.code))await exited;
    const row=state.db.prepare('SELECT status,owner,result FROM requests WHERE id=?').get(id);
    if(row?.status==='passed')return detach?{request:id,...JSON.parse(row.owner),submitted:true}:JSON.parse(row.result);
    if(child?.pid&&(child.exitCode!==null||child.signalCode!==null))
      state.db.prepare("UPDATE requests SET status='failed',error=? WHERE id=? AND status='queued' AND owner IS NULL")
        .run('Request process exited before recording ownership',id);
    if(!child?.pid)state.db.prepare("UPDATE requests SET status='failed',error=? WHERE id=?").run(error.message,id);
    throw error;
  }finally{signal?.removeEventListener('abort',cancel);await log.close();state.close();}
}

export async function trackRequest(root,id,operation) {
  const state=new State(join(root,'state.sqlite'));requests(state);
  const owner=await processIdentity();
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
