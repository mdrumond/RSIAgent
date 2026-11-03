import { spawn } from 'node:child_process';
import { mkdir,writeFile,open } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { State } from '../controller/state.mjs';
import { processIdentity } from '../controller/lifecycle.mjs';

export function requests(state) {
  state.db.exec(`CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,kind TEXT NOT NULL,status TEXT NOT NULL,owner TEXT,result TEXT,error TEXT)`);
  return state.db.prepare('SELECT id,kind,status,owner,result,error FROM requests').all().map(row=>({
    ...row,owner:row.owner?JSON.parse(row.owner):null,result:row.result?JSON.parse(row.result):null}));
}
export async function startBackground(root,command,action) {
  const id=randomUUID(),directory=join(root,'requests',id);await mkdir(directory,{recursive:true});
  const file=join(directory,'input.json');await writeFile(file,JSON.stringify(action),{mode:0o600});
  const state=new State(join(root,'state.sqlite'));requests(state);
  state.db.prepare("INSERT INTO requests(id,kind,status) VALUES (?,?,'queued')").run(id,command);
  const log=await open(join(directory,'output.log'),'a',0o600);
  try {
    const child=spawn(process.execPath,[fileURLToPath(new URL('./main.mjs',import.meta.url)),command,
      '--instance',root,'--config',file,'--request',id],{detached:true,stdio:['ignore',log.fd,log.fd]});
    await new Promise((ok,no)=>{child.once('spawn',ok);child.once('error',no);});child.unref();
    return {request:id,pid:child.pid,submitted:true};
  }catch(error){state.db.prepare("UPDATE requests SET status='failed',error=? WHERE id=?").run(error.message,id);throw error;}
  finally{await log.close();state.close();}
}
export async function trackRequest(root,id,operation) {
  const state=new State(join(root,'state.sqlite'));requests(state);
  const updated=state.db.prepare("UPDATE requests SET status='running',owner=? WHERE id=? AND status='queued'")
    .run(JSON.stringify(await processIdentity()),id);
  if(!updated.changes){state.close();throw new Error('Request is not queued');}
  try {
    const result=await operation();
    state.db.prepare("UPDATE requests SET status='passed',result=? WHERE id=?").run(JSON.stringify(result??null),id);
    return result;
  }catch(error){state.db.prepare("UPDATE requests SET status='failed',error=? WHERE id=?").run(error.message,id);throw error;}
  finally{state.close();}
}
