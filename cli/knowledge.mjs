import { spawn } from 'node:child_process';
import { mkdir, open, readFile, writeFile, unlink } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { createServer } from 'node:net';
import { setTimeout as delay } from 'node:timers/promises';
import { processIdentity, ownerStopped } from '../controller/lifecycle.mjs';
import { State } from '../controller/state.mjs';

const load=async path=>JSON.parse(await readFile(path,'utf8'));
async function freePort() {
  const server=createServer();await new Promise((ok,no)=>{server.once('error',no);server.listen(0,'127.0.0.1',ok);});
  const port=server.address().port;await new Promise(ok=>server.close(ok));return port;
}
async function healthy(url) {
  try {return (await fetch(url.replace('/mcp','/health'),{signal:AbortSignal.timeout(1000)})).ok;}
  catch{return false;}
}
export async function ensureKnowledgeService(config,{signal}={}) {
  const receipt=join(config.root,'knowledge-service.json'),legacyLock=receipt+'.lock';
  const deadline=performance.now()+60000,key='knowledge-startup';
  const state=new State(join(config.root,'state.sqlite')),me=JSON.stringify(await processIdentity());
  state.db.exec('CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY,value TEXT NOT NULL)');
  const wait=()=>delay(100,undefined,{signal});
  const inspect=async()=>{
    let previous;
    try {previous=await load(receipt);} catch(error){
      if(error.code==='ENOENT')return {};
      if(error instanceof SyntaxError)return {pending:true};
      throw error;
    }
    if(await ownerStopped(previous.owner))return {};
    return await healthy(previous.url)?{url:previous.url}:{pending:true};
  };
  try {
    while(performance.now()<deadline) {
      signal?.throwIfAborted();
      const previous=await inspect();
      if(previous.url)return previous.url;
      if(previous.pending){await wait();continue;}
      const row=state.db.prepare('SELECT value FROM control WHERE key=?').get(key);
      if(row) {
        let owner;try{owner=JSON.parse(row.value);}catch{await wait();continue;}
        if(!await ownerStopped(owner)){await wait();continue;}
      }
      state.db.exec('BEGIN IMMEDIATE');
      try {
        const current=state.db.prepare('SELECT value FROM control WHERE key=?').get(key);
        if(current?.value!==row?.value){state.db.exec('ROLLBACK');continue;}
        state.db.prepare('INSERT OR REPLACE INTO control(key,value) VALUES (?,?)').run(key,me);
        state.db.exec('COMMIT');
      }catch(error){state.db.exec('ROLLBACK');throw error;}
      try {
        // Another claimant may have launched a service since the first inspection.
        const latest=await inspect();
        if(latest.url)return latest.url;
        if(latest.pending){await wait();continue;}
        try {
          let owner;
          try{owner=await load(legacyLock);}catch(error){
            if(error instanceof SyntaxError){await wait();continue;}
            throw error;
          }
          if(!await ownerStopped(owner)){await wait();continue;}
          await unlink(legacyLock);
        }catch(error){if(error.code!=='ENOENT')throw error;}
        signal?.throwIfAborted();
        const port=await freePort(),url=`http://127.0.0.1:${port}/mcp`;
        await mkdir(join(config.root,'logs'),{recursive:true});
        const log=await open(join(config.root,'logs','knowledge.log'),'a',0o600);
        let child;
        try {
          const args=['-m','ascend_tools.kdb_service','--db',join(config.root,'knowledge.sqlite'),'--port',String(port)];
          if(config.embedding_factory)args.push('--embedding-factory',config.embedding_factory);
          if(config.embeddings?.cache_dir)args.push('--embedding-cache',config.embeddings.cache_dir);
          child=spawn(config.python??'python',args,{cwd:config.root,detached:true,
            stdio:['ignore',log.fd,log.fd],env:{...process.env,PYTHONPATH:resolve(new URL('..',import.meta.url).pathname)}});
          await new Promise((ok,no)=>{child.once('spawn',ok);child.once('error',no);});
        }finally{await log.close();}
        const owner=await processIdentity(child.pid);child.unref();
        await writeFile(receipt,JSON.stringify({url,owner}),{mode:0o600});
        while(performance.now()<deadline) {
          signal?.throwIfAborted();
          if(await healthy(url))return url;
          if(await ownerStopped(owner))throw new Error('KDB service exited; inspect logs/knowledge.log');
          await wait();
        }
        // A readiness timeout is not permission to replace or terminate a live service.
        throw new Error('KDB service still pending; retain its recorded owner');
      }finally{state.db.prepare('DELETE FROM control WHERE key=? AND value=?').run(key,me);}
    }
    throw new Error('KDB startup/readiness pending; inspect retained owner');
  }finally{state.close();}
}

export async function run(config,values) {
  if(!values.config)throw new Error('KDB action JSON required');
  const action=await load(values.config);
  const url=await ensureKnowledgeService(config,{signal:values.signal});
  const args=['-m','ascend_tools.kdb_client','--url',url,'--tool',action.tool,
    '--arguments',JSON.stringify(action.arguments??{})];
  return new Promise((ok,no)=>{
    const child=spawn(config.python??'python',args,{env:{...process.env,
      PYTHONPATH:resolve(new URL('..',import.meta.url).pathname)},signal:values.signal,stdio:['ignore','pipe','pipe']});
    let output='',error='';child.stdout.on('data',d=>output+=d);child.stderr.on('data',d=>error+=d);
    child.on('error',no);child.on('close',code=>{if(code)no(new Error(error));else{try{ok(JSON.parse(output));}catch(e){no(e);}}});
  });
}
