import { spawn } from 'node:child_process';
import { mkdir, open, readFile, writeFile, unlink } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { createServer } from 'node:net';
import { setTimeout as delay } from 'node:timers/promises';
import { processIdentity, ownerStopped } from '../controller/lifecycle.mjs';

const load=async path=>JSON.parse(await readFile(path,'utf8'));
async function freePort() {
  const server=createServer();await new Promise((ok,no)=>{server.once('error',no);server.listen(0,'127.0.0.1',ok);});
  const port=server.address().port;await new Promise(ok=>server.close(ok));return port;
}
async function healthy(url) {
  try {return (await fetch(url.replace('/mcp','/health'),{signal:AbortSignal.timeout(1000)})).ok;}
  catch{return false;}
}
export async function ensureKnowledgeService(config) {
  const receipt=join(config.root,'knowledge-service.json'),lock=receipt+'.lock';
  const deadline=performance.now()+60000;
  while(performance.now()<deadline) {
    try {
      const previous=await load(receipt);
      if(!await ownerStopped(previous.owner) && await healthy(previous.url))return previous.url;
    }catch(error){if(error.code && error.code!=='ENOENT')throw error;}
    let file;
    try {file=await open(lock,'wx',0o600);}catch(error){if(error.code!=='EEXIST')throw error;await delay(100);continue;}
    try {
      await file.writeFile(JSON.stringify(await processIdentity()));
      const port=await freePort(),url=`http://127.0.0.1:${port}/mcp`;
      await mkdir(join(config.root,'logs'),{recursive:true});
      const log=await open(join(config.root,'logs','knowledge.log'),'a',0o600);
      const args=['-m','ascend_tools.kdb_service','--db',join(config.root,'knowledge.sqlite'),'--port',String(port)];
      if(config.embedding_factory)args.push('--embedding-factory',config.embedding_factory);
      if(config.embeddings?.cache_dir)args.push('--embedding-cache',config.embeddings.cache_dir);
      const child=spawn(config.python??'python',args,{cwd:config.root,detached:true,
        stdio:['ignore',log.fd,log.fd],env:{...process.env,PYTHONPATH:resolve(new URL('..',import.meta.url).pathname)}});
      const spawned=new Promise((ok,no)=>{child.once('spawn',ok);child.once('error',no);});
      await log.close();await spawned;
      const owner=await processIdentity(child.pid);child.unref();
      await writeFile(receipt,JSON.stringify({url,owner}),{mode:0o600});
      while(performance.now()<deadline) {
        if(await healthy(url))return url;
        if(await ownerStopped(owner))throw new Error('KDB service exited; inspect logs/knowledge.log');
        await delay(100);
      }
      process.kill(child.pid,'SIGTERM');throw new Error('KDB readiness deadline exceeded');
    }finally{await file.close();await unlink(lock);}
  }
  throw new Error('KDB service startup locked; inspect its recorded owner before recovery');
}

export async function run(config,values) {
  if(!values.config)throw new Error('KDB action JSON required');
  const action=await load(values.config);
  const url=await ensureKnowledgeService(config);
  const args=['-m','ascend_tools.kdb_client','--url',url,'--tool',action.tool,
    '--arguments',JSON.stringify(action.arguments??{})];
  return new Promise((ok,no)=>{
    const child=spawn(config.python??'python',args,{env:{...process.env,
      PYTHONPATH:resolve(new URL('..',import.meta.url).pathname)},signal:values.signal,stdio:['ignore','pipe','pipe']});
    let output='',error='';child.stdout.on('data',d=>output+=d);child.stderr.on('data',d=>error+=d);
    child.on('error',no);child.on('close',code=>{if(code)no(new Error(error));else{try{ok(JSON.parse(output));}catch(e){no(e);}}});
  });
}
