/** Generation-keyed, fixed knowledge views; readiness is completed before timed actors. */
import assert from 'node:assert/strict';
import {mkdir,readFile,writeFile,rename,open,unlink,cp,stat} from 'node:fs/promises';
import {join,dirname} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {State} from '../controller/state.mjs';
import {attentionRevision} from '../controller/failure-attention.mjs';
import {ensureKnowledgeService,run as knowledgeRun} from './knowledge.mjs';
import {releaseKnowledgeService} from './campaign-memory.mjs';
const execute=promisify(execFile),hash=x=>createHash('sha256').update(JSON.stringify(x)).digest('hex');
const optional=async p=>{try{return JSON.parse(await readFile(p,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;return null;}};
const save=async(p,x)=>{const tmp=p+'.'+randomUUID();await writeFile(tmp,JSON.stringify(x,null,2),{mode:0o600});await rename(tmp,p);};
async function digest(path){const h=createHash('sha256');for await(const chunk of createReadStream(path,{highWaterMark:1048576}))h.update(chunk);return h.digest('hex');}
const backupScript=`import pathlib,sqlite3,sys
source=sqlite3.connect(pathlib.Path(sys.argv[1]).resolve().as_uri()+'?mode=ro',uri=True);target=sqlite3.connect(sys.argv[2]);source.backup(target,pages=256);target.close();source.close()
`;
export function admitWarmSearch(row){assert(!row.isError&&row.structured&&Number.isFinite(row.seconds)&&row.seconds<10,'KDB warm search unavailable or exceeds ten seconds');}
export async function prepareFailureMemory({config,directory,generation,signal},dependencies={}){
 assert(generation!==undefined&&generation!==null,'Frozen knowledge generation key required');
 const key=hash(generation),root=join(directory,key),file=join(root,'snapshot.json');await mkdir(root,{recursive:true});let snapshot=await optional(file);
 if(!snapshot){const lock=join(root,'snapshot.lock'),owner=await open(lock,'wx');await owner.close();
  try{const target=join(root,'knowledge.sqlite'),temporary=target+'.'+randomUUID();await (dependencies.backup??((source,destination)=>execute(config.python??'python3',['-c',backupScript,source,destination],{signal})))(join(config.root,'knowledge.sqlite'),temporary);await rename(temporary,target);
   // Preserve attention provenance without inheriting live writer/startup claims.
   await (dependencies.backup??((source,destination)=>execute(config.python??'python3',['-c',backupScript,source,destination],{signal})))(join(config.root,'state.sqlite'),join(root,'state.sqlite'));
   const ledger=new State(join(root,'state.sqlite'));let revision;try{if(ledger.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get())ledger.db.exec('DELETE FROM control');revision=attentionRevision(ledger.db);}finally{ledger.close();}
   await cp(join(config.root,'knowledge.sqlite.lessons'),join(root,'knowledge.sqlite.lessons'),{recursive:true}).catch(e=>{if(e.code!=='ENOENT')throw e;});
   snapshot={generation,key,sha256:await digest(target),bytes:(await stat(target)).size,attentionRevision:revision};await save(file,snapshot);
  }finally{await unlink(lock);}
 }else{assert.equal(hash(snapshot.generation),key,'Frozen knowledge generation changed');await stat(join(root,'knowledge.sqlite'));await stat(join(root,'state.sqlite'));}
 const privateConfig={...config,root,storage:undefined,skills:{...config.skills,tuningDisabled:true}},close=()=> (dependencies.release??releaseKnowledgeService)(privateConfig,{timeoutMs:20000});
 try{
  const cold=signal?AbortSignal.any([signal,AbortSignal.timeout(240000)]):AbortSignal.timeout(240000),url=await (dependencies.ensure??ensureKnowledgeService)(privateConfig,{signal:cold}),rows=[];cold.throwIfAborted();
  const queries=['kernel compilation','numerical correctness','memory indexing','bounded kernel structure'];
  for(const [index,query]of [queries[0],...queries].entries()){
   const action=join(root,'warm-'+index+'.json');await save(action,{tool:'search',arguments:{query,limit:3}});const begin=(dependencies.now??(()=>performance.now()))();
   const response=await (dependencies.run??knowledgeRun)(privateConfig,{config:action,signal:cold}),row={index,seconds:((dependencies.now??(()=>performance.now()))()-begin)/1000,isError:response.isError??false,structured:!!response.structuredContent};cold.throwIfAborted();rows.push(row);await save(join(root,'readiness.json'),rows);
   if(index)admitWarmSearch(row);else assert(!row.isError&&row.structured,'Cold KDB warmup failed');
  }
  return {config:privateConfig,knowledge:{url,readOnly:true,identity:snapshot.sha256,attentionRevision:snapshot.attentionRevision},snapshot,close};
 }catch(error){try{await close();}catch(cleanup){throw new AggregateError([error,cleanup],error.message+'; KDB cleanup: '+cleanup.message);}throw error;}
}
