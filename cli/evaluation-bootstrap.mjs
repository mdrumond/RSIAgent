import { readFile, writeFile, rename, stat } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { ownerStopped, processIdentity } from '../controller/lifecycle.mjs';
import { createEnvironment } from '../integrations/environments/index.mjs';

const receiptPath=root=>root+'.bootstrap.json';
const quote=value=>"'"+String(value).replaceAll("'","'\\''")+"'";
const save=async(root,value)=>{
  const temporary=receiptPath(root)+'.'+randomUUID();
  await writeFile(temporary,JSON.stringify(value),{mode:0o600});
  await rename(temporary,receiptPath(root));
};
const environment=(root,event=async()=>{},signal)=>createEnvironment({mode:'local'},
  {instancePath:root+'-setup',event,exclude:async(_label,fn)=>fn(),signal});
async function closed(root,receipt) {
  if(receipt.format!==1 || !receipt.owner?.pid || !receipt.owner.start || !receipt.owner.boot ||
    !['preparing','stopped','ready'].includes(receipt.status) || !Array.isArray(receipt.handles))return false;
  if(receipt.status==='preparing' && receipt.phase==='storage')return false;
  if(receipt.status!=='stopped' && receipt.status!=='ready' && !await ownerStopped(receipt.owner))return false;
  if(!receipt.handles.length)return true;
  const local=await environment(root);
  for(const handle of receipt.handles)if((await local.observe(handle)).state!=='stopped')return false;
  return true;
}
export async function bootstrapSettled(root) {
  let receipt;
  try{receipt=JSON.parse(await readFile(receiptPath(root),'utf8'));}
  catch(error){
    if(error.code!=='ENOENT')return false;
    // Canonical setup records its claim before creating either destination or
    // launching work. No claim and no destination is positive zero-dispatch proof.
    for(const path of [root,root+'-setup']) {
      try{await stat(path);return false;}catch(error){if(error.code!=='ENOENT')return false;}
    }
    return true;
  }
  try{return await closed(root,receipt);}catch{return false;}
}

// The enclosing skill-writer lock serializes claims. A receipt is written before
// touching the destination; a partial without this provenance is never continued.
export async function managedBootstrap(root,{binding,signal,remainingSeconds=28800},initialize) {
  signal?.throwIfAborted();
  const signature=createHash('sha256').update(JSON.stringify(binding)).digest('hex');
  let receipt,resume=false;
  try{receipt=JSON.parse(await readFile(receiptPath(root),'utf8'));}
  catch(error){if(error.code!=='ENOENT')throw error;}
  if(receipt) {
    if(receipt.signature!==signature)throw new Error('Evaluation bootstrap frozen binding changed');
    if(!await closed(root,receipt))throw new Error('Evaluation bootstrap remains unresolved');
    resume=true;
  } else {
    try{await stat(root);throw new Error('Unreceipted partial evaluation bootstrap');}
    catch(error){if(error.code!=='ENOENT')throw error;}
    receipt={format:1,signature,handles:[],completed:[]};
  }
  receipt.owner=await processIdentity();receipt.status='preparing';
  await save(root,receipt);
  const started=performance.now();
  let local;
  const setup=async(name,command,args,cwd,env)=>{
    signal?.throwIfAborted();
    if(receipt.completed.includes(name))return;
    local??=await environment(root,async(type,data)=>{
      if(type==='execution-handle'){receipt.handles.push(data.handle);await save(root,receipt);}
    },signal);
    const remaining=remainingSeconds-(performance.now()-started)/1000;
    if(!(remaining>0))throw new Error('Evaluation bootstrap elapsed allocation exhausted');
    const result=await local.execute([command,...args].map(quote).join(' '),cwd,{env,timeout:remaining,signal});
    signal?.throwIfAborted();
    if(!result.settled || result.exitCode!==0)throw new Error('Evaluation setup failed: '+name);
    receipt.completed.push(name);await save(root,receipt);
  };
  try {
    const result=await initialize({resume,setup,phase:async phase=>{receipt.phase=phase;await save(root,receipt);}});
    receipt.status='ready';await save(root,receipt);return result;
  } catch(error) {
    // A live owner can explicitly acknowledge closure only after every durable
    // setup handle is observed stopped; an unknown handle retains its fence.
    const probe={...receipt,status:'stopped'};
    if(await closed(root,probe)){receipt.status='stopped';await save(root,receipt);}
    throw error;
  }
}
