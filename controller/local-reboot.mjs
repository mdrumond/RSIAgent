import {readFile,realpath,writeFile,link,rm} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {createHash,randomUUID} from 'node:crypto';
import {hostname,arch} from 'node:os';
import {join,resolve,relative,isAbsolute,basename,dirname} from 'node:path';

const digest=value=>createHash('sha256').update(value).digest('hex');
const kind='operator-attested-same-host-reboot';
const json=async path=>JSON.parse(await readFile(path,'utf8'));
export async function localHostIdentity() {
 let machine='';try{machine=(await readFile('/etc/machine-id','utf8')).trim();}
 catch(error){if(error.code!=='ENOENT')throw error;}
 return {boot:(await readFile('/proc/sys/kernel/random/boot_id','utf8')).trim(),architecture:arch(),
  fingerprint:digest(JSON.stringify([machine,hostname(),arch()]))};
}
export async function fileSha256(path) {
 const hash=createHash('sha256');for await(const chunk of createReadStream(path))hash.update(chunk);return hash.digest('hex');
}
const sameHost=(a,b)=>a?.fingerprint===b.fingerprint&&a?.architecture===b.architecture;

// A boot mismatch is not proof: copied campaign receipts may belong to a live
// machine elsewhere. Only this explicit custody attestation can close them.
export async function attestLocalReboot(root,input) {
 const host=await localHostIdentity();root=await realpath(root);
 if(input?.format!==1||input.kind!==kind||input.operator?.sameHost!==true||input.operator?.migrated!==false||
   !input.operator.target||input.operator.previousTarget!==input.operator.target||
   input.operator.previousArchitecture!==host.architecture||!input.operator.reason?.trim())
  throw Error('Explicit same-host, non-migrated reboot attestation required');
 if(!sameHost(input.currentHost,host)||input.currentHost.boot!==host.boot||
   typeof input.previousBoot!=='string'||!input.previousBoot||input.previousBoot===host.boot)
  throw Error('Current host or reboot identity differs');
 if(!Array.isArray(input.evidence)||!input.evidence.length||!Array.isArray(input.receipts)||!input.receipts.length)
  throw Error('Hashed custody evidence and explicit execution receipts required');
 for(const evidence of input.evidence)
  if(!evidence.path||!/^[a-f0-9]{64}$/.test(evidence.sha256??'')||await fileSha256(evidence.path)!==evidence.sha256)
   throw Error('Custody evidence checksum differs');
 const entries=[],seen=new Set();
 for(const entry of input.receipts) {
  const path=await realpath(resolve(root,entry.path)),name=relative(root,path),directory=dirname(path);
  if(!name||name.startsWith('..')||isAbsolute(name)||basename(path)!=='execution.json'||
    basename(dirname(directory))!=='.executions'||!/^exec-[A-Za-z0-9]+$/.test(basename(directory))||seen.has(path))
   throw Error('Explicit instance-owned local execution receipt required');
  seen.add(path);
  if(!/^[a-f0-9]{64}$/.test(entry.sha256??'')||await fileSha256(path)!==entry.sha256)
   throw Error('Execution receipt checksum differs');
  const receipt=await json(path);
  if(receipt.boot!==input.previousBoot||typeof receipt.cwd!=='string'||
    receipt.hostIdentity&&(!sameHost(receipt.hostIdentity,host)||receipt.hostIdentity.boot!==receipt.boot))
   throw Error('Execution belongs to a different host or boot');
  entries.push({path,directory,handle:'local:'+basename(directory),sha256:entry.sha256});
 }
 const closures=[];
 for(const entry of entries) {
  const path=join(entry.directory,'reboot-closure.json');
  let previous;try{previous=await json(path);}catch(error){if(error.code!=='ENOENT')throw error;}
  if(previous&&(previous.kind!==kind||previous.receiptSha256!==entry.sha256||
     previous.previousBoot!==input.previousBoot||!sameHost(previous.observedHost,host)))
   throw Error('Different reboot closure already retained');
  if(input.apply===true&&!previous) {
   // Recheck immediately before publishing; execution.json is never rewritten.
   if(await fileSha256(entry.path)!==entry.sha256)throw Error('Execution receipt changed during attestation');
   const proof={format:1,kind,closed:true,handle:entry.handle,receiptSha256:entry.sha256,
    previousBoot:input.previousBoot,observedHost:host,operator:input.operator,evidence:input.evidence,
    recordedAt:new Date().toISOString()};
   const temporary=path+'.'+randomUUID();
   try{await writeFile(temporary,JSON.stringify(proof)+'\n',{mode:0o600});await link(temporary,path);}
   finally{await rm(temporary,{force:true});}
  }
  closures.push({handle:entry.handle,receipt:entry.path,proof:path,reused:Boolean(previous)});
 }
 return {applied:input.apply===true,currentHost:host,closures};
}

export async function observeRebootClosure(directory,receipt,host) {
 try {
  const proof=await json(join(directory,'reboot-closure.json'));
  if(proof.format!==1||proof.kind!==kind||proof.closed!==true||proof.handle!=='local:'+basename(directory)||
    proof.previousBoot!==receipt.boot||typeof proof.observedHost?.boot!=='string'||!proof.observedHost.boot||
    proof.observedHost.boot===receipt.boot||!sameHost(proof.observedHost,host)||
    proof.operator?.sameHost!==true||proof.operator?.migrated!==false||
    !proof.operator.target||proof.operator.previousTarget!==proof.operator.target||
    proof.operator.previousArchitecture!==host.architecture||!Array.isArray(proof.evidence)||!proof.evidence.length||
    proof.receiptSha256!==await fileSha256(join(directory,'execution.json')))return null;
  let exitCode;
  try{const text=(await readFile(join(directory,'exit'),'utf8')).trim();if(/^\d+$/.test(text)&&Number.isSafeInteger(Number(text)))exitCode=Number(text);}
  catch(error){if(error.code!=='ENOENT')throw error;}
  return {state:'stopped',...(exitCode===undefined?{}:{exitCode}),
   ...(receipt.outcome?{outcome:receipt.outcome}:exitCode===undefined?{outcome:{kind:'infrastructure',reason:'host-reboot'}}:{}),
   closure:{kind,previousBoot:proof.previousBoot,observedBoot:proof.observedHost.boot,receiptSha256:proof.receiptSha256}};
 }catch{return null;}
}
