import {readdir,readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';
import {ownerStopped} from '../../../controller/lifecycle.mjs';

// Only receipts inside this case's isolated temporary tree establish ownership.
export async function closeRegressionServices(root,{signal,remainingSeconds}) {
  const observations=[],receipts=[];
  async function walk(directory){
    for(const entry of await readdir(directory,{withFileTypes:true})) {
      const path=join(directory,entry.name);
      if(entry.isDirectory()&&!['node_modules','.git'].includes(entry.name))await walk(path);
      else if(entry.isFile()&&entry.name==='knowledge-service.json')receipts.push(path);
    }
  }
  await walk(root);
  const bounded=AbortSignal.any([signal,AbortSignal.timeout(Math.max(1,Math.floor(remainingSeconds*1000)))]);
  await Promise.all(receipts.map(async receipt=>{
    const proof={receipt,state:'unknown'};observations.push(proof);
    try {
      const {owner}=JSON.parse(await readFile(receipt,'utf8'));proof.owner=owner;
      if(!Number.isInteger(owner?.pid)||owner.pid<=1||!owner.start||!owner.boot)throw Error('Invalid service identity');
      bounded.throwIfAborted();
      if(!await ownerStopped(owner)) {
        try{process.kill(owner.pid,'SIGTERM');}catch(error){if(error.code!=='ESRCH')throw error;}
        while(!await ownerStopped(owner))await delay(20,undefined,{signal:bounded});
      }
      proof.state='stopped';
    }catch(error){proof.error=error.message;}
  }));
  await writeFile(join(root,'regression-service-closure.json'),JSON.stringify(observations,null,2));
  if(observations.some(o=>o.state!=='stopped'))throw Error('Regression service closure incomplete; inspect retained receipts');
  return observations;
}
