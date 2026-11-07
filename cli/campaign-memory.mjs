import {readFile,readdir} from 'node:fs/promises';
import {join} from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';
import {State} from '../controller/state.mjs';
import {ownerStopped,processIdentity} from '../controller/lifecycle.mjs';

const GiB=1024**3;
export const memoryDefaults={softBytes:24*GiB,checkpointBytes:28*GiB,lowBytes:20*GiB,hardBytes:32*GiB,
  reduceMs:30000,highMs:120000,coolMs:60000,pollMs:2000};
export function memorySettings(input) {
  const settings={...memoryDefaults,...input};
  if(!Object.values(settings).every(value=>Number.isSafeInteger(value)&&value>0)||
    !(settings.lowBytes<settings.softBytes&&settings.softBytes<settings.checkpointBytes&&settings.checkpointBytes<settings.hardBytes))
    throw Error('Ordered positive local memory thresholds required');
  return settings;
}
export function memoryDecision(previous,{rssBytes,now,maximum},settings=memoryDefaults) {
  if(!Number.isSafeInteger(rssBytes)||rssBytes<0||!Number.isFinite(now)||!Number.isSafeInteger(maximum)||maximum<1)
    throw Error('Measured memory, clock and worker limit required');
  const state={...previous,limit:Math.min(previous?.limit??maximum,maximum)};
  if(rssBytes>=settings.hardBytes)return {...state,action:'emergency',blocked:true};
  if(state.token) {
    state.lowSince=rssBytes<settings.lowBytes?(state.lowSince??now):null;
    return {...state,action:state.lowSince!==null&&now-state.lowSince>=settings.coolMs?'resume':'cooling',blocked:true};
  }
  const newlyHigh=state.highSince==null;
  if(rssBytes<settings.softBytes)return {...state,highSince:null,lastReduction:null,blocked:false,action:'run'};
  state.highSince??=now;
  if(newlyHigh||now-(state.lastReduction??now)>=settings.reduceMs) {
    state.limit=Math.max(1,state.limit-1);state.lastReduction=now;
  }
  const checkpoint=rssBytes>=settings.checkpointBytes||now-state.highSince>=settings.highMs;
  return {...state,blocked:true,action:checkpoint?(previous?.recovered&&previous?.limit===1?'exhausted':'checkpoint'):'throttle'};
}
function ledger(config,operation) {
  const state=new State(join(config.root,'state.sqlite'));
  try{return operation(state);}finally{state.close();}
}
export function writeMemoryControl(config,memory) {
  ledger(config,state=>state.db.prepare('INSERT OR REPLACE INTO control VALUES (?,?)')
    .run('memory-admission',JSON.stringify(memory)));
}
export function requestMemoryPause(config,token) {
  ledger(config,state=>{
    state.db.exec('BEGIN IMMEDIATE');
    try {
      const value=key=>JSON.parse(state.db.prepare('SELECT value FROM control WHERE key=?').get(key)?.value??'null');
      if(value('pause')||value('operator-stop'))throw Error('Operator pause or stop precedes memory recovery');
      const set=(key,value)=>state.db.prepare('INSERT OR REPLACE INTO control VALUES (?,?)').run(key,JSON.stringify(value));
      set('memory-pause',token);set('pause-at',Date.now());set('pause',true);state.db.exec('COMMIT');
    }catch(error){state.db.exec('ROLLBACK');throw error;}
  });
}
export async function releaseKnowledgeService(config) {
  let service;
  try{service=JSON.parse(await readFile(join(config.root,'knowledge-service.json'),'utf8'));}
  catch(error){if(error.code==='ENOENT')return;throw error;}
  if(!service.owner?.start||!service.owner?.boot)throw Error('Knowledge service ownership unknown');
  if(await ownerStopped(service.owner))return;
  const current=await processIdentity(service.owner.pid);
  if(current.start===service.owner.start&&current.boot===service.owner.boot)process.kill(current.pid,'SIGTERM');
  const until=performance.now()+30000;
  while(!await ownerStopped(service.owner)) {
    if(performance.now()>=until)throw Error('Knowledge service did not release memory after checkpoint');
    await delay(100);
  }
}
// Use comm, never argv/environment. Owned roots also cover detached services after supervisor restart.
export async function sampleCampaignMemory(config,owner,{proc='/proc',known=new Map()}={}) {
  const roots=new Map([[owner.pid,{...owner,role:'supervisor'}]]);
  ledger(config,state=>{
    const add=(raw,role)=>{const identity=typeof raw==='string'?JSON.parse(raw):raw;
      if(identity?.pid&&identity.start&&identity.boot===owner.boot&&!roots.has(identity.pid))roots.set(identity.pid,{...identity,role});};
    for(const row of state.db.prepare("SELECT key,value FROM control WHERE key IN ('owner','skill-writer','memory-writer','storage-maintenance')").all())add(row.value,row.key==='owner'?'controller':row.key);
    for(const row of state.db.prepare('SELECT owner FROM attempts WHERE owner IS NOT NULL').all())add(row.owner,'worker');
    if(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='requests'").get())for(const row of state.db.prepare("SELECT owner FROM requests WHERE owner IS NOT NULL AND status IN ('queued','running')").all())add(row.owner,'request');
  });
  try{const service=JSON.parse(await readFile(join(config.root,'knowledge-service.json'),'utf8'));
    if(service.owner?.boot===owner.boot)roots.set(service.owner.pid,{...service.owner,role:'knowledge'});
  }catch(error){if(error.code!=='ENOENT')throw error;}
  const rows=new Map();
  await Promise.all((await readdir(proc)).filter(name=>/^\d+$/.test(name)).map(async name=>{
    try{const stat=await readFile(join(proc,name,'stat'),'utf8'),end=stat.lastIndexOf(')'),fields=stat.slice(end+2).split(' ');
      if(!['Z','X'].includes(fields[0]))rows.set(Number(name),{pid:Number(name),parent:Number(fields[1]),start:fields[19],name:stat.slice(stat.indexOf('(')+1,end)});
    }catch(error){if(!['ENOENT','ESRCH'].includes(error.code))throw error;}
  }));
  const owned=new Map();
  for(const [pid,prior] of known)if(rows.get(pid)?.start===prior.start)owned.set(pid,{...rows.get(pid),role:prior.role});else known.delete(pid);
  for(const [pid,root] of roots)if(rows.get(pid)?.start===root.start)owned.set(pid,{...rows.get(pid),role:root.role});
  for(let size=-1;size!==owned.size;) {
    size=owned.size;
    for(const [pid,row] of rows)if(!owned.has(pid)&&owned.has(row.parent))owned.set(pid,{...row,role:owned.get(row.parent).role});
  }
  for(const [pid,row] of owned)known.set(pid,row);
  const processes=[];
  for(const row of owned.values()) {
    try {
      const status=await readFile(join(proc,String(row.pid),'status'),'utf8');
      const fields=(await readFile(join(proc,String(row.pid),'stat'),'utf8')).split(')').at(-1).trim().split(' ');
      if(fields[19]!==row.start)continue;
      const match=status.match(/^VmRSS:\s+(\d+) kB$/m);if(!match)continue;
      processes.push({...row,rssBytes:Number(match[1])*1024});
    }catch(error){if(!['ENOENT','ESRCH'].includes(error.code))throw error;}
  }
  return {rssBytes:processes.reduce((sum,row)=>sum+row.rssBytes,0),processCount:processes.length,
    largest:processes.sort((a,b)=>b.rssBytes-a.rssBytes).slice(0,12)};
}
