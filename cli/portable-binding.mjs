import {readFile,writeFile,rename,rm,realpath} from 'node:fs/promises';
import {createHash,randomUUID} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {DatabaseSync} from 'node:sqlite';
import {join,resolve,isAbsolute,sep} from 'node:path';
import {readConfig,updateConfig} from './instance.mjs';
import {processIdentity,ownerStopped} from '../controller/lifecycle.mjs';
import {administrativeHealthHistory,executionDigest} from '../controller/portable-proof.mjs';

const json=async path=>JSON.parse(await readFile(path,'utf8'));
async function verified(input) {
  if(!input?.path||!/^[a-f0-9]{64}$/.test(input.sha256??''))throw Error('Hashed source evidence required');
  const hash=createHash('sha256');for await(const chunk of createReadStream(input.path))hash.update(chunk);
  if(hash.digest('hex')!==input.sha256)throw Error('Source evidence checksum changed');
  return input.path;
}
const quote=name=>'"'+name.replaceAll('"','""')+'"';
function relocation(paths,root) {
  const roots=new Map(),exact=[];
  for(const [old,name] of Object.entries(paths??{})) {
    if(!isAbsolute(old)||isAbsolute(name)||name.split('/').includes('..'))throw Error('Invalid portable path mapping');
    const target=join(root,name),suffix='/'+name;
    if(old.endsWith(suffix))roots.set(old.slice(0,-suffix.length),root);
    else exact.push([old,target]);
  }
  // Root substitutions cover ordinary instance paths in one pass instead of
  // applying every collected-file mapping to every ledger text value.
  const mappings=[...exact,...roots].filter(([a,b])=>a!==b).sort((a,b)=>b[0].length-a[0].length);
  const destinations=new Map(mappings),pattern=mappings.length?new RegExp(mappings.map(([a])=>a.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|'),'g'):null;
  const rebase=value=>{
    if(typeof value==='string')return pattern?value.replace(pattern,match=>destinations.get(match)):value;
    if(Array.isArray(value))return value.map(rebase);
    if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([k,v])=>[rebase(k),rebase(v)]));
    return value;
  };
  return rebase;
}

export async function bindPortable(root,input) {
  root=await realpath(root);
  const restore=await json(join(root,'restore.json')),original=await readConfig(root);
  if(!restore.complete||!restore.paused||!restore.binding_required||!original.binding_required||original.environment!==null)
    throw Error('An unbound paused portable campaign is required');
  const packagePath=resolve(root,restore.source_package??'original-package.json');
  if(!packagePath.startsWith(root+sep))throw Error('Package metadata must belong to the restored campaign');
  await verified({path:packagePath,sha256:input.packageSha256});
  const metadata=await json(packagePath),campaign=original.storage?.campaign_id;
  if(metadata.format!==2||metadata.engine!=='restic'||metadata.scope!=='campaign'||!metadata.complete||
      metadata.missing?.length||metadata.campaign?.length!==1||metadata.campaign[0].id!==campaign)
    throw Error('Portable package campaign provenance mismatch');
  const shutdownPath=await verified(input.sourceShutdown),shutdown=await json(shutdownPath);
  if(shutdown.stopped!==true||!Number.isSafeInteger(shutdown.recordedScopes)||shutdown.recordedScopes<0||shutdown.remainingScopesStopped!==0)
    throw Error('Confirmed source shutdown audit required');
  const sourcePath=await verified(input.sourceState),source=new DatabaseSync(sourcePath,{readOnly:true});
  const staged=[],owner=await processIdentity();let activated=false;
  const stage=async(path,value)=>{
    const before=await readFile(path),temporary=path+'.'+randomUUID()+'.tmp';
    await writeFile(temporary,JSON.stringify(value)+'\n',{mode:0o600});staged.push({path,before,temporary});
  };
  try {
    if(source.prepare('SELECT id FROM storage_campaign').get()?.id!==campaign)throw Error('Source state campaign mismatch');
    if(typeof input.python!=='string'||!isAbsolute(input.python)||!input.environment?.module||!input.environment.options)
      throw Error('Destination Python and environment binding required');
    return await updateConfig(root,async(current,state)=>{
      if(!current.binding_required||current.environment!==null)throw Error('Portable campaign already bound');
      const controls=state.db.prepare("SELECT value FROM control WHERE key IN ('owner','supervisor-owner','skill-writer','memory-writer','storage-maintenance','knowledge-startup')").all();
      for(const row of controls)if(JSON.parse(row.value)&&!await ownerStopped(JSON.parse(row.value)))throw Error('Destination writer remains live');
      if(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='requests'").get()&&
          state.db.prepare("SELECT 1 FROM requests WHERE status IN ('queued','running')").get())throw Error('Destination background request remains active');
      const attempts=state.db.prepare('SELECT * FROM attempts').all(),oldAttempts=source.prepare('SELECT * FROM attempts').all();
      const projects=state.db.prepare('SELECT id FROM projects ORDER BY id').all(),oldProjects=source.prepare('SELECT id,frozen FROM projects ORDER BY id').all();
      if(JSON.stringify(projects.map(x=>x.id))!==JSON.stringify(oldProjects.map(x=>x.id))||attempts.length!==oldAttempts.length)
        throw Error('Source state inventory mismatch');
      const relocate=relocation(metadata.paths,root);
      for(const attempt of attempts) {
        const old=oldAttempts.find(x=>x.id===attempt.id);
        if(!old||['project','elapsed','outcome'].some(k=>attempt[k]!==old[k])||
            attempt.session!==(restore.paths_need_rebase?old.session:relocate(old.session))||attempt.owner!==null)
          throw Error('Portable attempt differs from source checkpoint');
      }
      const rebase=restore.paths_need_rebase?relocate:x=>x;
      const jobs=[];
      for(const attempt of attempts) {
        const file=join(root,'attempts',attempt.id,'job.json');let job;
        try{job=relocate(await json(file));}catch(error){if(error.code!=='ENOENT'||attempt.outcome===null)throw error;continue;}
        if(attempt.outcome===null) {
          const events=state.events(attempt.id),healthHandles=administrativeHealthHistory(events),slot=job.resource?.metadata?.hostSlot;
          const resources=(input.resources??[]).filter(r=>r.available===true&&r.metadata?.kind==='host-slot'&&r.metadata.hostSlot===slot);
          if(!Number.isSafeInteger(slot)||slot<0||resources.length!==1||!resources[0].id)throw Error('One available destination host slot required');
          jobs.push({attempt,healthHandles,resource:resources[0],previousResource:job.resource});job.resource=resources[0];
        }
        await stage(file,job);
      }
      if(restore.paths_need_rebase) {
        // Same portable tables/schema as the verified snapshot; only TEXT paths
        // change. Existing outcomes, sessions and elapsed allocations survive.
        state.db.function('rebase_portable',{deterministic:true},value=>value===null?null:rebase(value));
        for(const {name} of state.db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").all()) {
          const columns=state.db.prepare('PRAGMA table_info('+quote(name)+')').all().filter(x=>x.type.toUpperCase()==='TEXT');
          if(!columns.length)continue;
          state.db.exec('UPDATE '+quote(name)+' SET '+columns.map(x=>quote(x.name)+'=rebase_portable('+quote(x.name)+')').join(','));
        }
        state.db.exec("UPDATE storage_artifacts SET encoding='plain',descriptor=NULL");
        for(const repo of metadata.repositories??[])if(repo.kind==='source-git') {
          const file=join(root,repo.path,'rsi-source.json');try{await stage(file,rebase(await json(file)));}catch(error){if(error.code!=='ENOENT')throw error;}
        }
      }
      for(const p of oldProjects)state.db.prepare('UPDATE projects SET frozen=? WHERE id=?').run(p.frozen,p.id);
      for(const attempt of attempts)if(attempt.outcome!==null)
        state.db.prepare('UPDATE attempts SET owner=? WHERE id=?').run(oldAttempts.find(x=>x.id===attempt.id).owner,attempt.id);
      for(const job of jobs) {
        const events=state.events(job.attempt.id);
        state.db.prepare('UPDATE attempts SET owner=? WHERE id=? AND owner IS NULL').run(JSON.stringify(owner),job.attempt.id);
        state.event(job.attempt.id,'portable-bound',{eventCount:events.length,executionSha256:executionDigest(events),healthHandles:job.healthHandles,
          sourceShutdown:{...input.sourceShutdown,audit:shutdown},sourceState:input.sourceState,packageSha256:input.packageSha256,
          owner,previousResource:job.previousResource,resource:job.resource});
      }
      state.event(null,'portable-binding',{campaign,owner,sourceShutdown:input.sourceShutdown,sourceState:input.sourceState,
        packageSha256:input.packageSha256,resources:jobs.map(job=>({attempt:job.attempt.id,previous:job.previousResource,current:job.resource}))});
      const budget=state.db.prepare("SELECT value FROM control WHERE key='campaign-budget'").get();
      if(budget)state.db.prepare("UPDATE control SET value=? WHERE key='campaign-budget'").run(JSON.stringify({...JSON.parse(budget.value),running:false}));
      state.db.prepare("INSERT OR REPLACE INTO control VALUES ('pause','true')").run();
      await stage(join(root,'restore.json'),{...restore,binding_required:false,paths_need_rebase:false,bound:true});
      return {...rebase(current),environment:input.environment,python:input.python,binding_required:false,paused:true,
        projects:state.db.prepare('SELECT spec FROM projects').all().map(row=>JSON.parse(row.spec))};
    },{activate:async()=>{activated=true;try{for(const file of staged)await rename(file.temporary,file.path);}
        catch(error){for(const file of staged)await writeFile(file.path,file.before);activated=false;throw error;}},
      rollback:async()=>{if(activated)for(const file of staged)await writeFile(file.path,file.before);}});
  }finally {source.close();for(const file of staged)await rm(file.temporary,{force:true});}
}
