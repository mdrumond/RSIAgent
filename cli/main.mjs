#!/usr/bin/env node
import { parseArgs } from 'node:util';
import { readFile, writeFile, rename, rm, open } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';
import { join, resolve } from 'node:path';
import { bootstrap, readConfig, updateConfig, upgrade, importProject } from './instance.mjs';
import { State } from '../controller/state.mjs';
import {schemaOptions as storageOptions} from './storage.mjs';

// Storage status reads one metadata row without opening the evolution history.
export async function statusSkills(root, config, state) {
  const skills = {activeVersion: config.skills?.activeVersion ?? null, candidate: null, candidateBase: null};
  if (config.storage) {
    if (!state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='storage_skill_versions'").get()) return skills;
    const latest = state.db.prepare(`SELECT version AS candidate,base_version AS candidateBase,
      decision AS candidateDecision,reason AS candidateReason FROM storage_skill_versions
      ORDER BY rowid DESC LIMIT 1`).get();
    return {...skills, ...latest};
  }
  // Non-storage instances still keep their only candidate pointer in evolution.json.
  let file;
  try {
    file = await open(join(root, 'skills', 'evolution.json'), 'r');
    const buffer = Buffer.alloc(1024 * 1024 + 1);
    const {bytesRead} = await file.read(buffer, 0, buffer.length, 0);
    if (bytesRead === buffer.length) throw new Error('Evolution metadata exceeds status read limit');
    const evolution = JSON.parse(buffer.subarray(0, bytesRead).toString('utf8'));
    for (const key of ['candidate', 'candidateBase'])
      skills[key] = typeof evolution[key] === 'string' ? evolution[key] : null;
  } catch (error) {
    if (error.code !== 'ENOENT') skills.metadataError = error.message;
  } finally { await file?.close(); }
  return skills;
}

export function statusRecoveries(state) {
  if (!state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='recovery_queue'").get()) return [];
  return state.db.prepare(`SELECT project,origin,kind,eligible_wave,attempt,failed_rounds
    FROM recovery_queue ORDER BY project LIMIT 500`).all();
}

export async function main(argv=process.argv.slice(2),{signal}={}) {
  const {positionals,values}=parseArgs({args:argv,allowPositionals:true,options:{
    ...storageOptions,instance:{type:'string'},config:{type:'string'},baseline:{type:'string'},skills:{type:'string'},
    repository:{type:'string'},project:{type:'string'},mode:{type:'string'},port:{type:'string'},request:{type:'string'},
    concurrency:{type:'string'},detach:{type:'boolean'},'no-recovery':{type:'boolean'},help:{type:'boolean'}}});
  const [command,arg]=positionals;
  const root=resolve(values.instance??(command==='storage'?'.':arg)??'.');
  const operation=async()=>{
  if (values.help || !command) return {commands:['bootstrap','upgrade','status','config','skill','project','wave','campaign','resume','pause','stop','dashboard','kdb','tune','storage']};
  const storageAction=command==='storage'?(values.config?JSON.parse(await readFile(values.config,'utf8')):{...values,action:values.action??arg??'inspect'}):null;
  if(storageAction?.action==='unpack')return (await import('./storage.mjs')).runStandalone({...storageAction,signal});
  if (command==='bootstrap') {
    const config=values.config?JSON.parse(await readFile(values.config,'utf8')):{};
    return bootstrap(arg,{...config,baseline:values.baseline??config.baseline,
      skills:values.skills??config.skills,repository:values.repository??config.repository});
  }
  const config=await readConfig(root);
  if(command==='storage') {
    if(!config.storage) {
      if(storageAction.action==='inspect')return {available:false};
      throw new Error('Storage ledger required; existing campaigns are unchanged');
    }
    if(!values.request&&['compress','pack','prune','restore'].includes(storageAction.action))
      return (await import('./background.mjs')).startBackground(root,'storage',storageAction,{detach:Boolean(values.detach),signal});
    return (await import('./storage.mjs')).run(config,{...storageAction,signal});
  }
  if(!values.request&&['tune','kdb'].includes(command)) {
    if(command==='kdb'&&!values.config)throw new Error('KDB action JSON required');
    const action=values.config?JSON.parse(await readFile(values.config,'utf8')):{};
    return (await import('./background.mjs')).startBackground(root,command,action,{detach:Boolean(values.detach),signal});
  }
  if(command==='upgrade') {
    const input=JSON.parse(await readFile(values.config,'utf8'));
    return upgrade(root,input.machinery);
  }
  if (command==='config') {
    const skillRoot=join(root,'skills','versions',config.skills.activeVersion);
    const map=JSON.parse(await readFile(join(skillRoot,'skill-map.json'),'utf8'));
    if (!values.config)return {...config,skillsmap:map};
    const input=JSON.parse(await readFile(values.config,'utf8'));
    let staged,archive,baselineState,activated=false;
    try {
      return await updateConfig(root,async(current,ledger)=>{
        if(input.revision!==undefined && input.revision!==(current.revision??0))throw new Error('Inputs changed; refresh before saving');
        const currentMap=JSON.parse(await readFile(join(root,'skills','versions',current.skills.activeVersion,'skill-map.json'),'utf8'));
        if(input.skillsmap && JSON.stringify(input.skillsmap)!==JSON.stringify(currentMap)) {
          const {copyVersion}=await import('./skill-input.mjs');
          const version=await copyVersion(current,input.skillsmap);
          input.skills={...current.skills,...input.skills,activeVersion:version};
        }
        delete input.skillsmap;
        if(input.baselineKDB && input.baselineKDB!==current.baselineKDB) {
          if(ledger.db.prepare('SELECT COUNT(*) AS n FROM attempts').get().n)throw new Error('Baseline is frozen after first attempt; bootstrap a fresh instance');
          const {ownerStopped}=await import('../controller/lifecycle.mjs');
          if(ledger.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get())
            for(const row of ledger.db.prepare("SELECT value FROM control WHERE key IN ('owner','skill-writer','memory-writer','storage-maintenance','knowledge-startup')").all())
              if(row.value!=='null'&&!await ownerStopped(JSON.parse(row.value)))throw new Error('Stop campaign and knowledge writers before replacing the baseline');
          staged=join(root,'knowledge-next-'+randomUUID()+'.sqlite');
          const db=new DatabaseSync(input.baselineKDB,{readOnly:true});
          try{db.prepare('VACUUM INTO ?').run(staged);}finally{db.close();}
          if(current.storage)baselineState=await (await import('../controller/storage-knowledge.mjs'))
            .prepareStorageBaseline({...current,...input,root},staged,ledger);
        }
        return {...current,...input,format:1};
      },{activate:async()=>{
        if(!staged)return;
        let service;try{service=JSON.parse(await readFile(join(root,'knowledge-service.json'),'utf8'));}
        catch(error){if(error.code!=='ENOENT')throw error;}
        if(service) {
          const {ownerStopped}=await import('../controller/lifecycle.mjs');
          if(!service.owner?.start || !service.owner?.boot)throw new Error('KDB owner cannot be verified');
          if(!await ownerStopped(service.owner)) {
            process.kill(service.owner.pid,'SIGTERM');
            const {setTimeout:delay}=await import('node:timers/promises');
            const deadline=performance.now()+5000;
            while(!await ownerStopped(service.owner) && performance.now()<deadline)await delay(50);
            if(!await ownerStopped(service.owner))throw new Error('KDB service has not stopped');
          }
        }
        archive=join(root,'knowledge-before-'+randomUUID()+'.sqlite');
        await rename(join(root,'knowledge.sqlite'),archive);
        try{await rename(staged,join(root,'knowledge.sqlite'));await baselineState?.activate();activated=true;}
        catch(error){await baselineState?.rollback();await rename(archive,join(root,'knowledge.sqlite'));throw error;}
      },rollback:async()=>{
        if(activated){await baselineState?.rollback();await rm(join(root,'knowledge.sqlite'));await rename(archive,join(root,'knowledge.sqlite'));}
      }});
    }finally{if(staged)await rm(staged,{force:true});}
  }
  if(command==='skill') {
    const action=JSON.parse(await readFile(values.config,'utf8'));
    return (await import('./skill-input.mjs')).run(config,action);
  }
  if(command==='project') {
    if(!values.project) return config.projects;
    const spec=await importProject(root,{actor_seconds:config.actor_seconds,...JSON.parse(await readFile(values.project,'utf8'))});
    return updateConfig(root,(current,state)=>{
      if(state.project(spec.id)) {
        if(state.db.prepare('SELECT 1 FROM attempts WHERE project=? LIMIT 1').get(spec.id))
          throw new Error('Executed projects are frozen; clone under a new id');
        if(!spec.goal || !(spec.actor_seconds>0))throw new Error('Invalid project');
        state.db.prepare('UPDATE projects SET spec=? WHERE id=?').run(JSON.stringify(spec),spec.id);
      }else state.add(spec);
      return {...current,projects:[...current.projects.filter(p=>p.id!==spec.id),spec]};
    });
  }
  if(command==='browse') {
    const {readdir}=await import('node:fs/promises');
    const action=values.config?JSON.parse(await readFile(values.config,'utf8')):{};
    const path=resolve(action.path??root);
    return {path,entries:(await readdir(path,{withFileTypes:true})).filter(e=>e.isDirectory()).map(e=>({name:e.name,path:join(path,e.name),isDirectory:true}))};
  }
  if(command==='status') {
    const state=new State(join(root,'state.sqlite'));
    try {
      const {requests,reconcileRequests}=await import('./background.mjs');
      await reconcileRequests(root,{state});
      let controller={state:'stopped',wave:0,stopRequested:false};
      if(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get()) {
        const rows=Object.fromEntries(state.db.prepare('SELECT key,value FROM control').all().map(row=>[row.key,row.value]));
        try {
          const owner=rows.owner?JSON.parse(rows.owner):null;
          controller.wave=rows.wave?JSON.parse(rows.wave):0;
          controller.stopRequested=rows.stop?Boolean(JSON.parse(rows.stop)):false;
          if(owner) {
            const valid=Number.isInteger(owner.pid)&&owner.pid>0&&typeof owner.start==='string'&&owner.start&&typeof owner.boot==='string'&&owner.boot;
            const {ownerStopped}=await import('../controller/lifecycle.mjs');
            controller.state=valid?(await ownerStopped(owner)?'stopped':'running'):'unknown';
          }
        }catch{controller.state='unknown';}
      }
      const progress=(await import('../controller/progress.mjs')).campaignProgress(state,config);
      const controlValue=key=>state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get()
        ?JSON.parse(state.db.prepare('SELECT value FROM control WHERE key=?').get(key)?.value??'null'):null;
      return {controller,mode:config.mode,progress,skills:await statusSkills(root,config,state),recoveries:statusRecoveries(state),wait:controlValue('wait'),memory:controlValue('memory'),requests:requests(state),projects:state.db.prepare("SELECT id,status,frozen,json_extract(spec,'$.actor_seconds') AS actor_seconds,json_extract(spec,'$.goal') AS goal FROM projects").all(),
        attempts:state.db.prepare('SELECT id,project,elapsed,outcome,created FROM attempts ORDER BY created DESC').all()};
    } finally {state.close();}
  }
  if (['wave','campaign','resume','pause','stop'].includes(command)) {
    const {operate,campaign,resume}=await import('./runtime.mjs');
    if(values.detach&&['wave','campaign','resume'].includes(command))
      return (await import('./background.mjs')).startBackground(root,command,values,{signal});
    if(command==='resume')return resume(config,{...values,signal});
    if(command==='campaign' && !values.detach)return campaign(config,{...values,signal,mode:values.mode??config.mode});
    return operate(command,config,{...values,signal,mode:values.mode??config.mode});
  }
  if(command==='dashboard') {
    const {spawn}=await import('node:child_process');
    const child=spawn(config.python??'python',['-m','ascend_tools.dashboard','--instance',root,
      '--port',values.port??'46731','--node',process.execPath],{stdio:'inherit',env:{...process.env,
        PYTHONPATH:resolve(new URL('..',import.meta.url).pathname)}});
    return new Promise((res,rej)=>{child.on('error',rej);child.on('close',code=>code?rej(new Error('Dashboard failed')):res({stopped:true}));});
  }
  if(command==='kdb' || command==='tune') {
    const module=command==='kdb'?'./knowledge.mjs':'./skills.mjs';
    return (await import(module)).run(config,{...values,signal});
  }
  throw new Error(`Unknown command ${command}`);
  };
  if(values.request)return (await import('./background.mjs')).trackRequest(root,values.request,operation);
  const result=await operation();
  return result;
}
if(import.meta.main) {
  const abort=new AbortController();
  const stop=()=>abort.abort();process.on('SIGTERM',stop);process.on('SIGINT',stop);
  main(undefined,{signal:abort.signal}).then(value=>{if(value!==undefined)console.log(JSON.stringify(value));}).catch(error=>{
    console.error(JSON.stringify({error:error.message}));process.exitCode=1;
  }).finally(()=>{process.off('SIGTERM',stop);process.off('SIGINT',stop);});
}
