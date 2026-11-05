#!/usr/bin/env node
import { parseArgs } from 'node:util';
import { readFile, rename, rm } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';
import { join, resolve } from 'node:path';
import { bootstrap, readConfig, updateConfig, upgrade, importProject } from './instance.mjs';
import { State } from '../controller/state.mjs';

export async function main(argv=process.argv.slice(2),{signal}={}) {
  const {positionals,values}=parseArgs({args:argv,allowPositionals:true,options:{
    instance:{type:'string'},config:{type:'string'},baseline:{type:'string'},skills:{type:'string'},
    repository:{type:'string'},project:{type:'string'},mode:{type:'string'},port:{type:'string'},request:{type:'string'},
    concurrency:{type:'string'},detach:{type:'boolean'},'no-recovery':{type:'boolean'},help:{type:'boolean'}}});
  const [command,arg]=positionals;
  const root=resolve(values.instance??arg??'.');
  const operation=async()=>{
  if (values.help || !command) return {commands:['bootstrap','upgrade','status','config','skill','project','wave','campaign','resume','pause','stop','dashboard','kdb','tune','host-memory']};
  if (command==='bootstrap') {
    const config=values.config?JSON.parse(await readFile(values.config,'utf8')):{};
    return bootstrap(arg,{...config,baseline:values.baseline??config.baseline,
      skills:values.skills??config.skills,repository:values.repository??config.repository});
  }
  const config=await readConfig(root);
  if(command==='host-memory') {
    const action=values.config?JSON.parse(await readFile(values.config,'utf8')):{action:'status'};
    if(action.action==='status')return (await import('../controller/host-memory.mjs'))
      .createHostMemorySupervisor(config.host_memory).measure();
    if(action.action!=='configure')throw new Error('Unknown host memory action');
    const verifyInstanceStopped=async()=>{
      const {ownerStopped}=await import('../controller/lifecycle.mjs');
      const ledger=new State(join(root,'state.sqlite'));
      try {
        if(ledger.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get()) {
          const rows=ledger.db.prepare("SELECT value FROM control WHERE key IN ('owner','skill-writer')").all();
          for(const row of rows) {
            const owner=JSON.parse(row.value);
            if(owner&&!await ownerStopped(owner))return false;
          }
        }
        for(const row of ledger.db.prepare('SELECT owner FROM attempts').all())
          if(!await ownerStopped(row.owner))return false;
        return true;
      }finally{ledger.close();}
    };
    return (await import('./host-memory.mjs')).configureHostMemory(config.host_memory,{verifyInstanceStopped});
  }
  if(values.detach && ['tune','kdb'].includes(command)) {
    const action=JSON.parse(await readFile(values.config,'utf8'));
    return (await import('./background.mjs')).startBackground(root,command,action);
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
    let staged,archive,activated=false;
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
          staged=join(root,'knowledge-next-'+randomUUID()+'.sqlite');
          const db=new DatabaseSync(input.baselineKDB,{readOnly:true});
          try{db.prepare('VACUUM INTO ?').run(staged);}finally{db.close();}
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
        try{await rename(staged,join(root,'knowledge.sqlite'));activated=true;}
        catch(error){await rename(archive,join(root,'knowledge.sqlite'));throw error;}
      },rollback:async()=>{
        if(activated){await rm(join(root,'knowledge.sqlite'));await rename(archive,join(root,'knowledge.sqlite'));}
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
    const state=new State(join(root,'state.sqlite'));
    try {
      if(state.project(spec.id)) {
        if(state.db.prepare('SELECT 1 FROM attempts WHERE project=? LIMIT 1').get(spec.id))
          throw new Error('Executed projects are frozen; clone under a new id');
        if(!spec.goal || !(spec.actor_seconds>0))throw new Error('Invalid project');
        state.db.prepare('UPDATE projects SET spec=? WHERE id=?').run(JSON.stringify(spec),spec.id);
      }else state.add(spec);
    }finally{state.close();}
    return updateConfig(root,current=>({...current,projects:[...current.projects.filter(p=>p.id!==spec.id),spec]}));
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
      const {requests}=await import('./background.mjs');
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
      return {controller,mode:config.mode,progress,wait:controlValue('wait'),memory:controlValue('memory'),requests:requests(state),projects:state.db.prepare("SELECT id,status,frozen,json_extract(spec,'$.actor_seconds') AS actor_seconds,json_extract(spec,'$.goal') AS goal FROM projects").all(),
        attempts:state.db.prepare('SELECT id,project,elapsed,outcome,created FROM attempts ORDER BY created DESC').all()};
    } finally {state.close();}
  }
  if (['wave','campaign','resume','pause','stop'].includes(command)) {
    const {operate,campaign}=await import('./runtime.mjs');
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
  return operation();
}
if(import.meta.main) {
  const abort=new AbortController();
  const stop=()=>abort.abort();process.on('SIGTERM',stop);process.on('SIGINT',stop);
  main(undefined,{signal:abort.signal}).then(value=>console.log(JSON.stringify(value))).catch(error=>{
    console.error(JSON.stringify({error:error.message}));process.exitCode=1;
  }).finally(()=>{process.off('SIGTERM',stop);process.off('SIGINT',stop);});
}
