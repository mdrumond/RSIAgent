#!/usr/bin/env node
import { parseArgs } from 'node:util';
import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { bootstrap, readConfig, updateConfig } from './instance.mjs';
import { State } from '../controller/state.mjs';

export async function main(argv=process.argv.slice(2)) {
  const {positionals,values}=parseArgs({args:argv,allowPositionals:true,options:{
    instance:{type:'string'},config:{type:'string'},baseline:{type:'string'},skills:{type:'string'},
    repository:{type:'string'},project:{type:'string'},mode:{type:'string'},port:{type:'string'},
    concurrency:{type:'string'},detach:{type:'boolean'},'no-recovery':{type:'boolean'},help:{type:'boolean'}}});
  const [command,arg]=positionals;
  if (values.help || !command) return {commands:['bootstrap','status','config','project','wave','resume','stop','dashboard','kdb','tune']};
  if (command==='bootstrap') {
    const config=values.config?JSON.parse(await readFile(values.config,'utf8')):{};
    return bootstrap(arg,{...config,baseline:values.baseline??config.baseline,
      skills:values.skills??config.skills,repository:values.repository??config.repository});
  }
  const root=resolve(values.instance??arg??'.');
  const config=await readConfig(root);
  if (command==='config') {
    if (!values.config)return config;
    const input=JSON.parse(await readFile(values.config,'utf8'));
    return updateConfig(root,current=>({...current,...input,format:1}));
  }
  if(command==='project') {
    if(!values.project) return config.projects;
    const spec=JSON.parse(await readFile(values.project,'utf8'));
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
    return {path,entries:(await readdir(path,{withFileTypes:true})).filter(e=>e.isDirectory()).map(e=>e.name)};
  }
  if(command==='status') {
    const state=new State(join(root,'state.sqlite'));
    try {
      return {mode:config.mode,projects:state.db.prepare('SELECT id,status,frozen FROM projects').all(),
        attempts:state.db.prepare('SELECT id,project,elapsed,outcome,created FROM attempts ORDER BY created DESC').all()};
    } finally {state.close();}
  }
  if (['wave','resume','stop'].includes(command)) {
    const {operate}=await import('./runtime.mjs');
    return operate(command,config,{...values,mode:values.mode??config.mode});
  }
  if(command==='dashboard') {
    const {spawn}=await import('node:child_process');
    const child=spawn(config.python??'python',['-m','ascend_tools.dashboard','--instance',root,
      '--port',values.port??'46731'],{stdio:'inherit',env:{...process.env,
        PYTHONPATH:resolve(new URL('..',import.meta.url).pathname)}});
    return new Promise((res,rej)=>{child.on('error',rej);child.on('close',code=>code?rej(new Error('Dashboard failed')):res({stopped:true}));});
  }
  if(command==='kdb' || command==='tune') {
    const module=command==='kdb'?'./knowledge.mjs':'./skills.mjs';
    return (await import(module)).run(config,values);
  }
  throw new Error(`Unknown command ${command}`);
}
if(process.argv[1] && import.meta.url===pathToFileURL(resolve(process.argv[1])).href) {
  main().then(value=>console.log(JSON.stringify(value))).catch(error=>{
    console.error(JSON.stringify({error:error.message}));process.exitCode=1;
  });
}
