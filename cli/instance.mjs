import { cp, mkdir, readFile, writeFile, realpath, stat, rename, rm } from 'node:fs/promises';
import { join, resolve, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { dirname, delimiter } from 'node:path';
import { randomUUID } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';

export async function readConfig(root) {
  root=await realpath(root);
  const config=JSON.parse(await readFile(join(root,'instance.json'),'utf8'));
  if (config.format !== 1) throw new Error('Unsupported instance; bootstrap a fresh instance');
  return {...config,root};
}
const within=(parent,child)=>{const rel=relative(parent,child);return rel==='' || (!rel.startsWith('..'+sep) && rel!=='..');};

export async function bootstrap(destination,{baseline,skills,environment,projects=[],repository,
  machinery=fileURLToPath(new URL('..',import.meta.url)),mode='phase1',actor_seconds=28800,skillSettings={},install=true,failed_recovery_rounds=5,recovery=true,concurrency=1,tune_every=8,...settings}={}) {
  destination=resolve(destination);machinery=await realpath(machinery);
  if (within(machinery,destination)) throw new Error('Instance must be outside its source machinery');
  if (!['phase1','phase2','inference'].includes(mode)) throw new Error('Unknown mode');
  if (!baseline || !skills || !environment) throw new Error('Baseline KDB, skills and environment configuration required');
  const {validateSkillMap}=await import('./skill-input.mjs');
  await validateSkillMap(resolve(skills),JSON.parse(await readFile(join(skills,'skill-map.json'),'utf8')));
  const basestat=await stat(baseline);
  if (!basestat.isFile()) throw new Error('Baseline KDB must be a SQLite file');
  await mkdir(destination,{recursive:false,mode:0o700});
  const source=new DatabaseSync(baseline,{readOnly:true});
  try {
    // SQLite VACUUM snapshots WAL-backed source data without copying an inconsistent file.
    source.prepare('VACUUM INTO ?').run(join(destination,'knowledge.sqlite'));
  } finally {source.close();}
  await mkdir(join(destination,'skills','versions'),{recursive:true});
  await cp(skills,join(destination,'skills','versions','baseline'),{recursive:true,errorOnExist:true});
  await mkdir(join(destination,'projects'));
  if (repository) await cp(repository,join(destination,'repository'),{recursive:true,
    filter:path=>!['node_modules','.venv','__pycache__'].includes(path.split(sep).at(-1))});
  await copyMachinery(machinery,join(destination,'machinery'),install);
  const config={...settings,format:1,revision:0,mode,baselineKDB:resolve(baseline),actor_seconds,failed_recovery_rounds,recovery,
    concurrency,tune_every,skills:{...skillSettings,activeVersion:'baseline',runtimeModule:'machinery/cli/evaluation.mjs'},environment,projects};
  await writeFile(join(destination,'instance.json'),JSON.stringify(config,null,2)+'\n',{mode:0o600});

  return config;
}

export async function updateConfig(root,change) {
  const current=await readConfig(root);
  const next=await change(current);
  if (next.format !== 1) throw new Error('Current format required');
  next.revision=(current.revision??0)+1;
  delete next.root;
  const {rename}=await import('node:fs/promises');
  const path=join(current.root,'instance.json');
  await writeFile(path+'.tmp',JSON.stringify(next,null,2)+'\n',{mode:0o600});
  await rename(path+'.tmp',path);
  return next;
}

async function copyMachinery(machinery,release,install=true) {
  await mkdir(release);
  for (const item of ['controller','cli','ascend_tools','web','templates','package.json','package-lock.json','npm-shrinkwrap.json','pyproject.toml']) {
    const from=join(machinery,item);
    try {await stat(from);} catch(error) {if(error.code==='ENOENT')continue;throw error;}
    await cp(from,join(release,item),{recursive:true,filter:path=>!['node_modules','__pycache__'].includes(path.split(sep).at(-1))});
  }
  await mkdir(join(release,'integrations'));
  for (const item of ['environments','kdb-extension.mjs']) {
    try {await cp(join(machinery,'integrations',item),join(release,'integrations',item),{recursive:true});}
    catch(error) {if(error.code!=='ENOENT')throw error;}
  }
  if(install) {
    try {await stat(join(release,'package.json'));}
    catch(error){if(error.code==='ENOENT')return;throw error;}
    await new Promise((ok,no)=>{
      const child=spawn('npm',['ci','--ignore-scripts','--prefix',release],{
        env:{...process.env,PATH:dirname(process.execPath)+delimiter+process.env.PATH},stdio:['ignore','ignore','pipe']});
      let error='';child.stderr.on('data',data=>error+=data);child.on('error',no);
      child.on('close',code=>code?no(new Error('Instance dependency installation failed: '+error)):ok());
    });
  }
}

export async function upgrade(root,machinery) {
  const config=await readConfig(root);
  const {State}=await import('../controller/state.mjs');
  const {ownerStopped,attemptStopped}=await import('../controller/lifecycle.mjs');
  const {pathToFileURL}=await import('node:url');
  const ledger=new State(join(root,'state.sqlite'));
  try {
    if(ledger.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get()) {
      const row=ledger.db.prepare("SELECT value FROM control WHERE key='owner'").get();
      const owner=row && JSON.parse(row.value);
      if(owner && !await ownerStopped(owner))throw new Error('Stop the controller before upgrading');
    }
    const pending=ledger.db.prepare('SELECT * FROM attempts WHERE outcome IS NULL').all();
    const module=pending.length?await import(pathToFileURL(resolve(root,config.environment.module))):null;
    for(const attempt of pending) {
      const environment=await module.createEnvironment(config.environment.options,{
        instancePath:join(root,'attempts',attempt.id),event:(type,data)=>ledger.event(attempt.id,type,data),exclude:async(_label,fn)=>fn()});
      if(!await ownerStopped(attempt.owner) || !await attemptStopped(ledger,attempt,environment))
        throw new Error('Upgrade requires confirmed stopped local and external executions');
    }
    let receipt;
    try{receipt=JSON.parse(await readFile(join(root,'knowledge-service.json'),'utf8'));}
    catch(error){if(error.code!=='ENOENT')throw error;}
    if(receipt && !await ownerStopped(receipt.owner)) {
      if(!receipt.owner?.start || !receipt.owner?.boot)throw new Error('Cannot verify KDB process identity');
      process.kill(receipt.owner.pid,'SIGTERM');
      const {setTimeout:delay}=await import('node:timers/promises');
      const end=performance.now()+5000;
      while(!await ownerStopped(receipt.owner) && performance.now()<end)await delay(50);
      if(!await ownerStopped(receipt.owner))throw new Error('KDB process has not stopped');
    }
    const staged=join(root,'machinery-next-'+randomUUID());
    try {
      await copyMachinery(await realpath(machinery),staged);
      await mkdir(join(root,'archives'),{recursive:true});
      const archive=join(root,'archives','machinery-'+Date.now());
      await rename(join(root,'machinery'),archive);
      try {await rename(staged,join(root,'machinery'));}
      catch(error){await rename(archive,join(root,'machinery'));throw error;}
    }finally{await rm(staged,{recursive:true,force:true});}
    return {upgraded:true,format:1};
  }finally{ledger.close();}
}
