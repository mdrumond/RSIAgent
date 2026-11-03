import { cp, mkdir, readFile, writeFile, realpath, stat } from 'node:fs/promises';
import { join, resolve, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { dirname, delimiter } from 'node:path';
import { DatabaseSync } from 'node:sqlite';

export async function readConfig(root) {
  root=await realpath(root);
  const config=JSON.parse(await readFile(join(root,'instance.json'),'utf8'));
  if (config.format !== 1) throw new Error('Unsupported instance; bootstrap a fresh instance');
  return {...config,root};
}
const within=(parent,child)=>{const rel=relative(parent,child);return rel==='' || (!rel.startsWith('..'+sep) && rel!=='..');};

export async function bootstrap(destination,{baseline,skills,environment,projects=[],repository,
  machinery=fileURLToPath(new URL('..',import.meta.url)),mode='phase1',actor_seconds=28800,skillSettings={},install=true,...settings}={}) {
  destination=resolve(destination);machinery=await realpath(machinery);
  if (within(machinery,destination)) throw new Error('Instance must be outside its source machinery');
  if (!['phase1','phase2','inference'].includes(mode)) throw new Error('Unknown mode');
  if (!baseline || !skills || !environment) throw new Error('Baseline KDB, skills and environment configuration required');
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
  const release=join(destination,'machinery');
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
  const config={...settings,format:1,mode,baselineKDB:resolve(baseline),actor_seconds,failed_recovery_rounds:5,recovery:true,
    concurrency:1,tune_every:8,skills:{...skillSettings,activeVersion:'baseline',runtimeModule:'machinery/cli/evaluation.mjs'},environment,projects};
  await writeFile(join(destination,'instance.json'),JSON.stringify(config,null,2)+'\n',{mode:0o600});
  if(install) {
    try {await stat(join(release,'package.json'));}
    catch(error){if(error.code==='ENOENT')return config;throw error;}
    await new Promise((ok,no)=>{
      const child=spawn('npm',['ci','--ignore-scripts','--prefix',release],{
        env:{...process.env,PATH:dirname(process.execPath)+delimiter+process.env.PATH},stdio:['ignore','ignore','pipe']});
      let error='';child.stderr.on('data',data=>error+=data);child.on('error',no);
      child.on('close',code=>code?no(new Error('Instance dependency installation failed: '+error)):ok());
    });
  }
  return config;
}

export async function updateConfig(root,change) {
  const current=await readConfig(root);
  const next=await change(current);
  if (next.format !== 1) throw new Error('Current format required');
  delete next.root;
  const {rename}=await import('node:fs/promises');
  const path=join(current.root,'instance.json');
  await writeFile(path+'.tmp',JSON.stringify(next,null,2)+'\n',{mode:0o600});
  await rename(path+'.tmp',path);
  return next;
}
