import { readFileSync, writeFileSync, renameSync, rmSync } from 'node:fs';
import { cp, mkdir, readFile, writeFile, realpath, stat, rename, rm } from 'node:fs/promises';
import { join, resolve, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { dirname, delimiter } from 'node:path';
import { randomUUID,createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import { State, validateAcceptance } from '../controller/state.mjs';
import { syncCampaignPlan } from '../controller/progress.mjs';
import { CampaignStorage } from '../controller/storage.mjs';
import {assertPackAvailable} from './storage.mjs';
import {checkBundle,validateBundle,withBundleBudget} from './project-bundle.mjs';

export async function readConfig(root) {
  root=await realpath(root);
  const config=JSON.parse(await readFile(join(root,'instance.json'),'utf8'));
  if (config.format !== 1) throw new Error('Unsupported instance; bootstrap a fresh instance');
  return {...config,root};
}
async function fileDigest(path,{signal}={}) {
  const digest=createHash('sha256');for await(const chunk of createReadStream(path,{signal}))digest.update(chunk);return digest.digest('hex');
}
async function initializationCopy(from,to,options={},retain=false) {
  const filter=options.filter;
  await cp(from,to,{...options,force:!retain,filter:async path=>{
    if(filter && !await filter(path))return false;
    if(retain && (await stat(path)).isFile()) {
      const target=join(to,relative(from,path));
      try {
        if(await fileDigest(path)!==await fileDigest(target))
          throw new Error('Retained initialization file differs: '+target);
      }catch(error){if(error.code!=='ENOENT')throw error;}
    }
    return true;
  }});
}
const within=(parent,child)=>{const rel=relative(parent,child);return rel==='' || (!rel.startsWith('..'+sep) && rel!=='..');};

export async function bootstrap(destination,{baseline,skills,environment,projects=[],repository,
  machinery=fileURLToPath(new URL('..',import.meta.url)),mode='phase1',actor_seconds=28800,skillSettings={},install=true,failed_recovery_rounds=5,recovery=true,concurrency=1,tune_every=8,signal,initialization,...settings}={}) {
  signal?.throwIfAborted();
  destination=resolve(destination);machinery=await realpath(machinery);
  if (within(machinery,destination)) throw new Error('Instance must be outside its source machinery');
  if (!['phase1','phase2','inference'].includes(mode)) throw new Error('Unknown mode');
  if (!baseline || !skills || !environment) throw new Error('Baseline KDB, skills and environment configuration required');
  const {validateSkillMap}=await import('./skill-input.mjs');
  const skillMap=await validateSkillMap(resolve(skills),JSON.parse(await readFile(join(skills,'skill-map.json'),'utf8')));
  const basestat=await stat(baseline);
  if (!basestat.isFile()) throw new Error('Baseline KDB must be a SQLite file');
  const continuing=initialization?.resume===true;
  if(initialization && typeof initialization.setup!=='function')throw new Error('Managed initialization requires setup supervision');
  if(continuing) {
    try {await readConfig(destination);throw new Error('Cannot initialize a ready instance');}
    catch(error){if(error.code!=='ENOENT')throw error;}
    let hasState=false;try{await stat(join(destination,'state.sqlite'));hasState=true;}catch(error){if(error.code!=='ENOENT')throw error;}
    if(hasState) {
      const existing=new DatabaseSync(join(destination,'state.sqlite'),{readOnly:true});
      try{if(existing.prepare('SELECT 1 FROM attempts LIMIT 1').get())throw new Error('Cannot initialize existing actor work');}finally{existing.close();}
    }
  }
  await mkdir(destination,{recursive:continuing,mode:0o700});
  const source=new DatabaseSync(baseline,{readOnly:true});
  let legacy=false;
  try {
    legacy=source.prepare('PRAGMA table_info(chunks)').all().some(row=>row.name==='vector_json');
    // SQLite VACUUM snapshots WAL-backed source data without copying an inconsistent file.
    if(!legacy) {
      let retained=false;
      if(continuing)try{await stat(join(destination,'knowledge.sqlite'));retained=true;}catch(error){if(error.code!=='ENOENT')throw error;}
      if(!retained) {
        const temporary=join(destination,'knowledge-'+randomUUID()+'.tmp');
        source.prepare('VACUUM INTO ?').run(temporary);
        await rename(temporary,join(destination,'knowledge.sqlite'));
      }
    }
    if(legacy && initialization)throw new Error('Managed evaluation requires a modern KDB snapshot');
  } finally {source.close();}
  if(legacy)await new Promise((ok,no)=>{
    const child=spawn(settings.python??'python',['-m','ascend_tools.kdb_import','--source',resolve(baseline),
      '--destination',join(destination,'knowledge.sqlite')],{signal,env:{...process.env,PYTHONPATH:machinery},stdio:['ignore','ignore','pipe']});
    let error='';child.stderr.setEncoding('utf8');child.stderr.on('data',data=>{error=(error+data).slice(-65536);});
    child.on('error',no);child.on('close',code=>code?no(new Error('Baseline KDB import failed: '+error)):ok());
  });
  await mkdir(join(destination,'skills','versions'),{recursive:true});
  await initializationCopy(skills,join(destination,'skills','versions','baseline'),{recursive:true,errorOnExist:!continuing,
    filter:path=>!initialization || path!==join(skills,'skill-map.json')},continuing);
  const mapPath=join(destination,'skills','versions','baseline','skill-map.json'),mapContent=JSON.stringify(skillMap,null,2)+'\n';
  if(continuing)try {
    if(await readFile(mapPath,'utf8')!==mapContent)throw new Error('Retained initialization skill map differs');
  }catch(error){if(error.code!=='ENOENT')throw error;}
  await writeFile(mapPath,mapContent);
  await mkdir(join(destination,'projects'),{recursive:continuing});
  if (repository) await cp(repository,join(destination,'repository'),{recursive:true,
    filter:path=>!['node_modules','.venv','__pycache__','.env'].includes(path.split(sep).at(-1)) && !path.split(sep).at(-1).startsWith('.env.')});
  const importedProjects=[];
  for(const project of projects)importedProjects.push(await importProject(destination,{actor_seconds,...project},
    {continuing:!!initialization,signal,localExecution:settings.local_execution}));
  projects=importedProjects;
  await copyMachinery(machinery,join(destination,'machinery'),install,settings.python,signal,initialization);
  signal?.throwIfAborted();
  const config={...settings,format:1,revision:0,mode,baselineKDB:resolve(baseline),actor_seconds,failed_recovery_rounds,recovery,
    concurrency,tune_every,skills:{...skillSettings,activeVersion:'baseline',runtimeModule:'machinery/cli/evaluation.mjs'},environment,projects};
  const ledger=new State(join(destination,'state.sqlite'));
  try {
    syncCampaignPlan(ledger,config);
    for(const project of projects)if(!ledger.project(project.id))ledger.add(project);
    const storage=new CampaignStorage(ledger,destination);
    const campaign=await storage.initialize({target:config.target??null});
    config.storage={format:1,campaign_id:campaign.id};
    await initialization?.phase?.('storage');
    await (await import('../controller/storage-knowledge.mjs')).initializeStorageKnowledge(
      {...config,root:destination},ledger,{serialized:true,signal});
    await initialization?.phase?.('initialized');
  }finally {ledger.close();}
  await writeFile(join(destination,'instance.json'),JSON.stringify(config,null,2)+'\n',{mode:0o600});

  return config;
}

export async function updateConfig(root,change,{activate,rollback}={}) {
  const ledger=new State(join(root,'state.sqlite'));
  let staged;
  try {
    ledger.db.exec('BEGIN IMMEDIATE');
    await assertPackAvailable(ledger);
    const current=await readConfig(root),next=await change(current,ledger);
    syncCampaignPlan(ledger,next);
    if(next.format!==1)throw new Error('Current format required');
    next.revision=(current.revision??0)+1;delete next.root;
    const path=join(current.root,'instance.json');staged=path+'.'+randomUUID()+'.tmp';
    await writeFile(staged,JSON.stringify(next,null,2)+'\n',{mode:0o600});
    await activate?.();
    try {await rename(staged,path);}catch(error){await rollback?.();throw error;}
    ledger.db.exec('COMMIT');
    return next;
  } catch(error) {
    if(ledger.db.isTransaction)ledger.db.exec('ROLLBACK');
    throw error;
  } finally {ledger.close();if(staged)await rm(staged,{force:true});}
}

// Scheduler callbacks must not yield while holding SQLite's write lock: its
// elapsed timers write through another connection on this same event loop.
export async function updateConfigImmediate(root,change) {
  root=await realpath(root);
  const ledger=new State(join(root,'state.sqlite'));
  let staged;
  const pack=()=>ledger.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get()
    ?ledger.db.prepare("SELECT value FROM control WHERE key='pack-capture'").get()?.value??null:null;
  try {
    const token=pack();
    if(token && !await (await import('../controller/lifecycle.mjs')).ownerStopped(token))
      throw new Error('Pack capture active; writer admission refused');
    ledger.db.exec('BEGIN IMMEDIATE');
    if(pack()!==token)throw new Error('Pack capture changed; refresh before publishing');
    const path=join(root,'instance.json'),current={...JSON.parse(readFileSync(path,'utf8')),root};
    const next=change(current,ledger);
    if(next?.then)throw new Error('Immediate config callback must be synchronous');
    if(current.format!==1 || next.format!==1)throw new Error('Current format required');
    syncCampaignPlan(ledger,next);
    next.revision=(current.revision??0)+1;delete next.root;
    staged=path+'.'+randomUUID()+'.tmp';
    writeFileSync(staged,JSON.stringify(next,null,2)+'\n',{mode:0o600});
    renameSync(staged,path);ledger.db.exec('COMMIT');return next;
  }catch(error){if(ledger.db.isTransaction)ledger.db.exec('ROLLBACK');throw error;}
  finally{ledger.close();if(staged)rmSync(staged,{force:true});}
}

async function copyMachinery(machinery,release,install=true,python='python',signal,initialization) {
  signal?.throwIfAborted();
  await mkdir(release,{recursive:initialization?.resume===true});
  for (const item of ['controller','cli','ascend_tools','web','templates','scripts','package.json','package-lock.json','npm-shrinkwrap.json','pyproject.toml']) {
    signal?.throwIfAborted();
    const from=join(machinery,item);
    try {await stat(from);} catch(error) {if(error.code==='ENOENT')continue;throw error;}
    await initializationCopy(from,join(release,item),{recursive:true,filter:path=>!['node_modules','__pycache__'].includes(path.split(sep).at(-1))},initialization?.resume);
  }
  await mkdir(join(release,'integrations'),{recursive:initialization?.resume===true});
  for (const item of ['environments','kdb-extension.mjs']) {
    try {await initializationCopy(join(machinery,'integrations',item),join(release,'integrations',item),{recursive:true},initialization?.resume);}
    catch(error) {if(error.code!=='ENOENT')throw error;}
  }
  if(install) {
    try {await stat(join(release,'package.json'));}
    catch(error){if(error.code==='ENOENT')return;throw error;}
    if(initialization) {
      await initialization.setup('restic',python,['-m','ascend_tools.restic_install','--install'],release,{PYTHONPATH:release});
      await initialization.setup('dependencies','npm',['ci','--ignore-scripts','--prefix',release],release,
        {PATH:dirname(process.execPath)+delimiter+process.env.PATH});
      return;
    }
    await new Promise((ok,no)=>{
      const child=spawn(python,['-m','ascend_tools.restic_install','--install'],{
        signal,cwd:release,env:{...process.env,PYTHONPATH:release},stdio:['ignore','ignore','inherit']});
      child.on('error',no);
      child.on('close',code=>code?no(new Error('Restic installation failed')):ok());
    });
    await new Promise((ok,no)=>{
      const child=spawn('npm',['ci','--ignore-scripts','--prefix',release],{
        signal,env:{...process.env,PATH:dirname(process.execPath)+delimiter+process.env.PATH},stdio:['ignore','ignore','pipe']});
      let error='';child.stderr.on('data',data=>error+=data);child.on('error',no);
      child.on('close',code=>code?no(new Error('Instance dependency installation failed: '+error)):ok());
    });
  }
}

export async function upgrade(root,machinery) {
  const config=await readConfig(root);
  machinery=await realpath(machinery);
  const {ownerStopped,attemptStopped}=await import('../controller/lifecycle.mjs');
  const {pathToFileURL}=await import('node:url');
  const ledger=new State(join(root,'state.sqlite'));
  const staged=join(root,'machinery-next-'+randomUUID());
  async function stoppedOwners() {
    await assertPackAvailable(ledger);
    if(ledger.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get()) {
      for(const row of ledger.db.prepare("SELECT value FROM control WHERE key IN ('owner','skill-writer','supervisor-owner')").all()) {
        const owner=JSON.parse(row.value);
        if(owner && !await ownerStopped(owner))throw new Error('Stop the controller and skill writer before upgrading');
      }
    }
    for(const attempt of ledger.db.prepare('SELECT owner FROM attempts').all())
      if(!await ownerStopped(attempt.owner))throw new Error('Upgrade requires confirmed stopped local workers');
    if(ledger.db.prepare("SELECT 1 FROM sqlite_master WHERE name='requests'").get())
      for(const request of ledger.db.prepare("SELECT status,owner FROM requests WHERE status IN ('queued','running')").all())
        if(request.status==='queued' || !await ownerStopped(request.owner))
          throw new Error('Stop background requests before upgrading');
  }
  try {
    await stoppedOwners();
    await copyMachinery(machinery,staged,true,config.python);
    const pending=ledger.db.prepare('SELECT * FROM attempts WHERE outcome IS NULL').all();
    const installed=resolve(root,'machinery'),configured=resolve(root,config.environment.module);
    // Inspect with the installed dependencies of the exact staged replacement.
    // Older packaged plugins need not already implement the eligibility hook.
    const plugin=within(installed,configured)?resolve(staged,relative(installed,configured)):configured;
    const module=pending.length?await import(pathToFileURL(plugin)):null;
    for(const attempt of pending) {
      const environment=await module.createEnvironment({...config.environment.options,localExecution:config.local_execution},{
        instancePath:join(root,'attempts',attempt.id),event:async()=>{},exclude:async(_label,fn)=>fn()});
      const safe=environment.canUpgrade
        ? await environment.canUpgrade({attempt,events:ledger.events(attempt.id)})
        : await attemptStopped(ledger,attempt,environment);
      if(!safe)throw new Error('Upgrade requires confirmed stopped local and external executions');
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
    await mkdir(join(root,'archives'),{recursive:true});
    const archive=join(root,'archives','machinery-'+Date.now());
    ledger.db.exec('BEGIN IMMEDIATE');
    await stoppedOwners();
    await rename(join(root,'machinery'),archive);
    try{await rename(staged,join(root,'machinery'));}
    catch(error){await rename(archive,join(root,'machinery'));throw error;}
    ledger.db.exec('COMMIT');
    return {upgraded:true,format:1};
  }finally{
    if(ledger.db.isTransaction)ledger.db.exec('ROLLBACK');
    ledger.close();await rm(staged,{recursive:true,force:true});
  }
}

export async function importProject(root,spec,options={}) {
  return withBundleBudget(spec,options,budget=>importProjectFiles(root,spec,{...options,signal:budget.signal,budget}));
}
async function importProjectFiles(root,spec,{continuing=false,signal,localExecution,budget}={}) {
  validateAcceptance(spec);
  if(!spec.source || !spec.goal || !(spec.actor_seconds>0) || typeof spec.verifier?.command!=='string' || !spec.verifier.files?.length)
    throw new Error('Project requires goal, source folder, elapsed allocation and frozen verifier command/files');
  if(!/^[a-zA-Z0-9_-]+$/.test(spec.id))throw new Error('Project id must be a folder-safe identifier');
  signal?.throwIfAborted();await checkBundle(spec);
  const source=await realpath(spec.source),files=spec.verifier.files;
  const hash=path=>fileDigest(path,{signal});
  const expected={};
  for(const file of files) {
    signal?.throwIfAborted();
    const absolute=resolve(file),actual=await hash(absolute);
    if(spec.verifier.hashes&&spec.verifier.hashes[absolute]!==actual)throw new Error('Frozen verifier changed before import');
    expected[absolute]=actual;
  }
  if(!(await stat(source)).isDirectory())throw new Error('Project source must be a directory');
  const own=join(resolve(root),'projects',spec.id);
  if(within(own,source) && files.every(path=>within(own,resolve(path)))) {
    const retained={...spec,verifier:{...spec.verifier,hashes:expected}};
    await validateBundle(retained,join(own,'bundle-validation'),{signal,localExecution,budget});return retained;
  }
  let common=spec.bundle?await realpath(spec.bundle.root):source;
  for(const path of files)while(!within(common,resolve(path)))common=dirname(common);
  const tree=join(own,continuing?'inputs-managed':'inputs-'+randomUUID());await mkdir(tree,{recursive:true});
  const destination=join(tree,relative(common,source));
  const filter=path=>{signal?.throwIfAborted();return !['.git','node_modules','.venv','__pycache__','.env'].includes(path.split(sep).at(-1)) && !path.split(sep).at(-1).startsWith('.env.');};
  await initializationCopy(source,destination,{recursive:true,verbatimSymlinks:true,filter},continuing);
  const commands={command:spec.verifier.command,performance_command:spec.verifier.performance_command};
  let construction_command=spec.bundle?.construction_command;
  const copied=[],hashes={};
  for(const path of files) {
    signal?.throwIfAborted();
    const absolute=resolve(path),target=join(tree,relative(common,absolute));
    await mkdir(dirname(target),{recursive:true});await cp(absolute,target,{force:!continuing});copied.push(target);
    if(await hash(target)!==expected[absolute])throw new Error('Frozen verifier changed during import');
    hashes[target]=expected[absolute];
    if(construction_command)construction_command=construction_command.split(absolute).join(target);
    for(const key of Object.keys(commands))if(commands[key])commands[key]=commands[key].split(absolute).join(relative(destination,target));
  }
  for(const key of Object.keys(commands))if(commands[key]) {
    commands[key]=commands[key].split(source).join('.');
    if(spec.bundle)commands[key]=commands[key].split(common+sep).join(relative(destination,tree)+sep);
  }
  const result={...spec,source:destination};
  if(spec.verifier)result.verifier={...spec.verifier,...commands,files:copied,hashes};
  if(spec.bundle) {
    construction_command=construction_command.split(common+sep).join(tree+sep);
    const relocate=path=>join(tree,relative(common,path));
    result.bundle={...spec.bundle,root:tree,reference:relocate(spec.bundle.reference),
      negative_controls:spec.bundle.negative_controls.map(relocate),construction_command};
    await validateBundle(result,join(own,'bundle-validation'),{signal,localExecution,budget});
  }
  return result;
}
