import {readFile,realpath,stat} from 'node:fs/promises';
import {join,resolve,relative,isAbsolute} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawn} from 'node:child_process';
import {setTimeout as delay} from 'node:timers/promises';
import {State} from '../controller/state.mjs';
import {CampaignStorage} from '../controller/storage.mjs';
import {ownerStopped,failureSettled,processIdentity} from '../controller/lifecycle.mjs';
import {settled} from './evaluation.mjs';
import {withSkillLock} from './evolution-lock.mjs';

const read=async path=>{try{return JSON.parse(await readFile(path,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;return null;}};
const inside=(root,path)=>{const r=relative(root,path);return r!==''&&!r.startsWith('..')&&!isAbsolute(r);};
const exists=(state)=>Boolean(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='evaluation_archives'").get());
const rows=state=>exists(state)?state.db.prepare('SELECT * FROM evaluation_archives ORDER BY rowid').all():[];
function schema(state){state.db.exec(`CREATE TABLE IF NOT EXISTS evaluation_archives (
  root TEXT PRIMARY KEY,origin TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'pending',
  worker TEXT,phase TEXT,error TEXT,updated TEXT NOT NULL)`);}

export async function enqueueCompletedEvaluation(config,evolution) {
  if(!config.skills?.archiveEvaluations||!evolution?.lastEvaluationRoot)return false;
  const root=await realpath(config.root),evaluation=await realpath(evolution.lastEvaluationRoot);
  if(!inside(root,evaluation))throw new Error('Evaluation archive escaped campaign root');
  const round=await read(join(evaluation,'round.json'));
  if(round?.kind==='micro'||round?.status!=='completed')return false;
  const origin=[...(evolution.history??[])].reverse().find(task=>task.phase==='phase2'&&task.kind==='original'&&task.completed);
  if(!origin?.id)throw new Error('Completed evaluation requires originating Phase 2 evidence');
  const state=new State(join(root,'state.sqlite'));
  try{schema(state);return state.db.prepare('INSERT OR IGNORE INTO evaluation_archives(root,origin,updated) VALUES (?,?,?)')
    .run(evaluation,origin.id,new Date().toISOString()).changes>0;}finally{state.close();}
}

export async function evaluationArchiveStatus(config,evolution) {
  const state=new State(join(config.root,'state.sqlite'));
  try {
    const all=rows(state),pending=all.filter(row=>row.status!=='complete');
    const root=evolution?.lastEvaluationRoot;
    const untracked=Boolean(config.skills?.archiveEvaluations&&root&&!all.some(row=>row.root===resolve(root))&&
      (await read(join(root,'round.json')))?.status==='completed'&&(await read(join(root,'round.json')))?.kind!=='micro');
    return {pending:pending.length,complete:all.length-pending.length,untracked,
      hasWork:Boolean(config.skills?.archiveEvaluations&&(pending.length||untracked)),
      error:pending.find(row=>row.error)?.error??null};
  }finally{state.close();}
}

function groupStopped(group){try{process.kill(-group,0);return false;}catch(e){if(e.code==='ESRCH')return true;throw e;}}
export async function evaluationArchivesSettled(config) {
  const state=new State(join(config.root,'state.sqlite'));
  try{for(const row of rows(state)){
    if(row.phase==='dispatching')return false;
    if(row.worker){const worker=JSON.parse(row.worker);if(!await ownerStopped(worker)||!groupStopped(worker.pid))return false;}
  }return true;}catch{return false;}finally{state.close();}
}

async function closedInstance(path,sample) {
  const config=await read(join(path,'instance.json'));
  if(!config)throw new Error('Evaluation instance configuration missing');
  if(!await settled({instanceRoot:path}))throw new Error('Evaluation execution is not closed');
  const state=new State(join(path,'state.sqlite'));
  try {
    const attempts=state.db.prepare('SELECT * FROM attempts').all();
    if(sample&&attempts.length!==1)throw new Error('Evaluation sample is incomplete');
    for(const attempt of attempts)if(!attempt.outcome||!failureSettled(state,attempt.id)||
      !state.queryEvents(attempt.id,{types:['postprocess-completed'],limit:1}).length)
      throw new Error('Evaluation sample postprocessing is incomplete');
    if(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get())
      for(const row of state.db.prepare("SELECT value FROM control WHERE key IN ('owner','skill-writer','memory-writer','storage-maintenance','knowledge-startup','pack-capture')").all()) {
        const owner=JSON.parse(row.value);if(owner&&!await ownerStopped(owner))throw new Error('Evaluation writer remains live');
      }
    if(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='requests'").get())
      for(const row of state.db.prepare('SELECT status,owner FROM requests').all())
        if(['queued','running'].includes(row.status)||(row.owner&&!await ownerStopped(row.owner)))throw new Error('Evaluation storage request remains live');
    const receipt=await read(join(path,'knowledge-service.json'));
    if(receipt&&!await ownerStopped(receipt.owner))throw new Error('Evaluation KDB service remains live');
  }finally{state.close();}
}

async function scope(config,job,state) {
  const root=await realpath(config.root),evaluation=await realpath(job.root);
  if(!inside(root,evaluation))throw new Error('Evaluation archive escaped campaign root');
  const round=await read(join(evaluation,'round.json'));
  if(round?.status!=='completed'||round.runningAt!==undefined||!round.samples?.length||round.samples.some(s=>!s.result))
    throw new Error('Only completed comparisons can be archived');
  const origin=state.attempt(job.origin);
  if(!origin?.outcome||!await ownerStopped(origin.owner)||!failureSettled(state,origin.id)||
    !state.queryEvents(origin.id,{types:['postprocess-completed'],limit:1}).length)
    throw new Error('Originating project is not closed and postprocessed');
  const children=await Promise.all(round.samples.map(s=>realpath(s.instanceRoot)));
  if(new Set(children).size!==children.length||children.some(p=>!inside(evaluation,p)||p===join(evaluation,'inputs')))
    throw new Error('Evaluation sample escaped its comparison');
  for(const child of children)await closedInstance(child,true);
  const inputs=await realpath(join(evaluation,'inputs'));
  if(!inside(evaluation,inputs))throw new Error('Frozen evaluation inputs escaped comparison');
  await closedInstance(inputs,false);
  const files=[...children,inputs].map(p=>join(p,'knowledge.sqlite'));
  for(const path of files) {
    try {
      const actual=await realpath(path),info=await stat(path);
      if(actual!==path||!inside(evaluation,actual)||!info.isFile()||info.nlink!==1)throw new Error('Evaluation KDB is aliased or escaped comparison');
    } catch(error) {
      if(error.code!=='ENOENT')throw error;
      const archived=state.db.prepare('SELECT encoding FROM storage_artifacts WHERE path=?').get(relative(root,path));
      if(archived?.encoding!=='restic')throw new Error('Evaluation KDB missing without published archive');
    }
  }
  return {files,origin};
}

// Native parent Storage/Restic owns the repository, descriptors, validation and restore.
const maintenance=String.raw`
import fcntl,json,signal,sqlite3,sys
from pathlib import Path
from ascend_tools.storage_cli import Storage,execute
from ascend_tools.storage_archive import finalize_source

def stop(_signal,_frame): raise KeyboardInterrupt('Evaluation archive cancelled')
signal.signal(signal.SIGTERM,stop)
p=json.load(sys.stdin);root=Path(p['root'])
with (root/'.storage-maintenance.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if p['action']=='checkpoint':
  for name in p['paths']:
   with sqlite3.connect(name) as db:
    if db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0]:raise RuntimeError('KDB checkpoint is busy')
 elif p['action']=='compress':
  execute(root,dict(action='compress',kind='evaluation-kdb',role='evaluation',project=p['project'],delete_source=True,limit=1000))
 elif p['action']=='finalize':
  storage=Storage(root)
  try:
   for artifact in p['artifacts']:
    row,=storage.rows(dict(artifact=artifact))
    if row['encoding']!='restic':raise RuntimeError('Evaluation archive descriptor not published')
    storage.quiescent(row);storage.verify(row)
    if storage.path(row['path']).exists():finalize_source(storage.manifest(row))
  finally:storage.db.close()
print('{}')
`;

async function invoke(config,state,job,payload,signal) {
  signal?.throwIfAborted();
  state.db.prepare("UPDATE evaluation_archives SET phase='dispatching',worker=NULL WHERE root=?").run(job.root);
  const child=spawn(config.python??'python',['-c',maintenance],{detached:true,
    env:{...process.env,PYTHONPATH:fileURLToPath(new URL('..',import.meta.url))},stdio:['pipe','ignore','pipe']});
  let error='';child.stderr.on('data',chunk=>{error=(error+chunk).slice(-65536);});
  child.stdin.on('error',()=>{});child.stdin.end(JSON.stringify({root:config.root,...payload}));
  const closed=new Promise(ok=>{child.once('error',error=>ok({error}));child.once('close',code=>ok(code));});
  if(!child.pid){
    const outcome=await closed;state.db.prepare('UPDATE evaluation_archives SET phase=NULL,worker=NULL WHERE root=?').run(job.root);
    throw outcome.error??new Error('Evaluation archive worker was not spawned');
  }
  let owner;
  try{owner=await processIdentity(child.pid);}catch(error){
    if(error.code!=='ENOENT'&&error.code!=='ESRCH')throw error;
    const code=await closed;
    if(!groupStopped(child.pid))throw new Error('Evaluation archive descendants remain unresolved');
    state.db.prepare('UPDATE evaluation_archives SET phase=NULL,worker=NULL WHERE root=?').run(job.root);
    signal?.throwIfAborted();if(code)throw new Error(error||`Archive exited ${code}`);return;
  }
  state.db.prepare("UPDATE evaluation_archives SET phase='archiving',worker=? WHERE root=?").run(JSON.stringify(owner),job.root);
  const abort=()=>{try{process.kill(-child.pid,'SIGTERM');}catch(e){if(e.code!=='ESRCH')throw e;}};
  signal?.addEventListener('abort',abort,{once:true});if(signal?.aborted)abort();
  let code;
  try{code=await closed;}finally{signal?.removeEventListener('abort',abort);}
  const end=performance.now()+10000;
  while(!groupStopped(child.pid)&&performance.now()<end)await delay(25);
  if(!groupStopped(child.pid))throw new Error('Evaluation archive descendants remain unresolved');
  state.db.prepare('UPDATE evaluation_archives SET phase=NULL,worker=NULL WHERE root=?').run(job.root);
  signal?.throwIfAborted();if(code)throw new Error(error||`Evaluation archive worker exited ${code}`);
}

export async function drainEvaluationArchives(config,{signal}={}) {
  if(!config.skills?.archiveEvaluations)return;
  config={...config,root:await realpath(config.root)};
  return withSkillLock(config,async()=>{
    const state=new State(join(config.root,'state.sqlite'));
    try {
      const storage=CampaignStorage.open(state,config.root);if(!storage)throw new Error('Evaluation archives require parent campaign storage');
      for(const job of rows(state).filter(row=>row.status!=='complete')) {
        signal?.throwIfAborted();
        try {
          if(job.phase==='dispatching')throw new Error('Archive dispatch identity unknown; observe before retry');
          if(job.worker){const owner=JSON.parse(job.worker);if(!await ownerStopped(owner)||!groupStopped(owner.pid))throw new Error('Archive worker remains live or unresolved');}
          const {files,origin}=await scope(config,job,state);
          const fresh=files.filter(path=>!state.db.prepare('SELECT 1 FROM storage_artifacts WHERE path=?').get(relative(config.root,path)));
          if(fresh.length)await withSkillLock(config,()=>invoke(config,state,job,{action:'checkpoint',paths:fresh},signal),{key:'storage-maintenance',signal});
          const artifacts=[];
          for(const path of files) {
            const row=await storage.register(path,{attempt:origin.id,kind:'evaluation-kdb',role:'evaluation'});
            if(row.status==='open')await storage.seal(row.id);artifacts.push(row.id);
          }
          await withSkillLock(config,()=>invoke(config,state,job,{action:'compress',project:origin.project},signal),{key:'storage-maintenance',signal});
          await scope(config,job,state);
          await withSkillLock(config,()=>invoke(config,state,job,{action:'finalize',artifacts},signal),{key:'storage-maintenance',signal});
          state.db.prepare("UPDATE evaluation_archives SET status='complete',error=NULL,updated=? WHERE root=?").run(new Date().toISOString(),job.root);
        }catch(error){state.db.prepare('UPDATE evaluation_archives SET error=?,updated=? WHERE root=?').run(error.message,new Date().toISOString(),job.root);throw error;}
      }
    }finally{state.close();}
  },{key:'evaluation-archive',signal});
}
