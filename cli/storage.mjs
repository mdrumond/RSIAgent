import {spawn} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {State} from '../controller/state.mjs';
import {CampaignStorage} from '../controller/storage.mjs';
import {ownerStopped,failureSettled} from '../controller/lifecycle.mjs';

export const schemaOptions=Object.fromEntries([
  'action','artifact','project','role','phase','kind','outcome','stage','iteration',
  'skill-revision','memory-revision','cursor','after','limit','destination','inventory','package'
].map(key=>[key,{type:'string'}]));
Object.assign(schemaOptions,{'raw-provider':{type:'boolean'},'delete-source':{type:'boolean'},apply:{type:'boolean'}});

export async function sealClosedArtifacts(config,action={}) {
  const state=new State(join(config.root,'state.sqlite'));
  try {
    const storage=CampaignStorage.open(state,config.root);if(!storage)return [];
    state.db.exec('CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY,value TEXT NOT NULL)');
    const globalScan=['artifact','project','phase','role','kind','outcome','stage','iteration','skill-revision','memory-revision','skill_revision','memory_revision'].every(key=>action[key]===undefined);
    const cursor=action.after??(globalScan?JSON.parse(state.db.prepare("SELECT value FROM control WHERE key='storage-seal-cursor'").get()?.value??'""'):'');
    const sealed=[],clauses=["f.id>?","f.status='open'","f.kind!='source-git'"],args=[cursor];
    for(const [key,column] of Object.entries({artifact:'f.id',project:'COALESCE(f.project,a.project)',
      role:'f.role',kind:'f.kind',phase:'s.phase',outcome:'a.outcome',
      skill_revision:'s.skill_revision',memory_revision:'s.memory_revision'})) {
      const value=action[key]??action[key.replace('_','-')];
      if(value!==undefined){
        clauses.push(key==='project'&&action.action==='pack'
          ? `(${column}=? OR (f.kind='cited-text' AND f.attempt IS NULL AND f.project IS NULL))` : column+'=?');
        args.push(value);
      }
    }
    if(action.stage!==undefined||action.iteration!==undefined) {
      const terms=['i.attempt=f.attempt'];
      for(const [key,column] of [['stage','stage'],['iteration','ordinal']])
        if(action[key]!==undefined){terms.push('i.'+column+'=?');args.push(action[key]);}
      clauses.push('EXISTS(SELECT 1 FROM storage_iterations i WHERE '+terms.join(' AND ')+')');
    }
    const limit=Number(action.limit??100);
    if(!Number.isInteger(limit)||limit<1||limit>1000)throw new Error('limit must be 1..1000');
    const rows=state.db.prepare(`SELECT f.* FROM storage_artifacts f
      LEFT JOIN attempts a ON a.id=f.attempt LEFT JOIN storage_attempts s ON s.attempt=f.attempt
      WHERE ${clauses.join(' AND ')} ORDER BY f.id LIMIT ?`).all(...args,limit);
    if(action.after===undefined&&globalScan)state.db.prepare("INSERT OR REPLACE INTO control(key,value) VALUES ('storage-seal-cursor',?)")
      .run(JSON.stringify(rows.at(-1)?.id??''));
    for(const row of rows) {
      if(row.status!=='open'||row.external||row.kind==='source-git')continue;
      if(row.attempt) {
        const attempt=state.attempt(row.attempt);
        if(!attempt?.outcome||!await ownerStopped(attempt.owner)||!failureSettled(state,attempt.id))continue;
        if(!state.queryEvents(attempt.id,{types:['postprocess-completed'],limit:1}).length)continue;
      } else if(row.kind!=='cited-text'&&!state.db.prepare('SELECT 1 FROM storage_sessions WHERE (session=? OR provider=?) AND closed=1').get(row.path,row.path))continue;
      try {await storage.seal(row.id);sealed.push(row.id);}
      catch(error) {
        if(error.code!=='ENOENT'&&error.message!=='Artifact must be a file')throw error;
      }
    }
    return sealed;
  }finally{state.close();}
}

export async function run(config,values={}) {
  const action=values.config?JSON.parse(await readFile(values.config,'utf8')):{...values};
  action.action??=values.action??'inspect';
  const operation=async()=>{
    if(['compress','pack'].includes(action.action))await sealClosedArtifacts(config,action);
    return invoke(config,action,values.signal);
  };
  if(['compress','pack','prune','restore'].includes(action.action))
    return (await import('./evolution-lock.mjs')).withSkillLock(config,operation,{signal:values.signal,key:'storage-maintenance'});
  return operation();
}

export async function runStandalone(values={}) {
  const action=values.config?JSON.parse(await readFile(values.config,'utf8')):{...values};
  action.action='unpack';
  return invoke({python:values.python??'python',root:process.cwd()},action,values.signal);
}

function invoke(config,action,signal) {
  return new Promise((ok,no)=>{
    const child=spawn(config.python??'python',['-m','ascend_tools.storage_cli','--root',config.root,
      '--action-json','-'],{cwd:config.root,signal,
      env:{...process.env,PYTHONPATH:fileURLToPath(new URL('..',import.meta.url))},
      stdio:['pipe',action.action==='walk'?'inherit':'pipe','pipe']});
    child.stdin.on('error',no);child.stdin.end(JSON.stringify(action));
    let output='',error='';
    child.stdout?.on('data',chunk=>{output+=chunk;if(output.length>2*1024*1024){child.kill();no(new Error('Storage response exceeded limit'));}});
    child.stderr.on('data',chunk=>{error=(error+chunk).slice(-65536);});
    child.on('error',no);child.on('close',code=>{
      if(code)return no(new Error(error||`Storage exited ${code}`));
      if(action.action==='walk')return ok(undefined);
      try{ok(JSON.parse(output));}catch(e){no(e);}
    });
  });
}
