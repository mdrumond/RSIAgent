import {randomUUID,createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {mkdir,stat,realpath,writeFile,rename} from 'node:fs/promises';
import {resolve,relative,join,isAbsolute} from 'node:path';

const now=()=>new Date().toISOString();
const inside=(root,path)=>{const r=relative(root,path);return r===''||(!r.startsWith('..')&&!isAbsolute(r));};
export const hasStorage=state=>Boolean(state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='storage_campaign'").get());

// The existing state database remains the only campaign ledger. No sidecar DB.
export class CampaignStorage {
  constructor(state,root){this.state=state;this.db=state.db;this.root=resolve(root);}
  static open(state,root){return hasStorage(state)?new CampaignStorage(state,root):null;}
  async initialize({id=randomUUID(),target=null}={}) {
    this.db.exec(`CREATE TABLE IF NOT EXISTS storage_campaign (
      id TEXT PRIMARY KEY,created TEXT NOT NULL,target TEXT,format INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS storage_deployments (
      id TEXT PRIMARY KEY,created TEXT NOT NULL,machinery_revision TEXT);
      CREATE TABLE IF NOT EXISTS storage_attempts (
      attempt TEXT PRIMARY KEY REFERENCES attempts(id),phase TEXT NOT NULL,role TEXT NOT NULL,
      kind TEXT NOT NULL,parent TEXT REFERENCES attempts(id),wave INTEGER,deployment TEXT,
      memory_revision TEXT,skill_revision TEXT,knowledge_generation TEXT);
      CREATE INDEX IF NOT EXISTS storage_attempts_phase ON storage_attempts(phase,role,attempt);
      CREATE TABLE IF NOT EXISTS storage_iterations (
      id INTEGER PRIMARY KEY,attempt TEXT NOT NULL REFERENCES attempts(id),stage TEXT NOT NULL,
      ordinal INTEGER NOT NULL,reflection TEXT NOT NULL,source_hash TEXT,source_commit TEXT,
      UNIQUE(attempt,stage,ordinal));
      CREATE TABLE IF NOT EXISTS storage_artifacts (
      id TEXT PRIMARY KEY,attempt TEXT REFERENCES attempts(id),project TEXT REFERENCES projects(id),
      kind TEXT NOT NULL,role TEXT NOT NULL,path TEXT NOT NULL,external INTEGER NOT NULL DEFAULT 0,
      status TEXT NOT NULL,bytes INTEGER,sha256 TEXT,encoding TEXT NOT NULL DEFAULT 'plain',
      descriptor TEXT,created TEXT NOT NULL,UNIQUE(path));
      CREATE INDEX IF NOT EXISTS storage_artifacts_owner ON storage_artifacts(attempt,kind,id);
      CREATE TABLE IF NOT EXISTS storage_revisions (
      name TEXT PRIMARY KEY,revision TEXT NOT NULL,generation TEXT,updated TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS storage_sessions (
      session TEXT PRIMARY KEY,attempt TEXT REFERENCES attempts(id),phase TEXT NOT NULL,
      role TEXT NOT NULL,created TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS storage_retrievals (
      id INTEGER PRIMARY KEY,attempt TEXT REFERENCES attempts(id),role TEXT NOT NULL,
      tool_call TEXT,operation TEXT NOT NULL,arguments TEXT NOT NULL,generation TEXT,
      citations TEXT NOT NULL,created TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS storage_jobs (
      id TEXT PRIMARY KEY,artifact TEXT REFERENCES storage_artifacts(id),kind TEXT NOT NULL,
      status TEXT NOT NULL,progress TEXT NOT NULL DEFAULT '{}',updated TEXT NOT NULL);`);
    const campaign=this.db.prepare('SELECT * FROM storage_campaign').get();
    if(campaign)return campaign;
    this.db.prepare('INSERT INTO storage_campaign VALUES (?,?,?,1)').run(id,now(),target);
    await mkdir(join(this.root,'artifacts'),{recursive:true});
    await writeFile(join(this.root,'campaign.json'),JSON.stringify({id,format:1},null,2)+'\n',{flag:'wx',mode:0o600});
    return this.db.prepare('SELECT * FROM storage_campaign').get();
  }
  deployment(revision=null){const id=randomUUID();this.db.prepare('INSERT INTO storage_deployments VALUES (?,?,?)').run(id,now(),revision);return id;}
  attempt(id,{phase,role='actor',kind='original',parent=null,wave=null,deployment=null,skillRevision=null}={}) {
    if(!['phase1','phase2','inference'].includes(phase))throw new Error('Unknown storage phase');
    const revisions=Object.fromEntries(this.db.prepare('SELECT name,revision,generation FROM storage_revisions').all().map(r=>[r.name,r]));
    this.db.prepare(`INSERT OR IGNORE INTO storage_attempts VALUES (?,?,?,?,?,?,?,?,?,?)`)
      .run(id,phase,role,kind,parent,wave,deployment,revisions.memory?.revision??null,
        skillRevision??revisions.skills?.revision??null,revisions.memory?.generation??null);
    return this.db.prepare('SELECT * FROM storage_attempts WHERE attempt=?').get(id);
  }
  activate(name,revision,generation=null){
    if(!['memory','skills'].includes(name)||!revision)throw new Error('Invalid knowledge revision');
    this.db.prepare('INSERT OR REPLACE INTO storage_revisions VALUES (?,?,?,?)').run(name,revision,generation,now());
  }
  async session(path,{attempt=null,phase,role='actor'}={}) {
    const artifact=await this.register(path,{attempt,kind:'pi-session',role});
    this.db.prepare('INSERT OR IGNORE INTO storage_sessions VALUES (?,?,?,?,?)').run(artifact.path,attempt,phase,role,now());
    return artifact;
  }
  async retrieval({attempt=null,role='actor',toolCall=null,operation,arguments:args={},generation=null,entries=[]}) {
    const directory=join(this.root,'artifacts','citations');await mkdir(directory,{recursive:true});
    const citations=[];
    for(const entry of entries) {
      if(typeof entry.text!=='string')continue;
      const text=entry.text,sha=createHash('sha256').update(text).digest('hex'),path=join(directory,sha+'.txt');
      try{await writeFile(path,text,{flag:'wx',mode:0o600});}catch(error){if(error.code!=='EEXIST')throw error;}
      const artifact=await this.register(path,{kind:'cited-text',role:'kdb'});
      const {text:ignored,...reference}=entry;
      citations.push({...reference,artifact:artifact.id,text_sha256:sha});
    }
    this.db.prepare('INSERT INTO storage_retrievals(attempt,role,tool_call,operation,arguments,generation,citations,created) VALUES (?,?,?,?,?,?,?,?)')
      .run(attempt,role,toolCall,operation,JSON.stringify(args),generation,JSON.stringify(citations),now());
    return citations;
  }
  iteration(attempt,{stage,reflection='',sourceHash=null,sourceCommit=null}){
    return this.db.prepare(`INSERT INTO storage_iterations(attempt,stage,ordinal,reflection,source_hash,source_commit)
      SELECT ?,?,COALESCE(MAX(ordinal),0)+1,?,?,? FROM storage_iterations WHERE attempt=? AND stage=?`)
      .run(attempt,stage,reflection,sourceHash,sourceCommit,attempt,stage);
  }
  async register(path,{attempt=null,project=null,kind='artifact',role='actor',external=false}={}){
    let stored;
    if(external)stored=path;
    else {
      const absolute=resolve(this.root,path),root=await realpath(this.root);
      let actual;try{actual=await realpath(absolute);}catch(e){if(e.code!=='ENOENT')throw e;actual=resolve(await realpath(resolve(absolute,'..')),absolute.split('/').at(-1));}
      if(!inside(root,actual))throw new Error('Artifact must belong to its campaign');
      stored=relative(root,actual);
    }
    const existing=this.db.prepare('SELECT * FROM storage_artifacts WHERE path=?').get(stored);
    if(existing){if(existing.attempt!==attempt||existing.project!==project||existing.kind!==kind)throw new Error('Artifact ownership conflict');return existing;}
    const id=randomUUID();this.db.prepare('INSERT INTO storage_artifacts(id,attempt,project,kind,role,path,external,status,created) VALUES (?,?,?,?,?,?,?,?,?)')
      .run(id,attempt,project,kind,role,stored,Number(external),external?'not-collected':'open',now());
    return this.db.prepare('SELECT * FROM storage_artifacts WHERE id=?').get(id);
  }
  async seal(id){
    const row=this.db.prepare('SELECT * FROM storage_artifacts WHERE id=?').get(id);
    if(!row||row.external)throw new Error('Collect artifact before closing it');
    const path=resolve(this.root,row.path),actual=await realpath(path);
    if(!inside(await realpath(this.root),actual))throw new Error('Artifact escaped campaign');
    const before=await stat(actual);if(!before.isFile())throw new Error('Artifact must be a file');
    const hash=createHash('sha256');for await(const chunk of createReadStream(actual))hash.update(chunk);
    const after=await stat(actual);if(before.size!==after.size||before.mtimeMs!==after.mtimeMs||before.ino!==after.ino)throw new Error('Artifact changed while closing');
    this.db.prepare("UPDATE storage_artifacts SET status='closed',bytes=?,sha256=? WHERE id=?").run(after.size,hash.digest('hex'),id);
    return this.db.prepare('SELECT * FROM storage_artifacts WHERE id=?').get(id);
  }
  async manifest(){
    const value={format:1,campaign:this.db.prepare('SELECT * FROM storage_campaign').get(),
      revisions:this.db.prepare('SELECT * FROM storage_revisions').all()};
    const path=join(this.root,'campaign.json'),temporary=path+'.tmp';
    await writeFile(temporary,JSON.stringify(value,null,2)+'\n',{mode:0o600});await rename(temporary,path);return value;
  }
  list({attempt,kind,after='',limit=100}={}){
    if(!Number.isInteger(limit)||limit<1||limit>1000)throw new Error('Artifact limit must be 1..1000');
    const clauses=['id>?'],args=[after];
    for(const [k,v] of [['attempt',attempt],['kind',kind]])if(v!==undefined){clauses.push(k+'=?');args.push(v);}
    return this.db.prepare('SELECT * FROM storage_artifacts WHERE '+clauses.join(' AND ')+' ORDER BY id LIMIT ?').all(...args,limit);
  }
}
