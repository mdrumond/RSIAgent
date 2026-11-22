/** Explicit operator maintenance only; never registered as an agent tool. */
import {auditQueueRebuild,rebuildQueue} from './failure-queue.mjs';
import {DatabaseSync} from 'node:sqlite';
import {join} from 'node:path';
import {State} from '../controller/state.mjs';
import {ownerStopped} from '../controller/lifecycle.mjs';
import {attentionSnapshot,auditResolution,applyResolution,reopenResolution} from '../controller/failure-attention.mjs';

export async function runFailures(config,input={}){
 const action=input.action??'audit',path=join(config.root,'state.sqlite');
 if(!['audit','inspect','apply','reopen','rebuild-audit','rebuild','refresh-memory-provenance'].includes(action))throw Error('Unknown failures maintenance action');
 if(['audit','inspect','rebuild-audit'].includes(action)){
  const db=new DatabaseSync(path,{readOnly:true});
  try{if(action==='rebuild-audit')return await auditQueueRebuild(config,db,input);return action==='inspect'?attentionSnapshot(db):await auditResolution(db,input.manifest??input,{target:config.target});}finally{db.close();}
 }
 const state=new State(path),db=state.db;
 try{
  const controls=db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get()?db.prepare("SELECT value FROM control WHERE key IN ('owner','supervisor-owner','skill-writer','memory-writer')").all():[];
  for(const row of controls){
   const owner=JSON.parse(row.value);if(owner&&!await ownerStopped(owner))throw Error('Checkpoint live campaign writers before failure maintenance');
  }
  if(action==='refresh-memory-provenance'){
   const exists=name=>db.prepare("SELECT 1 FROM sqlite_master WHERE name=?").get(name);
   if(exists('skill_backlog_runs')&&db.prepare("SELECT 1 FROM skill_backlog_runs WHERE status='running'").get()||exists('skill_backlog_meta')&&JSON.parse(db.prepare("SELECT value FROM skill_backlog_meta WHERE key='microlearning-active'").get()?.value??'null'))throw Error('Settle active comparisons and learning before refreshing provenance');
   return await (await import('../controller/storage-knowledge.mjs')).refreshStorageMemoryProvenance(config,{state,serialized:true});
  }
  if(action==='rebuild')return await rebuildQueue(config,db,input);
  if(action==='reopen')return reopenResolution(db,input);
  return await applyResolution(db,input.manifest??input,{target:config.target});
 }finally{state.close();}
}
