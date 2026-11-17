/** Explicit operator maintenance only; never registered as an agent tool. */
import {DatabaseSync} from 'node:sqlite';
import {join} from 'node:path';
import {State} from '../controller/state.mjs';
import {ownerStopped} from '../controller/lifecycle.mjs';
import {attentionSnapshot,auditResolution,applyResolution,reopenResolution} from '../controller/failure-attention.mjs';

export async function runFailures(config,input={}){
 const action=input.action??'audit',path=join(config.root,'state.sqlite');
 if(!['audit','inspect','apply','reopen'].includes(action))throw Error('Unknown failures maintenance action');
 if(['audit','inspect'].includes(action)){
  const db=new DatabaseSync(path,{readOnly:true});
  try{return action==='inspect'?attentionSnapshot(db):await auditResolution(db,input.manifest??input,{target:config.target});}finally{db.close();}
 }
 const state=new State(path),db=state.db;
 try{
  const controls=db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get()?db.prepare("SELECT value FROM control WHERE key IN ('owner','supervisor-owner','skill-writer','memory-writer')").all():[];
  for(const row of controls){
   const owner=JSON.parse(row.value);if(owner&&!await ownerStopped(owner))throw Error('Checkpoint live campaign writers before failure maintenance');
  }
  if(action==='reopen')return reopenResolution(db,input);
  return await applyResolution(db,input.manifest??input,{target:config.target});
 }finally{state.close();}
}
