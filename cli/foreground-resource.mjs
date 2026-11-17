/** Move a stopped legacy foreground checkpoint off the permanent A/B host slot. */
import {isDeepStrictEqual} from 'node:util';
import {attemptStopped,ownerStopped,processIdentity} from '../controller/lifecycle.mjs';
import {environmentBinding} from '../controller/skill-evidence.mjs';
import {readConfig} from './instance.mjs';
import {readAttemptJob,publishAttemptJob,availableAttemptJob} from './attempt-job.mjs';

export async function migrateForegroundCheckpoint({state,config,job,resources,reserved,createEnvironment}) {
  if(!job.resume||job.finalize||!reserved)return null;
  const id=job.attempt.id,manifest=await readAttemptJob(state,config.root,id),prior=manifest.resource;
  if(prior?.id!==reserved.id||prior.metadata?.kind!=='host-slot'||prior.metadata.backgroundSlot!==true)return null;
  const controller=state.db.prepare("SELECT value FROM control WHERE key='owner'").get()?.value;
  if(!controller||!isDeepStrictEqual(JSON.parse(controller),await processIdentity()))return null;
  const attempt=state.attempt(id);
  if(!attempt||attempt.outcome!==null||!await ownerStopped(attempt.owner))return null;
  const project=state.project(attempt.project).spec;
  const binding=environmentBinding(config,project,prior);
  const next=resources.find(row=>row.available&&row.metadata?.kind==='host-slot'&&row.metadata.backgroundSlot===false&&
    Number.isInteger(row.metadata.hostSlot)&&row.metadata.hostSlot>=0&&row.metadata.pool===prior.metadata.pool&&
    (!config.resource_ids||config.resource_ids.includes(row.id))&&isDeepStrictEqual(row.env,prior.env)&&
    isDeepStrictEqual(environmentBinding(config,project,row),binding)&&
    !state.db.prepare('SELECT 1 FROM resource_leases WHERE resource=?').get(row.id));
  if(!next)return null;
  // The old binding owns reconciliation. Never route an unknown command or
  // physical-device lease onto the replacement slot to make it appear stopped.
  const environment=await createEnvironment(prior);
  if(!await attemptStopped(state,attempt,environment)||!environment.reconcileLease||!await environment.reconcileLease())return null;
  if(!isDeepStrictEqual((await readConfig(config.root)).environment,config.environment))return null;
  const moved={...manifest,resource:next};
  state.db.exec('BEGIN IMMEDIATE');
  try {
    const current=state.attempt(id);
    if(current?.owner!==attempt.owner||current.outcome!==null||
      state.db.prepare("SELECT value FROM control WHERE key='owner'").get()?.value!==controller||
      state.db.prepare("SELECT value FROM control WHERE key IN ('pause','stop') AND value='true'").get()||
      state.db.prepare('SELECT 1 FROM resource_leases WHERE resource=?').get(next.id)||
      state.db.prepare('SELECT attempt FROM resource_leases WHERE resource=?').get(prior.id)?.attempt!==id) {
      state.db.exec('ROLLBACK');return null;
    }
    state.db.prepare('UPDATE resource_leases SET resource=? WHERE resource=? AND attempt=?').run(next.id,prior.id,id);
    publishAttemptJob(state,id,moved);
    state.event(id,'foreground-resource-migrated',{from:prior,to:next,closure:'previous-execution-and-device-lease'});
    state.db.exec('COMMIT');
  }catch(error){state.db.exec('ROLLBACK');throw error;}
  // File publication can block this attempt, but not unrelated scheduling. The
  // committed lease and trusted job still agree and later reads repair the file.
  return await availableAttemptJob(state,config.root,id)?next:null;
}
