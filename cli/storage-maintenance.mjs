import {join} from 'node:path';
import {State} from '../controller/state.mjs';
import {startBackground,requests,reconcileRequests} from './background.mjs';
import {sealClosedArtifacts} from './storage.mjs';

// Submission only; the existing managed request owns compression and its receipts.
export async function queueCompression(config,{manager}={}) {
  if(!config.storage||config.storage.compress===false)return null;
  const state=new State(join(config.root,'state.sqlite'));
  try {
    await reconcileRequests(config.root,{state,...(manager?{manager}:{})});
    const pending=requests(state).find(r=>r.kind==='storage'&&['queued','running'].includes(r.status));
    if(pending)return {request:pending.id,retained:true};
    await sealClosedArtifacts(config,{limit:1000});
    if(!state.db.prepare(`SELECT 1 FROM storage_artifacts WHERE status='closed'
      AND encoding='plain' AND external=0 AND kind!='source-git' LIMIT 1`).get())return null;
    return await startBackground(config.root,'storage',{action:'compress',limit:1000,'delete-source':true},
      {detach:true,...(manager?{manager}:{})});
  }finally{state.close();}
}
