import {join} from 'node:path';
import {State} from '../controller/state.mjs';
import {startBackground,requests,reconcileRequests} from './background.mjs';
import {sealClosedArtifacts} from './storage.mjs';
import {stat} from 'node:fs/promises';

// Submission only; the existing background request owns compression and its receipts.
export async function queueCompression(config) {
  if(!config.storage||config.storage.compress===false)return null;
  const state=new State(join(config.root,'state.sqlite'));
  try {
    await reconcileRequests(config.root,{state});
    const pending=requests(state).find(r=>r.kind==='storage'&&['queued','running'].includes(r.status));
    if(pending)return {request:pending.id,retained:true};
    await sealClosedArtifacts(config,{limit:1000});
    let ready=false;
    for(const row of state.db.prepare(`SELECT path FROM storage_artifacts WHERE status='closed'
      AND encoding='plain' AND external=0 AND kind!='source-git' ORDER BY id LIMIT 1000`).all()) {
      try{if((await stat(join(config.root,row.path))).isFile()){ready=true;break;}}
      catch(error){if(error.code!=='ENOENT')throw error;}
    }
    if(!ready)return null;
    return await startBackground(config.root,'storage',{action:'compress',limit:1000},
      {detach:true});
  }finally{state.close();}
}
