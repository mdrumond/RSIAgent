// Separate controller process for real SDK native Bash worker-loss scenarios.
import {writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {State} from '../../../controller/state.mjs';
import {processIdentity} from '../../../controller/lifecycle.mjs';
import {runAttempt} from '../../../controller/pi.mjs';
import {runtime} from './lifecycle-helpers.mjs';
const [root,url]=process.argv.slice(2),state=new State(join(root,'state.sqlite'));
const abort=new AbortController();process.on('SIGTERM',()=>abort.abort(new Error('Fixture checkpoint')));
try {
  const attempt=state.admit('worker',await processIdentity());
  state.checkpoint(attempt.id,2.25);
  await writeFile(join(root,'attempt-id'),attempt.id);
  await runAttempt({state,attempt:state.attempt(attempt.id),project:state.project('worker').spec,
    attemptPath:join(root,'attempt'),skillMap:{},signal:abort.signal,...await runtime(url),
    environmentFactory:async()=>({prepare:async()=>{},execute:async()=>{throw Error('Unexpected target dispatch');}}),
    knowledge:{search:async()=>({content:[],evidence:[]})}});
  throw Error('Worker must remain open until interrupted');
}catch(error){
  if(!abort.signal.aborted)throw error;
  await writeFile(join(root,'worker-closed.json'),JSON.stringify({interrupted:true}));
}finally{state.close();}
