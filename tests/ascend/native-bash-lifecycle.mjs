import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { spawn } from 'node:child_process';
import { mkdtemp,mkdir,readFile,rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { setTimeout as wait } from 'node:timers/promises';
import { State } from '../../controller/state.mjs';
import { attemptStopped,processIdentity } from '../../controller/lifecycle.mjs';
async function until(check) {
  const deadline=performance.now()+8000;
  while(performance.now()<deadline){const result=await check();if(result)return result;await wait(10);}
  throw Error('fixture did not reach expected native lifecycle state');
}
for(const completed of [false,true])test(`native Bash ${completed?'completed permits':'unfinished blocks'} resume after SIGKILL`,{timeout:15000},async()=>{
  const root=await mkdtemp(join(tmpdir(),'native-bash-loss-')),source=join(root,'source');await mkdir(source);
  const state=new State(join(root,'state.sqlite'));let requests=0,worker,pid,exit;
  const command=`python3 -c "import os,time;open('child.pid','w').write(str(os.getpid()));time.sleep(${completed?0.05:60})"`;
  const server=createServer(async(req,res)=>{
    let raw='';for await(const data of req)raw+=data;const body=JSON.parse(raw);requests++;
    if(body.messages.some(m=>m.role==='tool'))return; // Keep the actor open after graceful tool completion.
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'fixture',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    frame({role:'assistant',tool_calls:[{index:0,id:'local',type:'function',function:{name:'bash',arguments:JSON.stringify({command})}}]});
    frame({},'tool_calls');res.end('data: [DONE]\n\n');
  });server.listen(0,'127.0.0.1');await once(server,'listening');
  try {
    state.add({id:'p',source,goal:'Local Python work',actor_seconds:30,verifier:{files:[],command:'true'}});
    worker=spawn(process.execPath,[new URL('./native-bash-worker.mjs',import.meta.url).pathname,root,`http://127.0.0.1:${server.address().port}/v1`],{stdio:['ignore','ignore','pipe']});
    let stderr='';worker.stderr.on('data',data=>{stderr+=data;});exit=once(worker,'exit');
    const id=await until(async()=>{try{return await readFile(join(root,'attempt-id'),'utf8');}catch{if(worker.exitCode!==null)throw Error(stderr);}});
    pid=await until(async()=>{try{return Number(await readFile(join(source,'child.pid'),'utf8'));}catch{if(worker.exitCode!==null)throw Error(stderr);}});
    await until(()=>state.events(id).find(e=>e.type===(completed?'native-bash-completed':'native-bash-started')));
    if(completed)await until(()=>requests===2);
    const before=state.attempt(id),count=requests;
    worker.kill('SIGKILL');await exit;
    const events=state.events(id),starts=events.filter(e=>e.type==='native-bash-started'),ends=events.filter(e=>e.type==='native-bash-completed');
    assert.equal(starts.length,1);assert.ok(starts[0].data.session);assert.ok(before.session);
    const environment={reconcileOperation:async()=>assert.fail('native Bash must not use target reconciliation'),observe:async()=>assert.fail('no target handles')};
    if(!completed){
      process.kill(pid,0); // The actual detached native shell child survived worker death.
      assert.equal(ends.length,0);
      await assert.rejects(state.resume(id,a=>attemptStopped(state,a,environment),await processIdentity()),/Previous execution/);
      assert.deepEqual(state.attempt(id),before);
      process.kill(pid,'SIGKILL');pid=null;
      assert.equal(await attemptStopped(state,before,environment),false); // Absence alone is not closure evidence.
    }else{
      assert.equal(ends.length,1);assert.equal(ends[0].data.operation,starts[0].data.operation);
      assert.ok(ends[0].data.actor_elapsed>before.elapsed);
      assert.throws(()=>process.kill(pid,0),{code:'ESRCH'});
      await state.resume(id,a=>attemptStopped(state,a,environment),await processIdentity());
      const resumed=state.attempt(id);assert.equal(resumed.id,id);assert.equal(resumed.session,before.session);
      assert.equal(resumed.outcome,null);assert.ok(resumed.elapsed>=ends[0].data.actor_elapsed);
      pid=null;
    }
    assert.equal(requests,count);assert.equal(state.events(id).filter(e=>e.type==='native-bash-started').length,1);
  }finally{
    if(worker?.exitCode===null && worker.signalCode===null){worker.kill('SIGKILL');await exit;}
    if(pid){try{process.kill(pid,'SIGKILL');}catch(error){if(error.code!=='ESRCH')throw error;}}
    server.closeAllConnections();await new Promise(ok=>server.close(ok));state.close();await rm(root,{recursive:true,force:true});
  }
});
