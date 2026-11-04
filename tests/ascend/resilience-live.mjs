import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {mkdtemp,mkdir,writeFile,rm,readdir} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {loadLiveProfile,createLiveFixture} from './resilience/live-harness.mjs';
import {cases} from './resilience/live.case.mjs';
import {runCases} from './resilience/runner.mjs';
import {stages} from '../../controller/workflow.mjs';

test('missing live credentials are blocked before fixture resources or paid dispatch',async t=>{
  const root=await mkdtemp(join(tmpdir(),'live-blocked-')),old=process.env.RESILIENCE_LIVE_PROFILE;
  t.after(async()=>{if(old===undefined)delete process.env.RESILIENCE_LIVE_PROFILE;else process.env.RESILIENCE_LIVE_PROFILE=old;await rm(root,{recursive:true,force:true});});
  delete process.env.RESILIENCE_LIVE_PROFILE;
  const report=await runCases(cases,{output:join(root,'report'),tier:'deepseek',caseId:'deepseek-readiness',deadlineSeconds:10,cleanupSeconds:2});
  assert.equal(report.status,'blocked');assert.equal(report.results[0].status,'blocked');
  assert.deepEqual(await readdir(join(root,'report/deepseek-readiness')),[]);
});

test('safe live preflight makes no request; real native canonical fixture uses a scripted loopback provider',{timeout:45000},async t=>{
  const root=await mkdtemp(join(tmpdir(),'live-native-')),requests=[],cleanup=[];let complete=false;const abort=new AbortController();
  const calls=[];for(const stage of stages){calls.push(['rsi_skill',{}]);
    if(['design','optimisation'].includes(stage))for(const kind of ['good','negative'])calls.push(['rsi_knowledge',{query:'integer sum',kind}]);
    if(stage==='implementation')calls.push(['write',{path:'solution.py',content:'def solve(values):\n    return sum(values)\n'}],['rsi_target',{command:'python -c "import solution;assert solution.solve([2,3])==5"'}]);
    if(stage==='validation')calls.push(['rsi_verify',{}]);calls.push(['rsi_advance',{reflection:'Observed independent integer evidence retained.'}]);
  }
  const server=createServer(async(req,res)=>{
    let raw='';for await(const chunk of req)raw+=chunk;const body=JSON.parse(raw);requests.push(body);
    const index=body.messages.reduce((count,m)=>count+(m.tool_calls?.length??0),0),call=calls[index];
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'live-fixture',object:'chat.completion.chunk',created:1,model:'deepseek-flash',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    if(call){frame({role:'assistant',tool_calls:[{index:0,id:'call-'+index,type:'function',function:{name:call[0],arguments:JSON.stringify(call[1])}}]});frame({},'tool_calls');}
    else{frame({role:'assistant',content:'Verified.'});frame({},'stop');}res.end('data: [DONE]\n\n');
  });server.listen(0,'127.0.0.1');await once(server,'listening');
  const key='LIVE_SANITY_KEY',previous=process.env[key],old=process.env.RESILIENCE_LIVE_PROFILE;
  process.env[key]='fixture-only';const path=join(root,'profile.json');process.env.RESILIENCE_LIVE_PROFILE=path;
  await writeFile(path,JSON.stringify({python:process.env.PYTHON??'python3',actor:{provider:'deepseek',model:'deepseek-flash',credential_env:key,base_url:`http://127.0.0.1:${server.address().port}/v1`},actor_seconds:30}));
  t.after(async()=>{
    const errors=[];for(const fn of cleanup.reverse())try{await fn({signal:AbortSignal.timeout(15000),remainingSeconds:15});}catch(error){errors.push(error.message);}
    server.closeAllConnections();await new Promise(ok=>server.close(ok));
    if(previous===undefined)delete process.env[key];else process.env[key]=previous;
    if(old===undefined)delete process.env.RESILIENCE_LIVE_PROFILE;else process.env.RESILIENCE_LIVE_PROFILE=old;
    if(complete&&!errors.length)await rm(root,{recursive:true,force:true});else t.diagnostic('Retained scripted fixture: '+root);assert.deepEqual(errors,[]);
  });
  const metadata=await loadLiveProfile();assert.equal(metadata.credentialsAvailable,true);assert.equal(metadata.provider,'deepseek');
  for(const name of ['actor','credential_env','credentials_file','key'])assert.equal(Object.hasOwn(metadata,name),false);
  assert.equal(requests.length,0);
  await mkdir(join(root,'case'));
  const fixture=await createLiveFixture({directory:join(root,'case'),signal:abort.signal,seed:1,profile:'normal',event:async()=>{},onCleanup:fn=>cleanup.push(fn)});
  const result=await fixture.run();assert.equal(result.accepted,true);fixture.verifyAccepted();
  assert.ok(requests.length>0);assert.ok(requests[0].tools.some(t=>t.function.name==='mcp__kdb__search'));
  assert.ok(fixture.observations.some(e=>e.tool==='rsi_target'&&!e.error));complete=true;
});
