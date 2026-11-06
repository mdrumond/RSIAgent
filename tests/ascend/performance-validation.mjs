import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp,mkdir,writeFile,readFile,rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { ModelRuntime,createLocalBashOperations } from '@earendil-works/pi-coding-agent';
import { InMemoryCredentialStore } from '@earendil-works/pi-ai';
import { State,validateAcceptance } from '../../controller/state.mjs';
import { runAttempt } from '../../controller/pi.mjs';
import { stages } from '../../controller/workflow.mjs';

const python = process.env.PYTHON ?? process.env.DRIVER_PYTHON ?? 'python3.12';
const quote = text => "'" + text.replaceAll("'", "'\\''") + "'";

test('performance goals require a frozen command before project or actor admission',async()=>{
  const state=new State(':memory:');
  const project={id:'p',goal:'numeric',actor_seconds:30,performance_goal:'meet the frozen speed criterion',
    verifier:{command:'python numeric.py',files:[]}};
  try {
    assert.throws(()=>validateAcceptance(project),/performance_command/);
    assert.throws(()=>state.add(project),/performance_command/);
    assert.equal(state.project('p'),undefined);
    await assert.rejects(runAttempt({project,environmentFactory:async()=>assert.fail('actor started')}),/performance_command/);
  }finally{state.close();}
});

for(const scenario of ['perf-miss','numeric-fail','perf-pass'])test(scenario,{timeout:15000},async()=>{
  const proofRoot=process.env.PERFORMANCE_PROOF_DIR??tmpdir();
  await mkdir(proofRoot,{recursive:true});
  const root=await mkdtemp(join(proofRoot,`ascend-performance-${scenario}-`));
  const state=new State(join(root,'state.sqlite')),calls=[],executions=[];
  const accepted=scenario==='perf-pass';let returnedVerification;
  for(const stage of stages) {
    calls.push(['rsi_skill',{}]);
    if(['design','optimisation'].includes(stage))for(const kind of ['good','negative'])
      calls.push(['rsi_knowledge',{query:'prior facts',kind}]);
    if(stage==='implementation')calls.push(['rsi_target',{command:`${quote(python)} -c "print(42)"`}]);
    if(stage==='validation') {calls.push(['rsi_verify',{}]);if(!accepted)break;}
    calls.push(['rsi_advance',{reflection:'Use independent numeric and performance criteria.'}]);
  }
  const server=createServer(async(req,res)=>{
    let raw='';for await(const chunk of req)raw+=chunk;
    const body=JSON.parse(raw),index=body.messages.reduce((n,m)=>n+(m.tool_calls?.length??0),0);
    const verified=body.messages.find(m=>m.role==='tool'&&m.tool_call_id==='call-'+calls.findIndex(([name])=>name==='rsi_verify'));
    if(verified)returnedVerification=JSON.parse(verified.content);
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'fixture',object:'chat.completion.chunk',
      created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    if(index<calls.length) {
      const [name,args]=calls[index];
      frame({role:'assistant',tool_calls:[{index:0,id:'call-'+index,type:'function',function:{name,arguments:JSON.stringify(args)}}]});
      frame({},'tool_calls');
    }else{frame({role:'assistant',content:'Finished checking the independent criteria.'});frame({},'stop');}
    res.end('data: [DONE]\n\n');
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');
  try {
    const source=join(root,'source');await mkdir(source);await writeFile(join(source,'answer.py'),'answer=42\n');
    const numeric=join(root,'numeric.py'),performance=join(root,'performance.py');
    await writeFile(numeric,`import sys\nsys.path.insert(0,'.')\nfrom answer import answer\nassert answer==${scenario==='numeric-fail'?43:42}\nprint('numeric oracle passed')\n`);
    await writeFile(performance,`import sys\nprint('performance oracle checked')\nsys.exit(${accepted?0:1})\n`);
    const skillMap={};for(const stage of [...stages,'global-kdb']) {
      skillMap[stage]=join(root,stage+'.md');await writeFile(skillMap[stage],'Use independent evidence and complete this stage.');
    }
    state.add({id:'p',goal:'Return the correct number',actor_seconds:30,source,
      performance_goal:'meet the frozen speed criterion',verifier:{files:[numeric,performance],
        command:`PYTHONDONTWRITEBYTECODE=1 ${quote(python)} ../numeric.py`,performance_command:`${quote(python)} ../performance.py`}});
    const attempt=state.admit('p',{pid:process.pid});
    const runtime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
    runtime.registerProvider('performance-fixture',{baseUrl:`http://127.0.0.1:${server.address().port}/v1`,models:[{
      id:'scripted',name:'Local fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,
      maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});
    await runtime.setRuntimeApiKey('performance-fixture','fixture-only');
    const local=createLocalBashOperations();
    const outcome=await runAttempt({state,attempt,project:state.project('p').spec,attemptPath:join(root,'attempt'),skillMap,
      environmentFactory:async()=>({prepare:async()=>{},instructions:'Local CPU oracle fixture',execute:async(command,cwd,options)=>{
        executions.push({command,...options});return {...await local.exec(command,cwd,options),settled:true};}}),
      knowledge:{search:async()=>({content:[],evidence:[]})},modelRuntime:runtime,model:runtime.getModel('performance-fixture','scripted')});
    assert.equal(outcome.accepted,accepted);
    assert.equal(state.project('p').status,accepted?'passed':'failed');
    const verdict=state.events(attempt.id).find(e=>e.type==='verification').data;
    assert.equal(verdict.correctness.passed,scenario!=='numeric-fail');
    assert.equal(verdict.performance.skipped,scenario==='numeric-fail'?true:undefined);
    assert.equal(verdict.performance.passed,accepted);
    assert.ok(executions.every(e=>e.signal && e.timeout>0 && e.timeout<=30));
    assert.deepEqual(executions.map(e=>e.purpose),scenario==='numeric-fail'
      ? ['performance','verification'] : ['performance','verification','performance']);
    assert.ok((await readFile(verdict.evidence.correctness,'utf8')).length>0);
    if(scenario!=='numeric-fail')assert.ok((await readFile(verdict.evidence.performance,'utf8')).length>0);
    assert.deepEqual(returnedVerification.correctness,verdict.correctness);
    assert.deepEqual(returnedVerification.performance,verdict.performance);
    const checks=[returnedVerification.correctness,returnedVerification.performance].filter(check=>check?.evidence);
    for(const check of checks)assert.equal(check.evidence_sha256,createHash('sha256').update(await readFile(check.evidence)).digest('hex'));
    const recorded=await promisify(execFile)(python,['-c',`
import asyncio,json,os,sys
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
async def main():
    params=StdioServerParameters(command=sys.executable,args=['-m','ascend_tools.kdb_service','--db',sys.argv[1],'--transport','stdio','--embedding-factory','kdb_fixture:Embeddings'],env={'PYTHONPATH':os.environ['PYTHONPATH']})
    async with stdio_client(params) as (reader,writer):
        async with ClientSession(reader,writer) as session:
            await session.initialize()
            for check in json.loads(sys.argv[2]):
                result=await session.call_tool('record_lesson',{'lesson':'Independent verifier result retained','outcome':'good' if check['passed'] else 'negative','applicability':'local-python','attempt_id':'verifier-proof','evidence':[{'path':check['evidence'],'sha256':check['evidence_sha256']}]})
                assert not result.isError,result
    print('accepted')
asyncio.run(main())
`,join(root,'knowledge.sqlite'),JSON.stringify(checks)],{env:{...process.env,
      PYTHONPATH:[process.cwd(),join(process.cwd(),'tests/ascend'),process.env.PYTHONPATH??''].join(':')}});
    assert.equal(recorded.stdout.trim(),'accepted');
    await writeFile(join(root,'verdict.json'),JSON.stringify({outcome,verdict,
      executions:executions.map(({command,purpose,timeout})=>({command,purpose,timeout}))},null,2));
  }finally {
    server.closeAllConnections();await new Promise(resolve=>server.close(resolve));state.close();
    if(!process.env.PERFORMANCE_PROOF_DIR)await rm(root,{recursive:true,force:true});
  }
});

for(const reason of ['elapsed','external'])test(`default native bash stops on ${reason} abort without target dispatch`,{timeout:10000},async()=>{
  const root=await mkdtemp(join(tmpdir(),'native-bash-abort-')),source=join(root,'source');await mkdir(source);
  const state=new State(join(root,'state.sqlite')),abort=new AbortController();let dispatches=0;
  const command=`${quote(python)} -c "import os,time;open('local.pid','w').write(str(os.getpid()));time.sleep(30)"`;
  const server=createServer(async(req,res)=>{
    let raw='';for await(const data of req)raw+=data;const body=JSON.parse(raw);
    const called=body.messages.some(m=>m.role==='tool');res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'abort',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    if(!called){frame({role:'assistant',tool_calls:[{index:0,id:'local',type:'function',function:{name:'bash',arguments:JSON.stringify({command})}}]});frame({},'tool_calls');}
    else{frame({role:'assistant',content:'Stopped.'});frame({},'stop');}res.end('data: [DONE]\n\n');
  });server.listen(0,'127.0.0.1');await once(server,'listening');
  try {
    const runtime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
    runtime.registerProvider('abort-fixture',{baseUrl:`http://127.0.0.1:${server.address().port}/v1`,models:[{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});
    await runtime.setRuntimeApiKey('abort-fixture','fixture-only');
    state.add({id:'p',source,goal:'Bounded local operation',actor_seconds:reason==='elapsed'?2:10,verifier:{files:[],command:'true'}});
    const attempt=state.admit('p',{});
    const running=runAttempt({state,attempt,project:state.project('p').spec,attemptPath:join(root,'attempt'),signal:abort.signal,
      environmentFactory:async()=>({prepare:async()=>{},execute:async()=>{dispatches++;throw Error('unexpected target dispatch');}}),
      knowledge:{search:async()=>({content:[],evidence:[]})},skillMap:{},modelRuntime:runtime,model:runtime.getModel('abort-fixture','scripted')});
    const {setTimeout:wait}=await import('node:timers/promises');let pid;const deadline=performance.now()+3000;
    while(!pid&&performance.now()<deadline){try{pid=Number(await readFile(join(source,'local.pid'),'utf8'));}catch{}await wait(10);}
    assert.ok(pid);
    if(reason==='external'){abort.abort();await assert.rejects(running,/interrupted/);assert.equal(state.attempt(attempt.id).outcome,null);}
    else{assert.equal((await running).accepted,false);assert.equal(state.attempt(attempt.id).outcome,'failed');}
    let stopped=false;while(!stopped&&performance.now()<deadline){try{process.kill(pid,0);}catch(error){if(error.code==='ESRCH')stopped=true;else throw error;}await wait(10);}
    assert.equal(stopped,true);assert.equal(dispatches,0);
  }finally{server.closeAllConnections();await new Promise(ok=>server.close(ok));state.close();await rm(root,{recursive:true,force:true});}
});
