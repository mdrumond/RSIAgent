import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { mkdtemp,mkdir,writeFile,readFile,rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { ModelRuntime,createLocalBashOperations } from '@earendil-works/pi-coding-agent';
import { InMemoryCredentialStore } from '@earendil-works/pi-ai';
import { State,validateAcceptance } from '../../controller/state.mjs';
import { runAttempt } from '../../controller/pi.mjs';
import { stages } from '../../controller/workflow.mjs';

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
  const accepted=scenario==='perf-pass';
  for(const stage of stages) {
    calls.push(['rsi_skill',{}]);
    if(['design','optimisation'].includes(stage))for(const kind of ['good','negative'])
      calls.push(['rsi_knowledge',{query:'prior facts',kind}]);
    if(stage==='implementation')calls.push(['bash',{command:'python3.12 -c "print(42)"'}]);
    if(stage==='validation') {calls.push(['rsi_verify',{}]);if(!accepted)break;}
    calls.push(['rsi_advance',{reflection:'Use independent numeric and performance criteria.'}]);
  }
  const server=createServer(async(req,res)=>{
    let raw='';for await(const chunk of req)raw+=chunk;
    const body=JSON.parse(raw),index=body.messages.reduce((n,m)=>n+(m.tool_calls?.length??0),0);
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
        command:'PYTHONDONTWRITEBYTECODE=1 python3.12 ../numeric.py',performance_command:'python3.12 ../performance.py'}});
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
    await writeFile(join(root,'verdict.json'),JSON.stringify({outcome,verdict,
      executions:executions.map(({command,purpose,timeout})=>({command,purpose,timeout}))},null,2));
  }finally {
    server.closeAllConnections();await new Promise(resolve=>server.close(resolve));state.close();
    if(!process.env.PERFORMANCE_PROOF_DIR)await rm(root,{recursive:true,force:true});
  }
});
