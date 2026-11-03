import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { once } from 'node:events';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { setTimeout as delay } from 'node:timers/promises';
import { createAgentSession, DefaultResourceLoader, ModelRuntime,
  SessionManager, SettingsManager } from '@earendil-works/pi-coding-agent';
import { createKdbBridge } from '../../integrations/kdb-extension.mjs';

// Uses a real Python MCP server and native Pi registration/tool hooks, without
// invoking a model provider or production BGE. PYTHON selects the helper runtime.
test('native Pi MCP retrieval preserves correlation, citations and permission gates',
  {timeout:20000}, async () => {
    const root = await mkdtemp(join(tmpdir(),'ascend-kdb-pi-'));
    const agentDir = join(root,'agent');
    await mkdir(agentDir);
    await writeFile(join(root,'kernel.py'),'vector addition with aligned loads\n');
    const socket = createServer();
    socket.listen(0,'127.0.0.1');
    await once(socket,'listening');
    const port = socket.address().port;
    await new Promise(resolve => socket.close(resolve));
    const child = spawn(process.env.PYTHON ?? 'python3.12', [
      '-m','ascend_tools.kdb_service','--db',join(root,'knowledge.sqlite'),
      '--port',String(port),'--embedding-factory','kdb_fixture:Embeddings'], {
        cwd:process.cwd(), stdio:['ignore','ignore','pipe'],
        env:{...process.env,PYTHONPATH:[resolve('.'),resolve('tests/ascend'),
          process.env.PYTHONPATH ?? ''].join(':')},
      });
    let stderr='';
    child.stderr.on('data',data=>{stderr=(stderr+data.toString()).slice(-3000);});
    let session;
    try {
      const url = `http://127.0.0.1:${port}`;
      const deadline = Date.now()+10000;
      while (true) {
        if (child.exitCode !== null) throw new Error(`MCP exited: ${stderr}`);
        try {if ((await fetch(`${url}/health`)).ok) break;} catch {}
        if (Date.now()>deadline) throw new Error(`MCP readiness timeout: ${stderr}`);
        await delay(25);
      }
      const events=[];
      let blocked=false;
      const bridge=createKdbBridge({url:`${url}/mcp`,attemptId:'pi-attempt',agentDir,
        target:'a3',event:async(type,data)=>events.push({type,...data})});
      const extension=pi=>{
        pi.registerTool(bridge.knowledgeTool());
        pi.on('tool_call',async event=>{
          if (blocked && event.toolName==='mcp__kdb__search') {
            return {block:true,reason:'test permission gate'};
          }
        });
      };
      const settings=SettingsManager.inMemory({});
      const loader=new DefaultResourceLoader({cwd:root,agentDir,settingsManager:settings,
        noExtensions:true,noSkills:true,extensionFactories:[bridge.extension,extension]});
      await loader.reload();
      const modelRuntime=await ModelRuntime.create({authPath:join(agentDir,'auth.json'),
        modelsPath:null,refreshOnCreate:false});
      ({session}=await createAgentSession({cwd:root,agentDir,modelRuntime,
        settingsManager:settings,sessionManager:SessionManager.inMemory(root),
        resourceLoader:loader,noTools:'builtin'}));
      await session.bindExtensions({});
      while (!session.getCallableToolNames().includes('mcp__kdb__search')) {
        if (Date.now()>deadline) throw new Error('Pi native MCP registration timeout');
        await delay(25);
      }
      assert.ok(session.getActiveToolNames().includes('mcp__kdb__search'));
      const ctx=session.extensionRunner.createToolContext('seed',new AbortController().signal);
      const indexed=await ctx.executeTool('mcp__kdb__add_repo',{
        path:root,collection:'baseline',language:'triton',target:'a3',paths:['kernel.py']});
      assert.equal(indexed.isError,false);
      const tool=session.agent.state.tools.find(tool=>tool.name==='rsi_knowledge');
      const result=await tool.execute('retrieve',{query:'aligned loads',kind:'good'});
      const data=JSON.parse(result.content[0].text);
      assert.equal(data.content.length,1);
      assert.equal(data.evidence[0].collection,'baseline');
      assert.equal(data.evidence[0].sha256.length,64);
      assert.ok(events.some(event=>event.tool==='mcp__kdb__search' &&
        event.parent_tool_call_id==='retrieve' && !event.failed));
      await assert.rejects(bridge.knowledge.search({query:'loads',attempt_id:'other'},ctx));
      blocked=true;
      await assert.rejects(tool.execute('blocked',{query:'loads',kind:'good'}),/blocked/);
      // Read persisted service telemetry via SQLite, without opening another service.
      const inspect=spawn(process.env.PYTHON ?? 'python3.12',['-c',
        'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(c.execute("SELECT count(*) FROM knowledge_events WHERE operation=\'search\' AND attempt_id=\'pi-attempt\'").fetchone()[0])',
        join(root,'knowledge.sqlite')]);
      let output='';
      inspect.stdout.on('data',data=>{output+=data;});
      const [code]=await once(inspect,'exit');
      assert.equal(code,0);
      assert.equal(output.trim(),'1');
    } finally {
      if (session) {
        await session.extensionRunner.emit({type:'session_shutdown',reason:'exit'});
        session.dispose();
      }
      if (child.exitCode===null) {
        const exited=once(child,'exit');
        child.kill('SIGTERM');
        await exited;
      }
      await rm(root,{recursive:true,force:true});
    }
  });

test('workflow adapter bounds hit text and rejects native tool failures',async()=>{
  const bridge=createKdbBridge({url:'http://127.0.0.1:8765/mcp',
    attemptId:'bounded',agentDir:'/tmp/kdb-test-agent',target:'a3'});
  const calls=[];
  const hit={collection:'baseline',chunk_id:'native-id',path:'kernel.py',
    start_line:1,end_line:80,source_sha256:'a'.repeat(64),text:'x'.repeat(4000)};
  const ctx={executeTool:async(name,args)=>{
    calls.push({name,args});
    return {isError:false,result:{structuredContent:{
      structuredContent:{result:Array(20).fill(hit)}}}};
  }};
  const result=await bridge.knowledge.search({query:'loads',kind:'negative'},ctx);
  assert.equal(calls[0].name,'mcp__kdb__search');
  assert.equal(calls[0].args.target,'a3');
  assert.equal(result.content.length,8);
  assert.equal(result.content.reduce((size,hit)=>size+hit.text.length,0),12000);
  assert.equal(result.evidence.length,8);
  assert.ok(result.content.every(hit=>hit.truncated));
  await assert.rejects(bridge.knowledge.search({query:'loads'},
    {executeTool:async()=>({isError:true})}),/failed or was blocked/);
});
