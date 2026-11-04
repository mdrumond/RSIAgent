import {createServer} from 'node:http';
import {once} from 'node:events';
import {appendFile} from 'node:fs/promises';
import {join} from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';
import {ModelRuntime} from '@earendil-works/pi-coding-agent';
import {InMemoryCredentialStore} from '@earendil-works/pi-ai';

export async function until(check,signal) {
  const deadline=performance.now()+15000;
  while(performance.now()<deadline){signal?.throwIfAborted();const value=await check();if(value)return value;
    await delay(15,undefined,{signal});}
  throw Error('Native lifecycle fixture did not reach the expected boundary');
}

// Loopback provider/MCP fixtures exercise official SDK clients, never paid providers.
export async function listen(ctx,handler) {
  const abort=new AbortController(),signal=AbortSignal.any([ctx.signal,abort.signal]);
  const server=createServer(async(req,res)=>{
    try {let raw='';for await(const data of req)raw+=data;
      await handler(req,res,raw?JSON.parse(raw):null,signal);
    }catch(error){if(!res.destroyed){res.writeHead(500);res.end(JSON.stringify({error:error.message}));}}
  });
  ctx.onCleanup(async()=>{abort.abort();server.closeAllConnections();
    await new Promise(resolve=>server.close(resolve));});
  server.listen(0,'127.0.0.1');await once(server,'listening');
  return `http://127.0.0.1:${server.address().port}`;
}
export async function provider(ctx,scheduler,reply) {
  const requests=[];
  const url=await listen(ctx,async(_req,res,body,signal)=>{
    requests.push(body);await appendFile(join(ctx.directory,'provider.jsonl'),JSON.stringify(body)+'\n');
    await scheduler.at('provider.request',{},signal);
    const response=await reply(body,signal);
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'lifecycle',object:'chat.completion.chunk',
      created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    if(typeof response==='string'){frame({role:'assistant',content:response});frame({},'stop');}
    else {frame({role:'assistant',tool_calls:[{index:0,id:response.id??'call',type:'function',
      function:{name:response.name,arguments:JSON.stringify(response.args)}}]});frame({},'tool_calls');}
    res.end('data: [DONE]\n\n');
  });
  return {url:url+'/v1',requests};
}
export async function runtime(url) {
  const modelRuntime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,
    allowModelNetwork:false,refreshOnCreate:false});
  modelRuntime.registerProvider('lifecycle-fixture',{baseUrl:url,models:[{id:'scripted',name:'Scripted fixture',
    api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,
    cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});
  await modelRuntime.setRuntimeApiKey('lifecycle-fixture','fixture-only');
  return {modelRuntime,model:modelRuntime.getModel('lifecycle-fixture','scripted')};
}
