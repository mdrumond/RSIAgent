import { join } from 'node:path';
import { writeFile } from 'node:fs/promises';
import { ModelRuntime } from '@earendil-works/pi-coding-agent';
import { InMemoryCredentialStore } from '@earendil-works/pi-ai';
import { State } from '../../controller/state.mjs';
import { processIdentity } from '../../controller/lifecycle.mjs';
import { runAttempt } from '../../controller/pi.mjs';
const [root,url]=process.argv.slice(2),state=new State(join(root,'state.sqlite'));
const runtime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
runtime.registerProvider('fixture',{baseUrl:url,models:[{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});
await runtime.setRuntimeApiKey('fixture','fixture-only');
const attempt=state.admit('p',await processIdentity());
await writeFile(join(root,'attempt-id'),attempt.id);
await runAttempt({state,attempt,project:state.project('p').spec,attemptPath:join(root,'attempt'),skillMap:{},
  environmentFactory:async()=>({prepare:async()=>{},execute:async()=>{throw Error('unexpected target dispatch');}}),
  knowledge:{search:async()=>({content:[],evidence:[]})},modelRuntime:runtime,model:runtime.getModel('fixture','scripted')});
