import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createAgentSession,DefaultResourceLoader,SessionManager,SettingsManager,shouldCompact} from '@earendil-works/pi-coding-agent';
import {skillProfile,skillConfig,skillModelFromProfile,skillModelIdentity} from '../../controller/skill-context.mjs';
import {microlearningConfig} from '../../cli/skill-microlearning.mjs';
import {readFile} from 'node:fs/promises';
import {modelFromProfile} from '../../controller/model.mjs';
const key='CONTEXT_LIMIT_FIXTURE';
function profile(overrides={}){return {provider:'deepseek',model:'deepseek-flash',credential_env:key,base_url:'http://127.0.0.1:1',...overrides};}
async function credentials(t){const previous=process.env[key];process.env[key]='fixture-only';t.after(()=>{if(previous===undefined)delete process.env[key];else process.env[key]=previous;});}

test('context limit governs known model catalogue and native session restoration/compaction',async t=>{
 await credentials(t);const original=await modelFromProfile(profile()),native=await skillModelFromProfile(profile({max_tokens:8192,metadata:{contextWindow:9999999}}));
 assert.ok(original.model.contextWindow>200000);assert.equal(native.model.contextWindow,200000);assert.equal(native.model.maxTokens,8192);
 assert.equal(native.modelRuntime.getModel('deepseek','deepseek-flash').contextWindow,200000);
 assert.equal(original.modelRuntime.getModel('deepseek','deepseek-flash').contextWindow,original.model.contextWindow);
 const cwd=await mkdtemp(join(tmpdir(),'model-context-'));t.after(()=>rm(cwd,{recursive:true,force:true}));
 const settingsManager=SettingsManager.inMemory(),manager=SessionManager.inMemory(cwd);
 manager.appendMessage({role:'user',content:'Retained history',timestamp:1});manager.appendMessage({role:'assistant',content:[{type:'text',text:'Recorded'}],api:native.model.api,provider:native.model.provider,model:native.model.id,stopReason:'stop',usage:{input:190000,output:1,cacheRead:0,cacheWrite:0,totalTokens:190001,cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}},timestamp:2});
 const resourceLoader=new DefaultResourceLoader({cwd,agentDir:cwd,settingsManager,noSkills:true,noContextFiles:true,noExtensions:true});await resourceLoader.reload();
 // Omitting explicit model exercises restoring the capped catalogue, rather than just passing a copy.
 const {session}=await createAgentSession({cwd,agentDir:cwd,modelRuntime:native.modelRuntime,sessionManager:manager,settingsManager,resourceLoader,tools:[]});
 try{
  const usage=session.getContextUsage();assert.equal(usage.contextWindow,200000);assert.ok(usage.percent>90);
  const settings=settingsManager.getCompactionSettings(session.model);
  assert.equal(shouldCompact(usage.tokens,session.model.contextWindow,settings),true);
  assert.equal(shouldCompact(usage.tokens,original.model.contextWindow,settings),false);
 }finally{await session.dispose();}
});

test('context cap cannot expand custom provider capacity and output capacity fits the effective window',async t=>{
 await credentials(t);const metadata={id:'custom',name:'Custom',api:'openai-completions',reasoning:false,input:['text'],contextWindow:64000,maxTokens:48000,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}},base=profile({provider:'context-fixture',model:'custom',metadata});
 const retained=await modelFromProfile({...base,max_context_tokens:200000});assert.equal(retained.model.contextWindow,64000);assert.equal(retained.model.maxTokens,48000);
 const clipped=await modelFromProfile({...base,max_context_tokens:32000,max_tokens:40000});assert.equal(clipped.model.contextWindow,32000);assert.equal(clipped.model.maxTokens,31999);
 assert.equal(metadata.contextWindow,64000);assert.equal(metadata.maxTokens,48000);
});

test('invalid context limits fail before loading credentials or constructing a provider',async()=>{
 for(const value of [null,0,1,-1,1.5,'200000',Infinity,Number.MAX_SAFE_INTEGER+1])await assert.rejects(modelFromProfile(profile({max_context_tokens:value,credentials_file:'/nonexistent/context-secret.env',credential_env:'ABSENT_CONTEXT_KEY'})),/max_context_tokens/);
});

// Pi keeps model headers in raw provider definitions, not its public model snapshots.
test('clamping custom metadata retains required native gateway request headers',async t=>{
 await credentials(t);const metadata={id:'custom',name:'Custom',api:'openai-completions',reasoning:false,input:['text'],contextWindow:1000000,maxTokens:8192,cost:{input:0,output:0,cacheRead:0,cacheWrite:0},headers:{'X-Gateway-Route':'fixture-route'}},base=profile({provider:'gateway-fixture',model:'custom',metadata});
 const uncapped=await modelFromProfile(base),capped=await modelFromProfile({...base,max_context_tokens:200000});
 for(const native of [uncapped,capped])assert.equal(native.modelRuntime.getCompatibilityRequestConfig(native.model).headers['X-Gateway-Route'],'fixture-route');
 assert.equal(capped.model.contextWindow,200000);assert.deepEqual(metadata.headers,{'X-Gateway-Route':'fixture-route'});
 const known=await modelFromProfile(profile({max_context_tokens:200000,metadata}));assert.equal(known.modelRuntime.getCompatibilityRequestConfig(known.model).headers?.['X-Gateway-Route'],undefined);
});


test('skill startup constrains all profiles without changing foreground configuration',async t=>{
 await credentials(t);
 const original={actor:profile(),curriculum_actor:profile({max_context_tokens:64000}),skill_writer_actor:profile({max_context_tokens:999999}),skills:{activeVersion:'fixture'},root:'/fixture'};
 const bounded=skillConfig(original);
 assert.equal(original.actor.max_context_tokens,undefined);assert.equal(original.skill_writer_actor.max_context_tokens,999999);
 assert.equal(bounded.actor.max_context_tokens,200000);assert.equal(bounded.curriculum_actor.max_context_tokens,64000);assert.equal(bounded.skill_writer_actor.max_context_tokens,200000);
 for(const key of ['actor','curriculum_actor','skill_writer_actor']){
  const native=await skillModelFromProfile(bounded[key]);assert.equal(native.model.contextWindow,bounded[key].max_context_tokens);
 }
 const child=microlearningConfig(original,{directory:'/child',request:{learningTarget:'cpu',question:'fixture'},resource:{id:'host'},seconds:30});
 assert.equal(child.actor.max_context_tokens,200000);assert.equal(child.curriculum_actor.max_context_tokens,64000);
 const cwd=await mkdtemp(join(tmpdir(),'skill-startup-'));t.after(()=>rm(cwd,{recursive:true,force:true}));
 const runtime=await skillModelFromProfile(profile(),{directory:cwd}),record=JSON.parse(await readFile(join(cwd,'model-limits.json'),'utf8'));
 assert.equal(record.contextWindow,runtime.model.contextWindow);assert.equal(record.contextWindow,200000);assert.equal(record.maxTokens,runtime.model.maxTokens);
 const identity=await skillModelIdentity(profile());assert.equal(identity.effectiveContextTokens,runtime.model.contextWindow);assert.equal(identity.effectiveMaxTokens,runtime.model.maxTokens);
 assert.equal(skillProfile(profile({max_context_tokens:8000})).max_context_tokens,8000);
 assert.throws(()=>skillProfile(profile({max_context_tokens:null})),/max_context_tokens/);
});
