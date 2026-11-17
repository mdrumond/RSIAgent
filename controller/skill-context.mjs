/** Skill experiments have a bounded context policy, independent of foreground profiles. */
import {writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
import {ModelRuntime} from '@earendil-works/pi-coding-agent';
import {InMemoryCredentialStore} from '@earendil-works/pi-ai';
import {modelFromProfile} from './model.mjs';

export const skillContextLimit=200000;
export function skillProfile(profile){
 if(!profile)return profile;
 const limit=profile.max_context_tokens;
 if(limit!==undefined&&(!Number.isSafeInteger(limit)||limit<2))throw Error('max_context_tokens must be a safe integer of at least 2');
 return {...profile,max_context_tokens:Math.min(limit??skillContextLimit,skillContextLimit)};
}
export function skillConfig(config){
 return {...config,...Object.fromEntries(['actor','curriculum_actor','skill_writer_actor'].filter(key=>config[key]).map(key=>[key,skillProfile(config[key])]))};
}
export async function skillModelFromProfile(profile,{directory}={}){
 const native=await modelFromProfile(skillProfile(profile));
 if(native.model.contextWindow>skillContextLimit)throw Error('Skill model context policy was not applied');
 if(directory){await mkdir(directory,{recursive:true});await writeFile(join(directory,'model-limits.json'),JSON.stringify({format:1,provider:native.model.provider,model:native.model.id,contextWindow:native.model.contextWindow,maxTokens:native.model.maxTokens},null,2)+'\n');}
 return native;
}

/** Catalogue-only inspection; no credentials or provider requests are needed to freeze limits. */
export async function skillModelIdentity(profile){
 profile=skillProfile(profile);
 const runtime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
 const model=runtime.getModel(profile.provider,profile.model)??profile.metadata;
 const contextWindow=model?Math.min(profile.max_context_tokens,model.contextWindow):null;
 const maxTokens=model?Math.min(model.maxTokens,profile.max_tokens||model.maxTokens,contextWindow-1):null;
 return {provider:profile.provider,model:profile.model,maxTokens:profile.max_tokens??null,maxContextTokens:profile.max_context_tokens,effectiveContextTokens:contextWindow,effectiveMaxTokens:maxTokens};
}
