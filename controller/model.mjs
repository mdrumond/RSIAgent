import { ModelRuntime } from '@earendil-works/pi-coding-agent';
import { readFile } from 'node:fs/promises';
import { parseEnv } from 'node:util';
import { InMemoryCredentialStore } from '@earendil-works/pi-ai';

export async function modelFromProfile(profile) {
  if (profile.max_context_tokens!==undefined && (!Number.isSafeInteger(profile.max_context_tokens)||profile.max_context_tokens<2))
    throw new Error('max_context_tokens must be a safe integer of at least 2');
  const key=process.env[profile.credential_env] ?? (profile.credentials_file
    ? parseEnv(await readFile(profile.credentials_file,'utf8'))[profile.credential_env] : undefined);
  if (!key) throw new Error(`Missing credential environment variable ${profile.credential_env}`);
  const runtime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),
    modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
  const customModel=!runtime.getModel(profile.provider,profile.model);
  if (customModel) {
    if (!profile.metadata) throw new Error('Unknown model requires explicit native Pi model metadata');
    runtime.registerProvider(profile.provider,{baseUrl:profile.base_url,models:[profile.metadata]});
  }
  runtime.registerProvider(profile.provider,{baseUrl:profile.base_url});
  await runtime.setRuntimeApiKey(profile.provider,key);
  const model=runtime.getModel(profile.provider,profile.model);
  const contextWindow=Math.min(model.contextWindow,profile.max_context_tokens??model.contextWindow);
  const maxTokens=Math.min(model.maxTokens,profile.max_tokens||model.maxTokens,contextWindow-1);
  if (profile.max_context_tokens!==undefined) {
    // Native session/model restoration must observe the same limits as the initial selection.
    runtime.registerProvider(profile.provider,{models:[{...model,...(customModel?{headers:profile.metadata.headers}:{}),contextWindow,maxTokens}]});
    return {modelRuntime:runtime,model:runtime.getModel(profile.provider,profile.model)};
  }
  return {modelRuntime:runtime,model:profile.max_tokens?{...model,maxTokens}:model};
}
