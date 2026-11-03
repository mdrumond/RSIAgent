import { ModelRuntime } from '@earendil-works/pi-coding-agent';
import { InMemoryCredentialStore } from '@earendil-works/pi-ai';

export async function modelFromProfile(profile) {
  const key=process.env[profile.credential_env];
  if (!key) throw new Error(`Missing credential environment variable ${profile.credential_env}`);
  const runtime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),
    modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
  if (!runtime.getModel(profile.provider,profile.model)) {
    if (!profile.metadata) throw new Error('Unknown model requires explicit native Pi model metadata');
    runtime.registerProvider(profile.provider,{baseUrl:profile.base_url,models:[profile.metadata]});
  }
  runtime.registerProvider(profile.provider,{baseUrl:profile.base_url});
  await runtime.setRuntimeApiKey(profile.provider,key);
  return {modelRuntime:runtime,model:runtime.getModel(profile.provider,profile.model)};
}
