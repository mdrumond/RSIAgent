/** Node SDK boundary only: Pi owns provider calls, tools, history and compaction. */
import {createInterface} from 'node:readline';
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {join} from 'node:path';

let session, active = false, failure, serial = 0;
const pending = new Map();
const send = value => process.stdout.write(JSON.stringify(value).split(process.env.RSI_NATIVE_API_KEY).join('[redacted]')+'\n');
function host(name, args, toolCallId) {
  const id = ++serial;
  send({type:'tool', id, name, arguments:args, tool_call_id:toolCallId});
  return new Promise((resolve, reject) => pending.set(id, {resolve, reject}));
}
async function start(config) {
  async function installed(name) {
    const root = join(config.modules, name);
    const manifest = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
    return import(pathToFileURL(join(root, manifest.exports?.['.']?.import || manifest.main)).href);
  }
  const sdk = await installed('@earendil-works/pi-coding-agent');
  const {InMemoryCredentialStore} = await installed('@earendil-works/pi-ai');
  const runtime = await sdk.ModelRuntime.create({credentials:new InMemoryCredentialStore(),
    modelsPath:null, allowModelNetwork:false, refreshOnCreate:false});
  const p = config.profile;
  if (!runtime.getModel(p.provider, p.model)) {
    if (!config.model_metadata) throw new Error('Registered profile has no native catalog/model metadata');
    runtime.registerProvider(p.provider, {baseUrl:p.base_url, models:[config.model_metadata]});
  }
  const catalog = runtime.getModel(p.provider, p.model);
  if (catalog.baseUrl !== p.base_url) runtime.registerProvider(p.provider, {baseUrl:p.base_url});
  const model = {...runtime.getModel(p.provider, p.model), maxTokens:Math.min(catalog.maxTokens, p.max_tokens)};
  await runtime.setRuntimeApiKey(p.provider, process.env.RSI_NATIVE_API_KEY);
  const native = config.tools_factory
    ? await (await import(pathToFileURL(config.tools_factory).href)).createNativeResources({sdk, host,
        root:config.root, cwd:config.cwd, options:config.tools_options}) : {};
  const tools = config.tools.map(tool => ({...tool, label:tool.name, executionMode:'sequential',
    execute:(id, args) => host(tool.name, args, id)}));
  const trace = pi => {
    for (const type of ['before_provider_request', 'after_provider_response', 'provider_stream_event']) {
      pi.on(type, event => {
        const {headers, ...safe} = event; // Native auth headers never enter evidence.
        send(safe);
      });
    }
  };
  const settings = sdk.SettingsManager.inMemory({compaction:{enabled:true}, retry:{enabled:true}});
  const loader = new sdk.DefaultResourceLoader({cwd:config.cwd, agentDir:join(config.root, '.pi'),
    settingsManager:settings, noExtensions:true, noContextFiles:true,
    ...config.resource_options, ...native.resourceOptions,
    systemPrompt:config.system_prompt, extensionFactories:[trace, ...(native.extensions || [])]});
  await loader.reload();
  ({session} = await sdk.createAgentSession({cwd:config.cwd, modelRuntime:runtime,
    model, thinkingLevel:p.reasoning?'high':'off',
    sessionManager:sdk.SessionManager.continueRecent(config.cwd, join(config.root, 'sessions')),
    settingsManager:settings, resourceLoader:loader, noTools:'builtin',
    customTools:[...tools, ...(native.tools || [])]}));
  session.subscribe(event => {
    if (event.type === 'message_end' && event.message.role === 'assistant')
      failure = event.message.stopReason === 'error'
        ? event.message.errorMessage || 'Native provider failed' : undefined;
    send(event);
  });
  send({type:'ready', session_file:session.sessionManager.getSessionFile()});
}
createInterface({input:process.stdin}).on('line', async line => {
  try {
    const command = JSON.parse(line);
    if (command.type === 'start') return await start(command);
    if (command.type === 'tool_result') {
      pending.get(command.id)?.resolve(command.result); pending.delete(command.id); return;
    }
    if (command.type === 'abort') { await session.abort(); return; }
    if (command.type === 'close') {
      await session.abort(); session.dispose(); process.exit(0);
    }
    if (command.type !== 'prompt' || active) throw new Error('Invalid or concurrent native prompt');
    active = true; failure = undefined;
    try { await session.prompt(command.text); }
    catch (error) { failure = error.message; }
    finally {
      active = false;
      send({type:'settled', status:failure?'failed':'settled', error:failure,
        session_file:session.sessionManager.getSessionFile()});
    }
  } catch (error) { send({type:'fatal', message:error.message}); }
});
