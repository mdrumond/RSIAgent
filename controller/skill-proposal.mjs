import { mkdir, writeFile, rename, appendFile } from 'node:fs/promises';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { Type } from 'typebox';
import { stages } from './workflow.mjs';

const scopes = ['global-kdb', ...stages];
const save = async (path, value) => {
  const tmp = `${path}.${randomUUID()}.tmp`;
  await writeFile(tmp, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
  await rename(tmp, path);
};

// Native Pi owns tool turns, correction, context, retry and compaction.
export async function generateCandidate({ directory, input, modelRuntime, model, native,
  seconds = 240, extensions = [], validate = () => {}, signal,onSession=async()=>{},onSessionClosed=async()=>{} }) {
  signal?.throwIfAborted();
  if (!Number.isFinite(seconds) || seconds <= 0) throw new Error('Writer elapsed seconds must be positive');
  const started = performance.now();
  const sdk = native ?? await import('@earendil-works/pi-coding-agent');
  await mkdir(directory, { recursive: true });
  await save(join(directory, 'current-guidance.json'), input.entries ?? []);
  await save(join(directory, 'current-usage.json'), { context: input.context, userinfo: input.userinfo,
    actualEvents: input.actualEvents, originals: input.originalCount ?? 0, pruneAfterTasks: input.pruneAfterTasks ?? 24 });
  await writeFile(join(directory, 'usage-history.jsonl'), (input.history ?? []).map(x => JSON.stringify(x)).join('\n'));
  const agentDir = join(directory, 'agent');
  await mkdir(agentDir, { recursive: true });
  const settingsManager = sdk.SettingsManager.inMemory({ compaction: { enabled: true }, retry: { enabled: true } });
  const manager = sdk.SessionManager.create(directory, directory);
  if(manager.getSessionFile)await onSession(manager.getSessionFile(),join(directory,'provider.jsonl'));
  let submitted;
  const id = Type.String({ pattern: '^[a-z0-9-]+$' });
  const contradiction = { contradictoryEvidence: Type.Optional(Type.Array(Type.String())) };
  const parameters = Type.Object({ changes: Type.Array(Type.Union([
    Type.Object({ action: Type.Literal('remove'), id, ...contradiction }),
    Type.Object({ action: Type.Literal('upsert'), id, scope: Type.Union(scopes.map(scope => Type.Literal(scope))),
      text: Type.String({ minLength: 1 }), critical: Type.Boolean(), negative: Type.Boolean(), ...contradiction })
  ])) });
  const extension = pi => {
    pi.registerTool({ name: 'submit_changes', label: 'Submit skill changes',
      description: 'Validate and submit candidate guidance. Correct any rejection using the returned reason.',
      parameters, execute: async (_id, { changes }) => {
        signal?.throwIfAborted();
        await validate(changes);
        signal?.throwIfAborted();
        submitted = changes;
        return { content: [{ type: 'text', text: 'Candidate changes accepted.' }], details: {} };
      } });
    for (const type of ['before_provider_request', 'provider_stream_event'])
      pi.on(type, async payload => appendFile(join(directory, 'provider.jsonl'), JSON.stringify(payload) + '\n'));
  };
  const loader = new sdk.DefaultResourceLoader({ cwd: directory, agentDir, settingsManager,
    noExtensions: true, noSkills: true, noContextFiles: true, extensionFactories: [extension, ...extensions] });
  await loader.reload();
  const { session } = await sdk.createAgentSession({ cwd: directory, agentDir, settingsManager, modelRuntime, model,
    tools: ['read', 'grep', 'find', 'ls', 'submit_changes', 'mcp__kdb__search', 'mcp__kdb__read_entry'],
    sessionManager: manager, resourceLoader: loader });
  let timer, expired = false, failure;
  const abort = () => { void session.abort().catch(() => {}); };
  signal?.addEventListener('abort', abort, { once: true });
  try {
    signal?.throwIfAborted();
    const remaining = seconds * 1000 - (performance.now() - started);
    if (remaining <= 0) throw new Error('Writer elapsed allocation exhausted');
    timer = setTimeout(() => { expired = true; void session.abort().catch(() => {}); }, remaining);
    await session.bindExtensions({});
    if (expired) throw new Error('Writer elapsed allocation exhausted');
    signal?.throwIfAborted();
    await session.prompt(`Read current-guidance.json and current-usage.json with native read tools.
Use usage-history.jsonl only for relevant prior usage. Resolve cited collection/entry_id with mcp__kdb__read_entry
and search positive/negative evidence with mcp__kdb__search; indexed source files may be absent.
Propose English guidance prioritizing error avoidance, detection and recovery before performance. Retain negatives.
Preserve critical guidance unless recorded contradictory evidence supports replacement; unused pruning must meet the original-task threshold in current-usage.json.
Submit via submit_changes; correct any rejection. A final text response does not publish changes.`);
    await session.waitForIdle();
    signal?.throwIfAborted();
    if (expired) throw new Error('Writer elapsed allocation exhausted');
    if (!submitted) throw new Error('Native writer did not submit validated changes');
    return submitted;
  } catch (error) { failure = error; throw error; } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
    const cleanup = async operation => { try { await operation(); } catch (error) { failure ??= error; } };
    await cleanup(() => session.exportToJsonl(join(directory, 'session.native.jsonl')));
    await cleanup(() => session.extensionRunner?.emit({ type: 'session_shutdown', reason: 'quit' }));
    await cleanup(() => session.dispose());
    if(manager.getSessionFile)await cleanup(()=>onSessionClosed(manager.getSessionFile()));
    if (failure) throw failure;
  }
}
