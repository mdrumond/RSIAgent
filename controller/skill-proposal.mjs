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
  seconds = 240, extensions = [], validate = () => {} }) {
  if (!Number.isFinite(seconds) || seconds <= 0) throw new Error('Writer elapsed seconds must be positive');
  const started = performance.now();
  const sdk = native ?? await import('@earendil-works/pi-coding-agent');
  await mkdir(directory, { recursive: true });
  await save(join(directory, 'current-guidance.json'), input.entries ?? []);
  await save(join(directory, 'current-usage.json'), { context: input.context, userinfo: input.userinfo,
    actualEvents: input.actualEvents, originals: input.originalCount ?? 0 });
  await writeFile(join(directory, 'usage-history.jsonl'), (input.history ?? []).map(x => JSON.stringify(x)).join('\n'));
  const agentDir = join(directory, 'agent');
  await mkdir(agentDir, { recursive: true });
  const settingsManager = sdk.SettingsManager.inMemory({ compaction: { enabled: true }, retry: { enabled: true } });
  const manager = sdk.SessionManager.create(directory, directory);
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
        await validate(changes);
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
  const responses = [];
  const unsubscribe = session.subscribe(event => {
    if (event.type === 'message_end') responses.push(event.message);
  });
  let timer, expired = false;
  try {
    const remaining = seconds * 1000 - (performance.now() - started);
    if (remaining <= 0) throw new Error('Writer elapsed allocation exhausted');
    timer = setTimeout(() => { expired = true; void session.abort().catch(() => {}); }, remaining);
    await session.bindExtensions({});
    await session.prompt(`Read current-guidance.json and current-usage.json with native read tools.
Use usage-history.jsonl only for relevant prior usage. Resolve cited collection/entry_id with mcp__kdb__read_entry
and search positive/negative evidence with mcp__kdb__search; indexed source files may be absent.
Propose English guidance prioritizing error avoidance, detection and recovery before performance. Retain negatives.
Preserve critical guidance unless recorded contradictory evidence supports replacement; unused pruning needs 24 distinct originals.
Submit via submit_changes; correct any rejection. A final text response does not publish changes.`);
    await session.waitForIdle();
    if (expired) throw new Error('Writer elapsed allocation exhausted');
    if (!submitted) throw new Error('Native writer did not submit validated changes');
    return submitted;
  } finally {
    clearTimeout(timer);
    try {
      await save(join(directory, 'responses.json'), responses);
      const messages = manager.getEntries?.().filter(entry => entry.type === 'message').map(entry => entry.message) ?? responses;
      await writeFile(join(directory, 'sft.native.jsonl'), messages.map(x => JSON.stringify(x)).join('\n'));
      await session.extensionRunner?.emit({ type: 'session_shutdown', reason: 'quit' });
    } finally { unsubscribe?.(); session.dispose(); }
  }
}
