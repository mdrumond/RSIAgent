import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
const load = async path => JSON.parse(await readFile(path, 'utf8'));

export function nativeWriter(onInput = () => {}, hang = false, read = load) {
  let listener, release, directory;
  const tools = {};
  const flags = { aborted: false, disposed: false };
  return { flags, native: {
    SettingsManager: { inMemory: () => ({}) }, SessionManager: { create: () => ({}) },
    DefaultResourceLoader: class {
      constructor(options) { this.options = options; directory = options.cwd; }
      async reload() { for (const factory of this.options.extensionFactories)
        await factory({ registerTool(tool) { tools[tool.name] = tool; }, on() {} }); }
    },
    createAgentSession: async options => {
      assert.ok(!options.tools.includes('write') && !options.tools.includes('bash') && !options.tools.includes('rsi_target'));
      return { session: {
        subscribe(fn) { listener = fn; return () => {}; }, async bindExtensions() {},
        async prompt() {
          const current = await read(join(directory, 'current-usage.json'));
          onInput({ ...current, entries: await read(join(directory, 'current-guidance.json')) });
          if (hang) { if (!flags.aborted) await new Promise(resolve => { release = resolve; }); }
          else if (!flags.aborted) await tools.submit_changes.execute('submit', { changes: [] });
          listener({ type: 'message_end', message: { role: 'assistant', content: [] } });
        },
        async waitForIdle() {}, async abort() { flags.aborted = true; release?.(); },
        dispose() { flags.disposed = true; }
      } };
    }
  } };
}
