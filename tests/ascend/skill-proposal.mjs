import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { generateCandidate } from '../../controller/skill-proposal.mjs';
import { nativeWriter } from './skill-proposal-fixture.mjs';

const guidance = { id: 'failure', scope: 'debug', text: 'Check the failing case.', critical: false, negative: true };
const task = n => ({ id: `original-${n}`, projectId: `project-${n}`, kind: 'original',
  phase: 'phase2', completed: true, events: [{ type: 'knowledge-retrieved', data: { kind: 'negative', evidence: 'failure.json' } }] });
const load = async path => JSON.parse(await readFile(path, 'utf8'));
async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'skill-proposal-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

test('native publisher injection reads saved evidence and disposes its session', async t => {
  const root = await fixture(t);
  let observed;
  const { native, flags } = nativeWriter(input => { observed = input; });
  assert.deepEqual(await generateCandidate({ directory: join(root, 'proposal'),
    input: { entries: [guidance], actualEvents: task(0).events }, native }), []);
  assert.equal(observed.actualEvents.length, 1);
  assert.ok(flags.disposed);
  assert.equal((await load(join(root, 'proposal', 'responses.json'))).length, 1);
});

test('actual Pi readonly KDB lookup and validated publisher repair a rejected proposal', { timeout: 20000 }, async t => {
  const sdk = await import('@earendil-works/pi-coding-agent');
  const { InMemoryCredentialStore } = await import('@earendil-works/pi-ai');
  const { createServer } = await import('node:http');
  const { spawn, execFileSync } = await import('node:child_process');
  const { resolve } = await import('node:path');
  const root = await fixture(t);
  const source = process.env.RSI_KDB_TEST_ROOT ?? resolve('.');
  const python = process.env.RSI_KDB_TEST_PYTHON ?? 'python3.12';
  const env = { ...process.env, PYTHONPATH: [source, join(source, 'tests', 'ascend')].join(':') };
  const evidenceText = `observed-${Date.now()} cancellation failure`;
  const evidenceFile = join(root, 'reference.md');
  await writeFile(evidenceFile, evidenceText);
  const seed = JSON.parse(execFileSync(python, ['-c',
    "from ascend_tools.kdb import KDB;from kdb_fixture import Embeddings;import sys,json;d=KDB(sys.argv[1],Embeddings());d.add_repo(sys.argv[2],'baseline','a3',paths=['reference.md']);print(json.dumps(d.search('cancellation')[0]));d.close()",
    join(root, 'knowledge.sqlite'), root], { env, encoding: 'utf8' }));
  await rm(evidenceFile);
  const { createServer: createSocket } = await import('node:net');
  const socket = createSocket();
  await new Promise(resolve => socket.listen(0, '127.0.0.1', resolve));
  const port = socket.address().port;
  await new Promise(resolve => socket.close(resolve));
  const service = spawn(python, ['-m', 'ascend_tools.kdb_service', '--db', join(root, 'knowledge.sqlite'),
    '--port', String(port), '--embedding-factory', 'kdb_fixture:Embeddings'], { env, stdio: ['ignore', 'ignore', 'pipe'] });
  let stderr = '', startupError;
  service.stderr.on('data', data => { stderr = (stderr + data).slice(-3000); });
  service.once('error', error => { startupError = error; });
  t.after(async () => { if (service.pid && service.exitCode === null && service.signalCode === null) {
    const { once } = await import('node:events'); const closed = once(service, 'exit'); service.kill(); await closed;
  } });
  const deadline = Date.now() + 5000;
  while (true) {
    if (startupError) throw startupError;
    if (service.exitCode !== null || service.signalCode !== null)
      throw new Error(`KDB service exited (${service.exitCode ?? service.signalCode}): ${stderr}`);
    try { if ((await fetch(`http://127.0.0.1:${port}/health`)).ok) break; } catch {}
    if (Date.now() > deadline) throw new Error('KDB readiness deadline');
    await new Promise(resolve => setTimeout(resolve, 20));
  }
  const directory = join(root, 'native-proposal');
  const proposal = [{ action: 'upsert', id: 'recovery', scope: 'debug', text: evidenceText,
    critical: false, negative: true }];
  const calls = [['read', { path: 'current-guidance.json' }], ['read', { path: 'current-usage.json' }],
    ['mcp__kdb__read_entry', { collection: seed.collection, entry_id: seed.chunk_id }],
    ['submit_changes', { changes: [{ action: 'remove', id: guidance.id }] }],
    ['submit_changes', { changes: proposal }]];
  let requests = 0;
  const server = createServer(async (request, response) => {
    let body = ''; for await (const chunk of request) body += chunk;
    const input = JSON.parse(body);
    const i = requests++;
    assert.ok(!input.tools.some(tool => ['write', 'edit', 'bash', 'rsi_target', 'mcp__kdb__record_lesson'].includes(tool.function.name)));
    const call = calls[i];
    const delta = call ? { tool_calls: [{ index: 0, id: `tool-${i}`, type: 'function',
      function: { name: call[0], arguments: JSON.stringify(call[1]) } }] } : { content: 'Done' };
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    const chunk = (delta, finish_reason = null) => 'data: ' + JSON.stringify({
      id: `proposal-${i}`, object: 'chat.completion.chunk', created: 1, model: 'proposal',
      choices: [{ index: 0, delta, finish_reason }] }) + '\n\n';
    response.end(chunk(delta) + chunk({}, call ? 'tool_calls' : 'stop') + 'data: [DONE]\n\n');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => { server.closeAllConnections(); return new Promise(resolve => server.close(resolve)); });
  const modelRuntime = await sdk.ModelRuntime.create({ credentials: new InMemoryCredentialStore(),
    modelsPath: null, refreshOnCreate: false, allowModelNetwork: false });
  modelRuntime.registerProvider('skill-test', {
    baseUrl: `http://127.0.0.1:${server.address().port}/v1`, api: 'openai-completions',
    models: [{ id: 'proposal', name: 'Proposal test', reasoning: false, input: ['text'],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 32768, maxTokens: 1024 }]
  });
  await modelRuntime.setRuntimeApiKey('skill-test', 'local-test-placeholder');
  const extension = sdk.createMcpExtension({ loadConfig: () => ({ servers: [{ name: 'kdb',
    source: join(directory, 'agent', 'mcp.json'), scope: 'extension', config: {
      url: `http://127.0.0.1:${port}/mcp`, headers: { 'X-Attempt-Id': 'writer' }, exposure: 'direct'
    } }], errors: [], autoEnableCodemode: false }), logPath: join(directory, 'mcp.log') });
  const submitted = await generateCandidate({ directory, input: { entries: [guidance], history: [],
    context: task(0), actualEvents: task(0).events }, modelRuntime,
    model: modelRuntime.getModel('skill-test', 'proposal'), extensions: [extension],
    validate: changes => { if (changes.some(change => change.action === 'remove')) throw new Error('Protected guidance'); } });
  assert.deepEqual(submitted, proposal);
  assert.equal(requests, 6);
  const messages = await load(join(directory, 'responses.json'));
  const toolResults = messages.filter(message => message.role === 'toolResult');
  assert.ok(toolResults.some(message => message.toolName === 'submit_changes' && message.isError));
  const lookup = toolResults.find(message => message.toolName === 'mcp__kdb__read_entry');
  assert.equal(JSON.parse(lookup.content[0].text).text, evidenceText);
  assert.ok((await readFile(join(directory, 'sft.native.jsonl'), 'utf8')).length > 0);
  assert.ok((await readFile(join(directory, 'provider.jsonl'), 'utf8')).length > 0);
  const count = execFileSync(python, ['-c',
    "import sqlite3,sys;c=sqlite3.connect(sys.argv[1]);print(c.execute(\"SELECT count(*) FROM knowledge_events WHERE operation='read_entry' AND attempt_id='writer'\").fetchone()[0])",
    join(root, 'knowledge.sqlite')], { encoding: 'utf8' });
  assert.equal(count.trim(), '1');
});

test('native writer elapsed allocation aborts and retains response evidence', async t => {
  const root = await fixture(t);
  const { native, flags } = nativeWriter(() => {}, true);
  await assert.rejects(generateCandidate({ directory: join(root, 'timed'), input: {}, native,
    seconds: 0.02 }), /elapsed allocation/);
  assert.ok(flags.aborted && flags.disposed);
  assert.equal((await load(join(root, 'timed', 'responses.json'))).length, 1);
});

test('external stop aborts native session, rejects submission and preserves partial messages', async t => {
  const root = await fixture(t), controller = new AbortController();
  const { native, flags } = nativeWriter(() => setTimeout(() => controller.abort(), 0), true);
  await assert.rejects(generateCandidate({ directory: join(root, 'stopped'), input: {}, native,
    signal: controller.signal }), { name: 'AbortError' });
  assert.ok(flags.aborted && flags.disposed);
  assert.equal((await load(join(root, 'stopped', 'responses.json'))).length, 1);
  await assert.rejects(generateCandidate({ directory: join(root, 'never-started'), input: {}, native,
    signal: controller.signal }), { name: 'AbortError' });
});

test('elapsed expiry during extension binding cannot start a later prompt', async t => {
  const root = await fixture(t), { native, flags } = nativeWriter();
  const create = native.createAgentSession; let prompts = 0;
  native.createAgentSession = async options => {
    const result = await create(options);
    result.session.bindExtensions = async () => new Promise(resolve => setTimeout(resolve, 150));
    result.session.prompt = async () => { prompts++; };
    return result;
  };
  await assert.rejects(generateCandidate({ directory: join(root, 'binding'), input: {}, native,
    seconds: 0.1 }), /elapsed allocation/);
  assert.equal(prompts, 0); assert.ok(flags.aborted && flags.disposed);
});

test('archive failures still shut down and dispose, preserving the earliest error', async t => {
  for (const promptFails of [false, true]) {
    const root = await fixture(t), directory = join(root, 'archive'), events = [];
    const { native } = nativeWriter(), create = native.createAgentSession, first = new Error('provider failed');
    native.createAgentSession = async options => {
      const result = await create(options), { session } = result, prompt = session.prompt;
      session.prompt = async () => { await prompt(); await mkdir(join(directory, 'responses.json'));
        if (promptFails) throw first; };
      session.extensionRunner = { async emit() { events.push('shutdown'); throw new Error('cleanup failed'); } };
      session.dispose = () => events.push('dispose');
      return result;
    };
    await assert.rejects(generateCandidate({ directory, input: {}, native }),
      error => promptFails ? error === first : error.code === 'EISDIR');
    assert.deepEqual(events, ['shutdown', 'dispose']);
    assert.equal((await readFile(join(directory, 'sft.native.jsonl'), 'utf8')).length > 0, true);
  }
});
