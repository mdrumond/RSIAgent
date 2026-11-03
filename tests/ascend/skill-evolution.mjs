import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { cadenceDecision, evaluateDecision, applyProposal, generateCandidate, tune } from '../../controller/skill-evolution.mjs';
import { run } from '../../cli/skills.mjs';

const task = (n, extra = {}) => ({ id: `original-${n}`, projectId: `project-${n}`,
  kind: 'original', phase: 'phase2', completed: true,
  events: [{ type: 'knowledge-retrieved', data: { kind: 'negative', evidence: 'failure.json' } }], ...extra });
const history = n => Array.from({ length: n }, (_, i) => task(i));
const guidance = { id: 'failure', scope: 'debug', text: 'Check the failing case.', critical: false, negative: true };
const config = { root: tmpdir(), mode: 'phase2', skills: { activeVersion: 'baseline', tuningProjectIds: ['reg-a', 'reg-b'] } };
async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'skills-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'skills', 'versions', 'baseline'), { recursive: true });
  await writeFile(join(root, 'instance.json'), JSON.stringify(config));
  await writeFile(join(root, 'skills', 'versions', 'baseline', 'guidance.json'), JSON.stringify([guidance]));
  return root;
}
const load = async path => JSON.parse(await readFile(path, 'utf8'));

test('cadence excludes phase1, retries, diagnostics, duplicates and repeated projects', () => {
  for (const extra of [{ phase: 'phase1' }, { completed: false }])
    assert.equal(cadenceDecision(history(7), task(7, extra)).analyse, false);
  for (const kind of ['retry', 'diagnostic', 'investigation']) {
    const decision = cadenceDecision(history(7), task(7, { kind }));
    assert.equal(decision.analyse, true);
    assert.equal(decision.evaluate, false);
    assert.equal(decision.projects, 7);
  }
  assert.equal(cadenceDecision(history(7), task(0)).analyse, false);
  assert.equal(cadenceDecision(history(7), task(7, { projectId: 'project-0' })).evaluate, false);
  assert.equal(cadenceDecision(history(7), task(7)).evaluate, true);
  assert.equal(cadenceDecision(history(23), task(23)).allowPruning, true);
  assert.equal(cadenceDecision(history(22), task(22)).allowPruning, false);
  assert.equal(cadenceDecision(history(8), task(8), 8).evaluate, false);
});

test('paired decision rejects correctness loss, missing samples, ties and regressions', () => {
  const results = ['reg-a', 'reg-b'].flatMap(projectId => [0, 1].flatMap(run =>
    ['active', 'candidate'].map(arm => ({ projectId, run, arm, correct: true, score: arm === 'candidate' ? 2 : 1 }))));
  assert.equal(evaluateDecision(results, config.skills.tuningProjectIds).promote, true);
  for (const patch of [{ correct: false }, { score: 1 }, { score: 0 }, { correct: undefined }]) {
    const changed = structuredClone(results);
    Object.assign(changed[1], patch);
    assert.equal(evaluateDecision(changed, config.skills.tuningProjectIds).promote, false);
  }
  assert.equal(evaluateDecision(results.slice(1), config.skills.tuningProjectIds).promote, false);
});

test('candidate repairs active failures before performance, but reverse never promotes', () => {
  const results = ['reg-a', 'reg-b'].flatMap(projectId => [0, 1].flatMap(run =>
    ['active', 'candidate'].map(arm => ({ projectId, run, arm, correct: true,
      score: arm === 'active' ? 2 : 1 }))));
  results[0].correct = false;
  delete results[0].score;
  assert.deepEqual(evaluateDecision(results, config.skills.tuningProjectIds),
    { promote: true, reason: 'reliability-improvement' });
  const reverse = results.map(r => ({ ...r, arm: r.arm === 'active' ? 'candidate' : 'active' }));
  assert.deepEqual(evaluateDecision(reverse, config.skills.tuningProjectIds),
    { promote: false, reason: 'correctness-regression' });
  results[3].correct = false;
  assert.equal(evaluateDecision(results, config.skills.tuningProjectIds).promote, false);
});

test('negative retention and evidence-gated critical pruning are enforced', () => {
  const remove = [{ action: 'remove', id: 'failure' }];
  const options = { allowPruning: true, history: history(24) };
  assert.throws(() => applyProposal([guidance], remove, options));
  const critical = { ...guidance, negative: false, critical: true };
  assert.throws(() => applyProposal([critical], remove, options));
  const contradictory = task(24, { events: [{ type: 'guidance-contradicted',
    data: { guidanceId: 'failure', evidence: 'verifier.json' } }] });
  assert.deepEqual(applyProposal([critical], [{ ...remove[0], contradictoryEvidence: ['verifier.json'] }],
    { ...options, history: [...options.history, contradictory] }), []);
  const ordinary = { ...critical, critical: false };
  assert.throws(() => applyProposal([ordinary], remove, { allowPruning: false, history: [] }));
  assert.deepEqual(applyProposal([ordinary], remove, options), []);
});

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

test('eight originals evaluate in isolated arms, preserve evidence and promote only passing candidate', async t => {
  const root = await fixture(t);
  const seen = [];
  const evaluate = async request => {
    seen.push(request);
    assert.equal(request.config.skills.tuningDisabled, true);
    await mkdir(request.instanceRoot, { recursive: true });
    await writeFile(join(request.instanceRoot, 'sft.json'), '["retained"]');
    return { correct: true, score: request.arm === 'candidate' ? 2 : 1 };
  };
  for (let i = 0; i < 8; i++) {
    const result = await tune({ instanceRoot: root, config, task: task(i), evaluate,
      generate: async ({ input }) => { assert.equal(input.actualEvents.length, 1); return []; } });
    if (i < 7) assert.equal(result.evaluate, false);
    else assert.equal(result.promote, true);
  }
  assert.equal(seen.length, 8);
  assert.equal(new Set(seen.map(x => x.instanceRoot)).size, 8);
  const active = (await load(join(root, 'instance.json'))).skills.activeVersion;
  assert.notEqual(active, 'baseline');
  assert.equal((await load(join(root, 'instance.json'))).revision, 1);
  assert.equal((await readdir(join(root, 'skills', 'versions'))).length, 9);
  assert.deepEqual(await load(join(seen[0].instanceRoot, 'sft.json')), ['retained']);
  assert.deepEqual(await load(join(root, 'skills', 'versions', 'baseline', 'guidance.json')), [guidance]);
});

test('inconclusive result keeps instance active; phase1 never invokes proposal or evaluator', async t => {
  const root = await fixture(t);
  await writeFile(join(root, 'skills', 'evolution.json'), JSON.stringify({ history: history(7), lastEvaluated: 0 }));
  const result = await tune({ instanceRoot: root, config, task: task(7), generate: async () => [],
    evaluate: async () => ({ correct: true, score: 1 }) });
  assert.equal(result.promote, false);
  assert.equal((await load(join(root, 'instance.json'))).skills.activeVersion, 'baseline');
  await tune({ instanceRoot: root, config, task: task(8, { phase: 'phase1' }),
    generate: async () => assert.fail('phase1 proposal') });
});

test('Phase1 CLI does not access action files, runtime or actor profile', async () => {
  assert.equal((await run({ mode: 'phase1' }, { config: '/absent' })).analyse, false);
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
    '--port', String(port), '--embedding-factory', 'kdb_fixture:Embeddings'], { env, stdio: 'ignore' });
  t.after(async () => { if (service.exitCode === null) {
    const { once } = await import('node:events'); const closed = once(service, 'exit'); service.kill(); await closed;
  } });
  const deadline = Date.now() + 5000;
  while (true) {
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
    assert.ok(!input.tools.some(tool => ['write', 'edit', 'bash', 'mcp__kdb__record_lesson'].includes(tool.function.name)));
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
    validate: changes => applyProposal([guidance], changes, { allowPruning: false, history: [] }) });
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

test('CLI uses instance.json and invokes bootstrap then canonical wave for each fresh arm', async t => {
  const root = await fixture(t);
  await writeFile(join(root, 'skills', 'evolution.json'), JSON.stringify({
    history: history(8), lastEvaluated: 0, candidate: 'baseline',
    pending: { active: 'baseline', candidate: 'baseline', projects: 8 }
  }));
  // Different immutable candidate prevents interrupted-promotion reconciliation.
  const pending = await load(join(root, 'skills', 'evolution.json'));
  pending.pending.candidate = 'candidate';
  await writeFile(join(root, 'skills', 'evolution.json'), JSON.stringify(pending));
  const calls = [];
  const runtime = {
    async bootstrap(request) { calls.push(['bootstrap', request.instanceRoot]); await mkdir(request.instanceRoot); },
    async wave(request) {
      calls.push(['wave', request.instanceRoot]);
      assert.deepEqual(request.projectIds.length, 1);
      assert.equal(request.evaluation, true);
      assert.equal(request.config.root, request.instanceRoot);
      return { correct: true, score: 1 };
    }
  };
  const result = await run({ ...config, root }, { runtime });
  assert.equal(result.promote, false);
  assert.equal(calls.length, 16);
  for (let i = 0; i < calls.length; i += 2) {
    assert.equal(calls[i][0], 'bootstrap');
    assert.equal(calls[i + 1][0], 'wave');
    assert.equal(calls[i][1], calls[i + 1][1]);
  }
});

test('CLI consumes action file task and requires an explicit writer profile', async t => {
  const root = await fixture(t);
  const action = join(root, 'action.json');
  await writeFile(action, JSON.stringify({ task: task(0), userinfo: 'Prefer recovery guidance.' }));
  await assert.rejects(run({ ...config, root }, { config: action,
    runtime: { async bootstrap() {}, async wave() {} } }), /skill_writer_actor or actor/);
  const proposals = await readdir(join(root, 'skills', 'proposals'));
  const evidence = await load(join(root, 'skills', 'proposals', proposals[0], 'evidence.json'));
  assert.equal(evidence.id, task(0).id);
  assert.equal((await run({ mode: 'phase1', root }, { config: '/missing-action.json' })).analyse, false);
});

const knowledgeStub = { async ensureKnowledgeService() { return 'http://127.0.0.1:8765/mcp'; },
  createKdbBridge() { return { extension: () => {} }; } };
function nativeWriter(onInput = () => {}, hang = false) {
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
      assert.ok(!options.tools.includes('write') && !options.tools.includes('bash'));
      return { session: {
        subscribe(fn) { listener = fn; return () => {}; }, async bindExtensions() {},
        async prompt() {
          const current = await load(join(directory, 'current-usage.json'));
          onInput({ ...current, entries: await load(join(directory, 'current-guidance.json')) });
          if (hang) await new Promise(resolve => { release = resolve; });
          else await tools.submit_changes.execute('submit', { changes: [] });
          listener({ type: 'message_end', message: { role: 'assistant', content: [] } });
        },
        async waitForIdle() {}, async abort() { flags.aborted = true; release?.(); },
        dispose() { flags.disposed = true; }
      } };
    }
  } };
}

test('first CLI update needs no A/B configuration; cadence defaults to two stable originals', async t => {
  const root = await fixture(t);
  const current = { ...config, root, skills: { activeVersion: 'baseline' } };
  await writeFile(join(root, 'instance.json'), JSON.stringify(current));
  const { native } = nativeWriter();
  const first = await run(current, { task: task(0), native, knowledge: knowledgeStub, modelRuntime: {}, model: {} });
  assert.ok(first.candidate);
  assert.equal(first.evaluate, false);
  const evaluated = [];
  for (let i = 1; i < 8; i++) await tune({ instanceRoot: root, config: current, task: task(i),
    generate: async () => [], evaluate: async request => {
      evaluated.push(request.projectId); return { correct: true, score: 1 };
    } });
  assert.deepEqual([...new Set(evaluated)], ['project-0', 'project-1']);
  assert.equal(evaluated.length, 8);
  assert.equal((await load(join(root, 'skills', 'evolution.json'))).history.length, 8);
});

test('manual CLI uses selected State events without increasing original cadence', async t => {
  const { State } = await import('../../controller/state.mjs');
  const root = await fixture(t);
  const state = new State(join(root, 'state.sqlite'));
  state.add({ id: 'selected', goal: 'numeric', actor_seconds: 20 });
  const attempt = state.admit('selected', { pid: 1 });
  state.event(attempt.id, 'knowledge-retrieved', { kind: 'negative', evidence: 'failed-case' });
  state.finish(attempt.id, 'failed'); state.close();
  const action = join(root, 'manual.json');
  await writeFile(action, JSON.stringify({ userinfo: 'repair', projectId: 'selected' }));
  let input;
  const { native } = nativeWriter(value => { input = value; });
  const result = await run({ ...config, root }, { config: action, native, knowledge: knowledgeStub, modelRuntime: {}, model: {},
    runtime: { async bootstrap(request) { await mkdir(request.instanceRoot); },
      async wave() { return { correct: true, score: 1 }; } } });
  assert.equal(result.promote, false);
  assert.equal(input.context.projectId, 'selected');
  assert.equal(input.actualEvents[0].data.evidence, 'failed-case');
  const saved = await load(join(root, 'skills', 'evolution.json'));
  assert.equal(saved.history.length, 0);
  assert.equal(saved.lastEvaluated, 0);
  assert.ok(saved.candidate);
  const noIds = { ...config, root, skills: { activeVersion: 'baseline' } };
  await writeFile(join(root, 'instance.json'), JSON.stringify(noIds));
  await writeFile(action, JSON.stringify({ userinfo: 'recover' }));
  assert.equal((await run(noIds, { config: action, native, knowledge: knowledgeStub, modelRuntime: {}, model: {} })).analyse, true);
  assert.equal((await load(join(root, 'skills', 'evolution.json'))).history.length, 0);
});

test('Markdown baseline seeds protected entries and preserves original relative assets in candidate', async t => {
  const { stages } = await import('../../controller/workflow.mjs');
  const root = await fixture(t);
  const baseline = join(root, 'skills', 'versions', 'baseline');
  await rm(join(baseline, 'guidance.json'));
  await mkdir(join(baseline, 'fork'));
  const map = {};
  for (const scope of ['global-kdb', ...stages]) {
    map[scope] = `fork/${scope}.md`;
    await writeFile(join(baseline, map[scope]), `# ${scope}\n[asset](reference.bin)`);
  }
  await writeFile(join(baseline, 'fork', 'reference.bin'), Buffer.from([1, 2, 3]));
  await writeFile(join(baseline, 'skill-map.json'), JSON.stringify(map));
  const result = await tune({ instanceRoot: root, config, task: task(0), generate: async ({ input }) => {
    assert.equal(input.entries.length, 10);
    assert.ok(input.entries.every(entry => entry.critical));
    return [];
  } });
  const candidate = join(root, 'skills', 'versions', result.candidate);
  const generatedMap = await load(join(candidate, 'skill-map.json'));
  assert.equal(generatedMap.design, 'fork/evolved-design.md');
  assert.deepEqual(await readFile(join(candidate, 'fork', 'reference.bin')),
    await readFile(join(baseline, 'fork', 'reference.bin')));
  assert.deepEqual(await readFile(join(candidate, map.design)), await readFile(join(baseline, map.design)));
  assert.equal((await load(join(candidate, 'guidance.json'))).length, 10);
  assert.equal((await load(join(root, 'instance.json'))).skills.activeVersion, 'baseline');
});

test('native writer elapsed allocation aborts and retains response evidence', async t => {
  const root = await fixture(t);
  const { native, flags } = nativeWriter(() => {}, true);
  await assert.rejects(generateCandidate({ directory: join(root, 'timed'), input: {}, native,
    seconds: 0.02 }), /elapsed allocation/);
  assert.ok(flags.aborted && flags.disposed);
  assert.equal((await load(join(root, 'timed', 'responses.json'))).length, 1);
});

test('investigation and retry update candidate but only originals advance cadence; state drops Pi telemetry', async t => {
  const root = await fixture(t);
  let analysed = 0;
  for (const [i, kind] of ['original', 'investigation', 'retry'].entries()) {
    const result = await tune({ instanceRoot: root, config, task: task(i, { kind,
      events: [...task(i).events, { type: 'pi-event', data: { duplicate: true } }] }),
      generate: async ({ input, validate }) => {
        analysed++;
        assert.ok(input.history.every(task => task.events.every(event => event.type !== 'pi-event')));
        validate([]); return [];
      } });
    assert.ok(result.candidate);
    assert.equal(result.evaluate, false);
    assert.equal(result.projects, 1);
  }
  assert.equal(analysed, 3);
  const state = await load(join(root, 'skills', 'evolution.json'));
  assert.equal(state.history.length, 3);
  const many = Array.from({ length: 30 }, (_, i) => task(i, { kind: 'investigation' }));
  assert.equal(cadenceDecision(many, task(31)).allowPruning, false);
});
