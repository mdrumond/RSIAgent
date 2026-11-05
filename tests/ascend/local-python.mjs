import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, writeFile, readFile, readdir, access } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import { setup, rows, until } from './driver_setup.mjs';
import { stages } from '../../controller/workflow.mjs';
import { readConfig } from '../../cli/instance.mjs';
import { run as tune } from '../../cli/skills.mjs';
import * as runtime from '../../cli/evaluation.mjs';
import { ownerStopped } from '../../controller/lifecycle.mjs';

const python = process.env.DRIVER_PYTHON ?? 'python3.12';
const oracle = resolve('examples/local-python/tasks.py');
const load = async path => JSON.parse(await readFile(path, 'utf8'));
const source = variant => execFileSync(python, [oracle, 'source', variant], { encoding: 'utf8' });
const tool = (name, args = {}) => ({ name, args });
const toolData = message => {
  try { return JSON.parse(typeof message.content === 'string' ? message.content :
    message.content.map(block => block.text ?? '').join('')); } catch { return null; }
};

function providerOptions(candidate, control) {
  return {
    writerChanges: n => n ? [] : [{ action: 'upsert', id: 'python-boundaries', scope: 'implementation',
      text: `Use the following implementation for these Python boundary contracts.\n\n\`\`\`python\n${candidate}\`\`\``,
      critical: false, negative: false }],
    workflowForGoal() {
      const calls = [];
      for (const stage of stages) {
        calls.push(tool('rsi_skill'));
        if (['design', 'optimisation'].includes(stage)) for (const kind of ['good', 'negative'])
          calls.push(tool('rsi_knowledge', { query: 'Python boundary cases', kind }));
        if (stage === 'implementation') calls.push(tool('write', { path: 'answer.py', content: '' }),
          tool('rsi_target', { command: 'python -B -c "import answer; print(answer.reduce_tiles([1,2,3,4,5]))"' }));
        if (stage === 'validation') calls.push(tool('rsi_verify'));
        calls.push(tool('rsi_advance', { reflection: 'Use independent frozen boundary-case evidence.' }));
      }
      return calls;
    },
    resolveCall(call, body) {
      const content = body.messages.find(message => message.role === 'user')?.content;
      const prompt = typeof content === 'string' ? content : (content ?? []).map(block => block.text ?? '').join('');
      if (!prompt.startsWith('Goal:')) return call;
      const values = body.messages.filter(message => message.role === 'tool').map(toolData).filter(Boolean);
      // Interrupt only between completed tools, before emitting another assistant call.
      // Counting emitted calls cannot recover a deliberately half-delivered tool turn.
      if (values.at(-1)?.stage === 'preflight' && values.at(-1)?.skill && control.onActor) {
        control.onActor(); return false;
      }
      if (values.at(-1)?.passed === false) return undefined;
      if (call?.name !== 'write') return call;
      const guidance = values.findLast(value => value.stage === 'implementation' && value.skill)?.skill ?? '';
      const code = /```python\n([\s\S]*?)```/.exec(guidance)?.[1];
      return tool('write', { ...call.args, content: code ?? source('baseline') });
    }
  };
}

async function fixture(t, variant) {
  const control = {};
  const f = await setup(t, 'local-python', ['reduction', 'parser'], { install: false,
    providerOptions: providerOptions(source(variant), control),
    projectFactory: async ({ id, root, python, quote }) => {
      const sourceRoot = join(root, id, 'source'); await mkdir(sourceRoot, { recursive: true });
      await writeFile(join(sourceRoot, 'answer.py'), source('baseline'));
      return { id, goal: `Implement ${id} boundary cases in answer.py`, source: sourceRoot, actor_seconds: 60,
        verifier: { files: [oracle], command: `${quote(python)} -B ${quote(oracle)} verify ${id} ${quote(sourceRoot)}` } };
    } });
  const configure = join(f.root, 'phase2.json'), current = await readConfig(f.root);
  const credentials = join(f.root, 'fixture.env');
  await writeFile(credentials, 'DRIVER_FIXTURE_KEY=fixture-only\n');
  await writeFile(configure, JSON.stringify({ mode: 'phase2', tune_every: 2,
    actor: { ...current.actor, credentials_file: credentials },
    curriculum: { phase2: ['reduction', 'parser'] } }));
  await f.run('config', '--config', configure);
  const cleanupSamples = async () => {
    try { await access(join(f.root, 'state.sqlite')); } catch (error) { if (error.code === 'ENOENT') return; throw error; }
    for (const event of rows(f.root, "SELECT data FROM events WHERE type='evaluation-started'")) {
      const root = JSON.parse(event.data).instanceRoot;
      try {
        const receipt = await load(join(root, 'knowledge-service.json'));
        if (!await ownerStopped(receipt.owner)) process.kill(receipt.owner.pid, 'SIGTERM');
      } catch (error) { if (!['ENOENT', 'ESRCH'].includes(error.code)) throw error; }
    }
  };
  t.after(cleanupSamples);
  await f.run('wave', '--mode', 'phase2');
  assert.deepEqual(rows(f.root, 'SELECT outcome FROM attempts').map(row => row.outcome), ['failed', 'failed']);
  assert.equal(rows(f.root, "SELECT count(*) AS n FROM events WHERE type='evaluation-started'")[0].n, 0);
  const proposals = await readdir(join(f.root, 'skills/proposals'));
  assert.equal(proposals.length, 2);
  for (const proposal of proposals) {
    const evidence = await load(join(f.root, 'skills/proposals', proposal, 'comparison-evidence.json'));
    assert.equal(evidence.current.outcome, 'failed'); assert.deepEqual(evidence.comparisons, {});
    assert.ok(evidence.current.references.some(reference => reference.type === 'verification'));
  }
  assert.equal((await f.run('status')).tuning.status, 'deferred');
  return { ...f, control, cleanupSamples };
}

for (const variant of ['repair', 'regression']) test(`local Python ${variant}: failure-only learning and canonical evaluation`,
  { timeout: 180000 }, async t => {
    const f = await fixture(t, variant), abort = new AbortController();
    const roots = [], resumes = [];
    const wrapped = { ...runtime,
      async wave(request) {
        roots.push(request.instanceRoot);
        if (variant === 'repair' && roots.length === 2) f.control.onActor = () => {
          f.control.onActor = null; abort.abort();
        };
        return runtime.wave(request);
      },
      async resume(request) { resumes.push(request.instanceRoot); return runtime.resume(request); }
    };
    const grant = { evaluationSeconds: 120, allocationId: `local-${variant}` };
    let result, interrupted;
    if (variant === 'repair') {
      await assert.rejects(tune(await readConfig(f.root), { ...grant, runtime: wrapped, signal: abort.signal }));
      const status = (await f.run('status')).tuning;
      assert.equal(status.status, 'interrupted'); assert.equal(status.completedSamples, 1);
      const journal = await load(join(status.evaluationRoot, 'round.json'));
      interrupted = journal.samples[1].instanceRoot;
      const retained = rows(interrupted, 'SELECT id,session FROM attempts')[0];
      assert.ok(retained.session);
      assert.equal(rows(interrupted, "SELECT count(*) AS n FROM events WHERE type='skill-read' AND json_extract(data,'$.stage')='preflight'")[0].n, 1);
      await until(async () => (await Promise.all(rows(interrupted, 'SELECT owner FROM attempts')
        .map(row => ownerStopped(JSON.parse(row.owner))))).every(Boolean));
      result = await tune(await readConfig(f.root), { ...grant, runtime: wrapped });
      assert.deepEqual(resumes, [interrupted]); assert.equal(roots.length, 8);
      assert.deepEqual(rows(interrupted, 'SELECT id,session FROM attempts')[0], retained);
    } else result = await tune(await readConfig(f.root), { ...grant, runtime: wrapped });
    const journal = await load(join(result.evaluationRoot, 'round.json'));
    const diagnostic = JSON.stringify({ reason: result.reason, status: result.evaluationStatus,
      samples: journal.samples.map(({ projectId, run, arm, result }) => ({ projectId, run, arm, result })) });
    assert.equal(result.evaluationStatus, 'completed', diagnostic);
    assert.equal(result.promote, variant === 'repair', diagnostic);
    assert.equal(result.reason, variant === 'repair' ? 'reliability-improvement' : 'correctness-regression', diagnostic);
    assert.equal(journal.samples.filter(sample => sample.result).length, 8);
    assert.equal(journal.allocatedSeconds, 120);
    assert.equal(new Set(journal.samples.map(sample => sample.instanceRoot)).size, 8);
    for (const sample of journal.samples) {
      assert.equal(rows(sample.instanceRoot, 'SELECT count(*) AS n FROM attempts')[0].n, 1);
      const verdicts = rows(sample.instanceRoot, "SELECT data FROM events WHERE type='verification'");
      assert.equal(verdicts.length, 1);
      const actual = JSON.parse(verdicts[0].data);
      assert.equal(actual.passed, sample.result.correct);
      assert.equal(sample.result.correct, sample.arm === 'candidate' &&
        (variant === 'repair' || sample.projectId === 'reduction'));
      assert.ok((await readFile(actual.evidence.correctness, 'utf8')).length > 0);
    }
    assert.equal((await readConfig(f.root)).skills.activeVersion === 'baseline', variant !== 'repair');
    await f.cleanupSamples(); f.pass();
  });
