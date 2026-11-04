/** Installed-package acceptance; all instance inputs and processes live outside the checkout. */
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {createServer} from 'node:net';
import {DatabaseSync} from 'node:sqlite';
import {mkdtemp, mkdir, writeFile, readFile, cp, lstat, realpath} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, dirname, delimiter, relative, isAbsolute} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';
import {provider} from './driver_fixture.mjs';

const checkout = fileURLToPath(new URL('../..', import.meta.url));
const parse = async path => JSON.parse(await readFile(path, 'utf8'));
const quote = value => "'" + value.replaceAll("'", "'\\''") + "'";
const inside = (root, path) => {
  const rel = relative(root, path);
  return rel === '' || (!isAbsolute(rel) && rel !== '..' && !rel.startsWith('../'));
};

async function until(check, description, seconds = 30, signal) {
  const deadline = performance.now() + seconds * 1000;
  while (performance.now() < deadline) {
    signal?.throwIfAborted();
    const result = await check();
    if (result) return result;
    await delay(50);
  }
  throw new Error(`Deadline waiting for ${description}`);
}

function rows(instance, sql) {
  const db = new DatabaseSync(join(instance, 'state.sqlite'), {readOnly: true});
  db.exec('PRAGMA busy_timeout=5000');
  try {return db.prepare(sql).all();} finally {db.close();}
}

function view(status) {
  return {
    controller: status.controller,
    projects: status.projects,
    attempts: status.attempts.map(({id, project, outcome}) => ({id, project, outcome})),
    requests: status.requests,
  };
}

export async function runInstalledPackagingCase({
  artifactRoot = process.env.INSTALLED_ARTIFACTS ?? tmpdir(),
  python = process.env.DRIVER_PYTHON ?? process.env.PYTHON ?? 'python3.12',
  signal, onCleanup = () => {}, event = async () => {},
} = {}) {
  signal?.throwIfAborted();
  await mkdir(artifactRoot, {recursive: true});
  const root = await mkdtemp(join(artifactRoot, 'ascend-installed-'));
  assert.equal(inside(await realpath(checkout), await realpath(root)), false,
    'Runtime artifacts must be outside the checkout');
  const assets = join(root, 'assets'), outside = join(root, 'clean launcher cwd');
  const host = join(root, 'package host'), instance = join(root, 'instance');
  for (const path of [assets, outside, host]) await mkdir(path);
  const env = {...process.env, INSTALLED_FIXTURE_KEY: 'fixture-only',
    PATH: dirname(process.execPath) + delimiter + process.env.PATH};
  delete env.NODE_PATH;
  delete env.PYTHONPATH;
  let sequence = 0, dashboard, fixture, cli, ownerStopped, cleaning = false, cleanupPromise;
  const commands = [], snapshots = [];
  const launch = (command, args, {cwd = outside, seconds = 180} = {}) => {
    if (!cleaning) signal?.throwIfAborted();
    const prefix = join(root, `process-${sequence++}`);
    const child = spawn(command, args, {cwd, env, detached: true, stdio: ['ignore', 'pipe', 'pipe']});
    let stdout = '', stderr = '';
    let expired = false, forced;
    const terminate = () => {
      if (child.exitCode !== null || child.signalCode !== null) return;
      try {process.kill(-child.pid, 'SIGTERM');} catch {}
      forced ??= setTimeout(() => {try {process.kill(-child.pid, 'SIGKILL');} catch {}}, 5000);
    };
    const abort = () => terminate();
    if (!cleaning) signal?.addEventListener('abort', abort, {once: true});
    if (!cleaning && signal?.aborted) abort();
    child.stdout.on('data', data => {stdout += data;});
    child.stderr.on('data', data => {stderr += data;});
    const timer = setTimeout(() => {
      expired = true;
      terminate();
    }, seconds * 1000);
    const output = (async () => {
      let code;
      try {[code] = await once(child, 'close');} finally {
        clearTimeout(timer); clearTimeout(forced); signal?.removeEventListener('abort', abort);
      }
      await writeFile(prefix + '.stdout', stdout);
      await writeFile(prefix + '.stderr', stderr);
      commands.push({command, args, cwd, pid: child.pid, code, expired,
        stdout: prefix + '.stdout', stderr: prefix + '.stderr'});
      await event({type: 'installed-process-closed', pid: child.pid, code, evidence: prefix});
      if (!cleaning) signal?.throwIfAborted();
      assert.equal(expired, false, `Command deadline expired; artifacts=${root}`);
      assert.equal(code, 0, `${command} ${args.join(' ')}\n${stderr}\nartifacts=${root}`);
      return stdout;
    })();
    // Long-lived dashboard output is awaited during cleanup; retain any rejection meanwhile.
    void output.catch(() => {});
    return {child, output, terminate};
  };
  const exec = async (command, args, options) => launch(command, args, options).output;
  const run = async (command, ...args) => JSON.parse(await exec(process.execPath,
    [cli, command, '--instance', instance, ...args]));
  const request = async (path, method = 'GET', body) => {
    const response = await fetch(dashboard.url + path, {method,
      ...(body === undefined ? {} : {headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}),
      signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(10000)]) : AbortSignal.timeout(10000)});
    assert.equal(response.status, 200, await response.clone().text());
    return response.json();
  };
  const compareStatus = async label => {
    const direct = await run('status'), http = await request('/api/status');
    assert.deepEqual(view(http), view(direct), `${label}: dashboard must reflect real installed CLI state`);
    snapshots.push({label, direct, http});
    await event({type: 'installed-status', label, controller: http.controller,
      projects: http.projects, attempts: http.attempts});
    await writeFile(join(root, 'status-transitions.json'), JSON.stringify(snapshots, null, 2));
    return http;
  };
  let result, failure;
  const cleanup = () => cleanupPromise ??= (async () => {
    cleaning = true;
    const errors = [];
    const closure = {};
    try {
      if (cli && ownerStopped) {
        await run('stop');
        await until(async () => (await run('status')).controller.state === 'stopped', 'cleanup controller');
        closure.controllerStopped = true;
        try {
          const service = await parse(join(instance, 'knowledge-service.json'));
          if (!await ownerStopped(service.owner)) {
            process.kill(service.owner.pid, 'SIGTERM');
            await until(() => ownerStopped(service.owner), 'owned KDB shutdown');
          }
          closure.knowledgeOwner = service.owner;
          closure.knowledgeStopped = await ownerStopped(service.owner);
        } catch (error) {if (error.code !== 'ENOENT' && error.code !== 'ESRCH') throw error;}
      }
    } catch (error) {errors.push(error);}
    if (dashboard) {
      dashboard.terminate();
      try {await dashboard.output;} catch (error) {
        if (!signal?.aborted) errors.push(error);
      }
      closure.dashboard = {pid: dashboard.child.pid, code: dashboard.child.exitCode,
        signal: dashboard.child.signalCode};
    }
    if (fixture) try {
      await writeFile(join(root, 'fixture-provider-requests.json'), JSON.stringify(fixture.requests, null, 2));
      await fixture.close(); closure.providerClosed = true;
    } catch (error) {errors.push(error);}
    closure.errors = errors.map(e => e.message);
    await writeFile(join(root, 'closure.json'), JSON.stringify(closure, null, 2));
    await writeFile(join(root, 'commands.json'), JSON.stringify(commands, null, 2));
    if (errors.length) throw new AggregateError(errors, 'Installed fixture cleanup failed');
  })();
  onCleanup(cleanup);
  try {
    const packed = JSON.parse(await exec('npm', ['pack', '--json', '--pack-destination', assets], {cwd: checkout}));
    const tarball = join(assets, packed[0].filename);
    await writeFile(join(host, 'package.json'), JSON.stringify({name: 'installed-acceptance-host', private: true}));
    await exec('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', '--prefix', host, tarball]);
    const packageRoot = join(host, 'node_modules', 'ascend-rsi');
    const packageCli = join(packageRoot, 'cli', 'main.mjs');
    cli = packageCli;
    const installedBin = join(host, 'node_modules', '.bin', 'ascend-rsi');
    assert.equal(await realpath(installedBin), await realpath(packageCli));
    const hostSdk = join(host, 'node_modules', '@earendil-works', 'pi-coding-agent');
    assert.equal((await lstat(hostSdk)).isSymbolicLink(), false);
    assert.ok(inside(host, await realpath(hostSdk)));

    // Reuse the existing deterministic embeddings and independent numeric oracle as input assets.
    const fixturePython = join(assets, 'installed_fixture.py');
    await cp(join(checkout, 'tests', 'ascend', 'driver_fixture.py'), fixturePython);
    python = (await exec(python, ['-c', 'import sys; print(sys.executable)'])).trim();
    env.PATH = dirname(process.execPath) + delimiter + dirname(python) + delimiter + process.env.PATH;
    env.PYTHONPATH = packageRoot;
    const baseline = join(assets, 'baseline.sqlite');
    await exec(python, [fixturePython, 'baseline', baseline]);
    const skills = join(assets, 'skills');
    await mkdir(skills);
    const scopes = ['preflight', 'task-construction', 'design', 'implementation', 'test',
      'debug', 'optimisation', 'validation', 'acceptance', 'global-kdb'];
    for (const scope of scopes) await writeFile(join(skills, scope + '.md'),
      'Read the current skill. Retrieve good and negative evidence in design and optimisation; use the frozen verifier.');
    await writeFile(join(skills, 'skill-map.json'), JSON.stringify(Object.fromEntries(scopes.map(s => [s, s + '.md']))));
    const source = join(assets, 'project');
    await mkdir(source);
    await writeFile(join(source, 'answer.py'), 'def answer(): return 0\n');
    fixture = await provider('interrupt', {nativeSession: async () =>
      rows(instance, 'SELECT session FROM attempts WHERE session IS NOT NULL LIMIT 1')[0]?.session});
    const settings = {
      baseline, skills, target: 'cpu', python, embedding_factory: 'installed_fixture:Embeddings',
      actor_seconds: 120, investigation_seconds: 120, curriculum_seconds: 30,
      concurrency: 1,
      environment: {module: 'machinery/integrations/environments/index.mjs', options: {mode: 'local'}},
      projects: [{id: 'installed', goal: 'installed arithmetic', source, actor_seconds: 120,
        verifier: {files: [fixturePython], command: `${quote(python)} ${quote(fixturePython)} verify ${quote(source)}`}}],
      actor: {provider: 'installed-fixture', model: 'scripted', credential_env: 'INSTALLED_FIXTURE_KEY', base_url: fixture.url,
        metadata: {id: 'scripted', name: 'Installed fixture', api: 'openai-completions', reasoning: false,
          input: ['text'], contextWindow: 128000, maxTokens: 8192,
          cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}}},
    };
    const input = join(assets, 'bootstrap.json');
    await writeFile(input, JSON.stringify(settings));
    // Public package entry point; default install=true performs its own normal npm ci.
    await exec(process.execPath, [installedBin, 'bootstrap', instance, '--config', input]);
    cli = join(instance, 'machinery', 'cli', 'main.mjs');
    const machinery = join(instance, 'machinery');
    await cp(fixturePython, join(instance, 'installed_fixture.py'));
    env.PYTHONPATH = machinery;
    const sdkPath = join(machinery, 'node_modules', '@earendil-works', 'pi-coding-agent');
    assert.equal((await lstat(sdkPath)).isSymbolicLink(), false);
    assert.ok(inside(machinery, await realpath(sdkPath)));
    assert.notEqual(await realpath(sdkPath), await realpath(hostSdk));
    ({ownerStopped} = await import(pathToFileURL(join(machinery, 'controller', 'lifecycle.mjs'))));
    const socket = createServer();
    socket.listen(0, '127.0.0.1'); await once(socket, 'listening');
    const port = socket.address().port;
    await new Promise(resolve => socket.close(resolve));
    const server = launch(process.execPath, [cli, 'dashboard', '--instance', instance, '--port', String(port)], {seconds: 180});
    dashboard = {...server, url: `http://127.0.0.1:${port}`};
    await until(async () => {
      if (dashboard.child.exitCode !== null) throw new Error(`Dashboard exited; artifacts=${root}`);
      try {return (await fetch(dashboard.url + '/api/status', {signal: AbortSignal.timeout(2000)})).ok;}
      catch {return false;}
    }, 'real installed dashboard HTTP startup', 30, signal);
    for (const [route, file] of [['/', 'index.html'], ['/assets/app.js', 'app.js']]) {
      const response = await fetch(dashboard.url + route);
      assert.equal(response.status, 200);
      assert.equal(await response.text(), await readFile(join(machinery, 'web', file), 'utf8'));
    }
    const queued = await compareStatus('queued');
    assert.equal(queued.projects[0].status, 'queued');
    assert.equal(queued.attempts.length, 0);
    const wave = await request('/api/wave', 'POST', {mode: 'phase1'});
    assert.equal(wave.submitted, true);
    await until(() => {
      const [attempt] = rows(instance, 'SELECT * FROM attempts');
      return attempt?.elapsed >= 1 && attempt.session &&
        rows(instance, "SELECT 1 FROM events WHERE type='execution-handle'").length;
    }, 'native actor target command before stop', 45, signal);
    const running = await compareStatus('running');
    assert.equal(running.controller.state, 'running');
    assert.equal(running.projects[0].status, 'running');
    assert.equal(running.attempts[0].outcome, null);
    const original = rows(instance, 'SELECT * FROM attempts')[0];
    await request('/api/stop', 'POST');
    await until(async () => (await run('status')).controller.state === 'stopped', 'controller checkpoint', 30, signal);
    const held = await compareStatus('held');
    assert.equal(held.controller.stopRequested, true);
    assert.equal(held.attempts[0].id, original.id);
    assert.equal(held.attempts[0].outcome, null);
    const checkpoint = rows(instance, 'SELECT * FROM attempts')[0];
    assert.equal(checkpoint.session, original.session);
    assert.ok(checkpoint.elapsed >= original.elapsed);
    const resumed = await request('/api/resume', 'POST');
    assert.equal(resumed.submitted, true);
    await until(async () => {
      const state = await run('status');
      if (state.attempts[0]?.outcome === 'failed') throw new Error(`Installed actor failed; artifacts=${root}`);
      return state.controller.state === 'stopped' && state.attempts[0]?.outcome === 'passed';
    }, 'same installed actor pass', 60, signal);
    const passed = await compareStatus('passed');
    assert.equal(passed.projects[0].status, 'passed');
    assert.equal(passed.attempts.length, 1);
    assert.equal(passed.attempts[0].id, original.id);
    assert.ok(passed.attempts[0].elapsed >= checkpoint.elapsed);
    assert.equal(rows(instance, 'SELECT session FROM attempts')[0].session, original.session);
    const events = rows(instance, 'SELECT type,data FROM events');
    assert.equal(events.filter(e => e.type === 'accepted').length, 1);
    assert.ok(events.some(e => e.type === 'verification' && JSON.parse(e.data).passed));
    assert.ok(events.some(e => e.type === 'memory-updated'));
    assert.equal(rows(instance, 'SELECT * FROM recovery_lessons WHERE recorded=0').length, 0);
    assert.deepEqual(fixture.failures, []);
    assert.ok(fixture.requests.some(r => r.messages.some(m => m.role === 'tool')));
    result = {passed: true, artifacts: root, tarball, packIntegrity: packed[0].integrity, installedCli: cli,
      dashboard: {host: '127.0.0.1', port}, transitions: snapshots.map(s => s.label),
      sameAttempt: original.id, independentSdkInstalls: [await realpath(hostSdk), await realpath(sdkPath)]};
  } catch (error) {
    failure = error;
    await writeFile(join(root, 'failure.txt'), error.stack ?? String(error));
  } finally {
    try {await cleanup();} catch (error) {failure ??= error;}
    await writeFile(join(root, 'result.json'), JSON.stringify(failure ?
      {passed: false, error: failure.message, partial: result} : result, null, 2));
  }
  if (failure) throw failure;
  return result;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const abort = new AbortController();
  const stop = () => abort.abort(new Error('Installed acceptance interrupted'));
  process.on('SIGTERM', stop); process.on('SIGINT', stop);
  runInstalledPackagingCase({signal: abort.signal}).then(result => console.log(JSON.stringify(result))).catch(error => {
    console.error(error.stack); process.exitCode = 1;
  }).finally(() => {process.off('SIGTERM', stop); process.off('SIGINT', stop);});
}
