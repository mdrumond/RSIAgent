import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile, readFile, mkdir, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { createEnvironment, createEnvironmentTools, readProcessStat } from '../../integrations/environments/index.mjs';

async function fixture(t, override) {
  const instancePath = await mkdtemp(join(tmpdir(), 'environment-test-'));
  t.after(() => rm(instancePath, { recursive: true, force: true }));
  const project = join(instancePath, 'project');
  await mkdir(project);
  await writeFile(join(project, 'program.py'), 'version=1');
  const events = [], excluded = [], calls = [], versions = [];
  let output = 'answer\n';
  const sdk = { createLocalBashOperations: () => ({}), createBashTool: (cwd, options) => ({ cwd, ...options }) };
  const context = { instancePath, sdk, event: async (type, data) => { events.push({ type, ...data }); },
    exclude: async (label, fn) => { assert.ok(['setup','transfer'].includes(label)); excluded.push(label); return fn(); },
    client: async args => {
      calls.push(args);
      if (args[0] === 'run' && args[args.indexOf('--file') + 1].endsWith('sync.sh')) return { handle: 'remote:fake:job:sync' };
      if (args[0] === 'observe' && args[1] === 'remote:fake:job:sync') return { state: 'completed', exit: 0 };
      const overridden = override && await override(args, { events, versions });
      if (overridden !== undefined) return overridden;
      if (args[0] === 'capabilities') return { state: 'available' };
      if (args[0] === 'upload') { if(args[2]===project)versions.push(await readFile(join(args[2], 'program.py'), 'utf8')); return { state: 'completed', handle: 'remote:fake:transfer:copy' }; }
      if (args[0] === 'download') { await writeFile(args[3], args[2].endsWith('stdout') ? output : ''); return { state: 'completed' }; }
      if (args[0] === 'run') return { handle: 'remote:fake:job:one' };
      return { state: 'completed', exit: 0 };
    } };
  const options = { mode: 'cpl', target: 'fake', pollMs: 0, workspacePath: '/registered', remoteDirectory: 'task-scope' };
  const environment = await createEnvironment(options, context);
  await environment.prepare({ project: { id: 'p', sourceDirectory: project }, attempt: { id: 'attempt' } });
  return { environment, events, excluded, sdk, instancePath, project, calls, versions, setOutput: value => { output = value; }, context, options };
}

test('stable attempt directories map local cwd and sync edits; full output reaches native tool', async t => {
  const f = await fixture(t);
  const [tool] = await createEnvironmentTools(f.environment, f.project, f.sdk);
  const full = 'large output\n'.repeat(10000);
  f.setOutput(full);
  let output = '';
  const result = await tool.operations.exec('python3 program.py', f.project, { timeout: 4, env: { ...process.env, ENVIRONMENT_TEST_VALUE: 'tool-override' }, onData: data => { output += data; } });
  assert.equal(result.exitCode, 0);
  assert.equal(output, full);
  await writeFile(join(f.project, 'program.py'), 'version=2');
  await tool.operations.exec('python3 program.py', f.project, { timeout: 4 });
  assert.deepEqual(f.versions, ['version=1', 'version=2']);
  const runs = f.calls.filter(args => args[0] === 'run' && args[args.indexOf('--file') + 1].endsWith('command.sh'));
  assert.equal(runs.length, 2);
  assert.ok(runs.every(args => !args.includes('--cwd') && !args.includes('--timeout')));
  assert.notEqual(f.calls.filter(args => args[0] === 'upload')[0][3], f.calls.filter(args => args[0] === 'upload')[1][3]);
  assert.equal(f.events.filter(e => e.type === 'execution-output').every(e => e.partial === false), true);
  assert.ok(f.excluded.includes('setup') && f.excluded.includes('transfer'));
  const receipt = JSON.parse(await readFile(join(f.instancePath, '.executions', (await readdir(join(f.instancePath, '.executions')))[0], 'execution.json'), 'utf8'));
  assert.ok(receipt.cwd.startsWith('/registered/task-scope/attempt/'));
  const script = await readFile(runs[0][runs[0].indexOf('--file') + 1], 'utf8');
  const config = JSON.parse(script.slice(script.lastIndexOf(" '{") + 2, -2));
  assert.deepEqual(config.env, { ENVIRONMENT_TEST_VALUE: 'tool-override' });
});

for (const failure of ['closed', 'closed-at-stop', 'not-admitted', 'unknown']) test(`sync setup proof: ${failure}`, async t => {
  const f = await fixture(t), reason = {stage:'ssh_preflight',message:'open failed'};
  const abort = new AbortController(), hasHandle = failure.startsWith('closed');
  const environment = await createEnvironment(f.options, {...f.context, client:async args => {
    if (args[0] === 'observe' && args[1] === 'remote:fake:job:sync') {
      if (failure === 'closed-at-stop') abort.abort();
      return failure === 'unknown' ? {state:'observation-unavailable'} : {state:'failed',exit:null,reason};
    }
    if (failure === 'not-admitted' && args[0] === 'run' && args[args.indexOf('--file')+1].endsWith('sync.sh'))
      return {admitted:false,reason};
    return f.context.client(args);
  }});
  await environment.prepare({project:{source:f.project},attempt:{id:failure}});
  await assert.rejects(environment.execute('must-not-run',f.project,{signal:abort.signal}), error => {
    if (failure === 'unknown') {assert.equal(error.closure,undefined);return true;}
    assert.equal(error.code,hasHandle ? 'SETUP_CLOSED' : 'CPL_NOT_ADMITTED');
    assert.equal(error.closure.closed,true);assert.equal(error.closure.payloadStarted,false);
    assert.deepEqual(error.closure.handles,hasHandle ? ['remote:fake:job:sync'] : []);
    assert.match(error.message,/open failed/);return true;
  });
  assert.equal(f.calls.some(a=>a[0]==='run'&&a[a.indexOf('--file')+1].endsWith('command.sh')),false);
});

for (const failure of ['failed-upload','unknown-upload','running-upload','unidentified-upload',
                        'failed-download','not-admitted-download']) test(`transfer setup proof: ${failure}`,async t=>{
  const handle='remote:fake:transfer:proof',reason={stage:'transfer',message:'copy failed'};
  let observations=0;
  const f=await fixture(t,async args=>{
    if(args[0]==='upload'&&failure.endsWith('upload')){
      if(failure==='unidentified-upload')return {state:'observation-unavailable'};
      return {handle,state:failure==='failed-upload'?'failed':'running',reason};
    }
    if(args[0]==='observe'&&args[1]===handle){
      if(failure==='running-upload'&&observations++===0)return {state:'running'};
      return {state:'observation-unavailable'};
    }
    if(args[0]==='download'&&failure.endsWith('download'))return failure==='not-admitted-download'
      ?{admitted:false,reason}:{handle,state:'failed',reason};
  });
  const work=failure.endsWith('upload')?f.environment.execute('must-not-run',f.project):f.environment.browse('artifact');
  await assert.rejects(work,error=>{
    if(failure==='failed-upload'){
      assert.equal(error.code,'SETUP_CLOSED');assert.deepEqual(error.closure.handles,[handle]);
      assert.equal(error.closure.payloadStarted,false);assert.deepEqual(error.closure.reason.reason,reason);
    }else assert.equal(error.closure,undefined);
    return true;
  });
  assert.equal(f.calls.filter(a=>a[0]==='upload').length,failure.endsWith('upload')?1:0);
  assert.equal(f.calls.some(a=>a[0]==='run'&&a[a.indexOf('--file')+1].endsWith('command.sh')),false);
});

test('disconnect preserves handle; new observer can stop without resubmission', async t => {
  let connected = false, cancelled = false;
  const f = await fixture(t, async (args, { events }) => {
    if (args[0] === 'observe') {
      assert.ok(events.some(e => e.type === 'execution-handle' && e.handle === args[1]));
      if (!connected) throw new Error('offline');
      return { state: cancelled ? 'cancelled' : 'running' };
    }
    if (args[0] === 'cancel') { cancelled = true; return {}; }
  });
  const result = await f.environment.execute('work', f.project);
  assert.equal(result.settled, false);
  const resumed = await createEnvironment(f.options, f.context);
  assert.equal((await resumed.observe(result.handle)).state, 'unknown');
  connected = true;
  assert.equal((await resumed.observe(result.handle)).state, 'running');
  assert.equal((await resumed.stop(result.handle)).state, 'stopped');
  assert.equal(f.calls.filter(args => args[0] === 'run' && args[args.indexOf('--file') + 1].endsWith('command.sh')).length, 1);
});

test('native submission receipt recovers identity; missing output is explicitly partial', async t => {
  const f = await fixture(t, async args => {
    if (args[0] === 'run') {
      await writeFile(args[args.indexOf('--receipt-file') + 1], JSON.stringify({ handle: 'remote:fake:job:receipt' }));
      throw new Error('disconnected after receipt');
    }
    if (args[0] === 'download') throw new Error('copy unavailable');
  });
  const result = await f.environment.execute('true', f.project);
  assert.equal(result.handle, 'remote:fake:job:receipt');
  assert.equal(result.settled, true);
  const output = f.events.find(e => e.type === 'execution-output');
  assert.equal(output.partial, true);
  assert.ok(output.archive.path.startsWith('task-scope/attempt/'));
});

test('copy latency is excluded from forwarded elapsed timeout', async t => {
  const f = await fixture(t, async args => {
    if (args[0] === 'upload') { await delay(80); return { state: 'completed' }; }
    if (args[0] === 'run') {
      const script = await readFile(args[args.indexOf('--file') + 1], 'utf8');
      // Parse the public Python JSON argument, rather than asserting source wording.
      const config = JSON.parse(script.slice(script.lastIndexOf(" '{") + 2, -2));
      assert.ok(config.timeout > 0.05 && config.timeout <= 0.1);
      assert.ok(!config.cwd.includes(f.project));
      return { handle: 'remote:fake:job:clock' };
    }
  });
  assert.equal((await f.environment.execute('true', f.project, { timeout: 0.1 })).exitCode, 0);
});

test('local native operations persist identity across observer restart and confirm child group stop', async t => {
  const sdk = await import(process.env.PI_SDK_MODULE ?? '@earendil-works/pi-coding-agent');
  const instancePath = await mkdtemp(join(tmpdir(), 'environment-local-'));
  t.after(() => rm(instancePath, { recursive: true, force: true }));
  let handle;
  const context = { instancePath, sdk, exclude: async (_, fn) => fn(), event: async (type, data) => { if (type === 'execution-handle') handle = data.handle; } };
  const original = await createEnvironment({ mode: 'local' }, context);
  const completion = original.execute('sleep 60 & wait', instancePath);
  const resumed = await createEnvironment({ mode: 'local' }, context);
  let state;
  for (let n = 0; n < 100; n++) {
    if (handle) state = await resumed.observe(handle);
    if (state?.state === 'running') break;
    await delay(10);
  }
  assert.equal(state?.state, 'running');
  const { execFile } = await import('node:child_process');
  const { promisify } = await import('node:util');
  const childCode = `import {createEnvironment} from ${JSON.stringify(new URL('../../integrations/environments/index.mjs', import.meta.url).href)};
    const env=await createEnvironment({mode:'local'},{instancePath:process.argv[2],sdk:{createLocalBashOperations:()=>({})},event:async()=>{},exclude:async(_,fn)=>fn()});
    console.log(JSON.stringify([await env.observe(process.argv[1]),await env.stop(process.argv[1])]));`;
  const child = await promisify(execFile)(process.execPath, ['--input-type=module','-e',childCode,handle,instancePath]);
  const states = JSON.parse(child.stdout);
  assert.equal(states[0].state, 'running');
  assert.equal(states[1].state, 'stopped');
  assert.equal((await completion).settled, true);
  const short = await original.execute('exit 7', instancePath);
  assert.equal(short.exitCode, 7);
  assert.equal((await resumed.observe(short.handle)).exitCode, 7);
});

test('Python supervisor enforces elapsed timeout and preserves build outputs while syncing source deletions', async t => {
  const sdk = await import(process.env.PI_SDK_MODULE ?? '@earendil-works/pi-coding-agent');
  const { cp } = await import('node:fs/promises');
  const instancePath = await mkdtemp(join(tmpdir(), 'environment-supervisor-'));
  t.after(() => rm(instancePath, { recursive: true, force: true }));
  const project = join(instancePath, 'source'), workspace = join(instancePath, 'remote');
  await mkdir(project); await mkdir(workspace);
  await writeFile(join(project, 'old.py'), 'old');
  await writeFile(join(project, 'layout-node'), 'old file');
  const references = join(project, 'references'); await mkdir(references);
  const names = Array.from({length:1800},(_,i)=>`${String(i).padStart(4,'0')}-${'reference-'.repeat(12)}.txt`);
  await Promise.all(names.map(name=>writeFile(join(references,name),'retained reference')));
  const jobs = new Map(); let sequence = 0, loseObserver = false;
  let manifestBytes = 0, largestSync = 0;
  const context = { instancePath, sdk, event: async () => {}, exclude: async (label, fn) => {
    assert.ok(['setup','transfer'].includes(label)); return fn();
  }, client: async args => {
    if (args[0] === 'capabilities') return { state: 'available' };
    if (args[0] === 'upload') {
      if (args[2].endsWith('/sync.json')) manifestBytes = Math.max(manifestBytes,(await readFile(args[2])).length);
      await mkdir(dirname(join(workspace, args[3])), { recursive: true }); await cp(args[2], join(workspace, args[3]), { recursive: true }); return { state: 'completed' };
    }
    if (args[0] === 'download') { await cp(join(workspace, args[2]), args[3]); return { state: 'completed' }; }
    if (args[0] === 'run') {
      const handle = `remote:fake:job:${++sequence}`;
      const job = { state: 'running', setup: args[args.indexOf('--file') + 1].endsWith('sync.sh') }; jobs.set(handle, job);
      const script = await readFile(args[args.indexOf('--file') + 1], 'utf8');
      if (job.setup) largestSync = Math.max(largestSync,Buffer.byteLength(script));
      job.completion = sdk.createLocalBashOperations().exec(script, workspace, { onData: () => {} }).then(result => {
        job.state = 'completed'; job.exit = result.exitCode;
      });
      return { handle };
    }
    if (args[0] === 'observe') {
      if (loseObserver && !jobs.get(args[1]).setup) { loseObserver = false; throw new Error('observer disconnected'); }
      const { state, exit } = jobs.get(args[1]); return { state, exit };
    }
    return { state: 'completed' };
  } };
  const environment = await createEnvironment({ mode: 'cpl', target: 'fake', workspacePath: workspace, remoteDirectory: 'attempts', pollMs: 1 }, context);
  await environment.prepare({ project: { id: 'p', sourceDirectory: project }, attempt: { id: 'a' } });
  assert.equal((await environment.execute('printf built > build.marker', project)).exitCode, 0);
  assert.ok(manifestBytes > 128 * 1024,'fixture must exceed the observed per-argument limit');
  assert.ok(largestSync < 16 * 1024,'sync control payload must remain bounded with a large manifest');
  assert.deepEqual((await readdir(join(workspace,'attempts/a/project/references'))).sort(),names.sort());
  assert.equal(await readFile(join(workspace,'attempts/a/project/references',names[0]),'utf8'),'retained reference');
  await rm(references,{recursive:true});
  await rm(join(project, 'old.py'));
  await rm(join(project, 'layout-node')); await mkdir(join(project, 'layout-node'));
  await writeFile(join(project, 'layout-node', 'nested.py'), 'new directory');
  await writeFile(join(project, 'new.py'), "from pathlib import Path\nassert not Path('old.py').exists()\nprint(Path('build.marker').read_text())\n");
  let output = '';
  assert.equal((await environment.execute('python3 new.py', project, { onData: data => { output += data; } })).exitCode, 0);
  assert.equal(output, 'built\n');
  await assert.rejects(readdir(join(workspace,'attempts/a/project/references')),{code:'ENOENT'});
  assert.equal((await environment.execute("python3 -c \"from pathlib import Path; assert Path('layout-node/nested.py').read_text() == 'new directory'; assert Path('build.marker').read_text() == 'built'\"", project)).exitCode, 0);
  await rm(join(project, 'layout-node', 'nested.py')); await rm(join(project, 'layout-node'), { recursive: true });
  await writeFile(join(project, 'layout-node'), 'new file');
  assert.equal((await environment.execute("python3 -c \"from pathlib import Path; assert Path('layout-node').read_text() == 'new file'; assert Path('build.marker').read_text() == 'built'\"", project)).exitCode, 0);
  // The caller disappears after dispatch; workload timeout still closes the child.
  loseObserver = true;
  const timeoutJob = await environment.execute("python3 -c 'import time; time.sleep(30)'", project, { timeout: 0.1 });
  for (const job of jobs.values()) await job.completion;
  assert.equal(jobs.get(timeoutJob.handle).exit, 124);
  const timed = await environment.observe(timeoutJob.handle);
  assert.ok(timed.executionSeconds > 0 && timed.executionSeconds < 5);
  const freshObserver = await createEnvironment({ mode: 'cpl', target: 'fake', workspacePath: workspace, remoteDirectory: 'attempts' }, context);
  assert.equal((await freshObserver.observe(timeoutJob.handle)).executionSeconds, timed.executionSeconds);
  const verifier = join(instancePath, 'verify.py');
  await writeFile(join(project, 'value.txt'), '7');
  await writeFile(verifier, "from pathlib import Path\nassert int(Path('value.txt').read_text()) == 7\nPath('verified.marker').write_text('ran')\nif Path('mutate').exists(): Path(__file__).write_text('changed')\nprint('verified')\n");
  const verified = await createEnvironment({ mode: 'cpl', target: 'fake', workspacePath: workspace, remoteDirectory: 'verifier-attempts', pollMs: 1 }, context);
  await verified.prepare({ project: { id: 'verified', source: project, verifier: { files: [verifier], command: 'python3 ../verify.py' } }, attempt: { id: 'v' } });
  let verifiedOutput = '';
  const verify = (purpose = 'verification') => verified.execute('python3 ../verify.py', project, { purpose, onData: data => { verifiedOutput += data; } });
  assert.equal((await verify()).exitCode, 0);
  assert.equal(verifiedOutput, 'verified\n');
  await writeFile(join(project, 'mutate'), '1');
  assert.equal((await verify('performance')).exitCode, 125); // performance self-mutation rejects success
  assert.equal((await verified.execute('rm verified.marker', project)).exitCode, 0);
  assert.equal((await verify()).exitCode, 125); // existing corruption prevents verifier launch
  assert.equal((await verify('performance')).exitCode, 125);
  assert.equal((await verified.execute('test ! -e verified.marker', project)).exitCode, 0);
});

test('missing submission receipts persist unresolved events before errors and survive observer restart without dispatch', async t => {
  for (const phase of ['work', 'setup', 'prepare']) {
    const f = await fixture(t), trace = [];
    let submissions = 0;
    const context = { ...f.context, event: async (type, data) => {
      await f.context.event(type, data);
      await writeFile(join(f.instancePath, 'events.json'), JSON.stringify(f.events));
      if (type === 'execution-unresolved') trace.push('persisted');
    }, client: async args => {
      const file = args[args.indexOf('--file') + 1];
      if ((phase === 'prepare' && args[0] === 'upload') ||
          (args[0] === 'run' && file.endsWith(phase === 'setup' ? 'sync.sh' : 'command.sh'))) {
        submissions++;
        if (phase === 'setup') return { state: 'queued' };
        throw new Error('HTTP400 text without structured admission evidence');
      }
      return f.context.client(args);
    } };
    const verifier = join(f.instancePath, 'verify.py');
    await writeFile(verifier, 'assert True');
    const spec = { project: { id: 'p', source: f.project, ...(phase === 'prepare' ? { verifier: { files: [verifier] } } : {}) }, attempt: { id: phase } };
    const environment = await createEnvironment(f.options, context);
    await assert.rejects(async () => {
      try { await environment.prepare(spec); if (phase !== 'prepare') await environment.execute('true', f.project); }
      catch (error) { trace.push('exception'); throw error; }
    });
    assert.deepEqual(trace, ['persisted', 'exception']);
    const marker = JSON.parse(await readFile(join(f.instancePath, 'events.json'), 'utf8')).at(-1);
    assert.equal(marker.type, 'execution-unresolved'); assert.equal(marker.handle, null);
    assert.match(marker.operation, /^[0-9a-f-]{36}$/);
    await createEnvironment(f.options, context);
    assert.equal(submissions, 1);
    assert.deepEqual(JSON.parse(await readFile(join(f.instancePath, 'events.json'), 'utf8')).at(-1), marker);
  }
  const rejected = await fixture(t, async args => args[0] === 'run' ? { admitted: false, state: 'rejected' } : undefined);
  await assert.rejects(rejected.environment.execute('true', rejected.project));
  assert.equal(rejected.events.some(event => event.type === 'execution-unresolved'), false);
});

test('native SDK session PATH and local session-file metadata do not override remote runtime', async t => {
  const sdk = await import(process.env.PI_SDK_MODULE ?? '@earendil-works/pi-coding-agent');
  const f = await fixture(t);
  const [tool] = await createEnvironmentTools(f.environment,f.project,sdk);
  await tool.execute('native-env', { command: 'true' }, undefined, undefined, {
    cwd: f.project, sessionManager: { getSessionId: () => 'native-session', getSessionFile: () => '/local/attempt/session.jsonl' },
  });
  const run = f.calls.find(args => args[0] === 'run' && args[args.indexOf('--file') + 1].endsWith('command.sh'));
  const script = await readFile(run[run.indexOf('--file') + 1], 'utf8');
  const config = JSON.parse(script.slice(script.lastIndexOf(" '{") + 2, -2));
  assert.equal(Object.hasOwn(config.env, 'PATH'), false);
  assert.equal(Object.hasOwn(config.env, 'PI_SESSION_FILE'), false);
  assert.equal(config.env.PI_SESSION_ID, 'native-session');
});

test('SIGKILL after native receipt recovers the same operation and completed duration without resubmission', async t => {
  const { spawn } = await import('node:child_process');
  const { once } = await import('node:events');
  const { appendFile } = await import('node:fs/promises');
  const root = await mkdtemp(join(tmpdir(), 'environment-crash-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const project = join(root, 'project'); await mkdir(project);
  await writeFile(join(project, 'source.py'), 'answer=42');
  const module = new URL('../../integrations/environments/index.mjs', import.meta.url).href;
  const script = `import {createEnvironment} from ${JSON.stringify(module)};
    import {writeFile,appendFile} from 'node:fs/promises';
    const root=process.argv[1],project=root+'/project';
    const environment=await createEnvironment({mode:'cpl',target:'fake',workspacePath:'/registered',remoteDirectory:'owned'}, {
      instancePath:root,sdk:{createLocalBashOperations:()=>({})},exclude:async(_,fn)=>fn(),
      event:async(type,data)=>appendFile(root+'/events.jsonl',JSON.stringify({type,data:{...data,operation:'crash-op',actor_elapsed:3}})+'\\n'),
      client:async args=>{
        if(args[0]==='capabilities')return {state:'available'};
        if(args[0]==='run') {
          const sync=args[args.indexOf('--file')+1].endsWith('sync.sh');
          const result={handle:'remote:fake:job:'+(sync?'sync':'orphan')};
          await writeFile(args[args.indexOf('--receipt-file')+1],JSON.stringify(result));
          if(!sync)process.kill(process.pid,'SIGKILL');
          return result;
        }
        return {state:'completed',exit:0};
      }});
    await environment.prepare({project:{id:'p',source:project},attempt:{id:'a'}});
    await environment.execute('unrepeated original command',project);`;
  const child = spawn(process.execPath, ['--input-type=module', '-e', script, root], { stdio: 'ignore' });
  const [code, signal] = await once(child, 'exit');
  assert.equal(code, null); assert.equal(signal, 'SIGKILL');
  const events = (await readFile(join(root, 'events.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
  const pending = events.find(event => event.type === 'execution-submitting' && !event.data.accountingExcluded);
  assert.ok(pending?.data.receiptFile);
  assert.equal(events.some(event => event.type === 'execution-handle' && event.data.handle === 'remote:fake:job:orphan'), false);
  const receiptDir = dirname(pending.data.receiptFile);
  const metadata = JSON.parse(await readFile(join(receiptDir, 'execution.json'), 'utf8'));
  assert.equal(metadata.handle, null); assert.ok(metadata.timingKey);
  let terminal = false, downloads = 0, submissions = 0, incomplete = false;
  const options = { mode: 'cpl', target: 'fake' };
  const context = { instancePath: root, sdk: { createLocalBashOperations: () => ({}) }, exclude: async (label, fn) => {
    assert.equal(label, 'transfer'); return fn();
  }, event: async (type, data) => {
    events.push({ type, data: { ...data, operation: 'crash-op' } });
    await appendFile(join(root, 'events.jsonl'), JSON.stringify(events.at(-1)) + '\n');
  }, client: async args => {
    if (['run','upload'].includes(args[0])) { submissions++; throw new Error('must not resubmit'); }
    if (args[0] === 'observe') return args[1].endsWith(':sync') ? { state: 'completed', exit: 0 } : { state: terminal ? 'completed' : 'running', exit: 0 };
    if (args[0] === 'download') {
      assert.equal(terminal, true); downloads++;
      await writeFile(args[3], JSON.stringify({ completed: !incomplete, executionSeconds: 30.125, key: metadata.timingKey }));
      return { state: 'completed' };
    }
    throw new Error('unexpected operation');
  } };
  const resumed = await createEnvironment(options, context);
  assert.equal((await resumed.reconcileOperation({ operation: 'crash-op', events, startElapsed: 3 })).closed, false);
  assert.equal(downloads, 0);
  terminal = true;
  const result = await resumed.reconcileOperation({ operation: 'crash-op', events, startElapsed: 3 });
  assert.equal(result.closed, true); assert.equal(result.executionSeconds, 30.125);
  assert.ok(result.handles.includes('remote:fake:job:orphan'));
  assert.equal((await resumed.reconcileOperation({ operation: 'crash-op', events, startElapsed: 3 })).executionSeconds, 30.125);
  assert.equal(downloads, 1); assert.equal(submissions, 0);
  incomplete = true; await rm(join(receiptDir, 'timing.json'));
  assert.equal((await (await createEnvironment(options, context)).reconcileOperation({ operation: 'crash-op', events })).closed, false);
  await rm(pending.data.receiptFile);
  assert.equal((await (await createEnvironment(options, context)).reconcileOperation({ operation: 'crash-op', events })).closed, false);
  assert.equal(submissions, 0);
});


test('reaped process proc descriptor reports absence without hiding permission failures', async t => {
  const { spawn } = await import('node:child_process');
  const { once } = await import('node:events');
  const { open } = await import('node:fs/promises');
  const child = spawn(process.execPath, ['-e', 'setInterval(()=>{},1000)'], { stdio: 'ignore' });
  t.after(() => { if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL'); });
  const descriptor = await open(`/proc/${child.pid}/stat`, 'r');
  try {
    const exited = once(child, 'exit'); child.kill('SIGKILL'); await exited;
    assert.equal(await readProcessStat(child.pid, () => descriptor.readFile('utf8')), null);
    await assert.rejects(readProcessStat(child.pid, async () => { throw Object.assign(new Error('permission denied'), { code: 'EACCES' }); }), { code: 'EACCES' });
  } finally { await descriptor.close(); }
});
