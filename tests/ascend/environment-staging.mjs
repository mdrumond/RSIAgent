import test from 'node:test';
import assert from 'node:assert/strict';
import { cp, mkdtemp, mkdir, readFile, writeFile, rm, chmod, stat, utimes, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname,basename } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { createEnvironment, createEnvironmentTools } from '../../integrations/environments/index.mjs';
import {normalizeInvestigationProject} from '../../cli/curriculum.mjs';

async function fixture(t) {
  const sdk = await import('@earendil-works/pi-coding-agent');
  const root = await mkdtemp(join(tmpdir(), 'staging-cache-'));
  const source = join(root, 'source'), workspace = join(root, 'remote');
  await mkdir(source); await mkdir(workspace);
  await writeFile(join(source, 'value.txt'), 'one');
  await writeFile(join(source, 'run.sh'), '#!/bin/bash\ncat value.txt\n', { mode: 0o755 });
  const calls = [], events = [], jobs = new Map(), excluded = [];
  const control = { failSync: false, disconnected: false, missingReceipt: false, uploadDelay: 0, uploadStates: null,
    backend:'remote-agent',failUpload:false,failDownload:false,parentUnknown:false,failParent:false,
    uploadExit:null,downloadExit:null,dropSetupReceipt:false };
  let sequence = 0;
  const operations = sdk.createLocalBashOperations();
  const context = {
    instancePath: root, sdk, event: async (type, data) => events.push({ type, ...data }),
    exclude: async (label, fn) => { assert.ok(['setup', 'transfer'].includes(label)); excluded.push(label); return fn(); },
  };
  const options = { mode: 'cpl', target: 'fake', runtime: 'declared', workspacePath: workspace, remoteDirectory: 'owned', pollMs: 1 };
  function clientFor(remote) {
    return async args => {
      calls.push(args);
      if (args[0] === 'capabilities') return { state: 'available',backend:control.backend };
      if (args[0] === 'preflight') return { state: 'completed' };
      if (args[0] === 'upload') {
        if(control.failUpload)throw new Error('Upload connection lost');
        if(control.backend==='ssh')assert.equal(args.length,4,'SSH upload has no retained-transfer flags');
        if(control.uploadExit!==null)return {backend:'ssh',operation:'upload',state:'failed',exit:control.uploadExit};
        await delay(control.uploadDelay);
        let destination = join(remote, args[3]);
        if(control.backend==='ssh')assert.equal((await stat(dirname(destination))).isDirectory(),true,'rsync requires an existing remote parent');
        else await mkdir(dirname(destination), { recursive: true });
        if(control.backend==='ssh'&&(await stat(args[2])).isDirectory()) {
          // Existing incoming directories occur on retry. Plain rsync nests a
          // directory basename unless its source argument ends with a slash.
          await mkdir(destination,{recursive:true});
          if(!args[2].endsWith('/'))destination=join(destination,basename(args[2]));
        }
        await cp(args[2], destination, { recursive: true });
        if (control.uploadStates) {
          const handle = `remote:${args[1]}:transfer:${++sequence}`;
          jobs.set(handle, { states: [...control.uploadStates] });
          control.uploadStates = null;
          return { state: 'queued', handle };
        }
        return { state: 'completed' };
      }
      if (args[0] === 'download') {
        if(control.failDownload)throw new Error('Download refused');
        if(control.backend==='ssh')assert.equal(args.length,4,'SSH download uses synchronous completion');
        if(control.downloadExit!==null)return {backend:'ssh',operation:'download',state:'failed',exit:control.downloadExit};
        await cp(join(remote, args[2]), args[3]); return { state: 'completed' };
      }
      if (args[0] === 'run') {
        const file = args[args.indexOf('--file') + 1], sync = file.endsWith('sync.sh'),parent=file.endsWith('upload-parent.sh');
        const handle = `remote:${args[1]}:job:${++sequence}`, job = { state: 'running', sync,parent };
        jobs.set(handle, job);
        if (!control.missingReceipt&&!(control.dropSetupReceipt&&(parent||sync))) await writeFile(args[args.indexOf('--receipt-file') + 1], JSON.stringify({ handle }));
        const script = parent&&control.failParent?'exit 19\n':sync && control.failSync ? 'exit 17\n' : await readFile(file, 'utf8');
        control.failSync = false;
        job.done = operations.exec(script, remote, { onData: () => {} }).then(result => {
          job.state = 'completed'; job.exit = result.exitCode;
        });
        return control.missingReceipt ? { state: 'queued' } : { handle, state: 'queued' };
      }
      if (args[0] === 'observe') {
        const job = jobs.get(args[1]);
        if (job.states) return { state: job.states.shift() ?? 'completed' };
        if(job.parent&&control.parentUnknown)return {state:'observation-unavailable'};
        if (job.sync && control.disconnected) return { state: 'observation-unavailable' };
        return { state: job.state, exit: job.exit };
      }
      throw Error('unexpected CPL call ' + args[0]);
    };
  }
  async function connect(changes = {}, attempt = 'a',verifierFiles=[]) {
    const selected = { ...options, ...changes };
    await mkdir(selected.workspacePath, { recursive: true });
    const env = await createEnvironment(selected, { ...context, client: clientFor(selected.workspacePath) });
    await env.prepare({ project: { id: 'p', source,verifier:{files:verifierFiles} }, attempt: { id: attempt } });
    return env;
  }
  const environment = await connect();
  t.after(async () => { await Promise.all([...jobs.values()].map(job => job.done)); await rm(root, { recursive: true, force: true }); });
  const sourceUploads = () => calls.filter(args => args[0] === 'upload' && args[2].replace(/\/$/,'') === source);
  const manifests = () => calls.filter(args => args[0] === 'upload' && args[2].endsWith('/sync.json'));
  const runs = sync => calls.filter(args => args[0] === 'run' && args[args.indexOf('--file') + 1].endsWith(sync ? 'sync.sh' : 'command.sh'));
  const run = async (command, env = environment, opts = {}) => {
    let output = '';
    const result = await env.execute(command, source, { ...opts, onData: data => { output += data; } });
    assert.equal(result.settled, true); assert.equal(result.exitCode, 0, output);
    return output;
  };
  return { root, source, workspace, sdk, environment, control, context, options, calls, events,
    connect, run, sourceUploads, manifests, runs, excluded, idle: () => Promise.all([...jobs.values()].map(job => job.done)) };
}

test('retained investigations stage their own verifier despite obsolete parent bundle metadata',async t=>{
  const f=await fixture(t),root=join(f.root,'investigation'),source=join(root,'source');
  await cp(f.source,source,{recursive:true});
  const verifier=join(root,'verify.py');
  await writeFile(verifier,"from pathlib import Path\nassert Path('value.txt').read_text()=='one'\nprint('independent check passed')\n");
  const legacy={id:'parent',source,sourceDirectory:f.source,actor_seconds:120,
    bundle:{version:1,root:f.source,reference:f.source,negative_controls:[],construction_command:'parent-check'},
    verifier:{files:[verifier],command:'python3 ../verify.py'}};
  // The old metadata fails before transfer, exactly as the retained campaign did.
  const old={...legacy};delete old.sourceDirectory;
  await assert.rejects(f.environment.prepare({project:old,attempt:{id:'child'}}),/Editable source must be inside its bundle root/);
  const before=JSON.stringify(legacy),project=normalizeInvestigationProject(legacy,f.root);
  assert.equal(JSON.stringify(legacy),before,'Resume must not rewrite the retained project manifest');
  await f.environment.prepare({project,attempt:{id:'child'}});
  let output='';
  const result=await f.environment.execute(project.verifier.command,source,{onData:data=>{output+=data;}});
  assert.equal(result.settled,true);assert.equal(result.exitCode,0,output);
  assert.equal(output.trim(),'independent check passed');
  assert.equal(await readFile(join(f.workspace,'owned/child/tree/verify.py'),'utf8'),await readFile(verifier,'utf8'));
  assert.equal(await readFile(join(f.source,'value.txt'),'utf8'),'one');
  // Normalization is scoped to the owning attempt and cannot waive its trust boundary.
  assert.throws(()=>normalizeInvestigationProject({...legacy,source:f.source},f.root),/current attempt/);
  for(const file of [join(source,'value.txt'),join(f.root,'unrelated.py')])
    assert.throws(()=>normalizeInvestigationProject({...legacy,verifier:{...legacy.verifier,files:[file]}},f.root),/outside editable source/);
  const moved={...project,bundle:{version:1,root:f.source}};
  await assert.rejects(f.environment.prepare({project:moved,attempt:{id:'other'}}),/Editable source must be inside its bundle root/);
});

test('advertised SSH transfers stage directory contents into an existing incoming path and collect synchronously',async t=>{
  const f=await fixture(t);f.control.backend='ssh';
  const environment=await f.connect({},'ssh');
  assert.equal(await f.run('cat value.txt; printf artifact > result.txt',environment),'one');
  const destination=join(f.root,'collected.txt');
  await environment.collectArtifact('owned/ssh/project/result.txt',destination);
  assert.equal(await readFile(destination,'utf8'),'artifact');
  assert.equal(f.calls.some(args=>args[0]==='fetch'),false);
  assert.equal(f.sourceUploads()[0][2],f.source+'/');
  assert.ok(f.manifests().every(args=>!args[2].endsWith('/')));
  const transfers=f.events.filter(row=>row.type==='execution-submitted'&&row.accountingExcluded&&row.closed);
  assert.ok(transfers.length>=3);assert.equal(transfers.every(row=>row.handle===null),true);
  const firstUpload=f.calls.findIndex(args=>args[0]==='upload');
  const parent=f.calls.slice(0,firstUpload).find(args=>args[0]==='run');
  assert.equal(parent[parent.indexOf('--cwd')+1],f.workspace);
  assert.equal(parent[parent.indexOf('--runtime')+1],'declared');
  const receipt=JSON.parse(await readFile(parent[parent.indexOf('--receipt-file')+1],'utf8'));
  assert.ok(f.calls.slice(0,firstUpload).some(args=>args[0]==='observe'&&args[1]===receipt.handle));
});

test('SSH frozen verifier upload prepares its scoped parent before staging source',async t=>{
  const f=await fixture(t);f.control.backend='ssh';const verifier=join(f.root,'oracle.py');
  await writeFile(verifier,"print('independent oracle')\n");
  const environment=await f.connect({},'ssh',[verifier]);
  const trustedUploads=f.calls.filter(args=>args[0]==='upload'&&args[3]==='owned/ssh/tree');
  assert.equal(trustedUploads.length,1);
  assert.ok(trustedUploads[0][2].endsWith('/tree/'));
  assert.equal(await readFile(join(f.workspace,'owned/ssh/tree/oracle.py'),'utf8'),await readFile(verifier,'utf8'));
  assert.equal(f.runs(false).length,0);
  assert.equal(await f.run('python3 ../oracle.py',environment),'independent oracle\n');
});

for(const failure of ['failed','unknown'])test(`SSH parent setup ${failure} prevents transfer and payload`,async t=>{
  const f=await fixture(t);f.control.backend='ssh';const environment=await f.connect({},'ssh');
  f.control.failParent=failure==='failed';f.control.parentUnknown=failure==='unknown';
  await assert.rejects(environment.execute('touch must-not-run',f.source),error=>{
    if(failure==='failed'){assert.equal(error.closure.closed,true);assert.equal(error.closure.reason.exitCode,19);}
    else assert.equal(error.closure,undefined);
    return /Upload parent setup/.test(error.message);
  });
  assert.equal(f.sourceUploads().length,0);assert.equal(f.runs(true).length,0);assert.equal(f.runs(false).length,0);
});

test('terminal SSH transfer failures retain exit evidence without falsely settling prior download work',async t=>{
  const f=await fixture(t);f.control.backend='ssh';const environment=await f.connect({},'ssh');
  f.control.uploadExit=23;
  await assert.rejects(environment.execute('touch must-not-run',f.source),error=>{
    assert.equal(error.closure.closed,true);assert.equal(error.closure.reason.exit,23);return true;
  });
  assert.equal(f.runs(true).length,0);assert.equal(f.runs(false).length,0);
  assert.equal(f.events.at(-1).reason.exit,23);assert.equal(f.events.at(-1).closed,true);
  f.control.downloadExit=23;
  await assert.rejects(environment.collectArtifact('missing',join(f.root,'unused')),error=>{
    assert.equal(error.closure,undefined);return /exit 23/.test(error.message);
  });
  assert.equal(f.events.at(-1).reason.exit,23);
});

test('SSH upload uncertainty prevents payload dispatch and download refusal does not settle earlier work',async t=>{
  const f=await fixture(t);f.control.backend='ssh';const environment=await f.connect({},'ssh');
  f.control.failUpload=true;
  await assert.rejects(environment.execute('touch must-not-run',f.source),/submission identity unknown/);
  assert.equal(f.runs(false).length,0);assert.equal(f.runs(true).length,0);
  assert.equal(f.events.at(-1).type,'execution-unresolved');
  f.control.failUpload=false;f.control.failDownload=true;
  await assert.rejects(environment.collectArtifact('unavailable',join(f.root,'unused')),error=>{
    assert.equal(error.closure,undefined);return /submission identity unknown/.test(error.message);
  });
  assert.equal(f.events.at(-1).type,'execution-unresolved');
});

test('an unexpected SSH transfer handle is retained without claiming synchronous closure',async t=>{
  const f=await fixture(t);f.control.backend='ssh';const environment=await f.connect({},'ssh');
  f.control.uploadStates=['completed'];
  await assert.rejects(environment.execute('touch must-not-run',f.source),/Synchronous transfer unresolved/);
  const submitted=f.events.find(row=>row.type==='execution-submitted'&&row.handle?.includes(':transfer:'));
  assert.ok(submitted.handle);assert.equal(submitted.closed,false);
  assert.equal(f.sourceUploads().length,1);assert.equal(f.runs(false).length,0);assert.equal(f.runs(true).length,0);
});

test('native SDK repeated Bash reuses one upload/manifest but restores source every time and retains build artifacts', async t => {
  const f = await fixture(t);
  const [tool] = await createEnvironmentTools(f.environment, f.source, f.sdk);
  const context = { cwd: f.source, sessionManager: { getSessionId: () => 'native', getSessionFile: () => null } };
  await tool.execute('first', { command: 'printf remote > value.txt; chmod 600 run.sh; printf compiled > build.marker' }, undefined, undefined, context);
  const response = await tool.execute('second', { command: './run.sh; cat build.marker' }, undefined, undefined, context);
  assert.ok(response.content.some(block => block.type === 'text' && block.text.includes('onecompiled')));
  assert.equal(f.sourceUploads().length, 1); assert.equal(f.manifests().length, 1);
  assert.equal(f.runs(true).length, 2); assert.equal(f.runs(false).length, 2);
  assert.deepEqual(f.events.filter(e => e.type === 'source-staged').map(e => e.reused), [false, true]);
  assert.ok(f.excluded.includes('transfer'));
});

test('unchanged target inputs retain inode and timestamp while edited inputs are restored',async t=>{
  const f=await fixture(t);
  await f.run('true');
  const target=join(f.workspace,'owned/a/project/value.txt');
  const oldTime=new Date('2001-01-01T00:00:00Z');
  await utimes(target,oldTime,oldTime);
  const before=await stat(target);
  await f.run('cat value.txt');
  const unchanged=await stat(target);
  assert.equal(unchanged.ino,before.ino);
  assert.equal(unchanged.mtimeMs,before.mtimeMs);
  await writeFile(target,'bad'); // Same byte length still requires a content check.
  await utimes(target,oldTime,oldTime);
  assert.equal(await f.run('cat value.txt'),'one');
  assert.notEqual((await stat(target)).mtimeMs,before.mtimeMs);
  await chmod(target,0o600);
  await f.run('true');
  assert.equal((await stat(target)).mode & 0o777,(await stat(join(f.source,'value.txt'))).mode & 0o777);
  assert.equal(f.sourceUploads().length,1);
});

test('multiple frozen files transfer as one scoped tree and preserve independent verification',async t=>{
  const f=await fixture(t),verifierRoot=join(f.root,'trusted');
  await mkdir(verifierRoot);
  const oracle=join(verifierRoot,'oracle.py'),data=join(verifierRoot,'expected.txt');
  await writeFile(oracle,"from pathlib import Path\nassert Path(__file__).with_name('expected.txt').read_text()=='one'\nassert Path('value.txt').read_text()=='one'\nprint('verified')\n");
  await writeFile(data,'one');
  const env=await f.connect({},'batch',[oracle,data]);
  const uploads=f.calls.filter(args=>args[0]==='upload');
  assert.equal(uploads.length,1);
  assert.equal(uploads[0][3],'owned/batch/tree');
  assert.equal(await f.run('python3 ../trusted/oracle.py',env,{purpose:'verification'}),'verified\n');
  await writeFile(join(f.workspace,'owned/batch/tree/trusted/expected.txt'),'tampered');
  const result=await env.execute('touch payload-must-not-run',f.source,{purpose:'verification'});
  assert.equal(result.settled,true);assert.equal(result.exitCode,125);
  await assert.rejects(stat(join(f.workspace,'owned/batch/tree/source/payload-must-not-run')),{code:'ENOENT'});
});

test('content with preserved timestamp, adds, deletes, file/directory refactors and executable modes invalidate staging', async t => {
  const f = await fixture(t);
  await f.run('printf kept > build.marker');
  const before = await stat(join(f.source, 'value.txt'));
  await writeFile(join(f.source, 'value.txt'), 'two');
  await utimes(join(f.source, 'value.txt'), before.atime, before.mtime);
  assert.equal(await f.run('cat value.txt'), 'two');
  await writeFile(join(f.source, 'added'), 'added');
  assert.equal(await f.run('cat added'), 'added');
  await rm(join(f.source, 'added'));
  await f.run('test ! -e added');
  await rm(join(f.source, 'value.txt')); await mkdir(join(f.source, 'value.txt'));
  await writeFile(join(f.source, 'value.txt', 'child'), 'directory');
  assert.equal(await f.run('cat value.txt/child'), 'directory');
  await rm(join(f.source, 'value.txt'), { recursive: true });
  await writeFile(join(f.source, 'value.txt'), 'file');
  assert.equal(await f.run('cat value.txt'), 'file');
  await chmod(join(f.source, 'run.sh'), 0o644);
  await f.run('test ! -x run.sh; test "$(cat build.marker)" = kept');
  assert.equal(f.sourceUploads().length, 7); assert.equal(f.manifests().length, 7);
  await f.run('test "$(cat value.txt)" = file');
  assert.equal(f.sourceUploads().length, 7);
});

test('confirmed staging survives observer restart; target/runtime/workspace/attempt identities require new snapshots', async t => {
  const f = await fixture(t);
  await f.run('true');
  const restarted = await f.connect();
  await f.run('true', restarted);
  assert.equal(f.sourceUploads().length, 1);
  for (const [changes, attempt] of [
    [{ target: 'other' }, 'a'], [{ runtime: 'changed-runtime' }, 'a'],
    [{ workspacePath: join(f.root, 'other-remote') }, 'a'], [{}, 'new-attempt'],
  ]) await f.run('cat value.txt', await f.connect(changes, attempt));
  assert.equal(f.sourceUploads().length, 5); assert.equal(f.manifests().length, 5);
});

test('failed synchronization is not cached; a missing retained snapshot or manifest stops before workload and permits explicit fresh retry', async t => {
  const f = await fixture(t); f.control.failSync = true;
  await assert.rejects(f.environment.execute('touch must-not-run', f.source), /Copy failed/);
  assert.equal(f.runs(false).length, 0);
  await f.run('true'); await f.run('true');
  assert.equal(f.sourceUploads().length, 2);
  for (const kind of ['snapshot', 'manifest']) {
    const item = kind === 'snapshot' ? f.sourceUploads().at(-1) : f.manifests().at(-1);
    await rm(join(f.workspace, item[3]), { recursive: true });
    const commands = f.runs(false).length;
    await assert.rejects(f.environment.execute('touch must-not-run', f.source), /Copy failed/);
    assert.equal(f.runs(false).length, commands);
    await f.run('true');
  }
  assert.equal(f.sourceUploads().length, 4);
});

test('unobservable synchronization and absent native receipt block restart without another upload, sync or command', async t => {
  for (const missingReceipt of [false, true]) {
    const f = await fixture(t);
    f.control.disconnected = !missingReceipt; f.control.missingReceipt = missingReceipt;
    await assert.rejects(f.environment.execute('touch must-not-run', f.source), /unresolved|unknown/);
    await f.idle();
    const calls = f.calls.filter(args => ['upload', 'run'].includes(args[0])).length;
    const restarted = await f.connect();
    await assert.rejects(restarted.execute('touch must-not-run', f.source), /unresolved/);
    assert.equal(f.calls.filter(args => ['upload', 'run'].includes(args[0])).length, calls);
    assert.equal(f.runs(false).length, 0);
    if (!missingReceipt) {
      f.control.disconnected = false;
      await f.run('true', restarted);
      assert.equal(f.sourceUploads().length, 1); assert.equal(f.manifests().length, 1);
      assert.equal(f.runs(true).length, 2);
    }
  }
});

test('symlink trees retain fresh-transfer behavior; staging latency stays excluded from nested timeout', async t => {
  const f = await fixture(t);
  await symlink('value.txt', join(f.source, 'alias'));
  f.control.uploadDelay = 60;
  await f.run('cat alias', f.environment, { timeout: 1 });
  await f.run('cat alias', f.environment, { timeout: 1 });
  assert.equal(f.sourceUploads().length, 2);
  for (const args of f.runs(false)) {
    const script = await readFile(args[args.indexOf('--file') + 1], 'utf8');
    const configuration = JSON.parse(script.slice(script.lastIndexOf(" '{") + 2, -2));
    assert.ok(configuration.timeout > 0.8 && configuration.timeout <= 1);
  }
});

test('partial snapshot deletion, content/type/mode changes, extra files and manifest tampering fail before workload', async t => {
  for (const corruption of [
    'rm ../exec-*/incoming/value.txt',
    'printf cached-poison > ../exec-*/incoming/value.txt',
    'chmod 700 ../exec-*/incoming/value.txt',
    'chmod 700 ../exec-*/incoming/assets',
    'printf extra > ../exec-*/incoming/extra.txt',
    'rm ../exec-*/incoming/value.txt; mkdir ../exec-*/incoming/value.txt',
    "printf '{}' > ../exec-*/sync.json",
  ]) {
    const f = await fixture(t);
    await mkdir(join(f.source, 'assets'));
    await chmod(join(f.source, 'assets'),0o755);
    await writeFile(join(f.source, 'assets', 'reference.txt'), 'reference');
    await f.run("incoming=$(printf '%s' ../exec-*/incoming); " + corruption.replaceAll('../exec-*/incoming', '"$incoming"') + '; printf poisoned > value.txt; printf kept > build.marker');
    assert.equal(f.sourceUploads().length, 1);
    const workloads = f.runs(false).length;
    await assert.rejects(f.environment.execute('cat value.txt', f.source), /Copy failed/, corruption);
    assert.equal(f.runs(false).length, workloads, corruption);
    assert.equal(await readFile(join(f.workspace, 'owned/a/project/value.txt'), 'utf8'), 'poisoned');
    assert.equal(await f.run('./run.sh; cat build.marker'), 'onekept');
    assert.equal(f.sourceUploads().length, 2);
  }
});


test('retained root mode and directory-to-symlink changes fail before restore; explicit fresh retry restores project root permissions', async t => {
  for (const corruption of ['chmod 555 "$incoming"', 'mv "$incoming" "$incoming.saved"; ln -s "$incoming.saved" "$incoming"']) {
    const f = await fixture(t);
    await chmod(f.source, 0o755);
    await f.run("incoming=$(printf '%s' ../exec-*/incoming); " + corruption + '; printf poisoned > value.txt');
    const workloads = f.runs(false).length;
    await assert.rejects(f.environment.execute('printf workload > should-not-exist', f.source), /Copy failed/, corruption);
    assert.equal(f.runs(false).length, workloads);
    const project = join(f.workspace, 'owned/a/project');
    assert.equal((await stat(project)).mode & 0o777, 0o755);
    assert.equal(await readFile(join(project, 'value.txt'), 'utf8'), 'poisoned');
    assert.equal(await f.run('./run.sh; printf built > build.output'), 'one');
    assert.equal(f.sourceUploads().length, 2);
    assert.equal((await stat(project)).mode & 0o777, 0o755);
    assert.equal(await readFile(join(project, 'build.output'), 'utf8'), 'built');
    // Restore the retired test snapshot's permissions solely for fixture cleanup.
    await chmod(join(f.workspace, f.sourceUploads()[0][3]), 0o755);
  }
});


test('native upload retrying is observed on the same handle before one workload delivers full output', async t => {
  const f = await fixture(t);
  f.control.uploadStates = ['retrying', 'retrying', 'completed'];
  const expected = 'complete-output\n'.repeat(10000);
  assert.equal(await f.run(`python3 -c "print(('complete-output'+chr(10))*10000,end='')"`), expected);
  assert.equal(f.sourceUploads().length, 1);
  assert.equal(f.manifests().length, 1);
  assert.equal(f.runs(true).length, 1); assert.equal(f.runs(false).length, 1);
  const observations = f.calls.filter(args => args[0] === 'observe' && args[1].includes(':transfer:'));
  assert.equal(observations.length, 3);
  assert.equal(new Set(observations.map(args => args[1])).size, 1);
  assert.equal(f.events.filter(e => e.type === 'execution-output' && e.partial === false).length, 1);
});

for (const state of ['unknown', 'observation-unavailable'])
  test(`native upload ${state} blocks without resubmitting or launching workload`, async t => {
    const f = await fixture(t); f.control.uploadStates = [state];
    await assert.rejects(f.environment.execute('touch must-not-run', f.source), /Transfer unresolved/);
    assert.equal(f.sourceUploads().length, 1); assert.equal(f.manifests().length, 0);
    assert.equal(f.runs(true).length, 0); assert.equal(f.runs(false).length, 0);
  });

// Exercise the controller's actual native/default tool registration, not just
// the environment helper: shadowing bash breaks the local read immediately.
test('canonical native bash edits persist locally and rsi_target restores updated execution snapshots',{timeout:20000},async t=>{
  const {State}=await import('../../controller/state.mjs');
  const {runAttempt}=await import('../../controller/pi.mjs');
  const {stages}=await import('../../controller/workflow.mjs');
  const {InMemoryCredentialStore}=await import('@earendil-works/pi-ai');
  const {createServer}=await import('node:http');const {once}=await import('node:events');
  const f=await fixture(t),state=new State(join(f.root,'state.sqlite'));t.after(()=>state.close());
  const q=value=>"'"+value.replaceAll("'","'\\''")+"'";
  const write=value=>`python3 -c ${q(`from pathlib import Path;Path('helper.py').write_text('value=${value}\\n')`)}`;
  const inspect=`python3 -c ${q("import runpy;print(runpy.run_path('helper.py')['value'])")}; cat value.txt`;
  const actions=[['bash',{command:write(1)}],['read',{path:'helper.py'}],
    ['rsi_target',{command:inspect+'; printf target-only > value.txt'}],['bash',{command:write(2)}],
    ['rsi_target',{command:inspect}],['rsi_target',{command:inspect}]];
  const plan=[];for(const stage of stages){plan.push(['rsi_skill',{}]);
    if(['design','optimisation'].includes(stage))for(const kind of ['good','negative'])plan.push(['rsi_knowledge',{kind,query:'numeric evidence'}]);
    if(stage==='implementation')plan.push(...actions);
    if(stage==='validation')plan.push(['rsi_verify',{}]);
    plan.push(['rsi_advance',{reflection:'Retain independent evidence and persistent local source.'}]);}
  const replies=new Map();
  const server=createServer(async(req,res)=>{
    let raw='';for await(const data of req)raw+=data;const body=JSON.parse(raw);
    for(const message of body.messages.filter(m=>m.role==='tool'))replies.set(message.tool_call_id,message.content);
    const index=body.messages.reduce((n,m)=>n+(m.tool_calls?.length??0),0),call=plan[index];
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const frame=(delta,finish_reason=null)=>res.write('data: '+JSON.stringify({id:'workspace',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
    if(call){frame({role:'assistant',tool_calls:[{index:0,id:'call-'+index,type:'function',function:{name:call[0],arguments:JSON.stringify(call[1])}}]});frame({},'tool_calls');}
    else{frame({role:'assistant',content:'Done.'});frame({},'stop');}res.end('data: [DONE]\n\n');
  });server.listen(0,'127.0.0.1');await once(server,'listening');
  t.after(async()=>{server.closeAllConnections();await new Promise(ok=>server.close(ok));});
  const runtime=await f.sdk.ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsPath:null,allowModelNetwork:false,refreshOnCreate:false});
  runtime.registerProvider('workspace-fixture',{baseUrl:`http://127.0.0.1:${server.address().port}/v1`,models:[{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:128000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});
  await runtime.setRuntimeApiKey('workspace-fixture','fixture-only');
  const verifier=join(f.root,'verify.py');await writeFile(verifier,`from pathlib import Path\nimport runpy\nassert Path.cwd().is_relative_to(Path(${JSON.stringify(f.workspace)}))\nassert runpy.run_path('helper.py')['value']==2\nassert Path('value.txt').read_text()=='one'\nprint('target oracle passed')\n`);
  const skillMap={};for(const stage of [...stages,'global-kdb']){skillMap[stage]=join(f.root,stage+'.md');await writeFile(skillMap[stage],'Use the independent oracle.');}
  state.add({id:'p',source:f.source,goal:'Persistent helper',actor_seconds:30,verifier:{files:[verifier],command:'PYTHONDONTWRITEBYTECODE=1 python3 ../verify.py'}});
  const attempt=state.admit('p',{});
  const result=await runAttempt({state,attempt,project:state.project('p').spec,attemptPath:join(f.root,'attempt'),skillMap,
    modelRuntime:runtime,model:runtime.getModel('workspace-fixture','scripted'),knowledge:{search:async()=>({content:[],evidence:[]})},environmentFactory:async()=>f.environment});
  assert.equal(result.accepted,true);
  assert.equal(await readFile(join(f.source,'helper.py'),'utf8'),'value=2\n');
  assert.equal(await readFile(join(f.source,'value.txt'),'utf8'),'one');
  const implementation=plan.findIndex(([name,args])=>name==='bash'&&args.command===write(1));
  assert.match(replies.get('call-'+(implementation+1)),/value=1/);
  for(const offset of [4,5])assert.match(replies.get('call-'+(implementation+offset)),/2\none/);
  assert.equal(f.runs(false).length,4); // Three target requests plus frozen verification.
  assert.deepEqual(f.events.filter(e=>e.type==='source-staged').map(e=>e.reused),[false,false,true,true]);
  assert.equal(state.events(attempt.id).find(e=>e.type==='verification').data.passed,true);
});


test('completed excluded parent and sync jobs do not need payload timing or a local submission receipt',async t=>{
  const f=await fixture(t);f.control.backend='ssh';f.control.dropSetupReceipt=true;
  const environment=await f.connect({},'lost-setup-receipt');
  assert.equal(await f.run('cat value.txt',environment),'one');
  const setup=new Set(f.events.filter(e=>e.type==='execution-submitted'&&e.accountingExcluded&&e.handle?.includes(':job:')).map(e=>e.handle));
  assert.ok(setup.size>=2);
  for(const handle of setup){
    const stopped=f.events.filter(e=>e.type==='execution-state'&&e.handle===handle&&e.state==='stopped');
    assert.ok(stopped.length);
    assert.ok(stopped.every(e=>e.accountingExcluded===true&&e.executionSeconds===0));
  }
  assert.equal(f.events.some(e=>e.type==='accounting-unresolved'&&setup.has(e.handle)),false);
  assert.equal(f.sourceUploads().length,1);
  assert.equal(f.runs(false).length,1);
});
