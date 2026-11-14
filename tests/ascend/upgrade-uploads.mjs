import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { syncBuiltinESMExports } from 'node:module';
import { State } from '../../controller/state.mjs';
import { RecoveryQueue } from '../../controller/recovery.mjs';
import { attemptStopped, processIdentity } from '../../controller/lifecycle.mjs';
import { upgrade } from '../../cli/instance.mjs';
import { createEnvironment } from '../../integrations/environments/index.mjs';

const generic = new URL('../../integrations/environments/index.mjs', import.meta.url).href;
async function fixture(t, manifest = false) {
  const root = await fs.mkdtemp(join(tmpdir(), 'upgrade-uploads-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const instance = join(root, 'instance'), release = join(root, 'release');
  for (const path of [instance, release]) await fs.mkdir(join(path, 'controller'), { recursive: true });
  await fs.mkdir(join(instance, 'machinery', 'controller'), { recursive: true });
  await fs.writeFile(join(instance, 'machinery', 'controller', 'marker.mjs'), 'old');
  await fs.writeFile(join(release, 'controller', 'marker.mjs'), 'new');
  const options = { mode: 'cpl', target: 'fake', workspacePath: '/registered', remoteDirectory: 'campaign', pollMs: 0 };
  const pluginPath = 'integrations/environments/fixture.mjs';
  await fs.mkdir(join(release, dirname(pluginPath)), { recursive: true });
  const observations = join(root, 'observations.json');
  await fs.writeFile(join(release, pluginPath), `import {readFile} from 'node:fs/promises';
import {createEnvironment as generic} from ${JSON.stringify(generic)};
export function createEnvironment(options,context){return generic(options,{...context,
 sdk:{createLocalBashOperations:()=>({})},client:async args=>{
 if(args[0]!=='observe')throw new Error('Upgrade must not dispatch or download');
 const result=JSON.parse(await readFile(${JSON.stringify(observations)},'utf8'))[args[1]];
 if(result?.observationError)throw new Error('Retained transport unavailable');
 return result??{state:'unknown'};
 }});}`);
  await fs.writeFile(join(instance, 'instance.json'), JSON.stringify({ format: 1, mode: 'phase1',
    environment: { module: 'machinery/' + pluginPath, options } }));
  const state = new State(join(instance, 'state.sqlite'));
  t.after(() => state.close());
  const dead = { ...await processIdentity(), start: 'previous-process' };
  state.add({ id: 'old', goal: 'retained task', actor_seconds: 90 });
  const attempt = state.admit('old', dead);
  const path = join(instance, 'attempts', attempt.id);
  await fs.mkdir(path, { recursive: true });
  const session = join(path, 'native.jsonl');
  await fs.writeFile(session, '{"retained":"native-session"}\n');
  state.checkpoint(attempt.id, 12.5, session);
  new RecoveryQueue(state);
  state.db.prepare('INSERT INTO resource_leases VALUES (?,?)').run('held-device', attempt.id);
  const project = join(root, 'project');await fs.mkdir(project);await fs.writeFile(join(project, 'candidate.py'), 'candidate');
  let upload = 0, receiptPath, handle;
  const statuses = {};
  const event = async (type, data) => state.event(attempt.id, type, { ...data, operation: 'old-operation', actor_elapsed: 12.5 });
  const environment = await createEnvironment(options, { instancePath: path, sdk: { createLocalBashOperations: () => ({}) },
    event, exclude: async (_, fn) => fn(), client: async args => {
      if (args[0] === 'capabilities') return { state: 'available' };
      if (args[0] === 'preflight') return { state: 'completed' };
      if (args[0] === 'download') return { state: 'completed' };
      if (args[0] === 'upload') {
        upload++;
        const current = `remote:fake:transfer:upload${upload}`;
        const receipt = args[args.indexOf('--receipt-file') + 1];
        await fs.writeFile(receipt, JSON.stringify({ operation: 'upload', target: 'fake', handle: current, state: 'queued' }));
        const pending = !manifest || upload === 2;
        statuses[current] = { state: pending ? 'unknown' : 'completed' };
        if (pending) { receiptPath = receipt; handle = current; }
        return { state: pending ? 'running' : 'completed', handle: current };
      }
      if (args[0] === 'observe') return statuses[args[1]];
      throw new Error('Unexpected dispatch');
    } });
  await environment.prepare({ project: { source: project }, attempt });
  await event('execution-starting', { operation: 'old-operation', startElapsed: 12.5 });
  await assert.rejects(environment.execute('do not replay', project), /Transfer unresolved/);
  await fs.writeFile(observations, JSON.stringify(statuses));
  const snapshot = async () => ({
    attempts: state.db.prepare('SELECT * FROM attempts').all(),
    projects: state.db.prepare('SELECT * FROM projects').all(),
    leases: state.db.prepare('SELECT * FROM resource_leases').all(),
    events: state.db.prepare('SELECT * FROM events').all(),
    session: await fs.readFile(session, 'utf8'),
    receipts: await Promise.all(Object.keys(statuses).map(async id => {
      const event = state.events(attempt.id).find(e => e.type === 'execution-submitted' && e.data.handle === id);
      const submission = state.events(attempt.id).find(e => e.type === 'execution-submitting' && e.data.submission === event.data.submission);
      return fs.readFile(submission.data.receiptFile, 'utf8');
    })) });
  return { instance, release, state, attempt, path, options, receiptPath, handle, statuses, observations,
    event, snapshot, environment, session };
}

for (const manifest of [false, true]) test(`upgrade installs machinery with pending ${manifest ? 'manifest' : 'source'} upload; resume remains guarded`, async t => {
  const f = await fixture(t, manifest), before = await f.snapshot();
  await fs.writeFile(f.observations, JSON.stringify({ ...f.statuses, [f.handle]: { observationError: true } }));
  await upgrade(f.instance, f.release);
  assert.equal(await fs.readFile(join(f.instance, 'machinery/controller/marker.mjs'), 'utf8'), 'new');
  const archives = await fs.readdir(join(f.instance, 'archives'));
  assert.equal(archives.length, 1);
  assert.equal(await fs.readFile(join(f.instance, 'archives', archives[0], 'controller/marker.mjs'), 'utf8'), 'old');
  assert.deepEqual(await f.snapshot(), before);
  assert.equal(await attemptStopped(f.state, f.state.attempt(f.attempt.id), f.environment), false);
  await assert.rejects(f.state.resume(f.attempt.id, a => attemptStopped(f.state, a, f.environment), await processIdentity()), /not confirmed stopped/);
  assert.deepEqual(f.state.attempt(f.attempt.id), before.attempts[0]);
});

for (const kind of ['worker', 'settled-worker', 'controller', 'writer', 'request', 'queued-request', 'unknown-owner'])
  test(`upgrade refuses ${kind} owner`, async t => {
    const f = await fixture(t), me = JSON.stringify(await processIdentity());
    if (kind.includes('worker') || kind === 'unknown-owner') {
      f.state.db.prepare('UPDATE attempts SET owner=?').run(kind === 'unknown-owner' ? '{}' : me);
      if (kind === 'settled-worker') f.state.finish(f.attempt.id, 'failed');
    } else if (kind === 'controller' || kind === 'writer') {
      f.state.db.exec('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
      f.state.db.prepare('INSERT INTO control VALUES (?,?)').run(kind === 'writer' ? 'skill-writer' : 'owner', me);
    } else {
      f.state.db.exec('CREATE TABLE requests(id TEXT,status TEXT,owner TEXT)');
      f.state.db.prepare('INSERT INTO requests VALUES (?,?,?)').run('request', kind === 'queued-request' ? 'queued' : 'running', me);
    }
    await assert.rejects(upgrade(f.instance, f.release), /[Ss]top/);
    assert.equal(await fs.readFile(join(f.instance, 'machinery/controller/marker.mjs'), 'utf8'), 'old');
  });

for (const kind of ['missing-receipt', 'wrong-target', 'wrong-handle', 'foreign-path', 'missing-response', 'run', 'download', 'unknown-submission', 'unmapped-handle', 'manifest-destination'])
  test(`upgrade refuses ${kind} evidence`, async t => {
    const f = await fixture(t, kind === 'manifest-destination');
    const receipt = JSON.parse(await fs.readFile(f.receiptPath, 'utf8'));
    if (kind === 'missing-receipt') await fs.rm(f.receiptPath);
    if (kind === 'wrong-target' || kind === 'wrong-handle') {
      receipt[kind === 'wrong-target' ? 'target' : 'handle'] = 'different';
      await fs.writeFile(f.receiptPath, JSON.stringify(receipt));
    }
    if (kind === 'manifest-destination') {
      const path = join(dirname(f.receiptPath), 'sync.json'), config = JSON.parse(await fs.readFile(path, 'utf8'));
      config.incoming = '/outside/attempt/incoming';await fs.writeFile(path, JSON.stringify(config));
    }
    if (kind === 'missing-response') f.state.db.prepare("DELETE FROM events WHERE type='execution-submitted'").run();
    if (['foreign-path', 'run', 'download'].includes(kind)) {
      const row = f.state.db.prepare("SELECT sequence,data FROM events WHERE type='execution-submitting'").get();
      const data = JSON.parse(row.data);
      if (kind === 'foreign-path') {
        const path = join(f.instance, 'foreign-transfer.json');await fs.writeFile(path, JSON.stringify(receipt));data.receiptFile = path;
      } else data.kind = kind;
      f.state.db.prepare('UPDATE events SET data=? WHERE sequence=?').run(JSON.stringify(data), row.sequence);
    }
    if (kind === 'unknown-submission') f.state.event(f.attempt.id, 'execution-unresolved', { operation: 'missing', handle: null });
    if (kind === 'unmapped-handle') f.state.event(f.attempt.id, 'execution-handle', { handle: 'remote:fake:job:unmapped' });
    const before = await fs.readFile(join(f.instance, 'machinery/controller/marker.mjs'), 'utf8');
    await assert.rejects(upgrade(f.instance, f.release), /external executions/);
    assert.equal(await fs.readFile(join(f.instance, 'machinery/controller/marker.mjs'), 'utf8'), before);
  });

test('known terminal native job permits installation; ambiguous job never does', async t => {
  const f = await fixture(t);
  const job = 'remote:fake:job:retained', submission = 'job-submission';
  f.state.event(f.attempt.id, 'execution-submitting', { submission, operation: 'job-operation', kind: 'run', accountingExcluded: true });
  f.state.event(f.attempt.id, 'execution-submitted', { submission, handle: job });
  for (const result of [{ state: 'running' }, { state: 'unknown' }, { observationError: true }]) {
    await fs.writeFile(f.observations, JSON.stringify({ ...f.statuses, [job]: result }));
    await assert.rejects(upgrade(f.instance, f.release), /external executions/);
  }
  await fs.writeFile(f.observations, JSON.stringify({ ...f.statuses, [job]: { state: 'completed', exit: 0 } }));
  await upgrade(f.instance, f.release);
});

for (const kind of ['download', 'upload', 'unknown-download', 'nonexcluded-download', 'closed-run'])
  test(`handleless synchronous transfer eligibility: ${kind}`, async t => {
    const f = await fixture(t);
    await f.environment.browse('/retained/output');
    const row = f.state.db.prepare("SELECT sequence,data FROM events WHERE type='execution-submitting' ORDER BY sequence DESC LIMIT 1").get();
    const data = JSON.parse(row.data);
    if (kind === 'upload') data.kind = 'upload';
    if (kind === 'closed-run') data.kind = 'run';
    if (kind === 'nonexcluded-download') data.accountingExcluded = false;
    f.state.db.prepare('UPDATE events SET data=? WHERE sequence=?').run(JSON.stringify(data), row.sequence);
    if (kind === 'unknown-download') {
      const response = f.state.db.prepare("SELECT sequence,data FROM events WHERE type='execution-submitted' ORDER BY sequence DESC LIMIT 1").get();
      f.state.db.prepare('UPDATE events SET data=? WHERE sequence=?')
        .run(JSON.stringify({ ...JSON.parse(response.data), closed: false }), response.sequence);
    }
    const before = await f.snapshot();
    if (kind === 'download' || kind === 'upload') await upgrade(f.instance, f.release);
    else await assert.rejects(upgrade(f.instance, f.release), /external executions/);
    assert.deepEqual(await f.snapshot(), before);
  });

test('failed staged activation restores archived machinery and leaves attempt evidence unchanged', async t => {
  const f = await fixture(t), before = await f.snapshot();
  const rename = fs.rename;
  fs.rename = async (from, to) => {
    if (from.includes('machinery-next-') && to === join(f.instance, 'machinery')) throw new Error('activation failure');
    return rename(from, to);
  };
  syncBuiltinESMExports();
  try { await assert.rejects(upgrade(f.instance, f.release), /activation failure/); }
  finally { fs.rename = rename;syncBuiltinESMExports(); }
  assert.equal(await fs.readFile(join(f.instance, 'machinery/controller/marker.mjs'), 'utf8'), 'old');
  assert.deepEqual(await f.snapshot(), before);
});

for(const kind of ['active-own','active-other','expired-own'])test(`public upgrade respects cleanup writer ownership: ${kind}`,async t=>{
 const f=await fixture(t);f.state.finish(f.attempt.id,'failed');
 const {queueDeviceCleanup,drainDeviceCleanup}=await import('../../integrations/environments/device-cleanup.mjs');
 const {main}=await import('../../cli/main.mjs');
 const directory=join(f.instance,'cleanup-queue'),other=join(f.instance,'other-campaign');await fs.mkdir(other);
 const campaignRoot=kind==='active-other'?other:f.instance;let ticket;
 await queueDeviceCleanup(directory,{id:'retained-cleanup',token:'retained-cleanup',receipt:'retained-receipt',pointer:'retained-pointer',
  campaignRoot,deadline:Date.now()+(kind==='expired-own'?-1:30000)},
  {launch:async(_directory,key)=>{ticket=key;return processIdentity();}});
 // Another campaign can share the service while this instance has no live jobs.
 if(kind==='active-other')await fs.copyFile(join(other,'device-cleanup-service.json'),join(f.instance,'device-cleanup-service.json'));
 if(kind==='expired-own')await drainDeviceCleanup(directory,ticket,{run:async()=>{throw Error('Expired job must not run');}});
 const action=join(f.instance,'upgrade.json');await fs.writeFile(action,JSON.stringify({machinery:f.release}));
 const run=()=>main(['upgrade','--instance',f.instance,'--config',action]);
 if(kind==='active-own'){
  await assert.rejects(run(),/active device cleanup writers/);
  assert.equal(await fs.readFile(join(f.instance,'machinery/controller/marker.mjs'),'utf8'),'old');
 }else{
  await run();assert.equal(await fs.readFile(join(f.instance,'machinery/controller/marker.mjs'),'utf8'),'new');
 }
});
