import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {Worker} from 'node:worker_threads';
import {once} from 'node:events';
import {DatabaseSync} from 'node:sqlite';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';

async function fixture(t,seconds=2,nativeWait=20) {
  const root=await mkdtemp(join(tmpdir(),'sqlite-backoff-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const file=join(root,'state.sqlite'),state=new State(file,{contentionSeconds:seconds});
  t.after(()=>state.close());
  if(nativeWait!==null)state.db.exec(`PRAGMA busy_timeout=${nativeWait}`);
  state.add({id:'one',goal:'persist actual state',actor_seconds:30});
  const attempt=state.admit('one',{});return {state,file,attempt};
}
async function lock(t,file,{milliseconds=350,read=false}={}) {
  const worker=new Worker(`const {parentPort,workerData}=await import('node:worker_threads');
    const {DatabaseSync}=await import('node:sqlite');const db=new DatabaseSync(workerData.file);
    db.exec(workerData.read?'BEGIN':'BEGIN IMMEDIATE');
    if(workerData.read)db.prepare('SELECT * FROM projects').all();
    parentPort.postMessage('locked');
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)),0,0,workerData.milliseconds);
    db.exec('ROLLBACK');db.close();`,{eval:true,workerData:{file,milliseconds,read}});
  t.after(()=>worker.terminate());await once(worker,'message');return worker;
}

test('event write survives an independent writer and is committed exactly once',async t=>{
  const f=await fixture(t);await lock(t,f.file);const start=performance.now();
  f.state.event(f.attempt.id,'observed',{value:7});
  assert.ok(performance.now()-start>=250);
  assert.deepEqual(f.state.queryEvents(f.attempt.id,{types:['observed']}).map(x=>x.data),[{value:7}]);
});

test('checkpoint and direct control writes survive contention without losing elapsed or session',async t=>{
  const f=await fixture(t);await lock(t,f.file);
  f.state.checkpoint(f.attempt.id,12.5,'session-one');
  assert.equal(f.state.attempt(f.attempt.id).elapsed,12.5);
  assert.equal(f.state.attempt(f.attempt.id).session,'session-one');
  f.state.db.exec('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT)');
  await lock(t,f.file);f.state.db.prepare('INSERT INTO control VALUES (?,?)').run('phase2-hold','true');
  assert.equal(f.state.db.prepare('SELECT value FROM control WHERE key=?').get('phase2-hold').value,'true');
});

test('admission waits before its transaction and does not duplicate attempts or outcomes',async t=>{
  const f=await fixture(t);f.state.add({id:'two',goal:'one atomic admission',actor_seconds:30});
  await lock(t,f.file);const attempt=f.state.admit('two',{});
  assert.equal(f.state.db.prepare('SELECT COUNT(*) n FROM attempts WHERE project=?').get('two').n,1);
  assert.equal(f.state.project('two').status,'running');
  await lock(t,f.file);f.state.finish(attempt.id,'passed');
  assert.equal(f.state.project('two').status,'passed');assert.equal(f.state.attempt(attempt.id).outcome,'passed');
  assert.throws(()=>f.state.admit('two',{}),/not eligible/);
  assert.equal(f.state.db.isTransaction,false);
});

test('a persistent writer fails within the elapsed allowance with original SQLite error and no event',async t=>{
  const f=await fixture(t,.14);await lock(t,f.file,{milliseconds:1000});const start=performance.now();
  assert.throws(()=>f.state.event(f.attempt.id,'uncommitted',{}),e=>e.code==='ERR_SQLITE_ERROR'&&e.errcode===5);
  const elapsed=performance.now()-start;assert.ok(elapsed>=100&&elapsed<350,`elapsed=${elapsed}`);
  assert.equal(f.state.queryEvents(f.attempt.id,{types:['uncommitted']}).length,0);
});

test('constraint errors are immediate and do not change attempt state',async t=>{
  const f=await fixture(t);const start=performance.now();
  assert.throws(()=>f.state.add({id:'one',goal:'duplicate',actor_seconds:30}),e=>e.code==='ERR_SQLITE_ERROR'&&(e.errcode&255)===19);
  assert.ok(performance.now()-start<150);
  assert.equal(f.state.project('one').status,'running');assert.equal(f.state.attempt(f.attempt.id).outcome,null);
});

test('stale WAL snapshots fail immediately rather than retrying a locked transaction',async t=>{
  const f=await fixture(t),other=new DatabaseSync(f.file);t.after(()=>other.close());
  f.state.db.exec('BEGIN');f.state.project('one');
  other.prepare('UPDATE projects SET frozen=1 WHERE id=?').run('one');const start=performance.now();
  assert.throws(()=>f.state.event(f.attempt.id,'stale-snapshot',{}),e=>e.code==='ERR_SQLITE_ERROR'&&e.errcode===517);
  assert.ok(performance.now()-start<150);assert.equal(f.state.db.isTransaction,true);
  f.state.db.exec('ROLLBACK');assert.equal(f.state.queryEvents(f.attempt.id,{types:['stale-snapshot']}).length,0);
});

test('arbitrary multi-statement exec never replays earlier side effects on BUSY',async t=>{
  const f=await fixture(t);let invoked=0;f.state.db.function('touch',()=>{invoked++;return 1;});
  await lock(t,f.file,{milliseconds:600});
  assert.throws(()=>f.state.db.exec("SELECT touch(); INSERT INTO events(type,data,created) VALUES ('blocked','{}','now')"),e=>e.errcode===5);
  assert.equal(invoked,1);assert.equal(f.state.queryEvents(null,{types:['blocked']}).length,0);
});

test('opening an existing State waits for schema initialization without rewriting prior events',async t=>{
  const f=await fixture(t);f.state.event(f.attempt.id,'retained',{before:true});
  await lock(t,f.file,{milliseconds:450});const other=new State(f.file,{contentionSeconds:2});t.after(()=>other.close());
  assert.deepEqual(other.queryEvents(f.attempt.id,{types:['retained']}).map(x=>x.data),[{before:true}]);
  other.event(f.attempt.id,'after-open',{});assert.equal(f.state.queryEvents(f.attempt.id,{types:['after-open']}).length,1);
});

test('COMMIT can wait for a reader without replaying the transaction body',async t=>{
  const f=await fixture(t);f.state.db.exec('PRAGMA journal_mode=DELETE');await lock(t,f.file,{read:true});
  f.state.db.exec('BEGIN IMMEDIATE');f.state.event(f.attempt.id,'commit-once',{});
  f.state.db.exec('COMMIT');assert.equal(f.state.db.isTransaction,false);
  assert.equal(f.state.queryEvents(f.attempt.id,{types:['commit-once']}).length,1);
});

// Constructors outside State still use unreplayed, multi-statement exec.
test('RecoveryQueue bootstrap retains native waiting while prepared retries restore caller policy',async t=>{
  const f=await fixture(t,2,null);
  await lock(t,f.file);const queue=new RecoveryQueue(f.state);
  f.state.finish(f.attempt.id,'failed');queue.enqueue(f.attempt.id,0);
  assert.equal(queue.entry('one').origin,f.attempt.id);
  f.state.db.exec('PRAGMA busy_timeout=20');await lock(t,f.file);
  f.state.event(f.attempt.id,'policy-restored',{});
  assert.equal(f.state.db.prepare('PRAGMA busy_timeout').get().timeout,20);
  assert.equal(f.state.queryEvents(f.attempt.id,{types:['policy-restored']}).length,1);
});
