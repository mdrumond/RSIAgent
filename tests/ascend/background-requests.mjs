import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm,writeFile,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {operate} from '../../cli/runtime.mjs';
import {State} from '../../controller/state.mjs';
import {processIdentity} from '../../controller/lifecycle.mjs';
import {requests,reconcileRequests,trackRequest,startBackground} from '../../cli/background.mjs';

async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'background-requests-'));
  const state=new State(join(root,'state.sqlite'));requests(state);
  t.after(async()=>{state.close();await rm(root,{recursive:true,force:true});});
  const insert=(id,status,owner=null,result=null)=>state.db.prepare(
    'INSERT INTO requests(id,kind,status,owner,result) VALUES (?,\'kdb\',?,?,?)')
    .run(id,status,owner?JSON.stringify(owner):null,result?JSON.stringify(result):null);
  return {root,state,insert};
}

test('observing an exited request marks failure without replaying its work',async t=>{
  const {root,state,insert}=await fixture(t);
  insert('exited','running',{pid:2147483647,start:'1',boot:'old'});
  const [row]=await reconcileRequests(root,{state});
  assert.equal(row.status,'failed');assert.equal(row.result,null);
  assert.equal(state.db.prepare('SELECT COUNT(*) n FROM requests').get().n,1);
  assert.deepEqual(await reconcileRequests(root,{state}),[row]);
});

test('observing a live request or a request without known ownership leaves it pending',async t=>{
  const {root,state,insert}=await fixture(t);
  insert('live','running',await processIdentity());insert('queued','queued');
  const rows=await reconcileRequests(root,{state});
  assert.equal(rows.find(row=>row.id==='live').status,'running');
  assert.equal(rows.find(row=>row.id==='queued').status,'queued');
});

test('a completed request retains its result after its owner exits',async t=>{
  const {root,state,insert}=await fixture(t);
  insert('completed','passed',{pid:2147483647,start:'1',boot:'old'},{indexed:3});
  const [row]=await reconcileRequests(root,{state});
  assert.equal(row.status,'passed');assert.deepEqual(row.result,{indexed:3});
});

test('request execution publishes success or failure once and refuses replay',async t=>{
  const {root,state,insert}=await fixture(t);let executions=0;
  insert('success','queued');insert('failure','queued');
  const operation=async()=>{executions++;return {indexed:2};};
  assert.deepEqual(await trackRequest(root,'success',operation),{indexed:2});
  await assert.rejects(trackRequest(root,'success',operation));
  assert.equal(executions,1);
  await assert.rejects(trackRequest(root,'failure',async()=>{throw new Error('index unavailable');}));
  const rows=requests(state);
  assert.equal(rows.find(row=>row.id==='failure').status,'failed');
  assert.equal(rows.find(row=>row.id==='success').status,'passed');
  assert.deepEqual(rows.find(row=>row.id==='success').result,{indexed:2});
});


test('explicit restart refuses a surviving worker without duplicating or killing it',async t=>{
  const {root,state}=await fixture(t);
  const child=spawn(process.execPath,['--max-old-space-size=32','-e','setInterval(()=>{},1000)'],{stdio:'ignore'});
  await once(child,'spawn');
  t.after(async()=>{const closed=once(child,'exit');child.kill('SIGTERM');await closed;});
  state.add({id:'still-running',goal:'unchanged',actor_seconds:30});
  const attempt=state.admit('still-running',await processIdentity(child.pid));
  await assert.rejects(operate('resume',{root,mode:'phase1',projects:[]}));
  assert.equal(state.db.prepare('SELECT COUNT(*) n FROM attempts').get().n,1);
  assert.equal(state.attempt(attempt.id).outcome,null);
  assert.equal(child.exitCode,null);
});


test('durable memory resumes carry their pause token and refuse a superseding operator control',async t=>{
  const {root,state}=await fixture(t);
  await writeFile(join(root,'instance.json'),JSON.stringify({format:1,mode:'phase1',projects:[]}));
  state.db.exec('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
  for(const stopped of [false,true]) {
    for(const [key,value] of Object.entries({'pause':true,'memory-pause':stopped?'old-memory-token':null,'operator-stop':stopped,'stop':stopped}))
      state.db.prepare('INSERT OR REPLACE INTO control VALUES (?,?)').run(key,JSON.stringify(value));
    const before=state.db.prepare('SELECT * FROM control ORDER BY key').all();
    await assert.rejects(startBackground(root,'resume',{'memory-pause-token':'old-memory-token'},{detach:false}),/superseded/);
    assert.deepEqual(state.db.prepare('SELECT * FROM control ORDER BY key').all(),before);
    const request=requests(state)[0];assert.equal(request.kind,'resume');assert.equal(request.status,'failed');
    const input=JSON.parse(await readFile(join(root,'requests',request.id,'input.json'),'utf8'));
    assert.equal(input['memory-pause-token'],'old-memory-token');
  }
  assert.equal(state.db.prepare('SELECT count(*) n FROM attempts').get().n,0);
  await writeFile(join(root,'instance.json'),JSON.stringify({format:1,mode:'phase1',projects:[],campaign_seconds:1}));
  const identity=await processIdentity();
  for(const [key,value] of Object.entries({'pause':true,'memory-pause':'current-token','operator-stop':false,'stop':true,
    operation:'campaign','campaign-budget':{cycle:'retained',seconds:1,elapsed:1,boot:identity.boot,monotonic:0,running:false}}))
    state.db.prepare('INSERT OR REPLACE INTO control VALUES (?,?)').run(key,JSON.stringify(value));
  const result=await startBackground(root,'resume',{'memory-pause-token':'current-token'},{detach:false});
  assert.equal(result.reason,'elapsed');assert.equal(requests(state)[0].status,'passed');
  assert.equal(JSON.parse(state.db.prepare("SELECT value FROM control WHERE key='pause'").get().value),false);
  assert.equal(JSON.parse(state.db.prepare("SELECT value FROM control WHERE key='campaign-budget'").get().value).cycle,'retained');
});
