import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {State} from '../../controller/state.mjs';
import {superviseCampaign} from '../../cli/campaign-supervisor.mjs';
import {workloads} from '../../cli/campaign-manifest.mjs';
import {processIdentity} from '../../controller/lifecycle.mjs';
import {memoryDecision,memorySettings,sampleCampaignMemory,requestMemoryPause,releaseKnowledgeService} from '../../cli/campaign-memory.mjs';

const settings=memorySettings({lowBytes:20,softBytes:24,checkpointBytes:28,hardBytes:32,reduceMs:30,highMs:120,coolMs:60,pollMs:2});
test('sustained memory reduces admission without killing work; cooldown preserves the reduced cap',()=>{
  const decide=(state,rssBytes,now)=>memoryDecision(state,{rssBytes,now,maximum:4},settings);
  let state=decide(null,25,0);assert.equal(state.limit,3);assert.equal(state.blocked,true);
  state=decide(state,25,29);assert.equal(state.limit,3);
  state=decide(state,25,30);assert.equal(state.limit,2);
  state=decide(state,25,60);assert.equal(state.limit,1);
  state=decide(state,25,120);assert.equal(state.action,'checkpoint');
  state={...state,token:'own-pause'};
  state=decide(state,19,122);assert.equal(state.action,'cooling');
  state=decide(state,21,180);assert.equal(state.lowSince,null);
  state=decide(state,19,182);state=decide(state,19,242);assert.equal(state.action,'resume');
  state=decide({...state,token:null,recovered:true},19,244);assert.equal(state.limit,1);assert.equal(state.blocked,false);
  assert.equal(decide(state,28,246).action,'exhausted');
  assert.equal(decide(state,32,246).action,'emergency');
  assert.throws(()=>memorySettings({softBytes:40}));
});

test('memory-pause ownership never replaces an operator pause or stop',async t=>{
  const root=await mkdtemp(join(tmpdir(),'memory-pause-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  state.db.exec('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT)');
  for(const key of ['pause','operator-stop']) {
    state.db.prepare('INSERT OR REPLACE INTO control VALUES (?,?)').run(key,'true');
    assert.throws(()=>requestMemoryPause({root},'mine'),/Operator/);
    assert.equal(state.db.prepare("SELECT value FROM control WHERE key='memory-pause'").get(),undefined);
    state.db.prepare('DELETE FROM control WHERE key=?').run(key);
  }
  requestMemoryPause({root},'mine');
  assert.equal(JSON.parse(state.db.prepare("SELECT value FROM control WHERE key='memory-pause'").get().value),'mine');
});

test('real allocated child memory is attributed once and released after a scaled pressure checkpoint',{timeout:15000},async t=>{
  const root=await mkdtemp(join(tmpdir(),'memory-process-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const state=new State(join(root,'state.sqlite'));state.db.exec('CREATE TABLE control(key TEXT PRIMARY KEY,value TEXT)');state.close();
  const child=spawn(process.execPath,['-e',"globalThis.data=Buffer.alloc(64*1024**2,1);console.log('ready');setInterval(()=>{},1000)"],{stdio:['ignore','pipe','ignore']});
  const exit=once(child,'exit');t.after(async()=>{if(child.exitCode===null&&child.signalCode===null)child.kill('SIGKILL');await exit;});
  await once(child.stdout,'data');const identity=await processIdentity(child.pid),owner=await processIdentity();
  await writeFile(join(root,'knowledge-service.json'),JSON.stringify({owner:identity}));
  const known=new Map(),baseline=await sampleCampaignMemory({root},owner,{known});
  const large=baseline.largest.find(row=>row.pid===child.pid);assert.ok(large.rssBytes>=64*1024**2);assert.equal(large.role,'knowledge');
  assert.equal(new Set(baseline.largest.map(row=>row.pid)).size,baseline.largest.length);
  assert.equal(Object.hasOwn(large,'argv'),false);
  const scaled=memorySettings({softBytes:baseline.rssBytes-2,checkpointBytes:baseline.rssBytes-1,
    lowBytes:baseline.rssBytes-32*1024**2,hardBytes:baseline.rssBytes+128*1024**2,coolMs:20});
  const q={expected:{sourceRevision:'fixture'},localValidation:{passed:true,sourceRevision:'fixture'},receipts:[]};
  for(const target of ['a3','a5']) {
    q.expected[target]={};
    for(const workload of workloads) {
      q.expected[target][workload]={sourceHash:'source',caseHash:'cases',verifierHash:'verifier',independentCheckHash:'checker',runtimeFingerprint:target,caseIds:[1]};
      q.receipts.push({format:1,qualified:true,target,workload,sourceHash:'source',caseHash:'cases',verifierHash:'verifier',
        independentChecks:{passed:true,workload,checkerHash:'checker'},runtime:{fingerprint:target},
        cases:[{id:1,passed:true,referenceHash:'ref',outputHash:'out',maxAbsError:0}],timing:{passed:true,samples:[{id:1,seconds:[1]}]}});
    }
  }
  const config={root,local_memory:{...scaled,pollMs:10,coolMs:30},concurrency:4,campaign_cohort:'fixture',campaign_source_revision:'fixture',campaign_seconds:600,campaign_plan:[{id:'entry'}]};
  const reports=[],budget={cycle:'same',elapsed:123};let resumed=false,ack=0;
  const result=await superviseCampaign({instance:root,cohort:'fixture',sourceRevision:'fixture',qualification:q},{
    read:async()=>config,identity:async()=>owner,claim:async()=>async()=>{},load:async()=>null,
    persist:async(_config,report)=>reports.push(structuredClone(report)),space:async()=>100*1024**3,closed:async()=>true,
    snapshot:async()=>{
      const db=new State(join(root,'state.sqlite')),get=key=>JSON.parse(db.db.prepare('SELECT value FROM control WHERE key=?').get(key)?.value??'null');
      try{return {budget,operation:'campaign',attempts:1,paused:Boolean(get('pause')),memoryPause:get('memory-pause'),
        status:{controller:{state:get('pause')?'stopped':'running'},progress:{done:resumed&&ack++>0}}};}finally{db.close();}
    },
    cli:async argv=>{
      if(argv[0]==='resume') {
        resumed=true;const db=new State(join(root,'state.sqlite'));
        db.db.prepare("UPDATE control SET value='false' WHERE key='pause'").run();db.close();return {request:'resume'};
      }
      return {};
    },
  });
  assert.equal(result.reason,'complete');assert.equal(resumed,true);assert.equal(result.memory.limit,3);
  assert.ok(reports.some(report=>report.state==='checkpointing'));assert.ok(reports.some(report=>report.state==='cooling'));
  assert.ok(reports.some(report=>report.reason==='memory-recovered'));assert.deepEqual(budget,{cycle:'same',elapsed:123});
  const released=await sampleCampaignMemory({root},owner,{known});assert.ok(baseline.rssBytes-released.rssBytes>=32*1024**2);
});
