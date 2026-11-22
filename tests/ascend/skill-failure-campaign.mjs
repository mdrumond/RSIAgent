import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {State} from '../../controller/state.mjs';
import {SkillBacklog} from '../../controller/skill-backlog.mjs';
import {claimLearningRequest} from '../../controller/skill-review.mjs';
import {main} from '../../cli/main.mjs';
import {runCampaignFailureStep,failureCampaignStatus,failureCampaignSettled} from '../../cli/skill-failure-campaign.mjs';
import {runFailureCampaign,failureProtocol,selectFailureSlots} from '../../cli/skill-failure-worker.mjs';

const slots=Array.from({length:4},(_,i)=>({resource:{id:'slot-'+i,available:true,metadata:{kind:'host-slot',pool:'cpu-fixture',hostSlot:i}},physical:i}));
const task=id=>({id,caseIds:['large','control'],controlCaseIds:['control'],files:{'kernel.py':'starter'},prompt:'Repair the isolated behavior'});
const family={id:'actual-failure',development:task('development'),transfer:[task('layout'),task('aggregate')],evidence:[]};
const result=correct=>({correct,mechanismMatch:!correct,caseResults:['large','control'].map(id=>({id,correct})),sourceSha256:'a'.repeat(64),artifacts:[],handles:[]});
async function fixture(t,target='a3'){
 const root=await mkdtemp(join(tmpdir(),'failure-campaign-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const config={format:1,root,target,mode:'phase1',projects:[],concurrency:5,actor:{},environment:{options:{target:'fixture'}},skills:{activeVersion:'base',background:true,tuningDisabled:false,failureFirst:{enabled:true,experimentPolicy:'legacy',sourceRevision:'a'.repeat(40)}}};
 await writeFile(join(root,'instance.json'),JSON.stringify(config));const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());state.db.exec('CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
 for(const [key,value]of Object.entries({pause:true,'campaign-budget':{seconds:10000,elapsed:125,cycle:'retained',boot:'prior-boot',monotonic:0,running:false}}))state.db.prepare('insert into control values(?,?)').run(key,JSON.stringify(value));
 state.event(null,'generation-rejected',{message:'Actual test fixture rejection'});const backlog=new SkillBacklog(state,{target});backlog.ingest();const g=backlog.failures()[0];backlog.acknowledge(g.id);
 const item=backlog.upsert({key:'first',workflow:'failure-first-v1',priority:10,reason:'Observed failure',objective:'correctness',skill:'implementation',base:'base',groupIds:[g.id],proposal:null,evidence:{groups:[g]}});
 const calls=[],memory={config:{root:join(root,'view')},knowledge:{identity:'frozen-memory'},close:async()=>calls.push('memory-close')};
 const ops={analyze:async()=>{},family:async()=>({status:'ready',family}),model:async()=>({id:'model'}),input:async()=>({text:'baseline-guidance',sha256:'skill-hash'}),
  runtime:async()=>({identify:async()=>({product:'fixture'}),qualify:async()=>({qualified:true,referencePassed:true,negativeFailed:true}),trial:async input=>{calls.push(['trial',input.phase,input.arm,input.actorSeconds,input.verificationSeconds]);return result(input.phase==='screen'?false:input.arm!=='baseline');},settle:async()=>({closed:true}),close:async()=>true}),
  design:async()=>async input=>{calls.push(['design',input.family]);return ['a','b','c'].map(id=>({id,text:id,evidence:[{reference:'observed'}]}));},
  publish:async input=>{calls.push(['publish',input.directory]);return {published:true};}};
 return {root,config,state,backlog,item,calls,memory,ops,values:{slots,memory,remainingSeconds:9000}};
}
test('production campaign step screens before writing and requires both transfers before publication',async t=>{
 const f=await fixture(t),outcome=await runCampaignFailureStep(f.config,f.values,f.ops);assert.equal(outcome.status,'accepted');
 const design=f.calls.findIndex(c=>c[0]==='design'),trials=f.calls.filter(c=>c[0]==='trial');assert.equal(f.calls.slice(0,design).filter(c=>c[0]==='trial').length,5);assert.equal(trials.length,45);
 assert(trials.every(c=>c[3]===180&&c[4]===300));assert.equal(trials.filter(c=>c[1]==='confirmation').length,20);assert.equal(f.calls.at(-1)[0],'publish');assert(f.calls.at(-1)[1].endsWith('/pilot'));
 assert.equal(f.backlog.run(f.item.id),null);assert.equal(f.backlog.get(f.item.id).status,'accepted');assert.equal((await failureCampaignStatus(f.config)).current,null);
});
test('baseline ceiling advances without designing or publishing an instruction',async t=>{
 const f=await fixture(t);const ops={...f.ops,runtime:async()=>({...await f.ops.runtime(),trial:async()=>result(true)})};
 const outcome=await runCampaignFailureStep(f.config,f.values,ops);assert.equal(outcome.status,'rejected');assert(!f.calls.some(c=>['design','publish'].includes(c[0])));assert.equal(f.backlog.get(f.item.id).status,'rejected');
});
test('unsupported evidence returns needs-evidence without launching legacy learning',async t=>{
 const f=await fixture(t);let runtime=0;const outcome=await runCampaignFailureStep(f.config,f.values,{...f.ops,family:async()=>({status:'needs-evidence',reason:'No recoverable actor source'}),runtime:async()=>{runtime++;}});
 assert.equal(outcome.reason,'needs-evidence');assert.equal(runtime,0);assert.equal(f.backlog.meta('microlearning-active'),null);
});
test('an isolated oracle gap queues one claimable Phase 1 request without running A/B',async t=>{
 const f=await fixture(t),verification={reference:'events:2',sourceSha256:'a'.repeat(64)},nextQuestion='Which dimension is reduced in the masked tail?';
 const disposition={blocker:'unreliable-oracle',route:'phase1',verification,nextQuestion};
 const outcome=await runCampaignFailureStep(f.config,f.values,{...f.ops,
  family:async()=>({status:'needs-evidence',reason:'The two reference formulas disagree on a masked tail',disposition}),
  runtime:async()=>assert.fail('Terminal research gap has no A/B runtime')});
 assert.equal(outcome.status,'learning');
 assert.equal(f.backlog.get(f.item.id).status,'learning');
 assert.equal(f.calls.length,0);
 assert.equal(claimLearningRequest(f.state,f.config,{draft:'wrong-target',target:'a5'}),null);
 const request=claimLearningRequest(f.state,f.config,{draft:'phase1',target:'a3'});
 assert.equal(request.id,outcome.learningRequest);
 assert.equal(request.workKind,'phase1');
 assert.equal(request.question,nextQuestion);
 assert.deepEqual(request.verification,verification);
 assert.equal(claimLearningRequest(f.state,f.config,{draft:'repeat',target:'a3'}),null);
});
test('scientific QA rejection finishes without development trials',async t=>{
 const f=await fixture(t);const outcome=await runCampaignFailureStep(f.config,f.values,{...f.ops,design:async()=>async()=>({status:'rejected',reason:'Unsupported instruction'})});assert.equal(outcome.status,'rejected');assert.equal(f.calls.filter(c=>c[0]==='trial').length,5);
});
test('terminal infrastructure lets admitted siblings finish and prevents new admissions',async t=>{
 const f=await fixture(t);let started=0,finished=0;
 await assert.rejects(runCampaignFailureStep(f.config,f.values,{...f.ops,runtime:async()=>({...await f.ops.runtime(),trial:async input=>{
  if(started++===0)return {infrastructure:true,correct:null,reason:'Memory connection lost'};
  await new Promise(resolve=>setTimeout(resolve,25));input.signal.throwIfAborted();finished++;
  return {infrastructure:true,correct:null,reason:'Independent failure'};
 }})}),/Memory connection lost/);
 assert(started<=4);assert.equal(finished,started-1);assert(!f.calls.some(c=>c[0]==='publish'));
});
test('closed unavailable design ends the item without disabling background evaluation',async t=>{
 const f=await fixture(t);
 const outcome=await runCampaignFailureStep(f.config,f.values,{...f.ops,design:async()=>async()=>({status:'infrastructure',infrastructure:true,correct:null,reason:'Writer allowance exhausted'})});
 assert.equal(outcome.status,'inconclusive');assert.equal(outcome.stage,'design');
 assert.equal(f.backlog.meta('failure-first-current'),null);assert(!f.calls.some(c=>c[0]==='publish'));
});
test('real CLI entrypoint stops A3 on startup failure without consuming projects or replacing campaign budget',async t=>{
 const f=await fixture(t);const before=f.state.db.prepare('select * from projects').all();let opened=0;
 const out=await main(['skills-ab','--instance',f.root,'--once'],{failureDependencies:{settled:async()=>true,slots:async()=>slots,memory:async()=>{opened++;throw Error('Cold memory unavailable');}}});
 assert.equal(out.status,'stopped');assert.equal(opened,1);const updated=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));assert(updated.skills.tuningDisabled&&!updated.skills.background);assert.equal(JSON.parse(f.state.db.prepare("select value from control where key='pause'").get().value),true);assert.deepEqual(f.state.db.prepare('select * from projects').all(),before);
 const budget=JSON.parse(f.state.db.prepare("select value from control where key='campaign-budget'").get().value);assert.equal(budget.cycle,'retained');assert.equal(budget.seconds,10000);assert(budget.elapsed>=125);assert.equal(budget.running,false);
 assert.equal((await main(['status','--instance',f.root])).backgroundTuning.status,'stopped');
});
test('production CLI performs memory readiness before any pipeline step and closes its service',async t=>{
 const f=await fixture(t),events=[];const out=await main(['skills-ab','--instance',f.root,'--once'],{failureDependencies:{settled:async()=>true,slots:async()=>slots,memory:async()=>{events.push('warm');return {...f.memory,close:async()=>events.push('close')};},step:async(_c,values)=>{events.push('step');assert.equal(values.slots.length,4);return {status:'rejected',reason:'baseline-ceiling'};}}});
 assert.equal(out.status,'rejected');assert.deepEqual(events,['warm','step','close']);assert.equal(failureProtocol(f.config).sampleSeconds,480);
});
test('legacy hypothesis grants remain capped and cannot consume failure-first items',async t=>{
 const f=await fixture(t);assert.throws(()=>f.backlog.claim({base:'base',seconds:7200}),/1800/);assert.equal(f.backlog.claim({base:'base'}),null);assert.equal(f.backlog.claim({base:'base',seconds:7200,workflow:'failure-first-v1'}).run.seconds,7200);
});
test('slot admission excludes busy and foreign-leased devices and rejects insufficient capacity',async t=>{
 const f=await fixture(t);const inventory=Array.from({length:6},(_,i)=>({available:i!==0,metadata:{physicalDevice:i,measurements:{idle:i!==0}}}));await writeFile(join(f.root,'foreign-lease'),'owned elsewhere');
 const selected=await selectFailureSlots(f.config,{directory:join(f.root,'admission')},{leaseReceipt:(_pool,id)=>join(f.root,id===1?'foreign-lease':'missing'),createEnvironment:async()=>({inspectDevices:async()=>inventory,resources:async()=>slots.map(s=>s.resource)})});assert.deepEqual(selected.map(s=>s.physical),[5,4,3,2]);
 await assert.rejects(selectFailureSlots(f.config,{directory:join(f.root,'few')},{createEnvironment:async()=>({inspectDevices:async()=>inventory.slice(0,4),resources:async()=>slots.map(s=>s.resource)})}),/Four healthy/);
});
test('settlement status does not start memory or new evaluations',async t=>{
 const f=await fixture(t);assert.equal(await failureCampaignSettled(f.config),true);
 const run=f.backlog.claim({base:'base',seconds:7200,workflow:'failure-first-v1'}).run;f.backlog.checkpoint(run.id,{elapsed:0,checkpoint:{runtimeDirectory:join(f.root,'retained')}});f.backlog.meta('failure-first-current',run.id);
 let observed=0;assert.equal(await failureCampaignSettled(f.config,{settled:async()=>{observed++;return {closed:false};}}),false);assert.equal(observed,1);
});
test('resuming retains the original allocation, elapsed isolation time and device bindings',async t=>{
 const f=await fixture(t),now=Date.now(),run=f.backlog.claim({base:'base',seconds:7200,workflow:'failure-first-v1'}).run;
 const checkpoint={workflow:'failure-first-v1',startedAt:now-1000000,directory:join(f.root,'retained-study'),slots};
 f.backlog.checkpoint(run.id,{elapsed:1000,checkpoint});f.backlog.meta('failure-first-current',run.id);
 const outcome=await runCampaignFailureStep(f.config,f.values,{...f.ops,now:()=>now,family:async()=>({status:'needs-evidence',reason:'Unavailable retained source'})});
 assert.equal(outcome.reason,'needs-evidence');const saved=f.backlog.run(run.id);assert.equal(saved.seconds,7200);assert.equal(saved.elapsed,1000);assert.equal(saved.checkpoint.startedAt,checkpoint.startedAt);assert.deepEqual(saved.checkpoint.slots,slots);
});
test('a resumed stage cannot spend more than the retained remaining parent clock',async t=>{
 const f=await fixture(t),now=Date.now(),run=f.backlog.claim({base:'base',seconds:7200,workflow:'failure-first-v1'}).run;
 f.backlog.checkpoint(run.id,{elapsed:7150,checkpoint:{workflow:'failure-first-v1',startedAt:now-1000,directory:join(f.root,'retained-study'),slots}});
 const result=await runCampaignFailureStep(f.config,f.values,{...f.ops,now:()=>now,family:async({seconds})=>{
  assert.equal(seconds,50);
  return {status:'expired',stage:'isolate',reason:'Isolation elapsed allowance exhausted'};
 },runtime:()=>assert.fail('A spent isolation cannot start qualification')});
 assert.equal(result.status,'inconclusive');assert.equal(result.reason,'stage-elapsed');assert.equal(result.stage,'isolate');
 assert.equal(f.backlog.run(run.id).elapsed,7150);
});
test('resuming never silently changes a frozen baseline or mutates its queue revision',async t=>{
 const f=await fixture(t),run=f.backlog.claim({base:'base',seconds:7200,workflow:'failure-first-v1'}).run;
 const before=f.backlog.get(f.item.id);await assert.rejects(runCampaignFailureStep({...f.config,skills:{...f.config.skills,activeVersion:'other'}},f.values,f.ops),/Frozen experiment baseline changed/);
 assert.deepEqual(f.backlog.get(f.item.id),before);assert.equal(f.backlog.run(run.id).seconds,7200);assert.equal(f.calls.length,0);
});
test('a resumed experiment refuses substitute devices before research or inference',async t=>{
 const f=await fixture(t),run=f.backlog.claim({base:'base',seconds:7200,workflow:'failure-first-v1'}).run;
 const checkpoint={workflow:'failure-first-v1',startedAt:Date.now(),directory:join(f.root,'retained-study'),slots};f.backlog.checkpoint(run.id,{elapsed:0,checkpoint});
 await assert.rejects(runCampaignFailureStep(f.config,{...f.values,slots:slots.map((s,i)=>({...s,physical:i+1}))},f.ops),/exact original resource bindings/);assert.equal(f.calls.length,0);assert.deepEqual(f.backlog.run(run.id).checkpoint.slots,slots);
});
test('higher-ranked legacy proposals stay frozen while failure-first work proceeds',async t=>{
 const f=await fixture(t),legacy=f.backlog.upsert({key:'legacy',priority:999,reason:'Retained older proposal',objective:'correctness',skill:'implementation',base:'base',groupIds:f.item.groupIds,proposal:{action:'update'},evidence:f.item.evidence});
 const before=f.backlog.get(legacy.id);assert.equal((await runCampaignFailureStep(f.config,f.values,f.ops)).status,'accepted');assert.deepEqual(f.backlog.get(legacy.id),before);
});
test('legacy correctness history neither suppresses classification nor claims new-mode work',async t=>{
 const f=await fixture(t);f.backlog.revise(f.item.id,{status:'rejected'});const legacy=f.backlog.upsert({key:'legacy',priority:999,reason:'Retained older proposal',objective:'correctness',skill:'implementation',base:'base',groupIds:f.item.groupIds,proposal:{action:'update'},evidence:f.item.evidence});let classified=0;
 const out=await runCampaignFailureStep(f.config,f.values,{...f.ops,analyze:async()=>{classified++;}});assert.equal(out.status,'idle');assert.equal(classified,1);assert.equal(f.backlog.get(legacy.id).status,'queued');assert.equal((await failureCampaignStatus(f.config)).hasWork,false);
});

test('unresolved submission reference execution aborts isolation and retains custody',async t=>{
 const f=await fixture(t),lost=Error('Reference command custody remains unresolved');
 const ops={...f.ops,
  family:async({checkReferences,signal})=>{
   await assert.rejects(checkReferences({family,seconds:90,signal}),error=>error===lost);
   assert.equal(signal.reason,lost);
   return {status:'needs-evidence',reason:'Model ended'};
  },
  runtime:async()=>({...await f.ops.runtime(),checkReferences:async()=>{throw lost;},close:async()=>false})
 };
 await assert.rejects(runCampaignFailureStep(f.config,f.values,ops),error=>error===lost);
 assert.equal(f.backlog.run(f.item.id).checkpoint.unresolved,true);
 assert.equal(f.calls.length,0);
});

for(const boundary of ['review profile','memory attention'])test('resume rejects changed '+boundary,async t=>{
 const f=await fixture(t);
 f.config.actor={model:'actor'};
 f.config.skill_writer_actor={model:'review-a'};
 f.memory.knowledge.attentionRevision=1;
 await writeFile(join(f.root,'instance.json'),JSON.stringify(f.config));
 let evaluations=0;
 const ops={...f.ops,model:async profile=>profile.model,pilot:async()=>{evaluations++;throw Error('Retained interruption');}};
 await assert.rejects(runCampaignFailureStep(f.config,f.values,ops),/Retained interruption/);
 if(boundary==='review profile'){
  f.config.skill_writer_actor={model:'review-b'};
  await writeFile(join(f.root,'instance.json'),JSON.stringify(f.config));
 }else f.memory.knowledge.attentionRevision=2;
 await assert.rejects(runCampaignFailureStep(f.config,f.values,ops),/Frozen experiment baseline, model, machinery or memory changed/);
 assert.equal(evaluations,1);
});

test('eligible family remains visible behind more than one page of legacy work',async t=>{
 const f=await fixture(t);
 for(let i=0;i<105;i++)f.backlog.upsert({key:'old-'+i,priority:100+i,reason:'Legacy proposal',objective:'correctness',skill:'old-'+i,base:'base',groupIds:f.item.groupIds,proposal:{action:'update'},evidence:f.item.evidence});
 assert.equal((await failureCampaignStatus(f.config)).hasWork,true);
 const result=await runCampaignFailureStep(f.config,f.values,{...f.ops,analyze:async()=>assert.fail('Eligible family is already queued')});
 assert.equal(result.status,'accepted');
});

test('expired retained experiment never starts a new publication',async t=>{
 const f=await fixture(t),run=f.backlog.claim({base:'base',seconds:7200,workflow:'failure-first-v1'}).run;
 f.backlog.checkpoint(run.id,{elapsed:7200,checkpoint:{workflow:'failure-first-v1',startedAt:Date.now()-7201000,directory:f.root,slots,phase:'publish'}});
 await assert.rejects(runCampaignFailureStep(f.config,f.values,{...f.ops,publish:async()=>assert.fail('Expired work cannot promote')}),/allowance exhausted before publication/);
});

test('A5 uses the same bounded failure-first worker through the normal CLI',async t=>{
 const f=await fixture(t);
 f.config.target='a5';
 await writeFile(join(f.root,'instance.json'),JSON.stringify(f.config));
 let evaluated=false;
 const result=await main(['skills-ab','--instance',f.root,'--once'],{failureDependencies:{
  settled:async()=>true,slots:async()=>slots,memory:async()=>f.memory,
  step:async(config,input)=>{assert.equal(config.target,'a5');assert.equal(input.slots.length,4);evaluated=true;return {status:'rejected',reason:'baseline-ceiling'};}}});
 assert.equal(result.status,'rejected');
 assert.equal(evaluated,true);
 assert.throws(()=>failureProtocol({...f.config,target:'cpu'}),/A3 or A5/);
});

test('A5 routes a settled focused development winner to searchable admission after failed transfer',async t=>{
 const f=await fixture(t);
 f.config.target='a5';
 await writeFile(join(f.root,'instance.json'),JSON.stringify(f.config));
 const calibrated=task=>({...task,controls:{calibration:{files:{'kernel.py':'calibration'}}}});
 const prepared={...family,development:calibrated(family.development),transfer:family.transfer.map(calibrated)};
 const winner={id:'approved'};
 let admissions=0;
 const ops={...f.ops,
  family:async()=>({status:'ready',family:prepared}),
  pilot:async({manifest})=>{
   assert.equal(manifest.mode,'focused-correctness');
   return {status:'complete',promising:false,winner,reason:'Transfer showed no gain'};
  },
  admit:async({result,directory})=>{
   assert.equal(result.winner,winner);
   assert(directory.endsWith('/pilot'));
   assert.equal(f.backlog.run(f.backlog.meta('failure-first-current')).checkpoint.phase,'admit-searchable');
   admissions++;
   return {admitted:true,id:'experimental'};
  },
  publish:async()=>assert.fail('Failed transfer must not activate guidance'),
 };
 const outcome=await runCampaignFailureStep(f.config,f.values,ops);
 assert.equal(admissions,1);
 assert.equal(outcome.searchable,true);
 assert.equal(outcome.admission.id,'experimental');
 assert.equal((await failureCampaignStatus(f.config)).current,null);
 assert.equal(JSON.parse(await readFile(join(f.root,'instance.json'),'utf8')).skills.activeVersion,'base');
});

async function retainedSearchableAdmission(t,elapsed=1){
 const f=await fixture(t);
 f.config.target='a5';
 await writeFile(join(f.root,'instance.json'),JSON.stringify(f.config));
 const claimed=f.backlog.claim({base:'base',itemId:f.item.id,seconds:7200,workflow:'failure-first-v1'});
 const current=claimed.run.id;
 f.backlog.meta('failure-first-current',current);
 const directory=join(f.root,'recovered');
 const checkpoint={workflow:'failure-first-v1',phase:'admit-searchable',directory,slots,startedAt:Date.now()-elapsed*1000};
 f.backlog.checkpoint(current,{elapsed,checkpoint});
 await mkdir(join(directory,'pilot'),{recursive:true});
 const saved={status:'complete',promising:false,winner:{id:'reviewed'}};
 await writeFile(join(directory,'pilot','pilot.json'),JSON.stringify(saved));
 const ops={...f.ops,pilot:async()=>assert.fail('Evaluation must not rerun'),
  family:async()=>assert.fail('Isolation must not rerun')};
 return {f,ops,saved};
}

test('searchable admission recovery reconciles an existing receipt after expiry without relaunching actors',async t=>{
 const {f,ops,saved}=await retainedSearchableAdmission(t,7201);
 const outcome=await runCampaignFailureStep(f.config,f.values,{...ops,admit:async input=>{
  assert.equal(input.recoverOnly,true);
  assert.deepEqual(input.result,saved);
  return {admitted:true,id:'resumed'};
 }});
 assert.equal(outcome.status,'inconclusive');
 assert.equal(outcome.admission.id,'resumed');
 assert.equal(f.backlog.meta('failure-first-current'),null);
});

test('missing searchable receipt after expiry closes the run without a new admission',async t=>{
 for(const [elapsed,remainingSeconds] of [[7201,600],[1,0]]){
  const {f,ops}=await retainedSearchableAdmission(t,elapsed);
  const attempts=[];
  const outcome=await runCampaignFailureStep(f.config,{...f.values,remainingSeconds},{...ops,admit:async ({recoverOnly})=>{
   attempts.push(recoverOnly);
   return {admitted:false,reason:'no-prior-admission'};
  }});
  assert.deepEqual(attempts,[true]);
  assert.equal(outcome.status,'inconclusive');
  assert.equal(f.backlog.meta('failure-first-current'),null);
  assert.equal(f.backlog.meta('failure-first-status').status,'inconclusive');
 }
});

test('searchable admission recovery starts a fresh write within the remaining allocation',async t=>{
 const {f,ops,saved}=await retainedSearchableAdmission(t);
 const outer=new AbortController();
 const attempts=[];
 const outcome=await runCampaignFailureStep(f.config,{...f.values,remainingSeconds:60,signal:outer.signal},
  {...ops,admit:async ({recoverOnly=false,signal,result})=>{
   attempts.push(recoverOnly);
   assert.deepEqual(result,saved);
   if(recoverOnly){
    assert.equal(signal,outer.signal);
    return {admitted:false,reason:'no-prior-admission'};
   }
   assert.notEqual(signal,outer.signal);
   signal.throwIfAborted();
   return {admitted:true,id:'written'};
  }});
 assert.deepEqual(attempts,[true,false]);
 assert.equal(outcome.status,'accepted');
 assert.equal(outcome.admission.id,'written');
});

for(const kind of ['no-submit','rejected-submissions'])test('A5 sets aside a closed '+kind+' family without scoring it',async t=>{
 const f=await fixture(t);
 f.config.target='a5';
 await writeFile(join(f.root,'instance.json'),JSON.stringify(f.config));
 const outcome=await runCampaignFailureStep(f.config,f.values,{
  ...f.ops,
  family:async()=>{throw Object.assign(Error('Review session '+kind),{reviewSession:{kind}});},
  pilot:async()=>assert.fail('Unsubmitted family cannot run a pilot'),
  admit:async()=>assert.fail('Unsubmitted family cannot admit a skill'),
 });
 assert.equal(outcome.status,'inconclusive');
 assert.equal(outcome.reason,'family-unsubmitted');
 assert.equal((await failureCampaignStatus(f.config)).current,null);
});

test('continuous worker releases resources after the last classification page blocks',async t=>{
 const f=await fixture(t);
 f.backlog.revise(f.item.id,{status:'deferred'});
 f.backlog.reclassify(f.item.groupIds[0]);
 const {analyzeBacklog}=await import('../../cli/skill-backlog-analysis.mjs');
 const version=join(f.root,'skills','versions','base');
 await mkdir(version,{recursive:true});await writeFile(join(version,'guidance.json'),'[]');
 let closed=false;
 const result=await runFailureCampaign(f.config,{},{
  lock:(_c,fn)=>fn(),slots:async()=>slots,settled:async()=>true,
  read:async()=>f.config,memory:async()=>({...f.memory,close:async()=>{closed=true;}}),
  update:()=>assert.fail('A blocked page must not disable tuning'),
  wait:()=>assert.fail('No classifiable page remains'),
  step:(config,args)=>runCampaignFailureStep(config,args,{analyze:input=>analyzeBacklog(input,{
   classify:async()=>{throw Object.assign(Error('No submission'),{reviewSession:{kind:'no-submit'}});}
  })})
 });
 assert.equal(result.classificationPage,'blocked');
 assert.equal(closed,true);
 assert.deepEqual(JSON.parse(f.state.db.prepare("SELECT value FROM control WHERE key='skill-background-resources'").get().value),[]);
});

test('scoped campaign repairs immutable writing and test independently before matched baseline-first trials',async t=>{
 const {prepareScopedWriting}=await import('../../cli/skill-failure-candidates.mjs');
 const {reviewScopedArtifact,reviewScopedTest}=await import('../../cli/skill-scoped-review.mjs');
 const {skillInput}=await import('../../cli/skill-microeval.mjs');
 const {stages}=await import('../../controller/workflow.mjs');
 for(const [target,baseline] of [['a5','pass'],['a5','unrelated'],['a5','reproduced'],['a3','reproduced']]){
  const f=await fixture(t,target),events=[],approved='Check the shape assumptions against all supported inputs.';
  delete f.config.skills.failureFirst.experimentPolicy;await writeFile(join(f.root,'instance.json'),JSON.stringify(f.config));
  const base=join(f.root,'skills','versions','base'),scopeNames=['global-kdb',...stages];
  await mkdir(base,{recursive:true});await writeFile(join(base,'skill-map.json'),JSON.stringify(Object.fromEntries(scopeNames.map(s=>[s,s+'.md']))));
  await writeFile(join(base,'guidance.json'),JSON.stringify(scopeNames.map(scope=>({id:scope,scope,text:'Base '+scope,critical:true,negative:false}))));
  for(const scope of scopeNames)await writeFile(join(base,scope+'.md'),'Base '+scope);
  const checker=join(f.root,'checker.py');await writeFile(checker,'pass');
  const groups=f.item.evidence.groups;
  const ops={...f.ops,input:skillInput,
   writing:args=>prepareScopedWriting(args,{session:async request=>{
    const text=request.prompt.feedback?approved:'Always change the shape.';events.push(['writer',text]);
    if(request.prompt.feedback)assert.equal(request.prompt.feedback.quote,'Always');
    const retrievals=[{entries:[{collection:'memory',chunk_id:'known',text:'A preserved shape. Shape changes can invalidate strides.'}]}];
    return {value:request.validate({text,scope:'implementation',memoryRefs:['memory:known'],claims:[{path:'text',proofs:[{reference:groups[0].occurrences[0].reference}]}]},retrievals),retrievals};
   },review:args=>reviewScopedArtifact(args,async request=>{
    const {fields,evidence}=request.prompt;
    assert.match(await readFile(evidence.find(e=>e.name==='memory:known').path,'utf8'),/invalidate strides/);
    events.push(['writing-review',fields.text]);
    return {value:request.validate(fields.text===approved?{verdict:'approved'}:{verdict:'revise',field:'text',quote:'Always',defect:'The recommendation is unconditional.'})};
   })}),
   family:async args=>{
    assert.equal(args.approvedInstruction.text,approved);let submitted;
    for(const prompt of ['Return zero','Repair the implementation']){
     events.push(['test-author',prompt]);const tasks=['development','stride','layout'].map((id,i)=>({...task(id),controlCaseIds:prompt==='Return zero'?['large']:['control'],files:{'kernel.py':'source '+i},prompt,checker,publicChecker:checker,changeType:['','layout','boundary'][i],relation:'Transfer',controls:{negatives:[{files:{}}],calibration:{files:{}},reference:{files:{}}}}));
     submitted={id:'family',identity:prompt,mechanism:{phase:'check',type:'shape'},development:tasks[0],transfer:tasks.slice(1)};
     try{await args.checkReferences({family:submitted,seconds:60,signal:args.signal});break;}
     catch(error){assert.equal(error.code,'FAILURE_REFERENCE_REJECTED');assert.match(error.message,/ignores the input/);}
    }
    return {status:'ready',family:submitted};
   },
   testReview:args=>reviewScopedTest(args,request=>reviewScopedArtifact(request,async model=>{
    const {fields}=model.prompt;assert.equal(fields['development.checker'],'pass');assert.equal(fields['stride.changeType'],'layout');assert(fields['development.controls']);assert.deepEqual(model.evidenceGroups,groups);events.push(['test-review',fields['development.prompt']]);
    return {value:model.validate(JSON.parse(fields['development.controlCaseIds']).includes('large')?{verdict:'revise',field:'development.controlCaseIds',quote:'large',defect:'The wrong control designation ignores the input.'}:{verdict:'approved'})};
   })),
   runtime:async()=>({...await f.ops.runtime(),checkReferences:async()=>({diagnostics:[]}),qualify:async()=>({qualified:true,referencePassed:true,negativeFailed:true,calibrationPassed:true}),
    trial:async input=>{events.push(['trial',input.arm]);if(input.arm!=='baseline')assert(input.guidance.includes(approved));
     const value=result(input.arm!=='baseline'||baseline==='pass');value.mechanismMatch=!value.correct&&baseline==='reproduced';return value;}}),
   publish:async args=>(await import('../../cli/skill-failure-candidates.mjs')).publishFailureWinner(args,{publish:async()=>true}),
   design:async()=>assert.fail('Legacy writing after screening'),};
  const outcome=await runCampaignFailureStep(f.config,f.values,ops);
  assert.deepEqual(events.slice(0,8).map(e=>e[0]),['writer','writing-review','writer','writing-review','test-author','test-review','test-author','test-review']);
  assert.deepEqual(events.slice(8,13).map(e=>e[1]),Array(5).fill('baseline'));
  assert.equal(outcome.status,baseline==='reproduced'?'accepted':'rejected',outcome.reason);
  assert.equal(events.filter(([event,lane])=>event==='trial'&&lane!=='baseline').length,baseline==='reproduced'?15:0);
 }
});

test('a closed writer provider timeout resumes its request within the original grant',async()=>{
 const {scopedSession}=await import('../../cli/skill-scoped-review.mjs');
 const grants=[];
 const value=await scopedSession(async args=>{
  grants.push(args.seconds);
  if(grants.length===1)throw Object.assign(Error('Request timed out'),{reviewSession:{kind:'provider-error',retryableTimeout:true}});
  return 'submitted';
 },{seconds:4});
 assert.equal(value,'submitted');assert.equal(grants.length,2);assert(grants[1]<grants[0]-.8);
});
test('a normally closed writer without submission releases its family and leaves the next eligible',async t=>{
 const f=await fixture(t);
 f.config.skills.failureFirst.experimentPolicy='scoped-v1';
 const error=Object.assign(Error('Finished without submission'),{reviewSession:{kind:'no-submit'}});
 const result=await runCampaignFailureStep(f.config,f.values,{...f.ops,writing:async()=>{throw error;},runtime:async()=>assert.fail('Closed writer requested evaluation')});
 assert.equal(result.status,'inconclusive');assert.equal(result.reason,'writing-no-submit');
 assert.equal(f.backlog.run(f.item.id),null);
});

test('an old running checkpoint retains its screened-before-writing protocol when fresh experiments default to scoped QA',async t=>{
 const f=await fixture(t);delete f.config.skills.failureFirst.experimentPolicy;
 const {run}=f.backlog.claim({base:'base',itemId:f.item.id,seconds:7200,workflow:'failure-first-v1'});
 f.backlog.checkpoint(run.id,{elapsed:0,checkpoint:{workflow:'failure-first-v1',startedAt:Date.now(),directory:join(f.root,'old-run'),slots}});
 const result=await runCampaignFailureStep(f.config,f.values,{...f.ops,writing:async()=>assert.fail('Changed retained writing contract')});
 assert.equal(result.status,'accepted');assert.equal(f.calls.filter(x=>x[0]==='design').length,1);
 assert.equal(failureProtocol(f.config).candidates,1);
});

test('a changed reviewed artifact gets a new native submission and unchanged inputs reuse it',async t=>{
 const {reviewScopedArtifact}=await import('../../cli/skill-scoped-review.mjs');
 const f=await fixture(t),answers=new Map();
 const session=async input=>{
  if(!answers.has(input.directory))answers.set(input.directory,input.validate(input.prompt.fields.text==='Known limitation'
   ?{verdict:'approved'}:{verdict:'revise',field:'text',quote:'invented',defect:'Unsupported by the evidence.'}));
  return {value:answers.get(input.directory)};
 };
 const args={directory:join(f.root,'shared-review'),scope:'writing',brief:{issue:'fixture'},seconds:5};
 for(const [text,verdict] of [['Known limitation','approved'],['An invented limit','revise'],['Known limitation','approved']])
  assert.equal((await reviewScopedArtifact({...args,fields:{text}},session)).verdict,verdict);
 assert.equal(answers.size,2);
});
