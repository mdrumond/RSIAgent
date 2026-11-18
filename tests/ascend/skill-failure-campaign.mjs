import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {State} from '../../controller/state.mjs';
import {SkillBacklog} from '../../controller/skill-backlog.mjs';
import {main} from '../../cli/main.mjs';
import {runCampaignFailureStep,failureCampaignStatus,failureCampaignSettled} from '../../cli/skill-failure-campaign.mjs';
import {failureProtocol,selectFailureSlots} from '../../cli/skill-failure-worker.mjs';

const slots=Array.from({length:4},(_,i)=>({resource:{id:'slot-'+i,available:true,metadata:{kind:'host-slot',pool:'cpu-fixture',hostSlot:i}},physical:i}));
const task=id=>({id,caseIds:['large','control'],controlCaseIds:['control'],files:{'kernel.py':'starter'},prompt:'Repair the isolated behavior'});
const family={id:'actual-failure',development:task('development'),transfer:[task('layout'),task('aggregate')],evidence:[]};
const result=correct=>({correct,mechanismMatch:!correct,caseResults:['large','control'].map(id=>({id,correct})),sourceSha256:'a'.repeat(64),artifacts:[],handles:[]});
async function fixture(t){
 const root=await mkdtemp(join(tmpdir(),'failure-campaign-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const config={format:1,root,target:'a3',mode:'phase1',projects:[],concurrency:5,actor:{},environment:{options:{target:'fixture'}},skills:{activeVersion:'base',background:true,tuningDisabled:false,failureFirst:{enabled:true,sourceRevision:'a'.repeat(40)}}};
 await writeFile(join(root,'instance.json'),JSON.stringify(config));const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());state.db.exec('CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
 for(const [key,value]of Object.entries({pause:true,'campaign-budget':{seconds:10000,elapsed:125,cycle:'retained',boot:'prior-boot',monotonic:0,running:false}}))state.db.prepare('insert into control values(?,?)').run(key,JSON.stringify(value));
 state.event(null,'generation-rejected',{message:'Actual test fixture rejection'});const backlog=new SkillBacklog(state,{target:'a3'});backlog.ingest();const g=backlog.failures()[0];backlog.acknowledge(g.id);
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
test('scientific QA rejection finishes without development trials',async t=>{
 const f=await fixture(t);const outcome=await runCampaignFailureStep(f.config,f.values,{...f.ops,design:async()=>async()=>({status:'rejected',reason:'Unsupported instruction'})});assert.equal(outcome.status,'rejected');assert.equal(f.calls.filter(c=>c[0]==='trial').length,5);
});
test('first infrastructure failure cancels the admitted cells and never starts a fifth',async t=>{
 const f=await fixture(t);let started=0,cancelled=0;
 await assert.rejects(runCampaignFailureStep(f.config,f.values,{...f.ops,runtime:async()=>({...await f.ops.runtime(),trial:async input=>{const n=started++;if(!n)return {infrastructure:true,correct:null,reason:'Memory connection lost'};if(input.signal.aborted)cancelled++;else await new Promise(resolve=>input.signal.addEventListener('abort',()=>{cancelled++;resolve();},{once:true}));return {infrastructure:true,correct:null,reason:'Cancelled sibling'};}})}),/Memory connection lost/);
 assert(started<=4);assert.equal(cancelled,started-1);assert(!f.calls.some(c=>c[0]==='publish'));
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
