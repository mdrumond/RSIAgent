import test from 'node:test';
import assert from 'node:assert/strict';
import {writeFile,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {setup,rows,until} from './driver_setup.mjs';

// Real controller/worker failure, Pi finalization, storage, KDB and phase cursor;
// only detached native execution observation is a deterministic fixture.
for(const proof of ['late','unknown'])test(`live campaign reconciles ${proof} native closure without actor replay`,{timeout:90000},async t=>{
  const f=await setup(t,'phase2',['holder','next']);
  const marker=join(f.root,'native-closed'),module=join(f.root,'late-proof.mjs');
  await writeFile(module,`import {readFile,access,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {join} from 'node:path';
import {createEnvironment as local} from './machinery/integrations/environments/index.mjs';
export async function createEnvironment(options,context){
 const base=await local(options,context);
 if(process.argv[2]==='worker'){
  const job=JSON.parse(await readFile(join(context.instancePath,'job.json'),'utf8'));
  if(job.attempt.project==='holder'&&job.kind==='original'&&!job.finalize){
   await context.event('execution-starting',{operation:'detached',startElapsed:0});
   await context.event('execution-submitting',{operation:'detached',submission:'native',kind:'run',accountingExcluded:false});
   await context.event('execution-submitted',{submission:'native',handle:'fixture:exact-native'});
   await context.event('execution-handle',{operation:'detached',handle:'fixture:exact-native'});
   await context.event('execution-unresolved',{operation:'detached',handle:'fixture:exact-native'});
   const evidence=join(context.instancePath,'native-failure.log'),content='Worker exited with detached native execution unresolved.\\n';
   await writeFile(evidence,content);
   await writeFile(join(context.instancePath,'findings.json'),JSON.stringify({findings:content,useful:false,evidence:[{path:evidence,sha256:createHash('sha256').update(content).digest('hex')}]}));
   process.exit(1);
  }
 }
 const closed=async()=>{try{await access(${JSON.stringify(marker)});return true;}catch{return false;}};
 return {...base,observe:async handle=>handle==='fixture:exact-native'?{state:await closed()?'stopped':'unknown'}:base.observe(handle),
 reconcileOperation:async({operation})=>operation==='detached'?{closed:await closed(),executionSeconds:2.5,handles:['fixture:exact-native']}:{closed:false}};
}
`);
  const initial=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));
  const settings=join(f.root,'live.json');
  await writeFile(settings,JSON.stringify({concurrency:1,recovery:false,skills:{...initial.skills,tuningDisabled:true},
    environment:{module,options:{mode:'local'}},campaign_plan:[{id:'p1',phase:'phase1',new_projects:1},{id:'p2',phase:'phase2',new_projects:1}],
    projects:initial.projects.map((p,i)=>({...p,phase:i?'phase2':'phase1',campaign_entry:i?'p2':'p1'}))}));
  await f.run('config','--config',settings);
  const campaign=f.start('campaign');void campaign.output.catch(()=>{});
  let drained=false;t.after(async()=>{if(!drained){await f.run('stop');await campaign.output;}});
  await until(()=>rows(f.root,"SELECT * FROM events WHERE type='failure-settlement' AND json_extract(data,'$.closed')=0").length);
  const original=rows(f.root,"SELECT * FROM attempts WHERE project='holder'")[0];
  assert.equal(original.outcome,'failed');
  assert.equal(rows(f.root,'SELECT * FROM resource_leases').length,1);
  assert.equal(rows(f.root,"SELECT * FROM attempts WHERE project='next'").length,0);
  if(proof==='late'){
    await writeFile(marker,'exact native proof');
    await until(()=>rows(f.root,"SELECT * FROM events WHERE type='postprocess-completed'").length===2,60000);
    await campaign.output;
    assert.equal(rows(f.root,'SELECT * FROM resource_leases').length,0);
    assert.equal(rows(f.root,"SELECT * FROM events WHERE type='job-admitted' AND json_extract(data,'$.kind')='original'").length,2);
    assert.equal(rows(f.root,`SELECT elapsed FROM attempts WHERE id='${original.id}'`)[0].elapsed,2.5);
    assert.equal(rows(f.root,"SELECT status FROM projects WHERE id='next'")[0].status,'passed');
  }else{
    // Wait past multiple periodic checks; unknown identity/closure never releases.
    await until(()=>rows(f.root,"SELECT * FROM events WHERE type='failure-settlement' AND json_extract(data,'$.closed')=0").length>=4,25000);
    assert.equal(rows(f.root,'SELECT * FROM resource_leases').length,1);
    assert.equal(rows(f.root,'SELECT * FROM attempts').length,1);
    assert.equal(rows(f.root,"SELECT * FROM events WHERE type='postprocess-completed'").length,0);
  }
  await f.run('stop');await campaign.output;drained=true;
  f.pass();
});

test('campaign postprocesses an investigation without blocking on its own unfinished outcome',{timeout:90000},async t=>{
  const {mkdir}=await import('node:fs/promises');
  const {State}=await import('../../controller/state.mjs');
  const {RecoveryQueue}=await import('../../controller/recovery.mjs');
  const {processIdentity}=await import('../../controller/lifecycle.mjs');
  const f=await setup(t,'recover',['holder','next']);
  // Obtain a real failed original and recorded recovery lesson first.
  const settings=join(f.root,'investigation.json');
  const before=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));
  await writeFile(settings,JSON.stringify({curriculum:{phase1:['holder']},
    campaign_plan:[{id:'first',phase:'phase1',new_projects:1},{id:'later',phase:'phase2',new_projects:1}],
    projects:before.projects.map(p=>({...p,campaign_entry:p.id==='holder'?'first':'later',phase:p.id==='holder'?'phase1':'phase2'}))}));
  await f.run('config','--config',settings);await f.run('wave');
  const original=rows(f.root,"SELECT * FROM attempts WHERE project='holder'")[0];
  const state=new State(join(f.root,'state.sqlite'));
  let investigation;
  try{
    const queue=new RecoveryQueue(state),owner={...await processIdentity(),start:'stopped-fixture-owner'};
    investigation=queue.claim('holder',2,owner,'local');
    const path=join(f.root,'attempts',investigation.attempt.id);await mkdir(path,{recursive:true});
    await writeFile(join(path,'job.json'),JSON.stringify({...investigation,mode:'phase1',wave:2,options:{},resource:{id:'local'},skillVersion:'baseline'}));
    await writeFile(join(path,'findings.json'),await readFile(join(f.root,'attempts',original.id,'findings.json')));
    state.finish(investigation.attempt.id,'failed',{settleProject:false});
  }finally{state.close();}
  const initial=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));
  await writeFile(settings,JSON.stringify({recovery:false,skills:{...initial.skills,tuningDisabled:true}}));
  await f.run('config','--config',settings);
  const campaign=f.start('campaign');void campaign.output.catch(()=>{});
  let drained=false;t.after(async()=>{if(!drained){await f.run('stop');await campaign.output;}});
  await until(()=>rows(f.root,`SELECT * FROM events WHERE attempt='${investigation.attempt.id}' AND type='postprocess-completed'`).length,30000);
  assert.ok(rows(f.root,`SELECT * FROM recovery_lessons WHERE attempt='${investigation.attempt.id}'`).length);
  assert.equal(rows(f.root,`SELECT * FROM attempts WHERE id='${investigation.attempt.id}'`)[0].outcome,'failed');
  assert.equal(rows(f.root,`SELECT * FROM events WHERE attempt='${investigation.attempt.id}' AND type='task-constructed'`).length,0);
  await f.run('stop');await campaign.output;drained=true;
  f.pass();
});
