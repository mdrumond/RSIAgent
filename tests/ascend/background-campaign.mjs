import test from 'node:test';
import assert from 'node:assert/strict';
import {writeFile,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {setup,rows,until} from './driver_setup.mjs';

// Only the A/B workload is controllable. Foreground Pi, Python verification,
// source Git, storage publication, controller and process lifecycle are real.
test('continuous A/B overlaps the next iteration and new originals select promoted skills',{timeout:150000},async t=>{
  let releaseSlow,slowEntered;
  const slowStarted=new Promise(resolve=>{slowEntered=resolve;});
  const slowGate=new Promise(resolve=>{releaseSlow=resolve;});
  t.after(()=>releaseSlow());
  const f=await setup(t,'phase2',['warm','seed','a-free','a-slow','z-later','tail'],{providerOptions:{resolveCall:async(call,body)=>{
    const content=body.messages.find(message=>message.role==='user')?.content;
    const prompt=typeof content==='string'?content:(content??[]).map(block=>block.text??'').join('');
    if(typeof prompt==='string'&&prompt.startsWith('Goal: a-slow')&&call?.name==='rsi_skill') {
      slowEntered();await slowGate;
    }
    return call;
  }}});
  const environment=join(f.root,'slots.mjs'),runtime=join(f.root,'ab-runtime.mjs'),release=join(f.root,'release-ab');
  await writeFile(environment,`import {createEnvironment as local} from './machinery/integrations/environments/index.mjs';
export const createEnvironment=(options,context)=>local(options,{...context,resources:async()=>[0,1].map(i=>({id:'local'+i,available:true,metadata:{}}))});\n`);
  await writeFile(runtime,`import {mkdir,readFile,writeFile,appendFile,access} from 'node:fs/promises';
import {dirname,join} from 'node:path';import {fileURLToPath} from 'node:url';
const root=dirname(fileURLToPath(import.meta.url));
export async function bootstrap(request){await mkdir(request.instanceRoot,{recursive:true});await writeFile(join(request.instanceRoot,'request.json'),JSON.stringify({config:request.config}));}
export async function wave(request){await appendFile(join(root,'ab-samples.jsonl'),JSON.stringify({root:request.instanceRoot,version:request.config.skills.activeVersion})+'\\n');
 while(true){request.signal?.throwIfAborted();try{await access(join(root,'release-ab'));break;}catch{}await new Promise(resolve=>setTimeout(resolve,20));}
 return {correct:true,score:request.config.skills.activeVersion==='baseline'?1:2};}
export async function resume(request){return wave({...request,...JSON.parse(await readFile(join(request.instanceRoot,'request.json'),'utf8'))});}
export async function settled(){return true;}\n`);
  const initial=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8'));
  const plan=[{id:'r1-p1',phase:'phase1',new_projects:1},{id:'r1-p2',phase:'phase2',new_projects:1},
    {id:'r2-p1',phase:'phase1',new_projects:1},{id:'r2-p2',phase:'phase2',new_projects:1},
    {id:'r3-p1',phase:'phase1',new_projects:1},{id:'r3-p2',phase:'phase2',new_projects:1}];
  const entries=plan;
  const config=join(f.root,'background.json');await writeFile(config,JSON.stringify({concurrency:2,recovery:false,campaign_seconds:30*24*60*60,pause_grace_seconds:0,
    campaign_plan:plan,environment:{module:environment,options:{mode:'local'}},
    projects:initial.projects.map((project,i)=>({...project,phase:entries[i].phase,campaign_entry:entries[i].id})),
    tune_every:1,skills:{...initial.skills,background:true,runsPerArm:2,tuningProjectIds:['seed','tail'],runtimeModule:runtime}}));
  await f.run('config','--config',config);
  let campaign=f.start('campaign');
  await Promise.race([slowStarted,campaign.output.then(()=>assert.fail('campaign ended before overlap'))]);
  await until(async()=>{try{return (await readFile(join(f.root,'ab-samples.jsonl'),'utf8')).trim().length>0;}catch{return false;}});
  assert.equal(rows(f.root,"SELECT status FROM projects WHERE id='a-free'")[0].status,'passed');
  const slow=rows(f.root,"SELECT id FROM attempts WHERE project='a-slow'")[0].id;
  assert.equal(JSON.parse(await readFile(join(f.root,'attempts',slow,'job.json'),'utf8')).skillVersion,'baseline');
  assert.equal(rows(f.root,'SELECT * FROM resource_leases').length,1);
  assert.ok(JSON.parse(rows(f.root,"SELECT value FROM control WHERE key='skill-background-resource'")[0].value));
  const before=JSON.parse(rows(f.root,"SELECT value FROM control WHERE key='campaign-budget'")[0].value);
  await f.run('pause');await campaign.output;
  const evolution=JSON.parse(await readFile(join(f.root,'skills','evolution.json'),'utf8'));
  const checkpoint=JSON.parse(await readFile(join(evolution.pending.evaluationRoot,'round.json'),'utf8'));
  assert.equal(checkpoint.status,'interrupted');assert.equal(checkpoint.runningAt,undefined);
  const retainedSession=rows(f.root,`SELECT session FROM attempts WHERE id='${slow}'`)[0].session;
  assert.ok(retainedSession);
  campaign=f.start('resume');
  await until(()=>rows(f.root,"SELECT value FROM control WHERE key='owner'").some(row=>JSON.parse(row.value)));
  assert.equal(rows(f.root,"SELECT id FROM attempts WHERE project='a-slow'")[0].id,slow);
  assert.equal(JSON.parse(rows(f.root,"SELECT value FROM control WHERE key='campaign-budget'")[0].value).cycle,before.cycle);
  await writeFile(release,'release');
  await until(async()=>JSON.parse(await readFile(join(f.root,'instance.json'),'utf8')).skills.activeVersion!=='baseline');
  const promoted=JSON.parse(await readFile(join(f.root,'instance.json'),'utf8')).skills.activeVersion;
  assert.equal(JSON.parse(await readFile(join(f.root,'attempts',slow,'job.json'),'utf8')).skillVersion,'baseline');
  releaseSlow();await campaign.output;
  for(const project of ['z-later','tail']) {
    const id=rows(f.root,`SELECT id FROM attempts WHERE project='${project}'`)[0].id;
    assert.equal(JSON.parse(await readFile(join(f.root,'attempts',id,'job.json'),'utf8')).skillVersion,promoted);
  }
  assert.equal(rows(f.root,`SELECT session FROM attempts WHERE id='${slow}'`)[0].session,retainedSession);
  assert.equal(rows(f.root,'SELECT * FROM attempts').length,6);
  assert.equal(rows(f.root,"SELECT count(*) n FROM events WHERE type='skill-evidence-queued'")[0].n,3);
  assert.equal(rows(f.root,"SELECT value FROM control WHERE key='skill-background-resource'")[0].value,'null');
  f.pass();
});
