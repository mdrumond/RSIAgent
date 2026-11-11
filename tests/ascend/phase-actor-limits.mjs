import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {State} from '../../controller/state.mjs';
import {phaseActorSeconds} from '../../controller/budget.mjs';
import {runAttempt} from '../../controller/pi.mjs';
import {campaignEntryConfig} from '../../cli/campaign-manifest.mjs';
import {updateConfig,readConfig} from '../../cli/instance.mjs';
import {operate} from '../../cli/runtime.mjs';
import {main} from '../../cli/main.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';

const caps={phase1:3600,phase2:14400};
async function temporary(t) {
  const root=await mkdtemp(join(tmpdir(),'phase-actor-limits-'));
  t.after(()=>rm(root,{recursive:true,force:true}));return root;
}

test('phase caps reach generation without enlarging smaller allocations or affecting legacy instances',()=>{
  const config={actor_seconds:7200,actor_seconds_by_phase:caps,campaign_plan:[
    {id:'p1',phase:'phase1',new_projects:8,actor_seconds:7200},
    {id:'p2',phase:'phase2',new_projects:1,actor_seconds:28800}]};
  assert.equal(campaignEntryConfig(config,'p1').actor_seconds,3600);
  assert.equal(campaignEntryConfig(config,'p2').actor_seconds,14400);
  assert.equal(phaseActorSeconds(config,'phase1',90),90);
  assert.equal(phaseActorSeconds(config,'inference',28800),28800);
  assert.equal(phaseActorSeconds({},'phase1',7200),7200);
  assert.equal(config.campaign_plan[0].actor_seconds,7200);
});

test('saving phase caps preserves historical project, checkpoints, outcomes, and retained jobs',async t=>{
  const root=await temporary(t),state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  const projects=['active','done'].map(id=>({id,goal:id,phase:'phase1',actor_seconds:7200}));
  for(const project of projects)state.add(project);
  const active=state.admit('active',{}),done=state.admit('done',{});
  state.checkpoint(active.id,3700,'retained-session.jsonl');state.finish(done.id,'passed');
  const job=join(root,'job.json');await writeFile(job,JSON.stringify({actor_seconds:7200}));
  await writeFile(join(root,'instance.json'),JSON.stringify({format:1,mode:'phase1',projects}));
  const before=state.db.prepare('SELECT * FROM attempts ORDER BY id').all();
  await updateConfig(root,current=>({...current,actor_seconds_by_phase:caps}));
  assert.deepEqual(state.db.prepare('SELECT * FROM attempts ORDER BY id').all(),before);
  assert.deepEqual(state.project('active').spec,projects[0]);
  const status=await main(['status','--instance',root]);
  assert.equal(status.projects.find(project=>project.id==='active').actor_seconds,7200);
  assert.equal(status.projects.find(project=>project.id==='active').effective_actor_seconds,3600);
  assert.equal(JSON.parse(await readFile(job,'utf8')).actor_seconds,7200);
  for(const invalid of [{phase1:0},{phase2:Infinity},{bad:3600},[],null]) {
    await assert.rejects(updateConfig(root,current=>({...current,actor_seconds_by_phase:invalid})),/positive elapsed/);
    assert.deepEqual((await readConfig(root)).actor_seconds_by_phase,caps);
  }
});

for(const phase of ['phase1','phase2'])test(`${phase} resume above its cap settles without running setup or losing source/session`,async t=>{
  const root=await temporary(t),source=join(root,'source');await mkdir(source);
  await writeFile(join(source,'kernel.py'),'answer=42\n');
  const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  const spec={id:'p',goal:'retained work',phase,source,actor_seconds:28800,verifier:{files:[],command:'true'}};
  state.add(spec);const attempt=state.admit('p',{}),elapsed=caps[phase]+10;
  state.checkpoint(attempt.id,elapsed,'retained-session.jsonl');
  await runAttempt({state,attempt:state.attempt(attempt.id),
    project:{...spec,actor_seconds:phaseActorSeconds({actor_seconds_by_phase:caps},phase,spec.actor_seconds)},
    attemptPath:join(root,'attempts',attempt.id),environmentFactory:async()=>({
      prepare:async()=>assert.fail('An exhausted attempt must not start setup'),
    })});
  const after=state.attempt(attempt.id);
  assert.equal(after.outcome,'failed');assert.equal(after.session,'retained-session.jsonl');
  assert.ok(after.elapsed>=elapsed);assert.equal(await readFile(join(source,'kernel.py'),'utf8'),'answer=42\n');
  assert.equal(state.project('p').spec.actor_seconds,28800);
  assert.equal(state.events(attempt.id).find(event=>event.type==='rejected').data.reason,'ACTOR_ELAPSED_EXHAUSTED');
});

for(const kind of ['original','resume','investigation','retry'])test(`real scheduler caps ${kind} jobs and keeps consumed elapsed`,async t=>{
  const resume=kind==='resume';
  const root=await temporary(t);await mkdir(join(root,'skills','versions','v'),{recursive:true});
  await writeFile(join(root,'environment.mjs'),`
    import {readFile,writeFile} from 'node:fs/promises';
    import {join} from 'node:path';
    export async function createEnvironment(options,{instancePath}) {
      if(process.argv[2]==='worker') {
        const job=JSON.parse(await readFile(join(instancePath,'job.json'),'utf8'));
        await writeFile(join(${JSON.stringify(root)},'captured-job.json'),JSON.stringify(job));
        process.exit(0);
      }
      return {resources:async()=>[{id:'cpu',available:true}],observe:async()=>({state:'stopped'})};
    }`);
  const config={format:1,root,mode:'phase1',python:join(root,'missing-python'),skills:{activeVersion:'v'},
    actor_seconds_by_phase:caps,environment:{module:'environment.mjs'},projects:[{id:'p',goal:'numeric',phase:'phase1',actor_seconds:7200}]};
  await writeFile(join(root,'instance.json'),JSON.stringify(config));
  let id;
  if(resume) {
    const state=new State(join(root,'state.sqlite'));state.add(config.projects[0]);
    const job=new RecoveryQueue(state).claim('p',1,{},'cpu');id=job.attempt.id;
    state.checkpoint(id,3700,'retained-session.jsonl');state.close();
    const directory=join(root,'attempts',id);await mkdir(directory,{recursive:true});
    await writeFile(join(directory,'job.json'),JSON.stringify({...job,mode:'phase1',options:{},wave:1,
      skillVersion:'v',resource:{id:'cpu'}}));
  }
  if(['investigation','retry'].includes(kind)) {
    const state=new State(join(root,'state.sqlite'));state.add(config.projects[0]);
    const queue=new RecoveryQueue(state),origin=queue.claim('p',0,{});
    const result={passed:false,findings:'observed failure',evidence:[{collection:'fixture',entry_id:'fact'}]};
    queue.recordOriginal(origin.attempt.id,result,0);queue.acknowledgeLesson(queue.lessons()[0].id);
    state.event(origin.attempt.id,'postprocess-completed',{});
    const directory=join(root,'attempts',origin.attempt.id);await mkdir(directory,{recursive:true});
    await writeFile(join(directory,'job.json'),JSON.stringify({...origin,mode:'phase1',options:{},wave:0,skillVersion:'v',resource:{id:'cpu'}}));
    if(kind==='retry') {
      const investigation=queue.claim('p',1,{});
      queue.complete(investigation.attempt.id,{...result,useful:true},1);
      queue.acknowledgeLesson(queue.lessons().find(lesson=>lesson.attempt_id===investigation.attempt.id).id);
      state.event(investigation.attempt.id,'postprocess-completed',{});
      state.db.exec('CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
      state.db.prepare('INSERT OR REPLACE INTO control VALUES (?,?)').run('wave','1');
    }
    state.close();
  }
  await operate(resume?'resume':'wave',config,{concurrency:1});
  const captured=JSON.parse(await readFile(join(root,'captured-job.json'),'utf8'));
  assert.equal(captured.actor_seconds,3600);
  assert.equal(captured.kind,resume?'original':kind);
  const state=new State(join(root,'state.sqlite'));t.after(()=>state.close());
  if(resume){assert.ok(state.attempt(id).elapsed>=3700);assert.equal(state.attempt(id).session,'retained-session.jsonl');}
  assert.equal(state.project('p').spec.actor_seconds,7200);
});
