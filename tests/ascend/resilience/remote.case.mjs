import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {randomUUID} from 'node:crypto';
import {Blocked} from './runner.mjs';
import {createEnvironment as genericEnvironment,createCplClient} from '../../../integrations/environments/index.mjs';
import {loadLiveProfile,createLiveFixture,constructionResume,digest} from './live-harness.mjs';

async function remoteOptions() {
  await loadLiveProfile(); // Local cached native model/credential checks, no API request.
  const profile=process.env.RESILIENCE_REMOTE_PROFILE;
  if(!profile)throw new Blocked('Set RESILIENCE_REMOTE_PROFILE to ignored registered CPU options');
  let input;
  try{input=JSON.parse(await readFile(profile,'utf8'));}
  catch{throw new Blocked('Registered CPU profile is not readable JSON');}
  const options=input.environment?.options??input;
  if(!options.target||!options.runtime||!options.workspacePath)
    throw new Blocked('Registered target/runtime/workspace required');
  return {...options,mode:'cpl',remoteDirectory:`resilience/live-${randomUUID()}`};
}
function contextFor(f,ctx) {
  return {instancePath:join(f.config.root,'attempts',f.attempt.id),signal:ctx.signal,
    event:(type,data)=>f.state.event(f.attempt.id,type,data),exclude:async(_,fn)=>fn()};
}
async function prepared(f,ctx) {
  const environment=await f.factory(contextFor(f,ctx));
  await environment.prepare({project:f.project,attempt:f.attempt});
  return environment;
}
async function runJson(environment,f,ctx,program) {
  let stdout='';
  const result=await environment.execute(`python3 -c '${program}'`,f.project.source,
    {signal:ctx.signal,timeout:30,onData:data=>{stdout+=data;}});
  assert.equal(result.settled,true);assert.equal(result.exitCode,0);
  return {handle:result.handle,value:JSON.parse(stdout.trim().split('\n').at(-1))};
}

export const cases=[
  {id:'remote-workspace',tier:'remote',async run(ctx){
    const options=await remoteOptions();
    const f=await createLiveFixture(ctx,{environment:{options},goal:
      'Fix solve(values) to sum integers. First use rsi_target for a baseline; edit solution.py persistently using native LOCAL bash, then run rsi_target again. Complete all nine stages, keep the frozen verifier unchanged, and keep reflections concise. If a stale source check rejects acceptance, use the canonical revision path and fresh validation.'});
    const path=join(f.project.source,'solution.py'),initial=await digest(path);let injected=false;
    await f.run({extensions:[pi=>{
      pi.on('tool_call',event=>{
        if(['write','edit'].includes(event.toolName)&&String(event.input?.path).endsWith('solution.py'))
          return {block:true,reason:'This case exercises persistent source repair through native LOCAL bash.'};
      });
      pi.on('tool_result',async event=>{
        if(event.toolName==='rsi_verify'&&!event.isError&&!injected&&
          f.state.events(f.attempt.id).findLast(e=>e.type==='verification')?.data.passed) {
          injected=true;await writeFile(path,(await readFile(path,'utf8'))+'\n# remote case stale-source boundary\n');
          await ctx.event({type:'source-revision',sha256:await digest(path)});
        }
      });
    }]});
    const evidence=f.verifyAccepted(),events=f.state.events(f.attempt.id);
    assert.equal(injected,true);assert.notEqual(await digest(path),initial);
    assert.ok(f.observations.some(e=>e.tool==='bash'&&!e.error));
    assert.ok(f.observations.filter(e=>e.tool==='rsi_target'&&!e.error).length>=2);
    assert.ok(f.observations.some(e=>e.tool==='rsi_advance'&&e.error));
    assert.ok(f.observations.some(e=>e.tool==='rsi_revise'&&!e.error));
    assert.ok(events.filter(e=>e.type==='verification'&&e.data.passed).length>=2);
    // An independent target read measures actual bytes after the final synchronization.
    const environment=await prepared(f,ctx);
    const remote=await runJson(environment,f,ctx,
      'import hashlib,json; print(json.dumps({"sha256":hashlib.sha256(open("solution.py","rb").read()).hexdigest()}))');
    const sourceHash=await digest(path);assert.equal(remote.value.sha256,sourceHash);
    await f.save('workspace-proof.json',{...evidence,sourceHash,handle:remote.handle,
      nativeBashPersisted:true,staleRejected:true,freshVerification:true});
    return {...evidence,sourceHash,handle:remote.handle,scope:'Local DeepSeek/Pi; registered target Python CPU'};
  }},
  {id:'remote-construction-resume',tier:'remote',async run(ctx){
    const options=await remoteOptions();
    const f=await createLiveFixture(ctx,{environment:{options},admit:false});
    // Shared native CLI stop/resume hook owns session/elapsed checks and provider use.
    const evidence=await constructionResume(f);
    const attempts=f.state.db.prepare('SELECT * FROM attempts WHERE id=?').all(evidence.id);
    assert.equal(attempts.length,1);assert.equal(attempts[0].outcome,'passed');
    assert.ok(f.state.events(evidence.id).some(e=>e.type==='execution-handle'&&e.data.handle));
    await f.save('construction-proof.json',evidence);
    return {...evidence,scope:'Local constructor/provider resumed; registered target Python CPU'};
  }},
  {id:'remote-observer-resume',tier:'remote',async run(ctx){
    const options=await remoteOptions(),native=createCplClient(options.clientPath);
    let armed=false,disconnected=false;const dispatches=[];
    const environmentFactory=async context=>{
      assert.ok(context.signal,'Every direct body/cleanup factory must supply its own signal');
      const client=async(args,settings={})=>{
        // Generic execute composes body signals; explicit stop/observe cleanup must
        // override an already-aborted body context.
        const signal=settings.signal??context.signal;
        const command=args[0]==='run'&&String(args[args.indexOf('--file')+1]).endsWith('command.sh');
        const result=await native(args,{...settings,signal});
        if(command)dispatches.push({handle:result.handle,observerBoundary:armed});
        if(command&&armed&&!disconnected) {
          assert.ok(result.handle);disconnected=true;
          await ctx.event({type:'lost-observer-response',handle:result.handle});
          // The native submission receipt already exists. Let the generic environment
          // recover that exact handle; never submit a replacement command.
          throw Error('Injected response loss after native submission receipt');
        }
        return result;
      };
      return genericEnvironment(options,{...context,client,
        resources:async()=>[{id:'remote-live-cpu',available:true}]});
    };
    const f=await createLiveFixture(ctx,{environment:{options},environmentFactory});
    await f.run();const evidence=f.verifyAccepted();
    const sourceHash=await digest(join(f.project.source,'solution.py'));
    const first=await prepared(f,ctx);armed=true;
    const result=await runJson(first,f,ctx,
      'import json; from pathlib import Path; p=Path("../outputs/observer-once.txt"); p.parent.mkdir(parents=True,exist_ok=True); p.open("a").write("dispatch\\n"); n=len(p.read_text().splitlines()); assert n==1; print(json.dumps({"dispatches":n}))');
    assert.equal(disconnected,true);assert.equal(result.value.dispatches,1);
    const work=dispatches.filter(d=>d.observerBoundary);
    assert.equal(work.length,1);assert.equal(work[0].handle,result.handle);
    const count=dispatches.length;
    // Recreate only the generic observer, retaining its instance path and native receipt.
    const observer=await f.factory(contextFor(f,ctx));
    const observed=await observer.observe(result.handle);
    assert.equal(observed.state,'stopped');assert.equal(observed.exitCode,0);
    assert.equal(dispatches.length,count);assert.equal(await digest(join(f.project.source,'solution.py')),sourceHash);
    await f.save('observer-proof.json',{...evidence,handle:result.handle,observed,
      dispatches,sourceHash,sameHandle:true,noReplay:true});
    return {...evidence,handle:result.handle,sameHandle:true,noReplay:true,
      scope:'Local DeepSeek/Pi; native GZ CPU submission and independent observer'};
  }}
];
