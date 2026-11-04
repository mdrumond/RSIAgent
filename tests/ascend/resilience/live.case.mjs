import assert from 'node:assert/strict';
import {readFile,writeFile,readlink,readdir} from 'node:fs/promises';
import {join} from 'node:path';
import {once} from 'node:events';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {createLiveFixture,constructionResume,digest,until} from './live-harness.mjs';
import {readProcessStat} from '../../../integrations/environments/index.mjs';
import {processIdentity,ownerStopped,attemptStopped} from '../../../controller/lifecycle.mjs';

export const cases=[
  {id:'deepseek-readiness',tier:'deepseek',async run(ctx){
    let before=0,after=0;
    const f=await createLiveFixture(ctx,{mcpDelayMs:11000});
    await f.run({extensions:[pi=>{
      pi.on('before_provider_request',event=>{before++;
        assert.ok(f.mcpTimeline.some(e=>e.type==='list-ready'));
        assert.ok(event.payload.tools.some(tool=>tool.function?.name==='mcp__kdb__search'));
      });
      pi.on('tool_result',event=>{if(event.toolName==='rsi_knowledge'&&!event.isError)after++;});
    }]});
    const evidence=f.verifyAccepted();assert.ok(before>0);assert.ok(after>=4);
    await f.save('readiness.json',f.mcpTimeline);
    return {...evidence,providerRequests:before,retrievals:after};
  }},
  {id:'deepseek-workspace-revision',tier:'deepseek',async run(ctx){
    const f=await createLiveFixture(ctx,{goal:'Fix solve(values) to sum integers. Run a baseline with rsi_target; use native LOCAL bash to repair solution.py persistently, then run rsi_target again. Complete all nine stages. Keep reflections concise; do not edit source after successful rsi_verify.'});
    const path=join(f.project.source,'solution.py'),initial=await digest(path);let changed=false;
    await f.run({extensions:[pi=>{
      pi.on('tool_call',event=>{
        if(['write','edit'].includes(event.toolName)&&String(event.input?.path).endsWith('solution.py'))return {block:true,reason:'This fixture exercises native local Bash editing; use bash for solution.py.'};
      });
      pi.on('tool_result',async event=>{
        if(event.toolName==='rsi_verify'&&!event.isError&&!changed&&f.state.events(f.attempt.id).findLast(e=>e.type==='verification')?.data.passed){
          changed=true;await writeFile(path,(await readFile(path,'utf8'))+'\n# fixture revision boundary\n');
          await ctx.event({type:'injected-source-revision',sha256:await digest(path)});
        }
      });
    }]});
    const evidence=f.verifyAccepted(),events=f.state.events(f.attempt.id);
    assert.equal(changed,true);assert.notEqual(await digest(path),initial);
    assert.ok(f.observations.some(e=>e.tool==='rsi_advance'&&e.error));
    assert.ok(f.observations.some(e=>e.tool==='rsi_revise'&&!e.error));
    assert.ok(events.filter(e=>e.type==='verification'&&e.data.passed).length>=2);
    assert.ok(f.observations.some(e=>e.tool==='bash'&&!e.error));assert.ok(f.observations.filter(e=>e.tool==='rsi_target'&&!e.error).length>=2);
    return {...evidence,staleRejected:true,sourcePersisted:true};
  }},
  {id:'deepseek-construction-resume',tier:'deepseek',async run(ctx){
    const f=await createLiveFixture(ctx,{admit:false});return constructionResume(f);
  }},
  {id:'deepseek-worker-loss',tier:'deepseek',async run(ctx){
    const f=await createLiveFixture(ctx,{admit:false,goal:'First read rsi_skill. Before advancing preflight use native LOCAL bash to run Python that writes its os.getpid() into worker-child.pid and sleeps60 seconds. This is an isolated worker-lifetime diagnostic; do not use rsi_target for that command.'});
    let nativeOwner,group;
    const fields=stat=>stat.slice(stat.lastIndexOf(')')+2).split(' ');
    const groupMembers=async()=>{const live=[];for(const name of await readdir('/proc')){
      if(!/^\d+$/.test(name))continue;const stat=await readProcessStat(name);if(!stat)continue;const row=fields(stat);
      if(Number(row[2])===group.pgid&&!['Z','X'].includes(row[0]))live.push(Number(name));
    }return live;};
    const child=spawn(process.execPath,[fileURLToPath(new URL('./live-worker.mjs',import.meta.url)),f.config.root],{stdio:['ignore','pipe','pipe']});
    let output='';child.stdout.on('data',d=>output+=d);child.stderr.on('data',d=>output+=d);const exited=once(child,'exit');void exited.catch(()=>{});
    async function capture(pid){
      assert.equal(await readlink(`/proc/${pid}/cwd`),f.project.source);
      const stat=await readProcessStat(pid);assert.ok(stat);let parent=Number(fields(stat)[1]),owned=false;
      while(parent>1){if(parent===child.pid){owned=true;break;}const ancestor=await readProcessStat(parent);if(!ancestor)break;parent=Number(fields(ancestor)[1]);}
      assert.ok(owned,'Milestone PID must descend from this exact native worker');
      nativeOwner=await processIdentity(pid);group={pgid:Number(fields(stat)[2]),start:fields(stat)[19]};
    }
    ctx.onCleanup(async cleanup=>{
      const errors=[];
      if(!nativeOwner)try{await capture(Number(await readFile(join(f.project.source,'worker-child.pid'),'utf8')));}
      catch(error){if(!['ENOENT','ESRCH'].includes(error.code))errors.push(error.message);}
      if(child.exitCode===null&&child.signalCode===null){child.kill('SIGKILL');await exited;}
      if(nativeOwner&&!group){const stat=await readProcessStat(nativeOwner.pid);if(stat){const row=fields(stat);group={pgid:Number(row[2]),start:row[19]};}}
      if(group){
        const stat=await readProcessStat(nativeOwner.pid),own=fields(await readProcessStat(process.pid));
        if(stat&&fields(stat)[19]===group.start&&Number(fields(stat)[2])===group.pgid){
          assert.ok(group.pgid>1&&group.pgid!==Number(own[2]),'Only the known detached fixture group may be killed');
          try{process.kill(-group.pgid,'SIGKILL');}catch(error){if(error.code!=='ESRCH')throw error;}
        }else if((await groupMembers()).length)errors.push('Native group identity unavailable; retain held/unknown evidence');
        if(!errors.length)await until(async()=>!(await groupMembers()).length,{signal:cleanup.signal});
      }else{
        for(const a of f.state.db.prepare('SELECT id FROM attempts').all()){
          const events=f.state.events(a.id),closed=new Set(events.filter(e=>e.type==='native-bash-completed').map(e=>e.data.operation));
          if(events.some(e=>e.type==='native-bash-started'&&!closed.has(e.data.operation)))errors.push('Native Bash closure identity missing');
        }
      }
      await f.save('process-closure.json',{workerClosed:child.exitCode!==null||child.signalCode!==null,nativeGroupStopped:group?!(await groupMembers()).length:errors.length===0,group,errors});
      assert.deepEqual(errors,[],'Unknown native Bash closure must fail cleanup');
      await f.save('worker-output.json',{output,owner:nativeOwner});
    });
    const pid=await until(async()=>{try{return Number(await readFile(join(f.project.source,'worker-child.pid'),'utf8'));}catch{return false;}},{signal:ctx.signal,seconds:60});
    await capture(pid);const attempt=f.state.db.prepare('SELECT * FROM attempts WHERE outcome IS NULL').get();assert.ok(attempt?.session);
    child.kill('SIGKILL');await exited;assert.equal(await ownerStopped(nativeOwner),false);
    const recorded=f.state.events(attempt.id),completed=new Set(recorded.filter(e=>e.type==='native-bash-completed').map(e=>e.data.operation));
    const starts=recorded.filter(e=>e.type==='native-bash-started');assert.equal(starts.filter(e=>!completed.has(e.data.operation)).length,1);
    await assert.rejects(f.state.resume(attempt.id,a=>attemptStopped(f.state,a,{}),await processIdentity()),/Previous execution/);
    assert.equal(f.state.attempt(attempt.id).outcome,null);assert.equal(f.state.attempt(attempt.id).session,attempt.session);
    assert.equal(f.state.events(attempt.id).filter(e=>e.type==='native-bash-started').length,starts.length);
    return {id:attempt.id,session:attempt.session,safeHeld:true,replayPrevented:true};
  }},
  {id:'deepseek-compaction',tier:'deepseek',async run(ctx){
    const f=await createLiveFixture(ctx,{admit:false}),session=await f.openSession(),file=session.sessionManager.getSessionFile();
    let exhausted=false;const timer=setTimeout(()=>{exhausted=true;session.abortCompaction();void session.abort();},(f.config.investigation_seconds??300)*1000),abort=()=>{session.abortCompaction();void session.abort();};ctx.signal.addEventListener('abort',abort,{once:true});
    let calls=0;session.subscribe(event=>{if(event.type==='tool_execution_end')calls++;});
    try{
      await session.prompt('Use native bash to write proof.txt containing the integer42 computed as6*7. Retrieve positive arithmetic evidence using mcp__kdb__search. Then report what the file and tool actually returned.');await session.waitForIdle();
      assert.ok(calls>=2);const first=calls;assert.equal((await readFile(join(f.project.source,'proof.txt'),'utf8')).trim(),'42');
      await session.compact('Keep the exact proof.txt path, numeric result42 and indexed citation; further tool work follows.');
      assert.ok(session.sessionManager.getEntries().some(e=>e.type==='compaction'));
      await session.prompt('After compaction use native read to inspect proof.txt, native bash to independently assert its integer equals sum(6 for _ in range(7)), and mcp__kdb__search for negative arithmetic evidence. Report only observed results.');await session.waitForIdle();
      ctx.signal.throwIfAborted();assert.equal(exhausted,false);assert.ok(calls>=first+3);assert.equal(session.sessionManager.getSessionFile(),file);
      assert.equal((await readFile(join(f.project.source,'proof.txt'),'utf8')).trim(),'42');return {session:file,compacted:true,toolCallsBefore:first,toolCallsAfter:calls-first};
    }finally{clearTimeout(timer);ctx.signal.removeEventListener('abort',abort);}
  }},
  {id:'deepseek-wave-recovery',tier:'deepseek',async run(ctx){
    const f=await createLiveFixture(ctx,{admit:false}),parent=await f.seedFailure(),original=structuredClone(f.state.project('numeric').spec);
    const first=await f.command('wave');assert.equal(first.results.length,1);
    const child=f.state.db.prepare('SELECT * FROM attempts WHERE id<>?').get(parent.id);assert.ok(child);f.verifyAccepted(child);
    const recovery=f.state.db.prepare('SELECT kind,eligible_wave FROM recovery_queue WHERE project=?').get('numeric');
    assert.equal(recovery.kind,'retry');assert.ok(recovery.eligible_wave>first.wave);
    await f.command('wave');const retry=f.state.db.prepare('SELECT * FROM attempts WHERE id<>? AND id<>?').get(parent.id,child.id);
    assert.ok(retry);f.verifyAccepted(retry);assert.equal(f.state.project('numeric').status,'passed');
    assert.deepEqual(f.state.project('numeric').spec,original);assert.deepEqual(f.state.attempt(parent.id),parent);
    assert.equal(f.state.db.prepare('SELECT COUNT(*) AS n FROM recovery_lessons WHERE recorded=0').get().n,0);
    return {original:parent.id,investigation:child.id,retry:retry.id,nextWaveRecovery:true};
  }}
];
