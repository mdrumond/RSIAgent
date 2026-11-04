import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdir,writeFile,readFile,cp,symlink,stat} from 'node:fs/promises';
import {join,resolve,relative,isAbsolute} from 'node:path';
import {fileURLToPath} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';
import {DatabaseSync} from 'node:sqlite';
import {createHash} from 'node:crypto';
import {provider} from '../driver_fixture.mjs';
import {Blocked} from './runner.mjs';
import {stages} from '../../../controller/workflow.mjs';
import {ownerStopped} from '../../../controller/lifecycle.mjs';

const repository=fileURLToPath(new URL('../../../',import.meta.url));
const oracle=join(repository,'tests/ascend/driver_fixture.py');
const quote=s=>"'"+s.replaceAll("'","'\\''")+"'";
export const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
export async function waitFor(check,signal,seconds=20) {
  const until=performance.now()+seconds*1000;
  for(;;) {
    signal?.throwIfAborted();
    const result=await check();if(result)return result;
    if(performance.now()>=until)throw Error('Campaign observable transition deadline exceeded');
    await delay(40,undefined,{signal});
  }
}

// Each fixture uses an external installed release, native Pi and the real KDB.
// The SDK is reused from the source installation; packaging is another case's remit.
export async function campaignFixture(ctx,scenario,ids,{loseDelivery=false}={}) {
  const directory=join(ctx.directory,scenario),instance=join(directory,'instance');
  await mkdir(directory,{recursive:true});
  const rel=relative(repository,instance);
  assert.ok(rel.startsWith('..')||isAbsolute(rel),'Resilience output must be outside source machinery');
  const python=process.env.DRIVER_PYTHON??'python3.12';
  const env={...process.env,DRIVER_FIXTURE_KEY:'fixture-only',PYTHONPATH:repository};
  const children=new Set(),processes=[];let sequence=0,server,cli=join(repository,'cli/main.mjs');
  const rows=(sql,file='state.sqlite',params=[])=>{
    const db=new DatabaseSync(join(instance,file),{readOnly:true});
    db.exec('PRAGMA busy_timeout=5000');
    try{return db.prepare(sql).all(...params);}finally{db.close();}
  };
  const execute=(file,args,signal=ctx.signal)=>{
    signal?.throwIfAborted();
    const number=sequence++,started=new Date().toISOString();
    const child=spawn(file,args,{cwd:directory,env,signal,stdio:['ignore','pipe','pipe']});
    children.add(child);let stdout='',stderr='',failure;
    child.stdout.on('data',d=>stdout+=d);child.stderr.on('data',d=>stderr+=d);
    child.on('error',error=>{failure=error;});
    const output=new Promise((ok,no)=>child.once('close',async(code,killedBy)=>{
      children.delete(child);
      try {
        const receipt={number,started,ended:new Date().toISOString(),code,signal:killedBy};
        processes.push(receipt);
        await Promise.all([writeFile(join(directory,`process-${number}.stdout`),stdout),
          writeFile(join(directory,`process-${number}.stderr`),stderr),
          writeFile(join(directory,'processes.json'),JSON.stringify(processes,null,2))]);
        if(failure)throw failure;
        assert.equal(code,0,`${file} exited ${code}; retained process-${number}.stderr`);
        ok(stdout.trim()?JSON.parse(stdout.trim().split('\n').at(-1)):null);
      }catch(error){no(error);}
    }));
    void output.catch(()=>{});return {child,output};
  };
  const start=(command,...args)=>execute(process.execPath,[cli,command,'--instance',instance,...args]);
  const run=async(command,...args)=>start(command,...args).output;
  // Register before launching any service/process, including partial bootstrap failures.
  ctx.onCleanup(async({signal,remainingSeconds}={})=>{
    const errors=[];
    const attempt=async fn=>{try{await fn();}catch(error){errors.push(error);}};
    await attempt(async()=>{
      try{await stat(join(instance,'instance.json'));}
      catch(error){if(error.code==='ENOENT')return;throw error;}
      await execute(process.execPath,[cli,'stop','--instance',instance],signal).output;
      await waitFor(async()=>{
        const owners=rows('SELECT owner FROM attempts').map(r=>JSON.parse(r.owner)).filter(Boolean);
        const controller=rows("SELECT value FROM control WHERE key='owner'")[0];
        if(controller)owners.push(JSON.parse(controller.value));
        return (await Promise.all(owners.filter(Boolean).map(ownerStopped))).every(Boolean);
      },signal,Math.min(15,remainingSeconds??15));
    });
    await attempt(async()=>{
      let receipt;
      try{receipt=JSON.parse(await readFile(join(instance,'knowledge-service.json'),'utf8'));}
      catch(error){if(error.code==='ENOENT')return;throw error;}
      assert.ok(receipt.owner?.pid&&receipt.owner?.start&&receipt.owner?.boot,'KDB owner identity required');
      if(!await ownerStopped(receipt.owner)) {
        process.kill(receipt.owner.pid,'SIGTERM');
        await waitFor(()=>ownerStopped(receipt.owner),signal,Math.min(10,remainingSeconds??10));
      }
    });
    for(const child of [...children])await attempt(async()=>{
      if(child.exitCode===null&&child.signalCode===null)child.kill('SIGTERM');
      await waitFor(()=>child.exitCode!==null||child.signalCode!==null,signal,5);
    });
    await attempt(async()=>{
      if(server) {
        await writeFile(join(directory,'provider.json'),JSON.stringify(server.requests));
        await server.close();
        assert.deepEqual(server.failures,[]);
      }
    });
    await writeFile(join(directory,'closure.json'),JSON.stringify({remainingSeconds,
      liveChildren:children.size,errors:errors.map(e=>e.message)},null,2));
    if(errors.length)throw new AggregateError(errors,'Campaign fixture cleanup failed');
  });
  try {
    await execute(python,['-c','import ascend_tools.kdb; import mcp; print("{}")']).output;
  }catch(error) {
    if(ctx.signal.aborted)throw error;
    throw new Blocked('Configured DRIVER_PYTHON lacks the existing KDB/MCP runtime; see process-0.stderr');
  }
  server=await provider(scenario,{nativeSession:async()=>rows(
    'SELECT session FROM attempts WHERE session IS NOT NULL ORDER BY rowid DESC LIMIT 1')[0]?.session});
  // Seed only source knowledge; all attempts and outcomes come from the actual CLI.
  const baseline=join(directory,'baseline.sqlite');
  await execute(python,[oracle,'baseline',baseline]).output;
  const skills=join(directory,'skills');await mkdir(skills);
  const map=Object.fromEntries([...stages,'global-kdb'].map(s=>[s,s+'.md']));
  for(const path of Object.values(map))await writeFile(join(skills,path),
    'Read this skill; retrieve good and negative knowledge in design and optimisation; follow canonical steps.');
  await writeFile(join(skills,'skill-map.json'),JSON.stringify(map));
  const projects=[];
  for(const id of ids) {
    const source=join(directory,id,'source');await mkdir(source,{recursive:true});
    await writeFile(join(source,'answer.py'),'def answer(): return 0\n');
    projects.push({id,goal:id,source,actor_seconds:120,
      verifier:{files:[oracle],command:`${quote(python)} ${quote(oracle)} verify ${quote(source)}`}});
  }
  const input=join(directory,'bootstrap.json');
  await writeFile(input,JSON.stringify({baseline,skills,projects,install:false,
    environment:{module:'machinery/integrations/environments/index.mjs',options:{mode:'local'}}}));
  await execute(process.execPath,[cli,'bootstrap',instance,'--config',input]).output;
  await symlink(join(repository,'node_modules'),join(instance,'machinery/node_modules'),'dir');
  cli=join(instance,'machinery/cli/main.mjs');
  await cp(oracle,join(instance,'machinery/ascend_tools/fixture_embeddings.py'));
  // Fault at the real durable sink boundary, not a fake outbox/state transition.
  if(loseDelivery)await writeFile(join(instance,'machinery/ascend_tools/campaign_delivery.py'),`
from pathlib import Path
from .fixture_embeddings import Embeddings
from .kdb import KDB
_original = KDB.record_lesson
_marker = Path(${JSON.stringify(join(directory,'delivery-lost.json'))})
def _once(self, *args, **kwargs):
    result = _original(self, *args, **kwargs)
    if not _marker.exists():
        import json
        _marker.write_text(json.dumps(result))
        raise RuntimeError('Injected lost delivery response after durable KDB commit')
    return result
KDB.record_lesson = _once
`);
  async function config(patch) {
    const path=join(directory,`config-${sequence}.json`);await writeFile(path,JSON.stringify(patch));
    return run('config','--config',path);
  }
  await config({python,embedding_factory:loseDelivery?'ascend_tools.campaign_delivery:Embeddings':
    'ascend_tools.fixture_embeddings:Embeddings',target:'cpu',investigation_seconds:120,
    failed_recovery_rounds:5,concurrency:1,curriculum_seconds:30,
    actor:{provider:'driver-fixture',model:'scripted',
      credential_env:'DRIVER_FIXTURE_KEY',base_url:server.url,
      metadata:{id:'scripted',name:'Scripted canonical campaign',api:'openai-completions',
        reasoning:false,input:['text'],contextWindow:128000,maxTokens:8192,
        cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}}});
  async function evidence() {
    assert.deepEqual(server.failures,[]);
    const requests=server.requests.map((request,index)=>({index,tools:request.tools.map(t=>t.function.name),
      messages:request.messages}));
    await writeFile(join(directory,'provider.json'),JSON.stringify(requests));
    const snapshots={attempts:rows('SELECT * FROM attempts'),projects:rows('SELECT * FROM projects'),
      recovery:rows('SELECT * FROM recovery_attempts'),lessons:rows('SELECT * FROM recovery_lessons'),
      events:rows('SELECT * FROM events')};
    await writeFile(join(directory,'ledger.json'),JSON.stringify(snapshots,null,2));
    return {directory,instance,providerRequests:requests.length,attempts:snapshots.attempts.length};
  }
  return {directory,instance,run,start,rows,config,evidence,server};
}
