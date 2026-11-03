import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, mkdir, writeFile, readFile, cp, rm } from 'node:fs/promises';
import { join, resolve, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { setTimeout as delay } from 'node:timers/promises';
import { DatabaseSync } from 'node:sqlite';
import { stages } from '../../controller/workflow.mjs';
import { ownerStopped } from '../../controller/lifecycle.mjs';
import { provider } from './driver_fixture.mjs';
const fixture = resolve('tests/ascend/driver_fixture.py');
const python = process.env.DRIVER_PYTHON ?? 'python3.12';
const quote = text => "'" + text.replaceAll("'", "'\\''") + "'";
export const rows = (root,sql,file='state.sqlite') => {
  const db = new DatabaseSync(join(root,file),{readOnly:true});
  db.exec('PRAGMA busy_timeout=5000');
  try {return db.prepare(sql).all();} finally {db.close();}
};
export async function until(check,timeout=15000) {
  const end = Date.now()+timeout;
  while (Date.now()<end) {const value=await check();if(value)return value;await delay(50);}
  throw new Error('Timed out waiting for observable driver transition');
}
export async function setup(t,scenario,ids) {
  const root = await mkdtemp(join(process.env.DRIVER_ARTIFACTS ?? tmpdir(),'driver-acceptance-'));
  const instance = join(root,'instance'), server = await provider(scenario,{nativeSession:async()=>
    rows(instance,'SELECT session FROM attempts WHERE session IS NOT NULL ORDER BY rowid DESC LIMIT 1')[0]?.session});
  const env = {...process.env,DRIVER_FIXTURE_KEY:'fixture-only',
    PATH:dirname(python)+':'+process.env.PATH,PYTHONPATH:resolve('.')};
  let sequence=0,okay=false;
  const execute=(executable,args) => {
    const child=spawn(executable,args,{env,stdio:['ignore','pipe','pipe']});
    let stdout='',stderr='';child.stdout.on('data',d=>stdout+=d);child.stderr.on('data',d=>stderr+=d);
    const output=(async()=>{
      const [code]=await once(child,'close');
      const name=join(root,`process-${sequence++}`);
      await writeFile(name+'.stdout',stdout);await writeFile(name+'.stderr',stderr);
      assert.equal(code,0,`${executable} ${args.join(' ')}\n${stderr}\nartifacts=${root}`);
      return stdout.trim() ? JSON.parse(stdout.trim().split('\n').at(-1)) : null;
    })();
    return {child,output};
  };
  let cli=resolve('cli/main.mjs');
  const start=(command,...args)=>execute(process.execPath,[cli,command,'--instance',instance,...args]);
  const run=async(command,...args)=>start(command,...args).output;
  t.after(async()=>{
    try {
      await run('stop');
      await until(async()=>{
        const control=rows(instance,"SELECT value FROM control WHERE key='owner'")[0];
        const owner=control && JSON.parse(control.value);
        return !owner || await ownerStopped(owner);
      },5000);
    }catch(error){t.diagnostic(`Controller cleanup retained for inspection: ${error.message}`);}
    try {
      const receipt=JSON.parse(await readFile(join(instance,'knowledge-service.json'),'utf8'));
      if(!await ownerStopped(receipt.owner))process.kill(receipt.owner.pid,'SIGTERM');
    }catch(error){if(error.code!=='ENOENT' && error.code!=='ESRCH')throw error;}
    await server.close();
    assert.deepEqual(server.failures,[],`Fixture protocol failures; artifacts=${root}`);
    if(okay && !process.env.DRIVER_ARTIFACTS)await rm(root,{recursive:true,force:true});
    else t.diagnostic(`Retained actual CLI/native session evidence: ${root}`);
  });
  const baseline=join(root,'baseline.sqlite');
  // Only the embedding backend is a fixture; indexing and serving use production KDB.
  const seed=spawn(python,[fixture,'baseline',baseline],{env,stdio:['ignore','pipe','pipe']});
  let seedError='';seed.stderr.on('data',d=>seedError+=d);
  assert.equal((await once(seed,'close'))[0],0,seedError);
  const skills=join(root,'skills');await mkdir(skills);
  const map=Object.fromEntries([...stages,'global-kdb'].map(stage=>[stage,stage+'.md']));
  for(const path of Object.values(map))await writeFile(join(skills,path),'Read current skill; query both kinds in design/optimisation; use rsi_advance defaults and frozen verifier.');
  await writeFile(join(skills,'skill-map.json'),JSON.stringify(map));
  const projects=[];
  for(const id of ids) {
    const source=join(root,id,'source');await mkdir(source,{recursive:true});
    await writeFile(join(source,'answer.py'),'def answer(): return 0\n');
    projects.push({id,goal:id,source,actor_seconds:120,
      verifier:{files:[fixture],command:`${quote(python)} ${quote(fixture)} verify ${quote(source)}`}});
  }
  const config={baseline,skills,projects,environment:{module:'machinery/integrations/environments/index.mjs',options:{mode:'local'}}};
  const input=join(root,'bootstrap.json');await writeFile(input,JSON.stringify(config));
  const boot=execute(process.execPath,[cli,'bootstrap',instance,'--config',input]);await boot.output;
  cli=join(instance,'machinery','cli','main.mjs');
  // Install only the configurable test embedding module, never replace runtime code.
  await cp(fixture,join(instance,'machinery','ascend_tools','fixture_embeddings.py'));
  const settings=join(root,'settings.json');await writeFile(settings,JSON.stringify({python,
    embedding_factory:'ascend_tools.fixture_embeddings:Embeddings',target:'cpu',investigation_seconds:120,
    failed_recovery_rounds:2,concurrency:1,curriculum_seconds:30,
    actor:{provider:'driver-fixture',model:'scripted',credential_env:'DRIVER_FIXTURE_KEY',base_url:server.url,
      metadata:{id:'scripted',name:'Scripted SSE acceptance',api:'openai-completions',reasoning:false,
        input:['text'],contextWindow:128000,maxTokens:8192,
        cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}}}));
  await run('config','--config',settings);
  return {root:instance,run,start,server,pass:()=>{okay=true;}};
}

