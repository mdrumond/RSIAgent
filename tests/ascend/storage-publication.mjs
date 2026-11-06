import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm, cp } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { State } from '../../controller/state.mjs';
import { CampaignStorage } from '../../controller/storage.mjs';
import { KnowledgeHistory } from '../../controller/knowledge-history.mjs';
import { stages } from '../../controller/workflow.mjs';
import { initializeStorageKnowledge, publishStorageMemory, recordStorageSkill } from '../../controller/storage-knowledge.mjs';
import { deliverLessons } from '../../cli/memory.mjs';
import { run as edit } from '../../cli/skill-input.mjs';

async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'publication-'));
  const state=new State(join(root,'state.sqlite')),ledger=new CampaignStorage(state,root);
  await ledger.initialize();
  t.after(async()=>{state.close();await rm(root,{recursive:true,force:true});});
  const config={format:1,root,storage:{format:1},target:'a5',skills:{activeVersion:'initial'}};
  const directory=join(root,'skills','versions','initial');await mkdir(directory,{recursive:true});
  const map=Object.fromEntries(['global-kdb',...stages].map(scope=>[scope,`${scope}.md`]));
  for(const path of Object.values(map))await writeFile(join(directory,path),'baseline guidance');
  await writeFile(join(directory,'skill-map.json'),JSON.stringify(map));
  await writeFile(join(root,'instance.json'),JSON.stringify(config));
  return {config,state,ledger,history:await new KnowledgeHistory(join(root,'knowledge.git')).init()};
}
const lesson=id=>({id,attempt_id:'attempt',outcome:'passed',findings:`lesson ${id}`,
  evidence:[{collection:'repo',entry_id:'entry',sha256:'abc',start_line:1,end_line:2}]});

test('KDB staging failure retains outbox; replay preserves evidence and acknowledges only active ledger',async t=>{
  const {config,state,history}=await fixture(t),item=lesson(1);let ack=0,fail=true;const calls=[];
  const queue={lessons:()=>ack?[]:[item],acknowledgeLesson:()=>{
    assert.equal(state.db.prepare("SELECT generation FROM storage_revisions WHERE name='memory'").get().generation,calls.at(-1).revision);ack++;
  }};
  const invoke=async(tool,args)=>{
    calls.push({tool,...args});
    if(tool==='stage_memory_revision'&&fail)throw new Error('index failed');
    return {generation:args.revision};
  };
  await assert.rejects(deliverLessons({config,queue,invoke}),/index failed/);assert.equal(ack,0);
  const committed=await history.ref('refs/heads/memory');
  fail=false;await deliverLessons({config,queue,invoke});assert.equal(ack,1);
  assert.equal(await history.ref('refs/heads/memory'),committed);
  const snapshot=await history.readRevision(committed);
  assert.deepEqual(snapshot.memory[0].evidence,item.evidence);
  assert.deepEqual(snapshot.memory[0].evidence_refs,['repo:entry']);
  assert.deepEqual(calls.map(c=>c.tool),['stage_memory_revision','stage_memory_revision','activate_memory_revision']);
  const count=calls.length;await publishStorageMemory(config,item,{invoke});assert.equal(calls.length,count);
});
test('activation failure and delayed older delivery cannot replace newer memory',async t=>{
  const {config,state,history}=await fixture(t);let fail=true;
  const invoke=async(tool,args)=>{
    if(tool==='activate_memory_revision'&&fail)throw new Error('activation failed');return {generation:args.revision};
  };
  await assert.rejects(publishStorageMemory(config,lesson(1),{invoke}),/activation failed/);
  assert.equal(state.db.prepare("SELECT * FROM storage_revisions WHERE name='memory'").get(),undefined);
  fail=false;await publishStorageMemory(config,lesson(1),{invoke});
  const newer=await publishStorageMemory(config,lesson(2),{invoke});
  await publishStorageMemory(config,lesson(1),{invoke:()=>{throw Error('must not reactivate');}});
  assert.equal((await history.state('memory')).revision,newer.revision);
  assert.equal((await history.readRevision(newer.revision)).memory.length,2);
});
test('baseline skills, rejected proposals and user edits map to real Git commits without changing memory',async t=>{
  const {config,state,history}=await fixture(t);
  await initializeStorageKnowledge(config,state);
  const initial=await history.state('skills');
  await cp(join(config.root,'skills','versions','initial'),join(config.root,'skills','versions','candidate'),{recursive:true});
  await writeFile(join(config.root,'skills','versions','candidate','global-kdb.md'),'rejected guidance');
  await recordStorageSkill(config,'candidate',{baseVersion:'initial'});
  const rejected=await recordStorageSkill(config,'candidate',{baseVersion:'initial',promote:false,reason:'correctness-regression'});
  assert.equal((await history.readRevision(rejected.revision)).event.metadata.reason,'correctness-regression');
  assert.deepEqual(await history.state('skills'),initial);
  const result=await edit(config,{scope:'global-kdb',content:'user guidance'});
  const mapping=state.db.prepare('SELECT * FROM storage_skill_versions WHERE version=?').get(result.version);
  assert.equal(mapping.base_version,'initial');assert.equal(mapping.decision,'promoted');
  const snapshot=await history.readRevision(mapping.revision);
  assert.equal(snapshot.skills['global-kdb.md'],'user guidance');
  assert.equal(snapshot.event.metadata.reason,'user-edit');
  assert.equal(JSON.parse(await readFile(join(config.root,'instance.json'))).skills.activeVersion,result.version);
  assert.equal((await history.readRevision(rejected.revision)).skills['global-kdb.md'],'rejected guidance');
});
test('storage helpers are inert on existing instances',async()=>{
  const config={root:'/not-created'};
  assert.equal(await initializeStorageKnowledge(config),null);
  assert.equal(await publishStorageMemory(config,lesson(1)),null);
});
test('bootstrap exports only lesson chunks offline; first delivery adopts baseline before indexing full snapshot',async t=>{
  const {config,state,history}=await fixture(t);
  const {DatabaseSync}=await import('node:sqlite');const db=new DatabaseSync(join(config.root,'knowledge.sqlite'));
  db.exec('CREATE TABLE collections(name TEXT,language TEXT); CREATE TABLE chunks(collection TEXT,chunk_id TEXT,path TEXT,start_line INTEGER,end_line INTEGER,text TEXT)');
  const baseline={delivery_id:'seed',lesson:'retain baseline',outcome:'good',applicability:'a5',evidence:[{path:'evidence',sha256:'abc'}]};
  const lines=(JSON.stringify(baseline,null,2)+'\n').split(/(?<=\n)/);
  db.prepare('INSERT INTO collections VALUES (?,?)').run('seed-lessons','lesson');
  db.prepare('INSERT INTO collections VALUES (?,?)').run('repo','python');
  const insert=db.prepare('INSERT INTO chunks VALUES (?,?,?,?,?,?)');
  insert.run('seed-lessons','a','seed.json',1,5,lines.slice(0,5).join(''));
  insert.run('seed-lessons','b','seed.json',4,lines.length,lines.slice(3).join(''));
  insert.run('repo','repo','code.py',1,1,'not JSON or lesson');db.close();
  const initialized=await initializeStorageKnowledge(config,state);
  assert.equal(initialized.memory,null);assert.deepEqual(initialized.baseline.collections,['seed-lessons']);
  assert.equal((await history.readRevision(initialized.baseline.revision)).memory[0].lesson,baseline.lesson);
  assert.deepEqual((await initializeStorageKnowledge(config,state)).baseline,initialized.baseline);
  const calls=[];
  await publishStorageMemory(config,lesson(2),{invoke:async(tool,args)=>{
    calls.push(tool);
    if(tool==='stage_memory_revision')assert.deepEqual(args.records.map(r=>r.lesson).sort(),['lesson 2','retain baseline']);
    return {generation:args.revision};
  }});
  assert.deepEqual(calls,['adopt_memory_revision','stage_memory_revision','activate_memory_revision']);
});
test('independent processes serialize Git writers without losing lessons',async t=>{
  const {config,history}=await fixture(t);
  const {execFile}=await import('node:child_process');const {promisify}=await import('node:util');const execute=promisify(execFile);
  const module=new URL('../../controller/storage-knowledge.mjs',import.meta.url).href;
  const code=`import {publishStorageMemory} from ${JSON.stringify(module)};
    const [config,item]=JSON.parse(process.argv[1]);
    await publishStorageMemory(config,item,{invoke:async(tool,args)=>({generation:args.revision})});`;
  await Promise.all([1,2].map(id=>execute(process.execPath,['--max-old-space-size=256','--input-type=module','-e',code,JSON.stringify([config,lesson(id)])])));
  const active=await history.state('memory');
  assert.deepEqual((await history.readRevision(active.revision)).memory.map(r=>r.lesson).sort(),['lesson 1','lesson 2']);
});
test('existing skill writer records tune proposals and evaluation decisions through the new hooks',async t=>{
  const {config,state}=await fixture(t);
  config.mode='phase2';config.skills.tuningProjectIds=['one','two'];
  await writeFile(join(config.root,'instance.json'),JSON.stringify(config));
  const {run}=await import('../../cli/skills.mjs');
  const {nativeWriter}=await import('./skill-proposal-fixture.mjs');
  const knowledge={ensureKnowledgeService:async()=> 'fixture',createKdbBridge:()=>({extension:()=>{}})};
  const result=await run(config,{manual:true,native:nativeWriter().native,modelRuntime:{},model:{},knowledge,
    runtime:{bootstrap:async request=>mkdir(request.instanceRoot),wave:async()=>({correct:true,score:1})}});
  assert.equal(result.promote,false);
  const row=state.db.prepare('SELECT * FROM storage_skill_versions WHERE version=?').get(result.candidate);
  assert.equal(row.decision,'rejected');assert.equal(row.reason,'inconclusive');assert.equal(row.base_version,'initial');
});

test('restart reconciles a config-selected skill proposal and records rollback history',async t=>{
 const {config,state,history}=await fixture(t);await initializeStorageKnowledge(config,state);
 const original=await history.state('skills');
 const {copyVersion}=await import('../../cli/skill-input.mjs');
 const map=JSON.parse(await readFile(join(config.root,'skills/versions/initial/skill-map.json')));
 const version=await copyVersion(config,map,{scope:'debug',content:'new debugging guidance'});
 await recordStorageSkill(config,version,{baseVersion:'initial'});
 const next={...config,skills:{...config.skills,activeVersion:version}};
 await writeFile(join(config.root,'instance.json'),JSON.stringify(next));
 await initializeStorageKnowledge(next,state);
 const updated=await history.state('skills');assert.notEqual(updated.revision,original.revision);
 assert.equal(state.db.prepare("SELECT revision FROM storage_revisions WHERE name='skills'").get().revision,updated.revision);
 await initializeStorageKnowledge(config,state);
 const rollback=await history.state('skills');assert.notEqual(rollback.revision,updated.revision);
 assert.equal((await history.readRevision(rollback.revision)).skills['debug.md'],'baseline guidance');
});

test('baseline export includes active memory only, excluding staged and superseded records',async t=>{
 const {config,state,history}=await fixture(t);const {DatabaseSync}=await import('node:sqlite');
 const db=new DatabaseSync(join(config.root,'knowledge.sqlite'));
 db.exec(`CREATE TABLE collections(name TEXT,language TEXT);
 CREATE TABLE chunks(collection TEXT,chunk_id TEXT,path TEXT,start_line INTEGER,end_line INTEGER,text TEXT);
 CREATE TABLE memory_collections(collection TEXT);CREATE TABLE active_generation(singleton INTEGER,revision TEXT);
 CREATE TABLE generation_collections(generation TEXT,collection TEXT);
 INSERT INTO active_generation VALUES (1,'active');`);
 for(const name of ['active','old','staged']){
  db.prepare('INSERT INTO collections VALUES (?,?)').run(name,'lesson');db.prepare('INSERT INTO memory_collections VALUES (?)').run(name);
  db.prepare('INSERT INTO chunks VALUES (?,?,?,?,?,?)').run(name,name,'lesson.json',1,1,JSON.stringify({delivery_id:name,lesson:name,outcome:'good',applicability:'a5',evidence:[{path:'e',sha256:'x'}]}));
  db.prepare('INSERT INTO generation_collections VALUES (?,?)').run(name,name);
 }
 db.close();const initialized=await initializeStorageKnowledge(config,state);
 assert.deepEqual(initialized.baseline.collections,['active']);
 assert.deepEqual((await history.readRevision(initialized.baseline.revision)).memory.map(r=>r.lesson),['active']);
});

test('pre-attempt baseline replacement updates Git provenance and adopts only the new collections',async t=>{
  const {DatabaseSync}=await import('node:sqlite');
  const {main}=await import('../../cli/main.mjs');
  const f=await fixture(t);
  const seed=(file,name)=>{
    const db=new DatabaseSync(file);
    try{
      db.exec('CREATE TABLE collections(name TEXT,language TEXT);CREATE TABLE chunks(collection TEXT,path TEXT,start_line INTEGER,end_line INTEGER,text TEXT,chunk_id TEXT)');
      db.prepare('INSERT INTO collections VALUES (?,?)').run(name,'lesson');
      db.prepare('INSERT INTO chunks VALUES (?,?,?,?,?,?)').run(name,'lesson.json',1,1,JSON.stringify({delivery_id:name,lesson:name,evidence_refs:['evidence://'+name],kind:'good'}),name);
    }finally{db.close();}
  };
  const old=join(f.config.root,'knowledge.sqlite'),next=join(f.config.root,'replacement.sqlite');
  seed(old,'old');seed(next,'new');
  const original=await initializeStorageKnowledge(f.config,f.state);
  const skills=await f.history.state('skills');
  const input=join(f.config.root,'baseline-input.json');await writeFile(input,JSON.stringify({baselineKDB:next}));
  await main(['config','--instance',f.config.root,'--config',input]);
  const changed=await initializeStorageKnowledge(f.config,f.state);
  assert.deepEqual(changed.baseline.collections,['new']);
  assert.deepEqual((await f.history.readRevision(changed.baseline.revision)).memory.map(r=>r.lesson),['new']);
  assert.deepEqual((await f.history.readRevision(original.baseline.revision)).memory.map(r=>r.lesson),['old']);
  assert.deepEqual(await f.history.state('skills'),skills);
  const calls=[];
  await publishStorageMemory(f.config,lesson(1),{invoke:async(tool,args)=>{
    calls.push({tool,...args});return {generation:args.revision};
  }});
  assert.deepEqual(calls.find(c=>c.tool==='adopt_memory_revision').collections,['new']);
  assert.deepEqual(calls.find(c=>c.tool==='stage_memory_revision').records.map(r=>r.lesson).sort(),['lesson 1','new']);
});

test('independent campaigns do not collide when their local outbox IDs restart',async t=>{
  const first=await fixture(t),second=await fixture(t);
  first.config.storage.campaign_id='first-campaign';second.config.storage.campaign_id='second-campaign';
  const invoke=async(_tool,args)=>({generation:args.revision});
  await publishStorageMemory(first.config,{...lesson(1),findings:'first campaign lesson'},{invoke});
  await rm(join(second.config.root,'knowledge.git'),{recursive:true,force:true});
  await cp(join(first.config.root,'knowledge.git'),join(second.config.root,'knowledge.git'),{recursive:true});
  const result=await publishStorageMemory(second.config,{...lesson(1),findings:'second campaign lesson'},{invoke});
  const records=(await second.history.readRevision(result.revision)).memory;
  assert.deepEqual(records.map(r=>r.lesson).sort(),['first campaign lesson','second campaign lesson']);
  assert.equal(new Set(records.map(r=>r.delivery_id)).size,2);
});

test('an existing pre-namespace delivery resumes without another memory commit',async t=>{
  const {lessonRecord}=await import('../../controller/storage-knowledge.mjs');
  const f=await fixture(t),item=lesson(1);
  const legacy=lessonRecord(f.config,{...item,delivery_id:'outbox-1'});
  const before=await f.history.commitMemory([legacy],{indexRevision:async revision=>revision,activate:async()=>{}});
  f.config.storage.campaign_id='campaign';
  const after=await publishStorageMemory(f.config,item,{invoke:()=>{throw Error('Already delivered');}});
  assert.equal(after.revision,before.revision);
  assert.equal((await f.history.readRevision(after.revision)).memory.length,1);
});
