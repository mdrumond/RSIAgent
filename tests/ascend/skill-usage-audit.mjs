import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {DatabaseSync} from 'node:sqlite';
import {auditSkillUsage} from '../../cli/skill-usage-audit.mjs';

async function fixture(t,{storage=true,critical=false}={}) {
  const root=await mkdtemp(join(tmpdir(),'skill-usage-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const directory=join(root,'skills','versions','baseline');await mkdir(directory,{recursive:true});
  const map={preflight:'preflight.md',optional:'optional.md',critical:'critical.md'},hashes={};
  for(const [scope,name] of Object.entries(map)) {
    const text=scope+' useful content';await writeFile(join(directory,name),text);hashes[scope]=createHash('sha256').update(text).digest('hex');
  }
  await writeFile(join(directory,'skill-map.json'),JSON.stringify(map));
  await writeFile(join(directory,'guidance.json'),JSON.stringify([{scope:'critical',critical:true},...(critical?[{scope:'optional',critical:true}]:[])]));
  await writeFile(join(root,'instance.json'),JSON.stringify({skills:{activeVersion:'baseline'}}));
  const db=new DatabaseSync(join(root,'state.sqlite'));t.after(()=>db.close());
  db.exec(`CREATE TABLE attempts(id TEXT PRIMARY KEY,outcome TEXT);
    CREATE TABLE events(sequence INTEGER PRIMARY KEY,attempt TEXT,type TEXT,data TEXT);
    CREATE INDEX events_attempt_type_sequence ON events(attempt,type,sequence);`);
  if(storage)db.exec('CREATE TABLE storage_retrievals(id INTEGER PRIMARY KEY,attempt TEXT,operation TEXT,citations TEXT)');
  const event=(id,type,data)=>db.prepare('INSERT INTO events(attempt,type,data) VALUES (?,?,?)').run(id,type,JSON.stringify(data));
  const expose=(id,scope,sha256=hashes[scope])=>event(id,'skill-read',{stage:scope,path:join(directory,map[scope]),sha256});
  for(const id of ['a','b']){db.prepare('INSERT INTO attempts VALUES (?,?)').run(id,'passed');expose(id,'preflight');}
  const retrieval=(citations,operation='mcp__kdb__search',id='a')=>{db.prepare('INSERT INTO storage_retrievals(attempt,operation,citations) VALUES (?,?,?)').run(id,operation,JSON.stringify(citations));event(id,'knowledge-retrieval-recorded',{entries:citations.length});};
  const audit=options=>auditSkillUsage({instance:root,minCompletedAttempts:2,...options});
  return {root,db,event,expose,retrieval,audit,hashes,map,directory};
}

test('optional unused content yields a searchable-only proposal while mandatory and critical instructions remain protected',async t=>{
  const f=await fixture(t);f.retrieval([]);const before=await readFile(join(f.root,'state.sqlite'));
  const result=await f.audit();
  assert.equal(result.readOnly,true);assert.equal(result.coverage.completedOpportunities,2);
  assert.deepEqual(result.demotionSuggestions.map(s=>[s.scope,s.action]),[['optional','move-to-searchable']]);
  assert.equal(result.scopes.find(s=>s.scope==='preflight').interpretation,'exposed-not-proven-useful');
  assert.equal(result.scopes.find(s=>s.scope==='critical').protected,true);
  assert.deepEqual(await readFile(join(f.root,'state.sqlite')),before);
  assert.equal(await readFile(join(f.directory,'optional.md'),'utf8'),'optional useful content');
});

test('observed injection, explicit retrieval and critical metadata each prevent a demotion proposal',async t=>{
  const f=await fixture(t);f.expose('a','optional');assert.deepEqual((await f.audit()).demotionSuggestions,[]);
  const g=await fixture(t);g.retrieval([{collection:'skills',chunk_id:'c',path:'skills/versions/baseline/optional.md',source_sha256:g.hashes.optional}]);
  assert.deepEqual((await g.audit()).demotionSuggestions,[]);
  const read=await fixture(t);read.retrieval([{evidence:[{collection:'skills',entry_id:'optional',path:'skills/versions/baseline/optional.md',sha256:read.hashes.optional}]}],'mcp__kdb__read_entry');
  assert.deepEqual((await read.audit()).demotionSuggestions,[]);
  const h=await fixture(t,{critical:true});assert.deepEqual((await h.audit()).demotionSuggestions,[]);
});

test('historical empty citation records cannot erase actual workflow retrieval evidence',async t=>{
  const f=await fixture(t);f.retrieval([]);f.event('a','knowledge-retrieved',{evidence:[{collection:'memory',entry_id:'lesson'}]});
  const result=await f.audit();
  assert.equal(result.coverage.citationLogging,'incomplete-historical');assert.equal(result.coverage.complete,false);
  assert.equal(result.retrieval.recordedCitationHits,0);assert.equal(result.retrieval.workflowHits,1);
  assert.deepEqual(result.retrieval.observedEntries,[{collection:'memory',entryId:'lesson',workflowHits:1,searchHits:0,readHits:0}]);
  assert.deepEqual(result.demotionSuggestions,[]);
});

test('successful native search and read records retain separate counts and stable item identity',async t=>{
  const f=await fixture(t);
  f.retrieval([{collection:'memory',chunk_id:'lesson'}]);
  f.retrieval([{text_sha256:'hash',evidence:[{collection:'memory',entry_id:'lesson'}]}],'mcp__kdb__read_entry');
  const result=await f.audit();
  assert.equal(result.retrieval.recordedSearches,1);assert.equal(result.retrieval.recordedReads,1);
  assert.deepEqual(result.retrieval.observedEntries,[{collection:'memory',entryId:'lesson',workflowHits:0,searchHits:1,readHits:1}]);
  assert.equal(result.retrieval.unseenInventory,'unknown');
});

test('absent instrumentation, unresolved recording and mismatched immutable content fail closed',async t=>{
  const f=await fixture(t,{storage:false});assert.equal((await f.audit()).coverage.complete,false);assert.deepEqual((await f.audit()).demotionSuggestions,[]);
  const g=await fixture(t);g.event('a','knowledge-retrieval-unavailable',{});assert.deepEqual((await g.audit()).demotionSuggestions,[]);
  const h=await fixture(t);h.expose('a','preflight','wrong');assert.deepEqual((await h.audit()).demotionSuggestions,[]);
});

for(const limit of [{maxAttempts:1},{maxEventsPerAttempt:1},{maxRetrievals:1}])test('bounded windows are reported and never imply complete zero usage '+JSON.stringify(limit),async t=>{
  const f=await fixture(t);f.expose('a','preflight');f.retrieval([]);f.retrieval([]);
  const result=await f.audit(limit);assert.equal(result.coverage.complete,false);assert.deepEqual(result.demotionSuggestions,[]);
});

test('incomplete attempts are not counted as finished exposure opportunities',async t=>{
  const f=await fixture(t);f.db.prepare("UPDATE attempts SET outcome=NULL WHERE id='b'").run();
  const result=await f.audit();assert.equal(result.coverage.completedOpportunities,1);assert.deepEqual(result.demotionSuggestions,[]);
});

test('shared skill files inherit protection and exposure from their mandatory scope',async t=>{
  const f=await fixture(t);f.map.optional='preflight.md';
  await writeFile(join(f.directory,'skill-map.json'),JSON.stringify(f.map));f.retrieval([]);
  const result=await f.audit(),optional=result.scopes.find(s=>s.scope==='optional');
  assert.equal(optional.protected,true);assert.equal(optional.exposedAttempts,2);
  assert.deepEqual(result.demotionSuggestions,[]);
});
