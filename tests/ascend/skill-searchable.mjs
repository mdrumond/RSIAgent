import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {cp,mkdir,mkdtemp,readFile,readdir,rm,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {admitSearchableSkill,importSearchableSkill,searchableMemoryFingerprint} from '../../cli/skill-searchable.mjs';
import {ensureKnowledgeService} from '../../cli/knowledge.mjs';
import {releaseKnowledgeService} from '../../cli/campaign-memory.mjs';
import {invokeStorageKdb,publishStorageMemory} from '../../controller/storage-knowledge.mjs';
import {CampaignStorage} from '../../controller/storage.mjs';
import {State} from '../../controller/state.mjs';
import {searchableExposure} from '../../controller/searchable-exposure.mjs';

const hash=text=>createHash('sha256').update(text).digest('hex');
const save=(file,value)=>writeFile(file,JSON.stringify(value));
const instruction='Vector reduction mask uses the validated keyword-only argument.';
const target='a5';
const pass=(correct,match=false)=>({correct,mechanismMatch:match,caseResults:[{id:'case',correct}],sourceSha256:'c'.repeat(64),artifacts:[],handles:[]});
async function fixture(t,{storage=false}={}){
 const root=await mkdtemp(join(tmpdir(),'searchable-skill-'));
 const state=new State(join(root,'state.sqlite'));
 const config={format:1,root,target,skills:{activeVersion:'retained'},python:process.env.PYTHON??'python3',embedding_factory:'kdb_fixture:Embeddings'};
 if(storage){config.storage={format:1};await new CampaignStorage(state,root).initialize();}
 t.after(async()=>{try{await releaseKnowledgeService(config,{timeoutMs:5000});}finally{state.close();await rm(root,{recursive:true,force:true});}});
 const initial=join(root,'skills','versions','retained');
 await mkdir(initial,{recursive:true});
 await writeFile(join(initial,'implementation.md'),'Existing implementation skill');
 await save(join(initial,'skill-map.json'),{implementation:'implementation.md'});
 await save(join(root,'instance.json'),config);
 await cp(new URL('./kdb_fixture.py',import.meta.url),join(root,'kdb_fixture.py'));
 return {root,config,state};
}
async function historical(root){
 const approvalDirectory=join(root,'historical'),writer=join(approvalDirectory,'write-1','writer.jsonl'),qa=join(approvalDirectory,'qa-1','qa.jsonl');
 await mkdir(join(approvalDirectory,'write-1'),{recursive:true});await mkdir(join(approvalDirectory,'qa-1'));
 await writeFile(writer,'{"type":"session"}\n');await writeFile(qa,'{"type":"session"}\n');
 await writeFile(join(approvalDirectory,'approved.txt'),instruction);
 await save(join(approvalDirectory,'approved'),{verdict:'approve',guidance:instruction,sha256:hash(instruction),writer,qa});
 await save(join(approvalDirectory,'write-1','submission.json'),{text:instruction});
 await save(join(approvalDirectory,'qa-1','input.json'),{candidate:instruction});
 await save(join(approvalDirectory,'qa-1','submission.json'),{verdict:'approve'});
 const outcome=join(approvalDirectory,'transfer.json');await save(outcome,{transfer:{baseline:1,candidate:1}});
 return {approvalDirectory,scope:'A5 SIMD reduction compile behavior',context:'Development improved; an independent masked transfer tied 1/5.',evidenceFiles:[outcome]};
}
async function pilot(root){
 const directory=join(root,'pilot'),design=join(directory,'operations','design--simd'),candidatePath=join(design,'principle-a'),qa=join(candidatePath,'qa');
 await mkdir(qa,{recursive:true});
 const proposal={changes:[{action:'upsert',id:'reviewed-vector-reduction',scope:'implementation',text:instruction}]};
 const winner={id:'principle-a',version:'candidate-version',scope:'implementation',instruction,proposal,text:instruction,qa:{verdict:'approved',approval:hash('review-evidence')}};
 const checker=join(directory,'oracle.py'),checkerBytes='none';
 await writeFile(checker,checkerBytes);
 const task=id=>({id,target,caseIds:['case'],checker,controls:{calibration:{files:{'kernel.py':'reference'}}}});
 const dev=task('development'),layout=task('layout'),tiles=task('tiles');
 const manifest={mode:'focused-correctness',baseline:'retained baseline',identities:{model:'model',runtime:'a5',memory:'snapshot'},families:[{id:'simd',development:dev,transfer:[layout,tiles]}]};
 const controls=Array.from({length:5},()=>pass(false,true));
 const successful=Array.from({length:5},()=>pass(true));
 const candidates=[winner,{id:'principle-b',text:'different A'},{id:'principle-c',text:'different B'}];
 const qualification={qualified:true,referencePassed:true,negativeFailed:true,calibrationPassed:true};
 const operations={};
 const register=(id,kind,result)=>{operations[id]={status:'complete',kind,directory:join(directory,'operations',id),result};};
 register('design--simd','design',candidates);
 for(const unit of [dev,layout,tiles])register('qualify--simd--'+unit.id,'qualify',qualification);
 const cells=(phase,task,arm,rows)=>rows.forEach((row,i)=>register([phase,'simd',task.id,arm,i].join('--'),'trial',row));
 cells('screen',dev,'baseline',controls);
 cells('development',dev,'principle-a',successful);
 cells('development',dev,'principle-b',controls);
 cells('development',dev,'principle-c',controls);
 cells('transfer-screen',layout,'baseline',controls);cells('confirmation',layout,'principle-a',controls);
 cells('transfer-screen',tiles,'baseline',successful);
 const result={status:'complete',promising:false,selectedFamily:'simd',winner,candidates,operations,manifestSha256:hash(JSON.stringify(manifest)),
  qualifications:Object.fromEntries([dev,layout,tiles].map(unit=>['simd--'+unit.id,qualification])),
  checkerHashes:Object.fromEntries([dev,layout,tiles].map(unit=>['simd--'+unit.id,hash(checkerBytes)])),
  screens:{simd:controls},
  development:{baseline:controls,'principle-a':successful,'principle-b':controls,'principle-c':controls},
  confirmation:{layout:{baseline:controls,'principle-a':controls},tiles:{baseline:successful}}};
 await save(join(directory,'manifest.json'),manifest);await save(join(directory,'pilot.json'),result);await save(join(directory,'winner.json'),winner);
 await save(join(design,'approved-design.json'),{identity:'exact-design',candidates});
 await save(join(candidatePath,'approved-candidate.json'),{identity:'exact-design',candidate:winner});
 await save(join(qa,'qa.json'),{format:2,verdict:'approved',approval:winner.qa.approval,proposal,rounds:[{proposal,review:{verdict:'approved'}}]});
 const verify=async()=>design;
 return {directory,result,verify};
}

test('settled scoped winner publishes exact reviewed bytes with transfer boundaries',async t=>{
 const {config,root,state}=await fixture(t);
 const data=await pilot(root),written=[];
 const dependencies={verify:data.verify,ensure:async()=> 'fixture',invoke:async(config,url,tool,args)=>{
  written.push({tool,args});return {collection:'stable-collection',entry_id:'stable-entry'};
 }};
 const active=await readFile(join(root,'instance.json'),'utf8');
 assert.equal(searchableMemoryFingerprint(state.db),null);
 const first=await admitSearchableSkill({config,...data},dependencies);
 assert.equal(first.admitted,true);
 assert.deepEqual(await admitSearchableSkill({config,...data},dependencies),first);
 assert.deepEqual(written[0],written[1]);
 assert.equal(written[0].tool,'record_lesson');
 assert(written[0].args.lesson.endsWith(instruction));
 assert(written[0].args.lesson.includes('no candidate trial')||written[0].args.lesson.includes('baseline-ceiling'));
 assert.equal(written[0].args.applicability,'a5');
 assert.equal(state.db.prepare('SELECT count(*) n FROM searchable_skill_admissions').get().n,1);
 assert.equal(searchableMemoryFingerprint(state.db),hash(JSON.stringify([first.id])));
 assert.equal(await readFile(join(root,'instance.json'),'utf8'),active);
});

test('refuses changed fixture, invalid controls or unresolved operations before QA and indexing',async t=>{
 const {config,root}=await fixture(t);
 const data=await pilot(root);
 const blocked={ensure:()=>assert.fail('Must not reach publication'),verify:()=>assert.fail('Must not reach QA verification')};
 assert.equal((await admitSearchableSkill({config,...data,result:{...data.result,status:'inconclusive'}},blocked)).admitted,false);
 const file=join(data.directory,'pilot.json');
 const invalid=structuredClone(data.result);
 invalid.qualifications['simd--development'].calibrationPassed=false;
 invalid.operations['qualify--simd--development'].result.calibrationPassed=false;
 await save(file,invalid);
 await assert.rejects(admitSearchableSkill({config,...data,result:invalid},blocked),/Development controls did not qualify/);
 const unresolved=structuredClone(data.result);
 unresolved.operations['screen--simd--development--baseline--0'].status='running';
 await save(file,unresolved);
 await assert.rejects(admitSearchableSkill({config,...data,result:unresolved},blocked),/operations remain unresolved/);
 const corrupted=structuredClone(data.result);
 corrupted.operations['screen--simd--development--baseline--0'].result=pass(true);
 await save(file,corrupted);
 await assert.rejects(admitSearchableSkill({config,...data,result:corrupted},blocked),/Trial operation result changed/);
});

test('recover-only completes an existing admission without starting KDB or creating a receipt',async t=>{
 const {config,root,state}=await fixture(t),data=await pilot(root);
 const recover=()=>admitSearchableSkill({config,...data,recoverOnly:true},{
  verify:data.verify,ensure:()=>assert.fail('Recovery must not reach KDB'),
 });
 assert.deepEqual(await recover(),{admitted:false,reason:'no-prior-admission'});
 await assert.rejects(readFile(join(root,'skills','searchable','admission.json')), {code:'ENOENT'});
 const first=await admitSearchableSkill({config,...data},{verify:data.verify,ensure:async()=> 'fixture',
  invoke:async()=>({collection:'stable-collection',entry_id:'stable-entry'}),
 });
 state.db.prepare('DELETE FROM searchable_skill_admissions').run();
 assert.equal(searchableMemoryFingerprint(state.db),null);
 assert.deepEqual(await recover(),first);
 assert.equal(searchableMemoryFingerprint(state.db),hash(JSON.stringify([first.id])));
 await save(first.evidence,{changed:true});
 await assert.rejects(recover(),/Searchable evidence identity changed/);
});

test('historical receipt must bind the retained QA packet to the exact writer bytes',async t=>{
 const {config,root}=await fixture(t),data=await historical(root);
 await save(join(data.approvalDirectory,'qa-1','input.json'),{candidate:'Other draft'});
 await assert.rejects(importSearchableSkill({config,...data},{ensure:()=>assert.fail('Must not publish mismatched approval')}),/Historical QA reviewed different text/);
});

test('storage-backed admission is visible to scoped KDB retrieval and replay does not activate guidance', {timeout:30000}, async t=>{
 const {config,root,state}=await fixture(t,{storage:true}),data=await historical(root);
 const active=await readFile(join(root,'instance.json'),'utf8');
 const first=await importSearchableSkill({config,...data});
 const url=await ensureKnowledgeService(config);
 const generation=state.db.prepare("SELECT generation FROM storage_revisions WHERE name='memory'").get().generation;
 const invoke=(tool,args)=>invokeStorageKdb(config,url,tool,args);
 const search=await invoke('search',{query:'Vector reduction mask',target:'a5',generation});
 const found=search.result.find(entry=>entry.text.includes('searchable-skill:'+first.id));
 assert(found,'New generation must return the exact scoped note to ordinary KDB search');
 const opened=await invoke('read_entry',{collection:found.collection,entry_id:found.chunk_id,generation});
 assert(opened.text.includes(instruction));
 assert.deepEqual(searchableExposure([opened],{collection:found.collection,entry_id:found.chunk_id}),[{id:first.id,collection:found.collection,entryId:found.chunk_id}]);
 assert.equal((await invoke('search',{query:'Vector reduction mask',target:'a3',generation})).result.length,0);
 // A later unrelated memory delivery must not make replay rename or duplicate the prior admission.
 await publishStorageMemory(config,{id:'other-note',lesson:'Unrelated test memory',kind:'good',target:'a5',evidence:[{path:data.evidenceFiles[0]}]},{url});
 assert.deepEqual(await importSearchableSkill({config,...data}),first);
 assert.equal(state.db.prepare('SELECT count(*) n FROM searchable_skill_admissions').get().n,1);
 assert.equal(await readFile(join(root,'instance.json'),'utf8'),active);
 assert.deepEqual(await readdir(join(root,'skills','versions')),['retained']);
 assert.equal(await readFile(join(root,'skills','versions','retained','implementation.md'),'utf8'),'Existing implementation skill');
});
