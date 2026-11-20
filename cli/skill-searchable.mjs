/** Searchable guidance is memory evidence, with a different gate from skill activation. */
import assert from 'node:assert/strict';
import {createHash,randomUUID} from 'node:crypto';
import {link,mkdir,readFile,realpath,unlink,writeFile} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {basename,dirname,join,relative,resolve} from 'node:path';
import {ensureKnowledgeService} from './knowledge.mjs';
import {State} from '../controller/state.mjs';
import {invokeStorageKdb,publishStorageMemory} from '../controller/storage-knowledge.mjs';
import {pilotWinner,validatePilotManifest,validatePilotTrial} from './skill-failure-pilot.mjs';

const read=path=>readFile(path,'utf8');
const json=path=>read(path).then(JSON.parse);
const optional=async path=>{try{return await json(path);}catch(error){if(error.code!=='ENOENT')throw error;return null;}};
const digest=value=>createHash('sha256').update(value).digest('hex');
const text=value=>typeof value==='string'&&value.trim();
const isHash=value=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
const objectHash=value=>digest(JSON.stringify(value));
const candidateId=value=>typeof value==='string'&&/^[a-z0-9][a-z0-9-]{0,127}$/.test(value);
const within=(root,path)=>{
 const part=relative(root,path);
 return part&&!part.startsWith('../')&&part!=='..';
};
const refs=async paths=>Promise.all(paths.map(async file=>{
 const path=await realpath(file),hash=createHash('sha256');
 for await(const chunk of createReadStream(path))hash.update(chunk);
 return {path,sha256:hash.digest('hex')};
}));
async function freeze(path,value){
 const data=JSON.stringify(value,null,2)+'\n',tmp=path+'.'+randomUUID();
 await writeFile(tmp,data,{mode:0o600,flag:'wx'});
 try{
  try{await link(tmp,path);}catch(error){if(error.code!=='EEXIST')throw error;}
  assert.equal(await read(path),data,'Searchable evidence identity changed');
 }finally{await unlink(tmp);}
}
const score=rows=>rows?.map(r=>({correct:r.correct,exact:r.mechanismMatch===true})).reduce((total,r)=>({
 passes:total.passes+(r.correct===true),exactFailures:total.exactFailures+(r.correct===false&&r.exact),
 unscored:total.unscored+(r.correct===null),
}),{passes:0,exactFailures:0,unscored:0})??null;
function arm(rows,task){
 if(rows===undefined)return undefined;
 assert(Array.isArray(rows)&&rows.length===5&&rows.every(r=>r?.infrastructure!==true),'Five settled non-infrastructure trials required');
 return rows.map(row=>validatePilotTrial(structuredClone(row),task));
}
function transferText(row){
 const show=score=>score?score.passes+'/5 passes ('+score.unscored+' unscored)':'not tested';
 return row.id+': current '+show(row.baseline)+', reviewed '+show(row.candidate)+(row.disposition?' ('+row.disposition+')':'');
}
export function searchableMemoryFingerprint(db){
 const table=db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='searchable_skill_admissions'").get();
 if(!table)return null;
 const ids=db.prepare('SELECT id FROM searchable_skill_admissions ORDER BY id').all().map(row=>row.id);
 return ids.length?objectHash(ids):null;
}
function recordAdmission(config,receipt){
 const state=new State(join(config.root,'state.sqlite'));
 try{
  state.db.exec('CREATE TABLE IF NOT EXISTS searchable_skill_admissions (id TEXT PRIMARY KEY, receipt TEXT NOT NULL)');
  const data=JSON.stringify(receipt);
  state.db.prepare('INSERT OR IGNORE INTO searchable_skill_admissions VALUES (?,?)').run(receipt.id,data);
  assert.equal(state.db.prepare('SELECT receipt FROM searchable_skill_admissions WHERE id=?').get(receipt.id).receipt,data,'Persisted searchable receipt changed');
 }finally{state.close();}
}
async function publishSearchable({config,review,scope,context,provenance,signal,recoverOnly=false},dependencies){
 assert(['a3','a5'].includes(config.target),'Searchable guidance requires an explicit A3 or A5 binding');
 assert(text(scope)&&scope.length<=300&&text(context)&&context.length<=1700,'Named bounded scope and evaluation context required');
 assert(text(review.instruction)&&review.instruction.length<=1500&&isHash(review.approval),'Exact reviewed instruction and approval required');
 signal?.throwIfAborted();
 const id=objectHash({format:1,target:config.target,scope,context,review:{instruction:review.instruction,approval:review.approval}});
 const directory=join(config.root,'skills','searchable',id);
 const evidence=join(directory,'evidence.json');
 const admission=join(directory,'admission.json');
 const evidenceRecord={format:1,id,target:config.target,scope,context,review,provenance};
 const identity={admitted:true,id,evidence,deliveryId:'searchable-skill-'+id};
 if(recoverOnly){
  const saved=await optional(admission);
  if(saved===null)return {admitted:false,reason:'no-prior-admission'};
  if(!config.storage)assert(text(saved.collection)&&text(saved.entryId),'Invalid searchable KDB receipt');
  assert.deepEqual(saved,{...identity,...(!config.storage?{collection:saved.collection,entryId:saved.entryId}:{})},'Persisted searchable receipt changed');
  assert.equal(await read(evidence),JSON.stringify(evidenceRecord,null,2)+'\n','Searchable evidence identity changed');
  signal?.throwIfAborted();
  recordAdmission(config,saved);
  return saved;
 }
 await mkdir(directory,{recursive:true,mode:0o700});
 await freeze(evidence,evidenceRecord);
 const records=await refs([evidence]);
 const lesson=[
  'Searchable guidance evidence. This note does not activate a skill.',
  'searchable-skill:'+id,'Scope: '+scope,'Evaluation: '+context,
  'Exact reviewed guidance:',review.instruction,
 ].join('\n\n');
 const {ensure=ensureKnowledgeService,invoke=invokeStorageKdb,publish=publishStorageMemory}=dependencies;
 const url=await ensure(config,{signal});
 const written=config.storage
  ?await publish(config,{id,delivery_id:'searchable-skill-'+id,lesson,kind:'good',target:config.target,evidence:records},{url,signal})
  :await invoke(config,url,'record_lesson',{lesson,outcome:'good',applicability:config.target,evidence:records,attempt_id:'searchable-skill-'+id},{signal});
 assert(written&&(config.storage?text(written.revision):text(written.collection)&&text(written.entry_id)),'Searchable KDB admission returned no stable identity');
 const receipt={...identity,...(!config.storage?{collection:written.collection,entryId:written.entry_id}:{})};
 await freeze(admission,receipt);
 recordAdmission(config,receipt);
 return receipt;
}

function completeOperation(report,directory,id,kind){
 const operation=report.operations?.[id];
 assert(operation?.status==='complete'&&operation.kind===kind&&operation.result!==undefined,'Missing complete '+kind+' operation: '+id);
 assert.equal(resolve(operation.directory),resolve(directory,'operations',id),'Operation directory changed: '+id);
 return operation;
}
function boundTrials(report,directory,family,task,phase,lane,rows){
 const trials=arm(rows,task);
 return trials?.map((row,index)=>{
  const id=[phase,family.id,task.id,lane,index].join('--');
  const op=completeOperation(report,directory,id,'trial');
  assert.deepEqual(row,validatePilotTrial(op.result,task),'Trial operation result changed: '+id);
  return row;
 });
}
async function qualification(report,directory,family,task){
 const key=family.id+'--'+task.id,retained=report.qualifications?.[key];
 // A never-scheduled transfer has no qualification receipt.
 if(!retained)return null;
 assert(task.controls?.calibration?.files,'Focused fixture has no calibration source');
 assert.equal(report.checkerHashes?.[key],digest(await readFile(task.checker)),'Frozen checker changed: '+task.id);
 assert.deepEqual(retained,completeOperation(report,directory,'qualify--'+key,'qualify').result,'Qualification operation result changed');
 return retained;
}
const qualified=receipt=>receipt?.qualified===true&&receipt.referencePassed===true&&receipt.negativeFailed===true&&receipt.calibrationPassed===true;
async function verifyScopedReport({directory,result,manifest,config}){
 assert.equal(manifest.mode,'focused-correctness','Searchable admission requires a focused correctness evaluation');
 assert.equal(result.manifestSha256,objectHash(manifest),'Frozen focused pilot manifest changed');
 assert(Object.values(result.operations??{}).every(op=>op.status==='complete'),'Pilot operations remain unresolved');
 const family=manifest.families.find(row=>row.id===result.selectedFamily),winner=result.winner;
 assert(family&&family.development.target===config.target&&family.transfer.every(task=>task.target===config.target),'Frozen target binding mismatch');
 assert.deepEqual(await json(join(directory,'winner.json')),winner,'Retained winner changed');
 const developmentQualification=await qualification(result,directory,family,family.development);
 assert(qualified(developmentQualification),'Development controls did not qualify');
 const base=boundTrials(result,directory,family,family.development,'screen','baseline',result.screens?.[family.id]);
 assert(base&&score(base).exactFailures>=1&&score(base).passes<=3,'Development baseline did not qualify');
 assert.deepEqual(arm(result.development?.baseline,family.development),base,'Development must reuse the screened baseline');
 assert(result.candidates?.length===3&&new Set(result.candidates.map(c=>c.id)).size===3,'Three distinct reviewed candidates required');
 const checked={baseline:base};
 for(const c of result.candidates){
  assert(candidateId(c.id),'Invalid reviewed candidate identifier');
  checked[c.id]=boundTrials(result,directory,family,family.development,'development',c.id,result.development?.[c.id]);
  assert(checked[c.id],'Missing development candidate comparison');
 }
 assert.deepEqual(pilotWinner(checked,result.candidates),winner,'Development winner changed');
 const transfers=[];
 for(const task of family.transfer){
  const controls=await qualification(result,directory,family,task),rows=result.confirmation?.[task.id];
  const baseline=boundTrials(result,directory,family,task,'transfer-screen','baseline',rows?.baseline);
  const treatment=boundTrials(result,directory,family,task,'confirmation',winner.id,rows?.[winner.id]);
  if(baseline||treatment)assert(qualified(controls)&&baseline,'Executed transfer missing qualified controls or baseline');
  if(treatment)assert(score(baseline).exactFailures>=1&&score(baseline).passes<=3,'Transfer baseline failed admission');
  const disposition=!controls?'not qualified before completion':!qualified(controls)?'controls did not qualify':!baseline?'no baseline trial':!treatment?'no candidate trial':'';
  const improved=Boolean(treatment&&pilotWinner({baseline,[winner.id]:treatment},[winner]));
  transfers.push({id:task.id,baseline:score(baseline),candidate:score(treatment),disposition,improved});
 }
 assert(!transfers.every(row=>row.improved),'Both transfers improved; searchable fallback is inapplicable');
 return {family,base,developmentQualification,transfers};
}

/** Admit a settled development winner with evidence limiting its transfer claim. */
export async function admitSearchableSkill({config,directory,result,item,signal,recoverOnly=false},dependencies={}){
 if(result?.status!=='complete'||result.promising!==false||!result.winner)
  return {admitted:false,reason:'No settled scoped candidate'};
 const root=await realpath(directory),pilot=join(root,'pilot.json'),manifestPath=join(root,'manifest.json');
 assert.deepEqual(await json(pilot),result,'Only the exact persisted pilot can admit a note');
 const manifest=validatePilotManifest(await json(manifestPath)),winner=result.winner;
 const {family,developmentQualification,transfers}=await verifyScopedReport({directory:root,result,manifest,config});
 const changes=winner.proposal?.changes;
 assert.equal(changes?.length,1,'One approved guidance change required');
 assert.equal(changes[0].action,'upsert');assert.equal(changes[0].text,winner.instruction);assert.equal(changes[0].scope,winner.scope);
 assert(candidateId(winner.version)&&winner.qa?.verdict==='approved'&&isHash(winner.qa.approval),'Approved candidate version identity invalid');
 const verify=dependencies.verify??(await import('./skill-failure-candidates.mjs')).verifyFocusedSearchableApproval;
 assert(typeof verify==='function','Focused approval verification is unavailable');
 const verified=await verify({config,item,result,manifest,family,directory:root,qualification:developmentQualification});
 assert(text(verified),'Verified focused approval provenance is missing');
 const design=await realpath(verified);
 assert(within(root,design),'Verified design is outside the pilot');
 const candidateDirectory=join(design,winner.id),qaDirectory=join(candidateDirectory,'qa');
 const initialQa=await optional(join(qaDirectory,'qa.json'));
 const qaFile=join(qaDirectory,initialQa?.format===2?'qa.json':'qa-v2.json');
 const sources=[join(design,'approved-design.json'),join(candidateDirectory,'approved-candidate.json'),qaFile];
 return publishSearchable({config,signal,recoverOnly,review:{instruction:winner.instruction,approval:winner.qa.approval},
  scope:config.target+' '+winner.scope+' guidance for '+family.id,
  context:'Development '+score(result.development.baseline).passes+'/5 to '+score(result.development[winner.id]).passes+'/5. Transfer confirmation did not pass: '+transfers.map(transferText).join('; ')+'.',
  provenance:{kind:'settled-failure-pilot',candidateId:winner.id,baseline:manifest.identities.skills??null,
   artifacts:await refs([pilot,manifestPath,...sources])}},dependencies);
}

/** Explicit import of a retained native writer/independent-QA approval. */
export async function importSearchableSkill({config,approvalDirectory,scope,context,evidenceFiles,signal},dependencies={}){
 assert.equal(config.target,'a5','Retained experimental receipts require the A5 campaign binding');
 assert(Array.isArray(evidenceFiles)&&evidenceFiles.length>0&&evidenceFiles.length<=16,'Historical admission requires retained evaluation artifacts');
 const root=await realpath(approvalDirectory),approved=join(root,'approved.txt'),receipt=join(root,'approved');
 const [instruction,record]=await Promise.all([read(approved),json(receipt)]);
 assert.equal(record.verdict,'approve','Historical QA did not approve');
 assert.equal(record.guidance,instruction,'Historical text differs from approval');
 assert.equal(record.sha256,digest(instruction),'Historical instruction identity changed');
 const writer=await realpath(record.writer),qa=await realpath(record.qa);
 assert(within(root,writer)&&within(root,qa)&&writer!==qa&&dirname(writer)!==dirname(qa),'Distinct native writer and QA sessions required');
 const writerSubmission=join(dirname(writer),'submission.json'),qaInput=join(dirname(qa),'input.json'),qaSubmission=join(dirname(qa),'submission.json');
 assert.equal((await json(writerSubmission)).text,instruction,'Historical writer submitted different text');
 assert.equal((await json(qaInput)).candidate,instruction,'Historical QA reviewed different text');
 assert.equal((await json(qaSubmission)).verdict,'approve','Historical review submitted a different verdict');
 const documents=[approved,receipt,writer,qa,writerSubmission,qaInput,qaSubmission,...evidenceFiles];
 return publishSearchable({config,signal,review:{instruction,approval:record.sha256},scope,context,
  provenance:{kind:'retained-native-approval',receipt:basename(receipt),artifacts:await refs(documents)}},dependencies);
}
