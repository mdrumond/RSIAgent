/** Evidence-only failure inventory and learning-request admission. */
import {DatabaseSync} from 'node:sqlite';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
const digest=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const has=(db,name)=>Boolean(db.prepare("SELECT 1 FROM sqlite_master WHERE name=?").get(name));
const negativeVerdict=data=>data&&typeof data==='object'&&(data.passed===false||data.correct===false||
 ['correctness','performance','verdict','result','verification'].some(key=>negativeVerdict(data[key])));
const failed=(type,data)=>{
 const actual=data.exitCode??data.exit_code,expected=data.expected_exit;
 if(expected===1&&actual===1)return false;
 return expected!==undefined&&actual!==undefined&&actual!==expected||negativeVerdict(data)||
  /error|failed|rejected|unresolved|blocked/.test(type)||['failure-recorded','project-invalid'].includes(type)||type==='verification'&&actual>0;
};
const normal=text=>String(text??'').replace(/[a-f0-9]{8}-[a-f0-9-]{27,}/gi,'<id>').replace(/\/[^\s'"<>]+/g,'<path>').slice(0,1500);
export function diagnosticIdentity(occurrence){
 const stable=value=>Array.isArray(value)?value.map(stable):value&&typeof value==='object'?
  Object.fromEntries(Object.entries(value).sort(([a],[b])=>a.localeCompare(b)).filter(([key])=>
   !/elapsed|duration|monotonic|timestamp|fingerprint|^handle$|^operation$|^owner$|^created$/i.test(key)).map(([key,item])=>[key,stable(item)])):
  typeof value==='string'?value.replace(/[a-f0-9]{8}-[a-f0-9-]{27,}/gi,'<id>').replace(/\/[^\s'"<>]+/g,'<path>'):value;
 return digest([stable(occurrence.data??{}),(occurrence.artifacts??[]).map(a=>a.sha256)]);
}
export async function inventoryFailures(sources) {
 const groups=new Map(),campaigns=[];
 for(const source of sources){
  const root=resolve(typeof source==='string'?source:source.root),config=JSON.parse(await readFile(join(root,'instance.json'),'utf8'));
  const sourceId=typeof source==='object'&&source.id||digest(root).slice(0,12);
  if(campaigns.some(c=>c.id===sourceId))throw Error('Duplicate review source ID');
  let artifactIndex={};try{artifactIndex=JSON.parse(await readFile(join(root,'evidence-files.json'),'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
  const artifacts=value=>{if(typeof value==='string')return artifactIndex[value]?.path?[{original:value,...artifactIndex[value],path:join(root,artifactIndex[value].path)}]:[];return value&&typeof value==='object'?Object.values(value).flatMap(artifacts):[];};
  const db=new DatabaseSync(join(root,'state.sqlite'),{readOnly:true});
  try{
   const projects=new Map(db.prepare('SELECT id,spec FROM projects').all().map(p=>[p.id,JSON.parse(p.spec)]));
   const attempts=new Map(db.prepare('SELECT id,project,outcome FROM attempts').all().map(a=>[a.id,a]));
   const jobs=new Map();for(const a of attempts.values())try{jobs.set(a.id,JSON.parse(await readFile(join(root,'attempts',a.id,'job.json'),'utf8')));}catch(e){if(e.code!=='ENOENT')throw e;}
   const reads=new Map([...attempts.keys()].map(id=>[id,db.prepare("SELECT type,data FROM events WHERE attempt=? AND type='skill-read' ORDER BY sequence").all(id).map(e=>JSON.parse(e.data))]));
   const related=new Map();
   const verdictPaths=['passed','correct','correctness.passed','performance.passed','result.passed','result.correct','verdict.passed','verdict.correct'];
   const verdictRows=db.prepare(`SELECT sequence,attempt,type,data,created FROM events WHERE attempt IS NOT NULL AND
    (type IN ('baseline-preflight-completed','project-invalid','verification') OR ${verdictPaths.map(path=>`json_type(data,'$.${path}')='false'`).join(' OR ')}) ORDER BY sequence`).all();
   for(const row of verdictRows){const data=JSON.parse(row.data);if(!failed(row.type,data))continue;
    const records=related.get(row.attempt)??[];records.push({reference:`${sourceId}:events:${row.sequence}`,type:row.type,data,artifacts:artifacts(data)});related.set(row.attempt,records);
   }
   const campaign={id:sourceId,root,target:config.target??'unknown',activeVersion:config.skills?.activeVersion??'baseline',attempts:attempts.size};
   campaigns.push(campaign);
   const add=(row,journal)=>{
    const data=typeof row.data==='string'?JSON.parse(row.data):row.data,attempt=attempts.get(row.attempt),spec=projects.get(attempt?.project);
    const message=data.message??data.error??data.reason??data.result?.error??row.type;
    const linked=(related.get(row.attempt)??[]).filter(d=>d.reference!==`${sourceId}:${journal}:${row.sequence}`);
    const artifactRefs=artifacts(data),linkedArtifacts=[...new Map(linked.flatMap(d=>d.artifacts).map(a=>[a.path,a])).values()];
    const diagnostic=diagnosticIdentity({data,relatedDiagnosticRefs:linked.map(d=>d.reference),artifacts:[...artifactRefs,...linkedArtifacts.slice(0,6)]});
    const key=digest([sourceId,row.type,normal(message),diagnostic,
     ['verification','failed-outcome'].includes(row.type)||message==='Actor completed without acceptance'?attempt?.project??row.draft:null]),reference=`${sourceId}:${journal}:${row.sequence}`;
    const group=groups.get(key)??{id:key,sourceId,target:campaign.target,type:row.type,message:normal(message),occurrences:[],linkedDiagnostics:[]};
    group.linkedDiagnostics=[...new Map([...group.linkedDiagnostics,...linked].map(d=>[d.reference,d])).values()];
    group.occurrences.push({reference,attemptId:row.attempt??null,draftId:row.draft??null,projectId:attempt?.project??null,
     phase:row.phase??jobs.get(row.attempt)?.mode??spec?.phase??config.mode,kind:jobs.get(row.attempt)?.kind??null,skillVersion:jobs.get(row.attempt)?.config?.skills?.activeVersion??jobs.get(row.attempt)?.skillVersion??null,skillReads:reads.get(row.attempt)??[],created:row.created??null,data,relatedDiagnosticRefs:linked.map(d=>d.reference),artifacts:[...artifactRefs,...linkedArtifacts.slice(0,6)]});groups.set(key,group);
   };
   for(const journal of ['events','generation_events'])if(has(db,journal)){
    let cursor=0;while(true){
     const page=db.prepare(`SELECT * FROM ${journal} WHERE sequence>? ORDER BY sequence LIMIT 500`).all(cursor);
     if(!page.length)break;
     for(const row of page){const data=JSON.parse(row.data);if(failed(row.type,data))add(row,journal);}
     cursor=page.at(-1).sequence;
    }
   }
   if(has(db,'generation_drafts'))for(const row of db.prepare('SELECT id,data FROM generation_drafts').all()){
    const draft=JSON.parse(row.data);if(['exhausted','invalid','failed'].includes(draft.status))add({sequence:row.id,draft:row.id,type:'generation-'+draft.status,phase:draft.phase,data:{message:draft.error??'Curriculum draft elapsed allocation exhausted',topic:draft.topic}},'generation_drafts');
   }
   for(const a of attempts.values())if(a.outcome&&a.outcome!=='passed'&&![...groups.values()].some(g=>g.occurrences.some(o=>o.attemptId===a.id)))
    add({sequence:a.id,attempt:a.id,type:'failed-outcome',data:{outcome:a.outcome,message:'Failed outcome; no recorded cause'}},'attempts');
  }finally{db.close();}
 }
 return {format:1,campaigns,groups:[...groups.values()]};
}
export async function guidanceEntries(root,version){
 const directory=join(root,'skills','versions',version);
 try{return JSON.parse(await readFile(join(directory,'guidance.json'),'utf8'));}
 catch(e){if(e.code!=='ENOENT')throw e;
  const map=JSON.parse(await readFile(join(directory,'skill-map.json'),'utf8'));
  return Promise.all(Object.entries(map).map(async([scope,file])=>({id:scope+'-baseline',scope,text:await readFile(join(directory,file),'utf8'),critical:true,negative:false})));
 }
}
export function validateClassification(value,group,entries){
 if(!value||typeof value!=='object')throw Error('Submit a classification object');
 if(!['actor','harness','specification','provider','unknown'].includes(value.cause))throw Error('cause must be actor, harness, specification, provider or unknown');
 if(!value.reason?.trim())throw Error('reason must explain the observed evidence; stage labels alone do not establish ownership');
 if(!['covered','uncovered','unknown'].includes(value.coverage))throw Error('coverage must be covered, uncovered or unknown');
 if(!Array.isArray(value.guidanceIds))throw Error('guidanceIds must be an array of existing guidance IDs');
 const unknown=value.guidanceIds.filter(id=>!entries.some(e=>e.id===id));if(unknown.length)throw Error('Unknown guidanceIds: '+JSON.stringify(unknown)+'; choose IDs from input.json');
 if(value.coverage==='covered'&&!value.guidanceIds.length)throw Error('coverage=covered requires at least one existing guidance ID');
 if(value.coverage!=='covered'&&value.guidanceIds.length)throw Error('coverage='+value.coverage+' requires guidanceIds=[]; use covered only when identifying applicable existing guidance');
 const refs=new Set([...group.occurrences,...(group.linkedDiagnostics??[])].map(o=>o.reference));
 if(!Array.isArray(value.evidenceRefs)||!value.evidenceRefs.length||value.evidenceRefs.some(ref=>!refs.has(ref)))throw Error('evidenceRefs must contain actual occurrence or linked-diagnostic reference IDs from this group');
 return {...value,groupId:group.id,intervention:value.cause==='harness'?'user-required':null};
}
export function validateClassificationBatch(value,groups,entries){
 const rows=value?.classifications;if(!Array.isArray(rows))throw Error('classifications must be an array with one result per supplied groupId');
 const ids=rows.map(row=>row?.groupId);
 if(rows.length!==groups.length||new Set(ids).size!==groups.length||ids.some(id=>!groups.some(g=>g.id===id)))
  throw Error('classifications must contain exactly one result for EVERY supplied groupId, with no duplicates or unknown IDs');
 return {classifications:rows.map(row=>validateClassification(row,groups.find(g=>g.id===row.groupId),entries))};
}
export function validateReview(value,groups){
 if(!value||!['update','learn','no-change'].includes(value.action)||!value.reason?.trim()||
  !Array.isArray(value.groupIds)||!value.groupIds.length||value.groupIds.some(id=>!groups.some(g=>g.id===id)))throw Error('Review requires a disposition and known failure groups');
 if(value.action==='update'&&(!Array.isArray(value.changes)||!value.changes.length||!Array.isArray(value.memoryRefs)||!value.memoryRefs.length))throw Error('Updates require changes and supporting memory references');
 const requests=[...(value.action==='learn'?[value]:[]),...(value.learningRequests??[])];
 if(!Array.isArray(value.learningRequests??[])||requests.some(r=>!['cpu','device'].includes(r.learningTarget)||!r.question?.trim()||!r.hypothesis?.trim()||!r.requiredEvidence?.trim()))throw Error('Learning requires a question, hypothesis and evidence criterion');
 return value;
}
// Called only when a NEW draft is allocated. Existing drafts retain their frozen request.
export function claimLearningRequest(state,config,{draft,topic,target}){
 if(config.skills?.review?.enabled!==true)return null;
 if(!has(state.db,'skill_learning_requests'))return null;
 const row=state.db.prepare(`SELECT id,data FROM skill_learning_requests WHERE status='pending' ORDER BY rowid`).all()
  .find(row=>{const r=JSON.parse(row.data);return r.intervention!=='user-required'&&r.target===target&&(r.topic===undefined||r.topic===topic);});
 if(!row)return null;
 const changed=state.db.prepare("UPDATE skill_learning_requests SET status='claimed',draft=? WHERE id=? AND status='pending'").run(draft,row.id).changes;
 return changed?{...JSON.parse(row.data),id:row.id}:null;
}
export function enqueueLearningRequest(state,request){
 if(request.intervention==='user-required')throw Error('Harness learning requires user intervention, not Phase 1 admission');
 if(!request.question||!request.target||!request.requiredEvidence)throw Error('Invalid learning request');
 state.db.exec(`CREATE TABLE IF NOT EXISTS skill_learning_requests(id TEXT PRIMARY KEY,data TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'pending',draft TEXT)`);
 const id=digest([request.target,request.topic??null,request.question,request.requiredEvidence]);
 state.db.prepare('INSERT OR IGNORE INTO skill_learning_requests(id,data) VALUES (?,?)').run(id,JSON.stringify(request));return id;
}
export function recordLearningOutcome(state,project,attempt){
 const request=project.generation?.learningRequest;if(!request||!has(state.db,'skill_learning_requests'))return;
 const row=state.db.prepare('SELECT data,status,draft FROM skill_learning_requests WHERE id=?').get(request.id);
 if(!row||row.draft!==project.generation.draft)return;
 const data=JSON.parse(row.data);data.findings={attemptId:attempt.id,projectId:project.id,outcome:attempt.outcome};
 state.db.prepare('UPDATE skill_learning_requests SET status=?,data=? WHERE id=?').run(attempt.outcome==='passed'?'verified':'inconclusive',JSON.stringify(data),request.id);
}
