/** Operator-approved attention overlays; scientific outcomes and journals are immutable. */
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {isAbsolute} from 'node:path';
import {ownerStopped} from './lifecycle.mjs';
import {terminalExecutionProofs} from './execution-proof.mjs';

const has=(db,name)=>Boolean(db.prepare("SELECT 1 FROM sqlite_master WHERE name=?").get(name));
const parse=JSON.parse,hash=value=>createHash('sha256').update(value).digest('hex');
export function initializeAttention(db){
 db.exec(`CREATE TABLE IF NOT EXISTS failure_attention_actions(sequence INTEGER PRIMARY KEY,incident TEXT NOT NULL,action TEXT NOT NULL,data TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS failure_attention_effects(sequence INTEGER PRIMARY KEY,action_sequence INTEGER NOT NULL,kind TEXT NOT NULL,identity TEXT NOT NULL,attention TEXT NOT NULL);
 CREATE INDEX IF NOT EXISTS failure_attention_identity ON failure_attention_effects(kind,identity,sequence);`);
}
export const attentionRevision=db=>has(db,'failure_attention_actions')?db.prepare('SELECT COALESCE(MAX(sequence),0) n FROM failure_attention_actions').get().n:0;
export function attentionStatus(db){
 if(!has(db,'failure_attention_actions'))return {revision:0,openIncidents:[],quarantinedOccurrences:0,retiredOccurrences:0};
 const counts=Object.fromEntries(db.prepare("SELECT attention,count(*) n FROM failure_attention_effects e WHERE kind='occurrence' AND sequence=(SELECT MAX(last.sequence) FROM failure_attention_effects last WHERE last.kind=e.kind AND last.identity=e.identity) GROUP BY attention").all().map(r=>[r.attention,r.n]));
 return {revision:attentionRevision(db),openIncidents:db.prepare("SELECT incident id,action FROM failure_attention_actions a WHERE sequence=(SELECT MAX(last.sequence) FROM failure_attention_actions last WHERE last.incident=a.incident) AND action!='retired'").all(),quarantinedOccurrences:counts.quarantined??0,retiredOccurrences:counts.retired??0};
}
const attention=(db,kind,id)=>has(db,'failure_attention_effects')?db.prepare('SELECT attention FROM failure_attention_effects WHERE kind=? AND identity=? ORDER BY sequence DESC LIMIT 1').get(kind,id)?.attention??'active':'active';
export const occurrenceAttention=(db,reference)=>attention(db,'occurrence',reference);
function scopeAttention(db,kind,id){
 const mode=attention(db,kind,id);if(mode==='active'||!has(db,'skill_backlog_occurrences'))return mode;
 const field=kind==='attempt'?'attemptId':'projectId';
 return db.prepare(`SELECT 1 FROM skill_backlog_occurrences o WHERE json_extract(o.data,'$.${field}')=? AND ${activeOccurrenceSql(db,'o')} LIMIT 1`).get(id)?'active':mode;
}
export const projectAttention=(db,id)=>scopeAttention(db,'project',id);
export const attemptAttention=(db,id)=>scopeAttention(db,'attempt',id);
export function activeOccurrenceSql(db,alias='o'){
 if(!/^[A-Za-z_][A-Za-z0-9_]*$/.test(alias))throw Error('SQL alias required');
 return has(db,'failure_attention_effects')?`COALESCE((SELECT e.attention FROM failure_attention_effects e WHERE e.kind='occurrence' AND e.identity=${alias}.reference ORDER BY e.sequence DESC LIMIT 1),'active')='active'`:'1';
}
export function activeItemSql(db,alias='i'){
 if(!/^[A-Za-z_][A-Za-z0-9_]*$/.test(alias))throw Error('SQL alias required');
 return has(db,'failure_attention_effects')?`(${alias}.status='running' OR EXISTS(SELECT 1 FROM json_each(${alias}.data,'$.groupIds') g JOIN skill_backlog_occurrences o ON o.group_id=g.value WHERE ${activeOccurrenceSql(db,'o')}))`:'1';
}
export function attentionSnapshot(db){
 if(!has(db,'failure_attention_actions'))return {revision:0,incidents:[],occurrences:[],attempts:[],projects:[]};
 const rows=db.prepare(`SELECT e.*,a.incident FROM failure_attention_effects e JOIN failure_attention_actions a ON a.sequence=e.action_sequence
 WHERE e.sequence=(SELECT MAX(last.sequence) FROM failure_attention_effects last WHERE last.kind=e.kind AND last.identity=e.identity) ORDER BY e.sequence`).all();
 const incidents=db.prepare(`SELECT * FROM failure_attention_actions a WHERE a.sequence=(SELECT MAX(last.sequence) FROM failure_attention_actions last WHERE last.incident=a.incident) ORDER BY sequence`).all().map(r=>({...r,state:r.action==='retired'?'resolved':'open',data:parse(r.data),manifest:parse(db.prepare("SELECT data FROM failure_attention_actions WHERE incident=? AND action IN ('retired','quarantined') ORDER BY sequence LIMIT 1").get(r.incident).data).manifest}));
 const pick=kind=>rows.filter(r=>r.kind===kind).map(r=>({id:r.identity,attention:kind==='occurrence'?r.attention:scopeAttention(db,kind,r.identity),incidentId:r.incident,revision:r.action_sequence}));
 return {revision:attentionRevision(db),incidents,occurrences:pick('occurrence'),attempts:pick('attempt'),projects:pick('project')};
}
const latest=(db,id)=>has(db,'failure_attention_actions')?db.prepare('SELECT * FROM failure_attention_actions WHERE incident=? ORDER BY sequence DESC LIMIT 1').get(id):null;
const cutoffs=db=>Object.fromEntries(['events','generation_events'].filter(t=>has(db,t)).map(t=>[t,db.prepare(`SELECT COALESCE(MAX(sequence),0) n FROM ${t}`).get().n]));
function scopeRows(db,input){
 if(!input?.id?.trim()||!input.target?.trim()||!input.diagnosis?.trim()||!input.operator?.reason?.trim()||!['quarantined','retired'].includes(input.attention)||
  !Array.isArray(input.references)||!input.references.length||new Set(input.references).size!==input.references.length)throw Error('Operator reason, incident, target, diagnosis, attention and exact unique references required');
 const rows=input.references.map(reference=>{
  const row=db.prepare('SELECT * FROM skill_backlog_occurrences WHERE reference=?').get(reference);
  if(!row)throw Error('Unknown failure occurrence: '+reference);
  const data=parse(row.data),attemptId=row.attempt??data.attemptId??null,projectId=attemptId?db.prepare('SELECT project FROM attempts WHERE id=?').get(attemptId)?.project:null;
  return {reference,groupId:row.group_id,attemptId,projectId:projectId??data.projectId??null};
 });
 for(const [field,key]of [['attemptIds','attemptId'],['projectIds','projectId']]){
  if(input[field]!==undefined&&(!Array.isArray(input[field])||new Set(input[field]).size!==input[field].length||input[field].some(id=>!rows.some(r=>r[key]===id))))throw Error(field+' must identify scopes in the exact references');
 }
 if(input.watchGroupIds!==undefined&&(!Array.isArray(input.watchGroupIds)||input.watchGroupIds.some(id=>!rows.some(r=>r.groupId===id))))throw Error('Watch groups must belong to the affected references');
 return rows;
}
function terminalScopes(db,input){
 const refs=new Set(input.references),coverage=new Set(input.attemptIds??[]);
 for(const project of input.projectIds??[]){
  const row=db.prepare('SELECT status FROM projects WHERE id=?').get(project);
  if(row?.status!=='failed')throw Error('Only failed terminal projects may leave active attention: '+project);
  for(const a of db.prepare('SELECT id FROM attempts WHERE project=?').all(project))coverage.add(a.id);
 }
 // Even reference-only overlays must not hide a live/unsettled attempt.
 const ids=new Set([...coverage,...scopeRows(db,input).map(row=>row.attemptId).filter(Boolean)]);
 const owners=[];
 for(const id of ids){
  const a=db.prepare('SELECT * FROM attempts WHERE id=?').get(id);
  const closure=db.prepare("SELECT data FROM events WHERE attempt=? AND type='failure-settlement' ORDER BY sequence DESC LIMIT 1").get(id);
  const passed=a?.outcome==='passed'&&db.prepare("SELECT 1 FROM events WHERE attempt=? AND type='postprocess-completed'").get(id);
  const investigation=passed&&has(db,'recovery_attempts')&&db.prepare("SELECT 1 FROM recovery_attempts WHERE attempt=? AND kind='investigation' AND completed=1").get(id);
  if(!(investigation||passed&&!coverage.has(id)||a?.outcome==='failed'&&closure)||closure&&parse(closure.data).closed!==true)throw Error('Explicit positive failure settlement or completed investigation required: '+id);
  if(!journalClosed(db,id))throw Error('Unmatched execution remains unresolved: '+id);
  if(a.owner)owners.push(parse(a.owner));
  if(!coverage.has(id))continue;
  const occurrences=db.prepare('SELECT reference FROM skill_backlog_occurrences WHERE attempt=?').all(id);
  if(!investigation&&!occurrences.length||occurrences.some(r=>!refs.has(r.reference)&&occurrenceAttention(db,r.reference)==='active'))throw Error('Mixed or uncovered failures remain active: '+id);
  // An incomplete ingest must not hide a known verifier failure in the same attempt.
  for(const row of db.prepare("SELECT sequence,data FROM events WHERE attempt=? AND type='verification'").all(id)){
   const v=parse(row.data);if((v.passed===false||v.correct===false||v.correctness?.passed===false||v.performance?.passed===false)&&!refs.has('events:'+row.sequence)&&occurrenceAttention(db,'events:'+row.sequence)==='active')throw Error('Uncovered verification failure: '+id);
  }
 }
 return owners;
}
// Positive operation completions cover preceding starts only; no transport or writes.
function journalClosed(db,id){
 const events=db.prepare('SELECT type,data FROM events WHERE attempt=? ORDER BY sequence').all(id).map(e=>({...e,data:parse(e.data)})),closed=new Map(),bash=new Map();
 const terminal=terminalExecutionProofs(events);
 for(const [index,e]of events.entries()){
  if(e.type==='execution-settled'||e.type==='execution-reconciled'&&e.data.closed===true)closed.set(e.data.operation,index);
  if(e.type==='native-bash-completed')bash.set(e.data.operation,index);
 }
 return events.every((e,index)=>{
  const op=e.data.operation??e.data.submission;
  if(e.type==='native-bash-started')return (bash.get(op)??-1)>index;
  // Reconciliation records recovered handle identities after its closure event.
  // Only immutable native terminal evidence may cover that late identity;
  // every subsequent launch/unresolved marker still needs its own closure.
  if(e.type==='execution-handle')return Boolean(e.data.handle)&&op!==undefined&&((closed.get(op)??-1)>index||closed.has(op)&&terminal.has(e.data.handle));
  if(['execution-starting','execution-submitting','execution-unresolved'].includes(e.type))return op!==undefined&&(closed.get(op)??-1)>index;
  return true;
 });
}
export async function auditResolution(db,input,{target=input?.target}={}){
 if(input.target!==target)throw Error('Resolution target differs from campaign');
 const rows=scopeRows(db,input),owners=terminalScopes(db,input);
 const basis=input.retirementBasis??'fixed';
 if(input.attention==='retired'&&(!['fixed','not-expected-to-recur'].includes(basis)||basis==='fixed'&&!/^[a-f0-9]{40}$/.test(input.fixRevision??'')))throw Error('Retirement requires a fix revision or explicit nonrecurrence basis');
 if(!Array.isArray(input.validation)||!input.validation.length)throw Error('Hashed diagnosis/validation evidence required');
 for(const evidence of input.validation){
  if(!isAbsolute(evidence.path??'')||!/^[a-f0-9]{64}$/.test(evidence.sha256??''))throw Error('Validation evidence checksum differs');
  const bytes=await readFile(evidence.path);if(hash(bytes)!==evidence.sha256)throw Error('Validation evidence checksum differs');
  if(input.attention==='retired'){
   let proof;try{proof=parse(bytes.toString('utf8'));}catch{throw Error('Retirement needs a positive structured validation receipt');}
   if(basis==='fixed'){
    if(proof.passed!==true||proof.target!==input.target||proof.sourceRevision!==input.fixRevision)throw Error('Retirement requires passed validation for this exact target and fix revision');
   }else if(input.expectedRecurrence!==false||proof.expectedRecurrence!==false||proof.retirementBasis!==basis||proof.target!==input.target||!proof.reason?.trim()||!['obsolete-benchmark','abandoned-experiment','documented-one-off'].includes(proof.kind)||JSON.stringify([...(proof.references??[])].sort())!==JSON.stringify([...input.references].sort()))throw Error('Nonrecurrence requires exact scoped obsolete/abandoned/one-off evidence, not age alone');
  }
 }
 for(const owner of owners)if(!await ownerStopped(owner))throw Error('Scoped worker remains live');
 return {revision:attentionRevision(db),manifest:structuredClone({...input,...(input.attention==='retired'?{retirementBasis:basis}:{})}),occurrences:rows,cutoffs:cutoffs(db)};
}
const transaction=(db,fn)=>{const nested=db.isTransaction;if(!nested)db.exec('BEGIN IMMEDIATE');try{const v=fn();if(!nested)db.exec('COMMIT');return v;}catch(e){if(!nested)db.exec('ROLLBACK');throw e;}};
function append(db,id,action,data,effects=[]){
 const seq=Number(db.prepare('INSERT INTO failure_attention_actions(incident,action,data) VALUES(?,?,?)').run(id,action,JSON.stringify(data)).lastInsertRowid);
 for(const [kind,identity,mode]of effects)db.prepare('INSERT INTO failure_attention_effects(action_sequence,kind,identity,attention) VALUES(?,?,?,?)').run(seq,kind,identity,mode);
 return seq;
}
export async function applyResolution(db,input,options={}){
 const audited=await auditResolution(db,input,options);initializeAttention(db);
 return transaction(db,()=>{
  const old=latest(db,input.id),signature=hash(JSON.stringify(input));
  if(old&&parse(old.data).signature===signature)return {applied:false,revision:old.sequence};
  if(old)throw Error('Incident exists; use a new incident or explicit reopen');
  if(attentionRevision(db)!==audited.revision)throw Error('Attention changed; audit again');
  const rows=scopeRows(db,input);terminalScopes(db,input);
  if(JSON.stringify(rows)!==JSON.stringify(audited.occurrences))throw Error('Failure scope changed; audit again');
  const effects=[...input.references.map(id=>['occurrence',id,input.attention]),...(input.attemptIds??[]).map(id=>['attempt',id,input.attention]),...(input.projectIds??[]).map(id=>['project',id,input.attention])];
  const revision=append(db,input.id,input.attention,{...audited,signature,at:new Date().toISOString()},effects);
  return {applied:true,revision,occurrences:rows.length};
 });
}
export function reopenResolution(db,{id,reason,references,projectIds=[],attemptIds=[]}){
 if(!reason?.trim()||!latest(db,id)||!Array.isArray(references)||!references.length)throw Error('Existing incident, reason and exact references required');
 return transaction(db,()=>{
  const effects=[...references.map(ref=>['occurrence',ref,'active']),...projectIds.map(p=>['project',p,'active']),...attemptIds.map(a=>['attempt',a,'active'])];
  for(const [kind,identity]of effects)if(!db.prepare('SELECT 1 FROM failure_attention_effects e JOIN failure_attention_actions a ON a.sequence=e.action_sequence WHERE a.incident=? AND e.kind=? AND e.identity=?').get(id,kind,identity))throw Error('Scope does not belong to incident');
  return {revision:append(db,id,'reopened',{reason,references,projectIds,attemptIds,at:new Date().toISOString()},effects)};
 });
}
/** Called only after a new journal occurrence is inserted, never for old replays. */
export function observeAttentionRecurrence(db,{reference,groupId,projectId,attemptId}){
 if(!has(db,'failure_attention_actions')||occurrenceAttention(db,reference)!=='active')return;
 const colon=reference.indexOf(':'),journal=reference.slice(0,colon),sequence=Number(reference.slice(colon+1));
 if(!Number.isSafeInteger(sequence))return;
 for(const row of db.prepare("SELECT * FROM failure_attention_actions WHERE action='retired'").all()){
  const data=parse(row.data),m=data.manifest;
  if(!m.watchGroupIds?.includes(groupId)||sequence<=(data.cutoffs[journal]??Infinity))continue;
  if(db.prepare("SELECT 1 FROM failure_attention_actions WHERE incident=? AND action='regression' AND json_extract(data,'$.reference')=?").get(row.incident,reference))continue;
  if(latest(db,row.incident)?.action!=='retired')continue;
  const effects=[];if(m.projectIds?.includes(projectId))effects.push(['project',projectId,'active']);if(m.attemptIds?.includes(attemptId))effects.push(['attempt',attemptId,'active']);
  append(db,row.incident,'regression',{reference,groupId,projectId,attemptId,fixRevision:m.fixRevision,at:new Date().toISOString()},effects);
 }
}
