import {createHash,randomUUID} from 'node:crypto';
import {isAbsolute} from 'node:path';
import {diagnosticIdentity} from './skill-review.mjs';
import {activeOccurrenceSql,activeItemSql,occurrenceAttention,observeAttentionRecurrence,attentionStatus} from './failure-attention.mjs';
const hash=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const parse=value=>JSON.parse(value);
const has=(db,name)=>Boolean(db.prepare("SELECT 1 FROM sqlite_master WHERE name=?").get(name));
const terminal=new Set(['accepted','rejected','inconclusive','archived']);
const analysisScopeSql=db=>{
 const scope=has(db,'skill_backlog_meta')?parse(db.prepare("SELECT value FROM skill_backlog_meta WHERE key='queue-rebuild'").get()?.value??'null'):null;
 if(!scope)return '1';
 const ids=JSON.stringify(scope.groupIds??[]).replaceAll("'","''"),fresh=['events','generation_events'].map(j=>`(o.reference LIKE '${j}:%' AND CAST(substr(o.reference,${j.length+2}) AS INTEGER)>${Number.isSafeInteger(scope.cutoffs?.[j])?scope.cutoffs[j]:Number.MAX_SAFE_INTEGER})`).join(' OR ');
 const newScopes=[['attempts','attemptId'],['generation_drafts','draftId']].filter(([t])=>has(db,t)).map(([t,field])=>`EXISTS(SELECT 1 FROM ${t} s WHERE s.id=json_extract(o.data,'$.${field}') AND s.rowid>${Number.isSafeInteger(scope.cutoffs?.[t])?scope.cutoffs[t]:Number.MAX_SAFE_INTEGER})`).join(' OR ');
 return `(f.id IN (SELECT value FROM json_each('${ids}')) OR EXISTS(SELECT 1 FROM skill_backlog_occurrences o WHERE o.group_id=f.id AND (${fresh}${newScopes?' OR '+newScopes:''})))`;
};

const statuses=new Set(['queued','running','needs-review','learning','user-required',...terminal]);
const semantic=(value,evidence=false)=>{
 if(Array.isArray(value)){const rows=value.map(v=>semantic(v,evidence));return evidence?[...new Map(rows.map(v=>[JSON.stringify(v),v])).entries()].sort(([a],[b])=>a.localeCompare(b)).map(([,v])=>v):rows;}
 if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).sort(([a],[b])=>a.localeCompare(b)).filter(([k])=>
  !/^(reference|references|evidenceRefs|groupIds|attemptId|draftId|projectId|created|elapsed|count|historicalCount|frequency|occurrences|firstSeen|lastSeen)$/.test(k)&&
  !(evidence&&/^(path|copiedPath|snapshotPath|archive|proposalDirectory|proposalElapsedSeconds|classification|classifications)$/.test(k))).map(([k,v])=>[k,semantic(v,evidence)]));
 return typeof value==='string'?value.replace(/[a-f0-9]{8}-[a-f0-9-]{27,}/gi,'<id>'):value;
};
const pick=(value,keys)=>Object.fromEntries(keys.filter(key=>Object.hasOwn(value,key)).map(key=>[key,value[key]]));
export function proposalIdentity(proposal){
 if(proposal?.action==='update')return {action:'update',changes:(proposal.changes??[]).map(change=>change.action==='remove'?pick(change,['action','id','scope']):
  {action:change.action,scope:change.scope,text:change.text?.trim().replace(/\s+/g,' '),critical:change.critical??false,negative:change.negative??false})};
 if(proposal?.action==='learn')return pick(proposal,['action','question','hypothesis','requiredEvidence','learningTarget','target','topic']);
 return proposal?.action==='no-change'?{action:'no-change'}:proposal;
}
function evidenceIdentity(item){
 const evidence=item.evidence;if(!evidence||!Array.isArray(evidence.memoryEvidence))return evidence;
 const memoryEvidence=evidence.memoryEvidence.map(entry=>({ref:entry.ref??(entry.collection?`${entry.collection}:${entry.entry_id??entry.chunk_id??entry.id}`:entry.entry_id??entry.chunk_id??entry.id),
  text:entry.text??entry.content,sha256:entry.sha256??entry.contentHash??entry.hash}));
 const cited=item.proposal?.memoryRefs;
 return {...evidence,memoryEvidence:(Array.isArray(cited)?memoryEvidence.filter(entry=>cited.includes(entry.ref)):memoryEvidence).map(({ref,...facts})=>facts)};
}
export const groundingIdentity=item=>{const evidence=evidenceIdentity(item);return hash([semantic(evidence?.memoryEvidence,true),evidence?.learningIdentity??diagnosticIdentity({data:semantic(evidence?.learning,true)}),semantic(item.tests??item.evaluation??null,true)]);};
export const hypothesisIdentity=item=>hash([item.objective,item.skill,semantic(proposalIdentity(item.proposal)),diagnosticIdentity({data:semantic(evidenceIdentity(item),true)}),item.base,semantic(item.tests??item.evaluation??null,true)]);
const fingerprint=hypothesisIdentity;
const journalFilter=`CASE WHEN type IN ('pi-event','curriculum-checkpoint','native-bash-started','native-bash-completed','knowledge-retrieved','skill-read') THEN 0
 WHEN type='execution-state' THEN CASE WHEN json_valid(data) THEN json_extract(data,'$.state')='stopped' AND COALESCE(json_extract(data,'$.accountingExcluded'),0)=0 AND
 (COALESCE(json_extract(data,'$.exitCode'),json_extract(data,'$.exit_code'),0)>0 OR json_extract(data,'$.nativeState')='failed') ELSE 0 END ELSE
 (type IN ('failure-recorded','project-invalid') OR type LIKE '%error%' OR type LIKE '%failed%' OR type LIKE '%rejected%' OR type LIKE '%unresolved%' OR type LIKE '%blocked%' OR
 CASE WHEN json_valid(data) THEN (json_type(data,'$.passed')='false' OR json_type(data,'$.correct')='false' OR json_type(data,'$.correctness.passed')='false' OR json_type(data,'$.performance.passed')='false' OR
 json_type(data,'$.verification.passed')='false' OR json_type(data,'$.result.passed')='false' OR json_type(data,'$.result.correct')='false' OR json_type(data,'$.verdict.passed')='false' OR
 type='verification' AND COALESCE(json_extract(data,'$.exitCode'),json_extract(data,'$.exit_code'),0)>0) ELSE 0 END) END`;
const negative=data=>data&&typeof data==='object'&&(data.passed===false||data.correct===false||
 ['correctness','performance','verification','verdict','result'].some(k=>negative(data[k])));
function failure(type,data){
 const actual=data.exitCode??data.exit_code,expected=data.expected_exit;
 if(data.operator===true||data.operatorRequested===true||['operator-pause','operator-stop'].includes(data.reason)||expected!==undefined&&expected!==0&&actual===expected||
  type==='execution-state'&&(data.accountingExcluded===true||['resource-health','device-lease-control'].includes(data.purpose)))return false;
 return negative(data)||actual!==undefined&&expected!==undefined&&actual!==expected||
  /error|failed|rejected|unresolved|blocked/.test(type)||['failure-recorded','project-invalid','generation-invalid','generation-exhausted'].includes(type)||type==='verification'&&actual>0||
  type==='execution-state'&&data.state==='stopped'&&(actual>0||data.nativeState==='failed');
}

/** Campaign-local durable evidence and hypothesis scheduling. No model calls. */
export class SkillBacklog {
 constructor(state,{target,now=Date.now}={}){
  this.db=state.db??state;this.target=target;this.now=now;
  this.db.exec(`CREATE TABLE IF NOT EXISTS skill_backlog_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
   CREATE TABLE IF NOT EXISTS skill_backlog_failures(id TEXT PRIMARY KEY,data TEXT NOT NULL,dirty INTEGER NOT NULL DEFAULT 1,fresh INTEGER NOT NULL DEFAULT 0,updated INTEGER NOT NULL);
   CREATE TABLE IF NOT EXISTS skill_backlog_occurrences(reference TEXT PRIMARY KEY,group_id TEXT NOT NULL,attempt TEXT,data TEXT NOT NULL);
   CREATE INDEX IF NOT EXISTS skill_backlog_occurrence_group ON skill_backlog_occurrences(group_id);
   CREATE INDEX IF NOT EXISTS skill_backlog_occurrence_attempt ON skill_backlog_occurrences(attempt);
   CREATE TABLE IF NOT EXISTS skill_backlog_items(id TEXT PRIMARY KEY,key TEXT NOT NULL UNIQUE,data TEXT NOT NULL,status TEXT NOT NULL,priority REAL NOT NULL,created INTEGER NOT NULL);
   CREATE TABLE IF NOT EXISTS skill_backlog_aliases(key TEXT PRIMARY KEY,item_id TEXT NOT NULL);
   CREATE TABLE IF NOT EXISTS skill_backlog_runs(id TEXT PRIMARY KEY,item_id TEXT NOT NULL,revision INTEGER NOT NULL,base TEXT,seconds REAL NOT NULL,elapsed REAL NOT NULL DEFAULT 0,status TEXT NOT NULL,snapshot TEXT NOT NULL,checkpoint TEXT,result TEXT,owner TEXT);
   CREATE UNIQUE INDEX IF NOT EXISTS skill_backlog_one_running ON skill_backlog_runs((1)) WHERE status='running';`);
 }
 atomic(fn){if(this.db.isTransaction)return fn();this.db.exec('BEGIN IMMEDIATE');try{const value=fn();this.db.exec('COMMIT');return value;}catch(error){this.db.exec('ROLLBACK');throw error;}}
 meta(key,value){if(value===undefined)return parse(this.db.prepare('SELECT value FROM skill_backlog_meta WHERE key=?').get(key)?.value??'null');this.db.prepare('INSERT OR REPLACE INTO skill_backlog_meta VALUES(?,?)').run(key,JSON.stringify(value));return value;}
 ingest({limit=100}={}){
  if(!Number.isSafeInteger(limit)||limit<1)throw Error('Positive ingestion page size required');
  return this.atomic(()=>{
   let observed=0,progress=false;
   const add=(row,journal,fresh=false)=>{
    const data=typeof row.data==='string'?parse(row.data):row.data;if(!failure(row.type,data))return;
    const attempt=row.attempt&&this.db.prepare('SELECT project FROM attempts WHERE id=?').get(row.attempt);
    const spec=attempt&&parse(this.db.prepare('SELECT spec FROM projects WHERE id=?').get(attempt.project)?.spec??'{}');
    const admission=row.attempt&&this.db.prepare("SELECT data FROM events WHERE attempt=? AND type='job-admitted' ORDER BY sequence LIMIT 1").get(row.attempt);
    const job=admission?parse(admission.data):{},selection=row.attempt&&this.db.prepare("SELECT data FROM events WHERE attempt=? AND type='skills-selected' AND sequence<=? ORDER BY sequence DESC LIMIT 1").get(row.attempt,journal==='events'?row.sequence:Number.MAX_SAFE_INTEGER);
    const diagnostic=diagnosticIdentity({data}),id=hash([this.target,row.type,diagnostic]);
    const reference=`${journal}:${row.sequence}`,occurrence={reference,attemptId:row.attempt??null,draftId:row.draft??null,projectId:attempt?.project??null,
     phase:row.phase??data.phase??job.mode??spec?.phase??null,kind:job.kind??null,skillVersion:(selection&&parse(selection.data).version)??job.skillVersion??job.config?.skills?.activeVersion??null,skillReads:[],artifacts:[],created:row.created??null,data,diagnostic};
    if(!this.db.prepare('INSERT OR IGNORE INTO skill_backlog_occurrences VALUES(?,?,?,?)').run(reference,id,row.attempt??null,JSON.stringify(occurrence)).changes)return;
    observeAttentionRecurrence(this.db,{reference,groupId:id,projectId:occurrence.projectId,attemptId:occurrence.attemptId});
    if(occurrenceAttention(this.db,reference)!=='active')return;
    const group={id,target:this.target,type:row.type,message:String(data.message??data.error??data.reason??row.type).slice(0,1500)};
    this.db.prepare('INSERT INTO skill_backlog_failures VALUES(?,?,1,?,?) ON CONFLICT(id) DO UPDATE SET dirty=1,fresh=MAX(fresh,excluded.fresh),updated=excluded.updated').run(id,JSON.stringify(group),Number(fresh),this.now());observed++;
   };
   for(const journal of ['events','generation_events'])if(has(this.db,journal)){
    let cursor=this.meta('cursor:'+journal);
    cursor??={end:this.db.prepare(`SELECT COALESCE(MAX(sequence),0) n FROM ${journal}`).get().n,history:0,fresh:0};
    cursor.fresh=Math.max(cursor.end,cursor.fresh);
    const scan=(position,end,fresh)=>{
     const upper=Math.min(end,position+limit*100),rows=this.db.prepare(`SELECT * FROM ${journal} WHERE sequence>? AND sequence<=? AND ${journalFilter} ORDER BY sequence LIMIT ?`).all(position,upper,limit);
     for(const row of rows)add(row,journal,fresh);const next=rows.length===limit?rows.at(-1).sequence:upper;progress||=next>position;return next;
    };
    cursor.fresh=scan(cursor.fresh,this.db.prepare(`SELECT COALESCE(MAX(sequence),0) n FROM ${journal}`).get().n,true);
    cursor.history=scan(cursor.history,cursor.end,false);
    this.meta('cursor:'+journal,cursor);
   }
   // Mutable rows can become terminal after their first pass. Cycle bounded pages.
   for(const table of ['attempts','generation_drafts'])if(has(this.db,table)){
    const cursor=this.meta('sweep:'+table)??0,rows=this.db.prepare(`SELECT rowid AS ordinal,* FROM ${table} WHERE rowid>? ORDER BY rowid LIMIT ?`).all(cursor,limit);
    for(const row of rows){
     if(table==='attempts'&&row.outcome==='failed'&&!this.db.prepare('SELECT 1 FROM skill_backlog_occurrences WHERE attempt=? LIMIT 1').get(row.id)){
      const operator=this.db.prepare(`SELECT 1 FROM events WHERE attempt=? AND type IN ('attempt-error','failure-recorded','attempt-stopped','operator-pause') AND
       CASE WHEN json_valid(data) THEN (json_extract(data,'$.operatorRequested')=1 OR json_extract(data,'$.operator')=1 OR json_extract(data,'$.reason') IN ('operator-pause','operator-stop')) ELSE 0 END LIMIT 1`).get(row.id);
      if(!operator)add({sequence:row.id,attempt:row.id,type:'failed-outcome',created:row.created,data:{outcome:'failed',message:'Failed outcome; diagnostic journal not yet observed'}},table);
     }
     if(table==='generation_drafts'){
      const d=parse(row.data);if(['invalid','exhausted','failed'].includes(d.status))add({sequence:row.id+':'+d.status,draft:row.id,phase:d.phase,type:'generation-'+d.status,data:{message:d.error??'Curriculum elapsed allocation exhausted',topic:d.topic}},table);
     }
    }
    this.meta('sweep:'+table,rows.length?rows.at(-1).ordinal:0);
   }
   return {observed,progress,...this.status()};
  });
 }
 failureCount(id){return this.db.prepare(`SELECT COUNT(*) n FROM skill_backlog_occurrences o WHERE group_id=? AND ${activeOccurrenceSql(this.db,'o')}`).get(id).n;}
 failure(id,{limit=20,offset=0,includeRetired=false}={}){
  const row=this.db.prepare('SELECT data,dirty FROM skill_backlog_failures WHERE id=?').get(id);if(!row)return null;
  const eligible=includeRetired?'1':activeOccurrenceSql(this.db,'o');
  const historicalCount=this.db.prepare('SELECT count(*) n FROM skill_backlog_occurrences WHERE group_id=?').get(id).n;
  const count=this.db.prepare(`SELECT count(*) n FROM skill_backlog_occurrences o WHERE group_id=? AND ${eligible}`).get(id).n;
  if(!includeRetired&&!count)return null;
  const occurrences=this.db.prepare(`SELECT data FROM skill_backlog_occurrences o WHERE group_id=? AND ${eligible} ORDER BY rowid DESC LIMIT ? OFFSET ?`).all(id,limit,offset).map(r=>parse(r.data));
  for(const occurrence of occurrences)if(occurrence.attemptId)occurrence.skillReads=this.db.prepare("SELECT data FROM events WHERE attempt=? AND type='skill-read' ORDER BY sequence DESC LIMIT 12").all(occurrence.attemptId).map(r=>parse(r.data));
  const linkedDiagnostics=[];
  for(const attempt of [...new Set(occurrences.map(o=>o.attemptId).filter(Boolean))].slice(0,2)){
   for(const row of this.db.prepare(`SELECT sequence,type,data FROM events WHERE attempt=? AND type IN ('verification','baseline-preflight-completed','project-invalid','attempt-error') ORDER BY sequence DESC LIMIT 12`).all(attempt)){
    const data=parse(row.data),reference='events:'+row.sequence;if(failure(row.type,data)&&(includeRetired||occurrenceAttention(this.db,reference)==='active'))linkedDiagnostics.push({reference,type:row.type,data,artifacts:[]});
   }
  }
  const frequency=this.db.prepare(`SELECT COUNT(DISTINCT json_extract(o.data,'$.projectId')) projects,COUNT(DISTINCT o.attempt) attempts,COUNT(DISTINCT json_extract(o.data,'$.draftId')) drafts FROM skill_backlog_occurrences o WHERE group_id=? AND ${eligible}`).get(id);
  return {...parse(row.data),dirty:Boolean(row.dirty)&&count>0,count,historicalCount,frequency:{...frequency},occurrences,linkedDiagnostics};
 }
 failures({dirty=true,limit=100,offset=0,occurrenceLimit=20,includeRetired=false}={}){return this.db.prepare(`SELECT id FROM skill_backlog_failures f WHERE ${dirty?'dirty=1':'1'} AND ${includeRetired?'1':analysisScopeSql(this.db)} ${includeRetired?'':`AND EXISTS(SELECT 1 FROM skill_backlog_occurrences o WHERE o.group_id=f.id AND ${activeOccurrenceSql(this.db,'o')})`} ORDER BY fresh DESC,updated DESC,rowid LIMIT ? OFFSET ?`).all(limit,offset).map(r=>this.failure(r.id,{limit:occurrenceLimit,includeRetired}));}
 acknowledge(id,{classification,count}={}){return this.atomic(()=>{const group=this.failure(id,{limit:0});if(!group)return;const data=this.db.prepare('SELECT data FROM skill_backlog_failures WHERE id=?').get(id);this.db.prepare('UPDATE skill_backlog_failures SET data=?,dirty=?,fresh=CASE WHEN ? THEN fresh ELSE 0 END WHERE id=?').run(JSON.stringify({...parse(data.data),...(classification?{classification}:{})}),Number(count!==undefined&&count<group.count),Number(count!==undefined&&count<group.count),id);});}
 get(id){const epoch=this.meta('queue-epoch'),key=epoch&&!id.startsWith('epoch:')?'epoch:'+epoch+':'+id:id;const row=this.db.prepare('SELECT data FROM skill_backlog_items WHERE id=? OR key=? OR id=(SELECT item_id FROM skill_backlog_aliases WHERE key=?)').get(id,key,key);return row?parse(row.data):null;}
 list({status,limit=100,includeRetired=false}={}){return this.db.prepare(`SELECT data FROM skill_backlog_items i WHERE ${status?'status=?':'1'} AND ${includeRetired?'1':activeItemSql(this.db,'i')} ORDER BY priority DESC,created,rowid LIMIT ?`).all(...(status?[status,limit]:[limit])).map(r=>parse(r.data));}
 store(item){this.db.prepare('INSERT INTO skill_backlog_items VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data,status=excluded.status,priority=excluded.priority')
  .run(item.id,item.key,JSON.stringify(item),item.status,item.priority,item.created);return item;}
 completed(item){const identity=fingerprint(item);for(const row of this.db.prepare("SELECT id,status,snapshot FROM skill_backlog_runs WHERE base IS ? AND status IN ('accepted','rejected','inconclusive') ORDER BY rowid DESC").iterate(item.base??null))if(fingerprint(parse(row.snapshot))===identity)return row;return null;}
 upsert(input){return this.atomic(()=>{
  const epoch=this.meta('queue-epoch');if(epoch&&!input.key.startsWith('epoch:'))input={...input,key:'epoch:'+epoch+':'+input.key};
  if(!input.key||!['harness','correctness','optimization'].includes(input.objective)||!Array.isArray(input.groupIds)||!input.groupIds.length)throw Error('Hypothesis key, objective and failure groups required');
  for(const id of input.groupIds)if(!this.db.prepare('SELECT 1 FROM skill_backlog_failures WHERE id=?').get(id))throw Error('Unknown failure group');
  let prior=this.get(input.key);
  if(!prior)for(const row of this.db.prepare('SELECT data FROM skill_backlog_items ORDER BY rowid').iterate()){const candidate=parse(row.data);if(fingerprint(candidate)===fingerprint(input)){prior=candidate;break;}}
  const item={...prior,...input,id:prior?.id??randomUUID(),key:prior?.key??input.key,created:prior?.created??this.now(),priority:input.priority??prior?.priority??0};
  item.fingerprint=fingerprint(item);const changed=prior&&fingerprint(prior)!==item.fingerprint;
  item.revision=prior?(prior.revision+(changed?1:0)):1;item.status=input.status??'queued';
  if(prior&&!changed&&terminal.has(prior.status))item.status=prior.status;
  const completed=this.completed(item);if(completed){item.status=completed.status;item.previousRun=completed.id;}
  if(prior?.status==='running')item.status='running';
  if(item.objective==='harness')item.status='user-required';
  if(!statuses.has(item.status)||!Number.isFinite(item.priority))throw Error('Valid status and priority required');
  this.store(item);if(input.key!==item.key)this.db.prepare('INSERT OR REPLACE INTO skill_backlog_aliases VALUES(?,?)').run(input.key,item.id);return item;
 });}
 revise(id,patch){const item=this.get(id);if(!item)throw Error('Unknown hypothesis');return this.upsert({...item,...patch,status:patch.status??'queued'});}
 rank(rows){return this.atomic(()=>rows.map(({id,priority,reason})=>{const item=this.get(id);if(!item||!Number.isFinite(priority)||!reason)throw Error('Known hypothesis, priority and reason required');return this.store({...item,priority,reason});}));}
 rebase(id,base){return this.atomic(()=>{const item=this.get(id);if(!item)throw Error('Unknown hypothesis');if(item.base===base)return item;if(item.status==='running')throw Error('Settle running comparison before rebasing');return this.revise(id,{base,status:'needs-review',reason:'Active skill changed; requalification required'});});}
 run(id){const row=this.db.prepare("SELECT * FROM skill_backlog_runs WHERE id=? OR (item_id=? AND status='running') ORDER BY rowid DESC LIMIT 1").get(id,id);return row?{...row,snapshot:parse(row.snapshot),checkpoint:row.checkpoint&&parse(row.checkpoint),result:row.result&&parse(row.result),owner:row.owner&&parse(row.owner)}:null;}
 claim({base,seconds=1800,owner,itemId}={}){return this.atomic(()=>{
  if(!Number.isFinite(seconds)||seconds<=0||seconds>1800)throw Error('Hypothesis allowance must fit 1800 elapsed seconds');
  let row=this.db.prepare("SELECT id FROM skill_backlog_runs WHERE status='running'").get();
  if(row){const run=this.run(row.id);if(run.base!==base)throw Error('Settle retained comparison before baseline change');if(owner!==undefined)this.db.prepare('UPDATE skill_backlog_runs SET owner=? WHERE id=?').run(JSON.stringify(owner),run.id);return {item:this.get(run.item_id),run:this.run(run.id),remaining:Math.max(0,run.seconds-run.elapsed),reused:true};}
  for(const item of itemId?[this.get(itemId)].filter(item=>item?.status==='queued'):this.list({status:'queued',limit:100})){
   if(!this.db.prepare(`SELECT 1 FROM skill_backlog_items i WHERE id=? AND ${activeItemSql(this.db,'i')}`).get(item.id))continue;
   if(item.base!==base){this.rebase(item.id,base);continue;}
   const completed=this.completed(item);if(completed){this.store({...item,status:completed.status,previousRun:completed.id});continue;}
   const id=randomUUID();this.db.prepare("INSERT INTO skill_backlog_runs(id,item_id,revision,base,seconds,status,snapshot,owner) VALUES(?,?,?,?,?,'running',?,?)")
    .run(id,item.id,item.revision,base,seconds,JSON.stringify(item),JSON.stringify(owner??null));this.store({...item,status:'running'});
   return {item:this.get(item.id),run:this.run(id),remaining:seconds,reused:false};
  }
  return null;
 });}
 /** Explicit procedural continuation; prior records stay immutable and time is never renewed. */
 continue(id,{procedure,owner,base,missingCases=false,caseSelectionRepair}={}){return this.atomic(()=>{
  const previous=this.run(id),reason=previous?.result?.reason;
  if(caseSelectionRepair!==undefined){
   const {fromRevision,toRevision,evidence}=caseSelectionRepair??{};
   if(!/^[a-f0-9]{40}$/.test(fromRevision??'')||!/^[a-f0-9]{40}$/.test(toRevision??'')||fromRevision===toRevision||
    !isAbsolute(evidence?.path??'')||!/^[a-f0-9]{64}$/.test(evidence?.sha256??''))throw Error('Case-selection repair requires distinct revisions and immutable evidence path/hash');
  }
  const repair=caseSelectionRepair!==undefined&&previous?.status==='inconclusive'&&previous.result?.status==='unsupported'&&previous.result?.missingCapability===true&&
   !previous.checkpoint?.prepared&&!previous.checkpoint?.reviewedContract&&!previous.checkpoint?.unresolved&&!previous.checkpoint?.publicationApproved&&!(previous.checkpoint?.priorQaElapsedSeconds>0);
  if(!previous||previous.status==='running'||!procedure?.trim()||previous.base!==base||
   previous.checkpoint?.evaluationDirectory||previous.result?.report?.results?.length||
   !(caseSelectionRepair!==undefined?repair:['qa-elapsed','qa-execution-failed','missing-cases','No target-compatible evaluation suite'].includes(reason)||missingCases&&previous.status==='learning'&&previous.result?.learningRequest))throw Error('Only audited pre-evaluation infrastructure failures may continue');
  const latest=this.db.prepare('SELECT id FROM skill_backlog_runs WHERE item_id=? ORDER BY rowid DESC LIMIT 1').get(previous.item_id);
  if(latest?.id!==id||fingerprint(this.get(previous.item_id))!==fingerprint(previous.snapshot))throw Error('Only the latest unchanged hypothesis may continue');
  if(this.db.prepare("SELECT 1 FROM skill_backlog_runs WHERE status='running'").get())throw Error('Background comparison already running');
  if(previous.elapsed>=previous.seconds)throw Error('Original hypothesis allowance exhausted');
  if(this.db.prepare("SELECT 1 FROM skill_backlog_runs WHERE json_extract(checkpoint,'$.continuedFrom')=?").get(id))throw Error('Continuation already exists');
  const next=randomUUID(),checkpoint={...previous.checkpoint,continuedFrom:id,continuationProcedure:procedure,continuationAt:new Date(this.now()).toISOString(),...(repair?{caseSelectionRepair:structuredClone(caseSelectionRepair)}:{})};
  checkpoint.priorQaElapsedSeconds=previous.checkpoint?.prepared?.qa?.elapsedSeconds??previous.checkpoint?.prepared?.elapsedSeconds??0;delete checkpoint.runningAt;delete checkpoint.prepared;
  this.db.prepare("INSERT INTO skill_backlog_runs(id,item_id,revision,base,seconds,elapsed,status,snapshot,checkpoint,owner) VALUES(?,?,?,?,?,?,'running',?,?,?)")
   .run(next,previous.item_id,previous.revision,base,previous.seconds,previous.elapsed,JSON.stringify(previous.snapshot),JSON.stringify(checkpoint),JSON.stringify(owner??null));
  this.store({...this.get(previous.item_id),status:'running'});return this.run(next);
 });}
 /** A baseline change invalidates samples, not the unfinished cohort's elapsed allocation. */
 requalify(id,{base,owner,procedure}={}){return this.atomic(()=>{
  const previous=this.run(id),item=previous&&this.get(previous.item_id),member=this.meta('hypothesis-cohort')?.members?.find(m=>m.id===previous?.item_id&&m.status==='pending'&&(m.runId===id||m.requalify===id||m.continuation?.id===id));
  if(!member||!previous||previous.status==='running'||['accepted','rejected'].includes(previous.status)||!procedure?.trim()||!base||previous.base===base||
   previous.elapsed>=previous.seconds||this.db.prepare('SELECT id FROM skill_backlog_runs WHERE item_id=? ORDER BY rowid DESC LIMIT 1').get(previous.item_id)?.id!==id||
   this.db.prepare("SELECT 1 FROM skill_backlog_runs WHERE status='running'").get()||hypothesisIdentity({...item,base:previous.base})!==hypothesisIdentity(previous.snapshot))throw Error('Only latest unchanged unfinished work may requalify after a baseline change');
  const rebased=this.rebase(item.id,base),next=randomUUID(),checkpoint={requalifiedFrom:id,continuationProcedure:procedure,proposalCharged:previous.checkpoint?.proposalCharged??false,
   priorQaElapsedSeconds:previous.checkpoint?.prepared?.qa?.elapsedSeconds??previous.checkpoint?.prepared?.elapsedSeconds??previous.checkpoint?.priorQaElapsedSeconds??0};
  if(this.completed(rebased))throw Error('Hypothesis already completed against the new baseline');
  this.db.prepare("INSERT INTO skill_backlog_runs(id,item_id,revision,base,seconds,elapsed,status,snapshot,checkpoint,owner) VALUES(?,?,?,?,?,?,'running',?,?,?)")
   .run(next,item.id,rebased.revision,base,previous.seconds,previous.elapsed,JSON.stringify(rebased),JSON.stringify(checkpoint),JSON.stringify(owner??null));
  this.store({...rebased,status:'running'});return this.run(next);
 });}
 checkpoint(id,{elapsed,checkpoint}={}){return this.atomic(()=>{const run=this.run(id);if(!run||run.status!=='running'||!Number.isFinite(elapsed)||elapsed<run.elapsed)throw Error('Running comparison and monotonic elapsed time required');this.db.prepare('UPDATE skill_backlog_runs SET elapsed=?,checkpoint=? WHERE id=?').run(elapsed,JSON.stringify(checkpoint??run.checkpoint),run.id);return this.run(run.id);});}
 finish(id,{status,result}={}){return this.atomic(()=>{
  const run=this.run(id);if(!run||run.status!=='running'||!statuses.has(status)||['running','queued','needs-review'].includes(status))throw Error('Running comparison and final disposition required');
  if(status==='accepted'&&run.elapsed>=run.seconds)throw Error('Expired comparison cannot promote');
  this.db.prepare('UPDATE skill_backlog_runs SET status=?,result=? WHERE id=?').run(status,JSON.stringify(result??null),run.id);
  const item=this.get(run.item_id),completed=item.revision!==run.revision&&this.completed(item);this.store({...item,status:item.revision===run.revision?status:completed?completed.status:item.objective==='harness'?'user-required':'queued',...(completed?{previousRun:completed.id}:{})});return this.run(run.id);
 });}
 status(){
  const attention=attentionStatus(this.db);
  const cursors=Object.fromEntries(this.db.prepare("SELECT key,value FROM skill_backlog_meta WHERE key LIKE 'cursor:%'").all().map(r=>[r.key.slice(7),parse(r.value)]));
  const journals=Object.fromEntries(['events','generation_events'].filter(name=>has(this.db,name)).map(name=>{
   const tail=this.db.prepare(`SELECT COALESCE(MAX(sequence),0) n FROM ${name}`).get().n,c=cursors[name];return[name,{...(c??{end:tail,history:0,fresh:tail}),tail,initialized:Boolean(c)}];
  }));
  return {dirtyFailures:this.db.prepare(`SELECT count(*) n FROM skill_backlog_failures f WHERE dirty=1 AND ${analysisScopeSql(this.db)} AND EXISTS(SELECT 1 FROM skill_backlog_occurrences o WHERE o.group_id=f.id AND ${activeOccurrenceSql(this.db,'o')})`).get().n,
   failures:this.db.prepare(`SELECT count(*) n FROM skill_backlog_occurrences o WHERE ${activeOccurrenceSql(this.db,'o')}`).get().n,
   historicalFailures:this.db.prepare('SELECT count(*) n FROM skill_backlog_occurrences').get().n,
   attention,
   items:this.db.prepare(`SELECT status,count(*) n FROM skill_backlog_items i WHERE ${activeItemSql(this.db,'i')} GROUP BY status`).all(),cursors,
   backfillComplete:Object.values(journals).every(c=>c.history>=c.end),ingestion:{journals,initialPending:Object.values(journals).some(c=>!c.initialized),freshPending:Object.values(journals).some(c=>c.fresh<c.tail)}};
 }
}
