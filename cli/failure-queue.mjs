/** Operator-only replacement of planning work, never scientific results. */
import {createHash} from 'node:crypto';
import {SkillBacklog} from '../controller/skill-backlog.mjs';
import {attentionRevision,activeOccurrenceSql} from '../controller/failure-attention.mjs';
import {enqueueLearningRequest} from '../controller/skill-review.mjs';
import {hypothesisCaseReadiness} from './skill-priority-cases.mjs';
const has=(db,name)=>Boolean(db.prepare("SELECT 1 FROM sqlite_master WHERE name=?").get(name));
const hash=v=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
const meta=(db,k)=>has(db,'skill_backlog_meta')?JSON.parse(db.prepare('SELECT value FROM skill_backlog_meta WHERE key=?').get(k)?.value??'null'):null;
const cutoffs=db=>Object.fromEntries(['events','generation_events','attempts','generation_drafts'].filter(t=>has(db,t)).map(t=>[t,db.prepare(`SELECT COALESCE(MAX(${['attempts','generation_drafts'].includes(t)?'rowid':'sequence'}),0) n FROM ${t}`).get().n]));
const plan=db=>{
 const requests=has(db,'skill_learning_requests')?db.prepare("SELECT * FROM skill_learning_requests WHERE status IN ('pending','claimed') ORDER BY id").all():[];
 const active=r=>r.draft&&(db.prepare('SELECT 1 FROM projects WHERE id=?').get(r.draft)||has(db,'generation_drafts')&&db.prepare("SELECT 1 FROM generation_drafts WHERE id=? AND json_extract(data,'$.status') IN ('draft','ready')").get(r.draft));
 const protectedIds=new Set(requests.filter(active).flatMap(r=>{const d=JSON.parse(r.data);return [d.backlogId,...(d.backlogIds??[])];}));
 return {revision:attentionRevision(db),cutoffs:cutoffs(db),items:db.prepare("SELECT * FROM skill_backlog_items WHERE status IN ('queued','learning','needs-review') ORDER BY id").all().filter(r=>!protectedIds.has(r.id)),
  requests:requests.filter(r=>!active(r)),preservedRequests:requests.filter(active),inconclusiveRequests:has(db,'skill_learning_requests')?db.prepare("SELECT * FROM skill_learning_requests WHERE status='inconclusive' ORDER BY id").all():[],cohort:meta(db,'hypothesis-cohort'),continuations:meta(db,'procedural-continuations')};
};
export async function auditQueueRebuild(config,db,input,{readiness=hypothesisCaseReadiness}={}){
 if(!input.id?.trim()||!input.operator?.reason?.trim()||!Array.isArray(input.seeds)||input.seeds.length>5||new Set(input.seeds.map(s=>s.id)).size!==input.seeds.length)throw Error('Rebuild requires an operator reason and at most five distinct seeds');
 if(meta(db,'microlearning-active')||db.prepare("SELECT 1 FROM skill_backlog_runs WHERE status='running'").get())throw Error('Settle active learning/comparisons before rebuilding');
 const backlog=new SkillBacklog({db},{target:config.target}),snapshot=plan(db),seeds=[];
 for(const seed of input.seeds){
  if(!seed.id?.trim()||!['correctness','optimization'].includes(seed.objective)||!seed.groupIds?.length)throw Error('Seed requires identity, correctness/optimization objective and active failure groups');
  const groups=seed.groupIds.map(id=>backlog.failure(id,{limit:6}));
  if(groups.some(g=>!g?.count))throw Error('Seed requires active evidence; retired incidents cannot seed new work');
  if(groups.some(g=>g.classification?.bucket==='harness'||g.classification?.intervention==='user-required'))throw Error('Harness incidents stay in human review');
  if(!seed.evidenceRefs?.length)throw Error('Seed evidence must name active supplied occurrences');
  for(const reference of seed.evidenceRefs){
   const row=db.prepare(`SELECT o.group_id,o.data FROM skill_backlog_occurrences o WHERE o.reference=? AND ${activeOccurrenceSql(db,'o')}`).get(reference),group=groups.find(g=>g.id===row?.group_id);
   if(!group)throw Error('Seed evidence must name active supplied occurrences');
   if(!group.occurrences.some(o=>o.reference===reference))group.occurrences.push(JSON.parse(row.data));
  }
  const proposal=seed.proposal??{action:'learn',question:seed.question,hypothesis:seed.hypothesis,requiredEvidence:seed.requiredEvidence,learningTarget:seed.learningTarget,workKind:'microlearning'};
  if(!['learn','update'].includes(proposal.action)||proposal.action==='learn'&&(!proposal.question?.trim()||!proposal.requiredEvidence?.trim()||!['cpu','device'].includes(proposal.learningTarget)))throw Error('Seed requires a grounded proposal or bounded learning question');
  const projects=new Set(),attempts=new Set();
  for(const id of seed.groupIds)for(const row of db.prepare(`SELECT o.data FROM skill_backlog_occurrences o WHERE group_id=? AND ${activeOccurrenceSql(db,'o')}`).all(id)){const o=JSON.parse(row.data);if(o.projectId)projects.add(o.projectId);if(o.attemptId)attempts.add(o.attemptId);}
  const item={...seed,target:config.target,base:config.skills.activeVersion,proposal,evidence:{groups,frequency:{projects:projects.size,attempts:attempts.size}}};
  seeds.push({...item,readiness:await readiness({config,backlog,item})});
 }
 seeds.sort((a,b)=>Number(!!b.readiness.ready)-Number(!!a.readiness.ready)||Number(!!b.readiness.supported)-Number(!!a.readiness.supported)||b.evidence.frequency.projects-a.evidence.frequency.projects||b.evidence.frequency.attempts-a.evidence.frequency.attempts||a.id.localeCompare(b.id));
 const humanReview=input.openInfrastructureBlocked??[];
 if(!Array.isArray(humanReview)||humanReview.some(r=>!r.requestId||!r.diagnosis?.trim()||!has(db,'skill_learning_requests')||!db.prepare('SELECT 1 FROM skill_learning_requests WHERE id=?').get(r.requestId)))throw Error('Infrastructure review requires known request IDs and operator diagnosis');
 const {auditSignature,action,...manifest}=input;return {id:input.id,signature:hash([snapshot,manifest,seeds]),snapshot,seeds,humanReview:humanReview.map(r=>({...r,attention:'open',retired:false}))};
}
export async function rebuildQueue(config,db,input,dependencies){
 const audit=await auditQueueRebuild(config,db,input,dependencies);
 if(input.auditSignature!==audit.signature)throw Error('Rebuild requires the unchanged dry-run signature');
 return new SkillBacklog({db},{target:config.target}).atomic(()=>{
  if(hash(plan(db))!==hash(audit.snapshot))throw Error('Planning state changed; audit again');
  db.exec('CREATE TABLE IF NOT EXISTS skill_queue_archives(id TEXT PRIMARY KEY,data TEXT NOT NULL)');
  if(db.prepare('SELECT 1 FROM skill_queue_archives WHERE id=?').get(input.id))throw Error('Rebuild identity already applied');
  db.prepare('INSERT INTO skill_queue_archives VALUES (?,?)').run(input.id,JSON.stringify({at:new Date().toISOString(),operator:input.operator,...audit}));
  for(const row of audit.snapshot.items){const data={...JSON.parse(row.data),status:'archived',archive:{id:input.id,reason:input.operator.reason}};db.prepare("UPDATE skill_backlog_items SET status='archived',data=? WHERE id=?").run(JSON.stringify(data),row.id);}
  for(const row of audit.snapshot.requests)db.prepare("UPDATE skill_learning_requests SET status='archived' WHERE id=?").run(row.id);
  const backlog=new SkillBacklog({db},{target:config.target});backlog.meta('queue-epoch',input.id);
  const cutoffs=audit.snapshot.cutoffs;
  backlog.meta('queue-rebuild',{id:input.id,groupIds:[...new Set(audit.seeds.flatMap(s=>s.groupIds))],cutoffs});
  for(const k of ['hypothesis-cohort','procedural-continuations','learning-scan'])backlog.meta(k,null);
  const items=audit.seeds.map((seed,index)=>{
   const {readiness,...value}=seed,item=backlog.upsert({...value,key:'seed:'+seed.id,priority:100-index,status:seed.proposal.action==='update'&&readiness.ready?'queued':'learning',reason:JSON.stringify({ready:!!readiness.ready,supported:!!readiness.supported,...seed.evidence.frequency})});
   if(item.status==='archived')throw Error('Seed repeats archived proposal; provide a distinct grounded question');
   if(seed.proposal.action==='learn')enqueueLearningRequest({db},{...seed.proposal,target:config.target,groupIds:seed.groupIds,backlogId:item.id,priority:item.priority});return item.id;
  });
  return {applied:true,id:input.id,archivedItems:audit.snapshot.items.length,archivedRequests:audit.snapshot.requests.length,seedIds:items};
 });
}
