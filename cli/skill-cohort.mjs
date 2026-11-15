/** A small completion-first cohort, persisted only in its campaign's backlog. */
import {randomUUID} from 'node:crypto';
const key='hypothesis-cohort';
const done=new Set(['accepted','rejected','inconclusive','learning','user-required']);
export function refreshCohort(backlog){
 const cohort=backlog.meta(key);if(!cohort)return null;
 for(const member of cohort.members){
  if(member.status!=='pending')continue;
  const item=backlog.get(member.id),run=member.runId&&backlog.run(member.runId);
  if(!item){member.status='blocked';member.reason='Hypothesis no longer exists';}
  else if(run&&done.has(run.status)){
   if(run.result?.reason==='active-skill-changed'){member.requalify=run.id;delete member.runId;}
   else{member.status=['accepted','rejected'].includes(run.status)?'completed':'blocked';member.reason=run.result?.reason??run.status;}
  }else if(!member.runId&&!member.continuation&&!member.requalify&&done.has(item.status)){
   member.status=['accepted','rejected'].includes(item.status)?'completed':'blocked';member.reason=item.status;
  }
 }
 cohort.status=cohort.members.some(m=>m.status==='pending')?'running':'complete';backlog.meta(key,cohort);return cohort;
}
export function cohortStatus(backlog){
 const cohort=backlog.meta(key);if(!cohort)return null;
 const counts={pending:0,completed:0,blocked:0};for(const m of cohort.members)counts[m.status]++;
 const pending=cohort.members.find(m=>m.status==='pending'),active=backlog.list({status:'running',limit:1})[0];
 return {...cohort,counts,nextAction:active?'resume-comparison':pending?'resolve-cases-and-compare':'analyze-new-evidence',
  deferredEvidence:Math.max(0,backlog.status().failures-cohort.failureCount)};
}
function candidates(backlog,base){
 const choices=new Map([...backlog.list({status:'queued',limit:Number.MAX_SAFE_INTEGER}),...backlog.list({status:'needs-review',limit:Number.MAX_SAFE_INTEGER})].map(item=>[item.id,{item}]));
 for(const continuation of backlog.meta('procedural-continuations')??[]){
  const run=backlog.run(continuation.id),item=run&&backlog.get(run.item_id);
  if(item&&continuation.base===base&&run.elapsed<run.seconds&&!choices.has(item.id))choices.set(item.id,{item,continuation});
 }
 return [...choices.values()].filter(({item})=>item.objective!=='harness');
}
export const hasCohortCandidates=(backlog,base)=>candidates(backlog,base).length>0;
export async function selectCohort(backlog,config,readiness){
 let cohort=refreshCohort(backlog);
 if(cohort?.status==='complete'){backlog.meta('hypothesis-cohort-last',cohort);backlog.meta(key,null);cohort=null;}
 if(!cohort){
  const choices=candidates(backlog,config.skills.activeVersion);
  for(const choice of choices)Object.assign(choice,await readiness({config,item:choice.item}));
  choices.sort((a,b)=>Number(Boolean(b.ready))-Number(Boolean(a.ready))||Number(Boolean(b.supported))-Number(Boolean(a.supported))||b.item.priority-a.item.priority||a.item.created-b.item.created);
  if(!choices.length)return null;
  cohort={id:randomUUID(),base:config.skills.activeVersion,createdAt:new Date().toISOString(),status:'running',failureCount:backlog.status().failures,
   members:choices.slice(0,5).map(({item,continuation,ready,supported})=>({id:item.id,priority:item.priority,status:'pending',ready:Boolean(ready),supported:Boolean(supported),...(continuation?{continuation}:{})}))};
  backlog.meta(key,cohort);
 }
 return cohort;
}
export async function claimCohort(backlog,config,owner,readiness){
 const cohort=await selectCohort(backlog,config,readiness);if(!cohort)return null;
 const pending=cohort.members.filter(m=>m.status==='pending');
 for(const member of pending)Object.assign(member,await readiness({config,item:backlog.get(member.id)}));
 pending.sort((a,b)=>Number(Boolean(b.ready))-Number(Boolean(a.ready))||Number(Boolean(b.supported))-Number(Boolean(a.supported))||b.priority-a.priority);
 for(const member of pending){
  try{
   let item=backlog.get(member.id);
   if(member.requalify||member.continuation){
    const source=member.requalify??member.continuation.id,old=backlog.run(source);
    if(old.base!==config.skills.activeVersion)backlog.requalify(source,{base:config.skills.activeVersion,owner,procedure:'Requalify unfinished cohort after active skill promotion'});
    else backlog.continue(source,{...member.continuation,owner});
    if(member.continuation)backlog.meta('procedural-continuations',(backlog.meta('procedural-continuations')??[]).filter(q=>q.id!==member.continuation.id));
    delete member.continuation;delete member.requalify;
   }else if(item.base!==config.skills.activeVersion){item=backlog.rebase(item.id,config.skills.activeVersion);}
   if(item.status==='needs-review')backlog.revise(item.id,{status:'queued'});
   const selected=backlog.claim({base:config.skills.activeVersion,owner,itemId:member.id});
   if(selected){member.runId=selected.run.id;cohort.base=config.skills.activeVersion;backlog.meta(key,cohort);return selected;}
   member.status='blocked';member.reason='Hypothesis is no longer eligible';
  }catch(error){
   member.status='blocked';member.reason=error.message;
   if(member.continuation){
    backlog.meta('procedural-continuations',(backlog.meta('procedural-continuations')??[]).filter(q=>q.id!==member.continuation.id));
    backlog.db.prepare("INSERT INTO events(type,data,created) VALUES('skill-continuation-skipped',?,?)").run(JSON.stringify({id:member.continuation.id,reason:error.message}),new Date().toISOString());
   }
  }
  backlog.meta(key,cohort);
 }
 refreshCohort(backlog);return null;
}
export function attachRetainedCohort(backlog,run){
 const cohort=backlog.meta(key)??{id:randomUUID(),base:run.base,createdAt:new Date().toISOString(),status:'running',failureCount:backlog.status().failures,members:[{id:run.item_id,priority:run.snapshot.priority,status:'pending'}]},member=cohort.members.find(m=>m.id===run.item_id);if(member){member.runId=run.id;backlog.meta(key,cohort);}
}
