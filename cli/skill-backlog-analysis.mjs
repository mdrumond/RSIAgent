/** Campaign-local failure analysis. Model work never holds the foreground ledger. */
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile,rename,open,realpath,readdir} from 'node:fs/promises';
import {join,resolve,sep,isAbsolute} from 'node:path';
import {guidanceEntries,validateClassificationBatch,enqueueLearningRequest} from '../controller/skill-review.mjs';
import {applyProposal,writeVersion} from '../controller/skill-evolution.mjs';
import {validateReviewProposal} from './skill-review.mjs';
import {runProposalQa} from './skill-review-qa.mjs';
const hash=x=>createHash('sha256').update(JSON.stringify(x)).digest('hex');
const load=async path=>{try{return JSON.parse(await readFile(path,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;return null;}};
const save=async(path,value)=>{await mkdir(resolve(path,'..'),{recursive:true});const temporary=path+'.tmp';await writeFile(temporary,JSON.stringify(value,null,2)+'\n');await rename(temporary,path);};
const unwrap=result=>result?.value??result;
async function operations(dependencies){const native=await import('./skill-review-native.mjs');return {classify:native.nativeClassifyBatch,review:native.nativeReview,qaAnswer:native.nativeQaAnswer,qaReview:native.nativeQaReview,session:native.reviewSession,...dependencies};}
function allocation(seconds,signal){
 if(!Number.isFinite(seconds)||seconds<=0)throw Error('Positive remaining elapsed allowance required');
 const started=performance.now(),remaining=()=>Math.max(0,seconds-(performance.now()-started)/1000);
 return {remaining,elapsed:()=>seconds-remaining(),call:async(fn,input,cap=120)=>{
  signal?.throwIfAborted();const left=Math.min(remaining(),cap);if(left<=0)throw Error('Analysis elapsed allocation exhausted');
  return unwrap(await fn({...input,seconds:left,signal:AbortSignal.any([...(signal?[signal]:[]),AbortSignal.timeout(Math.max(1,Math.ceil(left*1000)))])}));
 }};
}
function artifactPaths(value,key='',paths=[]){
 if(typeof value==='string'&&/^(path|log|stdout|stderr|stdout_path|stderr_path|evidence|correctness|performance)$/.test(key)&&value.length<=4096&&!/[\r\n]/.test(value)&&isAbsolute(value))paths.push(value);
 else if(value&&typeof value==='object')for(const [k,v]of Object.entries(value))artifactPaths(v,k,paths);
 return paths;
}
async function freezeGroups(root,groups,directory){
 const file=join(directory,'groups.json'),prior=await load(file);if(prior)return prior;
 await mkdir(join(directory,'evidence'),{recursive:true});const campaign=await realpath(root),frozen=structuredClone(groups);
 for(const group of frozen){delete group.classification;delete group.dirty;delete group.fresh;}
 for(const group of frozen)for(const observation of [...group.occurrences,...(group.linkedDiagnostics??[])]){
  const copies=[];
  for(const original of [...new Set([...(observation.artifacts??[]).map(a=>a.path),...artifactPaths(observation.data)])].slice(0,2)){
   let path;try{path=await realpath(original);}catch(e){if(e.code==='ENOENT')continue;throw e;}
   if(!path.startsWith(campaign+sep))continue;
   const handle=await open(path,'r');let bytes,info;try{info=await handle.stat();if(!info.isFile())continue;const length=Math.min(info.size,32768),buffer=Buffer.alloc(length);if(info.size>length){const half=length/2;await handle.read(buffer,0,half,0);await handle.read(buffer,half,half,info.size-half);}else await handle.read(buffer,0,length,0);bytes=buffer;}finally{await handle.close();}
   const copied=join(directory,'evidence',hash([observation.reference,path])+'.log');await writeFile(copied,bytes);
   copies.push({path:copied,sha256:createHash('sha256').update(bytes).digest('hex'),originalBytes:info.size,truncated:info.size>bytes.length,layout:info.size>bytes.length?'head-and-tail':'complete'});
  }
  observation.artifacts=copies;
 }
 await save(file,frozen);return frozen;
}
const learningIdentity=results=>hash(results.map(r=>({id:r.id,question:r.data.question,findings:r.data.findings,learningTarget:r.data.learningTarget})));
function learningFor(backlog,item){
 const db=backlog.db;if(!db.prepare("SELECT 1 FROM sqlite_master WHERE name='skill_learning_requests'").get())return [];
 return db.prepare("SELECT id,data,status FROM skill_learning_requests WHERE status='verified' AND (json_extract(data,'$.backlogId')=? OR EXISTS(SELECT 1 FROM json_each(data,'$.backlogIds') WHERE value=?))").all(item.id,item.id).map(r=>({...r,data:JSON.parse(r.data)}))
  .filter(r=>r.data.findings?.outcome==='passed').flatMap(r=>{
   const f=r.data.findings,attempt=db.prepare('SELECT id,project,outcome FROM attempts WHERE id=?').get(f.attemptId),project=attempt&&db.prepare('SELECT spec FROM projects WHERE id=?').get(attempt.project);
   if(!project||attempt.project!==f.projectId||attempt.outcome!=='passed'||!db.prepare("SELECT 1 FROM events WHERE attempt=? AND type='postprocess-completed'").get(attempt.id))return [];
   return [{...r,project:JSON.parse(project.spec),observations:db.prepare("SELECT sequence,type,data FROM events WHERE attempt=? AND type IN ('verification','accepted','stage-completed') ORDER BY sequence DESC LIMIT 12").all(attempt.id).map(e=>({reference:'events:'+e.sequence,type:e.type,attemptId:attempt.id,data:JSON.parse(e.data)}))}];
  });
}
async function freezeLearning(config,records,directory){
 const result=[],campaign=await realpath(config.root);
 for(const r of [...new Map(records.map(r=>[r.id,r])).values()]){
  const destination=join(directory,r.id),retained=await load(join(destination,'learning.json'));if(retained){result.push(retained);continue;}
  const copied=[],copy=async(path,name)=>{
   if(copied.length>=64)return null;
   let actual;try{actual=await realpath(path);}catch(e){if(e.code==='ENOENT')return null;throw e;}
   if(!actual.startsWith(campaign+sep))return null;
   const handle=await open(actual,'r');try{const info=await handle.stat();if(!info.isFile()||info.size>65536)return null;const target=join(destination,name);await mkdir(resolve(target,'..'),{recursive:true});const bytes=await handle.readFile();await writeFile(target,bytes);copied.push({path:target,sha256:createHash('sha256').update(bytes).digest('hex')});return target;}finally{await handle.close();}
  };
  const source=join(destination,'source');await mkdir(source,{recursive:true});let partial=false,scanned=0;
  const visit=async(path,relative='')=>{let list;try{list=await readdir(path,{withFileTypes:true});}catch(e){if(e.code==='ENOENT'){partial=true;return;}throw e;}
   for(const item of list){if(++scanned>128||copied.length>=32){partial=true;break;}if(item.name.startsWith('.')||['__pycache__','node_modules'].includes(item.name))continue;const rel=join(relative,item.name),pathName=join(path,item.name);
    if(item.isDirectory())await visit(pathName,rel);else if(item.isFile()&&/\.(py|cpp|c|h|hpp|cce|mjs|md|json)$/.test(item.name)){if(!await copy(pathName,join('source',rel)))partial=true;}}
  };
  if(r.project.source)await visit(r.project.source);else partial=true;
  const verifier=[];for(const [index,path]of (r.project.verifier?.files??[]).entries()){const target=await copy(path,join('verifier',String(index)+'-'+path.split(sep).at(-1)));if(target)verifier.push(target);else partial=true;}
  const groups=await freezeGroups(config.root,[{id:'learning-'+r.id,occurrences:r.observations,linkedDiagnostics:[]}],join(destination,'diagnostics'));
  let binding=null;try{const job=JSON.parse(await readFile(join(config.root,'attempts',r.data.findings.attemptId,'job.json'),'utf8'));const c=job.config??job;binding={target:c.target??null,environment:c.environment?.options?.mode??null,runtime:c.environment?.options?.runtime??null};}catch(e){if(e.code!=='ENOENT')throw e;}
  const receipt=groups[0].occurrences.find(o=>o.type==='verification'&&o.data.passed===true&&/^[a-f0-9]{64}$/.test(o.data.fingerprint??'')&&o.data.correctness?.passed===true&&o.data.correctness?.exitCode===0&&o.artifacts.length);
  const snapshot=[r.id,{result:{status:r.data.learningTarget==='device'?(binding?.target===config.target&&binding?.runtime&&receipt?'verified-device':'verified-project'):'verified-cpu',findings:r.data.findings,question:r.data.question,binding,
   project:{id:r.project.id,source,verifier:{files:verifier},partialEvidence:partial},evidence:[...copied.map(a=>({reference:'source:'+a.sha256,log:a.path,sha256:a.sha256})),...groups[0].occurrences.flatMap(o=>[o,...o.artifacts.map(a=>({reference:o.reference,log:a.path,sha256:a.sha256,truncated:a.truncated}))])]}}];
  await save(join(destination,'learning.json'),snapshot);result.push(snapshot);
 }
 return result;
}
function queueLearning(backlog,config,item,requests){
 if(item.objective==='harness')return [];
 return requests.map(request=>{
  const id=enqueueLearningRequest({db:backlog.db},{...request,target:config.target,backlogId:item.id,priority:item.priority,groupIds:request.groupIds??item.groupIds,intervention:null});
  const stored=JSON.parse(backlog.db.prepare('SELECT data FROM skill_learning_requests WHERE id=?').get(id).data);
  stored.backlogIds=[...new Set([stored.backlogId,...(stored.backlogIds??[]),item.id])];stored.priority=Math.max(stored.priority??0,item.priority);
  backlog.db.prepare('UPDATE skill_learning_requests SET data=? WHERE id=?').run(JSON.stringify(stored),id);return id;
 });
}
function priority(group){return Math.min(90,20+Math.min(30,group.count??group.occurrences.length)+(['execution-unresolved','project-invalid'].includes(group.type)?20:0));}

/** Analyze a bounded page. New evidence remains dirty after failures or elapsed expiry. */
export async function analyzeBacklog({config,backlog,directory,signal,seconds=120,limit=8},dependencies={}){
 const ops=await operations(dependencies),clock=allocation(seconds,signal),entries=await guidanceEntries(config.root,config.skills.activeVersion);
 const dirty=backlog.failures({limit,occurrenceLimit:6}),learningItems=backlog.list({status:'learning',limit:100}).map(item=>({item,results:learningFor(backlog,item)}))
  .filter(({item,results})=>results.length&&learningIdentity(results)!==item.evidence?.learningIdentity);
 const groups=[...new Map([...dirty,...learningItems.flatMap(({item})=>item.groupIds.map(id=>backlog.failure(id,{limit:6})))].filter(Boolean).map(g=>[g.id,g])).values()].slice(0,limit);
 if(!groups.length)return {analyzed:0,items:[],remainingSeconds:clock.remaining()};
 const batch=join(directory,hash([config.skills.activeVersion,groups,learningItems.map(({item,results})=>[item.id,results])])),frozen=await freezeGroups(config.root,groups,batch);
 const classifications=await clock.call(ops.classify,{directory:join(batch,'classify'),config,groups:frozen,entries,bucketPolicy:true,
  validate:value=>validateClassificationBatch(value,frozen,entries,{requireBucket:true})});
 const valid=validateClassificationBatch(classifications,frozen,entries,{requireBucket:true}).classifications,byGroup=Object.fromEntries(valid.map(c=>[c.groupId,c])),created=[];
 for(const group of frozen){
  signal?.throwIfAborted();if(clock.remaining()<=0)break;
  const classification=byGroup[group.id],base=config.skills.activeVersion,common={groupIds:[group.id],objective:classification.bucket,skill:classification.guidanceIds,
   priority:priority(group),reason:classification.reason,base},evidence={groups:[group]},classificationContext={classifications:{[group.id]:classification}};
  if(classification.intervention==='user-required'){
   created.push(backlog.upsert({...common,key:'harness:'+group.id,proposal:{action:'no-change',reason:classification.reason},evidence,preparation:classificationContext,status:'user-required'}));
   backlog.acknowledge(group.id,{classification,count:group.count});continue;
  }
  const learned=learningItems.filter(({item})=>item.groupIds.includes(group.id)),learning=await freezeLearning(config,learned.flatMap(({results})=>results),join(batch,'learning'));
  const proposalDirectory=join(batch,'proposals',group.id);await mkdir(proposalDirectory,{recursive:true});
  const spendFile=join(proposalDirectory,'writing-spend.json'),priorSpent=(await load(spendFile))?.elapsedSeconds??0,begun=performance.now();
  if(priorSpent>=1800){created.push(backlog.upsert({...common,key:'analysis:'+group.id,proposal:{action:'no-change',reason:'Proposal elapsed allowance exhausted'},evidence,preparation:{...classificationContext,proposalElapsedSeconds:priorSpent},status:'inconclusive'}));backlog.acknowledge(group.id,{classification,count:group.count});continue;}
  let review;try{review=validateReviewProposal(await clock.call(ops.review,{directory:proposalDirectory,config,groups:[group],entries,classifications:[classification],learning,
   validate:value=>validateReviewProposal(value,[group],byGroup)},Math.max(.001,1800-priorSpent)),[group],byGroup);}
  finally{await save(spendFile,{elapsedSeconds:priorSpent+(performance.now()-begun)/1000});}
  const records=await readFile(join(proposalDirectory,'retrievals.jsonl'),'utf8').catch(e=>{if(e.code==='ENOENT')return '';throw e;});
  const memoryEvidence=records.split('\n').filter(Boolean).flatMap(line=>JSON.parse(line).entries??[]);
  const actualRefs=new Set(memoryEvidence.map(e=>e.ref??`${e.collection}:${e.entry_id??e.chunk_id??e.id}`));
  if(review.action==='update'&&review.memoryRefs.some(ref=>!actualRefs.has(ref)))throw Error('Update requires memory retrieved in this proposal session');
  const preparation={...classificationContext,proposalDirectory,proposalElapsedSeconds:(await load(spendFile)).elapsedSeconds};
  const proposalEvidence={...evidence,memoryEvidence,...(learned.length?{learning,learningIdentity:learningIdentity(learned.flatMap(l=>l.results))}:{})};
  if(review.action==='no-change')for(const {item}of learned)created.push(backlog.upsert({...item,proposal:review,evidence:proposalEvidence,preparation,status:'inconclusive',reason:review.reason}));
  if(review.action==='update')for(const [index,change]of review.changes.entries()){
   const proposal={...review,changes:[change]},priorLearning=learned[index]?.item;
   const item=backlog.upsert({...common,key:priorLearning?.key??`update:${group.id}:${change.scope}:${change.id}`,skill:change.scope,proposal,evidence:proposalEvidence,preparation,status:'queued'});
   created.push(item);
  }
  const questions=[...(review.action==='learn'?[review]:[]),...(review.learningRequests??[])];
  for(const question of questions){
   const key=`learn:${group.id}:${hash([question.question,question.requiredEvidence])}`;
   const item=backlog.upsert({...common,key,proposal:{action:'learn',...question},evidence:proposalEvidence,preparation,status:'learning'});
   queueLearning(backlog,config,item,[question]);created.push(item);
  }
  backlog.acknowledge(group.id,{classification,count:group.count});
 }
 return {analyzed:frozen.filter(g=>!backlog.failure(g.id)?.dirty).length,items:created,elapsedSeconds:clock.elapsed(),remainingSeconds:clock.remaining()};
}

/** Rank open work only. Running comparisons retain their frozen inputs and finish first. */
export async function prioritizeBacklog({config,backlog,directory,signal,seconds=120},dependencies={}){
 const open=['queued','learning','needs-review'].flatMap(status=>backlog.list({status,limit:1000}));if(!open.length)return [];
 const seen=backlog.meta('priority-inputs')??{},signature=item=>hash([item.fingerprint,item.evidence?.groups?.map(g=>[g.id,g.count])]);
 const changed=open.filter(i=>seen[i.id]!==signature(i)).sort((a,b)=>b.priority-a.priority||b.created-a.created);
 const cursor=backlog.meta('priority-cursor')??0,ordered=open.slice().sort((a,b)=>b.priority-a.priority||a.created-b.created);
 const refresh=changed.length?changed.slice(0,20):ordered.slice(cursor,cursor+20);
 const items=[...new Map([...refresh,...ordered.slice(0,4)].map(i=>[i.id,i])).values()];
 const ops=await operations(dependencies),clock=allocation(seconds,signal),input=items.map(i=>({id:i.id,objective:i.objective,skill:i.skill,priority:i.priority,reason:i.reason,
  proposal:i.proposal,evidence:i.evidence?.groups?.map(g=>({id:g.id,type:g.type,message:g.message,count:g.count}))}));
 const validate=value=>{
  const rows=value?.priorities;if(!Array.isArray(rows)||rows.length!==items.length||new Set(rows.map(r=>r.id)).size!==items.length||rows.some(r=>!items.some(i=>i.id===r.id)||!Number.isFinite(r.priority)||r.priority<0||r.priority>100||!r.reason?.trim()))throw Error('Rank every open hypothesis once with priority 0–100 and evidence-supported reason');return rows;
 };
 const rows=await clock.call(ops.session,{directory:join(directory,hash(input)),config,knowledge:false,
  prompt:{task:'Rank every open hypothesis by expected impact: severity, recurrence, recoverable wasted elapsed time, evidence strength, and readiness. New urgent evidence may rank above all existing work. Do not use cumulative actor clocks as measured operation duration. Return {priorities:[{id,priority:0..100,reason}]}. Keep each reason to at most two short sentences and cite supplied evidence references when available; do not write an essay. Harness problems remain user handled; do not invent evidence.',items:input},validate});
 const ranked=validate(Array.isArray(rows)?{priorities:rows}:rows),result=backlog.rank(ranked);
 for(const item of items)seen[item.id]=signature(item);backlog.meta('priority-inputs',seen);backlog.meta('priority-cursor',cursor+20>=ordered.length?0:cursor+20);return result;
}

/** Prepare one candidate against the current baseline; approval admits A/B, never publication. */
export async function prepareHypothesis({config,backlog,item,directory,signal,remainingSeconds},dependencies={}){
 const ops=await operations(dependencies),clock=allocation(remainingSeconds,signal),identity=hash([item.id,item.revision,item.base,item.proposal,item.evidence]);
 await mkdir(directory,{recursive:true});const file=join(directory,'preparation.json'),prior=await load(file);
 if(prior){if(prior.identity!==identity)throw Error('Hypothesis preparation inputs changed');return {...prior,recovered:true};}
 if(item.base!==config.skills.activeVersion)throw Error('Rebase and requalify hypothesis against the current active skill');
 if(item.objective==='harness')return {status:'user-required',reason:'Harness faults require user intervention'};
 const groups=item.evidence?.groups??[],classifications=item.preparation?.classifications??{},entries=await guidanceEntries(config.root,item.base);
 const validateProposal=value=>validateReviewProposal(value,groups,classifications),proposal=validateProposal(structuredClone(item.proposal));
 if(proposal.action!=='update')return {status:'learning',reason:'Supported skill instruction required before comparison'};
 const cited=new Set((item.evidence.memoryEvidence??[]).map(e=>e.ref??`${e.collection}:${e.entry_id??e.chunk_id??e.id}`));
 if(proposal.memoryRefs.some(ref=>!cited.has(ref)))throw Error('Hypothesis memory evidence is missing');
 const qa=await runProposalQa({directory:join(directory,'qa'),config,proposal,groups,entries,objective:item.objective,target:config.target,
  memoryEvidence:item.evidence.memoryEvidence??[],proposalDirectory:item.preparation?.proposalDirectory,learning:item.evidence.learning??[],validateProposal,
  signal,seconds:Math.min(120,clock.remaining())},{qaAnswer:args=>clock.call(ops.qaAnswer,args,args.seconds),qaReview:args=>clock.call(ops.qaReview,args,args.seconds)});
 if(qa.verdict!=='approved'){
  queueLearning(backlog,config,item,qa.learningRequests??[]);
  return {status:qa.verdict==='learn'?'learning':'inconclusive',qa,reason:qa.reason,elapsedSeconds:clock.elapsed()};
 }
 if(clock.remaining()<=0)throw Error('Hypothesis preparation elapsed allocation exhausted');
 const final=validateProposal(qa.proposal),updated=applyProposal(entries,final.changes,{allowPruning:false,history:[]});
 const candidate=await writeVersion(config.root,updated,join(config.root,'skills','versions',item.base));
 const result={identity,status:'ready',active:item.base,candidate,proposal:final,qa,elapsedSeconds:clock.elapsed()};await save(file,result);return result;
}
