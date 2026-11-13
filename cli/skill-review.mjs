/** Isolated, resumable failure -> memory -> guidance/learning -> A/B review. */
import {mkdir,readFile,writeFile,rename,cp,realpath,access,rm} from 'node:fs/promises';
import {join,resolve,relative,sep} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {DatabaseSync} from 'node:sqlite';
import {State} from '../controller/state.mjs';
import {inventoryFailures,guidanceEntries,validateClassification,validateClassificationBatch,validateReview,enqueueLearningRequest} from '../controller/skill-review.mjs';
import {applyProposal,writeVersion} from '../controller/skill-evolution.mjs';
import {validateSkillGrounding} from '../controller/skill-evidence.mjs';
import {publishSkillVersion} from './skills.mjs';
import {evaluationSettings,retrievalIdentity} from '../controller/skill-objectives.mjs';
import {stages as workflowStages} from '../controller/workflow.mjs';
import {runProposalQa,qaIdentity} from './skill-review-qa.mjs';
import {withSkillLock} from './evolution-lock.mjs';
const load=async path=>JSON.parse(await readFile(path,'utf8'));
async function optional(path){try{return await load(path);}catch(e){if(e.code!=='ENOENT')throw e;return null;}}
async function save(path,value){const tmp=path+'.'+randomUUID();await writeFile(tmp,JSON.stringify(value,null,2)+'\n');await rename(tmp,path);}
async function fileHash(path){const d=createHash('sha256');for await(const chunk of createReadStream(path))d.update(chunk);return d.digest('hex');}
async function optionalHash(path){try{return await fileHash(path);}catch(error){if(error.code!=='ENOENT')throw error;return null;}}
async function dependentInputs(root){
 const db=new DatabaseSync(join(root,'state.sqlite'),{readOnly:true});let attempts;try{attempts=db.prepare('SELECT id FROM attempts ORDER BY id').all();}finally{db.close();}
 const jobs=[];for(const {id}of attempts)jobs.push({id,sha256:await optionalHash(join(root,'attempts',id,'job.json'))});
 const indexPath=join(root,'evidence-files.json'),indexHash=await optionalHash(indexPath),logs=[];
 if(indexHash){const index=await load(indexPath);for(const [original,record]of Object.entries(index).sort(([a],[b])=>a.localeCompare(b)))if(record.path){
  const actual=await fileHash(join(root,record.path));if(record.sha256&&record.sha256.toLowerCase()!==actual)throw Error('Copied evidence SHA256 mismatch: '+original);
  logs.push({original,path:record.path,sha256:actual});
 }}
 return {jobs,evidenceIndex:indexHash,logs};
}
const hash=x=>createHash('sha256').update(JSON.stringify(x)).digest('hex');
const stages=['inventory','classify','propose','qa','learn','evaluate','report','enqueue'];
async function forkCampaign(source,directory,options){
 let config=await optional(join(directory,'instance.json'));if(config)return {...config,root:directory,...(options.evaluations?{skill_review_evaluations:options.evaluations}:{})};
 const original=await load(join(source.root,'instance.json'));await mkdir(directory,{recursive:true});
 await cp(join(source.root,'skills','versions',source.activeVersion),join(directory,'skills','versions',source.activeVersion),{recursive:true});
 const knowledge=join(directory,'knowledge.sqlite');
 try{await access(knowledge);const retained=new DatabaseSync(knowledge,{readOnly:true});try{if(retained.prepare('PRAGMA quick_check').get().quick_check!=='ok')throw Error('Retained review knowledge copy is invalid');}finally{retained.close();}}
 catch(error){if(error.code!=='ENOENT')throw error;
  const temporary=join(directory,'knowledge-copy-'+randomUUID()+'.sqlite'),db=new DatabaseSync(join(source.root,'knowledge.sqlite'),{readOnly:true});
  try{db.prepare('VACUUM INTO ?').run(temporary);await rename(temporary,knowledge);}finally{db.close();await rm(temporary,{force:true});}
 }
 config={format:1,revision:0,root:directory,mode:'phase1',target:source.target,actor_seconds:300,
  actor:options.profile??original.skill_writer_actor??original.actor,skill_writer_actor:options.profile??original.skill_writer_actor??original.actor,
  python:options.python??original.python,embeddings:options.embeddings??original.embeddings,embedding_factory:options.embedding_factory??original.embedding_factory,
  ...(options.evaluations?{skill_review_evaluations:options.evaluations}:{}),skills:{activeVersion:source.activeVersion,tuningDisabled:true},environment:{mode:'local'},projects:[],recovery:false,concurrency:1};
 new State(join(directory,'state.sqlite')).close();await save(join(directory,'instance.json'),config);return config;
}
export function validateReviewProposal(result,groups,classifications){
 validateReview(result,groups);if(result.action!=='update')return result;
 const scopes=['global-kdb',...workflowStages];
 for(const change of result.changes)if(!scopes.includes(change.scope))throw Error('Guidance scope must be one of '+scopes.join(', ')+'. Evaluation buckets correctness, optimization and harness are not workflow guidance scopes.');
 const observations=groups.flatMap(g=>[...g.occurrences,...(g.linkedDiagnostics??[])].map(o=>({reference:o.reference,...(classifications[g.id]?.cause==='harness'||classifications[g.id]?.bucket==='harness'?{responsibility:'harness'}:{})})));
 const responses=['actor-change','report-blocker','reserve-validation-runway','retrieve-scoped-evidence'];
 for(const change of result.changes)if(change.response!==undefined&&!responses.includes(change.response))throw Error('response must be one of '+responses.join(', '));
 try{result.changes=validateSkillGrounding(result.changes,observations);}
 catch(error){
  const known=new Set(observations.map(o=>o.reference)),unknown=result.changes.flatMap(c=>(c.evidenceRefs??[]).filter(ref=>!known.has(ref)));
  if(unknown.length)throw new Error('Unknown evidenceRefs: '+JSON.stringify(unknown)+'. Use exact occurrence/linked-diagnostic reference IDs from input.json; KDB references belong in memoryRefs.',{cause:error});
  const scoped=result.changes.find(c=>['report-blocker','reserve-validation-runway'].includes(c.response)&&!['global-kdb','preflight'].includes(c.scope));
  if(scoped)throw new Error('response='+scoped.response+" requires scope='global-kdb' or scope='preflight'; change the scope field, not the changes-array order.",{cause:error});
  throw error;
 }
 return result;
}
function reviewReport(state){
 const groups=state.inventory?.groups??[];
 return {status:state.status,elapsedSeconds:state.elapsedSeconds,remainingSeconds:Math.max(0,state.seconds-(Date.now()-state.startedAt)/1000),failureGroups:groups.length,
   buckets:Object.fromEntries(['harness','correctness','optimization','legacy'].map(bucket=>[bucket,Object.values(state.classifications).filter(c=>(c.bucket??'legacy')===bucket).length])),
   failureObservations:groups.reduce((n,g)=>n+g.occurrences.length,0),classified:Object.keys(state.classifications).length,
   dispositions:Object.values(state.reviews).reduce((out,r)=>(out[r.action]=(out[r.action]??0)+1,out),{}),
   qa:Object.entries(state.qa??{}).map(([key,q])=>({key,verdict:q.verdict,reason:q.reason,elapsedSeconds:q.elapsedSeconds,approval:q.approval})),
   reviewFailures:Object.entries(state.reviews).filter(([,r])=>r.action==='inconclusive').map(([key,r])=>({key,directory:r.directory,error:r.error,reason:r.reason})),
   userInterventions:Object.values(state.classifications).filter(c=>c.intervention==='user-required').map(c=>c.groupId),
   learning:Object.values(state.learning).map(({id,status,question,result,intervention})=>({id,status,question,...(intervention?{intervention}:{}),...(result?.reason?{reason:result.reason}:{}),...(result?.error?{error:result.error}:{})})),
   candidates:Object.entries(state.candidates).map(([key,c])=>({key,version:c.version,decision:c.decision,published:c.published??false})),
   stats:state.stats,timings:state.timings??[],claim:'Isolated review; live campaigns unchanged. CPU microbenchmarks do not establish GPU kernel improvement.'};

}
export async function runReview(options,dependencies={}){
 if(!Array.isArray(options.sources)||!options.sources.length||!options.output)throw Error('Review requires snapshot sources and separate output');
 const output=resolve(options.output),sources=await Promise.all(options.sources.map(async s=>({...(typeof s==='string'?{}:s),root:await realpath(typeof s==='string'?s:s.root)})));
 for(const source of sources){const rel=relative(source.root,output);if(!rel||rel!=='..'&&!rel.startsWith('..'+sep))throw Error('Review output must not belong to source campaign');}
 const seconds=options.seconds??7200;if(!Number.isFinite(seconds)||seconds<=0)throw Error('Positive review elapsed allowance required');
 if(![options.maxCandidates??3,options.maxLearning??2].every(n=>Number.isSafeInteger(n)&&n>=0))throw Error('Nonnegative candidate and learning project targets required');
 const stage=options.stage??'all';if(stage!=='all'&&!stages.includes(stage))throw Error('Unknown review stage');
 let frozenEvaluations;
 if(options.evaluations){frozenEvaluations={};for(const objective of Object.keys(options.evaluations)){
  if(!['correctness','optimization'].includes(objective))throw Error('Unknown evaluation objective');
  const value=evaluationSettings(options.evaluations,objective);frozenEvaluations[objective]={...value,...(value.knowledge?.identity!==undefined?{knowledge:retrievalIdentity(value.knowledge)}:{}),tasks:await Promise.all(value.tasks.map(async task=>({...task,checker:await readFile(task.checker,'utf8'),...(task.pairedChecker?{pairedChecker:await readFile(task.pairedChecker,'utf8')}:{} )})))};
 }}
 if(options.bucketPolicy&&['all','evaluate'].includes(stage))for(const objective of ['correctness','optimization'])evaluationSettings(options.evaluations,objective);
 await mkdir(output,{recursive:true});
 const frozen=[],inputErrors=[];for(const source of sources){const cfg=await load(join(source.root,'instance.json'));let dependencies;try{dependencies=await dependentInputs(source.root);}catch(error){if(stage!=='report')throw error;inputErrors.push(error.message);}
 frozen.push({root:source.root,config:hash(cfg),state:await fileHash(join(source.root,'state.sqlite')),knowledge:await fileHash(join(source.root,'knowledge.sqlite')),guidance:hash(await guidanceEntries(source.root,cfg.skills.activeVersion)),dependencies});}
 const lockConfig={root:output,skill_queue_seconds:1};new State(join(output,'state.sqlite')).close();
 return withSkillLock(lockConfig,async()=>{
  const file=join(output,'review.json'),identity={sources,frozen,...(options.bucketPolicy?{bucketPolicy:true}:{}),profile:options.profile??null,python:options.python??null,seconds,maxCandidates:options.maxCandidates??3,maxLearning:options.maxLearning??2},signature=hash(identity),
   legacySignature=hash({...identity,frozen:frozen.map(({dependencies,...legacy})=>legacy)});
  let state=await optional(file);
  if(state&&(state.signature!==signature||inputErrors.length)){
   let proven=false;
   if(!inputErrors.length&&state.signature===legacySignature){
    const current=await inventoryFailures(sources),artifacts=current.groups.flatMap(g=>[...g.occurrences,...(g.linkedDiagnostics??[])].flatMap(o=>o.artifacts??[]));
    proven=hash(current)===hash(state.inventory)&&artifacts.every(a=>typeof a.sha256==='string');
   }
   if(proven){
    (state.fingerprintMigrations??=[]).push({from:state.signature,to:signature,at:new Date().toISOString(),proof:'same-effective-inventory-and-verified-copied-artifact-digests'});state.signature=signature;
   }else if(stage==='report'&&state.fingerprintVersion!==2){
    return {...reviewReport(state),legacyInspection:true,resumeAllowed:false,inputErrors:inputErrors.length?inputErrors:['Legacy input identity cannot be proven unchanged; model resume requires a fresh snapshot/review']};
   }else throw Error(state.fingerprintVersion===2?'Frozen review inputs or elapsed allocation changed':'Legacy review inputs cannot be proven unchanged; inspect report only or use a fresh snapshot');
  }
  if(inputErrors.length)throw Error(inputErrors.join('; '));
  state??={format:1,signature,startedAt:Date.now(),seconds,inventory:null,classifications:{},reviews:{},learning:{},candidates:{},errors:[],stats:[]};
  if(frozenEvaluations){const evaluationSignature=hash(frozenEvaluations);if(state.evaluationSignature&&state.evaluationSignature!==evaluationSignature)throw Error('Frozen review evaluation inputs or allocations changed');state.evaluationSignature=evaluationSignature;}
  state.fingerprintVersion=2;state.legacySignature=legacySignature;
  const remaining=()=>Math.max(0,seconds-(Date.now()-state.startedAt)/1000);
  const persist=async()=>{state.elapsedSeconds=(Date.now()-state.startedAt)/1000;await save(file,state);};
  const native=await import('./skill-review-native.mjs'),ops={classify:native.nativeClassify,classifyBatch:native.nativeClassifyBatch,review:native.nativeReview,qaAnswer:native.nativeQaAnswer,qaReview:native.nativeQaReview,learn:native.nativeLearn,evaluate:native.nativeEvaluate,evaluationSettled:native.evaluationSettled,...dependencies};
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(Error('Review elapsed allocation exhausted')),Math.min(2147483647,remaining()*1000));
  const signal=options.signal?AbortSignal.any([options.signal,controller.signal]):controller.signal;
  const invoke=async(name,input,cap=240)=>{signal.throwIfAborted();if(remaining()<=0)throw Error('Review elapsed allocation exhausted');
   const allocation=Math.min(cap,remaining()),bounded=AbortSignal.any([signal,AbortSignal.timeout(Math.max(1,Math.ceil(allocation*1000)))]);
   const started=performance.now(),telemetry={stage:name,key:input.key,...(input.groups?{groupCount:input.groups.length,groupIds:input.groups.map(g=>g.id)}:input.group?{groupCount:1,groupIds:[input.group.id]}:{}),...(input.sourceId?{sourceId:input.sourceId,guidanceVersion:input.guidanceVersion}:{})};
   try{const result=await ops[name]({...input,signal:bounded,seconds:allocation});if(result.stats)state.stats.push({...telemetry,...result.stats});return result.value??result;}
   finally{(state.timings??=[]).push({...telemetry,elapsedSeconds:(performance.now()-started)/1000});}};
  const selected=stage==='all'?stages.filter(s=>s!=='enqueue'):[stage];
  try{
   if(selected.includes('inventory')&&!state.inventory){state.inventory=await inventoryFailures(sources);await persist();}
   if(!state.inventory)throw Error('Run inventory first');
   const campaigns=new Map(state.inventory.campaigns.map(c=>[c.id,c]));
   const configFor=async id=>forkCampaign(campaigns.get(id),join(output,'campaigns',id),options);
   const rememberLearning=async()=>{for(const [key,review]of Object.entries(state.reviews)){
    for(const lesson of [...(review.action==='learn'?[review]:[]),...(review.learningRequests??[]),...(state.qa?.[key]?.learningRequests??[])]){
     const groupIds=lesson.groupIds??review.groupIds,id=hash([key,lesson.question,lesson.requiredEvidence,...(lesson.groupIds?[[...groupIds].sort()]:[])]),intervention=groupIds.some(group=>state.classifications[group]?.intervention==='user-required')?'user-required':null;
     if(state.learning[id]){if(intervention){state.learning[id].intervention=intervention;if(!state.learning[id].result)state.learning[id].status='user-required';await persist();}continue;}
     const request={id,key,target:campaigns.get(review.sourceId).target,intervention,learningTarget:lesson.learningTarget,question:lesson.question,hypothesis:lesson.hypothesis,requiredEvidence:lesson.requiredEvidence,groupIds};
     state.learning[id]={...request,status:intervention?'user-required':'pending'};await persist();
    }
   }
   };await rememberLearning();
   if(selected.includes('classify')){
    const pending=state.inventory.groups.filter(group=>!state.classifications[group.id]);
    while(pending.length){
     const first=pending[0],campaign=campaigns.get(first.sourceId),entries=await guidanceEntries(campaign.root,campaign.activeVersion);
     const legacy=await optional(join(output,'classifications',first.id,'native-session.json'));
     if(dependencies.classify&&!dependencies.classifyBatch||legacy){
      const group=pending.shift(),directory=join(output,'classifications',group.id);await mkdir(directory,{recursive:true});
      const result=await invoke('classify',{directory,key:group.id,sourceId:group.sourceId,guidanceVersion:campaign.activeVersion,bucketPolicy:options.bucketPolicy===true,config:await configFor(group.sourceId),group,entries,validate:r=>validateClassification(r,group,entries,{requireBucket:options.bucketPolicy===true})});
      state.classifications[group.id]=validateClassification(result,group,entries,{requireBucket:options.bucketPolicy===true});
     }else{
      const groups=[];for(let i=0;i<pending.length&&groups.length<8;){
       const group=pending[i];if(group.sourceId!==first.sourceId||await optional(join(output,'classifications',group.id,'native-session.json'))){i++;continue;}
       groups.push(...pending.splice(i,1));
      }
      const key=hash(groups.map(g=>g.id)),directory=join(output,'classification-batches',key);await mkdir(directory,{recursive:true});
      const result=await invoke('classifyBatch',{directory,key,sourceId:first.sourceId,guidanceVersion:campaign.activeVersion,bucketPolicy:options.bucketPolicy===true,config:await configFor(first.sourceId),groups,entries,validate:r=>validateClassificationBatch(r,groups,entries,{requireBucket:options.bucketPolicy===true})});
      for(const row of validateClassificationBatch(result,groups,entries,{requireBucket:options.bucketPolicy===true}).classifications)state.classifications[row.groupId]=row;
     }
     await persist();
    }
   }
   const buckets=new Map();
   for(const group of state.inventory.groups){const c=state.classifications[group.id];if(!c)continue;
    for(const id of c.guidanceIds.length?c.guidanceIds:[c.coverage]){const key=group.sourceId+':'+id+(c.bucket?':'+c.bucket:'');
     const bucket=buckets.get(key)??{key,sourceId:group.sourceId,guidanceId:id,...(c.bucket?{objective:c.bucket}:{}),groups:[]};bucket.groups.push(group);buckets.set(key,bucket);}}
   const propose=async()=>{for(const bucket of buckets.values()){
    const completed=state.reviews[bucket.key],learning=Object.entries(state.learning).filter(([,v])=>v.key===bucket.key&&v.result?.status==='verified-cpu');
    if(completed&&(!learning.length||completed.learningRevision===learning.length))continue;
    const config=await configFor(bucket.sourceId),entries=await guidanceEntries(config.root,config.skills.activeVersion),directory=join(output,'proposals',hash([bucket.key,learning.length]));await mkdir(directory,{recursive:true});
    const validate=result=>validateReviewProposal(result,bucket.groups,state.classifications);
    let result;try{result=validate(await invoke('review',{directory,key:bucket.key,objective:bucket.objective,config,groups:bucket.groups,entries,learning,
     classifications:bucket.groups.map(g=>state.classifications[g.id]),validate}));}
    catch(error){
     if(signal.aborted)throw error;
     result={action:'inconclusive',reason:'review-execution-failed',error:{name:error.name,message:error.message},groupIds:bucket.groups.map(g=>g.id)};
     state.errors.push({at:new Date().toISOString(),stage:'propose',bucket:bucket.key,message:error.message});
    }
    state.reviews[bucket.key]={...result,sourceId:bucket.sourceId,guidanceId:bucket.guidanceId,...(bucket.objective?{objective:bucket.objective}:{}),directory,learningRevision:learning.length};
    await rememberLearning();await persist();
   }};if(selected.includes('propose'))await propose();
   const qualify=async key=>{
    const review=state.reviews[key];if(review.action!=='update'||review.objective==='harness')return null;
    const retained=Object.entries(state.candidates).find(([id,c])=>(c.bucket===key||id===key)&&Array.isArray(c.changes)&&hash(c.changes)===hash(review.changes));
    if(retained&&(retained[1].decision||await optional(join(retained[1].directory,'evaluation','manifest.json'))))return {verdict:'grandfathered'};
    const bucket=buckets.get(key),campaign=campaigns.get(review.sourceId),entries=await guidanceEntries(campaign.root,campaign.activeVersion);
    const retrievalEvidenceHash=review.directory?await optionalHash(join(review.directory,'retrievals.jsonl')):null;
    const input={key,proposal:review,proposalDirectory:review.directory,retrievalEvidenceHash,entries,objective:review.objective,target:campaign.target,groups:bucket.groups,
     classifications:bucket.groups.map(g=>state.classifications[g.id]),config:await configFor(review.sourceId),signal,
     validateProposal:value=>validateReviewProposal(value,bucket.groups,state.classifications)};
    const prior=state.qa?.[key];if(prior?.approval===qaIdentity(input))return prior;
    const directory=join(output,'qa',hash([key,qaIdentity(input)]));
    const result=await runProposalQa({...input,directory,seconds:remaining()},{
     qaAnswer:args=>invoke('qaAnswer',args,args.seconds),qaReview:args=>invoke('qaReview',args,args.seconds)});
    (state.qa??={})[key]=result;
    if(result.verdict==='approved')state.reviews[key]={...review,...result.proposal};
    await rememberLearning();await persist();return result;
   };
   const qualifyAll=async()=>{for(const key of Object.keys(state.reviews))await qualify(key);};
   if(selected.includes('qa'))await qualifyAll();
   if(selected.includes('learn'))for(const [id,request]of Object.entries(state.learning)){
    if(request.result&&request.status!=='postprocessing-pending'||request.status==='device-evidence-needed'||request.intervention==='user-required')continue;
    if(request.learningTarget==='device'){
     request.status='device-evidence-needed';await persist();continue;
    }
    if(!request.executionStarted&&Object.values(state.learning).filter(r=>r.executionStarted).length>=(options.maxLearning??2))continue;
    request.executionStarted=true;await persist();
    const directory=join(output,'learning',id);await mkdir(directory,{recursive:true});
    try{request.result=await invoke('learn',{directory,key:request.key,config:await configFor(state.reviews[request.key].sourceId),request},600);}
    catch(error){
     if(signal.aborted)throw error;
     request.result={status:'inconclusive',reason:'learning-execution-failed',error:{name:error.name,message:error.message,...(error.code?{code:error.code}:{})},directory,evidence:[]};
     state.errors.push({at:new Date().toISOString(),stage:'learn',request:id,message:error.message});
    }
    request.status=request.result.status;await persist();
   }
   if(stage==='all'&&Object.values(state.learning).some(r=>r.result?.status==='verified-cpu')){await propose();await qualifyAll();await rememberLearning();}
   if(selected.includes('evaluate'))for(let [key,review]of Object.entries(state.reviews)){
    if(review.action!=='update'||review.objective==='harness')continue;
    const retained=Object.entries(state.candidates).find(([id,c])=>(c.bucket===key||id===key)&&Array.isArray(c.changes)&&hash(c.changes)===hash(review.changes));
    const frozen=retained&&await optional(join(retained[1].directory,'evaluation','manifest.json'));
    if(!frozen){const qa=await qualify(key);if(!['approved','grandfathered'].includes(qa?.verdict))continue;review=state.reviews[key];}
    const revision=hash(review.changes),legacy=state.candidates[key];
    const candidateKey=legacy&&Array.isArray(legacy.changes)&&hash(legacy.changes)===revision?key:key+':'+revision;
    if(state.candidates[candidateKey]?.decision)continue;
    if(!state.candidates[candidateKey]&&Object.keys(state.candidates).length>=(options.maxCandidates??3))continue;
    for(const [oldKey,old]of Object.entries(state.candidates))if(oldKey!==candidateKey&&(old.bucket===key||oldKey===key)&&!old.decision){
     if(!await ops.evaluationSettled({directory:join(old.directory,'evaluation')}))throw Error('Superseded comparison execution remains unresolved');
     old.decision={promote:false,reason:'superseded-proposal'};await persist();
    }
    const campaign=campaigns.get(review.sourceId),directory=join(output,'candidates',hash(candidateKey));await mkdir(directory,{recursive:true});
    let candidate=state.candidates[candidateKey];
    if(!candidate){
     const entries=await guidanceEntries(campaign.root,campaign.activeVersion);let updated;
     try{updated=applyProposal(entries,review.changes,{allowPruning:false,history:[]});}catch(error){state.candidates[candidateKey]={directory,bucket:key,proposalRevision:revision,changes:review.changes,decision:{promote:false,reason:'invalid-proposal',error:error.message}};await persist();continue;}
     await cp(join(campaign.root,'skills','versions',campaign.activeVersion),join(directory,'skills','versions',campaign.activeVersion),{recursive:true});
     const config={...await configFor(review.sourceId),root:directory,skills:{activeVersion:campaign.activeVersion,tuningDisabled:true}};
     new State(join(directory,'state.sqlite')).close();await save(join(directory,'instance.json'),config);
     const version=await writeVersion(directory,updated,join(directory,'skills','versions',campaign.activeVersion));
     candidate=state.candidates[candidateKey]={directory,bucket:key,proposalRevision:revision,active:campaign.activeVersion,version,changes:review.changes};await persist();
    }
    const evaluationConfig=await load(join(directory,'instance.json'));
    if(options.evaluations)evaluationConfig.skill_review_evaluations=options.evaluations;
    let result;try{result=await invoke('evaluate',{directory:join(directory,'evaluation'),key:candidateKey,objective:review.objective,config:evaluationConfig,
     active:join(directory,'skills','versions',candidate.active),candidate:join(directory,'skills','versions',candidate.version)},review.objective?(()=>{const e=evaluationSettings(options.evaluations,review.objective);return e.developmentSeconds+e.pairedSeconds;})():900);}
    catch(error){
     if(signal.aborted)throw error;
     if(!await ops.evaluationSettled({directory:join(directory,'evaluation')}))throw error;
     const partial=await optional(join(directory,'evaluation','results.json'));
     result={decision:{promote:false,reason:error.name==='TimeoutError'?'evaluation-elapsed':'evaluation-failed',error:error.message},report:partial};
     state.errors.push({at:new Date().toISOString(),stage:'evaluate',candidate:candidateKey,message:error.message});
    }
    candidate.decision=result.decision;candidate.results=result.report?.results??[];
    if(candidate.decision.promote){const current=await load(join(directory,'instance.json'));
     candidate.published=current.skills.activeVersion===candidate.version||Boolean(await publishSkillVersion(directory,{active:candidate.active,candidate:candidate.version,signal}));}
    await persist();
   }
   if(selected.includes('enqueue')){
    if(!options.learningInstance)throw Error('Explicit learningInstance required');
    const destination=await realpath(options.learningInstance);if(sources.some(s=>s.root===destination))throw Error('Cannot enqueue into a review source');
    const config=await load(join(destination,'instance.json'));if(config.skills?.review?.enabled!==true)throw Error('Enable skills.review.enabled before importing requests');
    const ledger=new State(join(destination,'state.sqlite'));try{for(const request of Object.values(state.learning))if(request.target===config.target&&request.status!=='verified-cpu'&&request.intervention!=='user-required')enqueueLearningRequest(ledger,request);}finally{ledger.close();}
   }
   state.status=remaining()>0?'checkpointed':'elapsed';
  }catch(error){state.errors.push({at:new Date().toISOString(),stage,message:error.message});state.status=signal.aborted?'elapsed':'failed';throw error;}
  finally{clearTimeout(timer);for(const campaign of state.inventory?.campaigns??[])await native.stopReviewKnowledge(join(output,'campaigns',campaign.id));for(const id of Object.keys(state.learning))await native.stopReviewKnowledge(join(output,'learning',id));await persist();}
  const report=reviewReport(state);
  await save(join(output,'report.json'),report);return report;
 },{key:'skills-review',signal:options.signal});
}
