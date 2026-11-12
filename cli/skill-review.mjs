/** Isolated, resumable failure -> memory -> guidance/learning -> A/B review. */
import {mkdir,readFile,writeFile,rename,cp,realpath,access,rm} from 'node:fs/promises';
import {join,resolve,relative,sep} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {DatabaseSync} from 'node:sqlite';
import {State} from '../controller/state.mjs';
import {inventoryFailures,guidanceEntries,validateClassification,validateReview,enqueueLearningRequest} from '../controller/skill-review.mjs';
import {applyProposal,writeVersion} from '../controller/skill-evolution.mjs';
import {validateSkillGrounding} from '../controller/skill-evidence.mjs';
import {publishSkillVersion} from './skills.mjs';
import {withSkillLock} from './evolution-lock.mjs';
const load=async path=>JSON.parse(await readFile(path,'utf8'));
async function optional(path){try{return await load(path);}catch(e){if(e.code!=='ENOENT')throw e;return null;}}
async function save(path,value){const tmp=path+'.'+randomUUID();await writeFile(tmp,JSON.stringify(value,null,2)+'\n');await rename(tmp,path);}
async function fileHash(path){const d=createHash('sha256');for await(const chunk of createReadStream(path))d.update(chunk);return d.digest('hex');}
const hash=x=>createHash('sha256').update(JSON.stringify(x)).digest('hex');
const stages=['inventory','classify','propose','learn','evaluate','report','enqueue'];
async function forkCampaign(source,directory,options){
 let config=await optional(join(directory,'instance.json'));if(config)return {...config,root:directory};
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
  skills:{activeVersion:source.activeVersion,tuningDisabled:true},environment:{mode:'local'},projects:[],recovery:false,concurrency:1};
 new State(join(directory,'state.sqlite')).close();await save(join(directory,'instance.json'),config);return config;
}
export async function runReview(options,dependencies={}){
 if(!Array.isArray(options.sources)||!options.sources.length||!options.output)throw Error('Review requires snapshot sources and separate output');
 const output=resolve(options.output),sources=await Promise.all(options.sources.map(async s=>({...(typeof s==='string'?{}:s),root:await realpath(typeof s==='string'?s:s.root)})));
 for(const source of sources){const rel=relative(source.root,output);if(!rel||rel!=='..'&&!rel.startsWith('..'+sep))throw Error('Review output must not belong to source campaign');}
 const seconds=options.seconds??7200;if(!Number.isFinite(seconds)||seconds<=0)throw Error('Positive review elapsed allowance required');
 if(![options.maxCandidates??3,options.maxLearning??2].every(n=>Number.isSafeInteger(n)&&n>=0))throw Error('Nonnegative candidate and learning project targets required');
 const stage=options.stage??'all';if(stage!=='all'&&!stages.includes(stage))throw Error('Unknown review stage');
 await mkdir(output,{recursive:true});
 const frozen=[];for(const source of sources){const cfg=await load(join(source.root,'instance.json'));frozen.push({root:source.root,config:hash(cfg),state:await fileHash(join(source.root,'state.sqlite')),knowledge:await fileHash(join(source.root,'knowledge.sqlite')),guidance:hash(await guidanceEntries(source.root,cfg.skills.activeVersion))});}
 const lockConfig={root:output,skill_queue_seconds:1};new State(join(output,'state.sqlite')).close();
 return withSkillLock(lockConfig,async()=>{
  const file=join(output,'review.json'),signature=hash({sources,frozen,profile:options.profile??null,python:options.python??null,seconds,maxCandidates:options.maxCandidates??3,maxLearning:options.maxLearning??2});
  let state=await optional(file);
  if(state&&state.signature!==signature)throw Error('Frozen review inputs or elapsed allocation changed');
  state??={format:1,signature,startedAt:Date.now(),seconds,inventory:null,classifications:{},reviews:{},learning:{},candidates:{},errors:[],stats:[]};
  const remaining=()=>Math.max(0,seconds-(Date.now()-state.startedAt)/1000);
  const persist=async()=>{state.elapsedSeconds=(Date.now()-state.startedAt)/1000;await save(file,state);};
  const native=await import('./skill-review-native.mjs'),ops={classify:native.nativeClassify,review:native.nativeReview,learn:native.nativeLearn,evaluate:native.nativeEvaluate,evaluationSettled:native.evaluationSettled,...dependencies};
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(Error('Review elapsed allocation exhausted')),Math.min(2147483647,remaining()*1000));
  const signal=options.signal?AbortSignal.any([options.signal,controller.signal]):controller.signal;
  const invoke=async(name,input,cap=240)=>{signal.throwIfAborted();if(remaining()<=0)throw Error('Review elapsed allocation exhausted');
   const allocation=Math.min(cap,remaining()),bounded=AbortSignal.any([signal,AbortSignal.timeout(Math.max(1,Math.ceil(allocation*1000)))]);
   const started=performance.now();try{const result=await ops[name]({...input,signal:bounded,seconds:allocation});if(result.stats)state.stats.push({stage:name,key:input.key,...result.stats});return result.value??result;}
   finally{(state.timings??=[]).push({stage:name,key:input.key,elapsedSeconds:(performance.now()-started)/1000});}};
  const selected=stage==='all'?stages.filter(s=>s!=='enqueue'):[stage];
  try{
   if(selected.includes('inventory')&&!state.inventory){state.inventory=await inventoryFailures(sources);await persist();}
   if(!state.inventory)throw Error('Run inventory first');
   const campaigns=new Map(state.inventory.campaigns.map(c=>[c.id,c]));
   const configFor=async id=>forkCampaign(campaigns.get(id),join(output,'campaigns',id),options);
   if(selected.includes('classify'))for(const group of state.inventory.groups){
    if(state.classifications[group.id])continue;
    const campaign=campaigns.get(group.sourceId),entries=await guidanceEntries(campaign.root,campaign.activeVersion);
    const directory=join(output,'classifications',group.id);await mkdir(directory,{recursive:true});
    state.classifications[group.id]=await invoke('classify',{directory,key:group.id,config:await configFor(group.sourceId),group,entries,validate:r=>validateClassification(r,group,entries)});
    state.classifications[group.id]=validateClassification(state.classifications[group.id],group,entries);await persist();
   }
   const buckets=new Map();
   for(const group of state.inventory.groups){const c=state.classifications[group.id];if(!c)continue;
    for(const id of c.guidanceIds.length?c.guidanceIds:[c.coverage]){const key=group.sourceId+':'+id;
     const bucket=buckets.get(key)??{key,sourceId:group.sourceId,guidanceId:id,groups:[]};bucket.groups.push(group);buckets.set(key,bucket);}}
   const propose=async()=>{for(const bucket of buckets.values()){
    const completed=state.reviews[bucket.key],learning=Object.entries(state.learning).filter(([,v])=>v.key===bucket.key&&v.result?.status==='verified-cpu');
    if(completed&&(!learning.length||completed.learningRevision===learning.length))continue;
    const config=await configFor(bucket.sourceId),entries=await guidanceEntries(config.root,config.skills.activeVersion),directory=join(output,'proposals',hash([bucket.key,learning.length]));await mkdir(directory,{recursive:true});
    const validate=result=>{
     validateReview(result,bucket.groups);
     if(result.action==='update')result.changes=validateSkillGrounding(result.changes,bucket.groups.flatMap(g=>g.occurrences.map(o=>({reference:o.reference,...(state.classifications[g.id].cause==='harness'?{responsibility:'harness'}:{})}))));
     return result;
    };
    const result=validate(await invoke('review',{directory,key:bucket.key,config,groups:bucket.groups,entries,learning,
     classifications:bucket.groups.map(g=>state.classifications[g.id]),validate}));
    state.reviews[bucket.key]={...result,sourceId:bucket.sourceId,guidanceId:bucket.guidanceId,directory,learningRevision:learning.length};await persist();
   }};if(selected.includes('propose'))await propose();
   const rememberLearning=async()=>{for(const [key,review]of Object.entries(state.reviews)){
    for(const lesson of [...(review.action==='learn'?[review]:[]),...(review.learningRequests??[])]){
     const id=hash([key,lesson.question,lesson.requiredEvidence]),intervention=review.groupIds.some(group=>state.classifications[group]?.cause==='harness')?'user-required':null;
     if(state.learning[id]){if(intervention){state.learning[id].intervention=intervention;if(!state.learning[id].result)state.learning[id].status='user-required';await persist();}continue;}
     const request={id,key,target:campaigns.get(review.sourceId).target,intervention,learningTarget:lesson.learningTarget,question:lesson.question,hypothesis:lesson.hypothesis,requiredEvidence:lesson.requiredEvidence,groupIds:review.groupIds};
     state.learning[id]={...request,status:intervention?'user-required':'pending'};await persist();
    }
   }
   };if(selected.includes('learn'))await rememberLearning();
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
   if(stage==='all'&&Object.values(state.learning).some(r=>r.result?.status==='verified-cpu')){await propose();await rememberLearning();}
   if(selected.includes('evaluate'))for(const [key,review]of Object.entries(state.reviews)){
    if(review.action!=='update')continue;
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
    let result;try{result=await invoke('evaluate',{directory:join(directory,'evaluation'),key:candidateKey,config:await load(join(directory,'instance.json')),
     active:join(directory,'skills','versions',candidate.active),candidate:join(directory,'skills','versions',candidate.version)},900);}
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
  const report={status:state.status,elapsedSeconds:state.elapsedSeconds,remainingSeconds:remaining(),failureGroups:state.inventory.groups.length,
   failureObservations:state.inventory.groups.reduce((n,g)=>n+g.occurrences.length,0),classified:Object.keys(state.classifications).length,
   dispositions:Object.values(state.reviews).reduce((out,r)=>(out[r.action]=(out[r.action]??0)+1,out),{}),
   userInterventions:Object.values(state.classifications).filter(c=>c.intervention==='user-required').map(c=>c.groupId),
   learning:Object.values(state.learning).map(({id,status,question,result,intervention})=>({id,status,question,...(intervention?{intervention}:{}),...(result?.reason?{reason:result.reason}:{}),...(result?.error?{error:result.error}:{})})),
   candidates:Object.entries(state.candidates).map(([key,c])=>({key,version:c.version,decision:c.decision,published:c.published??false})),
   stats:state.stats,timings:state.timings??[],claim:'Isolated review; live campaigns unchanged. CPU microbenchmarks do not establish GPU kernel improvement.'};
  await save(join(output,'report.json'),report);return report;
 },{key:'skills-review',signal:options.signal});
}
