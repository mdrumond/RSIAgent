import {projectSummaries} from './project-summary.mjs';
import {hypothesisCaseReadiness} from './skill-priority-cases.mjs';
/** Campaign-local failure analysis. Model work never holds the foreground ledger. */
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile,rename,open,realpath,readdir} from 'node:fs/promises';
import {join,resolve,sep,isAbsolute} from 'node:path';
import {guidanceEntries,validateClassificationBatch,enqueueLearningRequest} from '../controller/skill-review.mjs';
import {applyProposal,writeVersion} from '../controller/skill-evolution.mjs';
import {validateReviewProposal} from './skill-review.mjs';
import {realpathSync,existsSync} from 'node:fs';
import {DatabaseSync} from 'node:sqlite';
import {runProposalQa} from './skill-review-qa.mjs';
import {proposalIdentity,groundingIdentity} from '../controller/skill-backlog.mjs';
const hash=x=>createHash('sha256').update(JSON.stringify(x)).digest('hex');
const load=async path=>{try{return JSON.parse(await readFile(path,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;return null;}};
const save=async(path,value)=>{await mkdir(resolve(path,'..'),{recursive:true});const temporary=path+'.tmp';await writeFile(temporary,JSON.stringify(value,null,2)+'\n');await rename(temporary,path);};
const unwrap=result=>result?.value??result;
// An unfinished marker means the owner died before closing its elapsed ledger.
// Retain that uncertainty rather than renewing the unobserved portion of its grant.
async function trackSpending(path,prior,now,allocation={}){
 const begun=now();let writes=Promise.resolve(),failure;
 const snapshot=()=>({...allocation,elapsedSeconds:prior+(now()-begun)/1000,running:true});
 await save(path,snapshot());
 const timer=setInterval(()=>{const value=snapshot();writes=writes.then(()=>save(path,value)).catch(error=>{failure??=error;});},1000);
 return async()=>{clearInterval(timer);await writes;if(failure)throw failure;await save(path,{...allocation,elapsedSeconds:prior+(now()-begun)/1000,running:false});};
}
async function operations(dependencies){const native=await import('./skill-review-native.mjs');return {classify:native.nativeClassifyBatch,review:native.nativeReview,qaAnswer:native.nativeQaAnswer,qaReview:native.nativeQaReview,session:native.reviewSession,...dependencies};}
function allocation(seconds,signal,now=()=>performance.now()){
 if(!Number.isFinite(seconds)||seconds<=0)throw Error('Positive remaining elapsed allowance required');
 const started=now(),remaining=()=>Math.max(0,seconds-(now()-started)/1000);
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
export async function freezeGroups(root,groups,directory){
 const file=join(directory,'groups.json'),prior=await load(file);if(prior)return prior;
 await mkdir(join(directory,'evidence'),{recursive:true});const campaign=await realpath(root),frozen=structuredClone(groups);
 for(const group of frozen){delete group.classification;delete group.dirty;delete group.fresh;}
 const ledgerPath=join(campaign,'state.sqlite');
 const ledger=existsSync(ledgerPath)?new DatabaseSync(ledgerPath,{readOnly:true}):null;
 try{
  const event=ledger?.prepare("SELECT 1 FROM sqlite_master WHERE name='events'").get()
   ?ledger.prepare("SELECT data FROM events WHERE sequence=? AND type='verification'"):null;
  for(const group of frozen)for(const observation of [...group.occurrences,...(group.linkedDiagnostics??[])]){
   const copies=[];
   const sequence=/^events:(\d+)$/.exec(observation.reference??'');
   const retained=sequence&&event?.get(Number(sequence[1]));
   const primary=retained?JSON.parse(retained.data).correctness:null;
   const verified=typeof primary?.evidence==='string'&&isAbsolute(primary.evidence)
    &&/^[a-f0-9]{64}$/.test(primary.evidence_sha256??'')?primary:null;
   const paths=[...(verified?[verified.evidence]:[]),...(observation.artifacts??[]).map(a=>a.path),...artifactPaths(observation.data)];
   for(const original of [...new Set(paths)].slice(0,2)){
    let path;try{path=await realpath(original);}catch(e){if(e.code==='ENOENT')continue;throw e;}
    if(!path.startsWith(campaign+sep))continue;
    const handle=await open(path,'r');
    let bytes,info;
    try{
     info=await handle.stat();if(!info.isFile())continue;
     const length=Math.min(info.size,32768),buffer=Buffer.alloc(length);
     if(original===verified?.evidence){
      const sourceHash=createHash('sha256');
      for await(const chunk of handle.createReadStream({autoClose:false,start:0}))sourceHash.update(chunk);
      if(sourceHash.digest('hex')!==verified.evidence_sha256)continue;
     }
     if(info.size>length){const half=length/2;await handle.read(buffer,0,half,0);await handle.read(buffer,half,half,info.size-half);}
     else await handle.read(buffer,0,length,0);
     bytes=buffer;
    }finally{await handle.close();}
    const copied=join(directory,'evidence',hash([observation.reference,path])+'.log');await writeFile(copied,bytes);
    copies.push({path:copied,sha256:createHash('sha256').update(bytes).digest('hex'),originalBytes:info.size,...(original===verified?.evidence?{primaryVerification:true}:{}),truncated:info.size>bytes.length,layout:info.size>bytes.length?'head-and-tail':'complete'});
    if(original===verified?.evidence){
     const last=bytes.toString('utf8').trimEnd().split('\n').at(-1);
     try{
      const verdict=JSON.parse(last);
      if(verdict?.correct===false&&/^[a-f0-9]{64}$/.test(verdict.kernelSha256??''))
       observation.verifierVerdict={correct:false,kernelSha256:verdict.kernelSha256};
     }catch(error){if(!(error instanceof SyntaxError))throw error;}
    }

   }
   observation.artifacts=copies;
  }
 }finally{ledger?.close();}
 await save(file,frozen);return frozen;
}
const learningIdentity=results=>hash(results.map(r=>({id:r.id,question:r.data.question,findings:r.data.findings,learningTarget:r.data.learningTarget})));
function learningFor(backlog,item,campaignRoot){
 const db=backlog.db;if(!db.prepare("SELECT 1 FROM sqlite_master WHERE name='skill_learning_requests'").get())return [];
 return db.prepare("SELECT id,data,status FROM skill_learning_requests WHERE status='verified' AND (json_extract(data,'$.backlogId')=? OR EXISTS(SELECT 1 FROM json_each(data,'$.backlogIds') WHERE value=?))").all(item.id,item.id).map(r=>({...r,data:JSON.parse(r.data)}))
  .filter(r=>r.data.findings?.outcome==='passed').flatMap(r=>{
   const f=r.data.findings;let child,source=db;
   try{
    if(f.instanceRoot){
     const campaign=realpathSync(campaignRoot),root=realpathSync(f.instanceRoot);
     if(!root.startsWith(campaign+sep)||!existsSync(join(root,'state.sqlite')))return [];
     child=new DatabaseSync(join(root,'state.sqlite'),{readOnly:true});source=child;
    }
    const attempt=source.prepare('SELECT id,project,outcome FROM attempts WHERE id=?').get(f.attemptId),project=attempt&&source.prepare('SELECT spec FROM projects WHERE id=?').get(attempt.project);
    if(!project||attempt.project!==f.projectId||attempt.outcome!=='passed'||!source.prepare("SELECT 1 FROM events WHERE attempt=? AND type='postprocess-completed'").get(attempt.id))return [];
    return [{...r,project:JSON.parse(project.spec),observations:source.prepare("SELECT sequence,type,data FROM events WHERE attempt=? AND type IN ('verification','accepted','stage-completed') ORDER BY sequence DESC LIMIT 12").all(attempt.id).map(e=>({reference:(f.instanceRoot?'learning:'+r.id+':':'')+'events:'+e.sequence,type:e.type,attemptId:attempt.id,data:JSON.parse(e.data)}))}];
   }catch(error){if(error.code!=='ENOENT')throw error;return [];}finally{child?.close();}
  });
}
async function npuLearningReceipt(observations,kernelSha256){
 if(!kernelSha256)return null;
 for(const observation of observations){
  const v=observation.data;if(observation.type!=='verification'||v.passed!==true||!/^[a-f0-9]{64}$/.test(v.fingerprint??'')||v.correctness?.passed!==true||v.correctness?.exitCode!==0)continue;
  const receipts=[v.correctness.receipt];
  for(const artifact of observation.artifacts??[])try{receipts.push(JSON.parse((await readFile(artifact.path,'utf8')).trim().split('\n').at(-1)));}catch(error){if(!['ENOENT','SyntaxError'].includes(error.code??error.name))throw error;}
  for(const receipt of receipts)if(receipt?.correct===true&&receipt.kernelSha256===kernelSha256&&receipt.evidence&&typeof receipt.evidence==='object'&&!Array.isArray(receipt.evidence)&&/^npu:\d+$/.test(receipt.evidence.device??''))
   return {protocol:'npu-kernel-v1',reference:observation.reference,kernelSha256,device:receipt.evidence.device};
 }
 return null;
}
async function freezeLearning(config,records,directory){
 const result=[],campaign=await realpath(config.root);
 for(const r of [...new Map(records.map(r=>[r.id,r])).values()]){
  const destination=join(directory,r.id),retained=await load(join(destination,'learning.json'));if(retained){if(retained[1].result.status==='verified-device'&&retained[1].result.deviceEvidence?.protocol!=='npu-kernel-v1')retained[1].result.status='verified-project';result.push(retained);continue;}
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
  let binding=null;try{const job=JSON.parse(await readFile(join(r.data.findings.instanceRoot??config.root,'attempts',r.data.findings.attemptId,'job.json'),'utf8'));const c=job.config??job;binding={target:c.target??null,environment:c.environment?.options?.mode??null,runtime:c.environment?.options?.runtime??null};}catch(e){if(e.code!=='ENOENT')throw e;}
  const receipt=await npuLearningReceipt(groups[0].occurrences,copied.find(a=>a.path===join(source,'kernel.py'))?.sha256);
  const snapshot=[r.id,{result:{status:r.data.learningTarget==='device'?(binding?.target===config.target&&binding?.runtime&&receipt?'verified-device':'verified-project'):'verified-cpu',findings:r.data.findings,question:r.data.question,binding,...(receipt?{deviceEvidence:receipt}:{}),
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
function learningPage(backlog,config){
 const scan=backlog.meta('learning-scan')??{ids:backlog.db.prepare("SELECT id FROM skill_backlog_items WHERE status='learning' ORDER BY priority DESC,created,rowid").all().map(r=>r.id),offset:0};
 const ids=scan.ids.slice(scan.offset,scan.offset+20),items=ids.map(id=>backlog.get(id)).filter(item=>item?.status==='learning').map(item=>({item,results:learningFor(backlog,item,config.root)}))
  .filter(({item,results})=>results.length&&learningIdentity(results)!==item.evidence?.learningIdentity);
 backlog.meta('learning-scan',scan);backlog.meta('learning-pending',ids.length>0);
 return {items,finish(){
  const unconsumed=items.some(({item,results})=>learningIdentity(results)!==backlog.get(item.id)?.evidence?.learningIdentity);
  if(!unconsumed)scan.offset+=ids.length;const pending=unconsumed||scan.offset<scan.ids.length;
  backlog.meta('learning-scan',pending?scan:null);backlog.meta('learning-pending',pending);
 }};
}
function consumeLearning(backlog,contributors){
 for(const {item,results}of contributors){const current=backlog.get(item.id);if(current)backlog.upsert({...current,evidence:{...current.evidence,learningIdentity:learningIdentity(results)}});}
}
function priority(group){return Math.min(90,20+Math.min(30,group.frequency?group.frequency.projects+group.frequency.attempts+group.frequency.drafts:group.occurrences.length)+(['execution-unresolved','project-invalid'].includes(group.type)?20:0));}

/** Superseded coordinates remain historical evidence, never current blockers. */
export function currentAnalysisStages({config,backlog}){
 const epoch=backlog.meta('queue-epoch');
 return Object.entries(backlog.meta('analysis-stages')??{}).filter(([,s])=>s.epoch===epoch&&s.base===config.skills.activeVersion&&s.groupId&&backlog.failureCount(s.groupId)===s.count).map(([id,stage])=>({id,...stage}));
}

/** Analyze a bounded page. New evidence remains dirty after failures or elapsed expiry. */
export async function analyzeBacklog({config,backlog,directory,signal,seconds=120,remainingSeconds=seconds+3600,limit=2,classificationOnly=false},dependencies={}){
 const ops=await operations(dependencies),now=dependencies.now??(()=>performance.now()),clock=allocation(remainingSeconds,signal,now),entries=await guidanceEntries(config.root,config.skills.activeVersion);
 const policy={requireBucket:true,failureFirst:classificationOnly};
 const epoch=backlog.meta('queue-epoch'),stages=Object.fromEntries(Object.entries(backlog.meta('analysis-stages')??{}).filter(([,s])=>s.epoch===epoch)),coordinates=new Map(),identity=g=>{const id=hash([epoch,config.skills.activeVersion,g.id,g.count,...(classificationOnly?['evidence-routing']:[])]);coordinates.set(id,{epoch,base:config.skills.activeVersion,groupId:g.id,count:g.count});return id;};
 const persist=()=>{for(const [id,stage]of Object.entries(stages))Object.assign(stage,coordinates.get(id));backlog.meta('analysis-stages',stages);};
 const eligible=g=>stages[identity(g)]?.status!=='blocked';
 const dirty=[];
 for(let offset=0;dirty.length<limit;offset+=200){const rows=backlog.failures({limit:200,offset,occurrenceLimit:6});dirty.push(...rows.filter(eligible).slice(0,limit-dirty.length));if(rows.length<200)break;}
 const page=classificationOnly?{items:[],finish(){}}:learningPage(backlog,config),learningItems=page.items;
 const groups=[...new Map([...dirty,...learningItems.flatMap(({item})=>item.groupIds.map(id=>backlog.failure(id,{limit:6})))].filter(g=>g&&eligible(g)).map(g=>[g.id,g])).values()].slice(0,limit);
 if(!groups.length){page.finish();return {analyzed:0,items:[],remainingSeconds:clock.remaining()};}
 const batch=join(directory,hash([config.skills.activeVersion,groups,learningItems.map(({item,results})=>[item.id,results]),...(classificationOnly?['evidence-routing']:[])])),frozen=await freezeGroups(config.root,groups,batch);
 const classificationFile=join(batch,'classifications.json');let classifications=await load(classificationFile);
 if(!classifications&&frozen.every(g=>stages[identity(g)]?.classification))classifications={classifications:frozen.map(g=>stages[identity(g)].classification)};
 const stageError=(error,spent,cap)=>spent>=cap-.05||['rejected-submissions','no-submit','interrupted-unknown-spend'].includes(error.reviewSession?.kind);
 if(!classifications){
  // A retry retains its original grant membership. Fresh groups receive their own
  // page rather than inheriting an unrelated group's spent time or unknown owner.
  const pending=frozen.filter(g=>!stages[identity(g)]?.classification),partitions=new Map();
  for(const group of pending){const key=stages[identity(group)]?.spendFile??'fresh';partitions.set(key,[...(partitions.get(key)??[]),group]);}
  for(const [retainedFile,members]of partitions){
   const spendFile=retainedFile==='fresh'?join(batch,'classification',hash(members.map(g=>identity(g))),'classification-spend.json'):retainedFile,
    prior=await load(spendFile),grantSeconds=prior?.allocatedSeconds??members.map(g=>stages[identity(g)]?.allocatedSeconds).find(Number.isFinite)??(prior?120:seconds),
    spent=Math.max(prior?.elapsedSeconds??0,...members.map(g=>stages[identity(g)]?.elapsedSeconds??0)),begun=now();let finishClassification;
   try{
    if(prior?.running)throw Object.assign(Error('Interrupted classification elapsed custody is unknown; explicit review required'),{reviewSession:{kind:'interrupted-unknown-spend'}});
    for(const group of members)stages[identity(group)]={stage:'classification',status:'running',spendFile,allocatedSeconds:grantSeconds,elapsedSeconds:spent};persist();
    finishClassification=await trackSpending(spendFile,spent,now,{allocatedSeconds:grantSeconds});
    if(spent>=grantSeconds)throw Error('Classification elapsed allowance exhausted');
    const accepted=validateClassificationBatch(await clock.call(ops.classify,{directory:join(resolve(spendFile,'..'),'classify'),config,groups:members,entries,bucketPolicy:true,failureFirst:classificationOnly,
     validate:value=>validateClassificationBatch(value,members,entries,policy)},grantSeconds-spent),members,entries,policy);
    for(const group of members)stages[identity(group)]={stage:'proposal',status:'pending',classification:accepted.classifications.find(c=>c.groupId===group.id)};
    persist();
   }catch(error){for(const group of members)stages[identity(group)]={stage:'classification',status:stageError(error,spent+(now()-begun)/1000,grantSeconds)?'blocked':'retry',spendFile,allocatedSeconds:grantSeconds,reason:error.message,elapsedSeconds:spent+(now()-begun)/1000};persist();throw error;}
   finally{await finishClassification?.();}
  }
  classifications={classifications:frozen.map(g=>stages[identity(g)].classification)};await save(classificationFile,classifications);
 }

 const valid=validateClassificationBatch(classifications,frozen,entries,policy).classifications,byGroup=Object.fromEntries(valid.map(c=>[c.groupId,c])),created=[];
 for(const group of frozen){
  signal?.throwIfAborted();if(clock.remaining()<=0)break;
  const learned=learningItems.filter(({item})=>item.groupIds.includes(group.id));
  const classification=byGroup[group.id],base=config.skills.activeVersion,common={groupIds:[group.id],objective:classification.bucket,skill:classification.guidanceIds,
   priority:priority(group),reason:classification.reason,base},evidence={groups:[group]},classificationContext={classifications:{[group.id]:classification}};
  if(classificationOnly){
   const {route,observed}=classification.triage;
   const status={human:'user-required',baseline:'queued',phase1:'learning',probe:observed?.replayEvidence?'queued':'deferred'}[route]??'deferred';
   const measurements=route==='optimization'?(await projectSummaries(config,[group])).flatMap(summary=>
    summary.measurements.filter(m=>m.verified&&m.geometricMeanSpeedup>1).map(m=>({reference:m.reference,projectId:summary.projectId,geometricMeanSpeedup:m.geometricMeanSpeedup}))):[];
   const bonus=Math.min(10,Math.round(10*Math.log2(Math.max(1,...measurements.map(m=>m.geometricMeanSpeedup)))));
   const stored=backlog.atomic(()=>{
    const item=backlog.upsert({...common,objective:{human:'harness',phase1:'correctness'}[route]??classification.bucket,
     key:'failure-first:'+group.id,workflow:'failure-first-v1',proposal:null,evidence,preparation:{...classificationContext,...(measurements.length?{measurements}:{})},
     priority:common.priority+bonus,status});
    if(route==='phase1')queueLearning(backlog,config,item,[{workKind:'phase1',learningTarget:'device',
     question:'Can an explicit PyTorch reference reproduce the intended operation independently of the failed target baseline?',
     hypothesis:'Small cases with a separately specified oracle can establish the expected result on the target.',
     requiredEvidence:'Provide the reference and small cases with explicit expected outputs, then check the reference on the target. Original baseline verdict: '+observed.reference}]);
    backlog.acknowledge(group.id,{classification,count:group.count});return item;
   });
   created.push(stored);stages[identity(group)]={stage:'complete',status:'complete',classification};persist();continue;
  }
  if(classification.intervention==='user-required'){
   created.push(backlog.upsert({...common,key:'harness:'+group.id,proposal:{action:'no-change',reason:classification.reason},evidence,preparation:classificationContext,status:'user-required'}));
   consumeLearning(backlog,learned);backlog.acknowledge(group.id,{classification,count:group.count});stages[identity(group)]={stage:'complete',status:'complete',classification};persist();continue;
  }
  const learning=await freezeLearning(config,learned.flatMap(({results})=>results),join(batch,'learning'));
  const proposalDirectory=join(directory,'proposals',hash([identity(group),learned.map(({item,results})=>[item.id,learningIdentity(results)])]));await mkdir(proposalDirectory,{recursive:true});
  const spendFile=join(proposalDirectory,'writing-spend.json'),priorLedger=await load(spendFile),priorSpent=priorLedger?.elapsedSeconds??0,begun=now();
  if(priorLedger?.running){stages[identity(group)]={stage:'proposal',status:'blocked',classification,elapsedSeconds:priorSpent,reason:'Interrupted writing elapsed custody is unknown; explicit review required'};persist();continue;}
  if(priorSpent>=1800){created.push(backlog.upsert({...common,key:'analysis:'+group.id,proposal:{action:'no-change',reason:'Proposal elapsed allowance exhausted'},evidence,preparation:{...classificationContext,proposalElapsedSeconds:priorSpent},status:'inconclusive'}));stages[identity(group)]={stage:'proposal',status:'blocked',classification,elapsedSeconds:priorSpent,reason:'Proposal elapsed allowance exhausted'};persist();continue;}
  stages[identity(group)]={stage:'proposal',status:'running',classification,elapsedSeconds:priorSpent};persist();
  const finishWriting=await trackSpending(spendFile,priorSpent,now);
  let review;try{review=validateReviewProposal(await clock.call(ops.review,{directory:proposalDirectory,config,groups:[group],entries,classifications:[classification],learning,
   validate:value=>validateReviewProposal(value,[group],byGroup)},Math.max(.001,1800-priorSpent)),[group],byGroup);}
  catch(error){stages[identity(group)]={stage:'proposal',status:stageError(error,priorSpent+(now()-begun)/1000,1800)?'blocked':'retry',classification,reason:error.message,elapsedSeconds:priorSpent+(now()-begun)/1000};persist();throw error;}
  finally{await finishWriting();}
  const records=await readFile(join(proposalDirectory,'retrievals.jsonl'),'utf8').catch(e=>{if(e.code==='ENOENT')return '';throw e;});
  const memoryEvidence=records.split('\n').filter(Boolean).flatMap(line=>JSON.parse(line).entries??[]);
  const actualRefs=new Set(memoryEvidence.map(e=>e.ref??`${e.collection}:${e.entry_id??e.chunk_id??e.id}`));
  if(review.action==='update'&&review.memoryRefs.some(ref=>!actualRefs.has(ref)))throw Error('Update requires memory retrieved in this proposal session');
  const proposalEvidence={...evidence,memoryEvidence,...(learned.length?{learning,learningIdentity:learningIdentity(learned.flatMap(l=>l.results))}:{})};
  if(group.type==='skill-hypothesis-rejected'&&review.action==='update'){
   const effective=change=>hash(proposalIdentity({action:'update',changes:[change]})),grounding=groundingIdentity({proposal:review,evidence:proposalEvidence});
   const changes=review.changes.filter(change=>group.occurrences.filter(o=>o.data.previousProposal?.changes?.some(prior=>effective(prior)===effective(change)))
    .every(o=>o.data.previousEvidence&&grounding!==groundingIdentity({proposal:o.data.previousProposal,evidence:o.data.previousEvidence})));
   review=changes.length?{...review,changes}:{...review,action:'no-change',changes:[],reason:'Rejected instruction and factual grounding are unchanged; new knowledge or a substantive revision is required'};
  }
  const preparation={...classificationContext,proposalDirectory,proposalElapsedSeconds:(await load(spendFile)).elapsedSeconds};
  if(group.type==='skill-hypothesis-rejected'&&review.action==='no-change'&&!review.learningRequests?.length)
   created.push(backlog.upsert({...common,key:'feedback:'+group.id,proposal:review,evidence:proposalEvidence,preparation,status:'inconclusive',reason:review.reason}));
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
  consumeLearning(backlog,learned);backlog.acknowledge(group.id,{classification,count:group.count});
  stages[identity(group)]={stage:'complete',status:'complete',classification};persist();
 }
 page.finish();return {analyzed:frozen.filter(g=>!backlog.failure(g.id)?.dirty).length,items:created,elapsedSeconds:clock.elapsed(),remainingSeconds:clock.remaining()};
}

/** Rank open work only. Running comparisons retain their frozen inputs and finish first. */
export async function prioritizeBacklog({config,backlog,directory,signal,seconds=120},dependencies={}){
 const open=['queued','learning','needs-review'].flatMap(status=>backlog.list({status,limit:1000}));if(!open.length)return [];
 const seen=backlog.meta('priority-inputs')??{},signature=item=>hash([item.fingerprint,item.evidence?.groups?.map(g=>[g.id,g.count])]);
 const changed=open.filter(i=>seen[i.id]!==signature(i)).sort((a,b)=>b.priority-a.priority||b.created-a.created);
 const cursor=backlog.meta('priority-cursor')??0,ordered=open.slice().sort((a,b)=>b.priority-a.priority||a.created-b.created);
 const refresh=changed.length?changed.slice(0,20):ordered.slice(cursor,cursor+20);
 const items=[...new Map([...refresh,...ordered.slice(0,4)].map(i=>[i.id,i])).values()];
 const ops=await operations(dependencies),clock=allocation(seconds,signal),input=await Promise.all(items.map(async i=>({caseReadiness:await (dependencies.caseReadiness??hypothesisCaseReadiness)({config,backlog,item:i}),id:i.id,objective:i.objective,skill:i.skill,priority:i.priority,reason:i.reason,
  proposal:i.proposal,evidence:i.evidence?.groups?.map(g=>({id:g.id,type:g.type,message:g.message,frequency:g.frequency,observations:g.count}))})));
 const validate=value=>{
  const rows=Array.isArray(value)?value:value?.priorities;if(!Array.isArray(rows)||rows.length!==items.length||new Set(rows.map(r=>r.id)).size!==items.length||rows.some(r=>!items.some(i=>i.id===r.id)||!Number.isFinite(r.priority)||r.priority<0||r.priority>100||!r.reason?.trim()))throw Error('Rank every open hypothesis once with priority 0–100 and evidence-supported reason');return {priorities:rows};
 };
 const rows=await clock.call(ops.session,{directory:join(directory,hash(input)),config,knowledge:false,
  prompt:{task:'Rank every open hypothesis by expected impact: severity, independent affected projects/attempts, evidence strength, and actual case readiness; observation counts are polling telemetry, not independent failures. New urgent evidence may rank above all existing work. Do not use cumulative actor clocks as measured operation duration. Return {priorities:[{id,priority:0..100,reason}]}. Keep each reason to at most two short sentences and cite supplied evidence references when available; do not write an essay. Harness problems remain user handled; do not invent evidence.',items:input},validate});
 const ranked=validate(Array.isArray(rows)?{priorities:rows}:rows),result=backlog.rank(ranked.priorities);
 for(const item of items)seen[item.id]=signature(item);backlog.meta('priority-inputs',seen);backlog.meta('priority-cursor',cursor+20>=ordered.length?0:cursor+20);return result;
}

/** Prepare one candidate against the current baseline; approval admits A/B, never publication. */
export async function prepareHypothesis({config,backlog,item,directory,signal,remainingSeconds,evaluationContract,resumeFrom,priorElapsedSeconds},dependencies={}){
 if(item.base!==config.skills.activeVersion)throw Error('Rebase and requalify hypothesis against the current active skill');
 const ops=await operations(dependencies),clock=allocation(remainingSeconds,signal),identity=hash([item.id,item.revision,item.base,item.proposal,item.evidence,evaluationContract??null]);
 await mkdir(directory,{recursive:true});const file=join(directory,'preparation.json'),prior=await load(file);
 if(prior){if(prior.identity!==identity)throw Error('Hypothesis preparation inputs changed');return {...prior,recovered:true};}
 if(item.objective==='harness')return {status:'user-required',reason:'Harness faults require user intervention'};
 const groups=item.evidence?.groups??[],classifications=item.preparation?.classifications??{},entries=await guidanceEntries(config.root,item.base);
 const validateProposal=value=>validateReviewProposal(value,groups,classifications),proposal=validateProposal(structuredClone(item.proposal));
 if(proposal.action!=='update')return {status:'learning',reason:'Supported skill instruction required before comparison'};
 const cited=new Set((item.evidence.memoryEvidence??[]).map(e=>e.ref??`${e.collection}:${e.entry_id??e.chunk_id??e.id}`));
 if(proposal.memoryRefs.some(ref=>!cited.has(ref)))throw Error('Hypothesis memory evidence is missing');
 const qa=await runProposalQa({directory:join(directory,'qa'),config,proposal,groups,entries,objective:item.objective,target:config.target,
  memoryEvidence:item.evidence.memoryEvidence??[],proposalDirectory:item.preparation?.proposalDirectory,learning:item.evidence.learning??[],validateProposal,
  signal,evaluationContract,resumeFrom,priorElapsedSeconds,seconds:Math.min(300,clock.remaining())},{qaAnswer:args=>clock.call(ops.qaAnswer,args,args.seconds),qaReview:args=>clock.call(ops.qaReview,args,args.seconds)});
 if(qa.verdict!=='approved'){
  queueLearning(backlog,config,item,qa.learningRequests??[]);
  return {status:qa.verdict==='learn'?'learning':'inconclusive',qa,retryable:qa.retryable===true,reason:qa.reason,elapsedSeconds:clock.elapsed()};
 }
 if(clock.remaining()<=0)throw Error('Hypothesis preparation elapsed allocation exhausted');
 const final=validateProposal(qa.proposal),updated=applyProposal(entries,final.changes,{allowPruning:false,history:[]});
 const candidate=await writeVersion(config.root,updated,join(config.root,'skills','versions',item.base));
 const result={identity,status:'ready',active:item.base,candidate,proposal:final,qa,elapsedSeconds:clock.elapsed()};await save(file,result);return result;
}
