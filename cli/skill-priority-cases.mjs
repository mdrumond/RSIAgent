/** Select relevant frozen cases; missing evidence becomes curriculum work, never a generic fallback. */
import {loadSkillEvaluation} from './skill-evaluation-contract.mjs';
import {DatabaseSync} from 'node:sqlite';
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile,realpath,readdir} from 'node:fs/promises';
import {join,resolve,relative} from 'node:path';
import {prepareHypothesisTasks} from './microtasks/hypotheses.mjs';
import {reviewSession} from './skill-review-native.mjs';
import {prepareHypothesisEvaluation,hypothesisTarget} from './skill-hypothesis-evaluation.mjs';
import {curatedPacks,applicablePack,prepareCuratedSuites} from './skill-curated-suites.mjs';
import {randomUUID} from 'node:crypto';

const digest=value=>createHash('sha256').update(typeof value==='string'?value:JSON.stringify(value)).digest('hex');
export async function suiteFingerprint(suite,baseDirectory=process.cwd()){
 return digest({objective:suite.objective,target:suite.target,binding:suite.binding,tasks:await Promise.all(suite.tasks.map(async task=>({...task,checker:await readFile(resolve(baseDirectory,task.checker),'utf8'),pairedChecker:task.pairedChecker?await readFile(resolve(baseDirectory,task.pairedChecker),'utf8'):null})))});
}
export async function registerQualifiedSuite({config,suite,qualification,provenance={}}){
 if(qualification?.status!=='qualified'||suite.tasks?.length!==5)throw Error('Five reference/negative-qualified cases required');
 const familyKey=suite.familyKey??digest([suite.target,suite.objective,suite.mechanism]);
 const stored={...suite,status:'ready',familyKey,qualification:{protocol:'reference-negative-v1',fingerprint:await suiteFingerprint(suite,config.root),provenance}};
 const directory=join(config.root,'skills','priority','suites',familyKey);await mkdir(directory,{recursive:true});await writeFile(join(directory,'selected.json'),JSON.stringify(stored,null,2)+'\n');return stored;
}
export async function qualifiedSuites({config}){
 const root=join(config.root,'skills','priority','suites'),suites=[];
 for(const family of await readdir(root).catch(e=>{if(e.code!=='ENOENT')throw e;return [];})){
  try{const suite=JSON.parse(await readFile(join(root,family,'selected.json'),'utf8'));
   if(suite.status==='ready'&&suite.qualification?.fingerprint===await suiteFingerprint(suite,config.root))suites.push({...suite,id:'shared-'+family,description:suite.mechanism,learned:false});
  }catch(e){if(e.code!=='ENOENT')throw e;}
 }return suites;
}

async function learnedCases(config,backlog,item,requests){
 if(!backlog.db.prepare("SELECT 1 FROM sqlite_master WHERE name='skill_learning_requests'").get())return {suites:[],errors:[]};
 const rows=requests?requests.map(request=>({data:JSON.stringify(request)})):backlog.db.prepare("SELECT data FROM skill_learning_requests WHERE status='verified'").all();
 const suites=[],errors=[];
 for(const row of rows){
  const request=JSON.parse(row.data);if(request.backlogId!==item.id&&!request.backlogIds?.includes(item.id))continue;
  let child,project;try{
   if(request.findings?.instanceRoot){const root=await realpath(request.findings.instanceRoot),campaign=await realpath(config.root);if(relative(campaign,root).startsWith('..'))continue;await realpath(join(root,'state.sqlite'));child=new DatabaseSync(join(root,'state.sqlite'),{readOnly:true});}
   const db=child??backlog.db,f=request.findings;
   const attempt=child?db.prepare('SELECT project,outcome FROM attempts WHERE id=?').get(f?.attemptId??null):null;
   if(child&&(!attempt||attempt.outcome!=='passed'||attempt.project!==f.projectId||!db.prepare("SELECT 1 FROM events WHERE attempt=? AND type='postprocess-completed'").get(f.attemptId)))continue;
   project=db.prepare('SELECT spec FROM projects WHERE id=?').get(f?.projectId);
  }catch(error){if(error.code!=='ENOENT')throw error;}finally{child?.close();}if(!project)continue;
  const spec=JSON.parse(project.spec),file=spec.verifier?.files?.find(p=>p.endsWith('/skill-evaluation.json'));
  if(!file){errors.push('Learning project '+request.findings.projectId+' supplied no trusted skill-evaluation.json');continue;}
  try{
  const {contract,tasks}=await loadSkillEvaluation(spec,{required:true});
  suites.push({id:'learning-'+request.findings.projectId,description:request.question,target:contract.target,objective:contract.objective,tasks,
   learned:true,binding:contract.target==='cpu'?{target:'cpu'}:{target:config.target,environment:config.environment.options}});
  }catch(error){errors.push('Learning project '+request.findings.projectId+': '+error.message);}
 }
 return {suites,errors};
}
async function previousCases(config,item){
 const suites=[],root=await realpath(join(config.root,'skills','priority','hypotheses')).catch(()=>null);
 if(!root)return suites;
 const seen=new Set();
 for(const group of item.evidence?.groups??[])for(const occurrence of group.occurrences??[]){
  const path=occurrence.data?.evaluation?.path;if(!path||seen.has(path))continue;
  try{
   const file=await realpath(path);if(relative(root,file).startsWith('..'))continue;
   const suite=JSON.parse(await readFile(file,'utf8'));
   if(suite.status!=='ready'||suite.target!=='cpu'&&JSON.stringify(suite.binding?.environment)!==JSON.stringify(config.environment.options))continue;
   seen.add(path);suites.push({...suite,id:'previous-'+suites.length,description:'Previously qualified cases for '+occurrence.data.hypothesisId,learned:false});
  }catch(error){if(error.code!=='ENOENT')throw error;}
 }
 return suites;
}
const cpuMechanisms={
 'fault-ownership':/(?:traceback|fault|runtime).{0,80}(?:owner|attribut|origin|classif)|(?:owner|attribut|origin|classif|distinguish).{0,80}(?:traceback|fault|runtime)/i,
 'construction-admissibility':/(?:construct|bundle|proposal).{0,80}(?:admiss|layout|verifier|baseline)|(?:admiss|fixture ownership).{0,80}(?:construct|bundle|proposal)/i,
 'dtype-triage':/(?:dtype|float64|double).{0,80}(?:triage|unsupported|reject)|(?:triage|unsupported|reject).{0,80}(?:dtype|float64|double)/i
};
const applicableCpu=(hypothesis,proposal)=>hypothesis.proposalIds.some(id=>(proposal?.changes??[]).some(change=>change.id===id))||cpuMechanisms[hypothesis.id]?.test(JSON.stringify(proposal?.changes??proposal));
export async function hypothesisCaseReadiness({config,item}){
 const packs=config.target==='a3'?(await curatedPacks()).filter(pack=>applicablePack(pack,item)):[];
 const shared=await qualifiedSuites({config});
 const families=packs.map(pack=>'curated-'+pack.id);
 const {hypotheses}=await import('./microtasks/hypotheses.mjs');
 const cpu=item.objective==='correctness'&&hypotheses.some(h=>applicableCpu(h,item.proposal));
 return {ready:cpu||shared.some(s=>families.includes(s.familyKey)),supported:cpu||packs.length>0,families};
}
export async function resolveHypothesisCases({config,backlog,item,directory,qualificationDirectory=join(directory,'qualification'),resource,signal,seconds,event,beforeQualification,proposal=item.proposal},dependencies={}){
 const started=performance.now();
 await mkdir(directory,{recursive:true});
 const proposalIdentity=createHash('sha256').update(JSON.stringify([proposal.changes,item.objective,config.target])).digest('hex');
 const cached=await readFile(join(directory,'selected.json'),'utf8').then(JSON.parse).catch(error=>{if(error.code!=='ENOENT')throw error;return null;});
 const learned=await learnedCases(config,backlog,item);
 const shared=await qualifiedSuites({config});
 let suites=[...(config.skills?.priority?.evaluations??[]),...learned.suites,...shared.filter(s=>s.target==='cpu'||JSON.stringify(s.binding?.environment)===JSON.stringify(config.environment.options)),...await previousCases(config,item)];
 const applicable=config.target==='a3'?(await curatedPacks()).filter(pack=>applicablePack(pack,{...item,proposal})):[];
 let runtimeIdentity;
 if(applicable.length||suites.some(s=>s.target===config.target&&s.target!=='cpu'&&s.objective===item.objective)){
  const probeDirectory=join(directory,'runtime-'+randomUUID());await beforeQualification?.(probeDirectory);
  const binding={target:config.target,environment:config.environment.options};
  const probe=dependencies.runtimeIdentity??(args=>hypothesisTarget({root:probeDirectory,binding,resource,signal,stopSignal:signal,localExecution:config.local_execution,event},dependencies).runtimeIdentity(args));
  runtimeIdentity=await probe({directory:probeDirectory,seconds:Math.min(60,Math.max(.001,seconds-(performance.now()-started)/1000)),signal});
  suites=suites.filter(s=>s.target==='cpu'||!s.curated); // Current curated versions are materialized from source, never stale stored bindings.
  suites.push(...await (dependencies.curatedSuites??prepareCuratedSuites)({config,directory:join(config.root,'skills','priority','curated'),runtimeIdentity,item:{...item,proposal}}));
 }

 if(item.objective==='correctness'){
  const catalogue=await (dependencies.prepareTasks??prepareHypothesisTasks)(join(directory,'cpu'));
  suites.push(...catalogue.manifest.hypotheses.filter(h=>applicableCpu(h,proposal)).map(h=>({id:'cpu-'+h.id,description:h.id+'; cases: '+h.cases.join(', ')+'; CPU report/contract behavior only, not device arithmetic or latency',
   target:'cpu',objective:'correctness',tasks:catalogue.byHypothesis[h.id].map(t=>({...t,target:'cpu',bucket:'correctness'})),binding:{target:'cpu'}})));
 }
 suites=suites.filter(s=>s.objective===item.objective&&(s.target===config.target||s.target==='cpu'&&item.objective==='correctness'));
 const catalogueIdentity=digest({runtimeIdentity,suites:await Promise.all(suites.map(async suite=>({id:suite.id,description:suite.description,fingerprint:await suiteFingerprint(suite,config.root)})))});
 if(cached?.proposalIdentity===proposalIdentity&&cached.catalogueIdentity===catalogueIdentity)return cached;
 const selectionIdentity=digest([proposalIdentity,catalogueIdentity]);
 const validate=value=>{if(!value||!value.reason?.trim()||value.catalogueId!==null&&!suites.some(s=>s.id===value.catalogueId))throw Error('Choose a supplied matching suite or null with a reason');return value;};
 const selection=suites.length?await (dependencies.select??reviewSession)({directory:join(directory,'selection',selectionIdentity),config,signal,seconds:Math.min(60,seconds),
  prompt:{task:'Select one hypothesis-relevant evaluation suite or none. Return {catalogueId:null|string,reason}. The cases must directly exercise the proposed skill mechanism and boundaries. CPU contract/report tasks cannot validate device arithmetic, compiler behavior or kernel performance. Do not select merely because scopes overlap.',
   hypothesis:{objective:item.objective,proposal},catalogues:suites.map(({id,description,target,objective})=>({id,description,target,objective}))},validate}):{value:{catalogueId:null,reason:'No target-compatible evaluation suite'}};
 const chosen=validate(selection.value??selection);let suite=suites.find(s=>s.id===chosen.catalogueId);
 if(suite){
  const frozen=join(directory,'frozen',selectionIdentity);await mkdir(frozen,{recursive:true});
  suite={...suite,tasks:await Promise.all(suite.tasks.map(async(task,index)=>{
   const copied={...task};for(const key of ['checker','pairedChecker'])if(task[key]){
    copied[key]=join(frozen,index+'-'+key+'.py');await writeFile(copied[key],await readFile(resolve(config.root,task[key])));
   }return copied;
  }))};
  const prepared=prepareHypothesisEvaluation(suite);
  if(prepared.status==='ready'&&(suite.learned||suite.curated||suite.target!=='cpu'&&JSON.stringify(suite.binding?.runtimeIdentity)!==JSON.stringify(runtimeIdentity))){
   const qualify=dependencies.qualify??(await import('./skill-hypothesis-evaluation.mjs')).qualifyHypothesisTasks;
   if(suite.target!=='cpu')prepared.binding={...prepared.binding,runtimeIdentity};
   const actualDirectory=suite.curated?join(config.root,'skills','priority','suites',suite.familyKey,suite.packVersion,'qualification'):cached?join(qualificationDirectory,selectionIdentity):qualificationDirectory;
   await beforeQualification?.(actualDirectory);
   const qualification=await qualify({...prepared,directory:actualDirectory,config,signal,resource,
    remainingSeconds:Math.max(.001,seconds-(performance.now()-started)/1000)});
   if(qualification.status==='qualified'){suite.qualification=qualification;suite.qualificationDirectory=actualDirectory;}
   if(qualification.status!=='qualified'){prepared.status='needs-learning';prepared.reason=qualification.reason??'Reference/negative-control qualification did not pass';}
  }
  if(prepared.status==='ready'){prepared.proposalIdentity=proposalIdentity;prepared.catalogueIdentity=catalogueIdentity;
   if(suite.learned||suite.curated){Object.assign(prepared,await registerQualifiedSuite({config,suite:{...prepared,...(suite.curated?{familyKey:suite.familyKey,packVersion:suite.packVersion,curated:true,tags:suite.tags}:{}),mechanism:suite.mechanism??JSON.stringify(proposal.changes)},qualification:suite.qualification,provenance:{proposalIdentity,qualificationDirectory:suite.qualificationDirectory,qualificationReport:join(suite.qualificationDirectory,'micro','results.json')}}));}
   await writeFile(join(directory,'selected.json'),JSON.stringify(prepared,null,2)+'\n');return prepared;}
  chosen.reason=prepared.reason;
 }
 if(learned.errors.length)await event?.('skill-evaluation-qualification-failed',{hypothesis:item.id,errors:learned.errors});
 return {status:'unsupported',reason:chosen.reason,missingCapability:true,errors:learned.errors};
}

/** Suite learning is complete only after its protected export passes reference/negative controls. */
export async function qualifyLearningSuite({config,backlog,request,directory,resource,signal,seconds},dependencies={}){
 const item={id:request.backlogId},learned=await learnedCases(config,backlog,item,[request]);
 if(learned.suites.length!==1)return {status:'inconclusive',reason:learned.errors.join('; ')||'No trusted five-case suite export'};
 const suite=learned.suites[0],prepared=prepareHypothesisEvaluation(suite);
 if(prepared.status!=='ready')return {status:'inconclusive',reason:prepared.reason};
 const qualification=await (dependencies.qualify??(await import('./skill-hypothesis-evaluation.mjs')).qualifyHypothesisTasks)({...prepared,directory,config,resource,signal,remainingSeconds:seconds});
 if(qualification.status!=='qualified')return {status:'inconclusive',reason:qualification.reason??'Suite controls did not qualify'};
 const registered=await registerQualifiedSuite({config,suite:{...prepared,familyKey:request.familyKey,mechanism:request.hypothesis},qualification,provenance:{...request.findings,qualificationDirectory:directory,qualificationReport:join(directory,'micro','results.json')}});
 return {status:'qualified',suite:registered};
}
