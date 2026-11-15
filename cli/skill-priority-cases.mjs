/** Select relevant frozen cases; missing evidence becomes curriculum work, never a generic fallback. */
import {DatabaseSync} from 'node:sqlite';
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile,realpath,readdir} from 'node:fs/promises';
import {join,resolve,dirname,relative} from 'node:path';
import {prepareHypothesisTasks} from './microtasks/hypotheses.mjs';
import {reviewSession} from './skill-review-native.mjs';
import {enqueueLearningRequest} from '../controller/skill-review.mjs';
import {prepareHypothesisEvaluation} from './skill-hypothesis-evaluation.mjs';

const digest=value=>createHash('sha256').update(typeof value==='string'?value:JSON.stringify(value)).digest('hex');
export async function suiteFingerprint(suite){
 return digest({objective:suite.objective,target:suite.target,binding:suite.binding,tasks:await Promise.all(suite.tasks.map(async task=>({...task,checker:await readFile(task.checker,'utf8'),pairedChecker:task.pairedChecker?await readFile(task.pairedChecker,'utf8'):null})))});
}
export async function registerQualifiedSuite({config,suite,qualification,provenance={}}){
 if(qualification?.status!=='qualified'||suite.tasks?.length!==5)throw Error('Five reference/negative-qualified cases required');
 const familyKey=suite.familyKey??digest([suite.target,suite.objective,suite.mechanism]);
 const stored={...suite,status:'ready',familyKey,qualification:{protocol:'reference-negative-v1',fingerprint:await suiteFingerprint(suite),provenance}};
 const directory=join(config.root,'skills','priority','suites',familyKey);await mkdir(directory,{recursive:true});await writeFile(join(directory,'selected.json'),JSON.stringify(stored,null,2)+'\n');return stored;
}
export async function qualifiedSuites({config}){
 const root=join(config.root,'skills','priority','suites'),suites=[];
 for(const family of await readdir(root).catch(e=>{if(e.code!=='ENOENT')throw e;return [];})){
  try{const suite=JSON.parse(await readFile(join(root,family,'selected.json'),'utf8'));
   if(suite.status==='ready'&&suite.qualification?.fingerprint===await suiteFingerprint(suite))suites.push({...suite,id:'shared-'+family,description:suite.mechanism,learned:false});
  }catch(e){if(e.code!=='ENOENT')throw e;}
 }return suites;
}

async function learnedCases(config,backlog,item){
 if(!backlog.db.prepare("SELECT 1 FROM sqlite_master WHERE name='skill_learning_requests'").get())return {suites:[],errors:[]};
 const rows=backlog.db.prepare("SELECT data FROM skill_learning_requests WHERE status='verified'").all();
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
  const root=await realpath(dirname(file)),contract=JSON.parse(await readFile(file,'utf8'));
  const inside=async name=>{const path=await realpath(resolve(root,name));if(relative(root,path).startsWith('..'))throw Error('Learning evaluation files must stay within the trusted verifier');return path;};
  const files=async values=>Object.fromEntries(await Promise.all(Object.entries(values??{}).map(async([name,path])=>[name,await readFile(await inside(path),'utf8')])));
  const tasks=await Promise.all((contract.tasks??[]).map(async task=>({...task,
   checker:await inside(task.checker),...(task.pairedChecker?{pairedChecker:await inside(task.pairedChecker)}:{}),
   files:await files(task.files),controls:{reference:{files:await files(task.controls?.reference?.files)},
    negatives:await Promise.all((task.controls?.negatives??[]).map(async control=>({files:await files(control.files)})))}})));
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
export async function resolveHypothesisCases({config,backlog,item,directory,qualificationDirectory=join(directory,'qualification'),resource,signal,seconds,event,proposal=item.proposal},dependencies={}){
 const started=performance.now();
 await mkdir(directory,{recursive:true});
 const proposalIdentity=createHash('sha256').update(JSON.stringify([proposal.changes,item.objective,config.target])).digest('hex');
 const cached=await readFile(join(directory,'selected.json'),'utf8').then(JSON.parse).catch(error=>{if(error.code!=='ENOENT')throw error;return null;});
 const learned=await learnedCases(config,backlog,item);
 const shared=await qualifiedSuites({config});
 let suites=[...(config.skills?.priority?.evaluations??[]),...learned.suites,...shared.filter(s=>s.target==='cpu'||JSON.stringify(s.binding?.environment)===JSON.stringify(config.environment.options)),...await previousCases(config,item)];
 if(item.objective==='correctness'){
  const catalogue=await (dependencies.prepareTasks??prepareHypothesisTasks)(join(directory,'cpu'));
  const ids=new Set((proposal.changes??[]).map(change=>change.id));
  suites.push(...catalogue.manifest.hypotheses.filter(h=>h.proposalIds?.some(id=>ids.has(id))).map(h=>({id:'cpu-'+h.id,description:h.id+'; CPU report/contract behavior only, not device arithmetic or latency',
   target:'cpu',objective:'correctness',tasks:catalogue.byHypothesis[h.id].map(t=>({...t,target:'cpu',bucket:'correctness'})),binding:{target:'cpu'}})));
 }
 suites=suites.filter(s=>s.objective===item.objective&&(s.target===config.target||s.target==='cpu'&&item.objective==='correctness'));
 const catalogueIdentity=digest(await Promise.all(suites.map(async suite=>({id:suite.id,description:suite.description,fingerprint:await suiteFingerprint(suite)}))));
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
  if(prepared.status==='ready'&&suite.learned){
   const qualify=dependencies.qualify??(await import('./skill-hypothesis-evaluation.mjs')).qualifyHypothesisTasks;
   const qualification=await qualify({...prepared,directory:cached?join(qualificationDirectory,selectionIdentity):qualificationDirectory,config,signal,resource,
    remainingSeconds:Math.max(.001,seconds-(performance.now()-started)/1000)});
   if(qualification.status==='qualified')suite.qualification=qualification;
   if(qualification.status!=='qualified'){prepared.status='needs-learning';prepared.reason=qualification.reason??'Reference/negative-control qualification did not pass';}
  }
  if(prepared.status==='ready'){prepared.proposalIdentity=proposalIdentity;prepared.catalogueIdentity=catalogueIdentity;
   if(suite.learned){Object.assign(prepared,await registerQualifiedSuite({config,suite:{...prepared,mechanism:JSON.stringify(proposal.changes)},qualification:suite.qualification,provenance:{proposalIdentity}}));}
   await writeFile(join(directory,'selected.json'),JSON.stringify(prepared,null,2)+'\n');return prepared;}
  chosen.reason=prepared.reason;
 }
 if(learned.errors.length)await event?.('skill-evaluation-qualification-failed',{hypothesis:item.id,errors:learned.errors});
 const requiredEvidence='Supply a trusted verifier/skill-evaluation.json with {target,objective,tasks:[{id,kind,target,bucket,prompt,files,checker,pairedChecker?,controls:{reference:{files},negatives:[{files}]}}]} for exactly five small independent cases. files maps sample-relative names to UTF-8 file paths relative to that manifest; include kernel.py and all fixtures. Supply a correct reference and at least one incorrect control per case. Checkers are standalone, accept a sample directory, and must accept the reference and reject each negative control. Device checkers emit the existing trusted kernel receipt, with paired checkers for optimization. Construction generates only PyTorch references/specifications; actors implement device kernels. No CPU-only proof of device behavior. '+chosen.reason+' '+learned.errors.join('; ');
 const id=enqueueLearningRequest({db:backlog.db},{target:config.target,question:'Qualify an experiment for '+item.skill+': '+JSON.stringify(proposal),
  hypothesis:JSON.stringify(proposal),requiredEvidence,learningTarget:config.target==='cpu'?'cpu':'device',backlogId:item.id,priority:item.priority,groupIds:item.groupIds});
 const stored=JSON.parse(backlog.db.prepare('SELECT data FROM skill_learning_requests WHERE id=?').get(id).data);
 stored.workKind='suite';stored.familyKey=createHash('sha256').update(JSON.stringify([config.target,item.objective,item.skill,proposal.changes.map(c=>[c.scope,c.response,c.text])])).digest('hex');
 stored.backlogIds=[...new Set([stored.backlogId,...(stored.backlogIds??[]),item.id])];
 backlog.db.prepare('UPDATE skill_learning_requests SET data=? WHERE id=?').run(JSON.stringify(stored),id);
 return {status:'needs-learning',reason:chosen.reason,learningRequest:id};
}
