/** Select relevant frozen cases; missing evidence becomes curriculum work, never a generic fallback. */
import {mkdir,readFile,writeFile,realpath} from 'node:fs/promises';
import {join,resolve,dirname,relative} from 'node:path';
import {prepareHypothesisTasks} from './microtasks/hypotheses.mjs';
import {reviewSession} from './skill-review-native.mjs';
import {enqueueLearningRequest} from '../controller/skill-review.mjs';
import {prepareHypothesisEvaluation} from './skill-hypothesis-evaluation.mjs';

async function learnedCases(config,backlog,item){
 if(!backlog.db.prepare("SELECT 1 FROM sqlite_master WHERE name='skill_learning_requests'").get())return {suites:[],errors:[]};
 const rows=backlog.db.prepare("SELECT data FROM skill_learning_requests WHERE status='verified'").all();
 const suites=[],errors=[];
 for(const row of rows){
  const request=JSON.parse(row.data);if(request.backlogId!==item.id&&!request.backlogIds?.includes(item.id))continue;
  const project=backlog.db.prepare('SELECT spec FROM projects WHERE id=?').get(request.findings?.projectId);if(!project)continue;
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
export async function resolveHypothesisCases({config,backlog,item,directory,qualificationDirectory=join(directory,'qualification'),resource,signal,seconds,event},dependencies={}){
 const started=performance.now();
 await mkdir(directory,{recursive:true});
 try{return JSON.parse(await readFile(join(directory,'selected.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
 const learned=await learnedCases(config,backlog,item);
 let suites=[...(config.skills?.priority?.evaluations??[]),...learned.suites];
 if(item.objective==='correctness'){
  const catalogue=await (dependencies.prepareTasks??prepareHypothesisTasks)(join(directory,'cpu'));
  const ids=new Set((item.proposal.changes??[]).map(change=>change.id));
  suites.push(...catalogue.manifest.hypotheses.filter(h=>h.proposalIds?.some(id=>ids.has(id))).map(h=>({id:'cpu-'+h.id,description:h.id+'; CPU report/contract behavior only, not device arithmetic or latency',
   target:'cpu',objective:'correctness',tasks:catalogue.byHypothesis[h.id].map(t=>({...t,target:'cpu',bucket:'correctness'})),binding:{target:'cpu'}})));
 }
 suites=suites.filter(s=>s.objective===item.objective&&(s.target===config.target||s.target==='cpu'&&item.objective==='correctness'));
 const validate=value=>{if(!value||!value.reason?.trim()||value.catalogueId!==null&&!suites.some(s=>s.id===value.catalogueId))throw Error('Choose a supplied matching suite or null with a reason');return value;};
 const selection=suites.length?await (dependencies.select??reviewSession)({directory:join(directory,'selection'),config,signal,seconds:Math.min(60,seconds),
  prompt:{task:'Select one hypothesis-relevant evaluation suite or none. Return {catalogueId:null|string,reason}. The cases must directly exercise the proposed skill mechanism and boundaries. CPU contract/report tasks cannot validate device arithmetic, compiler behavior or kernel performance. Do not select merely because scopes overlap.',
   hypothesis:{objective:item.objective,proposal:item.proposal},catalogues:suites.map(({id,description,target,objective})=>({id,description,target,objective}))},validate}):{value:{catalogueId:null,reason:'No target-compatible evaluation suite'}};
 const chosen=validate(selection.value??selection);let suite=suites.find(s=>s.id===chosen.catalogueId);
 if(suite){
  const frozen=join(directory,'frozen');await mkdir(frozen,{recursive:true});
  suite={...suite,tasks:await Promise.all(suite.tasks.map(async(task,index)=>{
   const copied={...task};for(const key of ['checker','pairedChecker'])if(task[key]){
    copied[key]=join(frozen,index+'-'+key+'.py');await writeFile(copied[key],await readFile(resolve(config.root,task[key])));
   }return copied;
  }))};
  const prepared=prepareHypothesisEvaluation(suite);
  if(prepared.status==='ready'&&suite.learned){
   const qualify=dependencies.qualify??(await import('./skill-hypothesis-evaluation.mjs')).qualifyHypothesisTasks;
   const qualification=await qualify({...prepared,directory:qualificationDirectory,config,signal,resource,
    remainingSeconds:Math.max(.001,seconds-(performance.now()-started)/1000)});
   if(qualification.status!=='qualified'){prepared.status='needs-learning';prepared.reason=qualification.reason??'Reference/negative-control qualification did not pass';}
  }
  if(prepared.status==='ready'){await writeFile(join(directory,'selected.json'),JSON.stringify(prepared,null,2)+'\n');return prepared;}
  chosen.reason=prepared.reason;
 }
 if(learned.errors.length)await event?.('skill-evaluation-qualification-failed',{hypothesis:item.id,errors:learned.errors});
 const requiredEvidence='Supply a trusted verifier/skill-evaluation.json with {target,objective,tasks:[{id,kind,target,bucket,prompt,files,checker,pairedChecker?,controls:{reference:{files},negatives:[{files}]}}]} for exactly five small independent cases. files maps sample-relative names to UTF-8 file paths relative to that manifest; include kernel.py and all fixtures. Supply a correct reference and at least one incorrect control per case. Checkers are standalone, accept a sample directory, and must accept the reference and reject each negative control. Device checkers emit the existing trusted kernel receipt, with paired checkers for optimization. Construction generates only PyTorch references/specifications; actors implement device kernels. No CPU-only proof of device behavior. '+chosen.reason+' '+learned.errors.join('; ');
 const id=enqueueLearningRequest({db:backlog.db},{target:config.target,question:'Qualify an experiment for '+item.skill+': '+JSON.stringify(item.proposal),
  hypothesis:JSON.stringify(item.proposal),requiredEvidence,learningTarget:config.target==='cpu'?'cpu':'device',backlogId:item.id,priority:item.priority,groupIds:item.groupIds});
 const stored=JSON.parse(backlog.db.prepare('SELECT data FROM skill_learning_requests WHERE id=?').get(id).data);
 stored.backlogIds=[...new Set([stored.backlogId,...(stored.backlogIds??[]),item.id])];
 backlog.db.prepare('UPDATE skill_learning_requests SET data=? WHERE id=?').run(JSON.stringify(stored),id);
 return {status:'needs-learning',reason:chosen.reason,learningRequest:id};
}
