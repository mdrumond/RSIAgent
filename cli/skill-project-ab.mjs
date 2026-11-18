import {skillConfig,skillModelIdentity} from '../controller/skill-context.mjs';
/** Opt-in, immutable shadow experiment: one proposal covers five cases, four actors total. */
import {mkdir,readFile,writeFile,rename} from 'node:fs/promises';
import {join,dirname,resolve,relative,isAbsolute} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {unknownExecution} from './skill-microeval.mjs';
import {nativeProjectRole,projectDeadline} from './skill-project-native.mjs';
import {hypothesisTarget,hypothesisEvaluationSettled} from './skill-hypothesis-evaluation.mjs';
const sha=x=>createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
const save=async(path,value)=>{const tmp=path+'.'+randomUUID();await writeFile(tmp,JSON.stringify(value,null,2)+'\n');await rename(tmp,path);};
const text=x=>typeof x==='string'&&x.trim();
const successBases=['numerical-contract','compile-and-correct'];
const scoring={candidateFullPasses:2,baselineFullPassesAtMost:1,noCorrectnessRegression:true};
const stable=x=>Array.isArray(x)?x.map(stable):x&&typeof x==='object'?Object.fromEntries(Object.keys(x).sort().map(k=>[k,stable(x[k])])):x;
export function validateSkillProject(project){
 if(!text(project?.id)||project.objective!=='correctness'||!['a3','a5'].includes(project.target)||!text(project.prompt)||!text(project.checker)||!text(project.files?.['kernel.py']))throw Error('Prepared device correctness project required; optimization needs a separate paired-performance protocol');
 if(project.verificationSeconds!==undefined&&(!Number.isFinite(project.verificationSeconds)||project.verificationSeconds<=0))throw Error('project.verificationSeconds must be finite and positive');
 if(project.successBasis!==undefined&&!successBases.includes(project.successBasis))throw Error('Unknown project successBasis');
 if(project.cases?.length!==5||project.cases.some(c=>!/^[a-z0-9-]+$/.test(c.id)||!c.parameters||typeof c.parameters!=='object'||Array.isArray(c.parameters)||!Object.keys(c.parameters).length)||new Set(project.cases.map(c=>c.id)).size!==5||new Set(project.cases.map(c=>JSON.stringify(stable(c.parameters)))).size!==5)throw Error('Five distinct case IDs and parameter sets required');
 const maps=[project.files,project.controls?.reference?.files,...(project.controls?.negatives??[]).map(c=>c.files)];
 if(maps.length<3||maps.some(files=>!text(files?.['kernel.py'])))throw Error('Reference and negative control source required');
 for(const files of maps)for(const[name,value]of Object.entries(files)){if(['AB_GUIDANCE.md','AB_CASES.json'].includes(name))throw Error('Reserved experiment file: '+name);const rel=relative('/task',resolve('/task',name));if(isAbsolute(name)||!rel||rel==='..'||rel.startsWith('../')||!text(value))throw Error('Project files must be named UTF-8 task descendants');}
 return project;
}
export function validateSkillTestDesign(value,{project,evidence}){
 if(!['propose','learn','no-test'].includes(value?.decision)||!text(value.reason)||value.objective!==project.objective||value.target!==project.target||!Array.isArray(value.evidenceRefs)||!value.evidenceRefs.length)throw Error('Design requires an explicit decision, matching objective/target and exact evidence references');
 if((value.successBasis??'numerical-contract')!==(project.successBasis??'numerical-contract'))throw Error('Design successBasis must match the frozen project');
 const unknown=value.evidenceRefs.filter(ref=>!evidence.some(e=>e.reference===ref));if(unknown.length)throw Error('Unknown evidenceRefs: '+JSON.stringify(unknown)+'. Use only allowedEvidenceRefs, not JSON paths or abbreviated IDs.');
 if(value.decision!=='propose')return value;
 if(!Array.isArray(value.caseIds)||value.caseIds.length!==5||new Set(value.caseIds).size!==5||value.caseIds.some(id=>!project.cases.some(c=>c.id===id))||value.actorUnit!=='one-project-five-cases'||value.repetitions!==2||JSON.stringify(value.arms)!==JSON.stringify(['baseline','candidate'])||!text(value.disconfirmingOutcome)||!text(value.success)||!text(value.inconclusive)||JSON.stringify(stable(value.scoring))!==JSON.stringify(stable(scoring)))throw Error('Design must preserve all five cases, four independent actors, fair arms and falsifiable outcomes');
 return value;
}
export function validateSkillTestQa(value){
 if(!['approved','learn','reject'].includes(value?.verdict)||![value.reason,value.disconfirmation,value.leakage].every(text))throw Error('Independent design QA requires verdict, reason, disconfirmation and leakage assessment');return value;
}
export function groupedSkillVerdict(results,{successBasis='numerical-contract'}={}){
 if(!successBases.includes(successBasis))throw Error('Unknown successBasis');
 if(results.length!==4||results.some(r=>typeof r.correct!=='boolean'))return {promote:false,reason:'inconclusive',successBasis};
 const passes=Object.fromEntries(['baseline','candidate'].map(arm=>[arm,results.filter(r=>r.arm===arm&&r.correct).length]));
 if(['baseline','candidate'].some(arm=>[0,1].some(rep=>results.filter(r=>r.arm===arm&&r.rep===rep).length!==1)))return {promote:false,reason:'inconclusive',successBasis};
 const promote=passes.candidate===2&&passes.baseline<2;return {promote,reason:promote?(successBasis==='compile-and-correct'?'compatibility-success-improved':'correctness-improved'):passes.baseline===2&&passes.candidate===2?'baseline-ceiling':'no-demonstrated-benefit',passes,successBasis};
}
function receipt(row,files,checker,cases,successBasis){
 if(!row||row.sourceFilesMatch!==true||row.checkerSha256!==sha(checker))return {correct:null,reason:'invalid-verification-identity'};
 let value;try{value=JSON.parse(row.output.trim().split('\n').at(-1));}catch{return {correct:null,reason:'missing-verification-receipt'};}
 if(value.kernelSha256!==sha(files['kernel.py']))return {correct:null,reason:'kernel-identity-mismatch'};
 if(value.bucket==='harness'||['infrastructure','unknown-execution'].includes(value.evidence?.kind))return {correct:null,reason:'infrastructure-or-unknown-outcome',receipt:value};
 if(value.evidence?.kind==='case-resource-allowance'&&successBasis!=='compile-and-correct')return {correct:null,reason:'resource-outcome-not-numerical-verdict',receipt:value};
 if(row.exitCode===0&&value.correct===true&&/^npu:\d+$/.test(value.evidence?.device??'')&&cases.every(c=>value.evidence?.checkedCases?.includes(c.id)))return {correct:true,receipt:value};
 if(row.exitCode===1&&value.correct===false&&text(value.evidence?.reason))return {correct:false,receipt:value};
 return {correct:null,reason:'incomplete-or-infrastructure-verification',receipt:value};
}
/** directory must be fresh. A hard interruption is retained and never replayed by this API. */
export async function runSkillProjectAB({directory,config,project,guidance,evidence,binding,resource,seconds,actorSeconds=180,signal},dependencies={}){
 config=skillConfig(config);
 validateSkillProject(project);project={...project,successBasis:project.successBasis??'numerical-contract'};if(!text(guidance?.baseline)||!text(guidance?.candidate)||guidance.baseline===guidance.candidate||!evidence?.length||evidence.some(e=>!text(e.reference)||!text(e.text))||!binding?.environment||binding.environment.health?.product!==project.target||!Number.isFinite(seconds)||seconds<=60||!Number.isFinite(actorSeconds)||actorSeconds<=0)throw Error('Distinct guidance, evidence, target binding and positive elapsed allocation required');
 const began=performance.now(),elapsed=()=>(performance.now()-began)/1000,remaining=()=>Math.max(0,seconds-elapsed()),available=()=>Math.max(0,remaining()-30);
 await mkdir(dirname(directory),{recursive:true});await mkdir(directory);const checker=await readFile(project.checker,'utf8'),frozen={project:{...project,checker},guidance,evidence,binding,modelProfiles:Object.fromEntries(await Promise.all(['actor','skill_writer_actor'].map(async key=>[key,await skillModelIdentity(config[key]??config.actor)])))};
 const deadline=projectDeadline(seconds,signal,began),work=deadline.signal;await save(join(directory,'frozen.json'),frozen);const identity=sha(frozen),state={protocol:'skill-project-ab-v1',identity,seconds,startedAt:new Date().toISOString(),elapsedSeconds:elapsed(),status:'running',shadow:true,successBasis:project.successBasis,verdict:{promote:false,reason:'not-evaluated',successBasis:project.successBasis},results:[]};
 const persist=async()=>{state.elapsedSeconds=elapsed();await save(join(directory,'result.json'),state);};
 const role=dependencies.role??nativeProjectRole,call=async(name,args)=>{if(available()<=0)throw Error('Experiment elapsed allocation exhausted');if(args.editable&&available()<args.seconds)throw Error('Insufficient remaining elapsed time for a fair actor allocation');state.stage=name;await persist();const result=await role({...args,directory:join(directory,name),seconds:Math.min(args.seconds??120,available()),signal:work});if(result.error||result.expired)throw Object.assign(Error(result.error??'Model elapsed allocation exhausted'),{roleResult:result});return result;};
 const pendingCustody=new Set();
 const batch=async(root,entries)=>{if(available()<=0)throw Error('Experiment elapsed allocation exhausted');pendingCustody.add(root);const target=dependencies.target??hypothesisTarget({root,binding,resource,signal:work,stopSignal:work,localExecution:config.local_execution});return target.qualifyBatch({directory:root,entries,seconds:available(),signal:work,sampleStopSignal:work});};
 const settled=async root=>{const closed=await (dependencies.settled??hypothesisEvaluationSettled)({directory:root,config,binding,resource,signal:work});if(closed)pendingCustody.delete(root);return closed;};
 const packet={project:{id:project.id,objective:project.objective,target:project.target,successBasis:project.successBasis,prompt:project.prompt,cases:project.cases,files:project.files},runtimeIdentity:binding.runtimeIdentity,guidance,evidence,allowedEvidenceRefs:evidence.map(e=>e.reference),scoring};
 try{
  const proposed=await call('design',{files:{'input.json':JSON.stringify(packet,null,2)},submission:'proposal.json',validate:value=>validateSkillTestDesign(value,{project,evidence}),profile:config.skill_writer_actor??config.actor,prompt:'Read input.json. Write proposal.json as compact JSON: {decision:propose|learn|no-test,reason,objective,target,successBasis:input.project.successBasis,evidenceRefs:[only exact strings from input.allowedEvidenceRefs],caseIds:[all five input IDs],actorUnit:"one-project-five-cases",repetitions:2,arms:["baseline","candidate"],success,disconfirmingOutcome,inconclusive,scoring:input.scoring}. Do not cite JSON paths, filenames, field names or shortened IDs as evidenceRefs. Draft with stock write/edit and call submit_review with {} when ready; if submission is rejected, correct the draft and submit again within this same session. Propose a fair skill A/B on the trusted prepared project; sole treatment is guidance, not different code/cases/budgets. Do not require a historical failure to recur, infer kernel speed from host latency, or manufacture a negative control. Use project.successBasis exactly. numerical-contract measures output semantics; compile-and-correct measures producing correct outputs within the same predeclared elapsed compatibility allowance, not kernel speedup. Do not infer numerical error from a resource allowance outcome. Benefit means candidate full passes in both repetitions and fewer than two baseline full passes, not winning every pair. Public file contents are embedded in input.json; they are not external paths to locate. A proposed plan remains subject to independent control qualification; absence of fresh control results is not itself proof of a bad test. If the project is not informative, choose learn or no-test. The evaluator owns controls, runtime setup and execution. No outside files. Call submit_review after writing the file.'});
  state.design=validateSkillTestDesign(proposed.value,{project,evidence});await persist();if(state.design.decision!=='propose'){state.status='needs-learning';state.reason=state.design.reason;return state;}
  const reviewed=await call('design-qa',{files:{'input.json':JSON.stringify({...packet,design:state.design},null,2)},submission:'qa.json',validate:validateSkillTestQa,profile:config.skill_writer_actor??config.actor,prompt:'Read input.json and independently challenge the proposed A/B. Write qa.json {verdict:approved|learn|reject,reason,disconfirmation,leakage}. Can the proposed outcome disprove the skill benefit? Are cases identical across arms, reference/controls hidden from actors, correctness distinct from performance and historical bugs allowed to have changed? Respect project.successBasis: compile-and-correct is binary compilation-and-correctness within the same predeclared elapsed validity constraint, not kernel speedup; numerical-contract excludes resource allowance failures from numerical verdicts. You may reject unsupported or asymmetric bounds, but do not conflate explicitly declared compatibility success with a claim of numerical mismatch. Reject biased controls and unsupported attribution. Prefer learn to invented evidence. Do not execute or change the design. Draft compact JSON in qa.json and call submit_review with {}.'});
  const qa=validateSkillTestQa(reviewed.value);state.qa=qa;await persist();if(qa.verdict!=='approved'){state.status='needs-learning';state.reason=qa.reason;return state;}
  const controls=[['reference',project.controls.reference],...project.controls.negatives.map((c,i)=>['negative-'+i,c])].map(([key,c])=>({key,cwd:key,files:{...project.files,...c.files},checker,args:['.'],timeoutSeconds:project.verificationSeconds??180}));
  const controlRoot=join(directory,'controls');state.stage='qualification';await persist();state.controls=await batch(controlRoot,controls);state.controlsSettled=await settled(controlRoot);await persist();
  if(!state.controlsSettled){state.status='unresolved';state.reason='control-custody-unresolved';return state;}
  const qualified=controls.map(c=>({...c,...receipt(state.controls.find(r=>r.key===c.key),c.files,checker,project.cases,project.successBasis)}));
  if(qualified[0].correct!==true||qualified.slice(1).some(c=>c.correct!==false)){state.status='needs-learning';state.reason=qualified.slice(1).some(c=>c.correct===true)?'historical-negative-now-passes':'controls-inconclusive';return state;}
  const proposals=[];
  for(const[rep,arm]of [[0,'baseline'],[0,'candidate'],[1,'candidate'],[1,'baseline']]){
   const files={...project.files,'AB_GUIDANCE.md':guidance[arm],'AB_CASES.json':JSON.stringify(project.cases,null,2)},sample=await call(`actors/${rep}-${arm}`,{files,editable:['kernel.py'],seconds:actorSeconds,profile:config.actor,prompt:`Read AB_GUIDANCE.md, AB_CASES.json and these task files: ${Object.keys(project.files).join(', ')}. ${project.prompt}\nProduce one proposal covering all five cases. Edit only kernel.py using stock file tools. No device commands, setup, references or benchmark construction. The evaluator owns verification. Finish without claiming unverified success.`});
   const actual={};for(const name of Object.keys(files)){const bytes=await readFile(join(sample.cwd,name),'utf8');if(name in project.files)actual[name]=bytes;if(name!=='kernel.py'&&bytes!==files[name])throw Error('Protected actor task file changed: '+name);}
   proposals.push({rep,arm,files:actual});state.results.push({rep,arm,kernelSha256:sha(actual['kernel.py']),actorSeconds:sample.elapsedSeconds,stats:sample.stats,correct:null});await persist();
  }
  if(sha(JSON.parse(await readFile(join(directory,'frozen.json'),'utf8')))!==identity)throw Error('Frozen experiment inputs changed');
  const verifyRoot=join(directory,'verification');state.stage='verification';await persist();state.verification=await batch(verifyRoot,proposals.map(p=>({key:p.rep+'/'+p.arm,cwd:p.rep+'-'+p.arm,files:p.files,checker,args:['.'],timeoutSeconds:project.verificationSeconds??180})));state.verificationSettled=await settled(verifyRoot);
  for(const row of state.results){const p=proposals.find(p=>p.rep===row.rep&&p.arm===row.arm);Object.assign(row,receipt(state.verification.find(v=>v.key===row.rep+'/'+row.arm),p.files,checker,project.cases,project.successBasis));}
  state.verdict=state.verificationSettled?groupedSkillVerdict(state.results,{successBasis:project.successBasis}):{promote:false,reason:'custody-unresolved',successBasis:project.successBasis};state.status=state.verificationSettled?'complete':'unresolved';state.reason=state.verdict.reason;return state;
 }catch(error){state.status=unknownExecution(error)?'unresolved':'inconclusive';state.reason=error.message;state.error={name:error.name,code:error.code,handle:error.handle,roleResult:error.roleResult};return state;}
 finally{for(const root of pendingCustody){let closed=false;try{closed=await settled(root);}catch(error){(state.custodyErrors??=[]).push({directory:root,name:error.name,code:error.code,handle:error.handle,message:error.message});}if(!closed){state.status='unresolved';state.reason='target-custody-unresolved';state.verdict={promote:false,reason:state.reason,successBasis:project.successBasis};}}deadline.dispose();await persist();}
}
