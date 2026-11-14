import {evaluateNonInferiority} from './skill-noninferiority.mjs';
export const skillBuckets=['harness','correctness','optimization'];
/** Explicit pilot policies; historical resource-cost comparisons retain their gate. */
export function evaluateSkillObjective(results,{objective,projectIds,runs=2,pairedTimings,...legacy}={}){
 if(!objective||objective==='legacy')return evaluateNonInferiority(results,{projectIds,runs,...legacy});
 if(!skillBuckets.includes(objective))throw Error('Unknown skill objective');
 const report={objective,promote:false,reason:'inconclusive',projectIds,runs};
 if(objective==='harness')return {...report,reason:'user-intervention-required'};
 if(!Array.isArray(projectIds)||projectIds.length!==5||new Set(projectIds).size!==5||runs!==2)throw Error('Pilot requires five distinct cases and two repetitions');
 if(!Array.isArray(results)||results.length!==20)return report;
 const pairs=[];
 for(const id of projectIds)for(let run=0;run<runs;run++){
  const pair={projectId:id,run};
  for(const arm of ['active','candidate']){
   const found=results.filter(r=>r.projectId===id&&r.run===run&&r.arm===arm);
   if(found.length!==1||typeof found[0].correct!=='boolean')return report;
   pair[arm]=found[0];
  }
  pairs.push(pair);
 }
 const passes=Object.fromEntries(['active','candidate'].map(arm=>[arm,pairs.filter(p=>p[arm].correct).length]));
 if(objective==='correctness')return {...report,passes,extraPasses:passes.candidate-passes.active,promote:passes.candidate-passes.active>=2,
  reason:passes.candidate-passes.active>=2?'correctness-improved':'insufficient-correctness-improvement'};
 if(pairs.some(p=>!p.active.correct||!p.candidate.correct))return {...report,passes,reason:'incorrect-kernel'};
 if(!Array.isArray(pairedTimings)||pairedTimings.length!==10)return {...report,reason:'missing-paired-kernel-timing'};
 for(const pair of pairs){
  const receipts=pairedTimings.filter(r=>r.projectId===pair.projectId&&r.run===pair.run);
  if(receipts.length!==1)return {...report,reason:'missing-paired-kernel-timing'};
  const receipt=receipts[0];
  if(receipt.correct!==true||!receipt.evidence||typeof receipt.evidence!=='object')return {...report,reason:'invalid-paired-kernel-verification'};
  for(const arm of ['active','candidate']){
   if(!pair[arm].kernelSha256||receipt[arm+'KernelSha256']!==pair[arm].kernelSha256)return {...report,reason:'kernel-timing-identity-mismatch'};
   pair[arm]={...pair[arm],latencySeconds:receipt[arm+'LatencySeconds']};
  }
 }
 if(pairs.some(p=>['active','candidate'].some(a=>!Number.isFinite(p[a].latencySeconds)||p[a].latencySeconds<=0)))return {...report,reason:'missing-kernel-timing'};
 const tasks=projectIds.map(projectId=>({projectId,logSpeedup:pairs.filter(p=>p.projectId===projectId).reduce((s,p)=>s+Math.log(p.active.latencySeconds)-Math.log(p.candidate.latencySeconds),0)/runs}));
 const logSpeedup=tasks.reduce((s,t)=>s+t.logSpeedup,0)/tasks.length,speedup=Math.exp(logSpeedup),promote=logSpeedup>=Math.log(1.05)-1e-12;
 return {...report,passes,speedup,tasks:tasks.map(({projectId,logSpeedup})=>({projectId,speedup:Math.exp(logSpeedup)})),promote,
  reason:promote?'kernel-speedup':'insufficient-kernel-speedup'};
}

export function evaluationSettings(evaluations,objective){
 const value=evaluations?.[objective];
 if(!value||!Array.isArray(value.tasks)||value.tasks.length!==5||new Set(value.tasks.map(t=>t.id)).size!==5||
  !value.binding||typeof value.binding!=='object'||!Object.keys(value.binding).length)throw Error('Five frozen tasks and a registered runtime binding required for '+objective);
 for(const task of value.tasks)if(task.bucket!==objective||task.kind!=='a3-kernel'||!task.files?.['kernel.py']||!task.checker||!task.prompt||objective==='optimization'&&!task.pairedChecker)throw Error('Runnable objective-matched A3 tasks required');
 const developmentSeconds=value.developmentSeconds??3600,pairedSeconds=objective==='optimization'?(value.pairedSeconds??900):0;
 if(value.verificationSeconds!==undefined&&(!Number.isFinite(value.verificationSeconds)||value.verificationSeconds<0||value.verificationSeconds>=180))throw Error('Verification reserve must fit inside the sample elapsed allocation');
 if(!Number.isFinite(developmentSeconds)||developmentSeconds<=0||!Number.isFinite(pairedSeconds)||pairedSeconds<0||objective==='optimization'&&pairedSeconds===0)throw Error('Positive elapsed evaluation allocations required');
 return {...value,developmentSeconds,pairedSeconds};
}

// A caller-bound immutable retrieval snapshot may reconnect on a new local port.
export function retrievalIdentity(knowledge){
 if(knowledge.identity===undefined)return {url:knowledge.url,readOnly:true};
 if(typeof knowledge.identity!=='string'||!knowledge.identity.trim())throw Error('Nonempty retrieval snapshot identity required');
 return {readOnly:true,identity:knowledge.identity};
}
