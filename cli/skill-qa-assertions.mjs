import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {join} from 'node:path';

const hash=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const text=value=>typeof value==='string'&&Boolean(value.trim());
const roles=['material_claim','factual_premise','proposed_action'];
const labels=['supported','contradicted','insufficient'];
const bindingRoles=['text','source','actor','check','contract'];
function validateBindings(row,proofs){
 assert(row.bindings?.length&&row.bindings.every(binding=>bindingRoles.includes(binding.role)&&proofs.some(p=>p.reference===binding.reference)),'Every evidence binding needs a proof from the same source');
 for(const binding of row.bindings){
  if(['input:proposal','input:answers','input:answerClaims'].includes(binding.reference))assert.equal(binding.role,'text','Draft and QA text establish only what was written');
 }
 assert(bindingRoles.includes(row.basis),'Identify the type of evidence the assertion requires');
 if(row.label!=='insufficient')assert(row.bindings.some(binding=>binding.role===row.basis),'Decisive findings need a cited record for their evidence basis');
}

export function validateAssertionPlan(atoms,objections){
 assert(Array.isArray(atoms)&&atoms.length,'A nonempty assertion partition is required');
 const groups=new Map(objections.map(row=>[row.id,{row,atoms:[]}]));
 const ids=new Set();
 for(const atom of atoms){
  assert(text(atom.id)&&!ids.has(atom.id)&&groups.has(atom.objectionId),'Unknown objection or duplicate assertion ID');
  assert(roles.includes(atom.role)&&text(atom.assertion)&&['problem','description'].includes(atom.sourceField),'Each assertion needs a problem or description, role and statement');
  const source=groups.get(atom.objectionId).row;
  assert(text(source[atom.sourceField])||source.mentions?.some(mention=>text(mention.issue?.[atom.sourceField])),'Assertion source field is absent from its objection');
  ids.add(atom.id);groups.get(atom.objectionId).atoms.push(atom);
 }
 for(const group of groups.values())assert(group.atoms.some(atom=>atom.role==='material_claim'),'Every objection needs an explicit material claim about the reviewed text');
 return atoms;
}

export function assertionVerdict(atoms){
 const material=atoms.filter(row=>row.role==='material_claim');
 assert(material.length,'Missing material assertion');
 for(const row of atoms){
  assert(roles.includes(row.role)&&labels.includes(row.label)&&text(row.assertion)&&text(row.reason),'Incomplete assertion assessment');
  assert(row.proofs?.length&&row.proofs.every(p=>text(p.reference)&&text(p.quote)&&text(p.recordSha256)),'Missing materialized assertion proof');
  validateBindings(row,row.proofs);
 }
 const supported=material.filter(row=>row.label==='supported');
 const unresolved=material.filter(row=>row.label==='insufficient');
 const disposition=supported.length?'verified':unresolved.length?'unresolved':'disproved';
 const decisive=supported.length?supported:unresolved.length?unresolved:material;
 return {disposition,reason:decisive.map(row=>row.assertion+' — '+row.reason).join('\n'),proofs:decisive.flatMap(row=>row.proofs)};
}

// Each evaluator receives one assertion and the original target. Sibling verdicts are withheld.
export async function auditAssertions({objections,partition,identify,evaluate,concurrency,signal}){
 const atoms=validateAssertionPlan(partition??await identify(objections),objections),rows=[];
 const parents=new Map(objections.map(row=>[row.id,row]));
 let cursor=0;
 const workers=await Promise.allSettled(Array.from({length:Math.min(concurrency,atoms.length)},async()=>{
  while(cursor<atoms.length){
   signal?.throwIfAborted();
   const i=cursor++,atom=atoms[i];
   const result=await evaluate(atom,parents.get(atom.objectionId));
   rows[i]={...atom,label:result.label,reason:result.reason,basis:result.basis,bindings:result.bindings,proofs:result.claims.filter(c=>c.path==='reason').flatMap(c=>c.proofs)};
  }
 }));
 for(const worker of workers)if(worker.status==='rejected')throw worker.reason;
 const audits=[],claims=[];
 for(const objection of objections){
  const assertions=rows.filter(row=>row.objectionId===objection.id);
  const {proofs,...verdict}=assertionVerdict(assertions);
  const index=audits.length;
  audits.push({claimId:objection.id,...verdict,assertions});
  claims.push({path:'audits.'+index+'.reason',kind:'observation',text:verdict.reason,proofs});
 }
 return {audits,claims};
}

export async function nativeAssertionAudit({input,objections,session,schemas,proofRecords,sourceDirectory,indexFile,allowedPaths}){
 const start=performance.now();
 const invoke=async(directory,params)=>{
  input.signal?.throwIfAborted();
  const seconds=input.seconds-(performance.now()-start)/1000;
  assert(seconds>0,'Assertion audit elapsed allowance exhausted');
  return (await session({directory:join(input.directory,directory),config:input.config,signal:input.signal,seconds,
   knowledge:false,evidenceGroups:[],compactEvidence:true,...params})).value;
 };
 const task={draft:input.proposal,answers:input.answers,objective:input.objective,target:input.target,projectSummaries:[],learningProofEvidence:[]};
 return auditAssertions({objections,partition:input.panelAssertions?.filter(atom=>objections.some(row=>row.id===atom.objectionId)),
  concurrency:input.qaPanelConcurrency??5,signal:input.signal,
  identify:async group=>(await invoke('partition-'+hash(group),{resultSchema:schemas.partition,
   prompt:{...task,objections:group,task:'Separate each objection into independently checkable assertions, preserving all material allegations, their qualifiers and target. Record whether each is a material claim of a defect, supporting factual premise, or proposed action. Preserve meaningful distinctions between actors, source versions, receipts, and requirements. Include an explicit testable material claim for every objection. Use sourceField problem or description as appropriate. Assign unique IDs. Do not judge truth or return findings.'},
   validate:value=>{validateAssertionPlan(value.assertions,group);return value;}})).assertions,
  evaluate:async(atom,parent)=>invoke('assertion-'+hash([parent.id,atom]),{resultSchema:schemas.assessment,proofRequired:true,selectorOnly:true,
   proofRecords,inlineProofSelectors:false,allowedPaths:[sourceDirectory,...allowedPaths],
   prompt:{...task,assertion:atom,parent:{path:parent.path,problem:parent.problem,description:parent.description,citedSelectors:parent.citedSelectors},
    sourceIndex:indexFile,task:'Judge only this assertion. Separate static source, scored actor, verifier and task contract. Check the exact quoted draft or answer and opposing evidence. Trace source and run identities before making measured claims. Use supported, contradicted or insufficient. Cite primary evidence with selectors for reason. Set basis to the evidence type needed to decide this specific assertion. Bind each source used to the role it plays (text, source, actor, check, contract). An objection or the draft alone cannot establish an execution result. Treat a proposed action as a claim about whether the evidence warrants it.'},
   validate:value=>{validateBindings(value,value.claims.filter(c=>c.path==='reason').flatMap(c=>c.proofs));return value;}})});
}
