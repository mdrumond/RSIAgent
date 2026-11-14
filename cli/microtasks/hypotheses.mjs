/** Versioned, hypothesis-specific CPU cases; the legacy catalogue stays unchanged. */
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {constructionFixture} from './hypothesis-fixtures.mjs';
import {tasks as legacyTasks} from './tasks.mjs';
const sha=text=>createHash('sha256').update(text).digest('hex');
export const hypotheses=[
 {id:'fault-ownership',proposalIds:['preflight-fault-ownership','preflight-and-budget-triage'],deduplication:'The two saved grounded advice texts are identical; preserve both proposal provenances in the experiment manifest.',cases:['lease-ownership','runtime-identity','candidate-traceback','protected-traceback','missing-origin'],scopes:['global-kdb','preflight']},
 {id:'construction-admissibility',proposalIds:['task-construction-bundle-admissibility'],cases:['directory-form','fixture-ownership','command-nesting','baseline-entrypoint','implemented-mode'],scopes:['task-construction']},
 {id:'dtype-triage',proposalIds:['preflight-dtype-surface-from-baseline-log'],cases:['unsupported-source','warning-only','output-regression','cpu-reference','frozen-double'],scopes:['preflight','implementation']}
];
export function hypothesisForProposal(proposal){
 const ids=new Set((proposal.changes??proposal).map(c=>c.id)),matches=hypotheses.filter(h=>h.proposalIds.some(id=>ids.has(id)));
 if(matches.length!==1)throw Error('Choose exactly one known distinct hypothesis for this proposal');return matches[0];
}
export const recoveryDecisionsV2={...legacyTasks.find(t=>t.id==='recovery-decisions'),id:'recovery-decisions-events-v2',
 prompt:legacyTasks.find(t=>t.id==='recovery-decisions').prompt+' Event-list order is arbitrary: pair rows by operation and phase, then compare their at timestamps. A start row may appear after its matching end row in the list. Backwards means end.at < start.at, not list position. This version changes no old frozen task.'};
function fault(kind,n){
 const report={runtime:{expected:{build:'build-'+n},observed:{build:'build-'+n}},lease:'clear',wrapper:'baseline-preflight',candidatePrefix:'variant-'+n+'/candidate/',protectedPrefix:'variant-'+n+'/frozen/',trace:[]};
 if(kind==='lease-ownership')report.lease=n%2?'missing-receipt':'host-fence-unresolved';
 if(kind==='runtime-identity')report.runtime.observed.build='other-'+n;
 if(kind==='candidate-traceback')report.trace=[{path:report.protectedPrefix+'bench.py'},{path:report.candidatePrefix+'kernel.py',error:'Invalid tensor argument'}];
 if(kind==='protected-traceback')report.trace=[{path:report.candidatePrefix+'entry.py'},{path:report.protectedPrefix+'reference.py',error:'Trusted fixture failure'}];
 if(n===2){
  if(kind==='lease-ownership'){report.lease='clear';report.trace=[{path:report.candidatePrefix+'kernel.py',error:'Candidate argument failure'}];}
  if(kind==='runtime-identity'){report.runtime.observed={...report.runtime.expected};report.trace=[{path:report.candidatePrefix+'kernel.py',error:'Candidate argument failure'}];}
  if(kind==='candidate-traceback')report.trace.at(-1).path=report.protectedPrefix+'reference.py';
  if(kind==='protected-traceback')report.trace.at(-1).path=report.candidatePrefix+'kernel.py';
  if(kind==='missing-origin')report.lease='missing-receipt';
 }
 return report;
}
function dtype(kind,n){
 const report={binding:'synthetic-cpu-contract-'+n,operation:'operator-'+n,supported:['float16','float32'],requested:'float32',required:'float32',error:null,warning:'runtime may cast double to float',baselinePassed:true,verification:null,sourceHash:'source-'+n};
 if(kind==='unsupported-source'){report.requested='float64';report.error='dtype rejection in mutable candidate op';report.baselinePassed=false;}
 if(kind==='output-regression'){report.verification={passed:false,sourceHash:report.sourceHash};report.error='candidate output violates frozen contract';}
 if(kind==='cpu-reference'){report.referenceDtype='float64';report.verification={passed:true,sourceHash:report.sourceHash};}
 if(kind==='frozen-double'){report.required='float64';report.requested='float64';report.baselinePassed=false;report.error='target op rejects double';}
 if(n===1&&kind==='cpu-reference'){report.supported.push('float64');report.requested='float64';report.required='float64';}
 if(n===2){
  if(kind==='unsupported-source'){report.requested='float32';report.error=null;report.baselinePassed=true;}
  if(['warning-only','output-regression'].includes(kind)){report.verification={passed:true,sourceHash:report.sourceHash};report.error=null;}
  if(kind==='cpu-reference')report.verification.sourceHash='older-source';
  if(kind==='frozen-double'){report.supported.push('float64');report.error=null;report.baselinePassed=true;report.verification={passed:true,sourceHash:report.sourceHash};}
 }
 return report;
}
function brokenConstruction(kind,n){
 const fixture=constructionFixture(n?'variant-'+n+'/':''),p=fixture.proposal;
 if(kind==='directory-form'){p.source+='/implementation.py';p.reference+='/implementation.py';}
 if(kind==='fixture-ownership')p.verifier.files.push(p.reference+'/implementation.py',p.negative_controls[0]+'/implementation.py');
 if(kind==='command-nesting'){p.verifier.baseline_construction_command=p.baseline_construction_command;delete p.baseline_construction_command;}
 if(kind==='baseline-entrypoint')p.baseline_construction_command.entrypoint=(n?'variant-'+n+'/':'')+'quick.py';
 if(kind==='implemented-mode')p.baseline_construction_command.args=['--mode',n?'peek':'little'];
 return fixture;
}
const contract={classifications:['candidate-defect','operator-blocked','specification-defect','unproven','verified'],actions:['repair-source','report-operator','collect-evidence','run-verifier','accept'],
 supportScope:'supported describes ONLY this report binding/operator, not other operators, Torch-NPU, Catlass DSL or actual hardware.',
 semantics:'candidate-defect means mutable source violates an otherwise satisfiable contract; operator-blocked means runtime/closure/protected code prevents actor-owned repair; specification-defect means the frozen requirement cannot be supported; unproven means evidence does not establish correctness or an origin; verified requires a successful frozen check of this exact source. Return classification and action only. Never edit harness-state.json or mutate report arguments.'};
export async function prepareHypothesisTasks(directory,{node=process.execPath}={}){
 await mkdir(directory,{recursive:true});const template=await readFile(new URL('./hypothesis-check.py',import.meta.url),'utf8'),oracle=fileURLToPath(new URL('./construction-oracle.mjs',import.meta.url));
 const tasks=[],metadata=[];
 for(const h of hypotheses)for(const kind of h.cases){
  const id=h.id+'-'+kind,construct=h.id==='construction-admissibility';
  const variants=Array.from({length:3},(_,n)=>construct?brokenConstruction(kind,n):h.id==='fault-ownership'?fault(kind,n):dtype(kind,n));
  const files=construct?{...variants[0].files,'proposal.json':JSON.stringify(variants[0].proposal,null,2),
   'oracle.mjs':`import {oracleMain} from ${JSON.stringify(new URL('./construction-oracle.mjs',import.meta.url).href)}; await oracleMain(['--root',process.cwd(),'--proposal',process.argv[2]??'proposal.json',...(process.argv.includes('--controls')?['--controls']:[])]);\n`,
   'oracle-command.json':JSON.stringify([node,'oracle.mjs','proposal.json'])}:
   {'report.json':JSON.stringify(variants[0],null,2),'contract.json':JSON.stringify(contract,null,2),'harness-state.json':'{"owner":"controller","receipt":"immutable-fixture"}\n'};
  files['kernel.py']=construct?'def repair_proposal(proposal):\n    return proposal\n':h.id==='fault-ownership'?'def route(report):\n    raise NotImplementedError()\n':'def triage(report):\n    raise NotImplementedError()\n';
  const prompt=construct?'Implement repair_proposal(proposal), returning a corrected copy accepted by the pinned production construction oracle. Preserve goal, performance_goal, and all declared target command entrypoints/arguments. Only proposal layout/CPU-construction fields may change; do not edit fixture files. proposal.json is the public incident. oracle-command.json gives a callable oracle command; oracle.mjs also accepts --controls for actual CPU controls. Use the oracle as the admission contract.':`Implement ${h.id==='fault-ownership'?'route':'triage'}(report), returning exactly {classification, action} using contract.json. report.json is the public incident; other reports vary paths, runtime identities, proof freshness and analogous values. Trace frames run outermost to innermost. Reports are supplied CPU evidence, not live device probes. Preserve the report argument and harness-state.json.`;
  const protectedHashes=Object.fromEntries(Object.entries(files).filter(([name])=>name!=='kernel.py').map(([name,text])=>[name,sha(text)]));
  const config={hypothesis:h.id,kind,id,variants,protectedHashes,node,oracle};
  const checker=join(directory,id+'.py');await writeFile(checker,'import json\nCONFIG=json.loads('+JSON.stringify(JSON.stringify(config))+')\n'+template);
  tasks.push({id,kind:'hypothesis-case',scopes:h.scopes,prompt,files,checker});metadata.push({id,hypothesis:h.id,case:kind,heldoutVariants:2,checkerSha256:sha(await readFile(checker,'utf8')),publicInputSha256:sha(JSON.stringify(files)),promptSha256:sha(prompt)});
 }
 const manifest={format:1,catalogue:'skill-hypotheses-v2',hypotheses,cases:metadata,oracleSha256:sha(await readFile(oracle,'utf8')),productionAdmissionSha256:sha(await readFile(new URL('../project-generation.mjs',import.meta.url),'utf8')),commandCompilerSha256:sha(await readFile(new URL('../../controller/project-commands.mjs',import.meta.url),'utf8')),claim:'CPU contract/ownership evidence only; no device performance claim'};
 await writeFile(join(directory,'catalogue.json'),JSON.stringify(manifest,null,2));return {tasks,manifest,byHypothesis:Object.fromEntries(hypotheses.map(h=>[h.id,tasks.filter(t=>t.id.startsWith(h.id+'-'))]))};
}
