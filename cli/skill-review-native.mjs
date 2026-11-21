/** Native Pi/KDB and existing curriculum/evaluation adapters for isolated reviews. */
import {mkdir,readFile,writeFile,appendFile,cp,symlink,realpath,stat,rename} from 'node:fs/promises';
import {join,resolve,dirname} from 'node:path';
import {Type} from 'typebox';
import {randomUUID} from 'node:crypto';
import {proofSelectorCatalogue} from './claim-proofs.mjs';
import {Check,Errors} from 'typebox/value';
import {discoveryProofSources} from './skill-discovery-proofs.mjs';
import {frozenEvidencePacket,scopedBaseline,scopedQaGroups} from './skill-review-evidence.mjs';
import {projectSummaries,learningProofEvidence,claimEvidence,validateClaimedResult,proofInstructions} from './project-summary.mjs';
import {skillConfig,skillModelFromProfile} from '../controller/skill-context.mjs';
import {ensureKnowledgeService} from './knowledge.mjs';
import {createKdbBridge} from '../integrations/kdb-extension.mjs';
import {failureTriageBoundaries} from '../controller/skill-failure-triage.mjs';
import {State} from '../controller/state.mjs';
import {importProject} from './instance.mjs';
import {operate} from './runtime.mjs';
import {fileURLToPath} from 'node:url';
import {ownerStopped} from '../controller/lifecycle.mjs';
import {setTimeout as delay} from 'node:timers/promises';
import {refillProjects} from './project-generation.mjs';
import {runMicroeval,finalizeKernelPairs} from './skill-microeval.mjs';
import {tasks} from './microtasks/tasks.mjs';
import {stages as workflowStages} from '../controller/workflow.mjs';
import {validateSummaryTargets,qaFindingKinds} from './skill-review-qa.mjs';
import {compactSummary,sourcePointers} from './skill-evidence-summary.mjs';
import {diagnosticIdentity} from '../controller/skill-review.mjs';
import {evaluateSkillObjective,evaluationSettings} from '../controller/skill-objectives.mjs';

export const REVIEW_CONTRACT_VERSION=2;
const nonempty=()=>Type.String({minLength:1});
const proofFields={reference:nonempty(),quote:Type.Optional(nonempty()),factId:Type.Optional(nonempty()),value:Type.Optional(Type.Unknown()),startLine:Type.Optional(Type.Integer({minimum:1})),endLine:Type.Optional(Type.Integer({minimum:1})),segment:Type.Optional(Type.Integer({minimum:1})),field:Type.Optional(Type.String()),sourceSha256:Type.Optional(nonempty()),recordSha256:Type.Optional(nonempty()),sourceRefs:Type.Optional(Type.Array(nonempty())),sourcePaths:Type.Optional(Type.Array(nonempty()))};
const proofSchema=Type.Object(proofFields,{additionalProperties:false});
const claimSchema=Type.Object({path:nonempty(),text:Type.Optional(nonempty()),kind:Type.Union([Type.Literal('observation'),Type.Literal('hypothesis')]),proofs:Type.Array(proofSchema,{minItems:1})},{additionalProperties:false});
const claimsSchema=Type.Array(claimSchema,{minItems:1});
const selectorFields=Object.fromEntries(['reference','startLine','endLine','segment','field'].map(k=>[k,proofFields[k]]));
const selectorProofSchema=Type.Object(selectorFields,{additionalProperties:false});
const selectorClaims=(path=nonempty())=>Type.Array(Type.Object({path,kind:claimSchema.properties.kind,proofs:Type.Array(selectorProofSchema,{minItems:1})},{additionalProperties:false}),{minItems:1});
const selectorInstructions='Submit claims only as {path,kind,proofs:[{reference,startLine?,endLine?,segment?,field?}]}. The path identifies prose in your RESULT, not the input. Omit claim.text and all quotes, hashes, fact values, source lists and copied input metadata: software derives them. Lines are 1-based inclusive; excerpt records need segment. A whole typed fact uses only reference. JSON field selectors require a listed eligible input record and an RFC6901 pointer such as /changes/0/text; use proofSelectors for each reference and its supported selectors. Cite original primary references, never a summary as its own proof.';
const selectorShape=result=>({...result,...(result.issues?{issues:result.issues.map(issue=>typeof issue==='string'?issue:Object.fromEntries(['problem','description','path','field','summaryItemId','kind'].filter(k=>Object.hasOwn(issue,k)).map(k=>[k,issue[k]])))}:{}),claims:result.claims?.map(({text,...claim})=>({...claim,proofs:claim.proofs?.map(p=>Object.fromEntries(Object.keys(selectorFields).filter(k=>Object.hasOwn(p,k)).map(k=>[k,p[k]])))}))});
const choices=values=>Type.Union(values.map(value=>Type.Literal(value)));
const strings=()=>Type.Array(nonempty());
const classificationFields={cause:choices(['actor','harness','specification','provider','unknown']),bucket:Type.Optional(choices(['harness','correctness','optimization'])),reason:nonempty(),coverage:choices(['covered','uncovered','unknown']),guidanceIds:strings(),evidenceRefs:Type.Array(nonempty(),{minItems:1}),gap:Type.Optional(choices(['missing','insufficient','not-followed','unknown']))};
export const classificationSchema=Type.Object({...classificationFields,claims:selectorClaims(Type.Literal('reason'))},{additionalProperties:false});
const classificationBatch=(fields)=>Type.Object({classifications:Type.Array(Type.Object({groupId:nonempty(),...fields},{additionalProperties:false}),{minItems:1}),claims:selectorClaims(Type.String({pattern:String.raw`^classifications(?:[.][0-9]+|\[[0-9]+\])[.]reason$`}))},{additionalProperties:false});
export const classificationBatchSchema=classificationBatch(classificationFields);
const failureFirstBatchSchema=classificationBatch({...classificationFields,bucket:choices(['harness','correctness','optimization','unknown'])});
const classificationOptions=(input,batch=false)=>({...input,proofRequired:true,selectorOnly:true,resultSchema:batch?(input.failureFirst?failureFirstBatchSchema:classificationBatchSchema):classificationSchema,
 validate:async(value,retrievals)=>({...await input.validate(value,retrievals),claims:value.claims})});
export const proofResultSchema=Type.Object({claims:claimsSchema},{additionalProperties:true});
const summaryItemId=Type.Optional(Type.String({minLength:1,description:'For a defect in an evidence summary, copy the exact evidenceSummary.items[].id to route this issue back to its summarizer.'}));
const selectorIssueFields={summaryItemId,path:Type.Optional(nonempty()),field:Type.Optional(nonempty())};
const selectorIssueSchema=Type.Union([nonempty(),Type.Object({...selectorIssueFields,problem:nonempty(),description:Type.Optional(nonempty())},{additionalProperties:false}),Type.Object({...selectorIssueFields,description:nonempty(),problem:Type.Optional(nonempty())},{additionalProperties:false})]);
const qaFindingFields={...selectorIssueFields,kind:choices(qaFindingKinds)};
const typedQaIssueSchema=Type.Union([
 Type.Object({...qaFindingFields,problem:nonempty(),description:Type.Optional(nonempty())},{additionalProperties:false}),
 Type.Object({...qaFindingFields,description:nonempty(),problem:Type.Optional(nonempty())},{additionalProperties:false}),
]);
const issueFields={summaryItemId,...Object.fromEntries(Object.entries(proofFields).map(([k,v])=>[k,Type.Optional(v)])),path:Type.Optional(nonempty())};
const issueSchema=Type.Union([nonempty(),Type.Object({...issueFields,problem:nonempty(),description:Type.Optional(nonempty())},{additionalProperties:false}),Type.Object({...issueFields,description:nonempty(),problem:Type.Optional(nonempty())},{additionalProperties:false})]);
const requestSchema=Type.Object({groupIds:Type.Array(nonempty()),question:nonempty(),hypothesis:nonempty(),requiredEvidence:nonempty(),learningTarget:Type.Union([Type.Literal('cpu'),Type.Literal('device')]),workKind:Type.Optional(Type.Union([Type.Literal('microlearning'),Type.Literal('phase1')]))},{additionalProperties:false});
const contradiction={contradictoryEvidence:Type.Optional(strings())};
const changeSchema=Type.Union([Type.Object({action:Type.Literal('remove'),id:nonempty(),...contradiction},{additionalProperties:false}),Type.Object({action:Type.Literal('upsert'),id:nonempty(),scope:nonempty(),text:nonempty(),critical:Type.Boolean(),negative:Type.Boolean(),...contradiction,evidenceRefs:Type.Optional(strings()),response:Type.Optional(choices(['actor-change','report-blocker','reserve-validation-runway','retrieve-scoped-evidence'])),measurements:Type.Optional(Type.Array(Type.Object({reference:nonempty(),durationSeconds:Type.Number({minimum:0})},{additionalProperties:false})))},{additionalProperties:false})]);
export const proposalSchema=Type.Object({action:choices(['update','learn','no-change']),reason:nonempty(),groupIds:Type.Array(nonempty(),{minItems:1}),memoryRefs:Type.Optional(strings()),changes:Type.Optional(Type.Array(changeSchema)),question:Type.Optional(nonempty()),hypothesis:Type.Optional(nonempty()),requiredEvidence:Type.Optional(nonempty()),learningTarget:Type.Optional(choices(['cpu','device'])),workKind:Type.Optional(choices(['microlearning','phase1'])),learningRequests:Type.Optional(Type.Array(requestSchema)),claims:selectorClaims()},{additionalProperties:false});
const qaAnswerResultSchema=Type.Object({answers:Type.Object(Object.fromEntries(['action','mechanism','example','evidence','objective'].map(k=>[k,Type.Object({text:nonempty(),refs:Type.Array(nonempty())},{additionalProperties:false})])),{additionalProperties:false}),changes:Type.Optional(Type.Array(Type.Unknown())),memoryRefs:Type.Optional(Type.Array(nonempty())),reason:Type.Optional(nonempty()),claims:claimsSchema},{additionalProperties:false});
const qaReviewResultSchema=Type.Object({verdict:Type.Union(['approved','revise','learn','inconclusive'].map(x=>Type.Literal(x))),reason:nonempty(),issues:Type.Optional(Type.Array(issueSchema)),learningRequests:Type.Optional(Type.Array(requestSchema)),claims:claimsSchema},{additionalProperties:false});

async function reviewEvidence(directory,groups=[]){
 const previews=[];await mkdir(join(directory,'evidence'),{recursive:true});
 for(const group of groups){
  const representatives=[...new Map(group.occurrences.map(o=>[diagnosticIdentity(o),o])).values()];
  const linked=group.linkedDiagnostics??[],linkedFile=join(directory,'evidence',group.id+'-linked-diagnostics.jsonl');
  await writeFile(linkedFile,linked.map(d=>JSON.stringify(d)).join('\n')+'\n');
  const all=join(directory,'evidence',group.id+'-occurrences.jsonl'),diagnostics=join(directory,'evidence',group.id+'-diagnostics.jsonl');
  await writeFile(all,group.occurrences.map(o=>JSON.stringify(o)).join('\n')+'\n');
  await writeFile(diagnostics,representatives.map(o=>JSON.stringify(o)).join('\n')+'\n');
  previews.push({...group,linkedDiagnostics:linked.slice(0,6),linkedDiagnosticCount:linked.length,linkedDiagnosticsFile:linkedFile,occurrences:representatives.slice(0,6),occurrenceCount:group.occurrences.length,
   distinctDiagnostics:representatives.length,fullOccurrences:all,diagnosticsFile:diagnostics});
 }
 return {previews};
}
/** Resolve existing ancestors too: native write may create a new file below a symlink. */
export async function reviewWritePathAllowed(path,directory){
 if(!directory)return false;const root=await realpath(directory);let cursor=resolve(path);
 while(true){try{const actual=await realpath(cursor);return actual!==root&&actual.startsWith(root+'/')||actual===root&&cursor!==resolve(path);}catch(error){if(error.code!=='ENOENT')throw error;const parent=dirname(cursor);if(parent===cursor)return false;cursor=parent;}}
}
export async function reviewSession({directory,config,prompt,validate,signal,seconds,knowledge=false,evidenceGroups=[],allowedPaths=[],proofRequired=false,resultSchema,previousResultSchema,proofRecords,requiredReads=[],compactEvidence=false,selectorOnly=false,discoveryProofRecords,writableDirectory,continueFamilyDraft=false,onModelTiming},{wrapStream,retrySettings}={}){
 const sdk=await import('@earendil-works/pi-coding-agent'),agentDir=join(directory,'agent');
 await mkdir(agentDir,{recursive:true});
 // Use the pinned Pi path resolver so policy and native filesystem tools agree.
 const {resolveToCwd}=await import(new URL('./core/tools/path-utils.js',import.meta.resolve('@earendil-works/pi-coding-agent')).href);
 if(writableDirectory){await mkdir(writableDirectory,{recursive:true});writableDirectory=await realpath(writableDirectory);}
 const declarations=[directory,...(writableDirectory?[writableDirectory]:[]),...evidenceGroups.flatMap(g=>[...g.occurrences,...(g.linkedDiagnostics??[])].flatMap(o=>(o.artifacts??[]).map(a=>a.path))),...allowedPaths,...requiredReads];
 const allowed=await Promise.all([...new Set(declarations)].map(async path=>{try{const canonical=await realpath(path);return {path:canonical,directory:(await stat(canonical)).isDirectory()};}catch(error){if(error.code!=='ENOENT')throw error;return {path:resolve(path),directory:false};}}));
 const evidence=await reviewEvidence(directory,evidenceGroups);
 if(prompt.group)prompt={...prompt,group:evidence.previews[0]};if(prompt.groups)prompt={...prompt,groups:evidence.previews};
 const packet=compactEvidence?{observations:[],artifacts:[]}:await frozenEvidencePacket(evidenceGroups);
 prompt={...prompt,filesystemAccess:{allowed,requiredReads,...(writableDirectory?{writableDirectory}:{})},projectSummaries:prompt.projectSummaries??await projectSummaries(config,evidenceGroups),learningProofEvidence:prompt.learningProofEvidence??await learningProofEvidence(prompt.learning,prompt.learningEvidence),frozenEvidence:packet};
 const records=await readFile(join(directory,'retrievals.jsonl'),'utf8').catch(e=>{if(e.code!=='ENOENT')throw e;return '';});
 const retrievals=records.split('\n').filter(Boolean).map(line=>JSON.parse(line));
 const proofSources=()=>proofRecords?[...proofRecords,...claimEvidence({prompt:{entries:prompt.entries,baseline:prompt.baseline,proposal:prompt.proposal,answers:prompt.answers,answerClaims:prompt.answerClaims},retrievals})]:claimEvidence({prompt,packet,retrievals});
 const catalogueFile=join(directory,'proof-selectors.json');let catalogueWrites=Promise.resolve();
 const refreshCatalogue=()=>catalogueWrites=catalogueWrites.then(async()=>{const catalogue=proofSelectorCatalogue([...proofSources(),...(discoveryProofRecords?await discoveryProofRecords():[])]),temporary=catalogueFile+'.'+randomUUID()+'.tmp';await writeFile(temporary,JSON.stringify(catalogue,null,2));await rename(temporary,catalogueFile);return catalogue;});
 if(proofRequired)prompt={...prompt,proofInstructions:(selectorOnly?selectorInstructions:proofInstructions)+' After KDB search/read_entry, read proofSelectorsFile for newly retrieved references and actual line/segment bounds. Submit selectors only; never copy source prose into proof metadata.',proofSelectorsFile:catalogueFile,proofSelectors:await refreshCatalogue(),proofEvidence:compactEvidence?undefined:proofSources()};
 await writeFile(join(directory,'input.json'),JSON.stringify(prompt,null,2));
 const coverageFile=join(directory,'read-evidence.json');
 const seen=new Set(JSON.parse(await readFile(coverageFile,'utf8').catch(e=>{if(e.code!=='ENOENT')throw e;return '[]';})));
 const schema=resultSchema??(proofRequired?proofResultSchema:Type.Unknown());
 const isClassificationBatch=schema===classificationBatchSchema||schema===failureFirstBatchSchema;
 const check=async(result,cached=false)=>{
  if(isClassificationBatch&&Array.isArray(result?.claims))
   result={...result,claims:result.claims.map(claim=>typeof claim?.path==='string'
    ? {...claim,path:claim.path.replace(/^classifications\[(\d+)\]\.reason$/,'classifications.$1.reason')}
    : claim)};

  let candidate=cached&&selectorOnly?selectorShape(result):result;
  if(cached&&schema===classificationSchema){const {groupId,intervention,...submitted}=candidate;candidate=submitted;}
  if(cached&&isClassificationBatch)candidate={...candidate,classifications:candidate.classifications?.map(({intervention,triage,...submitted})=>submitted)};
  if((proofRequired||resultSchema)&&!Check(schema,candidate)){const paths=Errors(schema,candidate).slice(0,4).map(e=>(e.instancePath||'/')+' '+e.message).join('; ');
   throw Error((selectorOnly?'Submit only the result schema: claims contain path, kind and selector-only proofs {reference,startLine?,endLine?,segment?,field?}; omit claim.text, quote, hashes, fact values and input metadata.':'Submission must match the submit_review result schema: claims is a required TOP-LEVEL array of {path,kind,proofs} (text may be omitted); nested claims and reasonClaims do not replace it.')+' Invalid fields: '+paths);
  }
  if(requiredReads.some(path=>!seen.has('file:'+resolve(path))))throw Error('Read these primary source files before summary audit: '+JSON.stringify(requiredReads.filter(path=>!seen.has('file:'+resolve(path)))));
  if(proofRequired)validateClaimedResult(result,[...(proofRecords?proofSources():claimEvidence({prompt,packet,retrievals})),...(discoveryProofRecords?await discoveryProofRecords():[])]);
  return validate(result,retrievals);
 };
 const contractFile=join(directory,'review-contract.json');
 const contract={version:REVIEW_CONTRACT_VERSION,schema,...(writableDirectory?{writableDirectory}:{})};
 let oldContract;try{oldContract=JSON.parse(await readFile(contractFile,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
 // A narrower submission schema can reuse the old transcript and cached
 // proof only through the current proof and admission checks.
 const retainedContract=JSON.stringify(oldContract);
 const compatible=retainedContract===JSON.stringify(contract)||
  !!previousResultSchema&&retainedContract===JSON.stringify({...contract,schema:previousResultSchema});
 if(!compatible){const archive=join(directory,'contract-history',randomUUID());await mkdir(archive,{recursive:true});
  for(const file of ['submission.json','native-session.json','review-contract.json'])try{await rename(join(directory,file),join(archive,file));}catch(error){if(error.code!=='ENOENT')throw error;}
 }
 await writeFile(contractFile,JSON.stringify(contract));
 const prior=compatible?await readFile(join(directory,'submission.json'),'utf8').catch(e=>{if(e.code!=='ENOENT')throw e;return null;}):null;
 let priorRejection;
 if(prior)try{const value=await check(JSON.parse(prior),true);if(JSON.stringify(value)!==JSON.stringify(JSON.parse(prior)))await writeFile(join(directory,'submission.json'),JSON.stringify(value,null,2));const retained=await readFile(join(directory,'native-session.json'),'utf8').then(JSON.parse).catch(error=>{if(error.code==='ENOENT')return null;throw error;});return {value,retrievals,recovered:true,...(retained?.path?{session:retained.path}:{})};}catch(error){priorRejection=error.message;await appendFile(join(directory,'submission-rejections.jsonl'),JSON.stringify({cached:true,error:error.message})+'\n');}
 const bridge=knowledge?createKdbBridge({url:await ensureKnowledgeService(config,{signal}),attemptId:'review-'+directory.split('/').at(-1),agentDir,
  recordRetrieval:async record=>{retrievals.push(record);await appendFile(join(directory,'retrievals.jsonl'),JSON.stringify(record)+'\n');if(proofRequired)await refreshCatalogue();}}):null;
 let submitted,rejectedSubmissions=0,lastRejection,submitting=false;
 const sanitize=value=>String(value??'').replace(/(https?:\/\/)[^/\s@]+@/gi,'$1[redacted]@').replace(/Bearer\s+[^\s"<>]+/gi,'Bearer [redacted]').replace(/([?&](?:token|key|api_key|access_token)=)[^&\s"<>]+/gi,'$1[redacted]').slice(0,2000);
 const rejectedCalls=new Set();let rejectionWrites=Promise.resolve();
 const recordRejection=(toolCallId,error,result)=>{if(rejectedCalls.has(toolCallId))return rejectionWrites;rejectedCalls.add(toolCallId);rejectedSubmissions++;lastRejection=sanitize(error);
  rejectionWrites=rejectionWrites.then(()=>appendFile(join(directory,'submission-rejections.jsonl'),JSON.stringify({toolCallId,error:lastRejection,...(result===undefined?{}:{result})})+'\n'));return rejectionWrites;
 };
 const extension=pi=>{
  pi.on('tool_call',async event=>{
   if(['write','edit'].includes(event.toolName)){const path=resolveToCwd(event.input.path??'',directory);if(await reviewWritePathAllowed(path,writableDirectory))return;await appendFile(join(directory,'tool-policy.jsonl'),JSON.stringify({tool:event.toolName,path,blocked:true,toolCallId:event.toolCallId})+'\n');return {block:true,reason:'Write and edit are limited to filesystemAccess.writableDirectory. Evidence, session state and accepted files are read-only.'};}
   if(!['read','grep','find','ls'].includes(event.toolName))return;
   const requested=resolveToCwd(event.input.path??'.',directory);
   const path=await realpath(requested).catch(error=>{if(error.code!=='ENOENT')throw error;return requested;});
   if(allowed.some(a=>path===a.path||a.directory&&path.startsWith(a.path+'/')))return;
   await appendFile(join(directory,'tool-policy.jsonl'),JSON.stringify({tool:event.toolName,path,blocked:true,toolCallId:event.toolCallId})+'\n');
   return {block:true,reason:'Filesystem inspection is limited to this review directory and explicitly copied evidence/verified learning paths. Read input.json and correct the submitted fields; use KDB MCP search/read_entry for indexed knowledge, not workspace searches.'};
  });
  pi.on('tool_result',async event=>{if(event.toolName!=='read'||event.isError)return;
   const path=resolveToCwd(event.input.path??'',directory);if(requiredReads.some(p=>resolve(p)===path)){seen.add('file:'+path);await writeFile(coverageFile,JSON.stringify([...seen]));}
  });
  pi.registerTool({name:'submit_review',label:'Submit review',description:'Submit structured, evidence-supported results. The controller runs the checks defined for this task on submission and returns feedback you can use to revise.',parameters:Type.Object({result:schema}),
  execute:async(id,{result})=>{
   if(writableDirectory&&(submitting||submitted!==undefined))throw Error('A draft submission is already being checked or has been accepted');
   submitting=true;
   try{
    const accepted=await check(result);
    await writeFile(join(directory,'submission.json'),JSON.stringify(accepted,null,2));
    submitted=accepted;
    return {content:[{type:'text',text:'Accepted. Finish this session.'}],details:{}};
   }catch(error){await recordRejection(id,error.message,result);throw error;}
   finally{submitting=false;}
  }});};
 const sessionFile=join(directory,'native-session.json');let retained;try{retained=compatible?JSON.parse(await readFile(sessionFile,'utf8')):undefined;}catch(e){if(e.code!=='ENOENT')throw e;}
 const settingsManager=sdk.SettingsManager.inMemory({compaction:{enabled:true},retry:{enabled:true,...retrySettings}}),manager=retained?sdk.SessionManager.open(retained.path,directory):sdk.SessionManager.create(directory,directory);
 await writeFile(sessionFile,JSON.stringify({path:manager.getSessionFile()}));
 const loader=new sdk.DefaultResourceLoader({cwd:directory,agentDir,settingsManager,noExtensions:true,noSkills:true,noContextFiles:true,
  extensionFactories:[extension,...(bridge?[bridge.extension]:[])]});await loader.reload();
 const {session}=await sdk.createAgentSession({cwd:directory,agentDir,settingsManager,...await skillModelFromProfile(config.skill_writer_actor??config.actor,{directory}),thinkingLevel:'off',
  tools:['read','grep','find','ls',...(writableDirectory?['write','edit']:[]),'submit_review',...(knowledge?['mcp__kdb__search','mcp__kdb__read_entry']:[])],resourceLoader:loader,sessionManager:manager});
 if(wrapStream)session.agent.streamFunction=wrapStream(session.agent.streamFunction);
 let turnStarted,responseStarted,retryStarted,latestProviderError,retryError;
 let providerErrors=0,nativeRetries=0;
 const recordWait=()=>{
  if(turnStarted!==undefined){onModelTiming('piTurnWait',(performance.now()-turnStarted)/1000);turnStarted=undefined;}
 };
 const unsubscribe=session.subscribe(event=>{
  if(event.type==='tool_execution_end'&&event.toolName==='submit_review'&&event.isError){
   const error=(event.result?.content??[]).filter(c=>c.type==='text').map(c=>c.text).join('\n');void recordRejection(event.toolCallId,error);
  }
  if(event.type==='message_end'&&event.message.role==='assistant'){
   latestProviderError=event.message.stopReason==='error'?event.message.errorMessage??'Unknown provider error':undefined;
   if(latestProviderError)providerErrors++;
   if(event.message.stopReason!=='aborted')retryError=undefined;
  }
  if(event.type==='auto_retry_start'){nativeRetries++;retryError=event.errorMessage;}
  if(!onModelTiming)return;
  if(event.type==='turn_start')turnStarted=performance.now();
  if(event.type==='message_start'&&event.message.role==='assistant'){
   recordWait();responseStarted=performance.now();
  }
  if(event.type==='message_end'&&event.message.role==='assistant'&&responseStarted!==undefined){
   onModelTiming('piResponse',(performance.now()-responseStarted)/1000);responseStarted=undefined;
  }
  if(event.type==='auto_retry_start')retryStarted=performance.now();
  if(event.type==='auto_retry_end'&&retryStarted!==undefined){
   onModelTiming('piRetry',(performance.now()-retryStarted)/1000);retryStarted=undefined;
  }
 });
 const initialMessages=session.messages.length;
 const termination=async(kind,error)=>{
  await rejectionWrites;
  const terminalError=latestProviderError??(kind==='elapsed'?retryError:undefined);
  const retryableTimeout=(kind==='provider-error'||kind==='elapsed')&&/\b(?:timed[\s-]+out|timeout|ETIMEDOUT)\b/i.test(terminalError??'');
  const detail={kind,rejectedSubmissions,providerErrors,nativeRetries,...(retryableTimeout?{retryableTimeout}:{}),
   ...(lastRejection?{lastRejection}:{}),...(terminalError||error?{cause:sanitize(terminalError??error?.message??error)}:{})};
  await writeFile(join(directory,'termination.json'),JSON.stringify(detail,null,2));return detail;
 };
 let expired=false;const abort=()=>{void session.abort();},timer=setTimeout(()=>{expired=true;abort();},seconds*1000);signal?.addEventListener('abort',abort,{once:true});
 try{
  signal?.throwIfAborted();if(bridge)await bridge.knowledge.prepareSession(session,{signal});else await session.bindExtensions({});
  await session.prompt((priorRejection?'Previous cached submission was rejected: '+priorRejection+'. Correct it within this session. ':'')+'Read input.json and follow its task. filesystemAccess.allowed lists the actual permitted files and directory trees, including any primary-source catalogue and its files outside this session directory; those declared paths may be read. Read diagnosticsFile, linkedDiagnosticsFile and fullOccurrences only when supplied; absent diagnostic files are not an instruction to search for them. Read explicitly required primary files before audit submission. Inspect relevant diagnostics using native read offset/limit when needed; diagnostic coverage is not a submission gate. Use only supplied evidence or enabled KDB tools, without searching implementation source or live workspaces. Submit the requested JSON result using submit_review, correcting its field-specific validation feedback rather than finishing with ordinary prose. Unknown is valid: never invent a cause, repair, source identity, device execution or measurement. Harness faults require user intervention, not actor repairs.');
  await session.waitForIdle();signal?.throwIfAborted();if(expired)throw new DOMException('Review session elapsed allocation exhausted','TimeoutError');
  if(continueFamilyDraft&&submitted===undefined){
   const responses=session.messages.slice(initialMessages).filter(m=>m.role==='assistant');
   if(responses.at(-1)?.stopReason==='stop'){
    await session.prompt('Continue from the draft and evidence in this session. Call submit_review with status ready and the completed family manifest. If required evidence is genuinely unavailable, call submit_review with status needs-evidence and identify what is missing. Address any tool validation feedback before finishing.');
    await session.waitForIdle();signal?.throwIfAborted();if(expired)throw new DOMException('Review session elapsed allocation exhausted','TimeoutError');
   }
  }
  if(submitted===undefined){const kind=latestProviderError?'provider-error':rejectedSubmissions?'rejected-submissions':'no-submit',detail=await termination(kind);
   throw Object.assign(Error('Review session '+kind+(detail.cause?': '+detail.cause:detail.lastRejection?': '+detail.lastRejection:'')),{reviewSession:detail});}
  await termination('submitted');
  return {value:submitted,retrievals,stats:session.getSessionStats(),session:manager.getSessionFile()};
 }catch(error){
  if(submitted!==undefined&&!signal?.aborted&&!expired){
   await termination('submitted');
   return {value:submitted,retrievals,stats:session.getSessionStats(),session:manager.getSessionFile()};
  }
  if(!error.reviewSession){const elapsed=expired||signal?.reason?.name==='TimeoutError',detail=await termination(elapsed?'elapsed':signal?.aborted?'cancelled':'provider-error',error);
   if(!signal?.aborted||elapsed){const wrapped=Error('Review session '+detail.kind+': '+detail.cause);wrapped.name=elapsed?'TimeoutError':error.name;wrapped.reviewSession=detail;throw wrapped;}
  }
  throw error;
 }finally{
  recordWait();clearTimeout(timer);signal?.removeEventListener('abort',abort);unsubscribe();await rejectionWrites;
  if(responseStarted!==undefined)onModelTiming('piResponse',(performance.now()-responseStarted)/1000);
  if(retryStarted!==undefined)onModelTiming('piRetry',(performance.now()-retryStarted)/1000);
  try{await session.exportToJsonl(join(directory,'session.native.jsonl'));}finally{await session.extensionRunner?.emit({type:'session_shutdown',reason:'quit'});await session.dispose();}
 }
}
export async function nativeClassify(input){
 return reviewSession({...classificationOptions(input),evidenceGroups:[input.group],prompt:{task:`Classify this failure group. ${input.bucketPolicy?'Bucket is orthogonal to fault ownership: harness needs user intervention, correctness concerns functional behavior, optimization concerns kernel latency.':''} Return {cause:actor|harness|specification|provider|unknown,${input.bucketPolicy?'bucket:harness|correctness|optimization'+(input.failureFirst?'|unknown':'')+',':''}reason,coverage:covered|uncovered|unknown,guidanceIds:[],evidenceRefs:[],gap:missing|insufficient|not-followed|unknown}. Cite occurrence references, use existing guidance IDs only. Availability does not establish the actor read a skill. A pre-actor stage or wrapper label does NOT prove a harness root cause: read the linked same-attempt negative verdicts and copied logs, distinguish invalid specifications/baseline APIs from harness faults, and retain unknown if the actual cause is not established. Inspect copied log files in occurrence.artifacts with native read when the message alone cannot establish a cause; do not access original live artifact paths.`,group:input.group,entries:input.entries}});
}
export async function nativeClassifyBatch(input){
 return reviewSession({...classificationOptions(input,true),evidenceGroups:input.groups,prompt:{task:`Classify EVERY failure group using the shared guidance. ${input.bucketPolicy?'Bucket is orthogonal to fault ownership: harness needs user intervention, correctness concerns functional behavior, optimization concerns kernel latency.':''} Return {classifications:[{groupId,cause:actor|harness|specification|provider|unknown,${input.bucketPolicy?'bucket:harness|correctness|optimization'+(input.failureFirst?'|unknown':'')+',':''}reason,coverage:covered|uncovered|unknown,guidanceIds:[],evidenceRefs:[],gap:missing|insufficient|not-followed|unknown}]}. Include each supplied groupId exactly once; keep each reason at most two short sentences. Cite actual occurrence/linked-diagnostic references. covered requires existing guidance IDs; unknown/uncovered requires guidanceIds=[]. Read actual copied diagnostics before attributing ownership; a pre-actor stage alone is not proof of a harness fault. Preserve unknown when evidence is insufficient. ${input.failureFirst?'Report category separately from ownership. Use bucket=unknown when only a failing wrapper or verifier is established. A device disagreement alone cannot establish where an unexpected tensor first appeared. Cite the observed runtime boundary from runtimeBoundaries for harness or invalid baseline; keep cause=unknown otherwise. A candidate verdict establishes a failed check, while attribution needs separate evidence. Evidence may support correctness with unknown ownership. Include a proof selector for each reason. Do not design or run controls in classification.':''}`,groups:input.groups,entries:input.entries,...(input.failureFirst?{runtimeBoundaries:Object.fromEntries(input.groups.map(group=>[group.id,failureTriageBoundaries(group)]))}:{})}});
}
export async function nativeReview(input,dependencies={}){
 const sources=await discoveryProofSources(input.groups);
 const prompt={guidanceScopes:['global-kdb',...workflowStages],task:'Review this guidance bucket using KDB search and read_entry. Return {action:update|learn|no-change,reason,groupIds:[],memoryRefs:["collection:entry_id"],changes:[],question,hypothesis,requiredEvidence,learningTarget:cpu|device,learningRequests:[{groupIds:[originating failure group IDs],question,hypothesis,requiredEvidence,learningTarget:cpu|device}]}. Include outstanding learning questions even when proposing a supported update. Each learning request must identify its own originating groupIds, a nonempty subset of this bucket; do not attach unrelated harness faults to an actor/specification question. An update requires retrieved supporting memory, not a failure alone. For KDB search results, chunk_id is the entry_id argument to read_entry; do not substitute a lesson hash, Git hash or filename. memoryRefs uses exact collection:chunk_id from actual retrieval. evidenceRefs uses exact failure occurrence/linked-diagnostic IDs, not KDB IDs. Use only the enumerated guidanceScopes for changes.scope; correctness/optimization/harness are evaluation buckets, not guidance scopes. changes are {action:upsert,id,scope,text,critical:boolean,negative:boolean,evidenceRefs:[occurrence references],response:actor-change|report-blocker|reserve-validation-runway|retrieve-scoped-evidence}. report-blocker and reserve-validation-runway require scope=global-kdb or scope=preflight; fix that scope field rather than reordering changes. actor-change is the valid response label for actor-owned guidance. Prefer one concise new entry supplementing protected baseline. No device performance claims from CPU results or unsupported duration/speed claims. Cumulative actor clocks are not operation timings. Keep references and guidance compact. Detailed kernel cases remain searchable; guidance should explain applicability and retrieval. Questions requiring GPU evidence must stay pending if CPU cannot answer. Harness faults may only produce report/preserve guidance. Classify each learning request with workKind:microlearning when a single tiny controlled experiment or archived-fact inspection can finish inside 900 elapsed seconds including construction, verification and cleanup; otherwise workKind:phase1. This only routes future work; do not run benchmarks in this review.',groups:input.groups,entries:input.entries,classification:input.classifications,learning:input.learning??[],learningEvidence:input.learningEvidence??[]};
 const allowedPaths=(input.learning??[]).flatMap(([,record])=>['verified-cpu','verified-device','verified-project'].includes(record.result?.status)?[record.result.project?.source,...(record.result.project?.verifier?.files??[]),...(record.result.evidence??[]).flatMap(e=>e.log?[e.log]:[])].filter(Boolean):[]);
 return (dependencies.session??reviewSession)({...input,proofRequired:true,selectorOnly:true,resultSchema:proposalSchema,allowedPaths,evidenceGroups:input.groups,knowledge:true,discoveryProofRecords:async()=>(await discoveryProofSources(input.groups)).records,prompt:{...prompt,discoverySources:sources.indexes},validate:(result,retrievals)=>{
  const refs=new Set(retrievals.flatMap(r=>r.entries.map(e=>`${e.collection}:${e.entry_id??e.chunk_id??e.id}`)));
  if(result.action==='update'&&result.memoryRefs?.some(ref=>!refs.has(ref)))throw Error('Memory references must have been retrieved in this session');
  return input.validate(result);
 }});
}

const summaryResultSchema=Type.Object({items:Type.Array(Type.Object({text:nonempty(),kind:Type.Union([Type.Literal('observation'),Type.Literal('hypothesis')])},{additionalProperties:false}),{minItems:1,maxItems:3}),claims:selectorClaims(Type.String({pattern:'^items\\.[0-2]\\.text$'}))},{additionalProperties:false});
const summaryAuditSchema=Type.Object({verdict:Type.Union(['supported','revise','unknown'].map(x=>Type.Literal(x))),reason:nonempty(),claims:selectorClaims(Type.Literal('reason'))},{additionalProperties:false});
export async function nativeQaSummarize(input,dependencies={}){
 const {sourceCatalogue:catalogue,page}=input;
 return (dependencies.session??reviewSession)({...input,proofRequired:true,selectorOnly:true,resultSchema:summaryResultSchema,knowledge:false,evidenceGroups:[],compactEvidence:true,proofRecords:catalogue.records,allowedPaths:[catalogue.directory],prompt:{sources:page.sources,authoritativeFacts:catalogue.facts,previousReview:input.previousReview??[],task:'Return {items:[{text,kind}],claims:[{path:\'items.0.text\',kind,proofs:[{reference,startLine?,endLine?,segment?,field?}]}]}. Use at most three concise material items; omit claims.text, quotes, hashes and metadata. Do not repeat authoritative typed facts already supplied. Read the primary source files in this page and summarize the consequential observed evidence into concise items, each with observation|hypothesis kind and a top-level claim path items.N.text with primary proof selectors. Code/docstrings and memory reports prove text, not measured execution or backend defects. Keep project status separate from passing verifier counts; preserve uncertainty, source/case/runtime boundaries and conflicting results. Do not draft skills or infer skill benefit. Select source lines, whole typed facts or input JSON fields; do not retype quotes. A summary is not a new primary source. Fix previousReview findings using only these primary sources.'},validate:value=>input.validate(value)});
}
export async function nativeQaAuditSummary(input,dependencies={}){
 const {sourceCatalogue:catalogue,item}=input,refs=item.proofs.map(p=>p.reference),sources=sourcePointers(catalogue,refs),reads=sources.filter(s=>!s.factId).flatMap(s=>s.segments.map(p=>p.path));
 return (dependencies.session??reviewSession)({...input,proofRequired:true,selectorOnly:true,resultSchema:summaryAuditSchema,knowledge:false,evidenceGroups:[],compactEvidence:true,proofRecords:catalogue.records,allowedPaths:[catalogue.directory],requiredReads:reads,prompt:{item:{text:item.text,kind:item.kind,proofs:item.proofs},sources,relatedSources:catalogue.pages.find(p=>p.id===item.pageId)?.sources??[],sourceCatalogue:catalogue.indexFile,authoritativeFacts:catalogue.facts,task:'Independently verify this summary item against the primary files and authoritative facts. Read its primary source files before deciding. Return only {verdict:\'supported|revise|unknown\',reason,claims:[{path:\'reason\',kind:\'observation|hypothesis\',proofs:[{reference,startLine?,endLine?,segment?,field?}]}]}. Claims support your reason, never item.text or copied input claims. Omit claims.text, quotes, hashes, fact values and input metadata. Check entailment against the union of all cited primary excerpts and supplied typed facts, not merely literal copying. Every material clause needs support somewhere in that combined evidence; each individual citation need not prove the whole paragraph. Before reporting a missing citation, check whether another supplied proof already supports that clause. Mathematical deductions with explicit source-supported premises are not device measurements and do not require new NPU execution when their claim is scoped to CPU or generic mathematics. A code comment or memory report does not establish device execution, numerical correctness, a backend defect or causal performance. Whole-kernel ratios do not prove component contribution. Check project/pass counts and omitted contradictions. supported means the correctly scoped statement follows, or is explicitly an unverified hypothesis; revise means a fixable mismatch, unknown means necessary primary evidence is absent. The summary itself cannot be its own proof.'},validate:value=>{if(!['supported','revise','unknown'].includes(value.verdict))throw Error('Invalid summary audit verdict');return input.validate(value);}});
}

// QA deliberately reuses retrieved evidence rather than reopening KDB or executing kernels.
async function qaPrompt(input){
 let memory=input.memoryEvidence;
 if(!memory){
  const records=await readFile(join(input.proposalDirectory??input.directory,'retrievals.jsonl'),'utf8').catch(error=>{if(error.code!=='ENOENT')throw error;return '';});
  memory=records.split('\n').filter(Boolean).flatMap(line=>JSON.parse(line).entries??[]);
 }
 const cited=new Set(input.proposal.memoryRefs??[]);
 memory=memory.filter(item=>cited.has(item.ref??item.citationId??`${item.collection}:${item.entry_id??item.chunk_id??item.id}`));
 const groups=scopedQaGroups(input.groups,input.proposal);
 return {projectSummaries:input.projectSummaries,learningProofEvidence:input.learningProofEvidence,learningEvidence:input.learningEvidence??[],proposal:input.proposal,baseline:scopedBaseline(input.entries,input.proposal),objective:input.objective,target:input.target,groups,memory,
  learning:input.learning??[],evaluationContract:input.evaluationContract??null,questions:{action:'Is a concrete actor edit justified? If so, what is the smallest edit and what must remain unchanged? Otherwise state no justified edit and the missing fact.',mechanism:'What supported mechanism links the edit to the failure? If unknown, identify the unresolved question rather than inventing a cause.',example:'Give a source-supported worked example and applicability boundary; label mathematical deductions and unknowns.',evidence:'Separate observed execution, mathematical deduction, retained code, memory reports and untested claims; establish checkpoint and target applicability.',objective:'Is this correctness or optimization? Is the guidance already satisfied, documentation-only, or demanding unnecessary optimization?'},
  rules:'Only supplied evidence is available. Do not run benchmarks or search live sources. A retained source file is not necessarily the executed failing or passing checkpoint: require an explicit artifact identity link before attributing its behavior to a run. Memory reports establish what was reported, not independent execution. Mathematical deductions are allowed but must name their source-supported premises and remain distinct from measured facts; generic mathematical or explicitly CPU-scoped conclusions do not require an NPU run. Assess the union of all cited evidence: each material clause must be supported somewhere, not independently by every citation. No source edit is required when evidence is insufficient or the retained code already satisfies the proposed rule; identify the missing fact instead. Documentation-only corrections are not useful optimization approvals. Explicitly untested optimization hypotheses need no prior speedup measurement. The actor sees the skill text, not these answers.'};
}
export async function nativeQaAnswer(input,dependencies={}){
 let prompt=await qaPrompt(input);
 const compact=input.evidenceSummary&&input.sourceCatalogue;if(compact)prompt={proposal:prompt.proposal,baseline:prompt.baseline,objective:prompt.objective,target:prompt.target,questions:prompt.questions,rules:prompt.rules,evidenceSummary:compactSummary(input.evidenceSummary,input.sourceCatalogue),evaluationContract:prompt.evaluationContract};
 const failureFirst=input.evaluationContract?.workflow==='failure-first-v1',selectorOnly=!!compact||failureFirst;
 const correction=input.answerCorrection,resultSchema=correction
  ?Type.Object({answers:qaAnswerResultSchema.properties.answers,claims:selectorClaims()},{additionalProperties:false})
  :selectorOnly?Type.Object({...qaAnswerResultSchema.properties,claims:selectorClaims()},{additionalProperties:false}):qaAnswerResultSchema;
 const allowedPaths=(input.learning??[]).flatMap(([,record])=>['verified-cpu','verified-device','verified-project'].includes(record.result?.status)?[record.result.project?.source,...(record.result.project?.verifier?.files??[]),...(record.result.evidence??[]).flatMap(e=>e.log?[e.log]:[])].filter(Boolean):[]);
 return (dependencies.session??reviewSession)({...input,proofRequired:true,selectorOnly:selectorOnly||!!correction,resultSchema,previousResultSchema:failureFirst&&!compact&&!correction?qaAnswerResultSchema:undefined,knowledge:false,evidenceGroups:compact?[]:prompt.groups,compactEvidence:!!compact,proofRecords:compact?input.sourceCatalogue.records:undefined,allowedPaths:compact?[...allowedPaths,input.sourceCatalogue.directory]:allowedPaths,prompt:{...prompt,previousReview:input.previousReview,...(correction?{answerCorrection:correction}:{}),
  task:correction?'Correct the five prior answers using the reviewer findings and supplied evidence. Return only answers and proof-bearing claims for those answers. Keep supported answers and their source references, and address each blocking finding. The proposal is fixed.':'Answer all five questions concisely, at most two short sentences per answer. Return {answers:{action:{text,refs:[]},mechanism:{text,refs:[]},example:{text,refs:[]},evidence:{text,refs:[]},objective:{text,refs:[]}},changes?:[],memoryRefs?:[],reason?:string}. refs must identify supplied failure references, baseline IDs or already cited memory. When an edit is justified, revise changes and the optional proposal reason so essential actionable advice and boundaries appear in the actual skill text. Address the complete previousReview issue list together, retaining corrected facts instead of introducing replacements without evidence. When no edit is justified, say so in the five answers, leave changes omitted, and identify the missing evidence so independent review can choose learn or inconclusive; do not manufacture an instruction to fill the schema. Preserve the existing change schema. Do not merely explain a missing instruction in private answers.'}});
}
const panelAuditSchema=Type.Object({
 audits:Type.Array(Type.Object({claimId:nonempty(),disposition:choices(['verified','disproved','unresolved']),reason:nonempty()},{additionalProperties:false}),{minItems:1}),
 claims:selectorClaims(Type.String({pattern:'^audits[.][0-9]+[.]reason$'})),
},{additionalProperties:false});
export async function nativeQaPanelAudit(input,dependencies={}){
 const prompt=await qaPrompt(input),extra=input.panelAuditEvidence??[],compact=input.evidenceSummary&&input.sourceCatalogue;
 const objections=input.objections.map(({id,kind,path,problem,mentions})=>{
  const cited=new Set(mentions.flatMap(mention=>mention.proofs).map(proof=>JSON.stringify(Object.fromEntries(
   Object.keys(selectorFields).filter(key=>Object.hasOwn(proof,key)).map(key=>[key,proof[key]])))));
  return {id,kind,path,problem,citedSelectors:[...cited].map(value=>JSON.parse(value))};
 });
 const allowed=(input.learning??[]).flatMap(([,record])=>['verified-cpu','verified-device','verified-project'].includes(record.result?.status)?[record.result.project?.source,...(record.result.project?.verifier?.files??[]),...(record.result.evidence??[]).flatMap(e=>e.log?[e.log]:[])].filter(Boolean):[]);
 return (dependencies.session??reviewSession)({...input,resultSchema:panelAuditSchema,selectorOnly:true,proofRequired:true,knowledge:false,
  compactEvidence:!!compact,evidenceGroups:compact?[]:prompt.groups,proofRecords:compact?input.sourceCatalogue.records:undefined,
  discoveryProofRecords:extra.length?async()=>extra:undefined,allowedPaths:compact?[...allowed,input.sourceCatalogue.directory]:allowed,
  prompt:{...prompt,...(compact?{evidenceSummary:compactSummary(input.evidenceSummary,input.sourceCatalogue)}:{}),answers:input.answers,answerClaims:input.answerClaims,
   objections,...(extra.length?{primaryAuditEvidence:extra}:{}),
   task:'Check each objection against frozen primary evidence. Submit audits with the exact claimId, disposition (verified, disproved or unresolved), and a concise reason. Verify whether each objection is materially correct for the exact statement it challenges. Identify the source, attempt and execution role when that affects the claim. Distinguish final scores, intermediate edits, retained source, and static probes. An objection and the frequency of its repetition are not evidence. Read the cited primary excerpts in context. If an essential identity or measurement is missing, choose unresolved. Supply proof selectors for each audits.N.reason. Resolve wording-only questions from the actual draft or answer.'},
  validate:input.validate});
}
export async function nativeQaReview(input,dependencies={}){
 let prompt=await qaPrompt(input);
 const compact=input.evidenceSummary&&input.sourceCatalogue;if(compact)prompt={proposal:prompt.proposal,baseline:prompt.baseline,objective:prompt.objective,target:prompt.target,questions:prompt.questions,rules:prompt.rules,evidenceSummary:compactSummary(input.evidenceSummary,input.sourceCatalogue),evaluationContract:prompt.evaluationContract};
 const failureFirst=input.evaluationContract?.workflow==='failure-first-v1',selectorOnly=!!compact||failureFirst;
 const typed=['split','panel'].includes(input.qaPolicy);
 const allowedPaths=(input.learning??[]).flatMap(([,record])=>['verified-cpu','verified-device','verified-project'].includes(record.result?.status)?[record.result.project?.source,...(record.result.project?.verifier?.files??[]),...(record.result.evidence??[]).flatMap(e=>e.log?[e.log]:[])].filter(Boolean):[]);
 return (dependencies.session??reviewSession)({...input,validate:value=>input.validate(validateSummaryTargets(value,input.evidenceSummary?.items??[])),proofRequired:true,selectorOnly:selectorOnly||typed,resultSchema:selectorOnly||typed?Type.Object({...qaReviewResultSchema.properties,issues:Type.Optional(Type.Array(typed?typedQaIssueSchema:selectorIssueSchema)),claims:selectorClaims()},{additionalProperties:false}):qaReviewResultSchema,previousResultSchema:failureFirst&&!compact&&!typed?qaReviewResultSchema:undefined,knowledge:false,evidenceGroups:compact?[]:prompt.groups,compactEvidence:!!compact,proofRecords:compact?input.sourceCatalogue.records:undefined,allowedPaths:compact?[...allowedPaths,input.sourceCatalogue.directory]:allowedPaths,prompt:{...prompt,previousReview:input.previousReview,answers:input.answers,answerClaims:compact?selectorShape({claims:input.answerClaims??[]}).claims:input.answerClaims??[],
  task:'Independently review the draft, five answers and answerClaims against supplied evidence. Check the independently audited evidenceSummary against its primary source pointers as well as every final skill/answer claim; a summary is never its own primary proof. Check each claim against the union of its cited primary excerpts and supplied typed facts; an exact quote alone is not proof of entailment. Every material clause must have support somewhere in that combined evidence, but an individual citation need not prove the whole paragraph. Before calling a clause unsupported or a citation missing, check the other supplied proofs for that clause. Accept mathematical deductions with named, supported premises as deductions rather than demanding new device measurements for a generic mathematical or explicitly CPU-scoped conclusion. Report all material mismatches together in one revise verdict, with a concrete issue for each. When a material defect is in an evidenceSummary item, set summaryItemId on that issue to the exact item ID from evidenceSummary so the summary stage can repair it; do not ask the skill writer to rewrite a summary through a prose-only issue. For defects in draft or answer prose, use ordinary path/field metadata instead. Compare against the complete previousReview list: retain unresolved issues, recognize corrected ones, and explain new issues by changed text or previously missed primary evidence. Do not reserve known issues for later rounds or demand stylistic rewrites of already correct guidance. Return {verdict:approved|revise|learn|inconclusive,reason,issues?:[],learningRequests?:[{groupIds:[],question,hypothesis,requiredEvidence,learningTarget:cpu|device}]}. Approve only if a justified actor change is communicated in the skill text with its supported mechanism and material boundaries; private answers cannot repair absent guidance. If no justified edit or established mechanism exists, use learn for a concrete missing-evidence question or inconclusive when the review cannot establish one; do not require a speculative repair. A rule already satisfied by the retained source is not itself a new fix, and documentation-only corrections are not useful optimization approvals. Gate material contradictions, unsupported target claims and unnecessary optimization in correctness guidance. Do not demand prior performance proof of an explicitly untested optimization hypothesis. Use revise for fixable writing issues, learn for missing facts requiring CPU/device learning, inconclusive for insufficient review evidence. For each learning request set workKind:microlearning only for one tiny controlled question fitting 900 elapsed seconds including construction, verification and cleanup; larger workKind:phase1. Harness problems remain user handled. Cite concrete issues; approval admits A/B only.'+(typed?' Give each finding a kind: answer for a defect in an answer (path must name answers.<topic>.text), draft for a material defect in the instruction, evidence for a missing primary fact needed to decide, or advisory for an optional improvement. Keep factual or correctness defects material. Approve only if any findings are advisory.':'')}});
}
export async function nativeLearn({directory,config,request,signal,seconds},dependencies={}){
 const ops={construct:refillProjects,importProject,operate,ensureKnowledgeService,createKdbBridge,...dependencies};
 const state=new State(join(directory,'state.sqlite')),begun=performance.now();
 try{
  const existing=state.db.prepare('SELECT id FROM projects ORDER BY rowid LIMIT 2').all();
  if(existing.length>1)throw Error('Learning instance must contain only its original project');
  let cfg;try{cfg=JSON.parse(await readFile(join(directory,'instance.json'),'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
  cfg??={...config,root:directory,target:'cpu',mode:'phase1',actor_seconds:Math.min(300,seconds),curriculum_seconds:Math.min(600,seconds),
   environment:{module:fileURLToPath(new URL('../integrations/environments/index.mjs',import.meta.url)),options:{mode:'local'}},campaign_plan:[{id:'learning',phase:'phase1',new_projects:1}],phase_targets:{phase1:1},curriculum:{generation:{enabled:true,topics:[request.question],preprompt:'Investigate only this CPU-answerable learning question: '+JSON.stringify(request)+'. Keep fixtures tiny, deterministic and independently checked. Record actual observations, not hypothetical device behavior.'}},skills:{...config.skills,review:{enabled:false}}};
  cfg=skillConfig(cfg);
  let spec=existing.length?state.project(existing[0].id).spec:null;
  if(!spec){
   await cp(join(config.root,'skills'),join(directory,'skills'),{recursive:true});
   await symlink(join(config.root,'knowledge.sqlite'),join(directory,'knowledge.sqlite')).catch(e=>{if(e.code!=='EEXIST')throw e;});
   const bridge=ops.createKdbBridge({url:await ops.ensureKnowledgeService(config,{signal}),attemptId:request.id,agentDir:join(directory,'agent')});
   const drafts=await ops.construct({config:cfg,state,bridge,concurrency:1,path:join(directory,'drafts'),signal});
   if(!drafts[0]){
    const rows=state.db.prepare('SELECT data FROM generation_drafts').all().map(({data})=>{const d=JSON.parse(data);return {id:d.id,status:d.status,elapsed:d.elapsed,seconds:d.seconds,error:d.error??null};});
    const observations=state.db.prepare("SELECT sequence,draft,type,data FROM generation_events WHERE type IN ('generation-rejected','generation-construction-check','curriculum-interrupted') ORDER BY sequence").all().map(e=>({...e,data:JSON.parse(e.data)}));
    return {status:'inconclusive',reason:'curriculum-no-accepted-project',elapsedSeconds:(performance.now()-begun)/1000,drafts:rows,evidence:observations,claim:'No accepted learning project; constructor observations establish no successful repair'};
   }
   spec=await ops.importProject(directory,{...drafts[0],actor_seconds:Math.max(.001,Math.min(300,seconds-(performance.now()-begun)/1000))},{signal});
   state.add(spec);
  }
  cfg.actor_seconds=spec.actor_seconds;cfg.projects=[spec];cfg.curriculum={phase1:[spec.id],generation:{enabled:false}};
  await writeFile(join(directory,'instance.json'),JSON.stringify(cfg,null,2));
  const attempts=()=>state.db.prepare(`SELECT a.id,a.outcome,a.elapsed,a.session,
   EXISTS(SELECT 1 FROM events e WHERE e.attempt=a.id AND e.type='postprocess-completed') AS postprocessed FROM attempts a WHERE project=?`).all(spec.id);
  const previous=attempts();
  if(!previous.length||previous.some(a=>a.outcome===null||!a.postprocessed))await ops.operate(previous.length?'resume':'wave',cfg,{signal,'no-recovery':true});
  signal?.throwIfAborted();const completed=attempts(),passed=completed.some(a=>a.outcome==='passed'&&a.postprocessed),unfinished=completed.some(a=>!a.postprocessed);
  return {status:passed?'verified-cpu':unfinished?'postprocessing-pending':'inconclusive',project:spec,evidence:spec.generation?.checks??[],attempts:completed,claim:'Independent CPU verification and actor project evidence only; no GPU claim'};
 }finally{state.close();}
}
export async function nativeEvaluate({directory,active,candidate,config,signal,seconds,objective},dependencies={}){
 config=skillConfig(config);
 const ops={runMicroeval,finalizeKernelPairs,...dependencies},settings=objective&&objective!=='legacy'?evaluationSettings(config.skill_review_evaluations,objective):null;
 const selected=settings?.tasks??config.skill_review_tasks??tasks;
 let frozen;try{frozen=JSON.parse(await readFile(join(directory,'manifest.json'),'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
 if(settings&&frozen&&frozen.totalSeconds!==settings.developmentSeconds)throw Error('Frozen development allocation changed');
 let report=await ops.runMicroeval({directory,objective,binding:settings?.binding,knowledge:settings?.knowledge,skills:{active,candidate},tasks:selected,profile:config.actor,python:config.python??'python3',runs:2,seconds:settings?180:90,verificationSeconds:settings?.verificationSeconds??0,
  totalSeconds:frozen?.totalSeconds??settings?.developmentSeconds??seconds,signal,resume:true,shadow:true});
 if(objective==='optimization'&&report.status==='complete')report=await ops.finalizeKernelPairs({directory,python:config.python??'python3',seconds:settings.pairedSeconds,signal});
 return {decision:evaluateSkillObjective(report.results,{objective,projectIds:selected.map(t=>t.id),runs:2,margin:.05,pairedTimings:report.pairedTimings}),report};
}

export async function stopReviewKnowledge(root){
 let receipt;try{receipt=JSON.parse(await readFile(join(root,'knowledge-service.json'),'utf8'));}catch(e){if(e.code==='ENOENT')return;throw e;}
 if(await ownerStopped(receipt.owner))return;process.kill(receipt.owner.pid,'SIGTERM');
 const end=performance.now()+5000;while(!await ownerStopped(receipt.owner)&&performance.now()<end)await delay(50);
 if(!await ownerStopped(receipt.owner))throw Error('Owned review KDB service remains live: '+root);
}

export async function evaluationSettled({directory}){
 let report;try{report=JSON.parse(await readFile(join(directory,'results.json'),'utf8'));}catch(e){if(e.code==='ENOENT')return true;throw e;}
 return !report.runningSample&&!report.pairedReplay?.runningPair&&(!report.owner||await ownerStopped(report.owner));
}
