/** Bounded, independently reviewed admission; approval is not promotion. */
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile,rename} from 'node:fs/promises';
import {join} from 'node:path';
const digest=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
export const qaTopics=['action','mechanism','example','evidence','objective'];
export function validateQaAnswers(value,refs){
 if(!value?.answers)throw Error('QA requires five structured answers');
 for(const topic of qaTopics){const answer=value.answers[topic];
  if(!answer?.text?.trim()||!Array.isArray(answer.refs)||answer.refs.some(ref=>!refs.has(ref)))throw Error('QA answer requires text and actual evidence references: '+topic);
 }
 return value;
}
export function validateQaVerdict(value){
 if(!['approved','revise','learn','inconclusive'].includes(value?.verdict)||!value.reason?.trim())throw Error('QA requires a verdict and reason');
 if(value.verdict==='revise'&&(!Array.isArray(value.issues)||!value.issues.length))throw Error('Revision requires material issues');
 if(value.verdict==='learn'&&(!Array.isArray(value.learningRequests)||!value.learningRequests.length))throw Error('Learning verdict requires learning requests');
 return value;
}
export function qaIdentity(input,proposal=input.proposal){
 return digest({changes:proposal.changes,memoryRefs:proposal.memoryRefs,baseline:input.entries,objective:input.objective??null,target:input.target,evidence:input.groups,memoryEvidence:input.memoryEvidence??null,learning:input.learning??[],learningEvidence:input.learningEvidence??[],retrievalEvidenceHash:input.retrievalEvidenceHash??null,evaluationContract:input.evaluationContract??null});
}
export async function runProposalQa(input,ops){
 await mkdir(input.directory,{recursive:true});let file=join(input.directory,'qa.json');
 let state;try{state=JSON.parse(await readFile(file,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
 const identity=qaIdentity(input),now=ops.now??Date.now;
 if(state&&state.format!==2){const legacy=state;file=join(input.directory,'qa-v2.json');try{state=JSON.parse(await readFile(file,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;state={format:2,protocol:'skill-qa-v2',inputIdentity:identity,startedAt:now(),seconds:300,elapsedSeconds:legacy.elapsedSeconds??0,draft:input.proposal,rounds:[],resumeFrom:join(input.directory,'qa.json')};}}
 if(!state&&input.resumeFrom){const legacy=JSON.parse(await readFile(input.resumeFrom,'utf8'));state={format:2,protocol:'skill-qa-v2',inputIdentity:identity,startedAt:now(),seconds:300,elapsedSeconds:Math.max(input.priorElapsedSeconds??0,legacy.elapsedSeconds??0,legacy.inFlight?.reservedThrough??0),draft:input.proposal,rounds:[],resumeFrom:input.resumeFrom};}
 if(state&&state.inputIdentity!==identity)throw Error('QA input changed; use a new proposal QA directory');
 state??={format:2,protocol:'skill-qa-v2',inputIdentity:identity,startedAt:now(),seconds:300,elapsedSeconds:input.priorElapsedSeconds??0,draft:input.proposal,rounds:[]};
 // Reserve a bounded slice before provider work. A hard loss charges that slice,
 // never paused wall time; graceful completion releases the unused reservation.
 const begun=now(),charged=Math.max(state.elapsedSeconds??0,state.inFlight?.reservedThrough??0);
 delete state.inFlight;
 const elapsed=()=>charged+Math.max(0,(now()-begun)/1000);
 let writes=Promise.resolve();
 const persist=()=>{state.elapsedSeconds=elapsed();const snapshot=JSON.stringify(state,null,2)+'\n';writes=writes.then(async()=>{const tmp=file+'.tmp';await writeFile(tmp,snapshot);await rename(tmp,file);});return writes;};
 const remaining=()=>Math.max(0,state.seconds-elapsed());
 if(state.verdict&&!state.retryable)return state;
 if(state.retryable){delete state.verdict;delete state.reason;delete state.error;delete state.reasonKind;delete state.retryable;}
 let proposal=state.rounds.at(-1)?.proposal??state.draft;
 try{
  while(remaining()>0){
   if(state.rounds.length>=2&&state.rounds.at(-1)?.review?.verdict==='revise'){state.verdict='inconclusive';state.reason='qa-revision-limit';state.reasonKind='scientific';break;}
   const round=state.rounds.at(-1)?.review?.verdict==='revise'||!state.rounds.length?{proposal}:state.rounds.at(-1);
   if(round!==state.rounds.at(-1)){state.rounds.push(round);await persist();}
   const index=state.rounds.length-1;
   const call=async(name,payload)=>{
    if(remaining()<=0)throw Error('QA elapsed allocation exhausted');
    input.signal?.throwIfAborted();const seconds=Math.min(remaining(),input.seconds??300);
    const signal=AbortSignal.any([...(input.signal?[input.signal]:[]),AbortSignal.timeout(Math.max(1,Math.ceil(seconds*1000)))]);
    const directory=join(input.directory,'sessions',state.protocol,identity,`${index}-${name}`);await mkdir(directory,{recursive:true});
    const reserve=()=>{state.inFlight={stage:name,reservedThrough:Math.min(state.seconds,elapsed()+5)};return persist();};
    await reserve();let checkpointError;
    const heartbeat=setInterval(()=>{void reserve().catch(error=>{checkpointError=error;});},1000);
    try{const result=await ops[name]({...input,...payload,directory,seconds,signal});if(checkpointError)throw checkpointError;return result;}
    finally{clearInterval(heartbeat);await writes;delete state.inFlight;await persist();}
   };
   if(!round.answers){
    const refs=new Set([...input.groups.flatMap(g=>[...g.occurrences,...(g.linkedDiagnostics??[])].map(o=>o.reference)),...input.entries.map(e=>e.id),...(proposal.memoryRefs??[]),...(input.learning??[]).flatMap(([,record])=>(record.result?.evidence??[]).flatMap(e=>e.reference?[e.reference]:[]))]);
    const validate=value=>{
     validateQaAnswers(value,refs);
     if(value.memoryRefs?.some(ref=>!(input.proposal.memoryRefs??[]).includes(ref)))throw Error('QA revisions cannot introduce unretrieved memory references');
     if(value.changes){const revised=input.validateProposal({...proposal,changes:value.changes,...(value.memoryRefs?{memoryRefs:value.memoryRefs}:{})});return {...value,changes:revised.changes};}
     return value;
    };
    const result=validate(await call('qaAnswer',{proposal,previousReview:state.rounds[index-1]?.review,validate}));
    if(result.changes)proposal=input.validateProposal({...proposal,changes:result.changes,...(result.memoryRefs?{memoryRefs:result.memoryRefs}:{})});
    round.proposal=proposal;round.answers=result.answers;await persist();
   }
   if(!round.review){const validate=value=>{validateQaVerdict(value);if(value.learningRequests?.length)input.validateProposal({...round.proposal,learningRequests:value.learningRequests});return value;};
    round.review=validate(await call('qaReview',{proposal:round.proposal,answers:round.answers,validate}));await persist();}
   if(remaining()<=0)throw Error('QA elapsed allocation exhausted');
   if(round.review.verdict==='revise'){proposal=round.proposal;continue;}
   state.verdict=round.review.verdict;state.reason=round.review.reason;
   state.proposal=round.proposal;state.learningRequests=round.review.learningRequests??[];
   if(state.verdict==='approved')state.approval=qaIdentity(input,round.proposal);
   break;
  }
  if(!state.verdict){state.verdict='inconclusive';state.reason='qa-elapsed';}
 }catch(error){
  if(input.signal?.aborted){await persist();throw error;}
  state.verdict='inconclusive';state.reason=remaining()<=0||error.name==='TimeoutError'?'qa-elapsed':'qa-execution-failed';state.reasonKind='infrastructure';state.retryable=remaining()>0;state.error={name:error.name,message:error.message};
 }
 state.reasonKind??=state.reason==='qa-elapsed'?'infrastructure':'scientific';state.retryable??=false;await persist();return state;
}
