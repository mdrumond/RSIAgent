const infrastructureExits=new Set([124,125,126,127]);
const executed=exit=>Number.isInteger(exit)&&exit>=0&&exit<128&&!infrastructureExits.has(exit);

function observedBoundary(row,type){
 const data=row.data??{};
 if(['device-lease-blocked','execution-unresolved','preactor-infrastructure'].includes(type))
  return {kind:'infrastructure',reference:row.reference};
 if(['rejected','attempt-error'].includes(type)&&data.actorStatus?.kind==='provider-error')
  return {kind:'provider',reference:row.reference};
 if(type==='project-invalid'&&data.reason==='target-baseline-failed'&&data.passed===false&&executed(data.exitCode))
  return {kind:'invalid-baseline',reference:row.reference};
 if(type!=='verification')return null;
 const {correctness,performance}=data;
 if(correctness?.passed===true&&performance?.passed===false&&!performance.skipped&&executed(performance.exitCode))
  return {kind:'performance',reference:row.reference};
 if(correctness?.passed!==false||!executed(correctness.exitCode))return null;
 const receipt=correctness.receipt??row.verifierVerdict;
 if(receipt?.correct===false&&/^[a-f0-9]{64}$/.test(receipt.kernelSha256??''))
  return {kind:'candidate-verdict',reference:row.reference};
 return {kind:'verifier-exit',reference:row.reference,replayEvidence:row.artifacts?.some(artifact=>artifact.primaryVerification===true)===true};
}

export function failureTriageBoundaries(group){
 const records=[...group.occurrences.map(o=>[o,group.type]),...(group.linkedDiagnostics??[]).map(o=>[o,o.type])];
 return [...new Map(records.map(([row,type])=>observedBoundary(row,type)).filter(Boolean).map(boundary=>[boundary.reference,boundary])).values()];
}

export function routeFailureClassification(value,group){
 const all=failureTriageBoundaries(group),direct=new Set(group.occurrences.map(row=>row.reference));
 const scoped=all.some(boundary=>direct.has(boundary.reference))?all.filter(boundary=>direct.has(boundary.reference)):all;
 const boundaries=scoped.filter(boundary=>value.evidenceRefs.includes(boundary.reference));
 const find=kind=>boundaries.find(boundary=>boundary.kind===kind);
 const provider=find('provider'),infrastructure=find('infrastructure')??provider,invalid=find('invalid-baseline');
 const candidate=find('candidate-verdict'),exit=find('verifier-exit'),performance=find('performance');
 const attributed={harness:find('infrastructure'),provider,specification:invalid};
 if(value.cause!=='unknown'&&!attributed[value.cause])
  throw Error('Ownership '+value.cause+' needs a cited runtime boundary that establishes it. A failed command or verifier exit does not locate the fault. Use cause=unknown if that boundary is absent. Available runtime boundaries: '+JSON.stringify(scoped));
 if(value.bucket==='harness'&&!infrastructure)
  throw Error('Harness routing requires a cited infrastructure boundary. Use bucket=unknown when the origin of a failure remains unresolved.');
 if(infrastructure)return {route:'human',observed:infrastructure};
 if(invalid)return {route:'phase1',observed:invalid};
 if(value.bucket==='optimization'&&performance)return {route:'optimization',observed:performance};
 if(value.bucket==='correctness'&&candidate)return {route:'baseline',observed:candidate};
 if(exit)return {route:'unresolved',observed:exit,...(exit.replayEvidence?{
  question:'Can the retained source and original verifier reproduce a failure inside the candidate or its correctness oracle?'
 }: {})};
 return {route:'unresolved',observed:candidate??performance??null};
}
