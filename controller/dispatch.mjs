// Admission policy only. Execution, elapsed accounting and device selection stay
// with the canonical worker and its Pi extension.
export function selectAdmission(jobs,{running=[],concurrency=1,freshSinceRecovery=0}={}) {
  if(!Number.isSafeInteger(concurrency)||concurrency<1)throw new Error('Positive actor concurrency required');
  if(running.length>=concurrency)return null;
  const fresh=jobs.find(job=>job.kind==='original');
  const recovery=jobs.find(job=>['investigation','retry'].includes(job.kind));
  const recoveryActive=running.some(job=>['investigation','retry'].includes(job.kind));
  if(!recovery||recoveryActive)return fresh??null;
  if(!fresh)return recovery;
  if(freshSinceRecovery>=3)return recovery;
  if(concurrency===1)return fresh;
  // Reserve the final available worker for recovery, while giving fresh work
  // the first slots. A completed recovery never consumes another target's share.
  return running.length===concurrency-1?recovery:fresh;
}

export function memoryAdmission({maxBytes,currentBytes,workerPeakBytes,activeActors=0,requested=1,reserveBytes=1024**3}) {
  for(const value of [maxBytes,currentBytes,reserveBytes])if(!Number.isSafeInteger(value)||value<0)throw new Error('Measured byte counts required');
  if(!Number.isSafeInteger(activeActors)||activeActors<0||!Number.isSafeInteger(requested)||requested<1)throw new Error('Valid actor counts required');
  if(activeActors>=requested)return {admit:false,reason:'actor-capacity'};
  if(workerPeakBytes===undefined||workerPeakBytes===null) {
    // Establish a real worker peak before allowing concurrent actors.
    return activeActors===0&&currentBytes+reserveBytes<maxBytes
      ? {admit:true,learning:true} : {admit:false,reason:'memory-measurement'};
  }
  if(!Number.isSafeInteger(workerPeakBytes)||workerPeakBytes<=0)throw new Error('Positive measured worker peak required');
  const reservation=Math.ceil(workerPeakBytes*1.25);
  return currentBytes+reservation+reserveBytes<=maxBytes
    ? {admit:true,reservationBytes:reservation} : {admit:false,reason:'memory-capacity'};
}
