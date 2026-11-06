import {createHash} from 'node:crypto';

// Input is oldest-to-newest COMPLETION order, not admission order. The caller
// owns settlement and supplies factual failure-recorded data, never inferred causes.
export function failureInvestigations(completedAttempts, existingInvestigations=[]) {
  const known=new Set(existingInvestigations.map(item=>item.key));
  const groups=new Map();
  const completed=completedAttempts.filter(a=>a.outcome==='failed'||a.outcome==='passed').slice(-10);
  for(const attempt of completed) {
    if(attempt.outcome!=='failed')continue;
    const seen=new Set();
    for(const failure of (attempt.failures??[]).slice(-20)) {
      const {phase,kind,message,operation=''}=failure;
      if([phase,kind,message].some(value=>typeof value!=='string'||!value.trim()))continue;
      // Exact observations deliberately avoid normalizing numbers/paths into a
      // speculative common infrastructure cause. Target scopes remain distinct.
      const signature=JSON.stringify([attempt.target??null,phase,kind,message]);
      const key=createHash('sha256').update(signature).digest('hex');
      if(seen.has(key))continue;
      seen.add(key);
      const group=groups.get(key)??{key,target:attempt.target??null,observation:{phase,kind,message},attempts:[]};
      if(!group.attempts.some(a=>a.id===attempt.id))group.attempts.push({id:attempt.id,project:attempt.project,...(operation?{operation}:{}),evidence:(failure.evidence??[]).slice(0,16)});
      groups.set(key,group);
    }
  }
  return [...groups.values()].filter(group=>group.attempts.length>=3&&!known.has(group.key));
}

// A claim is bookkeeping only. It neither admits a job nor changes its goal,
// verifier, elapsed allowance, outcome, or recovery policy. Persist the key on
// the later job so scheduling retries reuse this same claim.
export function claimFailureInvestigation(state,candidate) {
  if(!candidate?.key||!Array.isArray(candidate.attempts)||new Set(candidate.attempts.map(a=>a.id)).size<3)throw new Error('Invalid failure investigation');
  state.db.exec('BEGIN IMMEDIATE');
  try {
    const prior=state.db.prepare("SELECT data FROM events WHERE attempt IS NULL AND type='failure-investigation-claimed' AND json_extract(data,'$.key')=? LIMIT 1").get(candidate.key);
    if(!prior)state.event(null,'failure-investigation-claimed',candidate);
    state.db.exec('COMMIT');
    return !prior;
  }catch(error){state.db.exec('ROLLBACK');throw error;}
}
