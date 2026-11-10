// Native remote handles are immutable: a positively recorded terminal state
// remains closure evidence after reconnect. Local identities need fresh proof.
export function isNativeTerminalClosure(data) {
  return /^remote:[^:]+:(job|transfer):[^:]+$/.test(data.handle ?? '') &&
    data.state === 'stopped' && ['completed', 'failed', 'cancelled'].includes(data.nativeState);
}

// This proves closure only. Operation reconciliation still owns elapsed accounting.
export function terminalExecutionProofs(events) {
  const proofs = new Map();
  for (const {type,data} of events)
    if (type === 'execution-state' && isNativeTerminalClosure(data))
      proofs.set(data.handle,{state:'stopped',nativeState:data.nativeState});
  return proofs;
}

// Exclusion is an accounting policy recorded before submission, never inferred
// from missing timing or a successful exit.
export function excludedExecutionProofs(events) {
  const submissions = new Map();
  for (const {type,data} of events)
    if (type === 'execution-submitting') submissions.set(data.submission,data.accountingExcluded === true);
  const excluded = new Map();
  for (const {type,data} of events) {
    if (type !== 'execution-submitted' || !data.handle) continue;
    const known = submissions.get(data.submission) === true && data.accountingExcluded !== false;
    excluded.set(data.handle,known && excluded.get(data.handle) !== false);
  }
  const proofs = new Map([...excluded].filter(([handle,known]) => known && handle.startsWith('remote:'))
    .map(([handle]) => [handle,null]));
  for (const {type,data} of events) {
    if (type !== 'execution-state' || data.state !== 'stopped' || data.accountingExcluded !== true || !proofs.has(data.handle)) continue;
    proofs.set(data.handle,{state:'stopped',executionSeconds:0,accountingExcluded:true,
      ...(Number.isInteger(data.exitCode) ? {exitCode:data.exitCode} : {})});
  }
  return proofs;
}
