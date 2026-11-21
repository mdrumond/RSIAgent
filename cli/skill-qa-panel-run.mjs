import {panelClaims, decideQaPanel} from './skill-qa-panel-decision.mjs';

export function panelVotes(panel) {
 return panel.slots.map(slot => slot?.status === 'submitted' ? slot.vote : null);
}

export function panelAudits(panel) {
 return (panel.audit?.audits ?? []).map((row, i) => ({
  ...row,
  proofs: (panel.audit.claims ?? []).filter(claim => claim.path === 'audits.' + i + '.reason').flatMap(claim => claim.proofs),
 }));
}

export function replayPanelDecision(panel) {
 return decideQaPanel(panelVotes(panel), panelAudits(panel));
}

// Re-open only missing votes after an explicit retry at lower provider concurrency.
export function reopenMissingVotes(round) {
 const panel=round?.panel;
 if(!panel?.decision?.missingSlots.length || panel.decision.verdict!=='inconclusive')return false;
 (panel.history??=[]).push({slots:panel.slots,audit:panel.audit,auditError:panel.auditError,decision:panel.decision});
 panel.slots=panel.slots.map(slot=>slot?.status==='submitted'?slot:null);
 for(const key of ['audit','auditError','auditElapsedSeconds','decision'])delete panel[key];
 delete round.review;
 return true;
}

const errorInfo = error => ({name: error.name, message: error.message, ...(error.reviewSession ? {reviewSession: error.reviewSession} : {})});

export async function runPanelRound({round, payload, index, call, persist, now, signal, concurrency}) {
 const panel = round.panel ??= {slots: Array(5).fill(null)};
 const suffix=panel.history?.length?'-retry-'+panel.history.length:'';
 const pending = panel.slots.flatMap((slot, i) => slot ? [] : [i]);
 async function worker() {
  while (pending.length && !signal.aborted) {
   const i = pending.shift(), started = now(), slot = i + 1;
   let result;
   try {
    const raw = await call('qaReview', payload, index+'-qaReview-'+i+suffix, true);
    const vote = payload.validate(raw?.value ?? raw);
    result = {status: 'submitted', vote, ...(raw?.stats ? {stats: raw.stats} : {})};
   } catch (error) {
    signal.throwIfAborted();
    result = {status: 'missing', error: errorInfo(error)};
   }
   panel.slots[i] = {slot, elapsedSeconds: Math.max(0, (now()-started)/1000), ...result};
   await persist();
  }
 }
 const workers = await Promise.allSettled(Array.from({length: Math.min(concurrency, pending.length)}, worker));
 for (const worker of workers) if (worker.status === 'rejected') throw worker.reason;
 signal.throwIfAborted();
 const objections = panelClaims(panelVotes(panel));
 if (objections.length && !panel.audit && !panel.auditError) {
  const started = now();
  const validate = value => {
   if (!Array.isArray(value?.audits)) throw Error('Audit requires a list of dispositions');
   const ids = new Set(objections.map(item => item.id));
   for (const row of value.audits) {
    if (!ids.delete(row.claimId)) throw Error('Unexpected or repeated objection: '+row.claimId);
    if (!['verified','disproved','unresolved'].includes(row.disposition) || !row.reason?.trim()) throw Error('Provide an audit disposition and its supporting reason');
   }
   if (ids.size) throw Error('Missing dispositions for objections: '+[...ids].join(', '));
   return value;
  };
  try {
   const raw = await call('qaPanelAudit', {...payload, objections, validate}, index+'-qaPanelAudit'+suffix, true);
   panel.audit = validate(raw?.value ?? raw);
  } catch (error) {
   signal.throwIfAborted();
   panel.auditError = errorInfo(error);
  }
  panel.auditElapsedSeconds = Math.max(0, (now()-started)/1000);
  await persist();
 }
 panel.decision = replayPanelDecision(panel);
 return panel.decision;
}
