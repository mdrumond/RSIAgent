import {createHash} from 'node:crypto';

const panelSize = 5;
const verdicts = new Set(['approved', 'revise', 'learn', 'inconclusive']);
const kinds = new Set(['answer', 'draft', 'evidence', 'advisory']);
const hasText = value => typeof value === 'string' && value.trim().length > 0;

// The runner validates the native review and its proofs before assigning a slot.
// Fail closed if a stored ballot is incomplete, including an approval with blockers.
function usable(vote) {
 if (!vote || !verdicts.has(vote.verdict) || !hasText(vote.reason)) return false;
 const issues = vote.issues ?? [];
 if (!Array.isArray(issues) || issues.some(issue => !kinds.has(issue?.kind) || !hasText(issue.problem ?? issue.description))) return false;
 const material = issues.some(issue => issue.kind !== 'advisory');
 if (vote.verdict === 'approved' && material || vote.verdict === 'revise' && !material) return false;
 return vote.verdict !== 'learn' || Array.isArray(vote.learningRequests) && vote.learningRequests.length > 0;
}

function ballotSlots(votes) {
 if (!Array.isArray(votes) || votes.length !== panelSize) throw Error('Expected five positional panel ballots');
 return Array.from(votes, vote => usable(vote) ? vote : null);
}

function groupedFindings(slots) {
 const groups = new Map(), advisories = [];
 for (const [index, vote] of slots.entries()) {
  if (!vote) continue;
  const slot = index + 1;
  const add = (issue, issueIndex, sources) => {
   const mention = {slot, issueIndex, issue, proofs: sources.flatMap(claim => claim.proofs ?? [])};
   if (issue.kind === 'advisory') {
    advisories.push(mention);
    return;
   }
   const path = issue.path ?? issue.field ?? null;
   // Keep separate claims that target the same paragraph or cite the same artifact.
   // Identical findings retain every reviewer's distinct citations for the auditor.
   const key = JSON.stringify([issue.kind, path, issue.summaryItemId ?? null, issue.problem?.trim(), issue.description?.trim()]);
   if (!groups.has(key)) {
    const id = createHash('sha256').update(key).digest('hex');
    groups.set(key, {id, kind: issue.kind, path, problem: issue.problem ?? issue.description, mentions: []});
   }
   groups.get(key).mentions.push(mention);
  };
  for (const [issueIndex, issue] of (vote.issues ?? []).entries()) {
   const prefix = 'issues.' + issueIndex + '.';
   add(issue, issueIndex, (vote.claims ?? []).filter(claim => claim.path?.startsWith(prefix)));
  }
  if (['learn', 'inconclusive'].includes(vote.verdict) && !(vote.issues ?? []).some(issue => issue.kind !== 'advisory')) {
   add({kind: 'evidence', problem: vote.reason}, null, (vote.claims ?? []).filter(claim => claim.path === 'reason'));
  }
 }
 return {claims: [...groups.values()], advisories};
}

export function panelClaims(votes) {
 return groupedFindings(ballotSlots(votes)).claims;
}

function resolveAudit(claim, audits) {
 const matching = audits.filter(audit => audit?.claimId === claim.id);
 const audit = matching[0];
 if (matching.length === 1 && hasText(audit.reason)) {
  if (audit.disposition === 'unresolved') return {...claim, audit};
  const materialized = Array.isArray(audit.proofs) && audit.proofs.length > 0 &&
   audit.proofs.every(proof => hasText(proof?.reference) && hasText(proof.quote) && hasText(proof.recordSha256));
  if (materialized && ['verified', 'disproved'].includes(audit.disposition)) return {...claim, audit};
 }
 return {...claim, audit: {claimId: claim.id, disposition: 'unresolved', reason: 'A unique evidence-backed audit is unavailable.', proofs: []}};
}

export function decideQaPanel(votes, audits = []) {
 const slots = ballotSlots(votes);
 const grouped = groupedFindings(slots);
 const claims = grouped.claims.map(claim => resolveAudit(claim, audits));
 const approvals = slots.filter(vote => vote?.verdict === 'approved').length;
 const missingSlots = slots.flatMap((vote, index) => vote ? [] : [index + 1]);
 const verified = claims.filter(claim => claim.audit.disposition === 'verified');
 const unresolved = claims.filter(claim => claim.audit.disposition === 'unresolved');
 const issues = verified.map(claim => ({
  ...claim.mentions[0].issue,
  claimId: claim.id,
  // Pass stable primary selectors to the correcting author. The full audited
  // excerpts remain in the panel record and can include large quoted inputs.
  proofs: claim.audit.proofs.map(proof => Object.fromEntries(
   ['reference', 'startLine', 'endLine', 'segment', 'field', 'factId', 'recordSha256']
    .filter(key => Object.hasOwn(proof, key)).map(key => [key, proof[key]])
  )),
  problem: claim.problem + '\nEvidence audit: ' + claim.audit.reason,
 }));
 const result = {approvals, missingSlots, claims, advisories: grouped.advisories, issues};
 if (unresolved.length) return {...result, verdict: 'inconclusive', route: 'inconclusive', reason: 'Material objections still require evidence.'};
 if (verified.length) {
  const route = verified.some(claim => claim.kind === 'evidence') ? 'evidence'
   : verified.some(claim => claim.kind === 'draft') ? 'writer' : 'answer';
  return {...result, verdict: 'revise', route, reason: issues.map(issue => issue.problem).join('\n\n')};
 }
 if (approvals >= 3) return {...result, verdict: 'approved', route: 'approved', reason: 'At least three reviewers approved and all material objections were disproved or absent.'};
 return {...result, verdict: 'inconclusive', route: 'inconclusive', reason: 'Fewer than three reviewers explicitly approved.'};
}
