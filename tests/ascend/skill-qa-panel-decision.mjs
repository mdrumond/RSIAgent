import test from 'node:test';
import assert from 'node:assert/strict';
import {decideQaPanel, panelClaims} from '../../cli/skill-qa-panel-decision.mjs';
import {materializeProof} from '../../cli/claim-proofs.mjs';

const source = {reference: 'trial:4', text: 'Intermediate edit: 1280 wrong.\nFinal score: pass.'};
const proof = materializeProof({reference: source.reference, startLine: 1, endLine: 1}, [source]);
const approved = {verdict: 'approved', reason: 'The evidence supports the instruction.'};
const draft = {kind: 'draft', path: 'changes.0.text', problem: 'The scored starter had 1280 wrong.'};
const answer = {kind: 'answer', path: 'answers.action.text', problem: 'The answer treats static analysis as a device measurement.'};
const evidence = {kind: 'evidence', problem: 'The final score is missing.'};
const revise = (...issues) => ({
 verdict: 'revise', reason: 'Concrete defects need correction.', issues,
 claims: issues.map((issue, index) => ({path: 'issues.' + index + '.problem', proofs: [proof]})),
});
const audit = (claim, disposition, reason = 'The output is from the intermediate edit.') => ({
 claimId: claim.id, disposition, reason, proofs: [proof],
});

test('a disproved minority objection does not veto three explicit approvals', () => {
 const votes = [approved, approved, approved, revise(draft), null];
 const [claim] = panelClaims(votes);
 assert.equal(claim.mentions[0].slot, 4);
 assert.deepEqual(claim.mentions[0].proofs, [proof]);
 const decision = decideQaPanel(votes, [audit(claim, 'disproved')]);
 assert.equal(decision.verdict, 'approved');
 assert.equal(decision.approvals, 3);
 assert.deepEqual(decision.missingSlots, [5]);
});

test('audit each distinct claim on a shared target and keep every duplicate mention', () => {
 const other = {...draft, problem: 'The hash identifies the tree.'};
 const votes = [approved, approved, approved, revise(draft, other), revise(draft)];
 const claims = panelClaims(votes);
 assert.equal(claims.length, 2);
 assert.deepEqual(claims[0].mentions.map(mention => mention.slot), [4, 5]);
 const decision = decideQaPanel(votes, [audit(claims[0], 'disproved'), {...audit(claims[1], 'verified', 'The artifact binds only the kernel.'), proofs: [
  materializeProof({reference: source.reference, startLine: 2, endLine: 2}, [source]),
 ]}]);
 assert.equal(decision.route, 'writer');
 assert.equal(decision.issues.length, 1);
 assert.match(decision.reason, /The artifact binds only the kernel[.]/);
 assert.equal(decision.issues[0].claimId, claims[1].id);
 // Feedback identifies the primary span used by the auditor, even when the
 // original reviewer cited a different span.
 assert.equal(decision.issues[0].proofs[0].recordSha256, proof.recordSha256);
 assert.equal(materializeProof(decision.issues[0].proofs[0], [source]).quote, source.text.split('\n')[1]);
});

test('a verified minority objection sends consolidated evidence to the appropriate author', () => {
 for (const [issues, route] of [[[answer], 'answer'], [[answer, draft], 'writer'], [[draft, evidence], 'evidence']]) {
  const votes = [approved, approved, approved, approved, revise(...issues)];
  const decision = decideQaPanel(votes, panelClaims(votes).map(claim => audit(claim, 'verified')));
  assert.equal(decision.verdict, 'revise');
  assert.equal(decision.route, route);
  assert.deepEqual(decision.issues.map(issue => issue.kind), issues.map(issue => issue.kind));
 }
});

test('unresolved, conflicting, absent, and unmaterialized audit evidence keeps the decision inconclusive', () => {
 const votes = [approved, approved, approved, approved, revise(draft)];
 const [claim] = panelClaims(votes);
 const verified = audit(claim, 'verified');
 for (const audits of [[], [audit(claim, 'unresolved')], [verified, audit(claim, 'disproved')], [{...verified, proofs: [{reference: source.reference}]}]]) {
  const decision = decideQaPanel(votes, audits);
  assert.equal(decision.verdict, 'inconclusive');
  assert.equal(decision.claims[0].audit.disposition, 'unresolved');
 }
});

test('missing slots and invalid approvals never reduce the majority threshold', () => {
 const votes = [approved, approved, {...approved, issues: [draft]}, null, null];
 const decision = decideQaPanel(votes);
 assert.equal(decision.approvals, 2);
 assert.deepEqual(decision.missingSlots, [3, 4, 5]);
 assert.equal(decision.verdict, 'inconclusive');
 assert.throws(() => decideQaPanel([approved, approved]));
});

test('even disproving every objection does not turn a revision vote into an explicit approval', () => {
 const votes = [approved, approved, revise(draft), revise(draft), revise(draft)];
 const [claim] = panelClaims(votes);
 assert.equal(decideQaPanel(votes, [audit(claim, 'disproved')]).verdict, 'inconclusive');
});

test('retain advisory notes and surface missing evidence expressed in a verdict reason', () => {
 const advisory = {kind: 'advisory', problem: 'A shorter example may help.'};
 const abstention = {verdict: 'inconclusive', reason: 'No final score exists.', claims: [{path: 'reason', proofs: [proof]}]};
 const votes = [{...approved, issues: [advisory]}, approved, approved, approved, abstention];
 const [claim] = panelClaims(votes);
 assert.equal(claim.kind, 'evidence');
 assert.deepEqual(claim.mentions[0].proofs, [proof]);
 const decision = decideQaPanel(votes, [audit(claim, 'disproved')]);
 assert.equal(decision.verdict, 'approved');
 assert.deepEqual(decision.advisories[0].issue, advisory);
});

test('checked premises and recommendations cannot establish a material defect or remove uncertainty',async()=>{
 const {assertionVerdict,validateAssertionPlan}=await import('../../cli/skill-qa-assertions.mjs');
 const proof={reference:'receipt:1',quote:'checker passed',recordSha256:'fixed'};
 const row=(id,role,label)=>({id,objectionId:'parent',sourceField:'problem',assertion:id,role,label,reason:'recorded result',basis:'check',bindings:[{role:'check',reference:proof.reference}],proofs:[proof]});
 const atoms=[row('a','material_claim','contradicted'),row('b','factual_premise','supported'),row('c','proposed_action','supported')];
 validateAssertionPlan(atoms,[{id:'parent',problem:'Compound assertion'}]);
 assert.equal(assertionVerdict(atoms).disposition,'disproved');
 atoms.push(row('d','material_claim','insufficient'));
 assert.equal(assertionVerdict(atoms).disposition,'unresolved');
 assert.throws(()=>validateAssertionPlan(atoms.slice(1,3),[{id:'parent',problem:'Compound assertion'}]),/material claim/);
 const bad={...atoms[0],proofs:[{...proof,reference:'input:answers'}],bindings:[{role:'check',reference:'input:answers'}]};
 assert.throws(()=>assertionVerdict([bad]),/only what was written/);
});
