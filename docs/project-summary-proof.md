# Project summary facts and claim proofs

At completion, the existing findings pass writes `project-summary.json` before
consulting its narrative cache. It enumerates every attempt and relevant event
for that project, separates project status from individual verification outcomes,
and exposes complete verification counts with a snapshot event cutoff. Legacy
narratives and summaries superseded by new evidence remain in `findings-history`.
Missing evidence and unknown verdicts stay explicit instead of becoming failures
or zero-valued measurements.

Performance facts come only from the exact log named by a verification record.
The reader checks recorded hashes, bounds file inspection, takes the aggregate
geometric-mean field rather than an individual case ratio, and preserves each
candidate/case/verifier/runtime identity separately. A log contradicting the
controller pass is not a verified measurement. These are recorded observations,
not proof of an optimization mechanism or transferable performance.

The existing summarizer receives these facts inline. Its output is a list of
claims, not an independent uncited narrative. Accepted claims are composed into
the compatible `findings` string (at most 4000 characters); hypotheses are labeled
unverified. A claim cites an exact source reference and contiguous copied quote:

```json
{"text":"One passing verification was recorded","kind":"observation",
 "proofs":[{"reference":"project:p:passingVerificationCount","quote":"1",
 "factId":"project:p:passingVerificationCount","value":1}]}
```

For inspected files, the summarizer uses `file:<exact path listed in paths>`.
Those files must satisfy the existing attempt-artifact boundary and bounded
complete-file check. Typed project facts require the exact `factId` and `value`.

Native classification, skill review and Q&A receive project facts and scoped
proof records inline. Every output `reason`, `text`, `findings`, structured issue `problem`/`description` or string issue needs a
claim whose `path` and `text` match that field, such as `answers.evidence.text`.
Structured issue inline citations are checked too. Rejections enumerate the
required prose paths; categorical `verdict` values do not need prose claims.
Unknown references, mismatched quotes, omitted-byte stitching and wrong typed
values reject submission with tool feedback within the same Pi session and
elapsed allowance. Cached native submissions undergo the same check.

A matching quote establishes copying, **not entailment**. Independent Q&A gets
the answer claims and the same project snapshot and must check whether the cited
source actually supports each statement. Fixable contradictions require revision.
Existing Q&A approvals without the proof protocol, or bound to changed project
facts, are archived and re-reviewed within the original remaining allocation.
Historical evaluation results are not rewritten.

This change does not restart campaigns, publish skills or establish that an
instruction improves actor performance. Local behavioral coverage is in
`tests/ascend/project-summary-proof.mjs` and the existing native workflow tests.


Isolated review forks retain a read-only `project_evidence_root` pointing to the
source campaign, including when resuming an older empty fork. Project facts are
read there; the large source database is neither copied nor mutated. Q&A's
original elapsed ledger remains authoritative, while native answer/review session
directories include the proof protocol and project-facts identity so superseded
native cached submissions cannot bypass re-review.

Verified learning contributes scoped record references and `file:` aliases for
its source, verifier and log files. CPU/device/project status and target binding
remain attached to those records; source code alone does not prove execution.
Inspection is bounded to complete 64 KiB files and a 256 KiB proof-byte packet;
missing, oversized or mismatched sources remain explicit unknowns. Additional
learning evidence cannot expand the verified source scope. Q&A answer references
may use only actual supplied project/learning IDs, never an arbitrary prefix.

Proof-required native submissions expose a typed tool schema with `claims` at the
result root. Q&A review additionally types verdict, reason, issues and learning
requests; misplaced `reasonClaims` or nested claims are rejected before admission.
Revision rounds share the original elapsed allocation; there is no round-count
stopping budget.
