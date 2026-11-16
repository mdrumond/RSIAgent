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
proof records inline. Every output `reason`, `text`, `findings` or issue needs a
claim whose `path` and `text` match that field, such as `answers.evidence.text`.
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
