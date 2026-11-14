# Continuous failure-driven skill updates

Set `skills.priority.enabled: true` alongside `skills.background: true` to use the
campaign-local priority worker. Each campaign owns its SQLite backlog, candidate
versions and experiments. Existing pending legacy comparisons finish first;
historical results and active skills are retained.

The worker incrementally inventories existing and new Phase 1, Phase 2,
curriculum, verification and recovery failures. It classifies responsibility,
retrieves supporting memory, and proposes focused hypotheses. Harness findings
require user intervention. Missing facts or relevant tests become learning
requests for a compatible future curriculum draft. Learning results return to
the originating hypotheses.

Open hypotheses are ranked by expected impact, recurrence, recoverable time and
evidence strength. New urgent work can overtake queued work. A running comparison
retains its frozen hypothesis until completion. Rejected instructions do not run
again merely because a duplicate occurrence or a new archive path appears.

Each hypothesis has 30 elapsed minutes, including proposal writing, Q&A,
evaluation and cleanup. Resume deducts prior spend. The independent Q&A review
must approve the actual skill text before A/B. Both arms use five frozen cases
with two repetitions. Correctness promotion requires two additional passes out
of ten; it does not require a speedup. Optimization requires correct kernels in
both arms and at least 5% geometric-mean speedup from paired device measurements.
Incomplete evidence cannot promote. Unknown commands retain their original
handles; positive closure ends an interrupted comparison without late credit.

`ascend-rsi status --instance INSTANCE` exposes the backlog, ingestion progress,
current hypothesis, allocated/spent time, candidate and blocked execution state
under `backgroundTuning`. The worker uses the existing reserved background slot;
foreground concurrency and memory admission remain controller policies.

## Hypothesis-relevant cases

An optional `skills.priority.evaluations` array supplies prequalified catalogues.
Each entry contains `id`, `description`, `objective`, `target`, `tasks` and
`binding`. A suite has exactly five distinct cases. Device cases use
`kind: target-kernel` with `target: a3` or `a5`; CPU contract cases use
`kind: hypothesis-case`, `target: cpu`, and correctness only. Each task declares
`id`, `bucket`, `prompt`, `files` (sample-relative names mapped to UTF-8 contents)
and a trusted `checker` path. Optimization also needs a `pairedChecker`.
Device bindings use the campaign's registered environment options. Machine paths
and runtime identities belong in ignored instance/profile configuration.

The worker selects a relevant suite, never the generic legacy four-task
catalogue. The built-in CPU cases are eligible only for their known proposal
identities and establish host contract behavior, not device arithmetic or timing.
Missing or incompatible suites produce learning requests.

A curriculum learning project may export a protected `skill-evaluation.json`
listed in its verifier files. It declares `target`, `objective` and five `tasks`
with the fields above. In this export, `files` maps sample-relative names to
file paths relative to the manifest; all referenced files and checkers must stay
within its trusted verifier directory. Each task also supplies
`controls: {reference: {files}, negatives: [{files}]}`, with a correct reference
and at least one independent incorrect implementation. Every control includes
`kernel.py`. Checkers are standalone and accept the sample directory; device
checkers emit the existing trusted kernel receipt.

The controller copies and freezes the exported cases, then runs the reference
and negatives before admitting A/B. References must pass; negatives must fail
with the numeric rejection exit code, not an invocation or harness error.
Receipts and elapsed spend survive resume. Curriculum construction still emits
only PyTorch workload specifications/references; actors implement device kernels.
