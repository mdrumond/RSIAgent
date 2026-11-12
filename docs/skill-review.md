# Failure-driven skill review

`node cli/main.mjs skills-review --config review.json` runs an isolated review.
It never publishes to the source campaigns. Supply consistent, quiescent snapshots,
not live directories. Snapshot `state.sqlite`, `knowledge.sqlite`, `instance.json`,
`skills/versions/` and `attempts/*/job.json`; retain generation events in state.
Optional `evidence-files.json` maps original artifact paths to copied relative
`{path,sha256,size}` records so classifiers can inspect retained verifier logs.

```json
{
  "sources": [{"id":"a3","root":"/snapshot/a3"}],
  "output": "/experiments/review",
  "stage": "all",
  "seconds": 7200,
  "python": "/configured/python",
  "maxCandidates": 3,
  "maxLearning": 2
}
```

The actor profile defaults to the snapshot's skill writer or actor; an explicit
`profile` can override it. Credentials remain in environment/ignored profile files.
Stages `inventory`, `classify`, `propose`, `learn`, `evaluate`, `report` are
independently runnable. Keep sources, profile, Python, limits and output unchanged
when resuming. The original elapsed deadline spans all invocations; report remains
available after it expires. Classification and review native sessions, searches,
evidence, decisions and original references are retained alongside `review.json`.
`report.json` shows coverage, user interventions, learning status and candidate
outcomes; token/cost telemetry is recorded but never stops work.

All saved failure observations are inventoried, including those in successful or
unfinished attempts. Related observations are grouped; absence of a cause is not
proof of a skill gap. Harness faults require user intervention. Guidance may teach
preserving work and reporting them, never repairing harness-owned state.

A memory-backed update becomes an immutable candidate. Existing protected-guidance
rules and the four-task, two-run CPU A/B comparison apply. Publication occurs only
inside the candidate's isolated instance. Its next fresh actor reads the accepted
version through the existing skill interface. CPU A/B acceptance does not establish
GPU kernel improvement. Rejected and inconclusive proposals remain inspectable.

Learning requests carry a question, hypothesis and required evidence. CPU requests
use the existing curriculum's independent controls and then a normal CPU actor wave;
verified results feed one further proposal review. Device questions remain pending.
Learning does not retroactively resolve or alter the original failed project.

For a later explicit rollout, stage `enqueue` with an explicit `learningInstance`
imports compatible requests into an opted-in destination. It rejects review source
instances. Programmatic `enqueueLearningRequest(state, request)` in
`controller/skill-review.mjs` idempotently adds a request to an instance. Enable
`skills.review.enabled` to let new Phase 1 drafts claim compatible target/topic
requests. Existing drafts, per-topic requirements, project counts and actor budgets
are unchanged. This opt-in hook does not run reviews or change live configurations
by itself. Exhausted or failed claimed drafts retain their request for inspection;
they do not automatically create repeated experiments. Successful actor postprocessing
records linked findings in the request and publishes ordinary memory through the
existing memory path. No new background scheduler is introduced.
