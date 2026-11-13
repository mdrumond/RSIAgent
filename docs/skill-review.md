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
Native classification batches up to eight groups sharing one source and current
guidance version in a Pi session; every group ID and evidence reference is validated
before the batch is saved. Existing individual native sessions resume in place.
Injected single-group classifier adapters remain supported. Timing/token receipts
include group IDs/counts so throughput can be measured directly.

Stages `inventory`, `classify`, `propose`, `learn`, `evaluate`, `report` are
independently runnable. Keep sources, profile, Python, limits and output unchanged
when resuming. Attempt-job metadata, the evidence index and the actual copied-log
SHA256 values are also frozen and verified; changed inputs require a fresh review
snapshot, rather than silently reusing older attribution. Legacy fingerprints
upgrade only after the old signature matches, actual copied digests verify and
recomputed effective inventory equals the saved inventory; clocks/results/sessions
are retained. Unprovable legacy histories remain report-inspectable without model
resume or mutation of their saved record. The original elapsed deadline spans all invocations; report remains
available after it expires. Classification and review native sessions, searches,
evidence, decisions and original references are retained alongside `review.json`.
Full observations and distinct diagnostic representatives are saved as JSONL files.
Native Pi filesystem tools are restricted by a native `tool_call` policy to the
current review directory and explicitly copied evidence or verified learning paths.
Broad workspace/config-root searches are blocked before execution; indexed
knowledge is retrieved through KDB MCP. Classification errors identify the invalid
field so the reviewer can correct its submission directly.
Native Pi `read` supports offset/limit paging; submission requires reading every
distinct diagnostic, while duplicate accounting events require no repeated reads.
`report.json` shows coverage, user interventions, learning status and candidate
outcomes; token/cost telemetry is recorded but never stops work. A settled
comparison that exhausts its allowance remains inconclusive and does not block
other candidates. Unknown execution closure still blocks dispatch. Changed
proposals receive separate candidate identities and retain previous decisions.

All saved failure observations are inventoried, including explicit negative
passed/correct verdicts in successful or unfinished attempts. Expected negative
controls that exit as expected are excluded; unexpected control outcomes remain
failures. Generic wrapper failures link to actual same-attempt verdicts and copied
logs, so a pre-actor label is not treated as proof of a harness defect. Related observations are grouped; absence of a cause is not
proof of a skill gap. Harness faults require user intervention; their learning requests remain in that
bucket and cannot automatically enter Phase 1. Guidance may teach
preserving work and reporting them, never repairing harness-owned state.

A memory-backed update becomes an immutable candidate. Existing protected-guidance
rules and the four-task, two-run CPU A/B comparison apply. Publication occurs only
inside the candidate's isolated instance. Its next fresh actor reads the accepted
version through the existing skill interface. CPU A/B acceptance does not establish
GPU kernel improvement. Rejected and inconclusive proposals remain inspectable. A writer/provider/item
timeout is a retained host inconclusive result, not a no-change knowledge finding;
other buckets and supported candidates continue. A failed frozen bucket is not
retried without new verified learning. Shared deadline/user cancellation still
stops the run. Scope/reference validation feedback identifies the fields to fix.

Learning requests carry a question, hypothesis, originating group IDs and required
evidence. Every accepted proposal immediately persists all questions before any
CPU execution; report/recovery also backfills legacy raw proposal questions without
model calls or a new allocation. Question-specific group IDs keep unrelated actor
questions eligible in mixed buckets; legacy questions without them conservatively
inherit their whole bucket. CPU requests
use the existing curriculum's independent controls and then a normal CPU actor wave;
verified results feed one further proposal review. Interrupted admitted learning
projects resume the original actor or postprocessing, preserving source, session
and elapsed allocation. Failed learning remains inconclusive and does not block
other candidates. Device questions remain pending.
Learning does not retroactively resolve or alter the original failed project.

The default native evaluator executes on the controller host. A frozen `binding`
records runtime identity; it does not select or activate a remote runtime. Run
the controller on the configured device host for this default path. Isolated
local-controller studies can supply `runMicroeval` sample/verification hooks and
`finalizeKernelPairs` measurement hooks using the existing Ascend/CPL environment;
`nativeSample` accepts its prepared target through `targetEnvironment`. Automatic
remote dispatch from the public `skills-review` configuration is not implemented.
Local-controller studies can set `nativeSample.localFilesystemSandbox` to run
Pi's Bash commands through Bubblewrap with only the sample work directory
writable and a private temporary directory and process namespace. Bubblewrap
and namespace support must be available; this option fails closed when they are
not. It protects the local controller filesystem, not remote target commands.

An optional `verificationSeconds` reserves part of the existing sample window
for independent checking. Preparation and actor work use the earlier portion;
after an elapsed actor cutoff, the checker may evaluate the current kernel only
once all execution is settled. `actorTimedOut` records this cutoff separately
from the overall sample deadline. The reserve is frozen on resume and adds no
sample time. Unknown commands still block advancement and replay.

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

`continueMicroevalSamples` in `cli/skill-microeval-continuations.mjs` creates
separate diagnostic continuations for settled whole-sample timeouts. Its default
600-second allowance includes original elapsed work and reserves up to 180
seconds for final verification. Pass the original `originDirectory`, unchanged
`skills` and `profile`, a separate `directory`, and optionally selected
`sampleKeys`. The normal frozen-resume contract and primary decisions are not
changed. A successful initial recheck needs no actor; a definite candidate
failure may continue a fork of the original Pi session. An original trial
without a session requires explicit `actorStarted: false` evidence before first
actor admission. Unknown execution or reservation identities remain blocked.

Continuations use the same sample/verification hooks as the primary evaluator.
Their separate manifest pins each selected result, task, work snapshot and
session; appending unrelated primary results does not invalidate that identity.
Resume reuses the fork and remaining allowance. Original files and sessions are
never overwritten, and actor session totals are marked cumulative because they
include the fork's original history. Continuations do not promote skills or
recompute original A/B decisions.
The initial check defaults to 120 seconds; an explicit `precheckSeconds` may
lower that cap and is frozen on resume. Verification hooks receive their own
`signal`/`seconds` plus the full remaining `sampleSignal`/`sampleSeconds`; cached
target contexts must use the latter for lifetime and the former for each request.
A3 actor admission requires a negative receipt bound to the current kernel hash
with candidate-failure evidence. Missing receipts and infrastructure timeouts do
not justify another model call.
