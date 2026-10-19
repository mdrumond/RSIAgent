# Optimizer-backed Phase 2

Fresh autonomous A3 Phase 2 starts use executable optimizer submissions for both
the final goal and focused practice. There is no ProjectSpec translation and no
native reference Actor. The curriculum's deterministic Torch reference is frozen
by the existing remote optimizer worker; native candidates must pass its suite.

Use the existing `run_ascend_campaign.py start --target a3 --phase 2` entrypoint
with `--project-target 1`, reviewed Phase 1 root/review, and initial memory.
Ignored campaign settings additionally name `optimizer_worker_config`, an absolute
path to the existing optimizer worker JSON documented in `optimizer-worker.md`.
That file selects the checked A3 profile/runtime/device, guide, provider and KDB.
Its hash is pinned at start. Runtime/model/guide/KDB treatment must agree with the
reviewed campaign composition. No machine paths belong in checked-in examples.

The default budgets remain three target attempts and up to two practices between
targets. Each attempt evaluates one native revision; compilation probes do not
consume this budget. Correct practice never substitutes for a final target retry.
Target retries reuse the same frozen suite. Host verdicts alone establish
correctness; speedup is optional, and instrumented profiling stays separate from
canonical timing. Full feedback is retained even when provider context is bounded.

State is versioned as `ascend-optimizer-phase2-v1` (`phase=optimizer-phase2`).
`seed-memory.json` authenticates the reviewed seed; `attempts/` records committed
target/practice outcomes; `tasks/` retains optimizer jobs, sources and evidence.
Successful host facts and explicitly labelled model interpretations enter memory;
failed attempts remain curriculum feedback. Resume keeps exact task identities,
attempt IDs and remote handles. Infrastructure uncertainty never authorizes a new
submission. Budget changes require a fresh campaign.

Retained legacy campaigns keep their original implementation. Explicit legacy
`--target-goal` contracts and A5 entrypoints are unchanged. New A3 validation does
not establish A5 compiler/device support for this optimizer path.
