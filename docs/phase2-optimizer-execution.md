# Phase 2 optimizer execution

`OptimizerTaskExecutor(directory, task, memory_snapshot, config)` binds a selected
`{task_id, submission, rationale, citations}` to one retained optimizer job. The
submission is the existing Torch source/entrypoint/signatures format. Config is
the existing ignored A3 optimizer worker configuration; runtime paths are never
inferred from old campaign records.

Call `prepare()` to freeze the reference and deterministic cases. Then call
`attempt("target-001", memory_snapshot=updated_snapshot, feedback=diagnosis)`.
Each new attempt permits one evaluated native candidate, with compile repairs,
memory reads and gated KDB queries before it. Compile actions do not launch a
kernel or consume evaluated trials. The native ABI remains `optimizer_kernel`;
there is no ProjectSpec conversion or private native-reference Actor.

Reuse the same attempt ID and inputs after interrupted observation. Native
operations retain their directories and handles. Start a different ID only
after committing the preceding result. Infrastructure exceptions propagate and
are not semantic outcomes. Updated authenticated memory affects Actor context
without replacing the immutable seed snapshot or reference.

Evidence is stored under `directory/jobs/<job_id>`. `JobStore` authenticates the
submission/event chain and content-addressed artifacts. Returned `baseline`,
`candidate`, `profiling`, and `result_artifact` are artifact records; the latter
contains the canonical result excluding its own record. Candidate artifacts
contain the optimizer's JSON-encoded source string. Host comparison, not Actor
claims or speedup, determines `verified`; measurements remain source/suite bound.
The Phase 2 coordinator owns target/practice budgets, not the standalone
optimizer's default 25-trial search budget.
