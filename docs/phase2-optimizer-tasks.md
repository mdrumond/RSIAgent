# Optimizer-format Phase 2 curriculum

`benchmarks.ascend.phase2_optimizer_tasks.select_optimizer_task` proposes both
final goals and focused practice in the existing optimizer submission format.
It returns `task_id`, `submission`, `rationale`, and host-observed KDB `citations`.
There is no conversion to the legacy operation graph or private native-reference
Actor. Validation here checks submission data only; the execution worker must run
and freeze the reference before learner execution.

```python
from optimizer.memory import freeze_reviewed_phase1
from benchmarks.ascend.phase2_optimizer_tasks import select_optimizer_task, reject_optimizer_task

seed = freeze_reviewed_phase1(admitted_phase1_snapshot)
task = select_optimizer_task(
    complete=provider_complete, snapshot=seed, objective=objective,
    root=selection_directory, query_knowledge=gated_kernel_search,
)
# On a semantic reference-worker rejection, retain diagnostic feedback:
reject_optimizer_task(selection_directory, "case-000: residual shape disagrees")
```

`complete(messages)` returns JSON text; `query_knowledge(query)` returns the
existing gated observation with `results[].citation`. For practice, pass
`kind="practice"` and `context` containing the target, diagnosed weakness,
prior evidence, and intended retry. Selection state and completed tool reads
survive provider interruptions. Three proposal attempts are shared with reference
rejections; infrastructure exceptions should propagate rather than be converted
to semantic rejections.

Memory presentation is bounded and references full immutable memories and source
artifacts by hash. `freeze_phase1_source(..., snapshot_root=...)` alternatively
performs existing reviewed native Phase 1 admission before export. Neither helper
requires or fabricates a completed Phase 2 campaign. Historical optimizer Phase 2
discovery remains unchanged.
