# Target-driven Phase 2 contracts

`benchmarks.ascend.phase2_targets.qualify_target` executes a working reference
through the selected `ProjectExecutor`. It enables the fixed `ProjectSpec` only
when the existing quantitative host oracle passes, retaining the reference
source, complete output, inputs, and oracle in `ContentAddressedArtifacts`.
The runtime identity must pin target, named runtime, product, compiler,
toolchain, and device. Revalidate after any of these or the project changes.
The host owns this configuration; a curriculum proposal cannot qualify itself.

The contract fixes the complete project, including shape, dtype, deterministic
inputs, output shape, padding, and existing dtype tolerances. A passing reference
proves that this specific task works in that environment. It does not prove a
speedup is possible. Optional measured device durations require explicit units,
measurement conditions, and a retained measurement artifact. They are observations,
never required speedup thresholds. A new shape or format needs another reference
run. Keep development goals separate from held-out evaluation tasks.

The separate `TargetCurriculumAgent` receives the fixed goal, latest host verdict,
diagnostics, accumulated lessons, budgets, and available tools. It selects one
focused practice with an observed weakness, controlled change, evidence plan,
and an explanation of how the result informs a target retry. It may instead
request a target retry, report readiness after a passing target, or report a
blocked experiment. Practice success cannot establish target readiness.
Reference source is excluded from the curriculum prompt.

These interfaces do not change broad Phase 1 selection or legacy Phase 2 runners.
The coordinator owns retry ordering, memory updates, budgets, and durable resume.
