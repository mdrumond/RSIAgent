# Shared target launcher

Select the target once. Existing A5 command arguments and defaults are preserved:

```sh
python -m benchmarks.ascend.phase1_cli --target a5 run <existing A5 arguments>
```

Native A3 uses the same adaptive controller, Actor, host numerical oracle, and
progressive-memory loop, with native compiler/runtime and measurement bindings:

```sh
python -m benchmarks.ascend.phase1_cli --target a3 run \
  --config <ignored-native-runtime.json> --root <new-campaign-directory> \
  --objective '<learning objective>' --initial-memory <memory-file> \
  --model openai/gpt-5.6-sol --knowledge without-kdb \
  --profiling without-profiling-guidance --project-target 1 --max-seconds 300
```

Use `resume` with identical identity-bearing inputs. `report` needs only root,
objective, and initial memory; it makes no model or remote calls. No experiment
launch is implicit in installing or importing this launcher.

Keep runtime JSON outside version control. Required fields are `guide` (native
programming-guide file), exact `cpl_skills_revision`, discovered `target`/`arch`,
`execution_profile` (`bz-a3-1` or `bz-a3-2`), `cpl_remote`, `validation_wrapper`,
`remote_workspace`, `physical_device`, and `environment_identity` containing
the observed `cann_version` and `profiler_version`. Credentials belong in the
environment or ignored `--env-file`, never the runtime JSON or campaign output.
The transport must identify the user-wide `cpl-remote` and checked neutral
`catlass-validation.sh`; no direct SSH execution is substituted.

Optional `dtypes` defaults to `float32`; configuring a dtype does not certify
device support. KDB-on adds `kdb`, `collection`, and `bge_cache` and validates the
native collection identity. `--profiling new` requires `profiling_guide`, the
user-wide skill root; its shared instructions and A3 reference are frozen into
the campaign identity. Guidance-off still permits host measurement actions.

Only the configured subset of `Basic` and `PipeUtilization` may be captured;
the current Actor profile action accepts `pipe-utilization`. Unsupported
memory/timeline diagnostics remain unavailable. Ordinary timing records repeated
launch-plus-synchronize samples, not instrumented kernel duration. Both initial
and measured output must agree with the host oracle. Timing sweeps reuse source
and recompile each shape; use `RSI_NATIVE_INPUT<i>_NUMEL`/`_DIM<axis>` and
`RSI_NATIVE_OUTPUT_NUMEL`/`_DIM<axis>` macros instead of hardcoded extents.
These describe physical allocations; corresponding `*_LOGICAL_NUMEL` and
`*_LOGICAL_DIM<axis>` macros exclude sentinel padding and define arithmetic bounds.

Campaign identities bind programming/profiling guide bytes, model, native target,
compiler architecture, toolchain/profiler versions, KDB treatment, and measurement
availability. Retained evidence additionally binds binary/source/input geometry,
selected device, replay settings, raw output, and report hashes. Real target
smokes are still required; fake-provider tests establish composition only.

## Reviewed native Phase 2 handoff

After a terminal native Phase1 run, a human/authorized reviewer supplies a review
with schema `ascend-phase1-usefulness-review-v1`, `cell_id` computed by
`benchmarks.ascend.phase2.source_cell_id(campaign_identity)`, and `memory_sha256`
from the controller's canonical final memory snapshot. Require explicit true
values for `accepted`, `useful`, `diverse`, `profiler_guidance_observed` and a
nonempty `rationale`. This does not curate or rewrite Phase1 memory. Admission
authenticates the actual published branch outcomes, source, and cited artifacts;
an A5 identity or uncommitted/pending wave is not a native source.

```bash
python -m benchmarks.ascend.phase2_cli --target a3 admit \
  --config /ignored/native-runtime.json --root /owned/phase2 \
  --phase1-root /owned/phase1 --initial-memory /owned/initial-memory.bin \
  --review /owned/review.json
```

Admission copies exact memory, source identity, publication records, and evidence
into an immutable destination snapshot. Change `admit` to `run` (and optionally
add `--env-file`) to execute the bounded three-attempt shared source/host-verifier
loop. Native Phase2 uses the same runtime JSON as Phase1 and checks the frozen
target/architecture, CANN/profiler identity, guide bytes, and KDB collection.
It never requires an A5 Catlass checkout. It uses the shared four-case fp16/bf16
attention suite; configuring those cases is not a claim the hardware passed.
Candidate source is Ascend C, complete raw output is host-graded, and failed
attempt diagnostics feed the next attempt. Committed evidence is reused on
resume; infrastructure failures leave the current semantic attempt uncommitted.

Omitting `--target` or selecting `a5` dispatches the existing A5 Phase2 CLI with
its original arguments and defaults. Old A3 vector-add entrypoints remain
separate compatibility targets.

## Adaptive Phase 2 curriculum composition

For fresh generic projects, construct `AdaptivePhase2Coordinator` from
`benchmarks.ascend.phase2_curriculum` with an admitted reviewed snapshot,
objective, `CurriculumAgent`, and the existing adaptive project executor. It
pins the snapshot identity and exact Phase 1 memory. Each selection sees reviewed
Phase 1 outcomes, previous verified Phase 2 learning, and semantic failure
summaries, then authors one new project through the existing closed grammar and
host oracle. `project_target` counts only verified Phase 2 publications; the loop
continues after a pass. Executor exceptions retain the pending project for resume.

Both `build_native_adaptive_phase1` and `build_adaptive_phase1` accept explicit
`reviewed_snapshot=` and optional `trace_hook=` arguments to select this mode
while preserving their target, model, guide, KDB and runtime composition. They
write a distinct Phase 2 campaign identity including the pinned snapshot ID.
Changed reviewed memory requires a new root. Existing Phase 2 attention CLI
entrypoints retain their compatibility behavior.

For either adaptive phase, `coordinator.continue_campaign(new_target,
reopen_terminal=True)` explicitly raises the retained target and reopens a
SATURATED or STALLED curriculum. Omit `reopen_terminal` when extending a campaign
that reached its target. Publications, selector history and prior stop decisions
remain retained. `resume()` alone never extends the target. A constructor target
that conflicts with retained state is rejected; use the continuation method.
The optional trace hook receives a deterministic event ID, verification flag,
summary and retained attempt directory after publication; exporters can also
inspect those retained directories after interrupted notification.


## Optional adaptive curriculum retrieval

The shared Phase 1/Phase 2 adaptive curriculum defaults to no direct KDB access.
For native A3, set `"curriculum_kdb": true` in the ignored native runtime config;
for the A5 adaptive CLI, pass `--curriculum-kdb`. Both require the existing
with-KDB treatment, configured collections, and pinned embeddings. Changing
curriculum access or its collection fingerprints requires a fresh campaign root;
historical default-off identities remain unchanged.

An enabled curriculum may issue `{"knowledge_query":{"query":"...","limit":5}}`
repeatedly before returning its normal decision. Retrieval is optional. The same
read-only citation gate used by actors validates excerpts. Selection session
history retains requests, excerpts and citations; `curriculum-knowledge.jsonl`
records queries and collection manifests. Retrieved examples are references,
not verification that a newly proposed experiment works.
