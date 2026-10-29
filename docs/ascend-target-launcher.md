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

## Reviewed Phase 1 memory

Current original-project retries can seed ordinary Phase 1 execution with an
immutable reviewed snapshot. `benchmarks.ascend.reviewed_memory` authenticates
native snapshots; `reviewed_catlass` binds Catlass source treatments. Copies
include published branch outcomes, source, cited artifacts and exact review
bytes. Changed memory or runtime identities require a fresh root.

## Optional adaptive curriculum retrieval

The shared Phase 1 and subject adaptive curriculum defaults to no direct KDB access.
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
