# Triton ProjectRunner binding

`rsi.triton_project_config.project_config` freezes compile, test and validation
commands for a real Triton candidate. The source folder contains `candidate.py`
with `ModelNew.forward`; host-owned correctness and runtime JSON files live
outside that folder. Only `candidate.py` is editable. This binding currently
supports add, row sum and strided copy foundation cases.

Use the shared `rsi.cpl_execution.CplExecutor`, not another transport. Construct
its `CplConfig` from the ignored registered profile and current device admission.
Use `trusted_resources(config)` to include the frozen cases, runtime manifest,
standalone exercise producer, frozen tool-command profile and shared health probe. `command_cwd` is the existing admitted listener/runtime
subdirectory; `workspace` is the absolute staging base. Select transfer flags
from the actual backend configuration. The interpreter argv must be compatible
with the selected named runtime. Production uses Hubble's mandatory `ExecutorProfiles` admission/command guard
factory on create and reopen; the domain binding does not bypass it with a raw
executor factory.

The standalone `rsi/triton_verifier.py` is copied by ProjectRunner as its frozen
verifier. Each command invokes it through `{verifier}`, so the controller can
hash-check and remap that file. No controller package imports are needed in the
target interpreter. `project_spec` freezes the resolved test command and a
7200-second elapsed actor budget without iteration or execution-count caps.
A separate, explicit, once-only 7200-second harness recovery grant belongs to
the owning controller ledger; this module does not grant or reset it.

The frozen correctness JSON has `project_id`, `operation`, and nonempty `cases`.
Each case supplies `case_id`, `shape`, `dtype`, `seed`, `layout`, `atol`, and
`rtol`. `layout` is `contiguous` or `transpose`. Add takes two equal-shaped
inputs, row sum reduces dimension 1, and strided copy preserves logical values
in a contiguous output. Goldens are computed on CPU before candidate invocation;
there is no fallback to an NPU oracle. Shape, dtype, input mutation and numerical
results are checked for each frozen case.

The frozen runtime manifest contains the admitted `physical_device` and
`expected_versions` for `torch`, `torch_npu`, and `triton`. It can also contain
`soc` and `expected_compiler_target`. The driver masks the selected physical
device and addresses logical zero. It probes and retains the active Triton
compiler target and runtime device name separately from the registry alias and
admitted SoC. An alias is never used to infer a compiler architecture. Main supplies the actual version-matched health profile and Jason's shared
standalone probe. The driver executes that probe before Torch/NPU imports and
before each device command; it never substitutes an old main health receipt.
Device selection/reservation remains with the canonical host admission.

Compile executes the first frozen case, causing actual Triton JIT compilation or
cache reuse and a native launch. Test and validation execute every case. A
scoped JIT run hook records returned dispatches and available compiler outputs;
warmup-only calls do not count. The hook is restored after failure. Per-operation
compiler caches stay under `caches/triton/`, separately from retained artifacts.
Numerical equivalence without an observed Triton dispatch fails.

Each invocation retains its inputs, raw Python/native stdout and stderr, launch
and case events, available compiler outputs, tensor inputs/goldens/outputs, result
JSON and a hash manifest in a unique `outputs/triton/` subdirectory. Exceptions
preserve partial files and their original traceback before propagating failure.
No failure cleanup deletes these records. Stdout outside the captured log contains
only a small result/manifest reference, never an embedded binary archive. Frozen exercise command profiles, source fixtures and dynamically produced raw
reports remain in the retained tree on success and failure. The common
execution backend captures `retained_paths=('outputs/',)` into CAS on success and
failure; the binding opts into that shared implementation. Merely excluding
generated outputs from the source hash would not retain them. Caches
are excluded from retention.

Validation produces the contract's `required_evidence` for the current candidate.
It first runs native correctness in a trusted child that exits before profiler or
debugger work, avoiding an idle NPU context in the validation orchestrator. The
standalone producer is hash checked and loaded from a frozen trusted resource.
It executes compiler-recovery and missing-feature probes, derives a printf
instrumentation copy from the current kernel, captures matching IR, and dispatches
version-matched native tool commands. Fresh facts bind the current source, cases,
runtime and compiler target. Historical failure/source fixtures are immutable
inputs; no candidate-specific evidence manifest is frozen before editing.

`exercise_profile.tools` freezes tool argv and raw report contracts, not outcomes.
Each tool has `steps` (argv arrays and optional literal environment), optional
`probe_argv`, elapsed `seconds`, and `required_files` relative globs. Placeholders
are `python`, `candidate`, `verifier`, `application` (safely quoted child argv),
`workload` (trusted wrapper that forwards retained text logs), `output_dir`,
`project`, `physical_device`, and `logical_device`. Captures must invoke the
current application/workload; analysis steps may use owned read-only helpers.
Profiler CSV parsing also freezes `kernel_column`, `duration_column` and
`duration_unit`; it requires a positive measurement of the verified kernel.
The isolation record comes from the actual shared health probe immediately before
the command. Debugger profiles supply `source_location_regex`. Sanitizer profiles
run a trusted parser and supply `summary_json` plus `errors_field`; nonzero errors
fail. IR capture must retain actual TTIR, TTAdapter and last-pass MLIR. Unsupported
and unconfigured studies produce typed non-passing records, never fabricated
success. Native msprof does not use controller `profile=True`/Python cProfile.

The compiler failure input freezes `compiler_failure.project_path`, `sha256`,
`diagnostic`, and expected `returncodes`. Conversion freezes `ascendc_reference`
with the real Ascend C source path/hash and pinned commit. A target-specific
missing-feature fixture may use `feature_probe`; otherwise the producer measures
whether a documented API is actually absent and reproduces that absence. If all
frozen API probes are supported it reports the study as not applicable, rather
than inventing a missing API. An environment study requires a real, provenance-
bearing `historical_environment` input and compares it with freshly executed
shared health/fault/runtime observations and the current verified candidate.
No shared-device fault is induced.

The shared executor owns CPL handles, submission receipts and reconciliation.
Unknown exceptions stay pending; callers reconcile retained work instead of
submitting again. This binding's source and unit tests do not establish device
correctness until the admitted target has actually compiled and run the cases.


For parallel waves, `rsi.triton_worker_binding.prepare_worker_bindings` constructs
host-owned per-device variants before dispatch, with each device's actual public
metadata, runtime manifest, target interpreter and executor identity. Its returned
path/hash is the ignored `project_configuration`; `profile_fields` supplies
`project_binder = rsi.triton_worker_binding:bind_project` and that configuration.
The owning child calls the shipped `bind_project(request, logical_admission,
metadata=..., configuration=..., remaining=...)` after receiving its actual lease.
The hook selects by `request.device`, never project ordinal, checks all frozen
resources and logical workload identities, and creates only dispatch-specific
`request.root.parent/lease-inputs/`. It never creates the Runner root or baseline.
Runtime/case/producer/health files become read-only project execution inputs and
the verifier remains external, so the generic transport needs no mutable
per-worker resource profile. The returned spec command equals commands.test.
The parent validates the resolution using the common project_binding helpers.
Resume reuses the exact saved resolution and never calls the binder again.

The builder freezes the actual shared `device_health_probe.standalone_source()`
and `standalone_hash()` into a content-addressed host resource. Targets verify and
execute that stdlib-only source, calling local_probe with physical_device,
configuration and remaining. There is no target import of the controller package
and no literal old guard hash. Ignored device_health may explicitly allow Alarm
for ordinary functional work; its actual degraded measurements are retained.
Alarm permission does not imply busy permission. Profiler probes set require_idle
and cannot admit Alarm, busy or unknown devices. Probes occur once per native
command before imports; there are no per-case re-probes after creating our own
context. Numeric children exit before strict profiler admission.
