# Triton ProjectRunner binding

`rsi.triton_project_config.project_config` freezes compile, test and validation
commands for a real Triton candidate. The source folder contains `candidate.py`
with `ModelNew.forward`; host-owned correctness and runtime JSON files live
outside that folder. Only `candidate.py` is editable. This binding currently
supports add, row sum and strided copy foundation cases.

Use the shared `rsi.cpl_execution.CplExecutor`, not another transport. Construct
its `CplConfig` from the ignored registered profile and current device admission.
Include frozen cases, runtime manifest and any exercise evidence directory in
`trusted_resources`. `command_cwd` is the existing admitted listener/runtime
subdirectory; `workspace` is the absolute staging base. Select transfer flags
from the actual backend configuration. The interpreter argv must be compatible
with the selected named runtime. The factory returned by `executor_factory`
resolves only the explicitly supplied opaque binding IDs and works on reopen.

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
admitted SoC. An alias is never used to infer a compiler architecture. Main must
supply fresh admission; this driver does not reserve or diagnose shared devices.

Compile executes the first frozen case, causing actual Triton JIT compilation or
cache reuse and a native launch. Test and validation execute every case. A
scoped JIT run hook records returned dispatches and available compiler outputs;
warmup-only calls do not count. The hook is restored after failure. Per-operation
compiler caches and generated artifacts stay under the declared output folder.
Numerical equivalence without an observed Triton dispatch fails.

Validation additionally checks the contract's `required_evidence`. Compilation,
launch, numeric verification and AST admission are produced by the driver. Other
exercise evidence must be supplied from trusted host producers in an immutable
manifest. That manifest binds `source_sha256`, `cases_sha256`, and
`runtime_manifest_sha256`; its `artifacts` entries carry relative `path`, `sha256`
and `producer_operation_id`. Artifact paths must stay inside its staged evidence
directory. Missing tools, stale source identity or changed artifacts fail
validation; a candidate-written status is not evidence. Native profiler commands
are separately frozen inputs and must not use controller `profile=True`, which
inserts Python cProfile rather than native msprof.

The shared executor owns CPL handles, submission receipts and reconciliation.
Unknown exceptions stay pending; callers reconcile retained work instead of
submitting again. This binding's source and unit tests do not establish device
correctness until the admitted target has actually compiled and run the cases.
