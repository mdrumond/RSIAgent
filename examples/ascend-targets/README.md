# Trusted target verifier

Run `python -m ascend_tools.targets --help` using the environment tool's selected
Python. This package uses no transport client. Freeze the verifier package and
case/runtime JSON outside the editable directory. Supply SHA256 digests of both
JSON files and the exact candidate file in the corresponding CLI arguments.
`--artifacts` must name a new directory outside the editable directory.

The CPU control ABI is `run(left, right)` and the case JSON is
`{"operation":"add","cases":[{"left":[1,-3],"right":[-2,7]}]}`.
Runtime JSON for this control has `target: "cpu"` and `python` equal to the exact
interpreter version. It provides numerical evidence only.

A3 Triton retains the existing `ModelNew.forward` ABI and frozen add, row_sum,
strided_copy cases (shape, layout, seed, dtype, case_id, atol, rtol). Runtime JSON
requires target `a3-triton`, exact Python 3.11 version, physical_device,
device_health, compiler_consumer_files and cann_metadata lists of {path,sha256},
expected_versions and module_sha256 maps for torch/torch_npu/triton, and
expected_compiler_target. Machine paths belong only in ignored environment
settings. The selected device is freshly checked before NPU imports and ordinary
commands use masked logical device zero. Health query timeouts are bounded.

The JSON stdout report separates compile, launch, correctness and performance;
raw Python/native stdout and stderr plus numerical tensors remain in artifacts.
Compile mode currently observes the ordinary dispatch path, including launch;
it makes no correctness claim. Performance is unmeasured and that mode fails
explicitly; elapsed wall time is not a device timing measurement. Failed studies
retain raw diagnostics without claiming hardware or language unsupported.

A5 Catlass uses `--target a5-catlass` with ABI version 1: the candidate exports
`kernel(a, b, out)` (or the frozen `entrypoint`), and the host owns allocation,
compilation and launch. The case JSON has `abi_version: 1`, `operation`, and
`cases` containing unchanged `length`, `seed`, `block_count` controls. Numerical
operations add/multiply/relu/convert retain the original independent oracle.
Physical padding to 64 elements is distinct from the logical length (1..512).

The runtime identity includes all common exact compiler/CANN file fingerprints,
Python3.11, physical_device, and device_health with product a5 and chip null.
Additional required fields: catlass_source, source_revision, dependency_gitlink,
dependency_revision, cann_version (9.3.x), cann_version_file,
cann_version_contents, pipeline_manifest, pipeline_manifest_sha256,
compiler_paths (TlaCompile/hivmc-a5), module_sha256 (catlass/torch/torch_npu), and
expected_versions (torch/torch_npu). Selected compiler files must be in
compiler_consumer_files with exact hashes. Paths are resolved exclusively from
ignored environment settings; the leaf never discovers or rebuilds dependencies.
A fresh per-run cache is prepared with `python -m catlass.bc_compile` before
compiling. The installed Catlass MLIR bindings must construct `ir.Context()`.

A5 compile mode stops after `tla.compile`; test mode separately launches,
synchronizes, retains tensors, checks unmodified inputs and compares numerical
outputs. A5 timing/profiling is not implemented and performance requests fail
explicitly as unmeasured. CPU contract tests cannot establish device behavior.
No old outer-CPL receipts or controller capabilities are consumed.

Explicit `device_health.allow_alarm: true` preserves user-authorized ordinary
Alarm-device tolerance and records degraded status; busy sharing separately
requires allow_busy and memory below 50%. Timing requires healthy idle regardless
of either opt-in. Unresponsive queries always exclude the device. Runtime import,
compile or launch failures remain retained observations, even after health
admission; they never establish a hardware or language unsupported claim.
