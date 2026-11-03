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
requires target `a3-triton`, exact Python 3.11 version,
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

A5 Catlass extraction is a separate dependent slice from the preserved 2d23
branch. This CLI does not yet claim A5 support. It must retain its independent
three-tensor ABI, oracle, CANN 9.3/compiler/gitlink identity checks and real
compile/launch evidence, without the old outer-CPL admission handshake.

Device targets require `--physical-device "$ASCEND_PHYSICAL_DEVICE"` in the
host-frozen trusted verifier command. The domain wrapper supplies that value
from its opaque lease and forces both visibility masks; ordinary application
addressing is logical0. CPU controls require no device flag. Frozen runtime JSON
contains toolchain/product/compiler/package identity, never physical_device.
Identical cases/runtime/verifier bytes therefore serve physical0 and1 leases.
Explicit project device affinity belongs in a resource selection filter, never
in a rewritten verifier or runtime profile. Conflicting assignment/masks reject;
no logical-to-physical fallback or old-profile device alias is supported.
