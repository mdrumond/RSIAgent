# Native optimizer worker

Run only in an isolated, bounded worker using the checked A3 profile. The worker
must select its actual NPU device before loading/creating the frozen reference.
`optimizer.native.evaluate(reference, candidate, directory, *, device, arch,
block_count=1, benchmark=False, profile=False, cache_directory=None)` accepts a `PreparedReference` or
an authenticated frozen reference directory. It never generates cases. Candidate
execution is arbitrary native code, so use one subprocess per candidate.

The evaluator compiles one case specialization, with a unique Torch dispatcher
namespace, per frozen case. `describe_case(prepared, row)` exposes the ABI contract.
Compile-only, evaluation, and profiling workers share a job-owned compiled cache
beside the frozen baseline. A hit requires the same candidate, full case contract,
architecture, block count, wrapper implementation, and observed toolchain/runtime
identity; both wrapper and binary hashes are checked before loading. Results record
`cache_key`, `cache_hit`, and the original binary/wrapper hashes. Torch registration
is reused within a process. Failed builds do not publish hits, and a job-level lock
protects concurrent compilation and eviction. The cache retains at most 2 GiB of
recent entries (plus a single current entry if that entry alone exceeds the limit).
Direct API callers can supply the same `cache_directory` to all three operations;
without it, a cache is created beside the case directories. Logs and wrappers remain
in each operation's evidence directory even on a hit; binaries remain remote.
The device entrypoint is:

```cpp
extern "C" __global__ __aicore__ void optimizer_kernel(
    GM_ADDR input0, GM_ADDR input1, /* tensor inputs, omitting scalar args */
    GM_ADDR state0, /* state_dict insertion order */
    GM_ADDR output0 /* tensor leaves, depth-first container insertion order */);
```

The host supplies `INPUT0_NUMEL`, `INPUT0_RANK`, `INPUT0_SIZE0`,
`INPUT0_STRIDE0`, equivalent `STATE`/`OUTPUT` constants, and numeric/bool/None
scalar constants `ARG0`, `ARG1`, etc. Keyword arguments follow positional args
in submission order; the contract records their names. Dtypes and state names
are in the JSON contract. Tensor storage retains strides; indexing must use the
declared layout. Outputs preserve reference shape, dtype, stride and nested
structure. Host-only allocations, dispatch and stream selection stay outside
candidate source. Compilation logs and exact commands remain per case.

Every case must pass the frozen CANNBot comparison policy before any timing or
profiling. Input/state mutation fails verification. Capture inputs are cloned
outside timing. Results retain each case contract, compiler/binary provenance,
correctness, optional Torch/Ascend C timings and speedup, and diagnostic captures.
Profiling errors remain unavailable without erasing successful correctness.
Only download compact artifact manifests; profiler report trees stay remote.

## Bounded elementwise runtime proof

In the checked A3 runtime (substitute the **discovered** architecture and device),
run this Python body. Scalar global-memory accesses deliberately prioritize a
small correctness proof over performance; it is not an optimized kernel.

```python
import json
import torch
import torch_npu
from optimizer.reference import prepare_reference, freeze_reference
from optimizer.native import evaluate

torch.npu.set_device(0)  # checked profile-selected logical device
spec = {'source': 'def kernel(x): return x + 1', 'entrypoint': 'kernel',
        'signatures': [{'args': [{'shape': [17], 'dtype': 'float32'}]}]}
prepared = prepare_reference(spec, device='npu:0', max_cases=3)
freeze_reference(prepared, 'proof-baseline')
candidate = r'''
extern "C" __global__ __aicore__ void optimizer_kernel(GM_ADDR input0, GM_ADDR output0) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<float> x, y;
  x.SetGlobalBuffer((__gm__ float*)input0, INPUT0_NUMEL);
  y.SetGlobalBuffer((__gm__ float*)output0, OUTPUT0_NUMEL);
  for (int64_t i = 0; i < INPUT0_NUMEL; ++i) y.SetValue(i, x.GetValue(i) + 1.0f);
}
'''
result = evaluate('proof-baseline', candidate, 'proof-candidate',
                  device='npu:0', arch='DISCOVERED_ARCH', benchmark=True)
assert result['status'] == 'verified', result
assert all(row['correctness']['passed'] for row in result['cases'])
assert all(row['timing']['torch']['latency_us'] > 0 and
           row['timing']['ascendc']['latency_us'] > 0 for row in result['cases'])
print(json.dumps(result, indent=2))
```

Reduction and matmul proofs use the same ABI with one or two input tensors and
the frozen output sizes. They must be executed on the real A3 target before
claiming device validation; CPU wrapper tests establish only host behavior.
