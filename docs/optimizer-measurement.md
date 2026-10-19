# Optimizer measurement interfaces

`optimizer.measurement.benchmark(fn, root, warmup=5, repeats=50, identity=None)`
captures a verified zero-argument callable using the retained CANNBot NPU
schedule. `identity` is a keyword-only JSON-compatible mapping supplied by the
caller: include source/binary hashes, exact kernel selector, case shape/dtype,
target/toolchain versions and load/frequency observations. The caller selects
the checked execution profile/device and prepares equivalent fresh inputs
outside the capture. Torch imports are lazy.

The result's `latency_us` is mean summed device-operator duration per invocation,
not host latency. Each operator retains raw durations, inferred and exact
launch counts, removed-row indices, threshold and discarded-prefix count.
The pinned CANNBot policy removes rows above max(median*50, 10us), then trims
the clean prefix to a multiple of repeats, sums each invocation segment and
averages. This can discard good rows after an outlier; that behavior is
deliberately retained and disclosed rather than silently changing comparison.

`optimizer.profiles.profile(fn, root, groups=None, identity=None)` captures
separate diagnostic replays. It records installed `msprof --help` and discovers
the Torch-NPU `AiCMetrics` enum; each capture confirms availability. Unsupported
groups and capture failures have explicit reasons. Counter values preserve
vendor names and units; missing values are null and measured zero is zero.
Ratios without documented units are not converted to percentages or bandwidth.

Full profiler trees remain at the remote `raw_report` locations. Only generated
compact JSON/CSV artifacts and manifests should be downloaded; no timeline
exports are included in the transfer manifest. Instrumented diagnostic times
are separate from the canonical CANNBot capture. Profiler diagnostics do not
establish correctness and never replace a successful correctness result.

Example on an already selected A3 runtime/device:

```python
import torch
import torch_npu
from optimizer.measurement import benchmark
from optimizer.profiles import profile

torch.npu.set_device(10)  # choose via checked profile, not a portable default
x = torch.arange(257, device='npu', dtype=torch.float32)
fn = lambda: x + 1
identity = {'case': 'add-257-f32', 'kernel': 'Torch add',
            'verification': 'caller must verify before benchmarking'}
timing = benchmark(fn, '/retained/job/baseline', identity=identity)
diagnostics = profile(fn, '/retained/job/baseline-profile', identity=identity)
```
