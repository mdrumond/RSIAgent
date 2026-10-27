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
Group availability requires at least one finite counter belonging to that group;
ordinary task durations and cycle totals do not qualify. `counter_columns` names
the qualifying columns and `counter_schema` identifies the supported CANN 9.1
summary names and retained long-name aliases. Unknown or all-unavailable schemas
remain retained remotely with `available=false` and `no group counters exported`.

## Bounded worker diagnostics

Worker configuration optionally accepts
`"profiling": {"max_cases": 1, "groups": ["PipeUtilization"]}`. These are the
defaults: choose the frozen case with the largest total input tensor element
count (stable suite-order ties) and ask one pipe-composition question. Selecting
24 cases and all names in `optimizer.profiles.GROUPS` explicitly opts into the
old expensive coverage; installed capabilities still determine availability.
This is diagnostic sampling, not reduced correctness coverage: evaluation and
canonical timing still cover every frozen case. A separate candidate diagnostic
rechecks only its selected cases, returns `status=diagnostic`, and captures only
Ascend C because the Torch baseline was already profiled during preparation.
Coverage records include suite hash, total count, selected case IDs and groups.

Normalization and analysis run on the remote worker. Portable digests contain
requested-group counters with original kernel/row identity, units and values,
plus hashes and remote paths for the full raw/normalized reports. Native rows
are filtered by the compiled namespace's kernel symbol; missing matching
counters are unavailable, not zero. No cross-row ratio averages are invented.
Full profile tables and timelines stay remote, while canonical timing samples
remain portable. The Actor consumes digests and locally retained artifacts;
`read-artifact` is not a general remote-file fetch interface.

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
