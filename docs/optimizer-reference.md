# Torch reference worker contract

`optimizer.cases` validates serializable submissions without running source.
`optimizer.reference` executes source **only in a caller-managed isolated worker**;
it is not a sandbox and must never be invoked inside the dashboard server.

```json
{
  "source": "import torch\ndef kernel(x, scale): return {'out': x * scale}",
  "entrypoint": "kernel",
  "signatures": [{"args": [
    {"shape": [32], "dtype": "float32"},
    {"scalar": 2}
  ]}]
}
```

For a Module entrypoint, set `entrypoint` to the class name and supply optional
JSON `init_args` and `init_kwargs`. Parameters and buffers are initialized once
under a recorded seed, moved to the selected device, frozen, and retained.
Inference runs use `eval()` and `torch.inference_mode()`.

Each signature accepts positional `args` and named `kwargs` containing tensor
hints or JSON scalar hints. Tensor hints have positive `shape` dimensions and
`dtype` (`float32`, `float16`, `bfloat16`, `float64`, integer types, or `bool`).
Optional `strides` preserves positive non-overlapping layouts; `layout` can be
`contiguous` or `transposed`. Optional `distribution` is `normal`, `uniform`,
`zeros`, `ones`, or `alternating`. Uniform/integer generation accepts `low` and
exclusive `high`. Per-dimension `constraints` may specify `min_shape`,
`max_shape`, or `multiple_of`. A tensor/storage is bounded to 16M elements.

`generate_cases(spec, max_cases=24, seed=0)` retains every representative then
adds small/odd/alignment-boundary and value cases deterministically. Equal input
dimensions change together to preserve common broadcast/matmul relationships.
It returns `cases` and explicit constraint-rejected variants. Every case retains
its reason, seed, ID, and signature SHA256; no supplied representative is dropped.
`materialize_case(case, device="cpu")` returns `(args, kwargs)` using a private CPU
generator, then transfers inputs before capture.

`prepare_reference(spec, device="cpu", max_cases=24, seed=0)` returns a
`PreparedReference` with `spec`, `model`, `cases`, `manifest`, and `invoke`.
Cases contain their signature, materialized arguments, and full nested outputs.
Input/module mutations, random/nondeterministic outputs, backward, unsupported
outputs, and failed representatives raise `ReferenceError`. Shape-incompatible
generated variants are retained as rejected with their error; valid variants
remain. Repeated reference outputs must be bit-exact, including NaN payloads.

`compare_outputs(expected, actual, inputs=(), non_compute=False)` checks every
leaf and exact container structure, tensor shape, and dtype. Pass both positional
and keyword tensor inputs in `inputs` for correct integer/quantization dispatch.
It returns structured leaf metrics or raises `ReferenceError` with metrics.
Floating comparisons adapt CANNBot's dtype-aware three **AND** checks:
maximum error cap, matched ratio >= 0.9, and mean relative error below the dtype
threshold. NaN locations and infinity locations/signs must match. Integer inputs
require exact integer output; floating-to-integer allows absolute difference <=1;
bool is exact. `non_compute=True` requires identical tensor bits.

Policy provenance: Huawei CANNBot `verify.py`, CANN Open Software License
Agreement v2, retained source SHA256
`0d80e9e78f8e0c7bc7861dfb2bd5dc9058103ffe5c187db098a8b36e2252b429`.
Thresholds and provenance are pinned in every baseline manifest.

`freeze_reference(prepared, directory)` writes new `reference.py`, `baseline.pt`,
and `manifest.json`, refusing overwrite. The manifest pins source, suite,
comparison policy, model state, full per-case input/output identities, and the
serialized baseline hash. Downstream evaluation must authenticate these before
loading tensors and never rewrite the bundle to accommodate a candidate.

Standalone CPU worker proof:

```sh
python -m optimizer.reference submission.json --output baseline --device cpu
```

The optimizer runner owns subprocess limits, durable logs, device selection,
transport, admission, and later CANNBot/profiler capture. This module does not
start remote workloads or claim CPU verification proves Ascend C correctness.
