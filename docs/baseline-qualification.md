# Frozen PyTorch baseline qualification

`python -m ascend_tools.baselines describe --workload WORKLOAD` emits the source,
case and verifier hashes and complete case IDs required by a launch gate. The
seven IDs are `fa`, `gdn`, `bsa`, `kimi_delta_attention`, `lact_ttt_chunk`,
`sparse_lightning_indexer_grad_kl_loss_enhance`, and `tensor_program_vm`.

The four CANNBench Level 5 packs retain the upstream source, specification,
20 public cases and license at revision
`f0488d95b4d44a6a0da1b8cd99e91fbd06181962`. Original source hashes are verified
before import. JSON case/specification files are generated from those pinned
YAML files. GDN/BSA retain the original supplied reference and all 50 cases,
not a previous optimized kernel or result. Their provenance records identify
these as supplied references, without claiming an upstream URL or license that
was not supplied. FA uses the four frozen small smoke contracts (sequence lengths
17, 32, 65 and 33; dimensions 16 or 32; both causal modes), a new unoptimized
PyTorch SDPA baseline and an independent explicit FP64 attention reference.
Sixteen additional frozen FA cases cover B1/2, H4/8, sequence128/512/1024/2048,
dimension64/128, FP16/BF16 and both causal modes. These20cases form the declared
standardSDPA contract (no GQA extension), not an entire external FA suite.
The largest FP64 score tensor is B1/H8/S2048 (~256MiB); qualification is serial.

The supplied GDN generator's hardcoded NPU placement is adapted to CPU through
an AST placement rewrite. GDN/BSA single-case construction retains original
case-index seeds and avoids allocating all 50 input groups simultaneously.
No reference arithmetic or public case size is changed. The sparse indexer
uses its upstream input hook to construct legal indices. Small independent
checks execute during qualification and their exact checker hash is bound in each receipt. They cover Kimi's NumPy recurrence, LaCT's closed-form updates versus autograd,
indexer gradients versus finite differences, GDN token recurrence, BSA masks,
and VM data dependencies. LaCT/VM full FP64 oracle paths share upstream core
logic; they are not described as independent implementations.

After caller-owned device health/admission and runtime preflight, execute:

```
python -m ascend_tools.baselines qualify --workload WORKLOAD --target a3 \
  --identity runtime.json --output NEW_OR_RETAINED_DIRECTORY --max-seconds 7200
```

The caller supplies the registered runtime and masks one admitted physical device
as logical NPU 0. `runtime.json` must contain `fingerprint`, observed `torch`,
`torchNpu`, and `deviceName` fields. Keep machine configuration outside source.
Each case runs in a separately supervised process. Elapsed expiry terminates
and reaps that process group; restart retains completed cases and conservatively
charges unknown running time. A still-live prior process identity blocks resume.
The allocation cannot change on resume. The campaign supervisor separately owns
the aggregate 16-hour per-target qualification allocation.

Qualification checks every returned tensor, including final states, for shape,
finiteness, numerical agreement and input mutation. Level 5 thresholds are frozen
from their documentation/specification (Kimi CPU acceptance thresholds; documented
MERE/MARE for other tasks). No undocumented small-value fallback is introduced.
A threshold failure blocks qualification. NPU baselines returning CPU outputs
are rejected. Timing is five warmups and twenty synchronized call-wall latency
samples, with input construction outside timing; it is not a kernel-duration or
profiler-derived measurement. Require idle device isolation before using it.

A successful receipt is `qualification.json` with `format: 1`, workload, target,
three contract hashes, runtime identity, all numerical case results and finite
positive timing samples. `validate_receipt(receipt, expected)` checks a receipt
against expected hashes, target, runtime fingerprint and exact case IDs. It is
not a substitute for executing qualification or checking runtime provenance.
Local CPU checks cannot produce an NPU-qualified receipt. Launch requires all
seven workloads qualified separately on A3 and A5.

## Sparse KL precision compatibility

The owned loader applies `inputs.py:_sparse_kl_fp32` to the frozen Sparse Lightning
Indexer Grad KL Loss Enhance reference. The upstream NPU branch calculates in the
input dtype: FP16 rounds `clamp_min(1e-8)` to zero before `log`, producing nonfinite
loss and gradients. The adapter changes only native-device selection and calculation
dtype: NPU inputs retain the native NPU layout path, including FP32 inputs, and all
floating calculations use FP32. The benchmark returns each gradient in its
corresponding input dtype and the loss in FP32. The CPU oracle remains unrounded
FP32; its cached outputs also supply the cast same-precision comparison control.
There is no CPU fallback for NPU inputs. Vendored bytes, cases and tolerances remain
unchanged. The manifest declares this adapter; source and verifier hashes bind it.
Reduced CPU tests reproduce the native-branch FP16 failure and exercise the repaired
branch across FP16/BF16/FP32. They do not establish NPU behavior; fresh full A3 and A5
qualification is required before launch.

## Five-case Phase 2 campaigns

`ascend_tools.phase2_focus.prepare_focused_project` creates a Phase 2 project whose
iteration checks and final score cover exactly five cases. It ranks the target's
fully qualified baseline cases by descending median of the twenty synchronized
device-call samples, breaking ties by ascending case ID. A3 and A5 select their
own cases. The project instructions list the selected IDs and measured medians.

The protected two-file verifier bundle binds this selection to the complete
qualification receipt and records `evaluationScope`, including selected cases,
full qualified case count and a scope hash. Correctness must pass for all five;
performance requires geometric-mean paired speedup greater than one across those
same five cases. A focused pass is not a full-suite score. Numerical tolerances,
dtype/structure/placement checks, mutation checks and the five-warmup/twenty-sample
paired timing protocol remain unchanged. The original `prepare_project` API still
creates full-case projects. Complete qualification evidence remains reusable:
the focused selector/scorer lives outside the immutable baseline package.
