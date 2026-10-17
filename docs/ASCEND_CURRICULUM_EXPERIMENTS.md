# Ascend curriculum experiments

The adaptive curriculum starts with `--project-target 64`. Increase the target
on `resume` to continue the same campaign; published memory and project identity
remain intact. Every wave contains at most eight projects, bounded by the number remaining,
so later waves can adapt to measured outcomes. A normally completed attempt
without a host-verified current source is skipped, not retried on resume. The
wave still waits for every sibling, then publishes only verified projects;
failed attempts do not count toward the verified project target. Provider,
context-budget, transport and infrastructure exceptions leave pending work
`RECOVERY_REQUIRED` rather than being mislabeled as kernel failures.
`TARGET_REACHED` means the verified project count was
reached, not that every requested topic was learned. Inspect `coverage_gaps`.

`report` includes the full `curriculum_memory`: concise lesson claims, explicit
`no_new_knowledge` outcomes, and completed failures. Each curriculum request
retains concise yield summaries for all prior outcomes and at most eight
detailed feedback entries: up to four recent failures and the remaining slots
for recent verified learning. Old failures cannot crowd out new lessons or
`no_new_knowledge` results, and selection context stays bounded.
Failure feedback includes retained source/evidence pointers, Actor status and
actual action counts and usage when available. A research-only budget exhaustion
does not establish a compiler or device limitation. Failed feedback is kept in
`state.json` separately from verified kernel memory, and the portable annex
retains its raw branch evidence. Completed project IDs cannot be reused, but a
smaller or changed experiment in the same operation family with a fresh ID is
allowed. These are advisory lessons for the curriculum, not hard family bans.

The saved curriculum session retains the complete selection chronology for
audit and resume; it is not a verbatim provider request transcript. Each new
selection sends only the system instruction and current authoritative host
context (including all prior outcomes, current coverage, and memory identity),
plus any validation corrections from that selection. Obsolete full contexts
from earlier selections are not resent. The shared provider request-size guard
still rejects oversized current contexts rather than silently truncating them.

Each hypothesis describes five parts: **Question; Controlled change; Tools;
Measurements; Expected lesson**. Text must be nonempty and trimmed, with no
arbitrary 500-character cap. Model response budgets still apply. Optional
`learning_focus` differentiates genuine experiments with the same mathematical
graph; existing saved projects without it retain their identities.

Coverage dimensions and coarse shape classes guide selection; they do not
require each project to introduce a new coverage token. New exact shapes within
a covered class and new combinations of known dimensions are allowed. Exact
project-information repeats of verified prior projects or wave siblings remain
rejected, including changes only to IDs, hypothesis wording, or launch tuning.

For example, a copy graph can first investigate `pipe-mte2`, then a distinct
`inter-pipe-sync`/`sync-performance` project can compare minimal correct ordering
with correct redundant barriers. A `sync-correctness` project instead diagnoses
and repairs an observed mismatch. Fixing the source requires new host validation;
old timing/profile evidence remains available as comparison evidence and cannot
satisfy final-source submission gates.

The curriculum covers SCALAR, VECTOR, CUBE, MTE1/2/3, FIX, legal producer/consumer
ordering, SIMD local-memory barriers, uniform SIMT barriers and Cube/Vector
cross-core handshakes. `msprof`, `msdebug`, `mssanitizer` and `tla.print` serve
specific experiments. Diagnostics and print-enabled sources run separately from
performance measurement. The host oracle remains the correctness authority.
Selected FP8/BF16/E8M0 values must be converted to printable FP32 or storage bytes;
check actual Catlass API restrictions for GM/UB/L1/L0C print views.

`request-timing` supports the current matrix shape without `timing_lengths`;
nonempty timing lengths remain a one-dimensional sweep. Output separates Python
launch-plus-synchronize latency from msprof OpBasicInfo instrumented device task
durations. Device rows are retained individually because AIC/AIV tasks may overlap.
Missing unit-bearing duration columns are explicitly unavailable. No timeline gap
is reported as a synchronization wait without explicit common-clock wait events.

FP8 E4M3FN/E5M2 and MXFP8 projects use supported low-precision runtime types.
MXFP8 uses separate E8M0 scales per 32 K elements and FP32 accumulation/reference
results based on quantized operands and scales. Start from small conversion,
movement and Cube exercises before tuning scale-bearing loads.

## Illustrative 64-project roadmap

This is an example campaign, not a fixed execution list. Each row is one project;
source variants live inside that project. Adapt later waves to observed failures
and lessons. Suspected hanging cases require a supported bounded route and are
not prerequisites. Every performance comparison verifies both sources first.

| # | Area / question | Dtype | Controlled comparison | Tools / evidence |
|---|---|---|---|---|
| 1 | Movement: MTE2 contiguous load | float32 | aligned vs tail GM→UB copy | msprof Memory |
| 2 | Movement: MTE3 output store | float16 | aligned vs padded UB→GM store | msprof Memory |
| 3 | Movement: SCALAR indexing | float32 | one vs multiple tiles, same output | msprof PipeTimeline |
| 4 | Movement: VECTOR conversion | bfloat16 | native vs FP32 intermediate conversion | tla.print + host oracle |
| 5 | Movement: MTE2 buffering | float16 | correct single vs double buffer | msprof PipeTimeline |
| 6 | Movement: MTE3 buffering | bfloat16 | correct immediate vs buffered store | msprof PipeTimeline |
| 7 | Movement: Tail addressing | float8_e4m3fn | aligned vs guarded tail copy | mssanitizer memcheck |
| 8 | Movement: FP8 storage lanes | float8_e5m2 | full lanes vs packed lanes, identical bytes | tla.print FP32 conversion |
| 9 | Vector: VECTOR add | float16 | one vs two vector tiles | msprof PipeUtilization |
| 10 | Vector: VECTOR multiply | bfloat16 | native vs FP32 intermediate | host oracle + msprof |
| 11 | Vector: VECTOR relu | float32 | aligned vs tail execution | mssanitizer memcheck |
| 12 | Vector: FP8 conversion | float8_e4m3fn | representable values vs rounding boundaries | tla.print FP32 conversion |
| 13 | Vector: FP8 conversion | float8_e5m2 | representable values vs rounding boundaries | tla.print FP32 conversion |
| 14 | Vector: FP8 VECTOR add | float8_e4m3fn | decode/compute/encode vs native supported path | host oracle + msprof |
| 15 | Vector: FP8 VECTOR multiply | float8_e5m2 | saturation inputs vs ordinary inputs | host oracle + mssanitizer |
| 16 | Vector: VECTOR buffering | float32 | correct serial vs overlapping copy/compute | msprof PipeTimeline |
| 17 | Cube pipes: MTE1 operand movement | float16 | one vs two K tiles | msprof PipeTimeline |
| 18 | Cube pipes: CUBE tile sizes | float16 | small vs larger legal tile | msprof PipeUtilization |
| 19 | Cube pipes: FIX output | float16 | FP32 accumulator conversion/store alternatives | msprof PipeTimeline |
| 20 | Cube pipes: MTE2 L1 staging | bfloat16 | single vs double L1 staging | msprof Memory |
| 21 | Cube pipes: CUBE FP8 | float8_e4m3fn | single vs multiple K tiles | host oracle + msprof |
| 22 | Cube pipes: CUBE FP8 | float8_e5m2 | tile/layout choices with identical quantized inputs | host oracle + msprof |
| 23 | Cube pipes: MTE1 reuse | bfloat16 | reload vs correct reuse of L0 operands | msprof PipeTimeline |
| 24 | Cube pipes: FIX tile drain | float32 | per-tile vs buffered correct output | msprof PipeTimeline |
| 25 | Inter-pipe correctness: MTE2→VECTOR dependency | float32 | diagnose missing producer ordering then repair | mssanitizer racecheck |
| 26 | Inter-pipe correctness: VECTOR→MTE3 dependency | float16 | diagnose stale store then repair | mssanitizer racecheck |
| 27 | Inter-pipe correctness: MTE2→MTE1 dependency | float16 | repair staged operand readiness | msdebug + host oracle |
| 28 | Inter-pipe correctness: MTE1→CUBE dependency | bfloat16 | repair operand consumer readiness | mssanitizer synccheck |
| 29 | Inter-pipe correctness: CUBE→FIX dependency | float16 | repair accumulator handoff | msdebug + host oracle |
| 30 | Inter-pipe correctness: Buffer reuse | float32 | repair overwrite-before-consume | mssanitizer memcheck/racecheck |
| 31 | Inter-pipe correctness: Flag pairing | float16 | repair ordinary unpaired flag case without deliberate hang | mssanitizer synccheck |
| 32 | Inter-pipe correctness: Tail initialization | float8_e4m3fn | repair uninitialized padded lanes | mssanitizer initcheck |
| 33 | Inter-pipe performance: MTE2→VECTOR ordering | float32 | correct minimal vs redundant barriers | msprof PipeTimeline + timing |
| 34 | Inter-pipe performance: VECTOR→MTE3 ordering | float16 | correct minimal vs redundant barriers | msprof PipeTimeline + timing |
| 35 | Inter-pipe performance: MTE2→MTE1 ordering | bfloat16 | correct minimal vs redundant flags | msprof PipeTimeline + timing |
| 36 | Inter-pipe performance: MTE1→CUBE ordering | float16 | correct minimal vs redundant flags | msprof PipeTimeline + timing |
| 37 | Inter-pipe performance: CUBE→FIX ordering | float16 | correct minimal vs redundant ordering | msprof PipeTimeline + timing |
| 38 | Inter-pipe performance: Double-buffer waits | float32 | correct one vs two buffers | msprof explicit wait events |
| 39 | Inter-pipe performance: FP8 movement overlap | float8_e5m2 | correct serial vs overlapped scheduling | msprof PipeTimeline + timing |
| 40 | Inter-pipe performance: SCALAR issue overhead | float32 | correct per-element vs tiled control | msprof PipeTimeline + timing |
| 41 | Local synchronization: SIMD local memory | float32 | repair store/load ordering with local_mem_bar | mssanitizer + host oracle |
| 42 | Local synchronization: SIMD barrier cost | float32 | correct minimal vs extra local barriers | msprof + timing |
| 43 | Local synchronization: SIMT shared UB | float16 | repair producer/consumer sync_threads | mssanitizer racecheck/synccheck |
| 44 | Local synchronization: SIMT barrier cost | float16 | correct minimal vs extra uniform barriers | msprof + timing |
| 45 | Local synchronization: SIMT reduction tree | float32 | repair intermediate-stage readiness | tla.print + host oracle |
| 46 | Local synchronization: SIMT reduction scheduling | bfloat16 | two correct reduction schedules | msprof + timing |
| 47 | Local synchronization: SIMT tail mask | float8_e4m3fn | repair uniform barrier participation with tail lanes | mssanitizer synccheck |
| 48 | Local synchronization: SIMT thread count | float32 | correct smaller vs larger thread group | msprof + timing |
| 49 | Cross-core synchronization: Cube→Vector handoff | float16 | repair ready-flag ownership | msdebug + host oracle |
| 50 | Cross-core synchronization: Vector→Cube acknowledgement | bfloat16 | repair buffer reuse acknowledgement | mssanitizer + host oracle |
| 51 | Cross-core synchronization: Two AIV consumers | float16 | repair mode-4 aiv_id0/1 handshake | msdebug + host oracle |
| 52 | Cross-core synchronization: Cross-core barrier cost | float16 | correct minimal vs redundant handshakes | msprof explicit wait events |
| 53 | Cross-core synchronization: Cross-core buffering | bfloat16 | correct one vs two shared buffers | msprof + timing |
| 54 | Cross-core synchronization: Cross-core final drain | float32 | repair final tile completion | tla.print + host oracle |
| 55 | Cross-core synchronization: Cross-core workload balance | float16 | two correct Cube/Vector tile ratios | msprof PipeTimeline + timing |
| 56 | Cross-core synchronization: Cross-core output tails | float8_e5m2 | repair consumer tail bounds | mssanitizer memcheck |
| 57 | MXFP8: MX E4M3 scale packing | mxfp8_e4m3fn | one vs multiple 32-K groups | tla.print decoded scales + host oracle |
| 58 | MXFP8: MX E5M2 scale packing | mxfp8_e5m2 | one vs multiple 32-K groups | tla.print decoded scales + host oracle |
| 59 | MXFP8: MX E4M3 padding | mxfp8_e4m3fn | even vs padded scale groups | mssanitizer memcheck + host oracle |
| 60 | MXFP8: MX E5M2 padding | mxfp8_e5m2 | even vs padded scale groups | mssanitizer memcheck + host oracle |
| 61 | MXFP8: MX E4M3 Cube tiles | mxfp8_e4m3fn | two legal tile sizes, identical quantized values/scales | msprof + timing |
| 62 | MXFP8: MX E5M2 Cube tiles | mxfp8_e5m2 | two legal tile sizes, identical quantized values/scales | msprof + timing |
| 63 | MXFP8: MX scale movement | mxfp8_e4m3fn | correct reload vs buffered scale-bearing loads | msprof PipeTimeline + timing |
| 64 | MXFP8: MX scale synchronization | mxfp8_e5m2 | repair scale readiness then compare correct minimal/extra ordering | mssanitizer then separate msprof + timing |

## Runnable MXFP8 project shape

MXFP8 supports only a single ordered matmul, without `padded-tail`; low-precision
Cube outputs are FP32. Plain FP8 permits movement/vector graphs as well, with
FP8 output only for copy-only graphs. Every non-copy FP8 graph widens operands,
intermediates and output to FP32. Scale tensors are host-owned and excluded from the
mathematical graph inputs. This strict project is accepted by the same parser
used for curriculum waves:

```json
{
  "project_id": "mx-scale-movement",
  "capabilities": ["cube-tiling", "profiling"],
  "shape": [16, 64, 16],
  "dtype": "mxfp8_e4m3fn",
  "block_count": 1,
  "padding_multiple": 1,
  "failure_mode": "none",
  "timing_lengths": [],
  "profile_metric": "timeline",
  "learning_focus": ["pipe-mte1", "pipe-cube"],
  "hypothesis": "Question: do buffered scale-bearing loads reduce operand waits? Controlled change: compare correct reload and buffered-load sources at fixed shape and quantized inputs/scales. Tools: msprof timeline and request-timing. Measurements: host error and per-task instrumented device duration for both sources, explicit wait events when available. Expected lesson: whether scale movement limits this tile.",
  "graph": {
    "inputs": [{"name": "a", "shape": [16, 64]}, {"name": "b", "shape": [64, 16]}],
    "nodes": [{"name": "out", "op": "matmul", "operands": ["a", "b"], "axis": null}],
    "output": "out"
  }
}
```
