# A5 SFT trace and dataset contract

This host-side pipeline retains new Phase 1/2 interactions for supervised
training without treating model output as correctness evidence. A trace is an
ordered, hash-chained JSONL stream containing the full redacted conversation,
tool activity, candidate source, and host validation provenance. Capture starts
prospectively; historical summaries are not reconstructed into training data.

The pinned source inventory is `config/a5/sft-source-manifest.json`. It records
the repositories and revisions made available to exploration. It does not
vendor those repositories, authorize redistribution, or make CANNBench golden
implementations eligible as labels. Review each source license before sending
derived data to an external service.

Phase 2 production wires `TraceRecorder` across its provider/verifier seam so
raw messages are retained and the trace is sealed only after host validation.
Phase 1 production binds `DirectActorDriver(trace_event=...)` per project,
buffers every exact message without sharing state across projects, and seals the
trace with the final candidate and host-evidence digest only after verification.
Declare the provider tool schema in the trace
header when tools are used. Secret-like fields and values are redacted before
anything reaches disk. Validation must refer to the exact candidate and to a
retained evidence digest.

Validate the inventory and export a Fireworks chat dataset with:

```bash
python -m benchmarks.a5kernels.sft_dataset validate-sources \
  config/a5/sft-source-manifest.json
python -m benchmarks.a5kernels.sft_dataset build \
  --traces artifacts/sft/traces --output artifacts/sft/dataset
```

The builder accepts only a terminal host pass. A repair trace is eligible only
when its final candidate passes. Earlier assistant turns receive `weight: 0`;
the terminal verified source receives `weight: 1`. The Fireworks rows preserve
the declared top-level `tools` and optional `reasoning_content`. Exact and
formatting/comment-only candidate duplicates are removed, then each operator
family is deterministically split 90/10. `manifest.json` binds split hashes,
provenance, counts, repairs, and validation evidence.

Run at least three independent Phase 1+2 rounds. Stop after a round contributes
less than 10% as many novel verified examples **and** repair-pattern clusters
as the preceding round, or after five rounds. The resulting adapter and dataset are frozen before the 2x2
Base/H/W/W+H evaluation.
