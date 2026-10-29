# A5 SFT trace and dataset contract

This host-side pipeline retains new Phase 1 interactions for supervised
training without treating model output as correctness evidence. A trace is an
ordered, hash-chained JSONL stream containing the full redacted conversation,
tool activity, candidate source, and host validation provenance. Capture starts
prospectively; historical summaries are not reconstructed into training data.
Candidate source is the exception to generic text redaction: its exact verified
bytes and SHA-256 are retained unchanged. Capture fails closed when source
contains a high-confidence credential signature; ordinary identifiers such as
`token = 0` are not secrets. Dataset export rejects any mismatch between the
retained source digest and the host-verified candidate digest.

Adaptive A3/A5 composition accepts an optional `trace_recorder_factory`, created
by `benchmarks.ascend.training_traces.recorder_factory(phase=..., round_id=...,
model=..., source_manifest_sha256=...)`. Each project gets a fresh recorder and
Actor observer under its own branch's `sft-traces`; concurrent projects share no
trace buffer. Write, failed compile, execution and infrastructure abort evidence
are retained. Each invocation has a fresh trace ID, so retrying the same project
preserves its earlier failed or infrastructure trajectory alongside the retry.
Only a host-verified submitted current source receives positive
supervision. Capture seals before terminal distillation, which remains in the
provider evidence annex. `supervised_source` preserves exact admitted source
bytes even for ordinary identifiers such as `token = 0`.

`SourceManifest.load` additionally accepts `ascend-sft-sources-v2` with the same
source fields and sorted unique names. This inventory lists repositories actually
observed for the run, including an empty list for a run with no repositories;
it does not require the historical eight-source inventory. Existing v1 source
manifests and trace files retain their contract.

For offline export, call `benchmarks.ascend.training_traces.export_annex(root,
output, artifact_store=...)`. The output must be outside the campaign. The
returned `annex-export.json` binds the inventory digest and lists accepted trace
paths, rounds and operator families, rejection reasons, annex gaps and the
existing dataset manifest. Sealed historical v1 traces are accepted only when
their exact source and terminal evidence bytes are also retained in the annex.
Raw historical ledgers without a sealed conversation/source/evidence binding
are reported ineligible; summary dialogue is never synthesized. Failed and
infrastructure attempts remain annex evidence without becoming positive labels.

The pinned source inventory is `config/a5/sft-source-manifest.json`. It records
the repositories and revisions made available to exploration. It does not
vendor those repositories, authorize redistribution, or make CANNBench golden
implementations eligible as labels. Review each source license before sending
derived data to an external service.

Phase 1 production wires `TraceRecorder` across its provider/verifier seam so
raw messages are retained and the trace is sealed only after host validation.
Legacy provider records created before prospective messages were added remain
resumable but deliberately produce no SFT trace.
Phase 1 production binds `DirectActorDriver(trace_event=...)` per project,
buffers every exact message without sharing state across projects, and seals the
trace with the final candidate and host-evidence digest only after verification.
Both phases add an explicit supervised user/assistant boundary containing the
exact verified source. Phase 1 distillation stays available for research memory
but is excluded from that training target. Host actions and prior failed project
attempts are retained as structured failure evidence so successful repairs are
classified from production traces, not inferred from prose.
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
formatting/comment-only Python candidate duplicates are removed. Native sources
with preprocessor directives or Ascend C markers use exact-source deduplication,
preserving directive values, case and distinct near-identical kernels. Each operator
family is deterministically split 90/10. `manifest.json` binds split hashes,
phase/model/source-manifest provenance, counts, stable repair-pattern IDs, and
validation evidence. A build rejects mixed source-manifest revisions; phases
and models are reported as deterministic sorted lists.

Run at least three independent Phase 1+2 rounds. Stop after a round contributes
less than 10% as many novel verified examples **and** repair-pattern clusters
as the preceding round, or after five rounds. The resulting adapter and dataset are frozen before the 2x2
Base/H/W/W+H evaluation.
The builder persists each round's two novelty counts and the resulting
`saturation_decision` in `manifest.json`, so rerunning the CLI reproduces the
same three-to-five-round stop decision.
