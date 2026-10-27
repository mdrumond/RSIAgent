# Retrospective Phase1 evidence

`benchmarks.ascend.history_import.import_history(roots, store)` reads explicit
historical producer roots and copies their content-addressed artifacts into a
consumer-owned `ContentAddressedArtifacts` store. A producer root contains
`state.json`, `campaign-identity.json` and `waves/*/plan.json`. Pass retained retry roots as well as the
original roots. The importer never writes producer files, admits memory, or
executes projects.

The returned JSON-compatible inventory has `memories`, `failures`, `exclusions`
and `counts`. A memory candidate requires a published memory payload, a matching
frozen assignment and terminal host-verified outcome, exact terminal artifact
hashes, the corresponding source, a valid evidence ledger and retained artifacts.
Each candidate is a complete project memory, including records with no distilled
lessons. Its `review_request` reconstructs with `ProjectReview(**request)` for
independent review; historical correctness is not a usefulness judgment.

Failures retain frozen specs and the readable evidence from all coalesced
attempts. Contract identity includes sanitized target, architecture and toolchain facts and
omits only `project_id` from the spec: hypotheses remain significant
because they describe distinct original repair bugs. A verified attempt removes
that same-binding contract from recovery, even if publication was interrupted.
Unknown bindings remain isolated by an opaque fingerprint of the canonical
producer location; the raw machine path is never retained. Known target/toolchain
bindings still deduplicate after relocation. Sanitized campaign binding
facts are retained as an artifact; provider/model settings and machine paths are
not included. Exact copied
attempts do not create additional recovery tasks. Unverified and interrupted
branches never become memory candidates. Malformed failure journals remain
readable raw artifacts with warnings, not verified host evidence.

The output store retains source, memory, execution and Actor/tool logs under their
original SHA-256 identities. Once import completes, consumers do not need producer
paths. Configuration files, credentials, provider settings and unrelated files
outside the branch artifact store are not imported. Exclusions report missing or
inconsistent evidence; callers must keep these visible rather than treating them
as successful reviews or repairs.
