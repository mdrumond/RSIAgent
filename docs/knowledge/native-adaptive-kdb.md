# Adaptive KDB in native Phase1

The native A3 launcher accepts `adaptive_kdb: true` in its ignored runtime
configuration. `curriculum_kdb: true` independently enables curriculum access;
Actor retrieval follows the existing `--knowledge with-kdb` treatment. Keep both
true when both roles should consult the librarian. The default remains the
existing deterministic retrieval path.

The librarian always uses the canonical DeepSeek profile and its separate
credential. Actor model selection does not change that librarian model. Its
asynchronous HTTP request has SDK retries disabled and a total elapsed timeout
that cancels the request, in addition to the SDK transport timeout.

`librarian_seconds` defaults to90. One shared elapsed allocation covers all
queries and internal artifact reads for each Actor project, and another covers
all curriculum queries in one selection. Allocations persist in the branch or
selection directory, charge interrupted work on resume, and remain capped by
the enclosing caller deadline. Starting a new query never renews the allocation.
Curriculum selection directories bind wave index and pre-wave memory identity.

Reference selection continues to use `kdb` and `collection`. To expose an
immutable learned-memory export without copying the reference database, configure
all three additional fields:

```json
{
  "adaptive_kdb": true,
  "curriculum_kdb": true,
  "librarian_seconds": 90,
  "memory_kdb": "/configured/export/index.sqlite3",
  "memory_collection": "configured-memory-collection",
  "memory_snapshot": "/configured/export/snapshot"
}
```

Omit all memory fields for the reference-only first wave. Export the desired
retained memory and linked artifacts, index that snapshot into a separate
collection, and supply all three fields to a fresh second-wave root. Export and
publication are explicit; this integration does not automatically promote a
failed example or change an existing campaign's pinned collection selection.
Reference and memory stores must use the same embedding model/revision. Snapshot
bindings, collection fingerprints, DeepSeek identity and retrieval allocation
are part of the retained campaign identity, so a resumed root rejects changes.

Actor observations retain the existing compact source presentation. Full source
chunks, source indexes, status, gaps and selection reasons are admitted as
readable content-addressed artifacts. Curriculum keeps its existing cited-excerpt
protocol and receives selection status/reasons alongside the results. Full raw
librarian decisions remain in ignored per-branch/selection audit files.

`build_native_adaptive_phase1(..., librarian_complete=callback)` permits bounded
provider injection for tests. Live validation must call the actual DeepSeek
adapter and inspect both Actor and curriculum results before device campaigns.
