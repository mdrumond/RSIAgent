# Expanded reference KDB

The production index uses pinned BGE-small-en-v1.5 embeddings and SQLite FTS5
hybrid retrieval. It is not the keyword-only demo database. Downloaded source
is reference material, not proof that a kernel compiles, is correct, or is fast.

## Ingest

```bash
python -m benchmarks.a5kernels.expanded_kdb ingest \
  --inventory /absolute/ignored/reference-inventory.json \
  --db /absolute/ignored/references.sqlite3 \
  --model-cache /absolute/ignored/huggingface/hub --batch-size 32
```

The inventory has `schema_version: 1` and a `sources` list. Each available source
has `name`, `repository`, immutable `revision`, absolute local `root`, `status`
(`ok`, `available`, `downloaded`, or `complete`), `licenses`, and `files` entries
with relative `path`, SHA-256 `sha256`, and integer `bytes`. Optional metadata:
`document_language` (default `mixed`), `kernel_languages` (`ascend-c`,
`catlass-dsl`, `triton`), and `reference_exposed` (default false).

Only inventory files are read; their bytes, lengths, and UTF-8 encoding are
checked. Failed/unavailable sources are not indexed. Keep the original inventory
and acquisition report to distinguish unavailable upstream code from coverage.
Repository paths never become query-client dependencies.

Each source gets an immutable collection whose identity includes its verified
inventory metadata, excluding the machine-local root. Embeddings are generated
in bounded batches, with progress on stderr. A sibling `.staging` DB retains
completed collections after interruption. Resume using the same inventory and
output path; an interrupted collection is rolled back and retried. Changing the
inventory requires a new output path. Publication replaces the requested DB only
after all available collections succeed; historical DBs are never migrated.
The sibling `.inventory.json` report records the published collection identities.

Document language and kernel language are different metadata. An English
historical manifest does not silently become Ascend C. New native A3 compositions
accept an explicitly `ascend-c`-capable manifest; existing A5 manifests still load.
The retained English embedding model does not imply validated multilingual
semantic retrieval; Chinese source remains available through code identifiers,
literal text search, and measured retrieval behavior.

## Gated retrieval

```bash
python -m benchmarks.a5kernels.expanded_kdb query \
  --db /absolute/ignored/references.sqlite3 \
  --model-cache /absolute/ignored/huggingface/hub \
  --journal /absolute/ignored/query-journal.jsonl \
  --kernel-language ascend-c --limit 5 \
  --query 'How does the kernel allocate and synchronize unified-buffer copies?'
```

Omit `--kernel-language` for cross-language references. Repeat `--collection` to
select a fixed subset of named collections; selection is configuration, not an
Actor action. Retrieval uses the read-only `KnowledgeAgent` gate, validates chunk
identities, and journals citations. Each result includes code text, exact
repository/revision/path/lines, whole-source hash, chunk hash, and lexical/vector
ranks. Agents can answer using this interface without repository access.

CANNBench solutions carry `reference_exposed: true`. Evaluations that consult
these collections are reference-exposed, not clean held-out evaluations. Ranking
scores remain upstream claims, not locally verified performance.

Acceptance requires five fresh independent agents answering code questions from
the same frozen index, using only this gated interface, with citations checked
against the pinned source. A smoke query or a keyword hit alone is not that gate.
