# A5 knowledge corpora

The corpus specifications in `benchmarks/a5kernels/corpora` pin authoritative
upstream repositories to exact commits and allowlist English documentation and
examples by path and SHA-256. Repository snapshots and databases are local
artifacts and must not be committed.

Prepare the collections from the repository root:

```bash
python -m benchmarks.a5kernels.corpus \
  --spec benchmarks/a5kernels/corpora/catlass-en.json \
  --artifacts artifacts/kdb-sources prepare
python -m benchmarks.a5kernels.corpus \
  --spec benchmarks/a5kernels/corpora/ascendc-en.json \
  --artifacts artifacts/kdb-sources prepare
```

Index each prepared collection with the pinned, offline-only BGE backend:

```bash
python -m benchmarks.a5kernels.corpus \
  --spec benchmarks/a5kernels/corpora/catlass-en.json \
  --artifacts artifacts/kdb-sources index \
  --db artifacts/kdb/knowledge.sqlite3 \
  --manifest artifacts/kdb/catlass-en.manifest.json \
  --model-cache artifacts/huggingface/hub
python -m benchmarks.a5kernels.corpus \
  --spec benchmarks/a5kernels/corpora/ascendc-en.json \
  --artifacts artifacts/kdb-sources index \
  --db artifacts/kdb/knowledge.sqlite3 \
  --manifest artifacts/kdb/ascendc-en.manifest.json \
  --model-cache artifacts/huggingface/hub
```

Preparation performs the only network access: it fetches the exact commit into
a temporary Git repository, extracts only allowlisted blobs, verifies every
hash, and atomically publishes the source tree. Repeated preparation is
offline and fails if the local tree is missing, changed, or contains extra
files. Indexing never downloads a model and fails closed unless the pinned BGE
revision is already present in the selected cache.

## Cross-layer A5 reference (AscendC/architecture evidence)

`a5-ascendc-architecture-en.json` pins
[`huawei-cpl-zurich/data-movement-benchmarks`](https://github.com/huawei-cpl-zurich/data-movement-benchmarks/tree/23f39974d9175ea39ddad6e9fe84a7510ff8a56e)
at commit `23f39974d9175ea39ddad6e9fe84a7510ff8a56e`. It is a separate
`a5-ascendc-architecture-en-23f3997` collection, not an extension of the Catlass
collection. Citations retain this AscendC/architecture label through their
collection identifier. This material can inform cross-layer A5 hypotheses;
it does **not** establish Catlass lowering, correctness, or performance.

The deterministic allowlist contains the upstream README, four AscendC kernels,
the profiler parser, three campaign runners, and five A5 result metadata files.
These cover MTE2/Fixpipe overlap, prefetch, SIMT loads, strides, and on-chip vector
and cube compute. Every selected blob has an exact SHA-256 in the specification.
Notebooks, raw CSVs, bytecode, and PR/review documents are deliberately excluded.
Metadata preserves upstream result hashes and collection conditions, but a CSV
hash alone is not a locally retained or independently reproduced measurement.
Source scripts are reference text only; corpus preparation and indexing never
execute them.

Keep upstream distinctions when citing this collection: requested-byte versus
unique-byte bandwidth, AIC pipeline time versus AIV whole-kernel time, and
exploratory/unverified-isolation versus timing-grade evidence. The pinned
snapshot is the reference revision; individual metadata files also record the
older revisions used for their campaigns. Neither a branch name nor current PR
status substitutes for either provenance layer.

```bash
python -m benchmarks.a5kernels.corpus \
  --spec benchmarks/a5kernels/corpora/a5-ascendc-architecture-en.json \
  --artifacts artifacts/kdb-sources prepare
python -m benchmarks.a5kernels.corpus \
  --spec benchmarks/a5kernels/corpora/a5-ascendc-architecture-en.json \
  --artifacts artifacts/kdb-sources index \
  --db artifacts/kdb/knowledge.sqlite3 \
  --manifest artifacts/kdb/a5-ascendc-architecture-en.manifest.json \
  --model-cache artifacts/huggingface/hub
```

Retain the checked-in spec alongside the exported collection manifest: the spec
binds repository/commit/path/hash, and the manifest binds the indexed bytes and
embedding revision. Once prepared, verification, indexing, and querying require
no upstream network access or mutable PR state. Missing, changed, or extra local
source files fail verification; no checkout is retained in the source tree or
under `reference_repos`.
