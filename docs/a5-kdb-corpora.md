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
