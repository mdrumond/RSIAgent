# Public ranked CANNBench references

Archive A3 (`910c`) per-operator rankings and the highest-ranked **available,
distinct** implementations:

```bash
python -m benchmarks.a5kernels.cannbench_sources \
  --root /path/to/ignored/artifacts/cannbench --workers 6
```

The default benchmark is `official-tasks` version `1.1.2`, with three requested
implementations per operator. `--operators exp gelu` limits a diagnostic run.
No credentials are read or sent. Downloaded code is never executed.

The downloader first archives complete paginated rankings, then examines
candidates in descending published order. It follows source pointers actually
published in public job/submission metadata. An authentication requirement is
an unavailable public source, not permission to access private storage.
Unrelated public implementations are not substituted for ranked submissions.

`inventory.json` contains integration-ready `sources` and operator coverage.
Each source records rank, published score, job/submission identity, original
download URL, bundle digest, extracted-file hashes, local root, and any bundled
license files. An absent license is reported as an empty list, not an inferred
license grant. `revision` is the downloaded archive SHA-256, not a Git commit.
Text wrappers, headers, and licenses remain with kernel code. Duplicate code
for the selected operator does not fill another of its three slots.

`responses/` retains URL, timestamp, HTTP status, body digest, and successful
response bodies; `operators/` records every examined rank and skipped-source
reason. Resume with the same command and artifact directory to reuse those
snapshots. Successful and authorization-denied responses are cached; use a new
artifact directory for a new acquisition snapshot. Transport failures are
retried on resume. An interrupted metadata fetch must not be interpreted as
exhausting the available ranking.

Coverage shortages are expected when source downloads require authentication.
Inspect each operator's `shortfall`, `exhausted`, and optional `error`; zero
process status alone does not mean three sources were acquired for every task.

These references are explicitly `reference_exposed`. Subsequent evaluation on
CANNBench is not held-out. Leaderboard correctness/performance values are
upstream reports, not local hardware validation.
