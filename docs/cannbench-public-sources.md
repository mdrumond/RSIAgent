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
published in public job/submission metadata: ZIP bundles, raw UTF-8 code files,
and public HTTPS Git repository pointers. Git references are frozen to the
published revision when supplied or the resolved repository HEAD otherwise.
Anonymous Git acquisition disables credential helpers and interactive prompts.
An authentication requirement is
an unavailable public source, not permission to access private storage.
Unrelated public implementations are not substituted for ranked submissions.

`inventory.json` contains integration-ready `sources` and operator coverage.
Each source records rank, published score, job/submission identity, original
download URL, bundle digest, extracted-file hashes, local root, and any bundled
license files. Available sources declare `status: complete`, each file includes
its byte length, and `licenses` lists retained license paths (`license_files`
is a compatibility alias). Kernel languages are inferred from explicit code
markers, defaulting to `common` when unknown, never from the A3 target alone.
An absent license is reported as an empty list, not an inferred
license grant. `revision` is the downloaded ZIP/raw-file SHA-256 for non-Git
sources, or a frozen Git commit for repository pointers.
Text wrappers, headers, and licenses remain with kernel code. Duplicate code
for the selected operator does not fill another of its three slots.
Language classification uses only the selected operator's code, not other
operators retained from the same bundle. Generated provenance manifests live
outside extracted source roots, preserving upstream files with the same name.

`responses/` retains URL, timestamp, HTTP status, body digest, and successful
response bodies; `operators/` records every examined rank and skipped-source
reason. Resume with the same command and artifact directory to reuse those
snapshots. Successful and authorization-denied responses are cached; use a new
artifact directory for a new acquisition snapshot. Transport failures are
retried on resume, including HTTP408/429 and server failures. Unsupported source
formats are recorded explicitly and prevent a claim of exhausted availability.
An interrupted metadata fetch must not be interpreted as
exhausting the available ranking.

Coverage shortages are expected when source downloads require authentication.
Inspect each operator's `shortfall`, `exhausted`, and optional `error`; zero
process status alone does not mean three sources were acquired for every task.

These references are explicitly `reference_exposed`. Subsequent evaluation on
CANNBench is not held-out. Leaderboard correctness/performance values are
upstream reports, not local hardware validation.
