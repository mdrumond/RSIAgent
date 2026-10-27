# Optimizer compact evidence ingestion

Remote compact archives are imported with `JobStore.put_artifacts(job_id,
entries)`, where `entries` yields `(name, bytes)` pairs. The store authenticates
the job once, writes content-addressed blobs, then publishes all ordinary
`artifact` events in one atomic event-chain replacement. Individual SHA-256,
size, and name records remain compatible with existing jobs and consumers.

An interrupted import can leave unreferenced blobs; its inventory is not
partially committed. The worker validates every archive member before committing
the inventory and only then publishes its local `result.json` completion marker.
A digest error or interrupted import therefore cannot masquerade as a completed
operation on resume. Exact remote handles continue to be observed rather than
blindly resubmitted.

Host-only scale regression and reproducible before/after benchmark:

```bash
python -m pytest -q -s tests/test_optimizer_jobs.py tests/test_optimizer_worker.py
python -m tests.benchmark_optimizer_artifacts --count 10000 --legacy-count 100 --memory-mib 56
```

The regression asserts one job authentication and one chain rewrite for 10,000
artifacts with a 56 MiB submission; wall-clock timings are informational, not a
flaky test threshold. The benchmark runs the pre-batch algorithm at 100 files
as well as the batch path at both 100 and 10,000 files. It uses temporary host
fixtures only: no credentials, model calls, device work, or campaign restart.
