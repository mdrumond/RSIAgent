# Campaign training evidence annex

`benchmarks.ascend.training_annex` inventories retained adaptive Phase 1 waves,
CAS objects and provider ledgers, native host evidence and measurements, and
existing Phase 2 cells, attempts, memory and snapshots. It preserves failed
attempts and infrastructure records as recorded; absent host evidence is a gap,
never an inferred kernel failure. It does not generate SFT examples from prose
summaries. Exact source versions, input/host verification records, trajectories,
timing semantics and profiler metadata remain in the original hashed files.

```python
from benchmarks.ascend.training_annex import inventory_campaign, package_campaign

manifest = inventory_campaign(campaign_root, artifact_store=report_store)
package_campaign(campaign_root, archive_path, artifact_store=report_store)
```

Inventory is read-only. The archive must be a new path outside the campaign,
normally under the orchestration workspace's ignored artifact directory.
Packages contain `manifest.json`, campaign-relative records under `campaign/`,
and verified raw profiler directories under `reports/<id>/`. Memory and evidence
travel together. Every file has a SHA-256 and byte size. Local absolute paths
are not archive member names; historical remote locations remain provenance.
Environment files, private configuration directories, credentials files, model
weights and reference repository bulk are excluded. Campaign evidence must not
contain embedded credentials.

`remote_reports` supplies transport descriptors: `id`, registry `target`,
`remote_path`, `expected_sha256`, `record_origin`, `kind="directory"`, and
`digest_algorithm="relative-path-sha256-v1"`. The downloader places the exact
directory contents at `report_store/<id>/`. The digest matches the native
profiler: hash the sorted sequence of each relative UTF-8 filename plus NUL and
the binary SHA-256 of its contents. A tar file digest is not interchangeable.
Descriptors retain their origin, even if several records reference one report.
The annex performs no remote I/O and never chooses targets from product labels.

Strict packaging rejects pending waves/attempts, missing publications, missing
or corrupt referenced CAS/snapshot/source files, and uncollected or corrupt raw
reports. A complete package describes a committed prefix; it need not mean the
campaign reached its requested project budget. `allow_pending=True` explicitly
creates a `pending-checkpoint` with all gaps listed, suitable for carrying an
unfinished continuation. No claim of complete evidence is made for that package.
Pause writers for packaging: changes to the selected inventory while copying
abort the archive and remove its partial output. Existing archives are never
overwritten.
