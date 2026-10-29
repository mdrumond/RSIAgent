# Campaign training evidence annex

`benchmarks.ascend.training_annex` inventories retained adaptive Phase 1 waves,
CAS objects and provider ledgers, native host evidence and measurements, and
current Phase 1 projects, memory and reviewed snapshots. It preserves failed
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

A5 CAS and inline capture payloads retain a full `remote_tree` plus a compact
archive SHA-256. The compact hash is never reused as a raw directory hash.
Such descriptors initially have `expected_sha256=None`,
`status="digest-unavailable"` and `digest_origin="unavailable"`. The collector
may query the remote tree's current digest before transfer, verify the downloaded
tree, then publish `report_store/<id>.receipt.json` with `descriptor_id`,
`source_sha256`, `downloaded_sha256`, and `digest_origin="collection-time"`.
Only when both receipt hashes match the local tree does inventory report it
available, with `resolved_sha256` and `digest_origin="collection-time"`.
The descriptor ID and absent capture-time digest remain unchanged; the receipt
travels in the package. Native records with an original report digest retain
`digest_origin="capture-time"`. A5 targets come from recorded execution profile
metadata or the enclosing campaign identity; absent routing metadata remains a
gap.

Strict packaging rejects pending waves/attempts, missing publications, missing
or corrupt referenced CAS/snapshot/source files, and uncollected or corrupt raw
reports. A complete package describes a committed prefix; it need not mean the
campaign reached its requested project budget. `allow_pending=True` explicitly
creates a `pending-checkpoint` with all gaps listed, suitable for carrying an
unfinished continuation. No claim of complete evidence is made for that package.
An unverified single-project wave may be closed with `closed-failure.json`
containing its wave fingerprint and raw outcome SHA-256. Inventory authenticates
that marker, branch identity, memory delta and execution evidence before treating
the failure as complete retained evidence. It never becomes a verified
publication; an existing `pending.json` still makes the package pending.
Pause writers for packaging: changes to the selected inventory while copying
abort the archive and remove its partial output. Existing archives are never
overwritten.
