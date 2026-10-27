# Pinned source references

`reference-sources.json` identifies actual upstream GitCode snapshots, not
reconstructed examples. To acquire and inventory them without building code:

```sh
python -m benchmarks.a5kernels.source_acquisition \
  --config config/knowledge/reference-sources.json \
  --destination /path/to/ignored/sources \
  --inventory /path/to/ignored/reference-inventory.json
```

The inventory has schema version 1. Each completed source records its upstream
URL, immutable Git revision, local root, UTF-8 file allowlist with SHA-256 and
byte counts, license paths, language metadata, and excluded-file reasons.
Hashes and indexed text come from canonical Git blobs, independent of checkout
filters and line-ending conversion. Identical checkouts retain their paths;
transformed checkouts get a separate raw-source export without modifying inputs.
Local roots belong only in ignored inventories, not the checked configuration.
Generated directories, symlinks/submodules, binary and non-UTF-8 files are not
indexed. Clones retain upstream licensing and complete repository context.

Use `--reuse NAME=/path/to/clone` to reuse an existing **upstream clone** after
checking its origin. This creates an isolated snapshot, never modifies the
original clone, and never copies uncommitted reconstructed material. Pinned
snapshots can be resumed offline; modifications fail validation. Acquisition
retains successes and reports failed sources rather than silently dropping
them. Exit status is nonzero if any source failed.

Snapshots provide code references only, not proof of compatibility, correctness
or performance on any particular target. The dedicated
`Ascend/triton-ascend-kernels` repository supplies kernel implementations with
host wrappers, benchmarks, tests and best-practice documentation; the
Triton-Ascend compiler repository supplements this with tutorials.

## Optional documentation allowlists

A source may supply `files`, a nonempty list of `{ "path": "...", "sha256":
"..." }` entries. Acquisition requires every entry to match a selected UTF-8
Git blob at the pinned revision. Only these files enter the inventory; remaining
text is recorded as `outside-source-allowlist`. Missing paths, wrong hashes, and
duplicates fail acquisition, including on offline resume. Omit `files` to retain
the historical whole-source selection.

`sanitizer-sources.json` uses this facility for the official English msSanitizer
guides. See [sanitizer KDB setup](../../docs/sanitizer-kdb.md) for publication,
new-campaign selection, and diagnostic limitations.
