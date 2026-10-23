# msSanitizer reference documentation for agents

msSanitizer is the correctness diagnostic tool; msprof measures performance.
The optional `config/knowledge/sanitizer-sources.json` pins official English
[Ascend/msSanitizer documentation](https://github.com/Ascend/mssanitizer/tree/78037658773f27796624148ce7680ccb9e24ae66/docs/en)
at commit `78037658773f27796624148ce7680ccb9e24ae66`. Each selected file has a
SHA-256 allowlist entry. The selection includes user and compilation guides,
examples, installation, troubleshooting, release notes, and licensing. It does
not execute upstream code or enable an agent sanitizer action.

## Prepare a new combined database

Acquire the docs through the same pinned-source workflow as code references:

```sh
python -m benchmarks.a5kernels.source_acquisition \
  --config config/knowledge/sanitizer-sources.json \
  --destination /absolute/ignored/sanitizer-sources \
  --inventory /absolute/ignored/sanitizer-inventory.json --workers 1
```

Merge the resulting `sources` list with the previously verified reference
inventory's `sources` list in a **new** inventory (schema version 1). Do not
change either source's metadata or file entries. Source names must remain
unique. Retain the old inventories and database. Then publish the combined
inventory to a new database path:

```sh
python -m benchmarks.a5kernels.expanded_kdb ingest \
  --inventory /absolute/ignored/combined-inventory.json \
  --db /absolute/ignored/references-with-sanitizer.sqlite3 \
  --model-cache /absolute/ignored/huggingface/hub --batch-size 32
```

The sibling `.inventory.json` report supplies the actual immutable collection
name and fingerprint. The sanitizer collection starts with `mssanitizer-en-`;
use the complete published name, not that prefix. Its metadata identifies
English documentation relevant to Ascend C and Triton. This is reference
applicability, not evidence of support on a particular chip or compiler.

## Offer the collection in both phases

For a **new** native Phase 1 campaign, set the KDB path to the new database and
extend `collection` to a list containing the existing Ascend C collection(s)
plus the published sanitizer collection. Preserve an explicitly Ascend C-capable
code collection. For a new Phase 2 campaign, retain the same collection list and
all manifest fingerprints from the reviewed Phase 1 handoff in its `knowledge`
configuration. Do not retrofit an existing experiment: its collection selection
and fingerprints are part of its identity. KDB-off configurations stay off.
A5 launchers accept repeated `--collection` options for the equivalent selection.

Validate retrieval through the existing read-only Knowledge Agent gate:

```sh
python -m benchmarks.a5kernels.expanded_kdb query \
  --db /absolute/ignored/references-with-sanitizer.sqlite3 \
  --model-cache /absolute/ignored/huggingface/hub \
  --journal /absolute/ignored/sanitizer-query.jsonl \
  --collection REPLACE_WITH_PUBLISHED_SANITIZER_COLLECTION --kernel-language ascend-c \
  --query 'How do I detect a race between DataCopy and vector compute, and what instrumentation is needed?'
```

Inspect returned citations for repository, immutable revision, path, line range,
and source/chunk hashes. A model smoke should actually invoke the production
knowledge action and cite retrieved evidence for race detection, instrumentation,
and report interpretation. A keyword hit alone does not establish agent use.

## Interpret diagnostics carefully

Consult the pinned user guide and installed tool help together. `memcheck`
checks memory accesses; `racecheck` checks read/write contention, including
inter-pipeline races; `initcheck` checks uninitialized reads; `synccheck` checks
mispaired synchronization. Run memory checks before other checks as advised by
the guide. Preserve the failing inputs, expected output, actual output, candidate
hash, command, tool/compiler version, and report when diagnosing a failure.

Full checking can require recompiling with sanitizer instrumentation. The
compilation guide describes `-g` and `--cce-enable-sanitizer`/`--sanitizer` for
supported compilation paths, and warns against running instrumented binaries
without the tool. Its old Kernel Launch sample explicitly excludes Atlas A3
and Ascend 950 products. Ascend 950 does not support those sanitizer flags or
`-O0` in that guide; do not copy legacy flags into Catlass DSL/A5 builds.
Determine compatibility with the actual target, runtime, and compiler first.
The pinned documentation revision need not match an installed tool release.

For supported Torch custom-operator checks, the guide recommends
`PYTORCH_NO_NPU_MEMORY_CACHING=1` to avoid allocator-pool ambiguity. Keep any
such setting local to the diagnostic process. An application replay can include
framework setup kernels: identify the candidate kernel and coverage rather
than attributing every report to it. A zero exit status or no report does not
prove instrumentation coverage, numerical correctness, or race freedom. Retest
the unchanged failing case against the host-owned oracle after a correction.
Keep diagnostic runs separate from performance timing, and do not change shared
driver/debug settings merely to make a check run.
