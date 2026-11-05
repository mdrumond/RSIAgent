# Campaign storage

New instances use one campaign identity and one SQLite ledger (`state.sqlite`) across phase 1, phase 2, inference, deployments and recovery. Existing campaigns are not migrated automatically. Attempts pin their knowledge generation and memory/skill Git revisions. Iterations retain their stage, reflection and source tree; an actor can make ordinary Git commits through Pi, but commits are optional.

Pi owns native JSONL sessions, context and exports. Raw provider telemetry is retained separately. The ledger records artifact paths, hashes, owners, citations and lineage; it does not duplicate complete conversations. Walkers stream original records without reconstructing sessions or interpreting Pi branches.

## Dependencies

Storage uses Git, Python SQLite and [Restic](https://restic.net/). Bootstrap installs the pinned official Restic binary into the user's cache if Restic is absent. Installation verifies its release checksum. For explicit setup:

```sh
python -m ascend_tools.restic_install --install
```

`RSI_RESTIC` selects an existing executable. Restic 0.17.0 or newer is required; setup rejects older versions. Archive operations never download software. Repositories use an empty password and private filesystem permissions: protect them as sensitive unencrypted data. These packages are not suitable for publication without separate review.

## Commands

Run these from the installed machinery directory, replacing paths with your instance and destination:

```sh
ascend-rsi storage inspect --instance /path/to/instance
ascend-rsi storage verify --instance /path/to/instance --project PROJECT
ascend-rsi storage walk --instance /path/to/instance --project PROJECT
ascend-rsi storage walk --instance /path/to/instance --raw-provider
ascend-rsi storage pack --instance /path/to/instance --project PROJECT --destination /path/to/package
ascend-rsi storage pack --instance /path/to/instance --destination /path/to/campaign-package
ascend-rsi storage unpack --package /path/to/package --destination /path/to/new-instance
```

Pack produces a standard local Restic repository plus RSI metadata. A project package includes its attempts, diagnostics, citations, required skill/memory history and source trees. A campaign package also includes its KDB. SQLite snapshots use its backup API. Unpack validates hashes, restores into a new destination and leaves execution paused. External evidence must be collected before a complete package can be made.

Walk filters include phase, role, project, outcome, stage and revision. `--cursor` resumes within a selected artifact; emitted metadata includes the next cursor and artifact identity. Inventory pages use `--after` and `--limit`. For fine tuning, retain both raw records and their ledger metadata; decide training selection separately from archival.

## Retention and memory

Managed maintenance runs serially with a 512 MiB aggregate memory ceiling and one CPU. Restic handles compression, deduplication, indexes and interrupted backups. It uses one reader and a bounded Go heap target. No custom archive/chunk engine is maintained.

Only sealed artifacts whose writers have stopped may be archived. Source deletion requires a published ledger reference, unchanged source identity and verified archive contents. Explicit prune first returns an inventory; apply requires that same inventory. No campaign history is automatically discarded.

Memory and skills share a local Git history without remotes. Memory is staged in KDB at its Git revision before activation. Attempts query their pinned generation; rejected skill proposals and negative findings remain evidence. Binary vectors and deduplicated chunk contents reduce KDB size. Pruning is explicit and preserves pinned or cited evidence.
