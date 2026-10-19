# Explicit Ascend campaigns and training annexes

`run_ascend_campaign.py` starts **one requested phase**, not an automatic
Phase 1 → Phase 2 pipeline. A3 uses Ascend C; A5 keeps the Catlass DSL
entrypoint and runtime bindings. Existing phase-specific scripts remain usable.
The launcher runs agents locally and uses the existing checked device adapters.

Prepare an ignored settings JSON outside the campaign directory:

```json
{
  "model": "openai/gpt-5.6-sol",
  "source_manifest": "/local/observed-sources.json",
  "phase1_arguments": [
    "--config", "/local/native-runtime.json",
    "--env-file", "/local/provider.env",
    "--model", "openai/gpt-5.6-sol",
    "--knowledge", "with-kdb",
    "--profiling", "new"
  ]
}
```

`phase1_arguments` are the existing target CLI's runtime/treatment arguments.
For A5 these include guide/Catlass revision/toolchain paths instead of native
`--config`; see [target composition](ascend-target-launcher.md). Campaign root,
objective, seed, phase, and budget belong to the launcher and cannot be
overridden in this list. Keys stay in the ignored environment file. The
source manifest uses the observed `ascend-sft-sources-v2` format described in
[SFT datasets](a5-sft-dataset.md): list only repositories actually available,
with their exact revisions and licenses. No weights or repositories are downloaded.

```bash
python run_ascend_campaign.py start --target a3 --phase 1 \
  --root /local/campaign --control /local/campaign-control \
  --config /local/settings.json --initial-memory /local/seed.md \
  --objective 'Learn reliable kernels and measured performance' --project-target 64
python run_ascend_campaign.py status --control /local/campaign-control --watch
python run_ascend_campaign.py resume --control /local/campaign-control
python run_ascend_campaign.py continue --control /local/campaign-control --project-target 96
```

GNU Screen and Python must be installed on this local host.
The harness can run from a Git checkout or a `git archive` source distribution.
Keep the exported `.git_archival.txt` (expanded by `.gitattributes` export-subst)
when deploying an archive: it records the exact harness commit without shipping
Git metadata. Unversioned copies or unexpanded placeholders are rejected rather
than assigned an invented revision.

Two separately
named Screen sessions retain compute and collection logs in the control
directory. A downloader failure never stops or waits on compute. `resume`
retries unfinished work without raising the target; `continue` explicitly raises
it and preserves successful publications and stop history. To reopen a curriculum
terminal, additionally pass `--reopen-terminal`. Infrastructure failures retain
the worker error and pending project; inspect them before explicit `resume`.
This launcher does not turn semantic failures into successful memory or silently
restart an expensive campaign. Status distinguishes verified from published
projects, reports wall-clock publication throughput, and polls every five minutes
with `--watch`.

### Parallel A3 projects within a Phase 1 wave

In the ignored native runtime JSON, use `"physical_devices": [0, 1]` instead
of `"physical_device": 0` to opt into two concurrent project workers. The list
must contain distinct nonnegative device IDs; select devices only after checking
current capacity through the registered target. The list takes precedence over
the legacy scalar for Phase 1. Phase 2 stays serial on the explicit
`physical_device`, or the first pool device if that scalar is omitted.
Omitting the pool preserves serial execution and existing paths.
This is an explicit device pool, not an automatic idle-device scheduler or a
persistent lease system. Do not assign its devices to another campaign.

Each project borrows one worker for its entire Actor loop, including compilation,
host verification, timing, and profiling. Local worker state and evidence live in
`native-workers/device-N/`; retained handles are under each worker's
`transport-state/`. Remote candidate folders retain their project/plan-derived
identities under the registered workspace. Project conversations, artifacts,
and KDB journals remain in their existing isolated wave branches. Queries share
the pinned embedding model but are serialized; model inference and device work
can overlap. All siblings see the same frozen pre-wave memory; publication still
waits at the wave barrier and consolidates in authored order.
An infrastructure or unexpected executor exception stops new dispatch from that
pool, while already running siblings finish. This avoids reusing a device with
an unobserved native operation; ordinary semantic project failures do not stop
other projects.

To change an existing campaign's pool, first stop its sole controller and settle
all retained native handles, including both legacy and worker transport folders.
Then resume the same root: verified sibling outcomes are retained without reruns.
Never start a second controller against that root. Phase 2 remains serial, and
this A3-only option does not change A5 entrypoints.

Phase 2 additionally requires `--phase1-root` and `--review` at `start` (and the
original Phase 1 initial memory as `--initial-memory`). The existing usefulness,
diversity, and profiler-guidance review must pass. The launcher copies and
verifies the reviewed snapshot once, then continues from that immutable copy:
later Phase 1 continuation cannot silently change a Phase 2 lineage.

New Phase 2 starts use `--project-target 1`. Without `--target-goal`, the curriculum
authors a new bounded kernel goal using reviewed Phase 1 lessons and gated KDB
searches when enabled. It rejects renamed Phase 1 tasks and unavailable tools.
An independent Actor constructs a private reference; the host then executes it
again through `qualify_target` before freezing the target. At most three candidate
proposals are allowed, with actual reference failures fed back to the curriculum.
Provider/transport interruptions retain the proposal and handles for `resume`;
they do not consume another proposal. `goal-selection.json` exposes selection,
qualification and terminal status before ordinary Phase 2 state exists. Private
source, conversations and reference-generated lessons stay in the control
directory's `bootstrap/`, never learner memory. Retain that directory alongside
campaign archives to preserve qualification evidence.

The learner attempts that fixed goal, performs focused curriculum-selected
practice and retries, using three target attempts and at most two practice
attempts between targets by default. Practice success alone is not readiness.
An explicit `--target-goal` still accepts an already host-qualified contract;
existing campaigns keep their retained behavior on resume. Fixed attention
benchmarks remain separate compatibility entrypoints. A different reviewed seed
requires a new Phase 2 campaign root.

Collection discovers pending raw profiler reports, transfers one at a time,
retains supported transport handles, backs off on poor links, and checks hashes;
see [collector behavior](artifact-collector.md). Once compute finishes, the
independent collector queues a strict archive of memory and all retained evidence
at `campaign-<target>.tar.gz` in the control directory. Compute does not wait for
missing reports or packaging. `package-status.json` explains pending downloads or
other incomplete evidence. To restart collection or make an explicit checkpoint:

```bash
python run_ascend_campaign.py collect --control /local/campaign-control --watch
python run_ascend_campaign.py package --control /local/campaign-control \
  --output /local/checkpoint.tar.gz --allow-pending
```

Without `--allow-pending`, packaging requires a verified committed boundary and
all referenced artifacts. Checkpoints explicitly retain a pending label. Archives
contain the public run identity, observed source manifest, memory, exact candidate
and host evidence, sealed conversations, failures, timings, and full available
raw reports. Local settings, environment files, credentials, weights, and control
files are not copied into the campaign/archive. Runtime settings are content-pinned
for a campaign; use a fresh root for a different treatment. Export training rows
offline through `benchmarks.ascend.training_traces.export_annex`; historical
summaries or unbound/failed traces remain evidence, not invented SFT targets.
