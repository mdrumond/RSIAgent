# Independent RSI instances

An instance owns a frozen copy of the RSI machinery, memory, baseline KDB, skill
bundle and mapping, run artifacts, dependency environments, and dashboard state.
The global CLI routes instance operations to that verified machinery snapshot.
Changes to a global checkout do not silently upgrade an existing instance.

Use Python 3.12 or 3.13 when enabling NVIDIA NOOA roles, and Node 22 or newer for
Pi. Install the packaged CLI, export starter skills, then bootstrap:

```sh
python -m pip install .
rsi skills export --builtin --destination ./starter-skills
rsi instance create ./my-rsi --baseline-kdb ./baseline.sqlite3 \
  --baseline-skills ./starter-skills --skill-map ./starter-skills/map.json
rsi --instance ./my-rsi instance setup --embeddings --node /absolute/path/to/node
rsi --instance ./my-rsi dashboard start --port 0
```

`baseline.sqlite3` is an existing KnowledgeDB database. Supply your existing
skills directory and mapping instead of starter templates when available. Add
`--repository REPO --repository-revision REV` to snapshot an optional repository.
Keep provider credentials in the launching process environment. Setup installs
legacy/embedding packages and the NOOA role worker into separate owned Python
environments; incompatible OpenAI SDK requirements do not share an environment.
Setup time is recorded separately from actor time.

Before starting jobs, set `kdb_collections` in the configuration payload to the
existing collection names in that database. Configure `embeddings.cache_dir` to
a prepared cache containing the pinned BGE model. Embedding setup installs Python
packages; it does not download model weights. Retrieval loads cached weights
only by default. To permit downloading the pinned weights, explicitly set
`embeddings.local_files_only` to `false` in the payload. Repository ingestion can
add a collection to the configured baseline after embeddings are available.

For CPU-only offline setup, provide a complete compatible wheelhouse including a
CPU PyTorch wheel and both separate role/legacy dependency sets, plus the pinned
npm cache:

```sh
rsi --instance ./my-rsi instance setup --embeddings --offline \
  --pip-cache /absolute/wheels --npm-cache /absolute/npm-cache \
  --node /absolute/path/to/node
```

Use the dashboard URL returned by `dashboard start`. The input editor controls
curricula, source/verifier folders, execution commands, independent heldouts,
performance goals, providers, prompts, budgets and recovery/tuning settings.
The execution page starts/stops jobs, revisits phases, selects accepted parent
inputs, edits baseline skills/mapping, ingests repositories, exports published
skills, triggers guided tuning, and inspects retained evidence/accounting.
Node folder browsing uses configured `browser_roots`.

The equivalent CLI is documented in [cli-jobs.md](cli-jobs.md). Configuration
reads return `{revision,payload}`; `config set --file INPUT.json` takes the payload,
with optional `--expect-revision REV` for conflict detection. An execution entry
in `project_configs` is keyed by curriculum `project_id` and supplies `source`,
`verifier`, `editable_files`, and `commands` (`compile`, `test`, `validation`,
optional `timing` and `profile`). Its test command must match the ProjectSpec
command. Both `test` and `validation` command arrays must pass `{verifier}` as a
standalone argument, for example `["python", "{verifier}"]`, so execution uses the
frozen independent verifier. Performance acceptance uses frozen host measurements and a declared
threshold, not model assertions.

Every actor uses the same host-controlled stages. Design and optimisation require
actual KDB retrieval, including negative observations, and retrospective evidence
before selecting a strategy. Mapped skills require real delivered reads. Missing
or ambiguous execution evidence cannot complete a stage. Failed targets remain
honest failures with retained source, receipts and elapsed costs.

The programming controller runs through a Pi extension. Pi retains the tool
conversation and manages automatic compaction; RSI executes host tools and checks
their completed results before advancing the workflow. Large source and evidence
records remain available through paged reads. Compaction work counts toward the
actor budget, and the original observations remain in the evidence log for SFT.

Phase1 keeps skills fixed and may simplify bounded recovery projects. Phase2
writes candidates after tasks and validates them through paired recovery and
independent A/B cases. Default tuning cadence is every five originals and wave
end, with five heldouts and two repetitions per arm. Ties, incomplete evidence
and provider-only variance keep the incumbent. Inference forks accepted heads
into independent mutable state and supports `--no-recovery`.

Actor budgets charge elapsed thinking, retrieval, compilation, execution,
debugging and profiling, including failures. Setup and measured remote data-copy
intervals are excluded. Overlapping charged intervals count once; category totals
can overlap. Resume preserves spent allocations. Unknown pending work must be
observed or explicitly reconciled from verified closure evidence; starting a new
job does not establish completion of an old one. All raw evidence remains
available for later SFT export.
