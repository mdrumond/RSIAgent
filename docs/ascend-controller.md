# Ascend controller guide

## Architecture and ownership

The supported implementation consists of:

| Path | Responsibility |
| --- | --- |
| `controller/` | SQLite current state, admission, elapsed budgets, canonical Pi worker and recovery queue |
| `cli/` | Bootstrap, configuration, projects, detached wave driver, stop/resume, KDB and tuning actions |
| `integrations/environments/` | User-configured local or CPL execution, source synchronization and durable execution handles |
| `integrations/kdb-extension.mjs` | Native Pi MCP registration, correlated retrieval and bounded evidence |
| `ascend_tools/` | Python KDB/BGE, shared MCP service, trusted target leaves and thin Flask dashboard |
| `web/` | Static dashboard HTML, JavaScript and CSS |
| `templates/skills/` | Baseline workflow skills |
| `tests/ascend/` | Current CPU behavioral and protocol tests |

Pi 1.3.0 owns provider requests, tool execution, session persistence, retries and
context management. The controller supplies workflow gates and tool operations;
it does not implement a second model/session protocol. Environment plugins own
transport details. Target leaves own device/runtime admission and independent
numeric verification. The controller does not import CPL or guess device IDs.

## Fresh bootstrap

Use Node 22.23.3. Python 3.12 is the tested, recommended host; packaging permits
Python 3.11 and newer. Full Python 3.11 host runtime acceptance remains untested.
Registered target kernels use Python 3.11 and their own pinned dependencies.
Install root Node dependencies and the Python helper package first. Production BGE queries additionally require the
`embeddings` extra and the pinned model already available in its configured
cache. No embedding model download or corpus reindex is part of bootstrap.

A private bootstrap JSON file has this shape; replace the example paths with
actual private paths before running it:

```json
{
  "baseline": "/path/to/baseline.sqlite",
  "skills": "/path/to/baseline-skills",
  "mode": "phase1",
  "actor_seconds": 28800,
  "environment": {
    "module": "/path/to/machinery/integrations/environments/index.mjs",
    "options": {"mode": "local", "instructionFile": "/path/to/private/access.md"}
  },
  "projects": []
}
```

For CPL execution, options select `mode: "cpl"`, a registered `target` and named
`runtime`, the registry's `workspacePath`, and a task-owned `remoteDirectory`.
Resolve these from private registry configuration. The driver imports
`environment.module` and passes `environment.options` to the extension; put
`instructionFile` inside `options`. Access instructions
explain how the actor should use the selected environment; credentials stay in
private configuration. Execution uses the installed remote-access client.

```bash
node cli/main.mjs bootstrap "$INSTANCE" --config "$BOOTSTRAP_JSON"
node "$INSTANCE/machinery/cli/main.mjs" status --instance "$INSTANCE"
```

The destination is new and outside the machinery source. Bootstrap snapshots
the baseline SQLite database, copies baseline skills into instance-owned
versioned storage, and copies the supported machinery. An optional `repository`
input copies a starting repository. The instance owns `instance.json`, current
`state.sqlite`, `knowledge.sqlite`, skills, projects and execution evidence.
Bootstrap automatically runs `npm ci --ignore-scripts` for the copied Node
machinery using its standard `npm-shrinkwrap.json`; source `node_modules` is not
copied. If installation needs repair, rerun
`npm ci --ignore-scripts --prefix "$INSTANCE/machinery"`. Extra bootstrap
settings, including actor profiles, are preserved in the instance configuration.

Only the current instance format is accepted. Existing historical state is not
migrated, adopted or replayed. Preserve old research evidence separately.

## Configuration and projects

```bash
node "$INSTANCE/machinery/cli/main.mjs" config --instance "$INSTANCE"
node "$INSTANCE/machinery/cli/main.mjs" config --instance "$INSTANCE" --config "$SETTINGS_JSON"
node "$INSTANCE/machinery/cli/main.mjs" project --instance "$INSTANCE" --project "$PROJECT_JSON"
```

The config command reads or writes JSON settings. Actor profiles are root
`actor` (kernel work), `curriculum_actor` (task generation) and
`skill_writer_actor` (skill generation). Each profile supplies its provider,
model and credential environment variable; private credential files and explicit
model metadata may be configured as needed. Bootstrap preserves these root
fields. The dashboard actor editor projects them as `{actor, curriculum, writer}`
and writes back those same canonical root fields.

Use config for actor allocation,
concurrency, environment instructions/options, model selection, curriculum,
skills map, baseline KDB configuration and user information. Store path and
machine values only in the private instance. Defaults are 28,800 actor seconds,
9,600 investigation seconds, five unsuccessful recovery rounds, concurrency one
and tuning cadence eight distinct originals. Modes are `phase1`, `phase2` and
`inference`; inference recovery is configurable.

A project supplies an ID, goal, editable `source`, target/profile,
`actor_seconds`, and an immutable `verifier` with `command` and `files`.
Verifier files and frozen cases must live outside the editable source. Preserve
original goals and verifiers across investigation/retry. Executed projects
cannot be edited into a different problem; add a new project ID for that case.
The dashboard project editor submits the same JSON contract to the CLI.

## Waves, stop and resume

```bash
node "$INSTANCE/machinery/cli/main.mjs" wave --instance "$INSTANCE" --mode phase1 --detach
node "$INSTANCE/machinery/cli/main.mjs" campaign --instance "$INSTANCE" --mode phase1 --detach
node "$INSTANCE/machinery/cli/main.mjs" wave --instance "$INSTANCE" --mode phase2 --detach
node "$INSTANCE/machinery/cli/main.mjs" wave --instance "$INSTANCE" --mode inference --detach
node "$INSTANCE/machinery/cli/main.mjs" wave --instance "$INSTANCE" --mode revisit --detach
node "$INSTANCE/machinery/cli/main.mjs" status --instance "$INSTANCE"
node "$INSTANCE/machinery/cli/main.mjs" stop --instance "$INSTANCE"
node "$INSTANCE/machinery/cli/main.mjs" resume --instance "$INSTANCE" --detach
```

Run a chosen wave when requested; these examples do not prescribe running all
modes in sequence. Detached commands return after starting the CLI driver.
The public `campaign [--detach] --mode phaseX` command advances successive
waves through the same canonical worker; choose phase1, phase2 or inference.
It runs in the foreground unless detached. The existing `stop` command stops
the campaign. An open attempt or no progress returns a blocked result requiring
inspection and `resume`, rather than silently admitting replacement attempts.
Optional private `config.campaign_seconds` bounds the campaign by elapsed time.
Starting a campaign is an explicit user action, including from the dashboard.

Status returns project, attempt and background-request JSON. Project states are queued, running,
passed and failed, with freezing tracked separately. An attempt is persisted
before setup or process execution starts.

Resume preserves the same interrupted attempt, Pi session and remaining elapsed
allocation. It first requires proof that its local owner and recorded external
handles have stopped. Unknown work remains unresolved and must be observed;
a disconnected observer is not authorization to launch a replacement.

Only measured setup and transfer scopes pause the actor clock. Provider calls,
compilation, execution and verification consume active elapsed time. Request,
context, concurrency and case-size limits are validity constraints, not alternate
runtime stopping budgets.

## Workflow, memory and skills

The canonical sequence is preflight, task-construction, design, implementation,
test, debug, optimisation, validation and acceptance. Each stage requires an
actual skill read. Design and optimisation require positive and negative KDB
retrieval plus reflection on prior evidence. Failed validation loops to design
or implementation. Acceptance requires a passing independent frozen verifier
on the current source fingerprint. Compile success alone is not acceptance.

Phase 1 updates memory; Phase 2 and inference may also evolve skills. Recovery
investigations become eligible in the next wave. Useful investigation findings
schedule an original-goal retry in a following wave. Unusable findings or a
failed original retry consume a recovery round; the initial failure itself does
not. A genuine retry receives a fresh allocation. At the configured limit, the
original freezes. Investigation success never substitutes for original-goal
acceptance. All findings enter the durable memory outbox; acknowledge delivery
only after the KDB write succeeds.

One MCP service owns the instance KDB and lazy production BGE backend. Native
Pi tool calls carry the attempt correlation header. Compact lessons reference
source/citation IDs or direct evidence hashes rather than recursively copying
archives. The pinned BGE model is
`BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.

```bash
node "$INSTANCE/machinery/cli/main.mjs" kdb --instance "$INSTANCE" --config "$KDB_ACTION_JSON"
node "$INSTANCE/machinery/cli/main.mjs" config --instance "$INSTANCE" --config "$SETTINGS_JSON"
node "$INSTANCE/machinery/cli/main.mjs" tune --instance "$INSTANCE" --config "$TUNE_ACTION_JSON" --detach
```

The config read view includes the actual active `skillsmap`. Skill-map edits
create a new immutable user skill version and update the active pointer.
`baselineKDB` records the bootstrap baseline path; changing it is permitted only
before any attempts exist. After an attempt, create a fresh instance for a
different baseline.

Persist `userinfo` through config before invoking a tuning action; the dashboard
performs this ordering automatically and detaches the tune request. The CLI
persists its action input in the instance before acknowledging launch; later
status reports the request process/outcome. Tuning counts distinct completed original
projects, excluding diagnostic/retry attempts, and evaluates candidates through
the shared scheduler. Action JSON is interpreted by the corresponding CLI
module, not by a separate Python orchestration engine.

## Dashboard and browsing

```bash
node "$INSTANCE/machinery/cli/main.mjs" dashboard --instance "$INSTANCE" --port 46731
node "$INSTANCE/machinery/cli/main.mjs" browse --instance "$INSTANCE" --config "$BROWSE_JSON"
node "$INSTANCE/machinery/cli/main.mjs" skill --instance "$INSTANCE" --config "$SKILL_JSON"
```

The dashboard listens on loopback only. It exposes settings/project JSON editors,
a CLI-backed path browser, skill-content editor, status, wave/stop/resume
controls, KDB actions and skilltune user information. The skill selector uses
active skillsmap scopes. A skill action `{scope}` reads content; `{scope, content}`
saves a new immutable user version and updates the active pointer. Save refreshes
the config projection and displays the returned content; Python performs
no direct skill-file writes. Browse JSON supplies a `path`; the CLI resolves and
lists it server-side. The browser never reads the user's machine filesystem
directly. Every dashboard action uses a structured subprocess argument array
and temporary JSON input file, with no shell interpolation or old RSI imports.

## Validation and retained evidence

Install `.[dev]` and use the serial CPU commands in the README. CI installs the
root lockfile and helper package, then runs Node tests with concurrency one and
Python tests separately. It installs no Torch/Transformers embedding extra and
runs no paid provider, remote job or NPU campaign.

Keep full native sessions, provider streams/SFT exports, verifier output, source
fingerprints and raw execution logs for real runs. Label CPU contract, real
provider, A3 device and A5 device evidence separately. A3 success does not prove
A5 behavior. A5 research requires the pinned CANN 9.3 environment and exact
source/dependency identity. Device timing requires healthy idle isolation;
compile, launch, numeric correctness and timing are distinct claims. Consult
the extracted target guide for the exact supported ABI and failure reports.

Historical documents elsewhere in `docs/` remain research records and may refer
to removed entrypoints. This guide and the README describe the supported source.
