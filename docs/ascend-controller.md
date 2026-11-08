# Ascend controller guide

## Architecture and ownership

The supported implementation consists of:

| Path | Responsibility |
| --- | --- |
| `controller/` | SQLite current state, admission, elapsed budgets, canonical Pi worker and recovery queue |
| `cli/` | Bootstrap, configuration, continuous admission, process lifecycle, pause/resume, KDB and tuning actions |
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

Ascend scheduler rows represent host worker slots. The Pi execution extension
selects a fresh idle, permitted physical device for each shell command. Local
device-host `bash` and `rsi_target` share that admission path. The command keeps
its device through compilation, execution and profiling; later commands may
use another device. Server aliases and runtimes share the registered NPU pool.
Unknown executions retain their reservations until positively reconciled.

For campaigns that permanently reserve an A/B device, the private Ascend profile
sets `backgroundDevice` to one physical ID in its explicit `deviceAffinity`.
Scheduler slot zero is marked for A/B and routes commands only to that device;
other workers exclude it. The first full iteration may use slot zero for
foreground work. Later iterations earmark it even while tuning is idle. An
unhealthy or disabled reserved device never falls back to a foreground device.
Scheduler inventories remain complete; ordinary command health and global lease
checks still decide whether execution can proceed. Profiles without this option
retain their existing routing.


## Continuous campaigns

Merge these portable policy settings into the private bootstrap profile:

```json
{
  "concurrency": "auto",
  "failed_recovery_rounds": 5,
  "pause_grace_seconds": 30,
  "phase_sequence": ["phase1", "phase2"],
  "phase_targets": {"phase1": 128},
  "curriculum": {
    "phase1": [],
    "phase2": ["end-to-end-project"],
    "generation": {
      "enabled": true,
      "topics": ["compiler recovery", "numeric debugging", "profiling"],
      "preprompt": "Use the configured Ascend language and existing repository evidence."
    }
  }
}
```

Phase 1 and Phase 2 refill their backlogs from the curriculum when generation is enabled.
Phase 1 constructs microprojects; Phase 2 constructs end-to-end projects with
declared performance goals preserved as `performance_goal` with
`verifier.performance_command`. These optional fields are preserved in either phase;
a non-null goal requires a performance command. Generated performance projects also
require `verifier.baseline_command`: an independent target self-test of the same
PyTorch baseline and placement path used by the performance command, covering all
frozen timing cases, warmup and timing calls against the trusted reference. This
command must work before any optimized candidate exists. The controller executes
it using the actor's admitted target resource before opening a model session.
A completed nonzero baseline verdict marks the generated project invalid and
freezes it without model summary or investigation retries. Transport uncertainty
remains an execution-settlement problem rather than a baseline verdict. Existing
supplied projects without this optional command retain their behavior.
New curriculum submissions declare each target command as
`{entrypoint: "trusted/verify.py", args: ["--mode", "correctness"]}`. The entrypoint
is relative to the generation draft and must be listed in the frozen verifier
files. RSI creates a quoted Python command with editable source as its working
directory, retaining entrypoint metadata so import can regenerate its paths.
Arguments use source-relative paths rather than absolute generation paths.
Commands cannot supply `cd` prefixes or shell fragments. Supplied legacy project
strings remain supported; their existing shell behavior is unchanged. A bundle's
declared root and relative source/trusted-file layout also survive target staging.
CPU construction checks establish
correctness only and do not execute that performance command. Optional
`curriculum.generation.by_phase.phase1` and `.phase2` objects shallowly override
`enabled`, `topics`, and `preprompt` for that phase. With a campaign plan, drafts
are pinned to a phase and entry; older entry drafts and projects cannot fill a
new batch. Remaining admissions cap staged projects, including pending drafts. Draft construction checks run
on CPU references and negative controls; rejected checks return to the same Pi
session for repair within the remaining construction allocation. An optional
`construction_command` lets these checks differ from the real target verifier. Generated specs retain a versioned
`bundle` with its root, trusted reference, negative controls, CPU construction
command and remaining curriculum seconds. Files in a coherent project subdirectory
outside editable source are frozen dependencies, including reference data and
verifier helpers. If artifacts sit directly in the generation session directory,
only declared verifier files and reference/control trees enter the bundle; list
all other helper files explicitly in `verifier.files`.

Import preserves this layout and reruns the unchanged CPU command from the copied
reference and every control before admission. Missing or changed dependencies and
non-discriminating controls reject the project. Import checks share the remaining
curriculum elapsed allocation, checkpoint it on interruption, and use the configured
local execution policy. A failed imported generated bundle is invalidated rather
than repeatedly receiving a fresh budget. Legacy supplied projects without a bundle
continue to import their declared source and verifier files.
Both phases can also consume supplied projects; inference consumes supplied projects. An empty supplied backlog waits
for input; a completed configured target advances the phase sequence.

A finished original requires a terminal result, settled execution, acknowledged
memory and completed postprocessing. Failures count; investigations and extra
retries do not. Recovery investigates smaller gaps and retains negative evidence.
It uses at most one worker per target; a single available worker admits three
fresh projects per recovery admission. Reaching the configured unsuccessful
recovery-round limit freezes the project for operator attention.

`campaign` continuously admits work. `pause` stops admission and pauses actors at
command boundaries, with a default 30-second grace before cancellation. `resume`
preserves sessions and elapsed allocations. `stop` requests immediate cancellation.
Pure device-queue waiting, actor setup and data transfer are excluded from actor
time; thinking, compilation, execution and profiling are charged. A configured
campaign elapsed allocation also survives controller restarts.

Public commands run as ordinary Node and Python processes. The CLI starts workers
and the instance KDB directly; no service-manager provisioning is required.
After a controller interruption, explicitly invoke `resume` to continue from
retained state. Native Pi sessions and elapsed allocations remain authoritative.
Choose an explicit concurrency appropriate to the host and target resources.
RSI does not install or enforce a shared host memory quota.

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

Skill proposals receive the current attempt's outcome and execution references.
When available, they also receive the most recent successful and failed attempts
with matching goal, initial source fingerprint, frozen verifier, and selected
target/runtime binding. Different skill versions may be compared; transient
health measurements do not define the binding. The binding identifies the
configured environment, not an independent audit of installed dependency versions.
Incomplete or historical identities omit comparisons. A failed attempt alone
can support a proposal; the tuner never runs a benchmark to obtain a successful
counterpart. Proposal directories retain `comparison-evidence.json` alongside
the original evidence. Observed failures support scoped lessons, not an untested
claim that a proposed alternative fixes the original project.

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

### Evaluation allocations and resume

Skill proposals accumulate without running benchmark projects unless
`skills.evaluationSeconds` supplies a positive elapsed allocation for a full
candidate evaluation round. This setting initializes each new round once;
resuming a round never refills it. Existing instances without this setting now
prepare candidates and report `deferred`, including manual tuning requests.
Promotion still requires two distinct projects and at least two runs per arm
(default `skills.runsPerArm: 2`). For inexpensive integration checks, explicitly
set `skills.runsPerArm: 1` to run four samples. A single run can reject correctness
regressions but never promote guidance; passing repair or timing observations
return `insufficient-repetitions`. These checks establish integration behavior,
not improvement efficacy.

To evaluate an already prepared candidate or add time to its pending round,
pass a tuning action such as
`{"evaluationSeconds":600,"allocationId":"user-extension-1"}`.
The seconds are an additional grant; use a new allocation ID for a new grant.
Replaying the same ID and seconds never adds time again, even after completion.
A grant-only action uses the existing candidate without invoking the writer.

The round snapshots settings and inputs, records stable sample paths and saves
completed results. Resume the same `tune` request to skip completed samples and
resume an interrupted sample through its existing canonical instance and
session. Unresolved owners or execution handles block replacement work.
Interrupting bootstrap before a complete instance exists requires inspection;
the partial directory is retained and is never automatically recreated.

The deadline covers bootstrap, dependency installation and execution, with
cancellation propagated to nested work. Graceful interruption checkpoints elapsed
time and excludes subsequent paused time only when stopped owners and operations
are positively proved. Unknown execution keeps charging across observation gaps
and failed reconciliation. Custom evaluator modules may export `settled(request)`
to provide this proof; missing or failed proof remains unknown. After process
loss, uncheckpointed elapsed time is conservatively charged until observation resumes. Exhaustion
keeps the candidate unpromoted until an explicit additional grant arrives.
Completed task evidence arriving while a round is blocked remains in history.
`status` includes `tuning` with its evaluation state, allocation, elapsed time,
remaining time and number of completed samples. States are `deferred`, `running`,
`interrupted`, `exhausted` and `completed`.

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

## Small local Python acceptance

`examples/local-python/README.md` describes a CPU-only acceptance path for skill
evolution using tiny Python workloads, the production native controller and
isolated evaluator, and independent frozen verifiers. Its deterministic provider
is local and free of API charges. An explicitly opted-in real-provider experiment
uses the same workloads with private configuration and separate retained evidence.

Fresh campaign manifests cap foreground actors plus the reserved A/B worker at four.
Their `local_memory` settings enable process-tree RSS observation every two seconds:
24 GiB blocks new admissions and reduces the retained worker cap every 30 seconds;
28 GiB or two minutes of pressure requests a checkpoint with the normal pause grace
plus 60 seconds. After local owners stop, the owned knowledge service is released.
The supervisor resumes only its own memory pause after usage stays below 20 GiB for
60 seconds. Reductions survive resume and supervisor restart; recurrent pressure at
one worker needs attention. Operator pauses/stops take precedence. The 32 GiB
emergency limit remains host containment, not an automatic restart trigger.
Supervisor status includes RSS, effective cap and the largest owned processes using
command names, never arguments or environment. Missing memory observations block new
work. A recovery that cannot settle or cool within 30 elapsed minutes needs attention.
Existing instances without `local_memory` keep their prior supervision behavior.

Fresh campaigns also set `local_execution` limits for each local command: 128
tasks (including threads), 4 GiB memory and no swap. Pi's native Bash operations
own spawning, streaming, timeout and cancellation; a user-systemd scope enforces
the resource boundary and cleans all descendants, including detached processes.
This requires a delegated Linux user manager with memory and pids controllers.
Missing containment fails closed rather than executing outside the scope.
The same policy covers native Pi Bash in actors and curriculum generation, and
CPU construction checks invoked by the submission tool. A limit violation is a
normal Pi tool error, so the agent can repair its command within the remaining
elapsed allocation. Device execution continues through the configured CPL target.
Legacy configurations without `local_execution` retain their existing behavior.

Memory-triggered resumes retain their expected pause token through the durable request
and recheck it atomically when claiming the native controller; a newer operator pause
or stop cannot be cleared by a queued recovery. If a retained one-worker cap cannot
serve foreground work alongside the permanent A/B reservation, supervision checkpoints
and reports `memory-at-minimum` instead of leaving the campaign silently stalled.
