# Instance CLI jobs

Bootstrap an instance with a release containing the job service, then configure its
curriculum and `project_configs` through `rsi --instance PATH config set --file INPUT`.
Instance commands execute that instance's verified machinery snapshot. Upgrading the
global CLI does not upgrade existing instances. Provider credentials remain in the
external process environment.

```console
rsi --instance PATH phase1 start --cycle microbench-wave
rsi --instance PATH phase2 start --cycle application-wave
rsi --instance PATH inference start --project problem.json --no-recovery
rsi --instance PATH job list
rsi --instance PATH job status JOB_ID
rsi --instance PATH job evidence JOB_ID
rsi --instance PATH phase1 stop --job JOB_ID
rsi --instance PATH phase1 resume --job JOB_ID
rsi --instance PATH phase1 revisit --job JOB_ID --cycle new-wave
```

Start returns a retained job ID immediately. One owned worker holds the instance
job lock until its service has drained and published its final result. `job status`
separates the worker's process state from the service result, such as `suspended`
or `awaiting_tuning`. Stop verifies process identity before signalling that worker;
the service unwinds canonical actor and child cleanup, preserving actual elapsed
time and pending operation identities. Interrupted work requires explicit resume
or revisit; unknown pending work is observed and never automatically replayed.

`revisit` starts a new cycle from the previous accepted input heads and current
curriculum/prompts, retaining predecessor lineage. `resume` opens the existing cycle. To steer upcoming tasks
in a retained cycle, stop it, edit configuration, then submit:

```console
rsi --instance PATH phase1 adopt --job JOB_ID
rsi --instance PATH phase1 resume --job ADOPTION_JOB_ID
```

Adoption uses the service's idle task/wave boundary transaction. It refuses edits
to admitted history, pending work, baselines and other frozen cycle policy. Existing
projects retain their source, skill, memory, verifier, budget and prompt contracts.

Repository ingestion and retained task export are also available as `kdb-add-repo`
and `evidence-export`. Instance ingestion publishes a new baseline database and
configuration revision while preserving already admitted cycles. Standalone
ingestion accepts explicit database and embedding inputs. Fine tuning runs the
same canonical runner on independently configured heldouts and retains the
incumbent until complete validation supports promotion. See
[quickstart.md](quickstart.md) for owned dependency setup and dashboard lifecycle.

## Baseline skills and timing evidence

- skills get [NAME] [--builtin]
- skills set NAME --file SKILL.md [--expect-bundle REF]
- skills map inspect
- skills map set --file mapping.json [--expect-bundle REF]
- skills tune --project CYCLE/OPERATION --guidance TEXT

Content and mapping edits validate the complete bundle and YAML metadata, publish
an immutable baseline copy, and revise configuration pointers with a conflict
check. Existing admitted bundles remain unchanged. A cycle baseline cannot be
switched by steering; start a new cycle to use an edited baseline.

Tuning resolves a retained phase2/inference host project receipt, also accepted as
its CAS reference, then starts an owned service job with structured user guidance.
The service performs the configured independent A/B evaluation; it retains the
incumbent on ties, regressions or incomplete evidence. Tuning is not a direct skill
promotion or a second programming workflow.

Job evidence includes an inspection section with task intervals, per-category
charged/excluded durations, the canonical actor union, remaining allocations and
skill references. Category durations can overlap: use actor_union_seconds for the
charged total. Active or uncertain actor totals remain null; bounded remaining
values that cannot be established are omitted. Disabled caps are explicitly null.

## Phase inputs and delivered skills

Start/revisit accept --from-cycle CYCLE, --inheritance auto|parent|baseline and
repeat --input-override memory|skills|kdb. Inputs enter a NEW cycle through the
shared service. Phase1 defaults to baselines; explicit parent inheritance may use
any accepted prior phase. Default revisit captures previous cycle accepted heads
without carrying old baseline overrides. Explicit baseline starts fresh.
Resume keeps the original frozen contract and accepts no new inheritance flags.
Automatic Phase2 uses accepted Phase1 memory; inference uses published validated
Phase2 heads. Explicit parent acknowledges inheritance; baseline ignores parents.
A component override uses the current baseline for that component. The service
records immutable choice/provenance. New curriculum/execution/prompts stay current.

skills get --cycle CYCLE reads published validated skills, optionally one NAME.
skills export --cycle CYCLE --destination NEWDIR delivers a reusable baseline.
skills export --builtin --destination NEWDIR delivers starter templates without
an instance. Read/export dispatch no actor and need no model setup receipt;
pinned instance machinery integrity is still verified.

The destination must be new. It contains NAME/SKILL.md, map.json, exact bundle.json
and export.json inventory/provenance/CAS reference/SkillBundle hash. Pending or
quarantined candidates are not selected. Atomic publication cannot replace an
existing destination. Use the folder as --baseline-skills and its map.json as
--skill-map in fresh instance creation, without opaque-CAS manual reconstruction.

## Opt-in project waves

An optional saved configuration selects admitted physical-device/executor identities:

```json
{
  "project_wave": {
    "max_parallel": 2,
    "devices": [
      {"device": "0", "executor_identity": "admitted-device-0"},
      {"device": "1", "executor_identity": "admitted-device-1"}
    ]
  }
}
```

Use the normal phase start/stop/resume commands and dashboard. Each child runs the
canonical `ProjectRunner` with the wave's frozen memory, skills, configuration and
read-only KDB. Source snapshots retain the complete configured resource tree;
`editable_files` still limits candidate edits. The parent admits memory only after
all siblings settle and host validation verifies their retained outcomes. Rejected,
exhausted and failed attempts remain distinct from accepted artifacts.

Wave projects preserve their frozen per-project actor allowance (7200 seconds
when unset), with iteration/attempt stopping
limits disabled. Parent actor and absolute wall allocations still constrain wave
admission and nested work. Explicit harness recovery can add 7200 seconds once,
retaining elapsed accounting. Configured semantic recovery and skill-validation
policies continue through the existing controllers with linked operation histories.

`RSI_EXECUTOR_PROFILES` names an ignored host JSON file with a `profiles` object.
A profile may contain the concrete `CplConfig` constructor settings, or a trusted
`factory: "module:callable"` plus `configuration` object. That callable receives
`(identity, configuration)` and returns a typed executor; this supports both a local
controller with CPL target commands and an externally admitted device controller
with local target commands. Credentials, machine routes, runtime directories and
outer admission settings stay in host configuration. Only the executor's public
identity/provenance is frozen in project receipts.

Stop prevents new admissions and drains owned workers. Inspection shows draining,
observation-required handles, unstarted siblings, true project outcomes, leases and
remaining actor allocations. Resume reuses accepted results and may admit ready
siblings; it never replays an unknown dispatch. Reconcile/recovery controls require
the corresponding pinned CLI capability:

```console
rsi --instance PATH phase1 reconcile --job JOB_ID
rsi --instance PATH phase1 reconcile --job JOB_ID --operation OPERATION --observation observed.json
rsi --instance PATH phase1 recover --job JOB_ID --operation OPERATION --observation observed.json
```

The optional host observation contains `controller_proof`, explicit closed actor
intervals when needed, and per-command `timing` receipts for target reconciliation.
Timing binds the original operation/source and authoritative controller-domain
execution bounds; remote epoch subtraction is insufficient. Target reconciliation
observes the original handle and finalizes its existing receipt. Missing closure,
source, or timing evidence keeps work pending. A harness continuation must close
its canonical pending admission before a new actor step; otherwise use the
configured, evidence-linked recovery workflow. Adoption and accepted-parent
inheritance wait for the wave's parent publication boundary.

For an outer-admitted A5 controller using default local target execution, select
`worker_profile` on each device and keep `executor_identity` absent or null. Its
ignored profile supplies public `metadata` with `controller_placement: "local"`,
`executor_identity: null`, physical device, runtime and actual `outer_handle`;
`worker_environment` names an optional shipped target callback, and
`worker_configuration` supplies that outer handle and physical device. Target
callback modules are delivered and validated by their target-specific PRs; the
generic integration neither imports nor ships a Catlass compiler/verifier. The child
supplies its own cache path, applies the returned environment before runner/target
imports, and never mutates the parent's environment. It preserves the helper's
source-hash behavior: stale `RSI_EXPECTED_SOURCE_HASH` is removed and the local
trusted verifier checks its own current project hash before/after command work.

Every NPU profile selects a shipped, host-trusted target guard using
`guard: "module:callable"` and optional `guard_configuration`. Profiles configure
the supplied implementation; users do not implement device-health callbacks. The callback accepts `(request, *, phase,
metadata, configuration, remaining)` for `admission` and each `command`. It must
perform fresh bounded per-device health and policy checks and return a public
receipt containing `admitted: true`, nonempty `observation` and the matching
`physical_device`. Missing guards or denied health leave work undispatched.
Concrete target guard/verifier implementations own actual bounded local or
registered-target probes and idle/memory/isolation policy. Frozen target verifier
gates run before Torch/NPU work under the admitted CPL route. The profile callback
must align with those same gates; metadata alone is never health evidence. Guard time is actor-billable.

The shipped guard is `rsi.device_admission:guard`, backed by the same stdlib-only
probe that target builders embed in frozen verifiers. `allow_alarm: true` permits
only explicitly authorized ordinary functional work on measured Alarm devices;
the receipt remains degraded and retains actual faults. It neither permits busy
devices nor relaxes healthy-idle requirements for timing/profiling. Defaults deny
Alarm, and terminal, unavailable or unknown devices remain excluded.

NPU profiles also select a shipped `project_binder: "module:bind_project"` with
`project_configuration` containing its ignored host builder inputs. After a
physical lease is selected, the child calls
`bind_project(request, logical_admission, *, metadata, configuration, remaining)`
before creating the runner. The binder creates/selects source/runtime inputs and
an independent verifier for that actual device; it never rewrites the baseline
or precreates the runner root. Its resolved spec/config and logical case/source
proof are sealed in CAS and linked to the runner's frozen backend identity.
Resume reuses the exact saved binding without running the binder again.

Logical case/oracle identity excludes physical argv and filesystem locations;
physical binding changes do not invent a different task. Parent validation checks
that proof and `commands.test == ProjectSpec.command`, preserving the original
logical goal, cases, budgets and edit/retention policies. The canonical actor task
context includes a whitelisted `execution_context` with actual target/runtime,
SoC/device, revisions and public admission facts. Private profiles, credentials
and target package paths are not copied into that context. Controllers do not
restore target Python paths globally; any registry-derived pre-import paths belong
to the target-only frozen verifier builder and need applicable runtime evidence.
