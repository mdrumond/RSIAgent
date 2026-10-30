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
and `evidence-export`; their help lists explicit database, embedding, outcome and
artifact inputs. Fine tuning is reported as awaiting its service rather than
claiming an unevaluated candidate was promoted. Dashboard serving and dependency
environment setup are separate commands supplied by their integration streams.

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
