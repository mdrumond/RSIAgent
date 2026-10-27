# Ascend subject curriculum

`run_ascend_subjects.py` replaces new separate Ascend Phase 2 launches. It uses
one ordered subject list for learning, repair, qualified target tasks and
optimizer work. Historical Phase 2 readers and controller APIs remain available;
the standalone Phase 2 scripts reject execution and retain `report`.

Copy `config/examples/subjects.json` and edit each subject's objective and
`min_waves` (default three). The curriculum may revisit earlier subjects when
accepted memory lacks coverage. A separate verifier reads source, execution
results, logs and related KDB memories, requests additional host checks, and
can send a host-passing project back for repair. Repair probes and controls
are explicitly linked to existing obligations.

Reuse an ignored runtime settings file containing `phase1_arguments`, the same
runtime/provider arguments used by the existing A3/A5 Phase 1 CLI. Optional
`verifier_model` and `curriculum_model` select registered model names; each
defaults to the configured Actor model. Each role has independent context.
Machine paths and provider credentials stay in ignored settings/environment.

```bash
python run_ascend_subjects.py run --target a3 --root /local/subject-run \
  --subjects /local/subjects.json --config /local/settings.json \
  --initial-memory /local/seed.txt --max-seconds 7200
python run_ascend_subjects.py resume --target a3 --root /local/subject-run \
  --subjects /local/subjects.json --config /local/settings.json \
  --initial-memory /local/seed.txt --max-seconds 7200
python run_ascend_subjects.py report --root /local/subject-run
```

A subject may attach `task: {"kind":"target","contract":{...},
"artifact_root":"/local/qualified-artifacts"}` using an existing qualified
`TargetContract`, or `task: {"kind":"optimizer","task":{...}}` using an
existing optimizer task. Optimizer subjects additionally require the ignored
`optimizer_worker_config` JSON path/settings with the same observed architecture
and physical device as the composed graph executor. Target qualification and
host correctness remain authoritative. Additional verifier checks accept a
complete variant `ProjectSpec`, retain separate host evidence, and leave the
original contract unchanged.

Accepted memory versions remain immutable. With KDB configured, each accepted
version receives a separate indexed collection and retrieval selects active
versions only; KDB-off needs no embedding backend. New graph Actors can follow
accepted memory code/evidence references in their frozen snapshots. Resume
preserves cached project execution and rejects changed subject/runtime settings.
A3 evidence establishes shared orchestration and Ascend C behavior; it does not
establish A5 Catlass execution behavior.

The existing optimizer task backend emits Ascend C, so `kind: optimizer` is
supported with the A3 Ascend C binding. An A5 Catlass optimizer subject is rejected
before composition; no Catlass optimizer backend is implied. A5 qualified graph
and graph performance subjects continue through their existing Catlass binding.

Memory evaluation can use `campaign.select_trial` and `campaign.execute_trial_arm`.
Every arm receives a fresh production Actor/session and isolated execution/cache
workspace while retaining the configured model, runtime and physical device.
Actor KDB retrieval follows the arm's frozen memory versions, even when the live
accepted catalog changes. `role_complete`, `model_identity` and `runtime_identity`
are used by the default memory verifier and admission composition.

Optimizer subjects retain the initial seed as a readable memory entry and export
accepted memory documents plus reachable code/evidence in authenticated optimizer
snapshot files. The existing `read-memory` and `read-artifact` tools can follow
the same memory links as graph Actors.

Runtime identity includes content hashes of referenced runtime/worker JSON files,
including optimizer worker settings. Editing a physical device or workspace at
the same filename is rejected before recomposition. Credential environment files
are excluded from these content identities.

Wave generation receives a bounded synopsis of the frozen memory snapshot,
prioritizing the current subject. It includes stable memory IDs/versions and
lesson summaries, preconditions and expectations, together with initial-guidance
preview and immutable hashes for the complete seed/snapshot. With KDB enabled,
the curriculum can retrieve reference and learned-memory evidence from that
same snapshot. Long result text is previewed; exact full text remains available
in retained artifacts, referenced by the query's search details. Background
memory admission cannot change an in-progress selection's memory view.

## Memory usefulness admission

New subject campaigns enable memory verification by default. The independent
memory verifier selects one small transfer project, freezes its host acceptance
contract, and runs fresh control and treatment Actors. The control sees accepted
memory; the treatment differs only by the proposed new or replacement memory.
The verifier reads retained results and Actor/tool logs before judging usefulness.
Equal scores may still support acceptance when logs demonstrate useful transfer;
invalid or incomplete runs never establish benefit.

A valid but inconclusive comparison can request a targeted follow-up with
`next_probe: {"question": "specific uncertainty and needed contrast"}`. The
curriculum selector authors a different neutral host contract; it does not put
the lesson's implementation advice into either Actor's task. Each probe has a
persisted ordinal, fresh arm directories and counterbalanced order. The judge
can read all earlier results and artifacts, including unfavorable evidence and
prior candidate revisions. Passing host checks does not preclude rejection.
Probes share the original admission elapsed allocation; there is no probe-count
stopping limit. A completed inconclusive result is cached and does not repeat
Actor or provider work when reopened.

Drafts stay outside accepted memory and ordinary KDB retrieval. Other waves may
start against the older accepted snapshot while verification runs. Acceptance
publishes a new immutable version for later waves. Appends and merges replace
specified accepted versions only after testing; history remains readable. A
concurrent replacement triggers reconciliation and a new comparison.

Unhelpful drafts return to the project verifier for up to two rewrites, using the
same frozen transfer task. Rejection after those opportunities leaves existing
memory untouched. Missing underlying evidence can instead send the original
project back to curriculum repair. Infrastructure failures remain inconclusive
and retain elapsed allocations and operation identities. An invalid trial is
held unpublished by default, including missing receipts and interrupted Actors.
An invalid or interrupted first arm also holds its unstarted peer.
Programmatic adapters can inject `MemoryAdmission(..., retry_ready=callback)`;
only a literal `True` certifying that all owned operations have settled and a
fresh trial is eligible permits a new isolated comparison. The callback receives
`(result, trial_directory, remaining_seconds=...)`. Default campaign composition
does not supply this certification, so it never blindly retries native work.
An unchanged completed uncertainty without a requested probe also stays held.

Optional settings are `memory_verifier_model` (defaults to the campaign model),
`memory_review_seconds` (3600), `memory_trial_seconds` (1800), and
`memory_arm_seconds` (600). Limits measure elapsed work and inherit the remaining
campaign allowance. `memory_verification: false` is available for project-review
smokes; ordinary campaigns should keep the default admission gate.

The `memory-reviews` directory retains draft versions, paired execution evidence,
judgments, and rewrite feedback. Accepted memory provenance contains the matched
evaluation identity and verifier trace hash. Full code and examples remain in
content-addressed artifact storage; snippets preserve exact source lines.

Project source audits and memory rewrites use the pinned Pi runtime for standard
live providers. Pi selects typed artifact-list/read, knowledge-query, optional
check, and review-submission tools; the Python host still validates artifact
identity, observed snippets, required evidence, citations and host correctness.
Missing-evidence feedback names the exact outstanding roles and SHA256 values.
If a verifier response emits multiple host tool calls, only its first queued
action is selected; later calls receive explicit rejection and have no effects.
The next host observation updates which evidence still needs reading. Pi receives
current review state, deduplicated retained artifact pages and the latest validation
feedback instead of replaying every prior action. Evidence fields contain bare
SHA256 values from the actually-read set; source snippet lines start at 1 and
citations must be exact objects returned by KDB (or an empty list). Malformed
reference formatting is reported separately from a valid but unread artifact.
`verifier_model` continues to select the model, including `deepseek-flash`.

`verifier_engine: "completion"` retains the JSON-completion adapter when needed;
injected/scripted provider transports use that adapter by default. Explicit
`verifier_engine: "pi"` also works with custom composition. The effective engine,
model and Pi runtime identity are persisted in `project-verifier-runtime.json`;
resume rejects a changed identity. Historical runtime directories without this
file retain completion delivery rather than silently switching engines. New
engine configurations require a new runtime directory. Pi does not change remote
job queues, host verification, or the independent A/B publication gates.

## Retrospective native runs

A3 campaigns may pass `compose_cpl_production` from
`benchmarks.ascend.retrospective_runtime` to `build_subject_campaign(compose=...)`.
The same ignored Phase1 settings select provider/model, reference KDB, guide and
workspace. The native JSON additionally requires `transport_target` and a named
registry `runtime`; `registry_workspace` is optional. This path never invokes the
legacy validation wrapper. Each fresh binding checks a known-good compile and
exact host oracle before Actor work or additional verifier execution. The check
consumes the current elapsed allocation. Optional `stop_file` belongs to this
campaign only. `campaign.check_environment(force=True)` explicitly refreshes the
check between batches; completed historical results do not require a device.

`campaign.review_retained(item, request, **review_options)` registers an immutable
historical graph plan before independent review, allowing extra host checks
without rerunning its original Actor. Recovery items may add
`recovery: {"reason": "...", "artifacts": {"role": "SHA256"}}`; referenced artifacts
must already exist in the campaign store. The Actor receives this original-failure
context separately from accepted learned memory, and the context is restored after
execution. Original project specs and frozen accepted-memory snapshots are unchanged.

The CPL adapter and environment probe retain reviewed behavior from PR186
(`48592246`) and PR200 (`37004d1`), respectively. Terminal result collection is
bounded read-only recovery: it cannot extend command execution or resubmit jobs.
