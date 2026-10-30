# Ascend subject curriculum

`run_ascend_subjects.py` runs one ordered curriculum of exploration, repair,
qualified target and optimizer subjects. There is no separate targeted-practice
stage or handoff campaign. Current subject execution owns task retries and
retains host-authenticated evidence.

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
in retained artifacts, referenced by the query's search details. Publication cannot change an in-progress selection's memory view.

## Memory publication

Source-approved reusable memories publish immediately as immutable versions in
accepted memory and the KDB. Appends and merges replace specified active versions;
historical source and evidence remain readable. Concurrent replacement requires
source-review reconciliation before publication. No separate admission experiment
or usefulness gate runs, and no removed A/B configuration is supported.

Project source audits and memory rewrites use the pinned Pi runtime for standard
live providers. Verifier reads, artifact listings, KDB queries and context reads
execute sequentially inside Pi's native conversation and return authoritative
observations immediately, including when a response requests several reads.
`submit_review` and `request_check` must each be standalone assistant calls;
mixed batches reject those operations while permitting their read-only calls.
Only a host-validated accept/repair decision ends the review. A model ending its
message without a verdict receives current state and may continue within its
remaining elapsed allocation; it is never automatically accepted or rejected.

The Python host validates original tool arguments before Pi coercion, artifact
identity, required evidence, observed snippets, citations and host correctness.
Read pages are at most 2 KiB with UTF-8 continuation offsets; large listings and
query results have lossless paged records with previews bounded to 4 KiB.
Review-wide immutable byte caching allows legitimate rereads and reports their
roles, coverage, cache hits and repeats. Completed prerequisites guide the model
toward a decision without relaxing admission gates or introducing request-count
stopping budgets. Raw proposed calls, dispositions and conversation are retained.

The gateway projects the entire serialized Chat/Responses request, including
instructions, schemas and current host state, before its existing hard byte
limit. Older complete tool exchanges become lossless `read_context` records;
the latest complete batch, host controls and known counterevidence remain visible.
Actor tool selection, default context packing and gateway behavior are unchanged.

`verifier_model` still selects the model, including `deepseek-flash`.
`verifier_engine: "completion"` retains the JSON-completion adapter; injected
transports default to that adapter. Explicit `verifier_engine: "pi"` supports
custom composition. Effective engine/model and native loop/context identities
are persisted in `project-verifier-runtime.json`; changed identities fail resume.
Historical directories without this metadata retain completion delivery, including
subsequent resumes. New Pi policies require a new runtime directory. Pi does not
change remote job queues.

## Retrospective native runs

A3 campaigns may pass `compose_cpl_production` from
`benchmarks.ascend.retrospective_runtime` to `build_subject_campaign(compose=...)`.
The same ignored Phase1 settings select provider/model, reference KDB, guide and
workspace. The native JSON additionally requires `transport_target` and a named
registry `runtime`; `registry_workspace` is optional. Configure `health_workspace`
as an existing absolute subdirectory of the registered workspace for read-only
checks before uploads. Pin product, architecture, compiler, CANN, Torch,
Torch-NPU and SoC identities. Optional `allowed_devices` restricts physical IDs. This path never invokes the
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

The latest complete batch and mandatory host controls are not silently dropped
to fit a request. If those inputs plus instructions and schemas alone exceed the
provider cap, the review is held as infrastructure/context failure before dispatch;
its raw evidence remains retained. This is a per-request capacity constraint,
not a request-count or token-based stopping budget.

Transport journals are retained per execution allocation. On admission, bounded
read-only observation settles owned retained handles before permitting new work;
ambiguous submissions and still-running operations continue to block admission.
Changing a permitted device preserves old journals and uses a distinct namespace.
