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
are exposed for the memory verifier composition; admission remains caller-owned.

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

Drafts stay outside accepted memory and ordinary KDB retrieval. Other waves may
start against the older accepted snapshot while verification runs. Acceptance
publishes a new immutable version for later waves. Appends and merges replace
specified accepted versions only after testing; history remains readable. A
concurrent replacement triggers reconciliation and a new comparison.

Unhelpful drafts return to the project verifier for up to two rewrites, using the
same frozen transfer task. Rejection after those opportunities leaves existing
memory untouched. Missing underlying evidence can instead send the original
project back to curriculum repair. Infrastructure failures remain inconclusive
and retain elapsed allocations and operation identities for resume.

Optional settings are `memory_verifier_model` (defaults to the campaign model),
`memory_review_seconds` (3600), `memory_trial_seconds` (1800), and
`memory_arm_seconds` (600). Limits measure elapsed work and inherit the remaining
campaign allowance. `memory_verification: false` is available for project-review
smokes; ordinary campaigns should keep the default admission gate.

The `memory-reviews` directory retains draft versions, paired execution evidence,
judgments, and rewrite feedback. Accepted memory provenance contains the matched
evaluation identity and verifier trace hash. Full code and examples remain in
content-addressed artifact storage; snippets preserve exact source lines.
