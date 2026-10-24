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
