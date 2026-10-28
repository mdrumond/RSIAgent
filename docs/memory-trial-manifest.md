# Inspect a completed memory trial

Export a readable local view after the memory verifier retains a judgment:

```bash
python -m benchmarks.ascend.experiment_manifest export \
  --trial /local/admission/revision-0/probe-1 \
  --output /local/inspection
```

The output directory must not already exist, its parent must exist, and it must
be outside the trial. The command prints the path to `manifest.json` and exits
with code 2 if the trial is active, has no completed judgment, or lacks required
evidence. A retained inconclusive judgment is exportable, including a trial that
never selected a task or started its second arm.

The directory contains `manifest.json`, `task.json` when a task was selected, and
`control-memory.json` / `treatment-memory.json`. Non-JSON memory is copied as
`.txt` instead. JSON snapshots are formatted for reading. The source records,
admission decisions and execution state are unchanged.

For example, the main fields look like this (the ID and source paths are
illustrative; the exporter also includes evidence locations and a scope note):

```json
{
  "evaluation_id": "existing-evaluation-id",
  "model": {"model": "provider/model"},
  "runtime": {"target": "a3", "product": "Ascend910", "cann_version": "9.1"},
  "task": "task.json",
  "memory": {"control": "control-memory.json", "treatment": "treatment-memory.json"},
  "sequence": 1,
  "order": ["treatment", "control"],
  "allocation_seconds": {"trial": 3500, "configured_trial": 3600, "configured_arm": 1200, "arm": 1100},
  "result": {"decision": "inconclusive", "reason": "Incomplete arm", "validity": {"treatment": true}},
  "cited_evidence": ["treatment:log:actor"]
}
```

Evidence and proposal paths are relative to the exported directory and point
into the original trial's retained objects; they are not copied. Keep both
directories in their relative locations. This is a local inspection view, not a
portable evidence archive or a new verification protocol. The existing
evaluation ID remains unchanged; the exporter computes no new hashes.

Model and runtime summaries use a small allowlist of descriptive fields and
omit absolute paths and URLs. Full identities stay in the referenced source
evaluation. Task, memory and existing evidence are retained content, so inspect
them before sharing an export; this command is not a general redaction tool.
