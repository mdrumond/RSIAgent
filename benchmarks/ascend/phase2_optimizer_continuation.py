"""Carry authenticated learning into a fresh fixed-goal Phase 2 campaign."""
from __future__ import annotations

import base64
from pathlib import Path

from optimizer.jobs import JobStore, canonical, sha
from optimizer.memory import verify_snapshot


def compose_prior_campaigns(reviewed_seed, predecessors):
    """Merge pinned READY predecessors without nesting entire campaign archives.

    Each predecessor's seed already contains its history, so normally only the
    latest root is needed. The original reviewed Phase 1 remains authoritative.
    """
    from .optimizer_phase2_artifacts import verify_campaign

    verify_snapshot(reviewed_seed)
    if not predecessors:
        return reviewed_seed
    original = reviewed_seed["snapshot_id"]
    files = dict(reviewed_seed["files"])
    entries = {sha(canonical(e)): e for e in reviewed_seed["memory_entries"]}

    def retain(name, data):
        files[name] = dict(sha256=sha(data), base64=base64.b64encode(data).decode())

    lineage = []
    for pinned in predecessors:
        root = Path(pinned["root"])
        before = (root / "state.json").read_bytes()
        if sha(before) != pinned["state_sha256"]:
            raise ValueError("prior Phase 2 state changed")
        state, seed = verify_campaign(root, require_ready=True)
        seed_origin = seed.get("reviewed_phase1_snapshot_id", seed["snapshot_id"])
        if (seed_origin != original or seed["source_identity"] != reviewed_seed["source_identity"]):
            raise ValueError("prior Phase 2 reviewed Phase 1 lineage differs")
        files.update(seed["files"])
        for entry in [*seed["memory_entries"], *state.get("memory_entries", [])]:
            entries[sha(canonical(entry))] = entry
        prefix = "prior-phase2/" + pinned["state_sha256"]
        retain(prefix + "/state.json", before)
        for row in state["attempts"]:
            retain(prefix + "/attempts/" + row["attempt_id"] + ".json", canonical(row))
            result = row["result"]
            store = JobStore(root / row["execution_directory"] / "jobs")
            for role in ("candidate", "baseline", "profiling", "result_artifact"):
                record = result.get(role)
                if record:
                    data = store.read_artifact(result["job_id"], record["sha256"])
                    retain("phase2/evidence/" + record["sha256"], data)
        if (root / "state.json").read_bytes() != before or (root / "pending.json").exists():
            raise ValueError("prior Phase 2 changed during continuation")
        lineage.append(dict(state_sha256=pinned["state_sha256"], terminal=state["terminal"]))
    body = {k: v for k, v in reviewed_seed.items() if k != "snapshot_id"}
    body.update(phase="continued-phase2", reviewed_phase1_snapshot_id=original,
                prior_campaigns=lineage, files=files, memory_entries=list(entries.values()))
    return {**body, "snapshot_id": sha(canonical(body))}
