"""Carry authenticated learning into a fresh fixed-goal Phase 2 campaign."""
from __future__ import annotations

import base64
import json
from pathlib import Path

from optimizer.jobs import JobStore, canonical, sha
from optimizer.memory import verify_snapshot


def _verify_origin(seed, reviewed_seed):
    verify_snapshot(seed)
    original = reviewed_seed["snapshot_id"]
    origin = seed.get("reviewed_phase1_snapshot_id", seed["snapshot_id"])
    reviewed_entries = {sha(canonical(e)) for e in reviewed_seed["memory_entries"]}
    inherited_entries = {sha(canonical(e)) for e in seed["memory_entries"]}
    if (origin != original or seed["source_identity"] != reviewed_seed["source_identity"]
            or not reviewed_entries <= inherited_entries
            or any(seed["files"].get(name) != record
                   for name, record in reviewed_seed["files"].items())):
        raise ValueError("prior Phase 2 reviewed Phase 1 lineage differs")
    # Continued lessons must still have their portable committed row and source
    # evidence, not merely a copied claim about the original reviewed snapshot.
    supported = set()
    for name, record in seed["files"].items():
        if not (name.startswith("prior-phase2/") and name.endswith("/state.json")):
            continue
        data = base64.b64decode(record["base64"], validate=True)
        state = json.loads(data)
        if name != "prior-phase2/" + sha(data) + "/state.json":
            raise ValueError("prior Phase 2 portable state mismatch")
        rows = {sha(canonical(row)): row for row in state["attempts"]}
        for entry in state.get("memory_entries", []):
            row = rows.get(entry.get("evidence_sha256"))
            if row is None or not row.get("verified"):
                raise ValueError("prior Phase 2 portable lesson missing committed evidence")
            for role in ("candidate", "baseline", "profiling", "result_artifact"):
                artifact = row["result"].get(role)
                if artifact and "phase2/evidence/" + artifact["sha256"] not in seed["files"]:
                    raise ValueError("prior Phase 2 portable lesson missing artifact")
            supported.add(sha(canonical(entry)))
    if not inherited_entries - reviewed_entries <= supported:
        raise ValueError("prior Phase 2 inherited lesson missing portable provenance")


def campaign_seed(root, reviewed_seed, predecessors):
    """Resume solely from the portable retained seed; read prior roots once."""
    verify_snapshot(reviewed_seed)
    path = Path(root) / "seed-memory.json"
    if not path.exists():
        return compose_prior_campaigns(reviewed_seed, predecessors)
    seed = json.loads(path.read_bytes())
    _verify_origin(seed, reviewed_seed)
    return seed


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
        _verify_origin(seed, reviewed_seed)
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
