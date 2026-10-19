import json

import pytest

from benchmarks.ascend.phase2_optimizer_continuation import compose_prior_campaigns
from optimizer.jobs import canonical, sha
from optimizer.memory import verify_snapshot
from test_optimizer_phase2_artifacts import campaign, write


def pin(root):
    return dict(root=str(root), state_sha256=sha((root / "state.json").read_bytes()))


def test_continuation_retains_authenticated_learning_without_raw_job_nesting(tmp_path):
    first = tmp_path / "first"
    state, _, _ = campaign(first)
    original = json.loads((first / "seed-memory.json").read_bytes())
    continued = compose_prior_campaigns(original, [pin(first)])
    verify_snapshot(continued)
    assert continued["reviewed_phase1_snapshot_id"] == original["snapshot_id"]
    assert len(continued["memory_entries"]) == 2
    assert any(p.startswith("prior-phase2/") for p in continued["files"])
    candidate = state["attempts"][0]["result"]["candidate"]["sha256"]
    assert "phase2/evidence/" + candidate in continued["files"]
    assert not any("jobs/" in p or "seed-memory" in p for p in continued["files"])

    second = tmp_path / "second"
    state2, _, _ = campaign(second, job_seed=continued)
    write(second / "seed-memory.json", continued)
    state2["snapshot_id"] = continued["snapshot_id"]
    state2["terminal"]["reason"] = "Second goal verified"
    state2["memory_entries"][0]["interpretation"] = "Another learned lesson"
    write(second / "state.json", state2)
    cumulative = compose_prior_campaigns(original, [pin(second)])
    verify_snapshot(cumulative)
    assert len(cumulative["memory_entries"]) == 3
    assert set(continued["files"]) <= set(cumulative["files"])
    # Shared evidence is content-addressed, rather than copied per goal.
    # Two namespaced state/row files and a new job-specific result artifact;
    # the identical candidate and baseline remain single CAS objects.
    assert len(cumulative["files"]) == len(continued["files"]) + 3
    assert compose_prior_campaigns(original, [pin(first), pin(second)])["memory_entries"] == cumulative["memory_entries"]
    assert compose_prior_campaigns(original, []) is original


@pytest.mark.parametrize("failure", ["changed", "pending", "stalled", "lineage", "identity", "artifact"])
def test_continuation_rejects_incompatible_or_uncommitted_predecessor(tmp_path, failure):
    root = tmp_path / "prior"
    state, store, key = campaign(root)
    original = json.loads((root / "seed-memory.json").read_bytes())
    pinned = pin(root)
    if failure == "changed":
        state["terminal"]["reason"] = "Changed after pinning"
        write(root / "state.json", state)
    elif failure == "pending":
        write(root / "pending.json", {})
    elif failure == "stalled":
        state["terminal"]["status"] = "STALLED"
        write(root / "state.json", state)
        pinned = pin(root)
    elif failure in {"lineage", "identity"}:
        if failure == "lineage":
            original["memory_entries"].append({"other": "reviewed learning"})
        else:
            original["source_identity"] = {"target": "other"}
        body = {k: v for k, v in original.items() if k != "snapshot_id"}
        original["snapshot_id"] = sha(canonical(body))
    else:
        record = state["attempts"][0]["result"]["candidate"]
        store.artifact_path(key, record["sha256"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        compose_prior_campaigns(original, [pinned])
