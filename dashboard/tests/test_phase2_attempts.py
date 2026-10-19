"""Attempt-scoped Phase 2 views, independent of the execution dependencies."""

import base64
import hashlib
import json

import pytest

from dashboard.app import create_app
from dashboard.store import DashboardStore


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup(tmp_path):
    root = tmp_path / "phase2"
    root.mkdir()
    config = tmp_path / "config.json"
    write(config, {"campaigns": [{"phase": 2, "root": str(root)}]})
    store = DashboardStore(config, tmp_path / "cache.sqlite3")
    client = create_app(config, tmp_path / "web.sqlite3", refresh_seconds=0).test_client()
    return root, store, client


@pytest.mark.parametrize("stage", ["selection", "qualification", "qualified", "failed"])
def test_bootstrap_visible_before_learner_state_without_private_reference(tmp_path, stage):
    root, store, client = setup(tmp_path)
    write(root / "goal-selection.json", {
        "stage": stage, "proposal_count": 2,
        "candidate": {"spec": {"project_id": "new-goal", "hypothesis": "New task"},
                      "rationale": "Inspired by reference kernels",
                      "citations": ["ops-example/kernel.cpp:24"],
                      "reference_source": "private candidate source"},
        "qualified_target_id": "qualified-123" if stage == "qualified" else None,
        "diagnostic": "Host verification pending",
        "private_conversation": "private reference discussion"})
    write(root / "control/bootstrap/reference.json", {"source": "private actor source"})
    view = store.refresh()
    campaign = view["campaigns"][0]
    assert not view["stale"]
    assert campaign["status"] == stage
    assert campaign["stats"] == {"attempted": 0, "verified": 0, "published": 0}
    public = json.dumps(client.get("/api/status").json)
    assert "private" not in public
    page = client.get("/phase/2").text
    assert "Goal bootstrap: " + stage in page
    assert "ops-example/kernel.cpp:24" in page
    assert "qualification is separate" in page
    assert client.get("/phase/2/project/new-goal").status_code == 404
    assert client.get("/api/phase/2/project/new-goal/trace").status_code == 404


def populate_attempt(root, wave, project_id, kind, verified, sample):
    spec = {"project_id": project_id, "hypothesis": "Try unchanged target",
            "dtype": "float32", "shape": [32]}
    write(root / f"waves/{wave}/plan.json", {"wave": {"projects": [{"spec": spec}]}})
    branch = root / f"waves/{wave}/branches/{project_id}"
    write(branch / "outcome.json", {"verified": verified, "classification": (
        "host-verified" if verified else "host-mismatch")})
    source = f"// kernel for {wave}\n"
    digest = hashlib.sha256(source.encode()).hexdigest()
    obj = branch / "artifacts/objects" / digest[:2] / digest[2:]
    obj.parent.mkdir(parents=True)
    obj.write_text(source)
    memory = {"source_revision": digest, "host_facts": [{
        "category": "performance-study", "statement": json.dumps({"timing": [{
            "measurement": "launch-plus-synchronize", "shape": [32],
            "samples_us": [sample]}]})}]}
    write(branch / "memory-delta.bin", {"summary": wave,
        "memory_base64": base64.b64encode(json.dumps(memory).encode()).decode()})
    write(branch / "execution-evidence.bin", {"attempt": wave, "verified": verified})
    return {"wave_id": wave, "project_id": project_id, "kind": kind,
            "verified": verified, "summary": wave}


def test_target_retries_keep_exact_sources_traces_and_performance(tmp_path):
    root, store, client = setup(tmp_path)
    attempts = [populate_attempt(root, "attempt-001", "goal", "target", False, 30),
                populate_attempt(root, "attempt-002", "practice", "practice", True, 20),
                populate_attempt(root, "attempt-003", "goal", "target", True, 10)]
    write(root / "state.json", {"phase": "target-driven-phase2", "attempts": attempts,
        "prior_outcomes": attempts[1:], "failed_outcomes": [attempts[0]],
        "terminal": {"status": "READY", "reason": "Target host verified"}})
    campaign = store.refresh()["campaigns"][0]
    assert campaign["stats"] == {"attempted": 3, "verified": 2, "published": 2}
    assert [row["kind"] for row in campaign["projects"]] == ["target", "practice", "target"]
    assert [row["project_id"] for row in campaign["projects"]] == ["goal", "practice", "goal"]
    for wave, project_id, sample in [("attempt-001", "goal", 30),
                                     ("attempt-002", "practice", 20),
                                     ("attempt-003", "goal", 10)]:
        identity = f"{wave}--{project_id}"
        page = client.get(f"/phase/2/project/{identity}")
        assert page.status_code == 200
        assert f"// kernel for {wave}" in page.text
        assert f"project {project_id}" in page.text
        base = f"/api/phase/2/project/{identity}"
        perf = client.get(base + "/performance").json
        assert perf["captures"][0]["statistics"]["median_us"] == sample
        events = client.get(base + "/trace").json["events"]
        verifier = next(row for row in events if row["label"] == "Verifier result")
        payload = client.get(verifier["payload_url"]).json
        assert json.loads(payload["text"])["attempt"] == wave
        assert client.get(verifier["download_url"]).data == payload["text"].encode()
        assert next(row for row in events if row["label"] == "Shared host ledger")["available"] is False
    assert client.get("/phase/2/project/goal").status_code == 404
    assert client.get("/api/phase/2/project/goal/trace").status_code == 404
    assert "READY" in client.get("/").text
    assert "Target host verified" in client.get("/phase/2").text


@pytest.mark.parametrize("status", ["READY", "STALLED", "BUDGET_EXHAUSTED"])
def test_terminal_status_wins_over_old_running_worker(tmp_path, status):
    root, store, client = setup(tmp_path)
    write(root / "state.json", {"phase": "target-driven-phase2", "attempts": [],
        "prior_outcomes": [], "terminal": {"status": status, "reason": "End reason"}})
    write(root.parent / "control/worker.json", {"status": "running"})
    assert store.refresh()["campaigns"][0]["status"] == status
    assert status in client.get("/").text
    assert "End reason" in client.get("/phase/2").text


def test_pending_attempt_kind_and_infrastructure_worker_status(tmp_path):
    root, store, client = setup(tmp_path)
    populate_attempt(root, "attempt-001", "goal", "target", False, 12)
    (root / "waves/attempt-001/branches/goal/outcome.json").unlink()
    write(root / "state.json", {"phase": "target-driven-phase2", "attempts": [],
                               "prior_outcomes": [], "terminal": None})
    write(root / "pending.json", {"kind": "target", "wave": {"wave_id": "attempt-001"}})
    write(root.parent / "control/worker.json", {"status": "infrastructure-failed"})
    campaign = store.refresh()["campaigns"][0]
    assert campaign["status"] == "infrastructure-failed"
    assert campaign["projects"][0]["kind"] == "target"
    assert campaign["projects"][0]["classification"] == "running"
    assert "infrastructure-failed" in client.get("/phase/2").text


def test_unavailable_attempt_artifact_does_not_route_to_another_attempt(tmp_path):
    root, store, client = setup(tmp_path)
    attempt = populate_attempt(root, "attempt-001", "goal", "target", False, 12)
    write(root / "state.json", {"phase": "target-driven-phase2", "attempts": [attempt]})
    (root / "waves/attempt-001/branches/goal/execution-evidence.bin").unlink()
    events = client.get("/api/phase/2/project/attempt-001--goal/trace").json["events"]
    assert not next(row for row in events if row["label"] == "Verifier result")["available"]
    assert client.get("/api/phase/2/project/attempt-002--goal/trace").status_code == 404


def test_malformed_bootstrap_keeps_last_cached_view_stale(tmp_path):
    root, store, _ = setup(tmp_path)
    write(root / "goal-selection.json", {"stage": "selection"})
    assert not store.refresh()["stale"]
    write(root / "goal-selection.json", {"stage": "not-a-real-stage"})
    view = store.refresh()
    assert view["stale"]
    assert view["campaigns"][0]["bootstrap"]["stage"] == "selection"
