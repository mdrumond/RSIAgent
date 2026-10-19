import json

from dashboard.app import create_app
from dashboard.store import DashboardStore
from optimizer.jobs import JobStore, canonical


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def test_optimizer_attempts_sources_downloads_and_measurements(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    task = {"task_id": "new-matmul-residual", "submission": {"source": "torch reference"},
            "rationale": "Matmul with residual beyond the legacy graph grammar"}
    store = JobStore(root / "tasks/new-matmul-residual/jobs")
    key = store.create("reference", "f", {}, {})["job_id"]
    source = "// kernel source"
    artifact = store.put_artifact(key, "candidate.cpp", canonical(source))
    base = store.put_artifact(key, "baseline", canonical({"cases": [{"case": "case-0",
        "timing": {"torch": {"latency_us": 12}}}]}))
    attempts = []
    for index, verified in enumerate((False, True), 1):
        row = {"attempt_id": f"attempt-{index:03}", "task_id": task["task_id"], "kind": "target",
            "task": task, "execution_directory": "tasks/new-matmul-residual", "verified": verified,
            "result": {"job_id": key, "classification": "host-verified" if verified else "host-mismatch",
                "summary": "Target result", "candidate": artifact, "baseline": base,
                "result": {"cases": [{"case": "case-0", "timing": {"ascendc": {"latency_us": 6}}}]}}}
        # Retrying the same candidate reuses its old source artifact: the new
        # attempt's event window contains other artifacts but no candidate event.
        store.put_artifact(key, "attempt-note.json", canonical({"attempt": index}))
        store.add_event(key, "phase2-attempt-result", row["result"])
        attempts.append(row)
        write(root / "attempts" / (row["attempt_id"] + ".json"), row)
    write(root / "state.json", {"phase": "optimizer-phase2", "attempts": attempts,
        "target": task, "terminal": {"status": "READY"}, "memory_entries": [{"lesson": "repair"}]})
    config = tmp_path / "dashboard.json"
    write(config, {"campaigns": [{"phase": 2, "root": str(root)}]})
    view = DashboardStore(config, tmp_path / "cache").refresh()
    assert not view["stale"]
    assert view["campaigns"][0]["stats"] == {"attempted": 2, "verified": 1, "published": 1}
    assert [p["id"] for p in view["campaigns"][0]["projects"]] == ["attempt-001", "attempt-002"]
    assert [p["source"] for p in view["campaigns"][0]["projects"]] == [source, source]
    client = create_app(config, tmp_path / "webcache", refresh_seconds=0).test_client()
    page = client.get("/phase/2/project/attempt-002")
    assert page.status_code == 200
    assert b"Optimizer measurements" in page.data and b"12" in page.data
    assert b"kernel source" in page.data
    trace = client.get("/api/phase/2/project/attempt-002/trace").json
    event = next(e for e in trace["events"] if e["label"] == "candidate.cpp")
    response = client.get(event["download_url"])
    assert response.status_code == 200
    assert json.loads(response.data) == source
    assert client.get("/api/phase/2/project/missing/trace").status_code == 404


def test_optimizer_failed_selection_remains_visible(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    write(root / "goal-selection.json", {"stage": "failed", "proposal_count": 3,
        "candidate": {"task": {"task_id": "proposal"}, "rationale": "New composition"},
        "diagnostic": "Reference raised a shape mismatch"})
    config = tmp_path / "dashboard.json"
    write(config, {"campaigns": [{"phase": 2, "root": str(root)}]})
    view = DashboardStore(config, tmp_path / "cache").refresh()
    assert not view["stale"]
    assert view["campaigns"][0]["status"] == "failed"
    assert view["campaigns"][0]["bootstrap"]["diagnostic"] == "Reference raised a shape mismatch"
