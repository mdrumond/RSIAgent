import base64
import json
import subprocess
import time

import pytest

from dashboard.app import create_app
from dashboard.store import DashboardStore


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture()
def campaign(tmp_path):
    root = tmp_path / "phase1"
    branch = root / "waves/wave-001/branches/vector-add"
    spec = {"project_id": "vector-add", "hypothesis": "Measure a bounded vector add.",
            "capabilities": ["simd-vector"], "shape": [1003]}
    write(root / "state.json", {"completed_waves": ["wave-001"], "prior_outcomes": [{}],
                                 "failed_outcomes": [], "project_target": 2})
    write(root / "waves/wave-001/plan.json", {"wave": {"projects": [{"spec": spec}]}})
    write(root / "waves/wave-001/published.json", {"projects": ["vector-add"]})
    write(branch / "outcome.json", {"project_id": "vector-add", "verified": True,
                                     "classification": "host-verified"})
    memory = {"project_spec": spec, "source_revision": "a" * 64,
              "actions": ["write", "compile", "run", "submit"],
              "host_facts": [{"category": "performance-study", "statement": json.dumps(
                  {"timing": [{"measurement": "launch-plus-synchronize", "median_us": 12.5,
                                "samples_us": [12, 13], "shape": [1003]}]})}],
              "agent_lessons": [{"category": "shape", "claim": "Tail remained bounded.",
                                   "confidence": "high"}], "agent_interpretations": []}
    write(branch / "memory-delta.bin", {"summary": "Verified padded vector add.",
                                         "memory_base64": base64.b64encode(
                                             json.dumps(memory).encode()).decode()})
    obj = branch / "artifacts/objects" / ("a" * 2) / ("a" * 62)
    obj.parent.mkdir(parents=True)
    obj.write_text("extern C { void kernel() {} }\n")
    old = "b" * 64
    old_obj = branch / "artifacts/objects" / old[:2] / old[2:]
    old_obj.parent.mkdir(parents=True)
    old_obj.write_text("extern C { void old_kernel() {} }\n")
    response = "c" * 64
    response_obj = branch / "artifacts/objects" / response[:2] / response[2:]
    response_obj.parent.mkdir(parents=True)
    response_obj.write_text(json.dumps({"write": {"slot": "kernel", "content":
                                                   "extern C { void first_draft() {} }\n"}}))
    (branch / "evidence.jsonl").write_text(
        json.dumps({"payload": {"role": "provider-response", "sha256": response}}) + "\n" +
        json.dumps({"payload": {"event": "candidate", "source_sha256": old}}) + "\n")
    control = tmp_path / "control"
    write(control / "worker.json", {"status": "running", "started": 10})
    config = tmp_path / "dashboard.json"
    write(config, {"campaigns": [{"phase": 1, "name": "A3 Phase 1", "root": str(root),
                                   "control": str(control)},
                                  {"phase": 2, "name": "A3 Phase 2",
                                   "root": str(tmp_path / "missing")}]})
    return config


def test_refresh_extracts_projects_memory_performance_and_revisions(campaign, tmp_path):
    store = DashboardStore(campaign, tmp_path / "cache.sqlite3")
    view = store.refresh()
    project = view["campaigns"][0]["projects"][0]
    assert view["campaigns"][0]["stats"] == {"attempted": 1, "verified": 1, "published": 1}
    assert project["description"] == "Measure a bounded vector add."
    assert project["summary"] == "Verified padded vector add."
    assert project["performance"][0]["median_us"] == 12.5
    assert project["memory"]["lessons"][0]["claim"] == "Tail remained bounded."
    assert project["revisions"][-1]["source"].startswith("extern C")
    assert "first_draft" in project["revisions"][0]["source"]
    assert project["diff"].startswith("---")
    assert view["campaigns"][0]["worker"]["status"] == "running"


def test_refresh_failure_serves_last_cached_snapshot(campaign, tmp_path):
    store = DashboardStore(campaign, tmp_path / "cache.sqlite3")
    assert store.refresh()["stale"] is False
    campaign.write_text("not json")
    stale = store.refresh()
    assert stale["stale"] is True
    assert stale["campaigns"][0]["projects"][0]["id"] == "vector-add"


def test_routes_render_escaped_data_and_project_views(campaign, tmp_path):
    app = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0)
    client = app.test_client()
    assert client.get("/").status_code == 200
    phase = client.get("/phase/1")
    assert b"vector-add" in phase.data and b"A3 Phase 1" in phase.data
    detail = client.get("/phase/1/project/vector-add")
    assert detail.status_code == 200
    assert b"Historic code evolution" in detail.data and b"Tail remained bounded" in detail.data
    assert b"void kernel" in detail.data and b"language-cpp" in detail.data
    assert client.get("/memory").status_code == 200
    assert client.get("/api/status").json["campaigns"][0]["worker"]["status"] == "running"
    assert client.get("/phase/1/project/../../etc/passwd").status_code == 404


def test_default_bind_is_loopback(monkeypatch, campaign, tmp_path):
    from dashboard import __main__
    called = {}
    monkeypatch.setattr(__main__, "create_app", lambda *a, **k: type(
        "App", (), {"run": lambda self, **kw: called.update(kw)})())
    __main__.main(["serve", "--config", str(campaign), "--cache-dir", str(tmp_path)])
    assert called["host"] == "127.0.0.1" and called["port"] == 8080


def controlled_config(tmp_path, *, phase=1):
    root, control = tmp_path / f"phase{phase}", tmp_path / f"control{phase}"
    config = tmp_path / "controlled.json"
    preset = {"id": f"phase{phase}", "phase": phase, "root": str(root),
              "control": str(control), "config": str(tmp_path / "runtime.json"),
              "initial_memory": str(tmp_path / "memory.bin"), "objective": "Learn A3",
              "project_target": 4}
    if phase == 2:
        preset.update(phase1_root=str(tmp_path / "phase1"), review=str(tmp_path / "review.json"))
    write(config, {"campaigns": [{"id": f"phase{phase}", "phase": phase,
                                   "root": str(root), "control": str(control)}],
                   "controls": {"enabled": True, "runner": "/repo/run_ascend_campaign.py",
                                "presets": [preset]}})
    return config, root, control


def wait_operation(client, operation_id):
    for _ in range(100):
        operation = client.get(f"/api/operations/{operation_id}").get_json()
        if operation["status"] not in {"queued", "running"}:
            return operation
        time.sleep(.01)
    raise AssertionError("operation did not finish")


def test_controls_disabled_by_default_and_csrf_required(campaign, tmp_path):
    app = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0)
    client = app.test_client()
    assert b"Start phase" not in client.get("/").data
    assert client.post("/api/control/sample", json={"campaign_id": "phase1"}).status_code == 403

    config, _, _ = controlled_config(tmp_path)
    client = create_app(config, tmp_path / "controlled.sqlite3", command_runner=lambda *a, **k: None).test_client()
    assert client.post("/api/control/start", json={"preset_id": "phase1",
                       "confirmation": "start:phase1"}).status_code == 403


def test_start_is_async_allowlisted_and_idempotent(tmp_path):
    config, _, _ = controlled_config(tmp_path, phase=2)
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, '{"session":"test"}\n', "")
    app = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0, command_runner=run)
    client = app.test_client()
    assert b"Start phase 2" in client.get("/").data
    csrf = client.get("/api/csrf").get_json()["token"]
    headers = {"X-CSRF-Token": csrf, "Idempotency-Key": "start-once"}
    body = {"preset_id": "phase2", "confirmation": "start:phase2"}
    first = client.post("/api/control/start", json=body, headers=headers)
    second = client.post("/api/control/start", json=body, headers=headers)
    assert first.status_code == second.status_code == 202
    assert first.get_json()["operation_id"] == second.get_json()["operation_id"]
    assert wait_operation(client, first.get_json()["operation_id"])["status"] == "succeeded"
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[:3] == [pytest.importorskip("sys").executable, "/repo/run_ascend_campaign.py", "start"]
    assert "--target" in argv and argv[argv.index("--target") + 1] == "a3"
    assert "--approval-mode" in argv and "--phase1-root" in argv and "--review" in argv
    assert kwargs["shell"] is False
    conflict = client.post("/api/control/start", json={**body, "confirmation": "wrong"}, headers=headers)
    assert conflict.status_code == 409


def test_sample_summary_approve_and_run_without_raw_provider_content(tmp_path):
    config, root, control = controlled_config(tmp_path)
    root.mkdir()
    write(root / "state.json", {"coverage": ["gm-ub-movement"], "prior_outcomes": [],
                                  "wave_index": 1})
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        command = argv[2]
        if command == "sample":
            proposal = {"base_wave_index": 1, "base_memory_sha256": "a" * 64,
                        "session": {"messages": [{"content": "private reasoning"}]},
                        "wave": {"projects": [
                            {"spec": {"project_id": "copy", "hypothesis": "Measure copy.",
                                      "capabilities": ["gm-ub-movement"]}},
                            {"spec": {"project_id": "vector", "hypothesis": "Measure vector.",
                                      "capabilities": ["simd-vector"]}}]}}
            write(control / "curriculum-approval.json", {"status": "proposed", "phase": 1,
                  "proposal_id": "proposal-1", "proposal": proposal})
        elif command == "approve":
            record = json.loads((control / "curriculum-approval.json").read_text())
            write(control / "curriculum-approval.json", {**record, "status": "approved",
                  "selected_project_ids": [argv[argv.index("--project-id") + 1]]})
        return subprocess.CompletedProcess(argv, 0, "{}\n", "")
    client = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0,
                        command_runner=run).test_client()
    csrf = client.get("/api/csrf").get_json()["token"]
    def post(action, body, key):
        response = client.post(f"/api/control/{action}", json=body,
            headers={"X-CSRF-Token": csrf, "Idempotency-Key": key})
        assert response.status_code == 202
        return wait_operation(client, response.get_json()["operation_id"])
    assert post("sample", {"campaign_id": "phase1", "confirmation": "sample:phase1"}, "s")["status"] == "succeeded"
    status = client.get("/api/status").get_json()
    proposals = status["controls"]["proposals"]
    assert proposals[0]["projects"][1]["new_coverage"] == ["simd-vector"]
    assert proposals[0]["projects"][0]["hypothesis"] == "Measure copy."
    assert "private reasoning" not in json.dumps(status)
    approval_body = {"campaign_id": "phase1", "proposal_id": "proposal-1",
                     "project_ids": ["vector"], "confirmation": "approve:proposal-1"}
    assert post("approve", approval_body, "a")["status"] == "succeeded"
    replay = client.post("/api/control/approve", json=approval_body,
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": "a"})
    assert replay.status_code == 202
    assert post("run-approved", {"campaign_id": "phase1", "proposal_id": "proposal-1",
        "confirmation": "run-approved:proposal-1"}, "r")["status"] == "succeeded"
    assert calls[-1][2] == "run-approved"


def test_selection_staleness_phase2_cardinality_failures_and_restart(tmp_path):
    config, root, control = controlled_config(tmp_path, phase=2)
    root.mkdir()
    write(root / "state.json", {"coverage": [], "prior_outcomes": [], "wave_index": 2})
    write(control / "curriculum-approval.json", {"status": "proposed", "phase": 2,
          "proposal_id": "p2", "proposal": {"base_wave_index": 1,
          "base_memory_sha256": "a" * 64, "wave": {"projects": [
              {"spec": {"project_id": "one", "hypothesis": "One", "capabilities": []}},
              {"spec": {"project_id": "two", "hypothesis": "Two", "capabilities": []}}]}}})
    cache = tmp_path / "cache.sqlite3"
    app = create_app(config, cache, refresh_seconds=0, command_runner=lambda *a, **k: None)
    client = app.test_client()
    csrf = client.get("/api/csrf").get_json()["token"]
    headers = {"X-CSRF-Token": csrf, "Idempotency-Key": "approval"}
    body = {"campaign_id": "phase2", "proposal_id": "p2", "project_ids": ["one", "two"],
            "confirmation": "approve:p2"}
    assert client.post("/api/control/approve", json=body, headers=headers).status_code == 409
    body["project_ids"] = ["one"]
    assert client.post("/api/control/approve", json=body,
        headers={**headers, "Idempotency-Key": "stale"}).status_code == 409

    manager = app.extensions["dashboard_controls"]
    operation = manager._insert_for_test("sample", "phase2", "running")
    manager.close()
    restarted = create_app(config, cache, refresh_seconds=0, command_runner=lambda *a, **k: None)
    interrupted = restarted.test_client().get(f"/api/operations/{operation}").get_json()
    assert interrupted["status"] == "failed" and "interrupted" in interrupted["error"]


def test_reject_uses_only_the_retained_proposal(tmp_path):
    config, root, control = controlled_config(tmp_path)
    root.mkdir()
    write(root / "state.json", {"coverage": [], "prior_outcomes": [], "wave_index": 1})
    write(control / "curriculum-approval.json", {"status": "proposed", "phase": 1,
          "proposal_id": "reject-me", "proposal": {"base_wave_index": 1,
          "base_memory_sha256": "a" * 64, "wave": {"projects": [
              {"spec": {"project_id": "one", "hypothesis": "One", "capabilities": []}}]}}})
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "{}", "")
    client = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0,
                        command_runner=run).test_client()
    csrf = client.get("/api/csrf").get_json()["token"]
    response = client.post("/api/control/reject", json={"campaign_id": "phase1",
        "proposal_id": "reject-me", "confirmation": "reject:reject-me"},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": "reject"})
    assert response.status_code == 202
    assert wait_operation(client, response.get_json()["operation_id"])["status"] == "succeeded"
    assert calls[0][2:] == ["reject", "--control", str(control),
                            "--proposal-id", "reject-me"]


def test_background_failure_is_sanitized_and_visible(tmp_path):
    config, _, _ = controlled_config(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("provider unavailable SECRET_TOKEN")
    client = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0,
                        command_runner=fail).test_client()
    csrf = client.get("/api/csrf").get_json()["token"]
    response = client.post("/api/control/sample", json={"campaign_id": "phase1",
        "confirmation": "sample:phase1"}, headers={"X-CSRF-Token": csrf,
        "Idempotency-Key": "failure"})
    operation = wait_operation(client, response.get_json()["operation_id"])
    assert operation["status"] == "failed"
    assert "provider unavailable" in operation["error"]
    assert "SECRET_TOKEN" not in operation["error"]
