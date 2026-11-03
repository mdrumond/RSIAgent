"""Behavioral contract tests; no running CLI, provider, or dashboard required."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ascend_tools.dashboard import create_app


@pytest.fixture
def harness(monkeypatch, tmp_path):
    calls = []
    config = {"format": 1, "environment": {"options": {"runtime": "named"}},
              "projects": [], "unknown": {"keep": True}}
    state = {"projects": [{"id": "p", "status": "queued"}], "attempts": []}
    replies = {}

    def run(argv, **kwargs):
        assert kwargs == {"capture_output": True, "text": True, "timeout": 60, "check": False}
        assert argv[:4] == ["node-test", str(tmp_path / "machinery/cli/main.mjs"),
                            argv[2], "--instance"]
        assert argv[4] == str(tmp_path / "instance with spaces")
        payload = None
        for flag in ("--config", "--project"):
            if flag in argv:
                payload = json.loads(Path(argv[argv.index(flag) + 1]).read_text())
        calls.append((argv, payload))
        command = argv[2]
        if command in replies:
            response = replies[command]
            if isinstance(response, Exception):
                raise response
            return response
        if command == "config":
            if payload is not None:
                config.clear()
                config.update(payload)
            value = config.copy()
        elif command == "status":
            value = state
        else:
            value = {"command": command, "accepted": True}
        return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    app = create_app(tmp_path / "instance with spaces", tmp_path / "machinery", "node-test")
    app.testing = True
    return app.test_client(), calls, config, replies, state


def test_settings_and_project_propagation(harness):
    client, calls, config, _, _ = harness
    original = client.get("/api/config").json
    settings = {**original, "baselineKDB": "/data/a b.sqlite", "skillsmap": {"design": "x"},
                "curriculum": ["add"], "actors": {"seconds": 240}}
    assert client.put("/api/config", json=settings).json == settings
    assert config == settings
    project = {"id": "p; literal", "goal": "g", "actor_seconds": 240}
    assert client.post("/api/project", json=project).status_code == 200
    assert calls[-1][1] == project
    assert "--project" in calls[-1][0]
    assert not Path(calls[-1][0][-1]).exists()


@pytest.mark.parametrize("mode", ["phase1", "phase2", "inference", "revisit"])
def test_wave_detaches_and_passes_mode(harness, mode):
    client, calls, *_ = harness
    assert client.post("/api/wave", json={"mode": mode}).status_code == 200
    assert calls[-1][0][-3:] == ["--detach", "--mode", mode]
    assert calls[-1][0][2] == "wave"


def test_stop_resume_and_status(harness):
    client, calls, _, _, state = harness
    assert client.get("/api/status").json == state
    assert client.post("/api/stop").status_code == 200
    assert calls[-1][0][2] == "stop"
    assert "--detach" not in calls[-1][0]
    assert client.post("/api/resume").status_code == 200
    assert calls[-1][0][2] == "resume"
    assert calls[-1][0][-1] == "--detach"


def test_tune_persists_userinfo_before_action(harness):
    client, calls, config, *_ = harness
    action = {"userinfo": "preferences\nkeep this", "action": "skilltune"}
    assert client.post("/api/tune", json=action).status_code == 200
    assert [call[0][2] for call in calls] == ["config", "config", "tune"]
    assert calls[1][1]["userinfo"] == action["userinfo"]
    assert config["unknown"] == {"keep": True}
    assert calls[2][1] == action


def test_failed_config_prevents_tuning(harness):
    client, calls, _, replies, _ = harness
    replies["config"] = SimpleNamespace(returncode=1, stdout="", stderr='{"error":"unavailable"}')
    assert client.post("/api/tune", json={"userinfo": "x"}).status_code == 502
    assert [call[0][2] for call in calls] == ["config"]


@pytest.mark.parametrize("route,payload", [("browse", {"path": "/a b/$(literal)"}),
                                           ("kdb", {"action": "search", "query": "q"})])
def test_actions_use_json_files(harness, route, payload):
    client, calls, *_ = harness
    assert client.post(f"/api/{route}", json=payload).status_code == 200
    assert calls[-1][1] == payload
    assert "--config" in calls[-1][0]


@pytest.mark.parametrize("response", [
    SimpleNamespace(returncode=1, stdout="", stderr="failure"),
    SimpleNamespace(returncode=0, stdout="not json", stderr=""),
    subprocess.TimeoutExpired("node", 60), FileNotFoundError("node missing")])
def test_driver_errors_are_http_errors(harness, response):
    client, _, _, replies, _ = harness
    replies["status"] = response
    result = client.get("/api/status")
    assert result.status_code == 502
    assert result.json["error"]


def test_bad_requests_do_not_execute_cli(harness):
    client, calls, *_ = harness
    assert client.post("/api/wave", json={"mode": "other"}).status_code == 400
    assert client.put("/api/config", json=[]).status_code == 400
    assert client.post("/api/tune", json={}).status_code == 400
    assert calls == []
