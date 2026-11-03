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
                "curriculum": ["add"], "actor": {"seconds": 240},
                "curriculum_actor": {"model": "curriculum"},
                "skill_writer_actor": {"model": "writer"}}
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


def test_executed_frontend_forwards_canonical_config_and_browse_contract(harness):
    """Run the shipped JS and replay its fetches through Flask into the CLI driver."""
    client, calls, config, replies, _ = harness
    original = {**config, "actor": {"model": "kernel"},
                "curriculum_actor": {"model": "curriculum"},
                "skill_writer_actor": {"model": "writer"},
                "skills": {"activeVersion": "baseline"},
                "skillsmap": {"design": "baseline/design.md"},
                "baselineKDB": "/baseline/old.sqlite"}
    edited = {"actor": {"model": "edited-kernel", "thinkingLevel": "off"},
              "curriculum": {"model": "edited-curriculum"},
              "writer": {"model": "edited-writer"}}
    skillsmap = {"design": "/private/skills/new design.md"}
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const elements = new Map();
function element() {
  return {value: '', textContent: '', children: [],
    addEventListener() {}, append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; }};
}
const document = {
  getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
  createElement: element, querySelectorAll: () => []
};
const requests = [];
const fetch = async (url, options) => {
  const body = options.body === undefined ? undefined : JSON.parse(options.body);
  requests.push({url, method: options.method, body});
  let value = {};
  if (url === '/api/config') value = body ?? input.config;
  if (url === '/api/status') value = {projects: [], attempts: []};
  if (url === '/api/browse') value = {path: body.path, entries: [
    {name: 'child', path: '/workspace/child', isDirectory: true}]};
  return {ok: true, json: async () => value};
};
(async () => {
  vm.runInNewContext(input.source, {document, fetch});
  const tick = () => new Promise(resolve => setImmediate(resolve));
  await tick();
  const initialActors = JSON.parse(elements.get('actors').value);
  elements.get('actors').value = JSON.stringify(input.actors);
  elements.get('skillsmap').value = JSON.stringify(input.skillsmap);
  elements.get('baseline').value = '/baseline/new database.sqlite';
  elements.get('settings').onsubmit({preventDefault() {}});
  await tick();
  document.getElementById('path').value = '/workspace';
  elements.get('browse').onsubmit({preventDefault() {}});
  await tick();
  await elements.get('folders').children[0].onclick();
  process.stdout.write(JSON.stringify({initialActors, requests,
    error: elements.get('message').textContent}));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    # Popen is separate from the monkeypatched subprocess.run CLI boundary.
    with subprocess.Popen(["node", "-e", script], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            stdout, stderr = process.communicate(json.dumps({
                "source": (Path(__file__).resolve().parents[2] / "web/app.js").read_text(),
                "config": original, "actors": edited, "skillsmap": skillsmap}), timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        assert process.returncode == 0, stderr
    result = json.loads(stdout)
    assert result["initialActors"] == {"actor": original["actor"],
        "curriculum": original["curriculum_actor"], "writer": original["skill_writer_actor"]}
    assert result["error"] == "Done"
    for req in result["requests"]:
        if req["method"] == "GET":
            continue
        if req["url"] == "/api/browse":
            replies["browse"] = SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
                "path": req["body"]["path"], "entries": [
                    {"name": "child", "path": "/workspace/child", "isDirectory": True}]}))
        response = client.open(req["url"], method=req["method"], json=req["body"])
        assert response.status_code == 200
    saved = next(payload for argv, payload in calls if argv[2] == "config")
    assert saved["actor"] == edited["actor"]
    assert saved["curriculum_actor"] == edited["curriculum"]
    assert saved["skill_writer_actor"] == edited["writer"]
    assert "actors" not in saved
    assert saved["baselineKDB"] == "/baseline/new database.sqlite"
    assert saved["skillsmap"] == skillsmap
    assert saved["skills"] == original["skills"]
    assert saved["unknown"] == original["unknown"]
    assert [payload["path"] for argv, payload in calls if argv[2] == "browse"] == [
        "/workspace", "/workspace/child"]
