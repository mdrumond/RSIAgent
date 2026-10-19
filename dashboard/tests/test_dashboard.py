import base64
import json
import shutil

import pytest

from dashboard.app import create_app
from dashboard.store import DashboardStore, _source


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
    planned = {"project_id": "planned-only", "hypothesis": "Not started yet."}
    write(root / "waves/wave-001/plan.json", {"wave": {"projects": [
        {"spec": spec}, {"spec": planned}]}})
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
    phase2 = tmp_path / "phase2"
    write(phase2 / "state.json", {"completed_waves": [], "prior_outcomes": []})
    write(config, {"campaigns": [{"phase": 1, "name": "A3 Phase 1", "root": str(root),
                                   "control": str(control)},
                                  {"phase": 2, "name": "A3 Phase 2",
                                   "root": str(phase2)}]})
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
    assert len(project["transitions"]) == 2
    assert all(row["diff"].startswith("---") for row in project["transitions"])
    assert view["campaigns"][0]["worker"]["status"] == "running"
    assert [row["id"] for row in view["campaigns"][0]["planned"]] == ["planned-only"]


def test_planned_projects_do_not_inflate_attempts_or_published(campaign, tmp_path):
    config = json.loads(campaign.read_text())
    state_path = tmp_path / "phase1/state.json"
    state = json.loads(state_path.read_text())
    state["prior_outcomes"] = []
    write(state_path, state)
    view = DashboardStore(campaign, tmp_path / "cache.sqlite3").refresh()
    phase = view["campaigns"][0]
    assert phase["stats"] == {"attempted": 1, "verified": 1, "published": 0}
    assert [project["id"] for project in phase["projects"]] == ["vector-add"]
    assert phase["planned"][0]["id"] == "planned-only"

    del state["prior_outcomes"]
    write(state_path, state)
    assert DashboardStore(campaign, tmp_path / "legacy.sqlite3").refresh()[
        "campaigns"][0]["stats"]["published"] == 1


def test_refresh_failure_serves_last_cached_snapshot(campaign, tmp_path):
    store = DashboardStore(campaign, tmp_path / "cache.sqlite3")
    assert store.refresh()["stale"] is False
    root = tmp_path / "phase1"
    shutil.rmtree(root)
    stale = store.refresh()
    assert stale["stale"] is True
    assert stale["campaigns"][0]["projects"][0]["id"] == "vector-add"
    assert store.cached()["stale"] is True

    app = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0)
    client = app.test_client()
    assert b"last cached snapshot" in client.get("/").data
    assert b"last cached snapshot" in client.get("/").data


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
    assert detail.data.count(b"Revision 1") == 1 and detail.data.count(b"Revision 2") == 1
    assert b"first_draft" in detail.data and b"old_kernel" in detail.data
    assert client.get("/memory").status_code == 200
    assert client.get("/api/status").json["campaigns"][0]["worker"]["status"] == "running"
    assert client.get("/phase/1/project/../../etc/passwd").status_code == 404


@pytest.mark.parametrize("digest", [
    "g" * 64,
    "A" * 64,
    "." * 64,
    "aa/" + "x" * 61,
    "../" + "x" * 61,
])
def test_artifact_source_rejects_noncanonical_digest_before_io(monkeypatch, tmp_path, digest):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("untrusted digest reached filesystem read")

    monkeypatch.setattr(type(tmp_path), "read_text", unexpected_read)
    assert _source(tmp_path, digest) is None


def test_invalid_source_revision_is_gracefully_unavailable(campaign, tmp_path):
    branch = tmp_path / "phase1/waves/wave-001/branches/vector-add"
    delta = json.loads((branch / "memory-delta.bin").read_text())
    memory = json.loads(base64.b64decode(delta["memory_base64"]))
    memory["source_revision"] = "aa/" + "x" * 61
    delta["memory_base64"] = base64.b64encode(json.dumps(memory).encode()).decode()
    write(branch / "memory-delta.bin", delta)
    (branch / "evidence.jsonl").unlink()

    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()
    detail = client.get("/phase/1/project/vector-add")
    assert detail.status_code == 200
    assert b"Source unavailable" in detail.data


def test_default_bind_is_loopback(monkeypatch, campaign, tmp_path):
    from dashboard import __main__
    called = {}
    monkeypatch.setattr(__main__, "create_app", lambda *a, **k: type(
        "App", (), {"run": lambda self, **kw: called.update(kw)})())
    __main__.main(["serve", "--config", str(campaign), "--cache-dir", str(tmp_path)])
    assert called["host"] == "127.0.0.1" and called["port"] == 8080
