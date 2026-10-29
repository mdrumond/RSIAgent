import base64
import json
import shutil
import subprocess
import time

import pytest

from dashboard.app import create_app
from dashboard.store import DashboardStore, _description_fields, _source


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def evidence_ledger_rows(payloads):
    previous, rows = "0" * 64, []
    for sequence, payload in enumerate(payloads):
        record = {"sequence": sequence, "kind": "artifact", "payload": payload,
                  "previous_sha256": previous}
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False).encode()
        record["entry_sha256"] = __import__("hashlib").sha256(canonical).hexdigest()
        previous = record["entry_sha256"]
        rows.append(record)
    return rows


@pytest.fixture()
def campaign(tmp_path):
    root = tmp_path / "phase1"
    branch = root / "waves/wave-001/branches/vector-add"
    spec = {"project_id": "vector-add", "hypothesis": "Measure a bounded vector add.",
            "capabilities": ["simd-vector"], "shape": [1003], "dtype": "float32"}
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
    actor_rows = evidence_ledger_rows([
        {"role": "provider-response", "sha256": response},
        {"event": "candidate", "source_sha256": old},
    ])
    (branch / "evidence.jsonl").write_text(
        "\n".join(json.dumps(row) for row in actor_rows) + "\n")
    control = tmp_path / "control"
    write(control / "worker.json", {"status": "running", "started": 10})
    config = tmp_path / "dashboard.json"
    write(config, {"campaigns": [{"phase": 1, "name": "A3 Phase 1", "root": str(root),
                                   "control": str(control)}]})
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


def test_performance_api_normalizes_attempts_statistics_bandwidth_and_pipe_metrics(
        campaign, tmp_path):
    branch = tmp_path / "phase1/waves/wave-001/branches/vector-add"
    delta = json.loads((branch / "memory-delta.bin").read_text())
    memory = json.loads(base64.b64decode(delta["memory_base64"]))
    timing = {"measurement": "launch-plus-synchronize", "instrumented": False,
              "shape": [1003], "physical_shape": [1024],
              "samples_us": [1, 10, 20, 30, 40]}
    distinct = {**timing, "samples_us": [10, 20, 30]}
    profile = {"metric": "PipeUtilization", "instrumented": True,
               "duration_semantics": "instrumented diagnostic; not canonical latency",
               "metric_values": [
                   ["report.csv:aiv_vec_ratio:0", 0.0],
                   ["report.csv:aiv_mte2_ratio:0", 0.361],
                   ["report.csv:aiv_mte2_time(us):0", 3.048],
                   ["report.csv:aic_mac_ratio:0", 0.25],
                   ["report.csv:aic_scalar_ratio:0", 25.0],
               ]}
    memory["host_facts"] = [
        {"category": "performance-study", "statement": json.dumps({"timing": [timing]})},
        {"category": "performance-study", "statement": json.dumps({"timing": [timing, distinct]})},
        {"category": "profiling", "statement": json.dumps(profile)},
        {"category": "profiling", "statement": json.dumps(profile)},
    ]
    delta["memory_base64"] = base64.b64encode(json.dumps(memory).encode()).decode()
    write(branch / "memory-delta.bin", delta)
    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()

    response = client.get("/api/phase/1/project/vector-add/performance")
    assert response.status_code == 200
    body = response.get_json()
    assert len(body["captures"]) == 2
    assert body["captures"][0]["attempt"] == 1
    assert body["captures"][0]["statistics"] == {
        "count": 5, "min_us": 1.0, "q1_us": 10.0, "median_us": 20.0,
        "q3_us": 30.0, "max_us": 40.0}
    assert body["captures"][0]["logical_shape"] == [1003]
    assert body["captures"][0]["physical_shape"] == [1024]
    # One float32 input and one output, divided by the ordinary host boundary.
    expected = (1003 * 2 * 4) / (20e-6) / (1024 ** 3)
    assert body["captures"][0]["useful_logical_io_gib_s"] == pytest.approx(expected)
    assert body["captures"][0]["boundary"] == "launch-plus-synchronize"
    assert len(body["profiles"]) == 1
    metrics = {row["id"]: row for row in body["profiles"][0]["metrics"]}
    assert metrics["aiv-vector"]["value"] == 0.0
    assert metrics["aiv-vector"]["available"] is True
    assert metrics["aiv-mte2"]["value"] == pytest.approx(36.1)
    assert metrics["aiv-mte2"]["time_us"] == pytest.approx(3.048)
    assert metrics["aic-cube"]["value"] == pytest.approx(25.0)
    assert metrics["aic-scalar"]["value"] == pytest.approx(25.0)
    assert metrics["aiv-mte3"]["value"] is None
    assert metrics["aiv-mte3"]["available"] is False
    assert body["mte_bandwidth"]["available"] is False


def test_performance_api_rejects_malformed_values_and_handles_shapes(campaign, tmp_path):
    branch = tmp_path / "phase1/waves/wave-001/branches/vector-add"
    delta = json.loads((branch / "memory-delta.bin").read_text())
    memory = json.loads(base64.b64decode(delta["memory_base64"]))
    memory["host_facts"] = [{"category": "performance-study", "statement": json.dumps({
        "timing": [
            {"measurement": "launch-plus-synchronize", "shape": [3, 7],
             "samples_us": [7]},
            {"measurement": "launch-plus-synchronize", "shape": [4],
             "samples_us": [1, "bad", 2]},
            {"measurement": "launch-plus-synchronize", "shape": [5],
             "samples_us": [float("inf")]},
            {"measurement": "launch-plus-synchronize", "shape": [6],
             "samples_us": [0, 1]},
            {"measurement": "launch-plus-synchronize", "shape": [7],
             "samples_us": [-1, 1]},
        ]})}]
    delta["memory_base64"] = base64.b64encode(json.dumps(memory).encode()).decode()
    write(branch / "memory-delta.bin", delta)
    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()
    body = client.get("/api/phase/1/project/vector-add/performance").get_json()
    assert len(body["captures"]) == 1
    assert body["captures"][0]["shape_axis"] == {"kind": "category", "value": "3×7"}
    assert body["captures"][0]["statistics"]["q1_us"] == 7.0
    assert body["captures"][0]["useful_logical_io_gib_s"] is None
    assert client.get("/api/phase/1/project/missing/performance").status_code == 404


@pytest.mark.parametrize("metric_values", [None, 7, {"bad": "shape"}, ["not-a-pair"]])
def test_performance_api_ignores_malformed_profile_facts(campaign, tmp_path, metric_values):
    branch = tmp_path / "phase1/waves/wave-001/branches/vector-add"
    delta = json.loads((branch / "memory-delta.bin").read_text())
    memory = json.loads(base64.b64decode(delta["memory_base64"]))
    memory["host_facts"] = [None, 3, ["wrong"],
        {"category": "profiling", "statement": json.dumps(
            {"metric": "PipeUtilization", "metric_values": metric_values})}]
    delta["memory_base64"] = base64.b64encode(json.dumps(memory).encode()).decode()
    write(branch / "memory-delta.bin", delta)
    response = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client().get(
        "/api/phase/1/project/vector-add/performance")
    assert response.status_code == 200
    assert response.get_json()["profiles"] == []


@pytest.mark.parametrize("dtype", [None, "unknown"])
def test_performance_bandwidth_requires_supported_explicit_dtype(campaign, tmp_path, dtype):
    plan_path = tmp_path / "phase1/waves/wave-001/plan.json"
    plan = json.loads(plan_path.read_text())
    spec = plan["wave"]["projects"][0]["spec"]
    if dtype is None:
        spec.pop("dtype")
    else:
        spec["dtype"] = dtype
    write(plan_path, plan)
    body = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client().get(
        "/api/phase/1/project/vector-add/performance").get_json()
    assert body["captures"][0]["useful_logical_io_gib_s"] is None


def test_multiple_profile_captures_keep_capture_identity(campaign, tmp_path):
    branch = tmp_path / "phase1/waves/wave-001/branches/vector-add"
    delta = json.loads((branch / "memory-delta.bin").read_text())
    memory = json.loads(base64.b64decode(delta["memory_base64"]))
    profiles = [{"metric": "PipeUtilization", "metric_values": [
        ["one.csv:aiv_vec_ratio:0", ratio]]} for ratio in (.1, .2)]
    memory["host_facts"] = [{"category": "profiling", "statement": json.dumps(row)}
                             for row in profiles]
    delta["memory_base64"] = base64.b64encode(json.dumps(memory).encode()).decode()
    write(branch / "memory-delta.bin", delta)
    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()
    body = client.get("/api/phase/1/project/vector-add/performance").get_json()
    assert [row["capture"] for row in body["profiles"]] == [1, 2]
    script = client.get("/static/app.js").data
    assert b"Capture ${metric.capture}" in script


@pytest.mark.parametrize(("spec_shape", "input_shapes", "output_shape", "node", "expected"), [
    ([33, 257], [[33, 257]], [33],
     {"name": "sum", "op": "reduce-sum", "operands": ["x"], "axis": 1},
     33 * 257 + 33),
    ([63, 65, 79], [[63, 65], [65, 79]], [63, 79],
     {"name": "product", "op": "matmul", "operands": ["x", "y"], "axis": None},
     63 * 65 + 65 * 79 + 63 * 79),
])
def test_performance_bandwidth_accepts_validated_graph_output_shape(
        campaign, tmp_path, spec_shape, input_shapes, output_shape, node, expected):
    plan_path = tmp_path / "phase1/waves/wave-001/plan.json"
    plan = json.loads(plan_path.read_text())
    spec = plan["wave"]["projects"][0]["spec"]
    spec.update({"shape": spec_shape, "dtype": "float32", "graph": {
        "inputs": [{"name": "x", "shape": input_shapes[0]}] +
                  ([{"name": "y", "shape": input_shapes[1]}] if len(input_shapes) == 2 else []),
        "nodes": [node], "output": node["name"]}})
    write(plan_path, plan)
    branch = tmp_path / "phase1/waves/wave-001/branches/vector-add"
    delta = json.loads((branch / "memory-delta.bin").read_text())
    memory = json.loads(base64.b64decode(delta["memory_base64"]))
    memory["host_facts"] = [{"category": "performance-study", "statement": json.dumps({
        "timing": [{"measurement": "launch-plus-synchronize", "shape": output_shape,
                    "samples_us": [10]}]})}]
    delta["memory_base64"] = base64.b64encode(json.dumps(memory).encode()).decode()
    write(branch / "memory-delta.bin", delta)
    body = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client().get(
        "/api/phase/1/project/vector-add/performance").get_json()
    assert body["captures"][0]["useful_logical_io_gib_s"] == pytest.approx(
        expected * 4 / 10e-6 / 1024 ** 3)


def test_project_performance_is_lazy_and_has_accessible_plot_scaffolds(campaign, tmp_path):
    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()
    detail = client.get("/phase/1/project/vector-add")
    assert b'data-performance-url="/api/phase/1/project/vector-add/performance"' in detail.data
    assert b'data-performance-chart="attempts"' in detail.data
    assert b'data-performance-chart="shapes"' in detail.data
    assert b'data-performance-chart="pipes"' in detail.data
    assert b'data-performance-chart="attempts" role="img"' not in detail.data
    assert b'data-performance-chart="pipes" role="img"' not in detail.data
    assert b'data-pipe-engine="AIC"' in detail.data and b'data-pipe-engine="AIV"' in detail.data
    assert b'class="pipe-table"' in detail.data
    assert b"Performance data loads when this section opens." in detail.data
    assert b"12.500" not in detail.data


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


def test_description_labels_are_presentation_only_and_lossless():
    text = ("Question: What changes? Controlled change: Only length. "
            "Tools: Exact comparison. Measurements: Mismatches. "
            "Expected lesson: Find the knee.")
    parsed = _description_fields(text)
    assert parsed == [
        {"label": "Question", "value": "What changes?"},
        {"label": "Controlled change", "value": "Only length."},
        {"label": "Tools", "value": "Exact comparison."},
        {"label": "Measurements", "value": "Mismatches."},
        {"label": "Expected lesson", "value": "Find the knee."},
    ]
    for malformed in ["Free form only", "Question: Q Tools: T", "Question: Q Question: again"]:
        assert _description_fields(malformed) == [{"label": "Description", "value": malformed}]


def test_project_sections_and_phase_summary_are_structured(campaign, tmp_path):
    plan_path = tmp_path / "phase1/waves/wave-001/plan.json"
    plan = json.loads(plan_path.read_text())
    hypothesis = ("Question: Does it copy? Controlled change: Length only. Tools: Native copy. "
                  "Measurements: Exact mismatch count. Expected lesson: Locate knees.")
    plan["wave"]["projects"][0]["spec"]["hypothesis"] = hypothesis
    write(plan_path, plan)
    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()
    phase = client.get("/phase/1").data
    assert b"Does it copy?" in phase and b"Controlled change" not in phase
    detail = client.get("/phase/1/project/vector-add").data
    for label in [b"Question", b"Controlled change", b"Tools", b"Measurements", b"Expected lesson"]:
        assert label in detail
    assert detail.count(b"<details open") == 2
    assert detail.count(b"<details") >= 6
    assert detail.index(b"Memory gained") < detail.index(b"Execution trace")


def test_trace_index_is_compact_and_payloads_are_lazy_authenticated(campaign, tmp_path):
    branch = tmp_path / "phase1/waves/wave-001/branches/vector-add"
    request = b'{"messages":[{"role":"user","content":"<script>bad()</script>"}]}'
    digest = __import__("hashlib").sha256(request).hexdigest()
    obj = branch / "artifacts/objects" / digest[:2] / digest[2:]
    obj.parent.mkdir(parents=True, exist_ok=True); obj.write_bytes(request)
    missing = "d" * 64
    actor_rows = evidence_ledger_rows([
        {"role": "provider-request", "sha256": digest, "stage": "action", "turn": 1},
        {"role": "provider-request", "sha256": digest, "stage": "action", "turn": 2},
        {"role": "provider-attempt-rejection", "sha256": digest,
         "stage": "action", "turn": 2},
        {"role": "host-observation", "sha256": missing, "turn": 2},
    ])
    (branch / "evidence.jsonl").write_text(
        "\n".join(json.dumps(row) for row in actor_rows) + "\n")
    host_rows = evidence_ledger_rows([
        {"project_id": "vector-add", "event": "verification"},
        {"project_id": "vector-add", "event": "verification"},
        {"project_id": "other", "event": "verification"},
    ])
    (tmp_path / "phase1/project-evidence.jsonl").write_text(
        "\n".join(json.dumps(row) for row in host_rows) + "\n")
    cache = tmp_path / "cache.sqlite3"
    client = create_app(campaign, cache, refresh_seconds=0).test_client()
    index = client.get("/api/phase/1/project/vector-add/trace").get_json()
    actor = [row for row in index["events"] if row.get("sha256") == digest and
             row["label"] == "Actor prompt"]
    assert len(actor) == 2 and [row["turn"] for row in actor] == [1, 2]
    assert "content" not in json.dumps(index)
    assert any(row["label"] == "Verifier prompt" and not row["available"] and
               "host-owned" in row["reason"] for row in index["events"])
    assert sum(row["label"] == "Host verification ledger entry"
               for row in index["events"]) == 2
    assert any(row["label"] == "Actor attempt rejection" for row in index["events"])
    assert any(row.get("sha256") == missing and not row["available"] for row in index["events"])
    payload = client.get(actor[0]["payload_url"])
    assert payload.status_code == 200 and payload.get_json()["text"] == request.decode()
    assert b"<script>" not in client.get("/phase/1/project/vector-add").data
    raw = client.get(actor[0]["download_url"])
    assert raw.data == request and raw.headers["Content-Disposition"].startswith("attachment")
    with __import__("sqlite3").connect(cache) as db:
        assert db.execute("SELECT body FROM payloads WHERE sha256=?", (digest,)).fetchone()[0] == request

    obj.write_bytes(b"tampered")
    with __import__("sqlite3").connect(cache) as db:
        db.execute("DELETE FROM payloads WHERE sha256=?", (digest,))
    assert client.get(actor[0]["payload_url"]).status_code == 409
    assert client.get("/api/phase/1/project/../../etc/passwd/trace").status_code == 404


def test_trace_payload_survives_backing_artifact_removal_after_cache(campaign, tmp_path):
    branch = tmp_path / "phase1/waves/wave-001/branches/vector-add"
    raw = b"\x00\xffbinary"
    digest = __import__("hashlib").sha256(raw).hexdigest()
    obj = branch / "artifacts/objects" / digest[:2] / digest[2:]
    obj.parent.mkdir(parents=True, exist_ok=True); obj.write_bytes(raw)
    (branch / "evidence.jsonl").write_text(json.dumps(evidence_ledger_rows([{
        "role": "provider-response", "sha256": digest,
        "stage": "action", "turn": 1}])[0]) + "\n")
    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()
    original_index = client.get("/api/phase/1/project/vector-add/trace").get_json()
    event = next(row for row in original_index["events"]
                 if row.get("sha256") == digest)
    assert client.get(event["download_url"]).data == raw
    shutil.rmtree(tmp_path / "phase1")
    campaign.unlink()
    retained = client.get("/api/phase/1/project/vector-add/trace")
    assert retained.status_code == 200
    assert [row["label"] for row in retained.get_json()["events"]] == [
        row["label"] for row in original_index["events"]
    ]
    assert client.get(event["download_url"]).data == raw
    assert client.get(event["payload_url"]).status_code == 415


def test_trace_uses_matching_retained_curriculum_session(campaign, tmp_path):
    state_path = tmp_path / "phase1/state.json"
    state = json.loads(state_path.read_text())
    plan = json.loads((tmp_path / "phase1/waves/wave-001/plan.json").read_text())
    admitted = {"wave_id": "wave-001", **plan["wave"]}
    invalid = json.loads(json.dumps(admitted))
    invalid["unexpected_top_level"] = "must be rejected"
    invalid["projects"][0]["unexpected_project_field"] = "must also be rejected"
    state["session"] = {"messages": [
        {"role": "system", "content": "curriculum system"},
        {"role": "user", "content": "choose wave-001"},
        {"role": "assistant", "content": "DECISION: WAVE\n" + json.dumps(invalid)},
        {"role": "user", "content": "correct the invalid hypothesis"},
        {"role": "assistant", "content": "DECISION: WAVE\n" + json.dumps(admitted)},
    ], "turns": 2}
    write(state_path, state)
    app = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0)
    client = app.test_client()
    events = client.get("/api/phase/1/project/vector-add/trace").get_json()["events"]
    curriculum = [row for row in events if row["category"] == "curriculum"]
    assert [row["label"] for row in curriculum] == [
        "Curriculum system prompt", "Curriculum user prompt",
        "Curriculum assistant attempt 1", "Curriculum user correction",
        "Curriculum admitted decision"]
    for label, expected in [("Curriculum system prompt", "curriculum system"),
                            ("Curriculum user prompt", "choose wave-001"),
                            ("Curriculum assistant attempt 1", "unexpected_top_level"),
                            ("Curriculum user correction", "correct the invalid"),
                            ("Curriculum admitted decision", '"wave_id": "wave-001"')]:
        event = next(row for row in events if row["label"] == label)
        assert expected in client.get(event["payload_url"]).get_json()["text"]


def test_running_wave_uses_exact_matching_pending_session(campaign, tmp_path):
    root = tmp_path / "phase1"
    plan = json.loads((root / "waves/wave-001/plan.json").read_text())
    wave = {"wave_id": "wave-001", **plan["wave"]}
    pending_session = {"messages": [
        {"role": "system", "content": "pending system"},
        {"role": "user", "content": "pending user"},
        {"role": "assistant", "content": "DECISION: WAVE\n" + json.dumps(wave)},
    ], "turns": 1}
    write(root / "pending.json", {"wave": wave, "session": pending_session})
    state = json.loads((root / "state.json").read_text())
    state["session"] = {"messages": [{"role": "assistant", "content": "stale state"}],
                        "turns": 1}
    write(root / "state.json", state)
    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()
    events = client.get("/api/phase/1/project/vector-add/trace").get_json()["events"]
    curriculum = [row for row in events if row["category"] == "curriculum"]
    assert [row["label"] for row in curriculum] == [
        "Curriculum system prompt", "Curriculum user prompt", "Curriculum admitted decision"]
    texts = [client.get(row["payload_url"]).get_json()["text"] for row in curriculum]
    assert texts[0] == "pending system" and texts[1] == "pending user"
    assert all("stale state" not in text for text in texts)


@pytest.mark.parametrize("pending", [
    "not json",
    json.dumps([]),
    json.dumps({"wave": {"wave_id": "another-wave"}, "session": {
        "messages": [{"role": "assistant", "content": "wrong pending"}]}}),
])
def test_running_wave_safely_falls_back_from_invalid_pending(campaign, tmp_path, pending):
    root = tmp_path / "phase1"
    plan = json.loads((root / "waves/wave-001/plan.json").read_text())
    wave = {"wave_id": "wave-001", **plan["wave"]}
    state = json.loads((root / "state.json").read_text())
    state["session"] = {"messages": [
        {"role": "system", "content": "state system"},
        {"role": "user", "content": "state user"},
        {"role": "assistant", "content": "DECISION: WAVE\n" + json.dumps(wave)},
    ], "turns": 1}
    write(root / "state.json", state)
    (root / "pending.json").write_text(pending)
    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()
    events = client.get("/api/phase/1/project/vector-add/trace").get_json()["events"]
    curriculum = [row for row in events if row["category"] == "curriculum"]
    texts = [client.get(row["payload_url"]).get_json()["text"] for row in curriculum]
    assert texts[:2] == ["state system", "state user"]
    assert all("wrong pending" not in text for text in texts)


@pytest.mark.parametrize("retained_in", ["pending", "state"])
def test_partial_approved_wave_matches_exact_origin_proposal(campaign, tmp_path, retained_in):
    root = tmp_path / "phase1"
    plan_path = root / "waves/wave-001/plan.json"
    plan = json.loads(plan_path.read_text())
    spec = {"project_id": "vector-add", "capabilities": ["simd-vector"],
            "shape": [1003], "dtype": "float32", "block_count": 1,
            "padding_multiple": 32, "failure_mode": "none", "timing_lengths": [],
            "profile_metric": None, "hypothesis": "Measure a bounded vector add.",
            "graph": {"inputs": [{"name": "x", "shape": [1003]},
                                  {"name": "y", "shape": [1003]}],
                      "nodes": [{"name": "sum", "op": "add",
                                 "operands": ["x", "y"], "axis": None}],
                      "output": "sum"}}
    selected = {"spec": spec, "requires": []}
    expected = {"wave_id": "wave-001", **plan["wave"], "pre_wave_coverage": [],
                "projects": [selected]}
    write(plan_path, {"wave": {key: value for key, value in expected.items()
                                if key != "wave_id"}})
    extra = {"spec": {**selected["spec"], "project_id": "valid-unselected"},
             "requires": []}
    admitted = {**expected, "projects": [selected, extra]}
    malformed = json.loads(json.dumps(admitted))
    malformed["projects"][1]["unexpected"] = "not a valid proposal project"
    unrelated = {**expected, "projects": [extra]}
    bad_spec_field = json.loads(json.dumps(admitted))
    bad_spec_field["projects"][1]["spec"]["unexpected"] = "not in ProjectSpec"
    bad_shape_type = json.loads(json.dumps(admitted))
    bad_shape_type["projects"][1]["spec"]["shape"] = [True]
    bad_dtype = json.loads(json.dumps(admitted))
    bad_dtype["projects"][1]["spec"]["dtype"] = "int8"
    session = {"messages": [
        {"role": "system", "content": "partial approval system"},
        {"role": "user", "content": "propose a full wave"},
        {"role": "assistant", "content": "DECISION: WAVE\n" + json.dumps(unrelated)},
        {"role": "user", "content": "include the selected project"},
        {"role": "assistant", "content": "DECISION: WAVE\n" + json.dumps(malformed)},
        {"role": "user", "content": "remove the malformed wrapper field"},
        {"role": "assistant", "content": "DECISION: WAVE\n" + json.dumps(bad_spec_field)},
        {"role": "user", "content": "remove the malformed spec field"},
        {"role": "assistant", "content": "DECISION: WAVE\n" + json.dumps(bad_shape_type)},
        {"role": "user", "content": "repair the invalid shape type"},
        {"role": "assistant", "content": "DECISION: WAVE\n" + json.dumps(bad_dtype)},
        {"role": "user", "content": "repair the unregistered dtype"},
        {"role": "assistant", "content": "DECISION: WAVE\n" + json.dumps(admitted)},
    ], "turns": 6}
    state = json.loads((root / "state.json").read_text())
    state["session"] = session if retained_in == "state" else {"messages": []}
    write(root / "state.json", state)
    if retained_in == "pending":
        write(root / "pending.json", {"wave": expected, "session": session})
    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()
    events = client.get("/api/phase/1/project/vector-add/trace").get_json()["events"]
    curriculum = [row for row in events if row["category"] == "curriculum"]
    assert curriculum[-1]["label"] == "Curriculum admitted decision"
    assert sum(row["label"].startswith("Curriculum assistant attempt")
               for row in curriculum) == 5
    admitted_body = client.get(curriculum[-1]["payload_url"]).get_json()["text"]
    assert admitted_body == "DECISION: WAVE\n" + json.dumps(admitted)


def test_trace_index_does_not_read_actor_payload_bodies(campaign, tmp_path, monkeypatch):
    branch = tmp_path / "phase1/waves/wave-001/branches/vector-add"
    body = b'{"nested":{"value":1}}'
    digest = __import__("hashlib").sha256(body).hexdigest()
    obj = branch / "artifacts/objects" / digest[:2] / digest[2:]
    obj.parent.mkdir(parents=True, exist_ok=True); obj.write_bytes(body)
    (branch / "evidence.jsonl").write_text(json.dumps(evidence_ledger_rows([{
        "role": "provider-response", "sha256": digest,
        "stage": "action", "turn": 1}])[0]) + "\n")
    original = type(obj).read_bytes
    def guarded_read(path):
        if "artifacts/objects" in str(path):
            raise AssertionError("trace index eagerly read actor payload")
        return original(path)
    monkeypatch.setattr(type(obj), "read_bytes", guarded_read)
    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()
    index = client.get("/api/phase/1/project/vector-add/trace")
    assert index.status_code == 200
    event = next(row for row in index.get_json()["events"] if row.get("sha256") == digest)
    assert event["available"] and event["size"] == len(body)
    script = client.get("/static/app.js").get_data(as_text=True)
    assert "JSON.stringify(JSON.parse(payload.text), null, 2)" in script


@pytest.mark.parametrize("damage", ["forged-object", "sequence", "previous", "digest"])
def test_actor_trace_rejects_corrupt_ledger_and_recovers(campaign, tmp_path, damage):
    branch = tmp_path / "phase1/waves/wave-001/branches/vector-add"
    legitimate, unrelated = b"legitimate actor response", b"unrelated retained object"
    legitimate_digest = __import__("hashlib").sha256(legitimate).hexdigest()
    unrelated_digest = __import__("hashlib").sha256(unrelated).hexdigest()
    for body, digest in ((legitimate, legitimate_digest), (unrelated, unrelated_digest)):
        source = branch / "artifacts/objects" / digest[:2] / digest[2:]
        source.parent.mkdir(parents=True, exist_ok=True); source.write_bytes(body)
    valid = evidence_ledger_rows([
        {"role": "provider-response", "sha256": legitimate_digest, "turn": 1},
        {"role": "host-observation", "sha256": legitimate_digest, "turn": 1},
    ])
    corrupt = json.loads(json.dumps(valid))
    if damage == "forged-object":
        corrupt[0]["payload"]["sha256"] = unrelated_digest
    elif damage == "sequence":
        corrupt[1]["sequence"] = 9
    elif damage == "previous":
        corrupt[1]["previous_sha256"] = "e" * 64
    else:
        corrupt[1]["entry_sha256"] = "f" * 64
    ledger = branch / "evidence.jsonl"
    ledger.write_text("\n".join(json.dumps(row) for row in corrupt) + "\n")
    store = DashboardStore(campaign, tmp_path / "cache.sqlite3")
    broken = store.trace_index(1, "vector-add")["events"]
    unavailable = [row for row in broken if row["label"] == "Actor trace unavailable"]
    assert len(unavailable) == 1 and not unavailable[0]["available"]
    assert "integrity" in unavailable[0]["reason"]
    assert not any(row["category"] == "actor" and row.get("sha256") in {
        legitimate_digest, unrelated_digest} for row in broken)

    ledger.write_text("\n".join(json.dumps(row) for row in valid) + "\n")
    repaired = store.trace_index(1, "vector-add")["events"]
    assert not any(row["label"] == "Actor trace unavailable" for row in repaired)
    actor = [row for row in repaired if row["category"] == "actor"]
    assert len(actor) == 2 and {row["sha256"] for row in actor} == {legitimate_digest}


def test_trace_event_urls_are_content_stable_across_growth_and_reordering(campaign, tmp_path):
    branch = tmp_path / "phase1/waves/wave-001/branches/vector-add"
    bodies = [b"first", b"second", b"inserted"]
    digests = [__import__("hashlib").sha256(body).hexdigest() for body in bodies]
    for body, digest in zip(bodies, digests):
        obj = branch / "artifacts/objects" / digest[:2] / digest[2:]
        obj.parent.mkdir(parents=True, exist_ok=True); obj.write_bytes(body)
    def ledger(order):
        rows = evidence_ledger_rows([
            {"role": "provider-request", "sha256": digests[index], "turn": index}
            for index in order])
        (branch / "evidence.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n")
    ledger([0, 0, 1])
    cache = tmp_path / "cache.sqlite3"
    client = create_app(campaign, cache, refresh_seconds=0).test_client()
    first = client.get("/api/phase/1/project/vector-add/trace").get_json()["events"]
    repeated = [row for row in first if row.get("sha256") == digests[0]]
    assert len({row["id"] for row in repeated}) == 2
    old_url = repeated[0]["payload_url"]

    ledger([2, 0, 0, 1])
    fresh = create_app(campaign, tmp_path / "fresh.sqlite3", refresh_seconds=0).test_client()
    grown = fresh.get("/api/phase/1/project/vector-add/trace").get_json()["events"]
    assert next(row for row in grown if row.get("sha256") == digests[0])["payload_url"] == old_url
    assert client.get(old_url).get_json()["text"] == "first"


def test_uncached_payload_disappearance_is_404_not_digest_conflict(campaign, tmp_path, monkeypatch):
    branch = tmp_path / "phase1/waves/wave-001/branches/vector-add"
    body = b"vanishing"
    digest = __import__("hashlib").sha256(body).hexdigest()
    obj = branch / "artifacts/objects" / digest[:2] / digest[2:]
    obj.parent.mkdir(parents=True, exist_ok=True); obj.write_bytes(body)
    (branch / "evidence.jsonl").write_text(json.dumps(evidence_ledger_rows([{
        "role": "provider-response", "sha256": digest, "turn": 1}])[0]) + "\n")
    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()
    event = next(row for row in client.get(
        "/api/phase/1/project/vector-add/trace").get_json()["events"]
        if row.get("sha256") == digest)
    original = type(obj).read_bytes
    def vanished(path):
        if path == obj:
            raise FileNotFoundError(path)
        return original(path)
    monkeypatch.setattr(type(obj), "read_bytes", vanished)
    assert client.get(event["payload_url"]).status_code == 404


def test_host_verification_chain_recovers_unattributed_rows_and_artifacts(campaign, tmp_path):
    root = tmp_path / "phase1"
    payloads = [{"compile": "diagnostic"}, {"classification": "host-verified"}]
    previous = "0" * 64
    rows = []
    for sequence, payload in enumerate(payloads):
        artifact = __import__("hashlib").sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        record = {"sequence": sequence, "kind": "artifact",
                  "previous_sha256": previous, "payload": {
            "artifact_kind": "compile" if sequence == 0 else "classification",
            "artifact_sha256": artifact, "payload": payload}}
        if sequence == 1:
            record["payload"]["project_id"] = "vector-add"
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        record["entry_sha256"] = __import__("hashlib").sha256(canonical).hexdigest()
        previous = record["entry_sha256"]
        rows.append(record)
    (root / "project-evidence.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()
    events = client.get("/api/phase/1/project/vector-add/trace").get_json()["events"]
    host = [row for row in events if row["category"] == "verification" and
            row["label"].startswith("Host verification")]
    assert [row["label"] for row in host] == [
        "Host verification ledger entry", "Host verification artifact: compile",
        "Host verification ledger entry", "Host verification artifact: classification"]
    texts = [client.get(row["payload_url"]).get_json()["text"] for row in host]
    assert "diagnostic" in texts[1] and "host-verified" in texts[3]


@pytest.mark.parametrize("damage", ["digest", "sequence", "previous"])
def test_host_verification_rejects_corrupt_ledger_and_recovers(campaign, tmp_path, damage):
    root = tmp_path / "phase1"
    payloads = [
        {"artifact_kind": "compile", "artifact_sha256": __import__("hashlib").sha256(
            b'{"ok":true}').hexdigest(), "payload": {"ok": True}},
        {"project_id": "vector-add", "artifact_kind": "classification",
         "artifact_sha256": __import__("hashlib").sha256(
             b'{"verified":true}').hexdigest(), "payload": {"verified": True}},
    ]
    valid = evidence_ledger_rows(payloads)
    corrupt = json.loads(json.dumps(valid))
    if damage == "digest":
        corrupt[1]["entry_sha256"] = "f" * 64
    elif damage == "sequence":
        corrupt[1]["sequence"] = 4
    else:
        corrupt[1]["previous_sha256"] = "e" * 64
    ledger = root / "project-evidence.jsonl"
    ledger.write_text("\n".join(json.dumps(row) for row in corrupt) + "\n")
    store = DashboardStore(campaign, tmp_path / "cache.sqlite3")
    broken = store.trace_index(1, "vector-add")["events"]
    unavailable = [row for row in broken if row["label"] == "Host verification unavailable"]
    assert len(unavailable) == 1 and not unavailable[0]["available"]
    assert "integrity" in unavailable[0]["reason"]
    assert not any(row["label"] == "Host verification ledger entry" for row in broken)

    ledger.write_text("\n".join(json.dumps(row) for row in valid) + "\n")
    repaired = store.trace_index(1, "vector-add")["events"]
    assert not any(row["label"] == "Host verification unavailable" for row in repaired)
    assert sum(row["label"] == "Host verification ledger entry" for row in repaired) == 2


def test_host_verification_chain_stops_at_prior_project_attribution(campaign, tmp_path):
    root = tmp_path / "phase1"
    definitions = [
        ("other-project", {"project": "other-terminal"}),
        ("vector-add", {"project": "selected-start"}),
        (None, {"project": "selected-unattributed"}),
        ("vector-add", {"project": "selected-terminal"}),
    ]
    previous, rows = "0" * 64, []
    for sequence, (project_id, artifact_payload) in enumerate(definitions):
        artifact = __import__("hashlib").sha256(json.dumps(
            artifact_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        payload = {"artifact_kind": f"kind-{sequence}", "artifact_sha256": artifact,
                   "payload": artifact_payload}
        if project_id is not None:
            payload["project_id"] = project_id
        record = {"sequence": sequence, "kind": "artifact",
                  "previous_sha256": previous, "payload": payload}
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        record["entry_sha256"] = __import__("hashlib").sha256(canonical).hexdigest()
        previous = record["entry_sha256"]
        rows.append(record)
    (root / "project-evidence.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()
    events = client.get("/api/phase/1/project/vector-add/trace").get_json()["events"]
    host = [row for row in events if row["label"] == "Host verification ledger entry"]
    assert len(host) == 3
    texts = [client.get(row["payload_url"]).get_json()["text"] for row in host]
    assert all("selected" in text for text in texts)
    assert all("other-terminal" not in text for text in texts)


def test_host_verification_collects_interleaved_project_chains(campaign, tmp_path):
    root = tmp_path / "phase1"
    definitions = [
        ("vector-add", "selected-one"), (None, "selected-unattributed"),
        ("vector-add", "selected-two"), ("sibling", "sibling-one"),
        (None, "sibling-unattributed"), ("sibling", "sibling-two"),
        ("vector-add", "selected-reset"),
    ]
    payloads = []
    for attributed, marker in definitions:
        artifact_payload = {"marker": marker}
        artifact = __import__("hashlib").sha256(json.dumps(
            artifact_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        payload = {"artifact_kind": marker, "artifact_sha256": artifact,
                   "payload": artifact_payload}
        if attributed:
            payload["project_id"] = attributed
        payloads.append(payload)
    rows = evidence_ledger_rows(payloads)
    (root / "project-evidence.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    client = create_app(campaign, tmp_path / "cache.sqlite3", refresh_seconds=0).test_client()
    events = client.get("/api/phase/1/project/vector-add/trace").get_json()["events"]
    ledger = [row for row in events if row["label"] == "Host verification ledger entry"]
    texts = [client.get(row["payload_url"]).get_json()["text"] for row in ledger]
    assert len(texts) == 4
    assert [next(marker for marker in (
        "selected-one", "selected-unattributed", "selected-two", "selected-reset")
        if marker in text) for text in texts] == [
            "selected-one", "selected-unattributed", "selected-two", "selected-reset"]
    assert all("sibling" not in text for text in texts)


def test_completed_trace_catalog_refreshes_when_partial_sync_finishes(campaign, tmp_path):
    root = tmp_path / "phase1"
    branch = root / "waves/wave-001/branches/vector-add"
    actor_body = b'{"response":"arrived later"}'
    actor_digest = __import__("hashlib").sha256(actor_body).hexdigest()
    (branch / "evidence.jsonl").write_text(json.dumps(evidence_ledger_rows([{
        "role": "provider-response", "sha256": actor_digest, "turn": 1}])[0]) + "\n")
    cache = tmp_path / "cache.sqlite3"
    store = DashboardStore(campaign, cache)
    initial = store.trace_index(1, "vector-add")["events"]
    initial_actor = next(row for row in initial if row.get("sha256") == actor_digest)
    assert not initial_actor["available"]
    assert any(row["label"] == "Host verification unavailable" for row in initial)

    obj = branch / "artifacts/objects" / actor_digest[:2] / actor_digest[2:]
    obj.parent.mkdir(parents=True, exist_ok=True); obj.write_bytes(actor_body)
    artifact_payload = {"classification": "host-verified"}
    artifact_digest = __import__("hashlib").sha256(json.dumps(
        artifact_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    record = {"sequence": 0, "kind": "artifact",
              "previous_sha256": "0" * 64, "payload": {
        "project_id": "vector-add", "artifact_kind": "classification",
        "artifact_sha256": artifact_digest, "payload": artifact_payload}}
    record["entry_sha256"] = __import__("hashlib").sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (root / "project-evidence.jsonl").write_text(json.dumps(record) + "\n")

    refreshed = store.trace_index(1, "vector-add")["events"]
    refreshed_actor = next(row for row in refreshed if row.get("sha256") == actor_digest)
    assert refreshed_actor["available"]
    assert not any(row["label"] == "Host verification unavailable" for row in refreshed)
    assert sum(row["label"].startswith("Host verification") for row in refreshed) == 2


def test_unchanged_completed_trace_reuses_catalog(campaign, tmp_path, monkeypatch):
    store = DashboardStore(campaign, tmp_path / "cache.sqlite3")
    expected = store.trace_index(1, "vector-add")
    monkeypatch.setattr(store, "_trace_sources", lambda *_: pytest.fail(
        "unchanged completed source should reuse compact catalog"))
    assert store.trace_index(1, "vector-add") == expected


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


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_controlled_cli_accepts_only_unambiguous_loopback_hosts(monkeypatch, tmp_path, host):
    from dashboard import __main__
    config, _, _ = controlled_config(tmp_path)
    called = {}
    monkeypatch.setattr(__main__, "create_app", lambda *a, **k: type(
        "App", (), {"run": lambda self, **kw: called.update(kw)})())
    __main__.main(["serve", "--config", str(config), "--cache-dir", str(tmp_path),
                   "--host", host])
    assert called["host"] == host


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.1", "example.invalid"])
def test_controlled_cli_rejects_non_loopback_bind(monkeypatch, tmp_path, host):
    from dashboard import __main__
    config, _, _ = controlled_config(tmp_path)
    monkeypatch.setattr(__main__, "create_app", lambda *a, **k:
                        pytest.fail("app must not be created for unsafe bind"))
    with pytest.raises(SystemExit):
        __main__.main(["serve", "--config", str(config), "--cache-dir", str(tmp_path),
                       "--host", host])


def test_read_only_cli_retains_explicit_host_override(monkeypatch, campaign, tmp_path):
    from dashboard import __main__
    called = {}
    monkeypatch.setattr(__main__, "create_app", lambda *a, **k: type(
        "App", (), {"run": lambda self, **kw: called.update(kw)})())
    __main__.main(["serve", "--config", str(campaign), "--cache-dir", str(tmp_path),
                   "--host", "0.0.0.0"])
    assert called["host"] == "0.0.0.0"


def controlled_config(tmp_path, *, phase=1):
    root, control = tmp_path / f"phase{phase}", tmp_path / f"control{phase}"
    config = tmp_path / "controlled.json"
    preset = {"id": f"phase{phase}", "target": "a3", "phase": phase, "root": str(root),
              "control": str(control), "config": str(tmp_path / "runtime.json"),
              "initial_memory": str(tmp_path / "memory.bin"), "objective": "Learn A3",
              "project_target": 4}
    if phase == 2:
        preset.update(phase1_root=str(tmp_path / "phase1"), review=str(tmp_path / "review.json"))
    write(config, {"campaigns": [{"id": f"phase{phase}", "target": "a3", "phase": phase,
                                   "root": str(root), "control": str(control)}],
                   "controls": {"enabled": True, "runner": "/repo/run_ascend_campaign.py",
                                "extension_step": 8,
                                "presets": [preset]}})
    return config, root, control


def test_extend_is_fixed_server_side_idempotent_and_serialized(tmp_path):
    config, root, control = controlled_config(tmp_path)
    root.mkdir()
    write(root / "state.json", {"project_target": 64, "prior_outcomes": [{}] * 64,
          "terminal": None, "wave_index": 9})
    write(control / "job.json", {"project_target": 64, "approval_mode": True})
    write(control / "worker.json", {"status": "waiting-approval",
                                      "result": {"status": "TARGET_REACHED"}})
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[2] == "continue":
            state = json.loads((root / "state.json").read_text())
            write(root / "state.json", {**state, "project_target": 72})
            write(control / "job.json", {"project_target": 72, "approval_mode": True})
            write(control / "worker.json", {"status": "waiting-approval"})
        return subprocess.CompletedProcess(argv, 0, "{}", "")
    app = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0, command_runner=run)
    client = app.test_client(); csrf = client.get("/api/csrf").get_json()["token"]
    body = {"campaign_id": "phase1", "confirmation": "extend:phase1",
            "project_target": 999, "command": "evil"}
    headers = {"X-CSRF-Token": csrf, "Idempotency-Key": "extend-once"}
    first = client.post("/api/control/extend", json=body, headers=headers)
    replay = client.post("/api/control/extend", json=body, headers=headers)
    assert first.status_code == replay.status_code == 202
    assert first.get_json()["operation_id"] == replay.get_json()["operation_id"]
    assert wait_operation(client, first.get_json()["operation_id"])["status"] == "succeeded"
    assert calls == [[pytest.importorskip("sys").executable, "/repo/run_ascend_campaign.py",
                      "continue", "--control", str(control), "--project-target", "72"]]


def test_distinct_extend_requests_cannot_queue_concurrently(tmp_path):
    import threading
    config, root, control = controlled_config(tmp_path)
    root.mkdir()
    write(root / "state.json", {"project_target": 64, "prior_outcomes": [{}] * 64,
          "terminal": None, "wave_index": 9})
    write(control / "job.json", {"project_target": 64, "approval_mode": True})
    write(control / "worker.json", {"status": "waiting-approval",
                                      "result": {"status": "TARGET_REACHED"}})
    entered, release = threading.Event(), threading.Event()
    def run(argv, **kwargs):
        entered.set(); assert release.wait(2)
        state = json.loads((root / "state.json").read_text())
        write(root / "state.json", {**state, "project_target": 72})
        write(control / "job.json", {"project_target": 72, "approval_mode": True})
        write(control / "worker.json", {"status": "waiting-approval"})
        return subprocess.CompletedProcess(argv, 0, "{}", "")
    app = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0, command_runner=run)
    client = app.test_client(); csrf = client.get("/api/csrf").get_json()["token"]
    body = {"campaign_id": "phase1", "confirmation": "extend:phase1"}
    first = client.post("/api/control/extend", json=body, headers={
        "X-CSRF-Token": csrf, "Idempotency-Key": "first"})
    assert first.status_code == 202 and entered.wait(2)
    second = client.post("/api/control/extend", json=body, headers={
        "X-CSRF-Token": csrf, "Idempotency-Key": "second"})
    assert second.status_code == 409
    release.set()
    assert wait_operation(client, first.get_json()["operation_id"])["status"] == "succeeded"


def test_extend_pins_validated_target_before_background_execution(tmp_path):
    import threading
    config, root, control = controlled_config(tmp_path)
    settings = json.loads(config.read_text())
    settings["controls"]["presets"].append({**settings["controls"]["presets"][0],
        "id": "blocker", "root": str(tmp_path / "unused"),
        "control": str(tmp_path / "unused-control")})
    write(config, settings)
    root.mkdir()
    write(root / "state.json", {"project_target": 64, "prior_outcomes": [{}] * 64,
          "terminal": None, "wave_index": 9})
    write(control / "job.json", {"project_target": 64, "approval_mode": True})
    write(control / "worker.json", {"status": "waiting-approval",
                                      "result": {"status": "TARGET_REACHED"}})
    entered, release, calls = threading.Event(), threading.Event(), []
    def run(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            entered.set(); assert release.wait(2)
        else:
            write(control / "job.json", {"project_target": 72, "approval_mode": True})
            write(control / "worker.json", {"status": "waiting-approval"})
        return subprocess.CompletedProcess(argv, 0, "{}", "")
    app = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0, command_runner=run)
    client = app.test_client(); csrf = client.get("/api/csrf").get_json()["token"]
    blocker = client.post("/api/control/start", json={"preset_id": "blocker",
        "confirmation": "start:blocker"}, headers={"X-CSRF-Token": csrf,
        "Idempotency-Key": "blocker"})
    assert blocker.status_code == 202 and entered.wait(2)
    extension = client.post("/api/control/extend", json={"campaign_id": "phase1",
        "confirmation": "extend:phase1", "project_target": 999}, headers={
        "X-CSRF-Token": csrf, "Idempotency-Key": "pinned"})
    assert extension.status_code == 202
    state = json.loads((root / "state.json").read_text())
    write(root / "state.json", {**state, "project_target": 72,
                                  "prior_outcomes": [{}] * 72})
    release.set()
    assert wait_operation(client, extension.get_json()["operation_id"])["status"] == "succeeded"
    assert calls[-1][-2:] == ["--project-target", "72"]


@pytest.mark.parametrize("worker_failure", [False, True])
def test_extend_waits_for_detached_worker_and_propagates_failure(tmp_path, worker_failure):
    import threading
    config, root, control = controlled_config(tmp_path)
    root.mkdir()
    write(root / "state.json", {"project_target": 64, "prior_outcomes": [{}] * 64,
          "terminal": None, "wave_index": 9})
    write(control / "job.json", {"project_target": 64, "approval_mode": True})
    write(control / "worker.json", {"status": "waiting-approval",
                                      "result": {"status": "TARGET_REACHED"}})
    release = threading.Event()
    def mutate():
        assert release.wait(2)
        if worker_failure:
            write(control / "worker.json", {"status": "failed", "error": "continuation broke"})
        else:
            state = json.loads((root / "state.json").read_text())
            write(root / "state.json", {**state, "project_target": 72})
            write(control / "job.json", {"project_target": 72, "approval_mode": True})
            write(control / "worker.json", {"status": "waiting-approval"})
    def run(argv, **kwargs):
        threading.Thread(target=mutate).start()
        return subprocess.CompletedProcess(argv, 0, '{"session":"detached"}', "")
    app = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0, command_runner=run)
    app.extensions["dashboard_controls"].EXTENSION_POLL_SECONDS = .001
    client = app.test_client(); csrf = client.get("/api/csrf").get_json()["token"]
    response = client.post("/api/control/extend", json={"campaign_id": "phase1",
        "confirmation": "extend:phase1"}, headers={"X-CSRF-Token": csrf,
        "Idempotency-Key": "async"})
    operation_id = response.get_json()["operation_id"]
    time.sleep(.02)
    assert client.get(f"/api/operations/{operation_id}").get_json()["status"] == "running"
    release.set()
    operation = wait_operation(client, operation_id)
    assert operation["status"] == ("failed" if worker_failure else "succeeded")
    if worker_failure:
        assert "continuation broke" in operation["error"]


def test_extension_timeout_persists_intent_and_retry_does_not_increment_again(tmp_path):
    config, root, control = controlled_config(tmp_path)
    root.mkdir()
    write(root / "state.json", {"project_target": 64, "prior_outcomes": [{}] * 64,
          "terminal": None, "wave_index": 9})
    write(control / "job.json", {"project_target": 64, "approval_mode": True,
                                  "adopted": True})
    write(control / "adoption.json", {"project_target": 64})
    write(control / "worker.json", {"status": "waiting-approval",
                                      "result": {"status": "TARGET_REACHED"}})
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, '{"session":"detached"}', "")
    app = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0, command_runner=run)
    manager = app.extensions["dashboard_controls"]
    manager.EXTENSION_WAIT_SECONDS = .01
    manager.EXTENSION_POLL_SECONDS = .001
    client = app.test_client(); csrf = client.get("/api/csrf").get_json()["token"]
    body = {"campaign_id": "phase1", "confirmation": "extend:phase1"}
    first = client.post("/api/control/extend", json=body, headers={
        "X-CSRF-Token": csrf, "Idempotency-Key": "timeout"})
    assert wait_operation(client, first.get_json()["operation_id"])["status"] == "failed"
    intent = json.loads((control / "extension-intent.json").read_text())
    assert intent["project_target"] == 72
    retry = client.post("/api/control/extend", json=body, headers={
        "X-CSRF-Token": csrf, "Idempotency-Key": "retry"})
    assert retry.status_code == 202
    assert wait_operation(client, retry.get_json()["operation_id"])["status"] == "failed"
    assert [call[-1] for call in calls] == ["72", "72"]


@pytest.mark.parametrize("change", ["pending", "terminal", "short", "proposal", "worker"])
def test_extend_requires_clean_target_reached_boundary(tmp_path, change):
    config, root, control = controlled_config(tmp_path)
    root.mkdir()
    state = {"project_target": 64, "prior_outcomes": [{}] * 64,
             "terminal": None, "wave_index": 9}
    write(root / "state.json", state)
    write(control / "job.json", {"project_target": 64, "approval_mode": True})
    write(control / "worker.json", {"status": "waiting-approval",
                                      "result": {"status": "TARGET_REACHED"}})
    if change == "pending": write(root / "pending.json", {})
    elif change == "terminal": state["terminal"] = {"status": "STALLED"}; write(root / "state.json", state)
    elif change == "short": state["prior_outcomes"].pop(); write(root / "state.json", state)
    elif change == "proposal": write(control / "curriculum-approval.json", {"status": "proposed"})
    elif change == "worker": write(control / "worker.json", {"status": "running"})
    client = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0,
                        command_runner=lambda *a, **k: pytest.fail("must not run")).test_client()
    csrf = client.get("/api/csrf").get_json()["token"]
    response = client.post("/api/control/extend", json={"campaign_id": "phase1",
        "confirmation": "extend:phase1"}, headers={"X-CSRF-Token": csrf,
        "Idempotency-Key": change})
    assert response.status_code == 409


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


def test_operation_details_are_hidden_when_controls_are_disabled(campaign, tmp_path):
    config, _, _ = controlled_config(tmp_path)
    cache = tmp_path / "operations.sqlite3"
    enabled = create_app(config, cache, command_runner=lambda *a, **k: None)
    operation_id = enabled.extensions["dashboard_controls"]._insert_for_test(
        "sample", "phase1", "failed")
    assert enabled.test_client().get(f"/api/operations/{operation_id}").status_code == 200
    disabled = create_app(campaign, cache)
    assert disabled.test_client().get(f"/api/operations/{operation_id}").status_code == 403


def test_a5_campaign_and_preset_are_rejected_before_invocation(tmp_path):
    config, _, _ = controlled_config(tmp_path)
    settings = json.loads(config.read_text())
    settings["campaigns"][0]["target"] = "a5"
    settings["controls"]["presets"][0]["target"] = "a5"
    write(config, settings)
    calls = []
    client = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0,
                        command_runner=lambda *a, **k: calls.append(a)).test_client()
    csrf = client.get("/api/csrf").get_json()["token"]
    headers = {"X-CSRF-Token": csrf, "Idempotency-Key": "a5"}
    assert client.post("/api/control/sample", json={"campaign_id": "phase1",
        "confirmation": "sample:phase1"}, headers=headers).status_code == 409
    assert client.post("/api/control/start", json={"preset_id": "phase1",
        "confirmation": "start:phase1"}, headers={**headers,
        "Idempotency-Key": "a5-start"}).status_code == 409
    assert calls == []


def test_start_is_async_allowlisted_and_idempotent(tmp_path):
    config, _, _ = controlled_config(tmp_path, phase=1)
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, '{"session":"test"}\n', "")
    app = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0, command_runner=run)
    client = app.test_client()
    assert b"Start phase 1" in client.get("/").data
    csrf = client.get("/api/csrf").get_json()["token"]
    headers = {"X-CSRF-Token": csrf, "Idempotency-Key": "start-once"}
    body = {"preset_id": "phase1", "confirmation": "start:phase1"}
    first = client.post("/api/control/start", json=body, headers=headers)
    second = client.post("/api/control/start", json=body, headers=headers)
    assert first.status_code == second.status_code == 202
    assert first.get_json()["operation_id"] == second.get_json()["operation_id"]
    assert wait_operation(client, first.get_json()["operation_id"])["status"] == "succeeded"
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[:3] == [pytest.importorskip("sys").executable, "/repo/run_ascend_campaign.py", "start"]
    assert "--target" in argv and argv[argv.index("--target") + 1] == "a3"
    assert "--approval-mode" in argv and "--phase1-root" not in argv and "--review" not in argv
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
                            {"spec": {"project_id": "copy", "hypothesis":
                                      "Question: Copy? Controlled change: Length. Tools: Native. "
                                      "Measurements: Mismatches. Expected lesson: Copy safely.",
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
    assert proposals[0]["projects"][0]["description_fields"][0] == {
        "label": "Question", "value": "Copy?"}
    home = client.get("/").data
    assert b"Controlled change" in home and b"Expected lesson" in home
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


@pytest.mark.parametrize("already_published", [False, True])
def test_running_approval_retry_and_reconciliation_are_idempotent(tmp_path, already_published):
    config, root, control = controlled_config(tmp_path)
    root.mkdir()
    state = {"coverage": [], "prior_outcomes": [], "wave_index": 1}
    write(root / "state.json", state)
    proposal = {"base_wave_index": 1, "base_memory_sha256": "a" * 64,
                "wave": {"projects": [{"spec": {"project_id": "one",
                    "hypothesis": "One", "capabilities": []}}]}}
    write(control / "curriculum-approval.json", {"status": "approved", "phase": 1,
          "proposal_id": "retry", "proposal": proposal, "selected_project_ids": ["one"]})
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        record = json.loads((control / "curriculum-approval.json").read_text())
        if len(calls) == 1:
            write(control / "curriculum-approval.json", {**record, "status": "running"})
            raise OSError("worker connection lost")
        write(control / "curriculum-approval.json", {**record, "status": "complete",
              "recovered": already_published})
        return subprocess.CompletedProcess(argv, 0, "{}", "")
    client = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0,
                        command_runner=run).test_client()
    csrf = client.get("/api/csrf").get_json()["token"]
    body = {"campaign_id": "phase1", "proposal_id": "retry",
            "confirmation": "run-approved:retry"}
    headers = {"X-CSRF-Token": csrf, "Idempotency-Key": "first"}
    first = client.post("/api/control/run-approved", json=body, headers=headers)
    first_id = first.get_json()["operation_id"]
    assert wait_operation(client, first_id)["status"] == "failed"
    restarted = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0,
                           command_runner=run)
    client = restarted.test_client()
    csrf = client.get("/api/csrf").get_json()["token"]
    headers = {"X-CSRF-Token": csrf, "Idempotency-Key": "first"}
    replay = client.post("/api/control/run-approved", json=body, headers=headers)
    assert replay.get_json()["operation_id"] == first_id and len(calls) == 1
    if already_published:
        write(root / "state.json", {**state, "wave_index": 2,
                                     "prior_outcomes": [{"project_id": "one"}]})
    page = client.get("/")
    assert b"Resume/reconcile" in page.data
    retry = client.post("/api/control/run-approved", json=body, headers={
        "X-CSRF-Token": csrf, "Idempotency-Key": "explicit-retry"})
    assert retry.status_code == 202
    assert wait_operation(client, retry.get_json()["operation_id"])["status"] == "succeeded"
    assert len(calls) == 2


def test_stale_selection_and_interrupted_operations_recover_after_restart(tmp_path):
    config, root, control = controlled_config(tmp_path, phase=1)
    root.mkdir()
    write(root / "state.json", {"coverage": [], "prior_outcomes": [], "wave_index": 2})
    write(control / "curriculum-approval.json", {"status": "proposed", "phase": 1,
          "proposal_id": "p2", "proposal": {"base_wave_index": 1,
          "base_memory_sha256": "a" * 64, "wave": {"projects": [
              {"spec": {"project_id": "one", "hypothesis": "One", "capabilities": []}},
              {"spec": {"project_id": "two", "hypothesis": "Two", "capabilities": []}}]}}})
    cache = tmp_path / "cache.sqlite3"
    app = create_app(config, cache, refresh_seconds=0, command_runner=lambda *a, **k: None)
    client = app.test_client()
    csrf = client.get("/api/csrf").get_json()["token"]
    headers = {"X-CSRF-Token": csrf, "Idempotency-Key": "approval"}
    body = {"campaign_id": "phase1", "proposal_id": "p2", "project_ids": ["one", "two"],
            "confirmation": "approve:p2"}
    assert client.post("/api/control/approve", json=body, headers=headers).status_code == 409
    body["project_ids"] = ["one"]
    assert client.post("/api/control/approve", json=body,
        headers={**headers, "Idempotency-Key": "stale"}).status_code == 409

    manager = app.extensions["dashboard_controls"]
    running = manager._insert_for_test("sample", "phase1", "running")
    queued = manager._insert_for_test("sample", "phase1", "queued")
    manager.close()
    restarted = create_app(config, cache, refresh_seconds=0, command_runner=lambda *a, **k: None)
    client = restarted.test_client()
    for operation in (running, queued):
        interrupted = client.get(f"/api/operations/{operation}").get_json()
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


def test_terminal_proposal_summary_and_zero_project_approval(tmp_path):
    config, root, control = controlled_config(tmp_path)
    root.mkdir()
    write(root / "state.json", {"coverage": [], "prior_outcomes": [], "wave_index": 3})
    write(control / "curriculum-approval.json", {"status": "proposed", "phase": 1,
          "proposal_id": "terminal", "proposal": {"base_wave_index": 3,
          "base_memory_sha256": "a" * 64, "session": {"messages": [
              {"content": "private terminal reasoning"}]},
          "terminal": {"status": "SATURATED", "reason": "Coverage is complete."}}})
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "{}", "")
    client = create_app(config, tmp_path / "cache.sqlite3", refresh_seconds=0,
                        command_runner=run).test_client()
    status = client.get("/api/status").get_json()
    proposal = status["controls"]["proposals"][0]
    assert proposal["terminal"]["status"] == "SATURATED"
    assert proposal["terminal"]["reason"] == "Coverage is complete."
    assert "private terminal reasoning" not in json.dumps(status)
    csrf = client.get("/api/csrf").get_json()["token"]
    response = client.post("/api/control/approve", json={"campaign_id": "phase1",
        "proposal_id": "terminal", "project_ids": [],
        "confirmation": "approve:terminal"}, headers={"X-CSRF-Token": csrf,
        "Idempotency-Key": "terminal"})
    assert response.status_code == 202
    assert wait_operation(client, response.get_json()["operation_id"])["status"] == "succeeded"
    assert calls == [[pytest.importorskip("sys").executable,
                      "/repo/run_ascend_campaign.py", "approve", "--control", str(control),
                      "--proposal-id", "terminal"]]


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




def test_explanations_are_visible_and_trace_bytes_remain_exact(campaign, tmp_path):
    import hashlib
    branch = tmp_path / 'phase1/waves/wave-001/branches/vector-add'
    raw = b'{"classification":"host-verified","missing":["observed-host-mismatch"]}'
    digest = hashlib.sha256(raw).hexdigest()
    obj = branch / 'artifacts/objects' / digest[:2] / digest[2:]
    obj.parent.mkdir(parents=True, exist_ok=True); obj.write_bytes(raw)
    (branch / 'evidence.jsonl').write_text('\n'.join(json.dumps(row) for row in evidence_ledger_rows([
        {'role': 'host-observation', 'sha256': digest, 'turn': 1}])) + '\n')
    app = create_app(campaign, tmp_path / 'explained-cache', refresh_seconds=0)
    try:
        client = app.test_client()
        detail = client.get('/phase/1/project/vector-add').text
        index = client.get('/api/phase/1/project/vector-add/trace').json
        event = next(row for row in index['events'] if row.get('sha256') == digest)
        response = client.get(event['payload_url']).json
        assert response['text'].encode() == raw
        assert response['sha256'] == digest and response['size'] == len(raw)
        explanations = response['presentation']['status_explanations']
        assert any(row['code'] == 'observed-host-mismatch' for row in explanations)
        assert client.get(event['download_url']).data == raw
        campaign_view = client.get('/api/status').json['campaigns'][0]
        assert campaign_view['status_explanations'] and campaign_view['worker']['status_explanations']
    finally:
        app.extensions['dashboard_controls'].close()


@pytest.mark.parametrize("section", ["campaigns", "presets"])
def test_unsupported_campaign_configuration_rejected_before_cache_writes(tmp_path, section):
    config = tmp_path / "dashboard.json"
    row = dict(id="old", phase=2, target="a3", root=str(tmp_path / "old-run"))
    write(config, {"campaigns": [row]} if section == "campaigns" else
          {"campaigns": [], "controls": {"presets": [row]}})
    cache = tmp_path / "cache" / "dashboard.sqlite"
    with pytest.raises(ValueError):
        create_app(config, cache)
    assert not cache.parent.exists()
    assert not (tmp_path / "old-run").exists()
