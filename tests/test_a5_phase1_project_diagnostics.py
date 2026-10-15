from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActorProjectExecutor
from benchmarks.a5kernels.phase1_project_diagnostic_driver import _capture, diagnose
from benchmarks.a5kernels.phase1_project_diagnostics import diagnostic_action
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from benchmarks.a5kernels.trial import parse_trial_action
from tests.test_a5_phase1_adaptive_actor import SourceBackend, ScriptedActor, _request
from tests.test_a5_phase1_project_bz import FakeExecutor, _backend, _request as bz_request


@pytest.mark.parametrize("text", [
    '{"request-debug":{"command":"run"}}',
    '{"request-sanitizer":{"tool":"custom"}}',
    '{"request-sanitizer":{"tool":[]}}',
    '{"request-sanitizer":{"tool":"memcheck","args":[]}}',
])
def test_diagnostic_parser_rejects_actor_commands(text):
    assert parse_trial_action(text) is None


def _arguments(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("# exact diagnostic source\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "candidate_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "abi": {"kernel_name": "kernel"},
    }))
    return ["host_driver.py", str(source), str(manifest), "diagnostic-memcheck", *(["p"] * 6)]


def test_missing_tool_is_an_observation_not_a_clean_check(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    record = diagnose(_arguments(tmp_path))
    assert record["available"] is False
    assert record["host_verified"] is False
    assert record["coverage"] == "unconfirmed"
    assert "execution" not in record


@pytest.mark.parametrize("tool", ["msdebug", "memcheck", "racecheck", "initcheck", "synccheck"])
def test_installed_tool_replays_exact_source_without_claiming_coverage(tmp_path, monkeypatch, tool):
    calls = []
    arguments = _arguments(tmp_path)
    arguments[3] = "diagnostic-" + tool
    monkeypatch.setattr("shutil.which", lambda name: "/tool/" + name)

    def capture(command, **kwargs):
        calls.append((command, kwargs))
        return {"exit_code": 0, "termination": None, "output": "tool returned zero"}

    monkeypatch.setattr("benchmarks.a5kernels.phase1_project_diagnostic_driver._capture", capture)
    record = diagnose(arguments)
    command, options = calls[-1]
    assert options["environment"]["PYTORCH_NO_NPU_MEMORY_CACHING"] == "1"
    application = (sys.executable, "-B", *arguments[:3], "run", *arguments[4:])
    assert command[-len(application):] == application
    assert record["coverage"] == "unconfirmed"
    assert record["host_verified"] is False
    assert "passed" not in record
    if tool == "msdebug":
        assert "--batch" in command and "--no-lldbinit" in command
        assert "ascend info cores" in command
    else:
        assert command[:4] == ("/tool/mssanitizer", "--tool=" + tool,
                               "--max-debuglog-size=1", "--")


def test_diagnostic_rejects_changed_source(tmp_path):
    arguments = _arguments(tmp_path)
    Path(arguments[1]).write_text("changed")
    with pytest.raises(ValueError, match="differs from manifest"):
        diagnose(arguments)


def test_tool_timeout_is_bounded_and_retains_partial_transcript():
    result = _capture((sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"),
                      environment=dict(os.environ), timeout=0.15)
    assert result["termination"] == "timeout"
    assert "started" in result["output"]


def test_tool_output_is_bounded():
    result = _capture((sys.executable, "-c", "print('x' * 100000)"), environment=dict(os.environ))
    assert result["output_truncated"]
    assert len(result["output"]) == 48000


def test_checked_backend_retains_source_bound_diagnostic(tmp_path):
    request = bz_request(tmp_path, "movement")
    executor = FakeExecutor([{
        "status": "diagnostic-observation", "diagnostic": "tool unavailable",
        "candidate_sha256": hashlib.sha256(request.candidate_source.encode()).hexdigest(),
        "available": False, "host_verified": False, "coverage": "unconfirmed",
    }])
    backend = _backend(tmp_path, executor)
    result = backend.diagnostic(request, tool="synccheck")
    assert result["artifact"]["mode"] == "diagnostic-synccheck"
    invocation = executor.invocations[0]
    assert invocation.argv[-1] == "diagnostic-synccheck"
    assert any(item.relative_path == "project_diagnostics.py" for item in invocation.files)
    assert result["artifact"]["handle"].startswith("bz-a5:")


def test_actor_can_debug_failed_source_but_must_run_verified_final_source(tmp_path):
    backend = SourceBackend()
    diagnostics = []

    def diagnostic(request, *, tool):
        diagnostics.append((request, tool))
        artifact = {"diagnostic": tool, "source": request.candidate_source}
        from benchmarks.a5kernels.evidence import canonical_digest
        return {"artifact": artifact, "evidence_sha256": canonical_digest(artifact),
                "record": {"available": True, "host_verified": False}}

    backend.diagnostic = diagnostic
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "evidence.jsonl"))
    actor = ScriptedActor((
        '{"request-debug":{}}',
        '{"write":{"slot":"kernel","content":"compile failure"}}',
        '{"request-sanitizer":{"tool":"synccheck"}}',
        '{"request-debug":{}}',
        '{"submit":{}}',
        '{"write":{"slot":"kernel","content":"fixed source"}}',
        '{"run":{}}',
        '{"submit":{}}',
    ))
    AdaptiveActorProjectExecutor(
        executor, actor, object(), request_diagnostic=diagnostic_action(backend),
    )(_request(tmp_path, executor))
    assert "write candidate source" in actor.observations[0]["error"]
    assert [tool for _, tool in diagnostics] == ["synccheck", "msdebug"]
    assert all(request.candidate_source == "compile failure" for request, _ in diagnostics)
    assert "HOST_VERIFIED" in actor.observations[4]["error"]
    admitted = actor.distill_calls[0][1]
    assert "diagnostic" in admitted.values()
    assert "host-verification" in admitted.values()


@pytest.mark.parametrize("tool", ["msdebug", "memcheck", "racecheck", "initcheck", "synccheck"])
def test_diagnostic_sessions_fit_checked_wrapper_and_retries_stay_distinct(tmp_path, tool):
    request = bz_request(tmp_path, "movement")
    record = {
        "status": "diagnostic-observation", "diagnostic": "tool unavailable",
        "candidate_sha256": hashlib.sha256(request.candidate_source.encode()).hexdigest(),
        "available": False, "host_verified": False, "coverage": "unconfirmed",
    }
    executor = FakeExecutor([record, record])
    backend = _backend(tmp_path, executor)
    results = [backend.diagnostic(request, tool=tool) for _ in range(2)]
    sessions = [invocation.argv[2] for invocation in executor.invocations]
    assert max(map(len, sessions)) <= 63
    assert sessions[0] != sessions[1]
    assert sessions[0].endswith("-000001")
    assert sessions[1].endswith("-000002")
    assert [result["artifact"]["handle"] for result in results] == [
        "bz-a5:" + session for session in sessions
    ]
    assert [result["artifact"]["attempt"] for result in results] == [1, 2]
    assert all(result["artifact"]["mode"] == "diagnostic-" + tool for result in results)
