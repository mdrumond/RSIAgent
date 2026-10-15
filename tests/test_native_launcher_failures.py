"""Exercise real BZ adapters with failed terminal/retained wrapper responses."""

import json
import subprocess

import pytest

from test_a3_bz_remote_candidate import _backend, _terminal, _completed, _upload_terminal
from test_ascend_target_production import spec
from benchmarks.ascend.production import NativeProjectBackend, NativeProjectBinding
from benchmarks.ascend.phase1_cli import NativeProjectMeasurements
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError
from benchmarks.a5kernels.evidence import EvidenceLedger


def request_and_backend(tmp_path, process):
    transport = _backend(tmp_path, process)
    backend = NativeProjectBackend(transport, tmp_path / "evidence", target="discovered-a3", arch="discovered-arch", profile="bz-a3-1")
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "ledger"), binding=NativeProjectBinding())
    prepared = executor.prepare(spec())
    return executor.request(prepared, prepared.abi.signature + " {}"), backend


@pytest.mark.parametrize("outcome", ["prepare", "ambiguous", "timeout", "provenance", "host"])
def test_compile_infrastructure_never_becomes_agent_source_failure(tmp_path, outcome):
    def process(argv, **kwargs):
        if argv[0].endswith("cpl-remote"):
            return _completed(argv, _upload_terminal(), code=0) if outcome != "prepare" else _completed(argv, stderr="upload unavailable", code=1)
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(argv, 1)
        if outcome == "provenance":
            return _terminal("bz-a3-1", "bz-a3-1:compile", "A3CANDIDATE_COMPILED=not-a-digest")
        body = "BZ_A3_JOB_STATE=transport-failed" if outcome == "ambiguous" else "A3REMOTE_STAGE=compile"
        return _terminal("bz-a3-1", "bz-a3-1:compile", body, state="failed", code=1, stderr="host unavailable")
    request, backend = request_and_backend(tmp_path, process)
    with pytest.raises(ProjectInfrastructureError):
        backend.compile(request)
    measurements = NativeProjectMeasurements(backend, {}, tmp_path)
    with pytest.raises(ProjectInfrastructureError):
        measurements._measurement(request)


def test_actual_native_compiler_failure_remains_repair_feedback(tmp_path):
    def process(argv, **kwargs):
        if argv[0].endswith("cpl-remote"):
            return _completed(argv, _upload_terminal())
        return _terminal("bz-a3-1", "bz-a3-1:compile", "A3CANDIDATE_COMPILE_FAILED=1", state="failed", code=1, stderr="candidate.cpp:1: error: undefined device API")
    request, backend = request_and_backend(tmp_path, process)
    observation = backend.compile(request)
    assert observation.passed is False
    assert "undefined device API" in observation.diagnostic
    with pytest.raises(ValueError, match="compilation failed"):
        NativeProjectMeasurements(backend, {}, tmp_path)._measurement(request)


def test_pending_compile_preserves_original_handle_for_observation(tmp_path):
    calls = []
    def process(argv, **kwargs):
        calls.append(argv)
        if argv[0].endswith("cpl-remote"):
            return _completed(argv, _upload_terminal())
        return _terminal("bz-a3-1", "bz-a3-1:keep-this", state="observation-unavailable", code=1)
    request, backend = request_and_backend(tmp_path, process)
    for _ in range(2):
        with pytest.raises(ProjectInfrastructureError, match="keep-this"):
            backend.compile(request)
    assert len([a for a in calls if a[0].endswith("cpl-remote")]) == 1
    assert "observe" in calls[-1] and calls[-1][-1] == "bz-a3-1:keep-this"


def test_pending_measurement_preserves_handle_instead_of_agent_repair(tmp_path):
    from test_a3_native_measurements import setup_backend
    calls = []
    def process(argv, **kwargs):
        calls.append(argv)
        return _terminal("bz-a3-1", "bz-a3-1:timing-keep", state="observation-unavailable", code=1)
    backend = setup_backend(tmp_path, runner=process)
    for _ in range(2):
        with pytest.raises(ProjectInfrastructureError, match="timing-keep"):
            backend.timing(launch_count=1)
    assert "observe" in calls[-1] and calls[-1][-1] == "bz-a3-1:timing-keep"


def test_measurement_wrapper_timeout_is_infrastructure(tmp_path):
    from test_a3_native_measurements import setup_backend
    def process(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 1)
    with pytest.raises(ProjectInfrastructureError, match="timed out"):
        setup_backend(tmp_path, runner=process).timing(launch_count=1)


@pytest.mark.parametrize("compiler_invoked", [False, True])
def test_compile_failure_marker_requires_actual_compiler_exit(tmp_path, monkeypatch, capsys, compiler_invoked):
    import sys
    from types import SimpleNamespace
    from test_a3_native_measurements import load_driver
    def compile_only(*args):
        if compiler_invoked:
            raise subprocess.CalledProcessError(1, ["bisheng", "candidate.cpp"])
        raise RuntimeError("bisheng compiler not found on PATH")
    monkeypatch.setitem(sys.modules, "legacy_driver", SimpleNamespace(_compile=compile_only))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    driver = load_driver("native_driver", monkeypatch)
    driver.__file__ = str(tmp_path / "host_driver.py")
    (tmp_path / "build.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["host_driver.py", "--compile-only"])
    with pytest.raises(SystemExit if compiler_invoked else RuntimeError):
        driver.main()
    assert ("A3CANDIDATE_COMPILE_FAILED=1" in capsys.readouterr().out) is compiler_invoked


@pytest.mark.parametrize("extra", ["TORCH_LIBRARY(fake, m) {}", "auto result = torch::add(x, x);"])
def test_obvious_host_framework_substitution_is_rejected(extra):
    from benchmarks.a3kernels.native_candidate import NativeCandidatePlan, NativeTensor
    with pytest.raises(ValueError, match="device code"):
        NativeCandidatePlan('extern "C" __global__ __aicore__ void copy(GM_ADDR x, GM_ADDR out) {}\n' + extra,
            "copy", (NativeTensor("x", (1,), "float32", (1.0,)),), NativeTensor("out", (1,), "float32"),
            "r", "a", "p", "bz-a3-1", "actual-a3", "native-arch")


def test_unsupported_native_profile_metric_is_repaired_before_execution():
    from test_ascend_target_curriculum import spec_mapping
    from benchmarks.ascend.targets import a3_curriculum_policy
    from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent, CurriculumContext, _grammar
    valid = spec_mapping()
    valid.update(capabilities=["vector-reduction", "profiling"], profile_metric="pipe-utilization")
    responses = iter((dict(valid, profile_metric="timeline"), valid))
    messages = []
    def complete(history):
        messages.append(history)
        return "DECISION: WAVE\n" + json.dumps(dict(wave_id="wave-001", pre_wave_memory_sha256="a" * 64,
            pre_wave_coverage=[], projects=[dict(spec=next(responses), requires=[])]))
    decision, session = CurriculumAgent(complete, target_policy=a3_curriculum_policy()).select(
        CurriculumContext("Learn native profiling", 1, "a" * 64, (), ()))
    assert session.turns == 2
    assert decision.wave.projects[0].spec.profile_metric.value == "pipe-utilization"
    assert "profiling metric unavailable" in messages[1][-1]["content"]
    assert a3_curriculum_policy().grammar(_grammar())["profile_metrics"] == ["pipe-utilization"]
    assert "timeline" in _grammar()["profile_metrics"]
