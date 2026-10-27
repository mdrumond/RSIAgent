"""Direct native payloads keep the host oracle and avoid transport wrappers."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.ascend.local_native import LocalNativeCandidateBackend
from benchmarks.ascend.phase1_cli import load_native_config, native_transport
from benchmarks.a3kernels.native_candidate import NativeCandidatePlan, NativeTensor
from benchmarks.a3kernels.phase1_protocol import FailedEvidence


def config(tmp_path):
    return dict(execution="local", workspace=str(tmp_path / "workspace"),
                physical_device=0, arch="discovered-arch")


def plan():
    return NativeCandidatePlan(
        'extern "C" __global__ __aicore__ void compute(GM_ADDR x, GM_ADDR y) {}',
        "compute", (NativeTensor("x", (2,), "float32", (1., 2.)),),
        NativeTensor("y", (2,), "float32"), "request", "attempt", "project",
        "local", "discovered-a3", "discovered-arch")


def test_local_native_stages_exact_driver_and_returns_ungraded_receipt(tmp_path):
    calls = []
    class Runner:
        def run(self, argv, operation_directory, **options):
            calls.append((argv, options))
            root = Path(options["cwd"])
            assert argv[0] == sys.executable
            assert Path(argv[1]) == root / "host_driver.py"
            if "--compile-only" in argv:
                (root / "a3_candidate.so").write_bytes(b"real-compiled-library-fixture")
                digest = hashlib.sha256((root / "a3_candidate.so").read_bytes()).hexdigest()
                return subprocess.CompletedProcess(argv, 0, "A3CANDIDATE_COMPILED=" + digest, "")
            return subprocess.CompletedProcess(argv, 0, "A3KERNEL_OUTPUT=[123,456]", "")
    backend = LocalNativeCandidateBackend(config(tmp_path), runner=Runner())
    candidate = plan()
    compiled = backend.compile(candidate, tmp_path / "evidence")
    receipt = backend.execute_receipt(compiled)
    assert receipt.output == (123., 456.)  # Host oracle, not transport, grades values.
    assert receipt.job_handle.startswith("local:")
    root = backend.remote_candidate_directory(candidate)
    assert json.loads((root / "build.json").read_text())["arch"] == "discovered-arch"
    assert json.loads((root / "input.json").read_text())["inputs"][0]["values"] == [1., 2.]
    assert all("remote" not in str(arg) for argv, _ in calls for arg in argv)
    (root / "a3_candidate.so").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="library changed"):
        backend.execute_receipt(compiled)


def test_local_native_distinguishes_compiler_failure_from_host_failure(tmp_path):
    class Runner:
        def run(self, argv, directory, **options):
            return subprocess.CompletedProcess(argv, 1, "A3CANDIDATE_COMPILE_FAILED=1", "bad candidate")
    result = LocalNativeCandidateBackend(config(tmp_path), runner=Runner()).compile(plan(), tmp_path / "evidence")
    assert isinstance(result, FailedEvidence)
    assert result.error_type == "NativeCompileError"
    assert "exit code 1" in result.detail
    assert "A3CANDIDATE_COMPILE_FAILED=1" in result.detail
    assert "bad candidate" in result.detail


def test_native_worker_config_replaces_remote_requirements_before_transport(tmp_path, monkeypatch):
    import benchmarks.ascend.execution as execution
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps(dict(guide="guide.md", target="discovered-a3", cpl_skills_revision="a" * 40)))
    worker = tmp_path / "worker.json"
    worker.write_text(json.dumps(config(tmp_path)))
    monkeypatch.setattr(execution, "preflight_local", lambda settings: dict(
        cann_version="discovered-cann", profiler_version="discovered-msprof", compiler_version="discovered-compiler"))
    settings = load_native_config(workflow, worker)
    assert settings["execution_profile"] == "local"
    assert "cpl_remote" not in settings
    transport = native_transport(settings, tmp_path, factory=lambda **kw: pytest.fail("remote constructor invoked"))
    assert isinstance(transport, LocalNativeCandidateBackend)


def test_local_report_collection_copies_digest_checked_tree_without_transport(tmp_path, monkeypatch):
    from benchmarks.ascend.artifact_collector import ArtifactCollector
    from benchmarks.ascend.training_annex import report_digest
    import benchmarks.ascend.artifact_collector as collector
    source = tmp_path / "profiler-output"
    source.mkdir()
    (source / "metrics.csv").write_text("kernel,duration\ncompute,12\n")
    worker = ArtifactCollector(tmp_path / "campaign", tmp_path / "reports", tmp_path / "state.json",
                               execution="local", free_floor_bytes=0)
    monkeypatch.setattr(collector.subprocess, "run", lambda *a, **kw: pytest.fail("remote transport invoked"))
    entry = dict(id="report-one", target="local", remote_path=str(source), status="pending")
    state = dict(entries={entry["id"]: entry})
    worker._transfer(state, entry, 0)
    assert entry["status"] == "complete"
    assert report_digest(tmp_path / "reports/report-one") == report_digest(source)
    assert entry["expected_sha256"] == report_digest(source)


@pytest.mark.parametrize("entrypoint", ["run_a3_phase1", "run_a3_phase2"])
def test_fixed_a3_entrypoint_accepts_worker_config_without_remote_arguments(tmp_path, entrypoint):
    import importlib
    from run_a3_phase1 import _worker_settings
    worker = tmp_path / "worker.json"
    worker.write_text(json.dumps(config(tmp_path)))
    args = importlib.import_module(entrypoint)._parser().parse_args([
        "preflight", "--state-root", str(tmp_path / "state"), "--worker-config", str(worker)])
    _worker_settings(args)
    assert args.profile == "local"
    assert args.cpl_remote is None
    assert args.physical_device == 0


def test_legacy_vectoradd_plan_uses_local_compiler_architecture(tmp_path):
    from benchmarks.a3kernels.live_composition import RemoteCandidateBundle
    from benchmarks.a3kernels.candidate import CANDIDATE_EXPORTED_SIGNATURE
    bundle = RemoteCandidateBundle(LocalNativeCandidateBackend(config(tmp_path)))
    candidate = bundle._plan(CANDIDATE_EXPORTED_SIGNATURE + " {}", dict(
        request_id="request", attempt_id="attempt", length=4, seed=1))
    assert candidate.execution_profile == "local"
    build = next(item for item in candidate.files if item.relative_path == "build.json")
    assert json.loads(build.content)["arch"] == "discovered-arch"


def test_failed_native_execution_explains_signal_even_with_correct_looking_stdout(tmp_path):
    class Runner:
        def run(self, argv, operation_directory, **options):
            if "--compile-only" in argv:
                library = Path(options["cwd"]) / "a3_candidate.so"
                library.write_bytes(b"compiled-library")
                digest = hashlib.sha256(library.read_bytes()).hexdigest()
                return subprocess.CompletedProcess(argv, 0, "A3CANDIDATE_COMPILED=" + digest, "")
            return subprocess.CompletedProcess(argv, -15, "A3KERNEL_OUTPUT=[1,2]", "termination diagnostic")
    backend = LocalNativeCandidateBackend(config(tmp_path), runner=Runner())
    compilation = backend.compile(plan(), tmp_path / "evidence")
    result = backend.execute_receipt(compilation)
    assert isinstance(result, FailedEvidence)
    assert result.error_type == "NativeRuntimeError"
    assert "exit code -15" in result.detail
    assert "SIGTERM" in result.detail
    assert "A3KERNEL_OUTPUT=[1,2]" in result.detail
    assert "termination diagnostic" in result.detail
