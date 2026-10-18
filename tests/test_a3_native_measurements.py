import json
import hashlib
import importlib.util
from importlib.resources import files
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks.a3kernels.candidate import CandidateCompilation
from benchmarks.a3kernels.native_smoke import smoke_plan
from benchmarks.a3kernels.phase1_protocol import ExecutionReceipt
from benchmarks.a3kernels.native_measurements import NativeMeasurementBackend


def setup_backend(tmp_path, *, output=None, runner=None, **kwargs):
    plan = smoke_plan(target="discovered-a3", arch="discovered-arch")
    compilation = CandidateCompilation(plan, "b" * 64, "compile", "", "a" * 64)
    expected = tuple(float(3 * i - 1) for i in range(32))
    receipt = ExecutionReceipt(0, expected if output is None else output)
    candidate = SimpleNamespace(execute_receipt=lambda _: receipt)
    backend = NativeMeasurementBackend(
        compilation=compilation, candidate_backend=candidate, expected_output=expected,
        validation_wrapper="/checked/catlass-validation.sh", cpl_remote="/tools/cpl-remote",
        profile="bz-a3-1", remote_candidate_directory="/owned/candidate",
        evidence_directory=tmp_path, physical_device=1, process_runner=runner,
        environment_identity=dict(cann_version="discovered-cann", profiler_version="discovered-profiler"),
        **kwargs,
    )
    return backend


def terminal(mode, body):
    meta = dict(mode=mode, target="discovered-a3", kernel_name="sum_three", logical_device=0,
                language="ascend-c", runtime="native-ascend-c", remote_report="/owned/report" if mode == "profile" else None)
    output = "A3KERNEL_OUTPUT=" + json.dumps([float(3 * i - 1) for i in range(32)])
    return subprocess.CompletedProcess((), 0, "\n".join((body, output, "A3PROFILE_META=" + json.dumps(meta),
        "CATLASS_VALIDATION_PROFILE=bz-a3-1", "CATLASS_VALIDATION_STATE=completed",
        "CATLASS_VALIDATION_HANDLE=bz-a3-1:retained", "CATLASS_VALIDATION_EXIT=0")), "")


def test_timing_is_host_verified_binds_shape_dtype_and_reuses_retained_result(tmp_path):
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        return terminal("timing", "A3TIMING_US=10\nA3TIMING_US=12")
    backend = setup_backend(tmp_path, runner=runner)
    result = backend.timing(warm_up=1, launch_count=2)
    assert result["summary"]["median_us"] == 11
    assert result["summary"]["instrumented"] is False
    assert result["identity"]["kernel_name"] == "sum_three"
    assert result["identity"]["inputs"][0]["shape"] == [2, 16]
    assert backend.timing(warm_up=1, launch_count=2) == result
    assert len(calls) == 1
    assert "msprof" not in calls[0]


def test_wrong_host_output_blocks_measurement_before_transport(tmp_path):
    backend = setup_backend(tmp_path, output=(0.0,) * 32, runner=lambda *_a, **_kw: pytest.fail("launched"))
    with pytest.raises(ValueError, match="host verification"):
        backend.timing()


def test_measured_replay_output_is_also_host_checked(tmp_path):
    def runner(argv, **kwargs):
        completed = terminal("timing", "A3TIMING_US=10")
        output = json.dumps([float(3 * i - 1) for i in range(32)])
        completed.stdout = completed.stdout.replace(output, json.dumps([0.0] * 32))
        return completed
    with pytest.raises(ValueError, match="host verification"):
        setup_backend(tmp_path, runner=runner).timing(launch_count=1)


def test_missing_samples_are_not_reported_as_success(tmp_path):
    with pytest.raises(RuntimeError, match="sample count"):
        setup_backend(tmp_path, runner=lambda *a, **k: terminal("timing", "A3TIMING_US=10")).timing(launch_count=2)


def test_profile_exact_manifest_kernel_keeps_raw_metrics_and_report_identity(tmp_path):
    compact = dict(exported_kernels=["sum_three"], metric_values=[["report.csv:Vector Ratio:0", 0.5]],
                   timeline=[], report_sha256="f" * 64, selected_columns=["report.csv:Vector Ratio"])
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        return terminal("profile", "A3PROFILE_COMPACT=" + json.dumps(compact))
    result = setup_backend(tmp_path, runner=runner).capture("PipeUtilization")
    assert result["summary"]["instrumented"] is True
    assert result["summary"]["metric_values"][0][1] == 0.5
    assert result["summary"]["report_sha256"] == "f" * 64
    assert calls[0][calls[0].index("--kernel") + 1] == "sum_three"


def test_unavailable_metric_is_not_a_successful_empty_capture(tmp_path):
    backend = setup_backend(tmp_path, runner=lambda *_a, **_kw: pytest.fail("launched"))
    with pytest.raises(ValueError, match="unavailable"):
        backend.capture("Memory")


def test_replay_plan_keeps_frozen_host_driver_assets(monkeypatch):
    from benchmarks.a3kernels import native_candidate
    plan = smoke_plan(target="discovered-a3", arch="discovered-arch")
    identity = plan.execution_id
    monkeypatch.setattr(native_candidate, "files", lambda *_: pytest.fail("reread mutable installed assets"))
    assert plan.execution_id == identity


def load_driver(name, monkeypatch):
    root = files("benchmarks.a3kernels.candidate_runtime")
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(name, root.joinpath(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_benchmark_warms_up_outside_repeated_synchronized_samples(tmp_path, monkeypatch, capsys):
    import sys
    monkeypatch.setitem(sys.modules, "legacy_driver", SimpleNamespace(_compile=lambda *a: pytest.fail("compiled during timing")))
    driver = load_driver("native_driver", monkeypatch)
    driver.__file__ = str(tmp_path / "host_driver.py")
    (tmp_path / "input.json").write_text(json.dumps(dict(inputs=[dict(values=[1], dtype="float32", shape=[1])], output=dict(shape=[1]))))
    events = []
    output = SimpleNamespace(shape=(1,), cpu=lambda: output, float=lambda: output,
                             reshape=lambda _: output, tolist=lambda: [2.0])
    def launch(_):
        events.append("launch")
        return output
    torch = SimpleNamespace(float32="fp32", tensor=lambda *a, **k: output,
        ops=SimpleNamespace(load_library=lambda _: None, rsi_a3native=SimpleNamespace(launch=launch)),
        npu=SimpleNamespace(set_device=lambda _: None, synchronize=lambda: events.append("sync")))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(sys, "argv", ["host_driver.py", "--mode", "benchmark", "--warm-up", "2", "--launch-count", "2", "input.json"])
    ticks = iter((1000, 3000, 5000, 9000))
    driver.main(clock_ns=lambda: next(ticks))
    assert events == ["launch", "sync"] * 4
    stdout = capsys.readouterr().out
    assert "A3TIMING_US=2.000000" in stdout and "A3TIMING_US=4.000000" in stdout


def test_generic_profile_driver_uses_actual_selector_and_preserves_report_units(tmp_path, monkeypatch, capsys):
    driver = load_driver("native_measurement_driver", monkeypatch)
    (tmp_path / "build.json").write_text(json.dumps(dict(kernel_name="sum_three", target="actual-a3")))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(execution_id="e", source_fingerprint="s")))
    binary = b"compiled"
    (tmp_path / "a3_candidate.so").write_bytes(binary)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        report = next(a.removeprefix("--output=") for a in argv if a.startswith("--output="))
        from pathlib import Path
        (Path(report) / "report.csv").write_text("Kernel Name,Vector Ratio,Duration(us)\nsum_three,0.5,7\nother,0.9,100\n")
        return subprocess.CompletedProcess(argv, 0, "A3KERNEL_OUTPUT=[2]\n", "")
    driver.main(["profile", "--candidate-dir", str(tmp_path), "--kernel", "sum_three", "--execution-id", "e", "--source-fingerprint", "s", "--library-sha256", hashlib.sha256(binary).hexdigest(), "--metric", "PipeUtilization"], runner=run)
    assert calls[0][0] == "msprof"
    text = capsys.readouterr().out
    compact = json.loads(next(line.removeprefix("A3PROFILE_COMPACT=") for line in text.splitlines() if line.startswith("A3PROFILE_COMPACT=")))
    assert compact["exported_kernels"] == ["sum_three"]
    assert compact["metric_values"][0][1] == 0.5
    assert compact["timeline"][0][1] == 7
