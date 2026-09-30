from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

import benchmarks.a3kernels.candidate as candidate_module
from benchmarks.a3kernels.candidate_runtime import a3_profile_driver, host_driver
from benchmarks.a3kernels.candidate import (
    A3CandidateBackend,
    CandidateCompilation,
    FailedEvidence,
    VerifiedResult,
    validate_candidate_source,
)


SOURCE = r'''#include "kernel_operator.h"
extern "C" __global__ __aicore__ void vector_add(
    GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output,
    uint32_t count, uint32_t buffer_bytes) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::QuePosition::VECIN> a_buffer;
  AscendC::TBuf<AscendC::QuePosition::VECIN> b_buffer;
  AscendC::TBuf<AscendC::QuePosition::VECOUT> output_buffer;
  pipe.InitBuffer(a_buffer, buffer_bytes);
  pipe.InitBuffer(b_buffer, buffer_bytes);
  pipe.InitBuffer(output_buffer, buffer_bytes);
  AscendC::GlobalTensor<float> a;
  AscendC::GlobalTensor<float> b;
  AscendC::GlobalTensor<float> c;
  a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input_a), count);
  b.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input_b), count);
  c.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output), count);
  auto a_local = a_buffer.Get<float>();
  auto b_local = b_buffer.Get<float>();
  auto output_local = output_buffer.Get<float>();
  AscendC::DataCopyExtParams copy{};
  copy.blockCount = 1;
  copy.blockLen = count * sizeof(float);
  copy.srcStride = 0;
  copy.dstStride = 0;
  AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
  AscendC::DataCopyPad(a_local, a, copy, padding);
  AscendC::DataCopyPad(b_local, b, copy, padding);
  auto loaded = pipe.FetchEventID<AscendC::HardEvent::MTE2_V>();
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(loaded);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(loaded);
  AscendC::Add(output_local, a_local, b_local, count);
  auto computed = pipe.FetchEventID<AscendC::HardEvent::V_MTE3>();
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(computed);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(computed);
  AscendC::DataCopyPad(c, output_local, copy);
}
'''


def _output(cwd: Path) -> list[float]:
    payload = json.loads((cwd / "input.json").read_text())
    return [a + b for a, b in zip(payload["input_a"], payload["input_b"])]


def test_candidate_owns_one_exact_source_and_identity_changes_with_it():
    backend = A3CandidateBackend(command_runner=lambda *a, **k: None)
    first = backend.plan(SOURCE, request_id="r1", attempt_id="a1", length=33, seed=7)
    second = backend.plan(SOURCE + "\n", request_id="r1", attempt_id="a1", length=33, seed=7)

    assert [item.relative_path for item in first.files] == [
        "a3_profile_driver.py", "build.json", "candidate.cpp", "host_driver.py", "host_wrapper.inc"
    ]
    assert [item.relative_path for item in first.files if item.relative_path == "candidate.cpp"] == [
        "candidate.cpp"
    ]
    assert first.argv == ("python", "host_driver.py", "input.json")
    assert first.logical_device == 0
    assert first.source_fingerprint != second.source_fingerprint
    assert first.execution_id != second.execution_id


@pytest.mark.parametrize(
    "changed_path",
    ["a3_profile_driver.py", "build.json", "host_driver.py", "host_wrapper.inc"],
)
def test_every_host_asset_is_identity_bound_and_staged_from_plan(monkeypatch, tmp_path, changed_path):
    backend = A3CandidateBackend(command_runner=lambda *args, **kwargs: None)
    original = backend.plan(
        SOURCE, request_id="r1", attempt_id="a1", length=4, seed=0
    )
    original_host = tuple(item for item in original.files if item.relative_path != "candidate.cpp")
    changed_host = tuple(
        type(item)(item.relative_path, item.content + "\n// changed")
        if item.relative_path == changed_path else item
        for item in original_host
    )
    monkeypatch.setattr(candidate_module, "_host_source_files", lambda: changed_host)
    changed = backend.plan(
        SOURCE, request_id="r1", attempt_id="a1", length=4, seed=0
    )

    assert changed.source_fingerprint != original.source_fingerprint
    assert changed.execution_id != original.execution_id
    backend._stage(changed, tmp_path)
    assert {
        item.relative_path: (tmp_path / item.relative_path).read_text()
        for item in changed.files
    } == {item.relative_path: item.content for item in changed.files}


def test_profile_driver_asset_has_explicit_bytes_and_digest_contract():
    asset = candidate_module.profile_driver_asset()

    assert asset.relative_path == "a3_profile_driver.py"
    assert asset.content.startswith('"""Checked A3 timing')
    assert len(asset.sha256) == 64


def test_checked_driver_emits_unprofiled_warmups_and_timing_samples(tmp_path, capsys):
    _candidate_directory(tmp_path, length=4)
    calls = []
    ticks = iter((0, 2000, 3000, 7000))

    def process(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, "A3KERNEL_OUTPUT=[3,3,3,3]\n", "")

    result = a3_profile_driver.main(
        ["timing", *_driver_args(tmp_path), "--warm-up", "1", "--launch-count", "2"],
        process_runner=process,
        clock_ns=lambda: next(ticks),
    )

    output = capsys.readouterr().out
    assert result == 0 and len(calls) == 3
    assert all(call[0][-2:] == ("host_driver.py", "input.json") for call in calls)
    assert "A3TIMING_US=2.000000" in output
    assert "A3TIMING_US=4.000000" in output


@pytest.mark.parametrize("metric", ["Basic", "PipeUtilization"])
def test_checked_driver_profiles_one_raw_metric_and_retains_report(tmp_path, capsys, metric):
    _candidate_directory(tmp_path, length=4)
    calls = []

    def process(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[0] == "msprof":
            output = Path(next(item.split("=", 1)[1] for item in argv if item.startswith("--output=")))
            report = output / "PROF_1" / "mindstudio_profiler_output"
            report.mkdir(parents=True)
            (report / f"{metric}.csv").write_text(
                "Op Name,Duration(us),Raw Counter\nvector_add,12.5,7\n"
            )
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "A3KERNEL_OUTPUT=[3,3,3,3]\n", "")

    assert a3_profile_driver.main(
        ["profile", *_driver_args(tmp_path), "--metric", metric, "--kernel", "vector_add"],
        process_runner=process,
        report_directory_factory=lambda root, selected: root / f"report-{selected}",
    ) == 0

    output = capsys.readouterr().out
    compact = json.loads(next(line.split("=", 1)[1] for line in output.splitlines()
                              if line.startswith("A3PROFILE_COMPACT=")))
    meta = json.loads(next(line.split("=", 1)[1] for line in output.splitlines()
                           if line.startswith("A3PROFILE_META=")))
    assert compact["exported_kernels"] == ["vector_add"]
    table = f"PROF_1/mindstudio_profiler_output/{metric}.csv"
    assert compact["metric_values"] == [[f"{table}:Raw Counter:0", 7.0]]
    assert compact["timeline"] == [[f"{table}:Duration(us):0", 12.5]]
    assert len(compact["report_sha256"]) == 64
    assert Path(meta["remote_report"]).is_dir()
    profile = next(call for call in calls if call[0] == "msprof")
    assert f"--aic-metrics={metric}" in profile


def test_checked_driver_rejects_failed_verification_and_unsupported_blocks(tmp_path):
    _candidate_directory(tmp_path, length=4)

    def failed(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 2, "", "launch failed")

    with pytest.raises(RuntimeError, match="launch failed"):
        a3_profile_driver.main(["timing", *_driver_args(tmp_path)], process_runner=failed)
    with pytest.raises(ValueError, match="block-count"):
        a3_profile_driver.main(
            ["timing", *_driver_args(tmp_path), "--block-count", "2"],
            process_runner=failed,
        )


def _candidate_directory(root: Path, *, length: int) -> None:
    for name in ("host_driver.py", "input.json", "a3_candidate.so"):
        (root / name).write_text("{}" if name == "input.json" else "staged")
    (root / "input.json").write_text(json.dumps({"input_a": [1] * length, "input_b": [2] * length}))


def _driver_args(root: Path) -> list[str]:
    return [
        "--candidate-dir", str(root.resolve()), "--logical-device", "0",
        "--length", "4", "--block-count", "1", "--warm-up", "0",
        "--launch-count", "1",
    ]


@pytest.mark.parametrize(
    "source",
    [
        "void vector_add() {}",
        SOURCE.replace("extern \"C\" ", ""),
        SOURCE.replace("uint32_t buffer_bytes", "uint64_t buffer_bytes"),
        SOURCE + "\nextern \"C\" void vector_add();",
        SOURCE + "\n// catlass fallback",
    ],
)
def test_candidate_requires_exact_exported_signature(source):
    with pytest.raises(ValueError, match="candidate"):
        validate_candidate_source(source)


def test_compile_only_uses_fixed_host_command_and_records_library(tmp_path):
    calls = []

    def command(argv, *, cwd, text, capture_output, check):
        calls.append(tuple(argv))
        assert (cwd / "candidate.cpp").read_text() == SOURCE
        assert "TORCH_LIBRARY(rsi_a3candidates" in (cwd / "host_wrapper.inc").read_text()
        return subprocess.CompletedProcess(argv, 0, "A3CANDIDATE_COMPILED=" + "a" * 64 + "\n", "")

    result = A3CandidateBackend(command).compile(
        SOURCE, tmp_path, request_id="r1", attempt_id="a1"
    )

    assert isinstance(result, CandidateCompilation)
    assert result.library_sha256 == "a" * 64
    assert result.attestation_sha256
    assert calls == [("python", "host_driver.py", "--compile-only")]
    assert result.plan.source_fingerprint


@pytest.mark.parametrize("length", [32, 33])
def test_run_compiles_then_executes_aligned_and_padded_shapes(tmp_path, length):
    calls = []

    def command(argv, *, cwd, text, capture_output, check):
        calls.append(tuple(argv))
        if "--compile-only" in argv:
            return subprocess.CompletedProcess(argv, 0, "A3CANDIDATE_COMPILED=" + "b" * 64 + "\n", "")
        return subprocess.CompletedProcess(
            argv, 0, "A3KERNEL_OUTPUT=" + json.dumps(_output(cwd)) + "\n", ""
        )

    result = A3CandidateBackend(command).run(
        SOURCE, tmp_path / str(length), request_id=f"r{length}",
        attempt_id="a1", length=length, seed=9,
    )

    assert isinstance(result, VerifiedResult)
    assert result.passed and result.max_abs_error == 0.0
    assert calls == [
        ("python", "host_driver.py", "--compile-only"),
        ("python", "host_driver.py", "input.json"),
    ]


def test_compile_failure_is_structured_and_stops_before_run(tmp_path):
    calls = []

    def command(argv, **kwargs):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 2, "", "bisheng: candidate.cpp:8: error")

    result = A3CandidateBackend(command).run(
        SOURCE, tmp_path, request_id="r", attempt_id="a", length=4
    )

    assert isinstance(result, FailedEvidence)
    assert result.stage == "compile" and result.error_type == "CompileError"
    assert "candidate.cpp:8" in result.detail
    assert len(calls) == 1


def test_runtime_failure_is_structured(tmp_path):
    def command(argv, **kwargs):
        if "--compile-only" in argv:
            return subprocess.CompletedProcess(argv, 0, "A3CANDIDATE_COMPILED=" + "c" * 64 + "\n", "")
        return subprocess.CompletedProcess(argv, 3, "", "ACL launch failed")

    result = A3CandidateBackend(command).run(
        SOURCE, tmp_path, request_id="r", attempt_id="a", length=4
    )

    assert isinstance(result, FailedEvidence)
    assert result.stage == "execute" and result.error_type == "RuntimeError"
    assert "ACL launch failed" in result.detail


@pytest.mark.parametrize("stdout", ["", "A3KERNEL_OUTPUT=nope\n", "A3KERNEL_OUTPUT=[1]\n"])
def test_malformed_or_wrong_length_output_is_structured_verification_failure(tmp_path, stdout):
    def command(argv, **kwargs):
        if "--compile-only" in argv:
            return subprocess.CompletedProcess(argv, 0, "A3CANDIDATE_COMPILED=" + "d" * 64 + "\n", "")
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    result = A3CandidateBackend(command).run(
        SOURCE, tmp_path, request_id="r", attempt_id="a", length=4
    )

    assert isinstance(result, FailedEvidence)
    assert result.stage == "verify" and result.error_type == "OutputError"


def test_agent_has_no_argv_or_build_descriptor_surface():
    with pytest.raises(TypeError):
        A3CandidateBackend(lambda *a, **k: None).plan(
            SOURCE, request_id="r", attempt_id="a", length=4, seed=0,
            argv=("bash", "-c", "anything"),
        )


def test_fixed_driver_compiles_dav_2201_without_a_shell(monkeypatch, tmp_path):
    torch_root, npu_root = tmp_path / "torch", tmp_path / "torch_npu"
    torch_root.mkdir()
    npu_root.mkdir()
    fake_torch = type("Torch", (), {
        "__file__": str(torch_root / "__init__.py"),
        "_C": type("C", (), {"_GLIBCXX_USE_CXX11_ABI": True}),
    })
    fake_npu = type("Npu", (), {"__file__": str(npu_root / "__init__.py")})
    (tmp_path / "candidate.cpp").write_text(SOURCE)
    (tmp_path / "host_wrapper.inc").write_text("// fixed wrapper")
    calls = []
    monkeypatch.setattr(host_driver.shutil, "which", lambda _name: "/cann/bin/bisheng")

    def command(argv, *, cwd, check):
        calls.append(tuple(argv))
        (tmp_path / "a3_candidate.so").write_bytes(b"library")

    monkeypatch.setattr(host_driver.subprocess, "run", command)
    output = host_driver._compile(
        tmp_path, dict(host_driver._EXPECTED_BUILD), fake_torch, fake_npu
    )

    assert output == tmp_path / "a3_candidate.so"
    assert calls[0][:7] == (
        "/cann/bin/bisheng", "-x", "asc", "--npu-arch=dav-2201",
        "-shared", "-fPIC", "-std=c++17",
    )
    assert not any(value in {"bash", "sh", "-c"} for value in calls[0])


def test_fixed_driver_owns_device_zero_allocation_launch_and_sync(monkeypatch, capsys, tmp_path):
    class Tensor:
        def __init__(self, values): self.values = values
        def cpu(self): return self
        def tolist(self): return self.values

    events = []
    torch = type("Torch", (), {})()
    torch.float32 = object()
    torch.tensor = lambda values, **kwargs: (
        events.append(("allocate", kwargs["device"])), Tensor(values)
    )[1]
    torch.npu = type("Npu", (), {
        "set_device": lambda _self, value: events.append(("device", value)),
        "synchronize": lambda _self: events.append("sync"),
    })()
    torch.ops = type("Ops", (), {
        "load_library": lambda _self, path: events.append(("load", path)),
        "rsi_a3candidates": type("Candidate", (), {
            "vector_add": lambda _self, a, b: (
                events.append("launch"),
                Tensor([x + y for x, y in zip(a.values, b.values)]),
            )[1],
        })(),
    })()
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    monkeypatch.setitem(__import__("sys").modules, "torch_npu", type("N", (), {})())
    monkeypatch.setattr(host_driver, "__file__", str(tmp_path / "host_driver.py"))
    (tmp_path / "build.json").write_text(json.dumps(host_driver._EXPECTED_BUILD))
    (tmp_path / "a3_candidate.so").write_bytes(b"library")
    (tmp_path / "input.json").write_text('{"input_a":[1,2],"input_b":[3,4]}')

    assert host_driver.main(["input.json"]) == 0
    assert events == [
        ("load", str(tmp_path / "a3_candidate.so")), ("device", 0),
        ("allocate", "npu:0"), ("allocate", "npu:0"), "launch", "sync",
    ]
    assert capsys.readouterr().out == "A3KERNEL_OUTPUT=[4.0,6.0]\n"
