from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import venv

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


def test_plan_identity_binds_logical_padded_and_block_dimensions():
    backend = A3CandidateBackend(command_runner=lambda *a, **k: None)
    base = backend.plan(
        SOURCE, request_id="r1", attempt_id="a1", length=33,
        padded_length=64, block_count=2, seed=7,
    )

    assert (base.logical_length, base.padded_length, base.block_count) == (33, 64, 2)
    assert len(base.input_a) == len(base.input_b) == 64
    assert base.input_a[33:] == base.input_b[33:] == (0.0,) * 31
    assert base.execution_id != backend.plan(
        SOURCE, request_id="r1", attempt_id="a1", length=33,
        padded_length=65, block_count=2, seed=7,
    ).execution_id
    assert base.execution_id != backend.plan(
        SOURCE, request_id="r1", attempt_id="a1", length=33,
        padded_length=64, block_count=3, seed=7,
    ).execution_id


def test_compile_and_run_bind_explicit_bz_execution_profile(tmp_path):
    commands = []

    def command(argv, **kwargs):
        commands.append(tuple(argv))
        if "--compile-only" in argv:
            return subprocess.CompletedProcess(
                argv, 0, "A3CANDIDATE_COMPILED=" + "a" * 64 + "\n", ""
            )
        payload = json.loads((kwargs["cwd"] / "input.json").read_text())
        values = [a + b for a, b in zip(payload["input_a"], payload["input_b"])]
        return subprocess.CompletedProcess(
            argv, 0, "A3KERNEL_OUTPUT=" + json.dumps(values) + "\n", ""
        )
    backend = A3CandidateBackend(command)
    compiled = backend.compile(
        SOURCE, tmp_path / "compile", request_id="r", attempt_id="a",
        length=4, execution_profile="bz-a3-1",
    )
    verified = backend.run(
        SOURCE, tmp_path / "run", request_id="r", attempt_id="b",
        length=4, execution_profile="bz-a3-2",
    )
    assert compiled.plan.execution_profile == "bz-a3-1"
    assert verified.passed
    assert verified.execution_id == backend.plan(
        SOURCE, request_id="r", attempt_id="b", length=4, seed=0,
        execution_profile="bz-a3-2",
    ).execution_id
    assert commands


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


def test_checked_driver_emits_unprofiled_multiblock_timing_samples(tmp_path, capsys):
    _candidate_directory(tmp_path, length=4, padded_length=6, block_count=1)
    calls = []
    def process(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        assert json.loads((tmp_path / argv[-1]).read_text())["block_count"] == 3
        return subprocess.CompletedProcess(
            argv, 0,
            "A3INNER_TIMING_US=2.000000\nA3INNER_TIMING_US=4.000000\n"
            "A3KERNEL_OUTPUT=[3,3,3,3,0,0]\n", "",
        )

    result = a3_profile_driver.main(
        ["timing", *_driver_args(tmp_path, block_count=3), "--warm-up", "1", "--launch-count", "2"],
        process_runner=process,
    )

    output = capsys.readouterr().out
    assert result == 0 and len(calls) == 1
    assert calls[0][0][-8:] == (
        "host_driver.py", "--mode", "benchmark", "--warm-up", "1",
        "--launch-count", "2", ".a3-profile-input.json",
    )
    assert "A3TIMING_US=2.000000" in output
    assert "A3TIMING_US=4.000000" in output
    staged = json.loads((tmp_path / ".a3-profile-input.json").read_text())
    assert (staged["logical_length"], staged["padded_length"], staged["block_count"]) == (4, 6, 3)


@pytest.mark.parametrize(
    "metric", ["Basic", "ArithmeticUtilization", "PipeUtilization"]
)
def test_checked_driver_profiles_one_raw_metric_and_retains_report(
    tmp_path, capsys, metric,
):
    _candidate_directory(tmp_path, length=4, padded_length=6, block_count=1)
    calls = []

    def process(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[0] == "msprof":
            output = Path(next(item.split("=", 1)[1] for item in argv if item.startswith("--output=")))
            report = output / "PROF_1" / "mindstudio_profiler_output"
            report.mkdir(parents=True)
            (report / "op_summary_0.csv").write_text(
                "Op Name,Duration(us),Raw Counter\nvector_add,12.5,7\n"
            )
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "A3KERNEL_OUTPUT=[3,3,3,3,0,0]\n", "")

    assert a3_profile_driver.main(
        ["profile", *_driver_args(tmp_path, block_count=4), "--metric", metric, "--kernel", "vector_add"],
        process_runner=process,
        report_directory_factory=lambda root, selected: root / f"report-{selected}",
    ) == 0

    output = capsys.readouterr().out
    compact = json.loads(next(line.split("=", 1)[1] for line in output.splitlines()
                              if line.startswith("A3PROFILE_COMPACT=")))
    meta = json.loads(next(line.split("=", 1)[1] for line in output.splitlines()
                           if line.startswith("A3PROFILE_META=")))
    assert compact["exported_kernels"] == ["vector_add"]
    table = "PROF_1/mindstudio_profiler_output/op_summary_0.csv"
    assert compact["metric_values"] == [[f"{table}:Raw Counter:0", 7.0]]
    assert compact["timeline"] == [[f"{table}:Duration(us):0", 12.5]]
    assert compact["selected_columns"] == [
        f"{table}:Duration(us)", f"{table}:Op Name", f"{table}:Raw Counter"
    ]
    assert len(compact["report_sha256"]) == 64
    assert Path(meta["remote_report"]).is_dir()
    profile = next(call for call in calls if call[0] == "msprof")
    host_calls = [call for call in calls if call[0] != "msprof"]
    assert len(host_calls) == 1
    assert host_calls[0][-8:] == (
        "host_driver.py", "--mode", "replay", "--warm-up", "0",
        "--launch-count", "1", ".a3-profile-input.json",
    )
    assert f"--aic-metrics={metric}" in profile
    assert not any(item.startswith("--application=") for item in profile)
    assert profile[3:] == (
        a3_profile_driver.sys.executable,
        str((tmp_path / "host_driver.py").resolve()),
        "--mode", "replay", "--warm-up", "0",
        "--launch-count", "1", ".a3-profile-input.json",
    )
    staged = json.loads((tmp_path / ".a3-profile-input.json").read_text())
    assert (staged["logical_length"], staged["padded_length"], staged["block_count"]) == (4, 6, 4)


def test_replay_preserves_real_symlinked_venv_interpreter(tmp_path, monkeypatch):
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    interpreter = environment / "bin" / "python"
    assert interpreter.is_symlink()
    assert interpreter.resolve() != interpreter
    monkeypatch.setattr(a3_profile_driver.sys, "executable", str(interpreter))

    argv = a3_profile_driver._replay_argv(
        tmp_path, "input.json", argparse.Namespace(launch_count=1)
    )

    assert argv[0] == str(interpreter)


def test_checked_driver_rejects_failed_verification_and_out_of_range_blocks(tmp_path):
    _candidate_directory(tmp_path, length=4, block_count=1)

    def failed(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 2, "", "launch failed")

    with pytest.raises(RuntimeError, match="launch failed"):
        a3_profile_driver.main(["timing", *_driver_args(tmp_path)], process_runner=failed)
    with pytest.raises(ValueError, match="block-count"):
        a3_profile_driver.main(
            ["timing", *_driver_args(tmp_path, block_count=33)],
            process_runner=failed,
        )


def _candidate_directory(
    root: Path, *, length: int, block_count: int, padded_length: int | None = None,
) -> None:
    padded = padded_length or length
    padding = [0] * (padded - length)
    for name in ("host_driver.py", "input.json", "a3_candidate.so"):
        (root / name).write_text("{}" if name == "input.json" else "staged")
    (root / "input.json").write_text(json.dumps({
        "input_a": [1] * length + padding, "input_b": [2] * length + padding,
        "logical_length": length, "padded_length": padded, "block_count": block_count,
    }))


def _driver_args(root: Path, *, block_count: int = 1) -> list[str]:
    return [
        "--candidate-dir", str(root.resolve()), "--logical-device", "0",
        "--length", "4", "--block-count", str(block_count), "--warm-up", "0",
        "--launch-count", "1",
    ]


@pytest.mark.parametrize(
    "source",
    [
        "void vector_add() {}",
        SOURCE.replace("extern \"C\" ", ""),
        SOURCE.replace("uint32_t buffer_bytes", "uint64_t buffer_bytes"),
        SOURCE.replace("GM_ADDR input_a", "GM_ADDRinput_a"),
        SOURCE.replace("__global__ __aicore__", "__global____aicore__"),
        SOURCE.replace('"C" __global__', '"C"__global__'),
        SOURCE + "\nextern \"C\" void vector_add();",
        SOURCE + "\n// catlass fallback",
    ],
)
def test_candidate_requires_exact_exported_signature(source):
    with pytest.raises(ValueError, match="candidate"):
        validate_candidate_source(source)


def test_candidate_signature_allows_compact_punctuation():
    source = candidate_module.CANDIDATE_EXPORTED_SIGNATURE.replace(", ", ",") + "{}"

    validate_candidate_source(source)


def test_compile_only_uses_fixed_host_command_and_records_library(tmp_path):
    calls = []

    def command(argv, *, cwd, text, capture_output, check):
        calls.append(tuple(argv))
        assert (cwd / "candidate.cpp").read_text() == SOURCE
        wrapper = (cwd / "host_wrapper.inc").read_text()
        assert "TORCH_LIBRARY(rsi_a3candidates" in wrapper
        assert "int64_t logical_length, int64_t padded_length" in wrapper
        assert "int64_t block_count" in wrapper
        assert "static_cast<uint32_t>(block_count)" in wrapper
        return subprocess.CompletedProcess(argv, 0, "A3CANDIDATE_COMPILED=" + "a" * 64 + "\n", "")

    result = A3CandidateBackend(command).compile(
        SOURCE, tmp_path, request_id="r1", attempt_id="a1"
    )

    assert isinstance(result, CandidateCompilation)
    assert result.library_sha256 == "a" * 64
    assert result.attestation_sha256
    assert calls == [("python", "host_driver.py", "--compile-only")]
    assert result.plan.source_fingerprint


@pytest.mark.parametrize("length,padded_length", [(32, 32), (33, 64)])
def test_run_compiles_then_executes_aligned_and_padded_shapes(
    tmp_path, length, padded_length
):
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
        attempt_id="a1", length=length, padded_length=padded_length,
        block_count=2, seed=9,
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


@pytest.mark.parametrize("logical,padded", [(16, 64), (400, 448)])
def test_fixed_driver_allocates_padding_but_launches_logical_count(
    monkeypatch, capsys, tmp_path, logical, padded
):
    class Tensor:
        def __init__(self, values): self.values = values
        def cpu(self): return self
        def tolist(self): return self.values

    events = []
    torch = type("Torch", (), {})()
    torch.float32 = object()
    torch.tensor = lambda values, **kwargs: (
        events.append(("allocate", kwargs["device"], len(values))), Tensor(values)
    )[1]
    torch.npu = type("Npu", (), {
        "set_device": lambda _self, value: events.append(("device", value)),
        "synchronize": lambda _self: events.append("sync"),
    })()
    torch.ops = type("Ops", (), {
        "load_library": lambda _self, path: events.append(("load", path)),
        "rsi_a3candidates": type("Candidate", (), {
            "vector_add": lambda _self, a, b, count, padded_length, block_count: (
                events.append(("launch", count, padded_length, block_count)),
                Tensor(
                    [x + y for x, y in zip(a.values[:count], b.values[:count])]
                    + [99.0] * (padded_length - count)
                ),
            )[1],
        })(),
    })()
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    monkeypatch.setitem(__import__("sys").modules, "torch_npu", type("N", (), {})())
    monkeypatch.setattr(host_driver, "__file__", str(tmp_path / "host_driver.py"))
    (tmp_path / "build.json").write_text(json.dumps(host_driver._EXPECTED_BUILD))
    (tmp_path / "a3_candidate.so").write_bytes(b"library")
    (tmp_path / "input.json").write_text(json.dumps({
        "input_a": [1.0] * logical + [0.0] * (padded - logical),
        "input_b": [2.0] * logical + [0.0] * (padded - logical),
        "logical_length": logical,
        "padded_length": padded,
        "block_count": 3,
    }))

    assert host_driver.main(["input.json"]) == 0
    assert events == [
        ("load", str(tmp_path / "a3_candidate.so")), ("device", 0),
        ("allocate", "npu:0", padded), ("allocate", "npu:0", padded),
        ("launch", logical, padded, 3), "sync",
    ]
    values = json.loads(capsys.readouterr().out.removeprefix("A3KERNEL_OUTPUT="))
    assert values[:logical] == [3.0] * logical
    assert values[logical:] == [0.0] * (padded - logical)


def test_fixed_driver_benchmarks_in_process_after_warmup(monkeypatch, capsys, tmp_path):
    class Tensor:
        def __init__(self, values): self.values = values
        def cpu(self): return self
        def tolist(self): return self.values

    events = []
    torch = type("Torch", (), {})()
    torch.float32 = object()
    torch.tensor = lambda values, **kwargs: Tensor(values)
    torch.npu = type("Npu", (), {
        "set_device": lambda _self, value: events.append(("device", value)),
        "synchronize": lambda _self: events.append("sync"),
    })()
    torch.ops = type("Ops", (), {
        "load_library": lambda _self, path: events.append("load"),
        "rsi_a3candidates": type("Candidate", (), {
            "vector_add": lambda _self, a, b, logical, padded, blocks: (
                events.append(("launch", blocks)),
                Tensor([x + y for x, y in zip(a.values, b.values)]),
            )[1],
        })(),
    })()
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    monkeypatch.setitem(__import__("sys").modules, "torch_npu", type("N", (), {})())
    monkeypatch.setattr(host_driver, "__file__", str(tmp_path / "host_driver.py"))
    (tmp_path / "build.json").write_text(json.dumps(host_driver._EXPECTED_BUILD))
    (tmp_path / "a3_candidate.so").write_bytes(b"library")
    _candidate_directory(tmp_path, length=4, block_count=3)
    ticks = iter((0, 2000, 3000, 7000))

    assert host_driver.main(
        ["--mode", "benchmark", "--warm-up", "2", "--launch-count", "2", "input.json"],
        clock_ns=lambda: next(ticks),
    ) == 0

    assert events.count(("launch", 3)) == 4
    assert events.count("sync") == 4
    assert events.count("load") == 1
    output = capsys.readouterr().out
    assert output.count("A3INNER_TIMING_US=") == 2
    assert "A3INNER_TIMING_US=2.000000" in output
    assert "A3INNER_TIMING_US=4.000000" in output
    assert output.endswith("A3KERNEL_OUTPUT=[3.0,3.0,3.0,3.0]\n")


def test_fixed_wrapper_launches_the_requested_block_count():
    wrapper = next(
        item.content for item in candidate_module._host_source_files()
        if item.relative_path == "host_wrapper.inc"
    )
    assert "int64_t logical_length, int64_t padded_length" in wrapper
    assert "int64_t block_count" in wrapper
    assert 'registry.def("vector_add(Tensor a, Tensor b, int logical_length, int padded_length, int block_count)' in wrapper
    assert "logical_length >= 1 && logical_length <= padded_length" in wrapper
    assert "const auto count = static_cast<uint32_t>(logical_length);" in wrapper
    assert "torch::empty_like(a)" in wrapper
    assert "block_count >= 1 && block_count <= 32" in wrapper
    assert "const auto launch_blocks = static_cast<uint32_t>(block_count);" in wrapper
    assert "::vector_add<<<launch_blocks" in wrapper
    assert wrapper.index("block_count >= 1") < wrapper.index("static_cast<uint32_t>(block_count)")
