from __future__ import annotations

from dataclasses import FrozenInstanceError
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.a3kernels import A3KernelRunner, RunRequest, VerifiedResult
from benchmarks.a3kernels.ascendc_runtime import host_driver
from benchmarks.a3kernels.fixture import source_files
import run_a3kernels as cli


def test_fixture_is_a3_specific_and_complete():
    sources = {item.relative_path: item.content for item in source_files()}
    descriptor = json.loads(sources["build.json"])

    assert tuple(sources) == ("build.json", "host_driver.py", "kernel.cpp")
    assert descriptor["arch"] == "dav-2201"
    assert descriptor["target"] == "Ascend910B4"
    assert descriptor["logical_device"] == 0
    assert descriptor["runtime"] == "py311-torch"
    assert "TORCH_LIBRARY(rsi_a3kernels" in sources["kernel.cpp"]
    assert "A3KERNEL_OUTPUT=" in sources["host_driver.py"]
    assert "A5KERNEL" not in "".join(sources.values())


def test_plan_inputs_and_identity_are_deterministic_and_immutable():
    request = RunRequest(length=7, seed=42)
    runner = A3KernelRunner()

    first = runner.prepare(request)
    second = runner.prepare(request)

    assert first == second
    assert first.request_id == request.request_id
    assert first.execution_id == second.execution_id
    assert first.logical_device == 0
    assert first.runtime == "gz-a3-native-py311-torch"
    with pytest.raises(FrozenInstanceError):
        request.seed = 1


def test_host_runner_scores_raw_output_and_attests_evidence(tmp_path):
    def command(argv, *, cwd, text, capture_output, check):
        assert argv[1:] == ("host_driver.py", "input.json")
        assert text and capture_output and not check
        payload = json.loads((cwd / "input.json").read_text())
        output = [a + b for a, b in zip(payload["input_a"], payload["input_b"])]
        return subprocess.CompletedProcess(
            argv, 0, "A3KERNEL_OUTPUT=" + json.dumps(output) + "\n", ""
        )

    result = A3KernelRunner(command).run(RunRequest(length=5, seed=3), tmp_path)

    assert result.passed is True
    assert result.max_abs_error == 0
    assert result.stdout.startswith("A3KERNEL_OUTPUT=")
    assert result.stderr == ""
    assert result.output_parse_error is None
    assert len(result.a3_evidence_sha256) == 64
    assert len(result.attestation_sha256) == 64
    assert (tmp_path / "kernel.cpp").is_file()


def test_host_runner_rejects_incorrect_output_even_on_zero_exit(tmp_path):
    def command(argv, **unused):
        return subprocess.CompletedProcess(argv, 0, "A3KERNEL_OUTPUT=[99]\n", "")

    result = A3KernelRunner(command).run(RunRequest(length=1), tmp_path)

    assert result.passed is False
    assert result.max_abs_error > 1


@pytest.mark.parametrize(
    "stdout, message",
    [
        ("compiler banner\n", "exactly one"),
        ("A3KERNEL_OUTPUT=[]\nA3KERNEL_OUTPUT=[]\n", "exactly one"),
        ("A3KERNEL_OUTPUT=not-json\n", "Expecting value"),
        ('A3KERNEL_OUTPUT=[1,"bad"]\n', "numeric array"),
    ],
)
def test_malformed_success_output_becomes_attested_failure(tmp_path, stdout, message):
    def command(argv, **unused):
        return subprocess.CompletedProcess(argv, 0, stdout, "runtime warning")

    result = A3KernelRunner(command).run(RunRequest(length=1), tmp_path)

    assert result.passed is False
    assert result.exit_code == 0
    assert result.max_abs_error is None
    assert message in result.output_parse_error
    assert result.stdout == stdout
    assert result.stderr == "runtime warning"
    assert len(result.a3_evidence_sha256) == len(result.attestation_sha256) == 64


def test_native_failure_diagnostics_are_retained_and_attested(tmp_path):
    def command(argv, **unused):
        return subprocess.CompletedProcess(
            argv, 4, "bisheng: compiling kernel.cpp\n", "linker: missing library\n"
        )

    result = A3KernelRunner(command).run(RunRequest(length=7), tmp_path)

    assert result.passed is False
    assert result.exit_code == 4
    assert result.stdout == "bisheng: compiling kernel.cpp\n"
    assert result.stderr == "linker: missing library\n"
    assert result.output_parse_error is None
    changed = A3KernelRunner(
        lambda argv, **unused: subprocess.CompletedProcess(argv, 4, "", "different")
    ).run(RunRequest(length=7), tmp_path)
    assert changed.a3_evidence_sha256 != result.a3_evidence_sha256
    assert changed.attestation_sha256 != result.attestation_sha256


@pytest.mark.parametrize("length", [0, 4097, True])
def test_request_rejects_invalid_lengths(length):
    with pytest.raises(ValueError, match="length"):
        RunRequest(length=length)


@pytest.mark.parametrize("seed", [True, False, 1.0, "1", None])
def test_request_rejects_non_integer_seed_before_identity(seed):
    with pytest.raises(ValueError, match="seed must be an integer"):
        RunRequest(seed=seed).request_id


def test_compile_uses_bisheng_and_fixed_a3_arch(monkeypatch, tmp_path):
    torch_root = tmp_path / "torch"
    npu_root = tmp_path / "torch_npu"
    torch_root.mkdir()
    npu_root.mkdir()
    fake_torch = SimpleNamespace(
        __file__=str(torch_root / "__init__.py"),
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
    )
    fake_npu = SimpleNamespace(__file__=str(npu_root / "__init__.py"))
    (tmp_path / "kernel.cpp").write_text("// staged")
    calls = []
    monkeypatch.setattr(host_driver.shutil, "which", lambda name: "/cann/bin/bisheng")

    def fake_run(argv, *, cwd, check):
        calls.append((argv, cwd, check))
        (tmp_path / "a3_kernel.so").touch()

    monkeypatch.setattr(host_driver.subprocess, "run", fake_run)
    output = host_driver._compile(
        tmp_path, dict(host_driver._EXPECTED_BUILD), fake_torch, fake_npu
    )

    assert output == tmp_path / "a3_kernel.so"
    argv, cwd, check = calls[0]
    assert argv[:7] == [
        "/cann/bin/bisheng", "-x", "asc", "--npu-arch=dav-2201",
        "-shared", "-fPIC", "-std=c++17",
    ]
    assert cwd == tmp_path and check is True
    assert not any(item in {"sh", "bash", "-c"} for item in argv)


def test_driver_uses_logical_device_zero_and_a3_namespace(monkeypatch, capsys, tmp_path):
    class Tensor:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    events = []
    torch = ModuleType("torch")
    torch.float32 = object()
    torch.tensor = lambda values, **kw: Tensor(values)
    torch.npu = SimpleNamespace(
        set_device=lambda device: events.append(("device", device)),
        synchronize=lambda: events.append("sync"),
    )
    torch.ops = SimpleNamespace(
        load_library=lambda path: events.append(("load", path)),
        rsi_a3kernels=SimpleNamespace(
            vector_add=lambda a, b: Tensor(
                [left + right for left, right in zip(a.values, b.values)]
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(host_driver, "_compile", lambda *args: tmp_path / "a3_kernel.so")
    (tmp_path / "build.json").write_text(json.dumps(host_driver._EXPECTED_BUILD))
    inputs = tmp_path / "input.json"
    inputs.write_text('{"input_a":[1.5,-2],"input_b":[2.25,1]}')
    monkeypatch.setattr(host_driver, "__file__", str(tmp_path / "host_driver.py"))
    monkeypatch.setattr(sys, "argv", ["host_driver.py", str(inputs)])

    assert host_driver.main() == 0

    assert events == [("load", str(tmp_path / "a3_kernel.so")), ("device", 0), "sync"]
    assert capsys.readouterr().out == "A3KERNEL_OUTPUT=[3.75,-1.0]\n"


def test_input_parser_rejects_untrusted_fields_and_bad_shapes():
    with pytest.raises(ValueError, match="only input_a"):
        host_driver._read_vectors(
            StringIO('{"input_a":[1],"input_b":[2],"command":"sh"}'), 4096
        )
    with pytest.raises(ValueError, match="equal lengths"):
        host_driver._read_vectors(StringIO('{"input_a":[1],"input_b":[2,3]}'), 4096)


def test_public_cli_validation_suite_covers_native_boundary_lengths(
    monkeypatch, capsys, tmp_path
):
    calls = []

    class FakeRunner:
        def run(self, request, workdir):
            calls.append((request.length, request.seed, workdir))
            return VerifiedResult(
                request_id=request.request_id,
                execution_id=f"execution-{request.length}",
                passed=True,
                max_abs_error=0.0,
                exit_code=0,
                source_fingerprint="source",
                output_sha256="output",
                a3_evidence_sha256="evidence",
                stdout="A3KERNEL_OUTPUT=[]\n",
                stderr="",
                output_parse_error=None,
                attestation_sha256="attestation",
            )

    monkeypatch.setattr(cli, "A3KernelRunner", FakeRunner)

    assert cli.main(["--validation-suite", "--seed", "9", "--workdir", str(tmp_path)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["validation_lengths"] == [1, 33, 4096]
    assert [item[:2] for item in calls] == [(1, 9), (33, 9), (4096, 9)]
    assert [item[2] for item in calls] == [
        tmp_path / "length-1",
        tmp_path / "length-33",
        tmp_path / "length-4096",
    ]
