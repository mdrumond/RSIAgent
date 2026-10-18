"""Generic native candidates keep device results separate from host oracles."""

import json
import tarfile
import subprocess
import importlib.util
from importlib.resources import files
import sys
from types import SimpleNamespace

import pytest

from benchmarks.a3kernels.native_candidate import NativeCandidatePlan, NativeTensor
from benchmarks.a3kernels.remote_candidate_bz import BZA3RemoteCandidateBackend
from benchmarks.a3kernels.phase1_protocol import ExecutionReceipt, FailedEvidence


def plan(**overrides):
    values = dict(
        source='extern "C" __global__ __aicore__ void compute(GM_ADDR x, GM_ADDR y) {}',
        kernel_name="compute", inputs=(NativeTensor("x", (2, 3), "float16", tuple(range(6))),),
        output=NativeTensor("y", (2,), "float32"),
        request_id="r", attempt_id="a", project_id="reduce",
        execution_profile="bz-a3-1", target="discovered-a3", arch="discovered-arch",
    )
    return NativeCandidatePlan(**(values | overrides))


def backend(tmp_path):
    return BZA3RemoteCandidateBackend(
        cpl_remote="/tools/cpl-remote", validation_wrapper="/checked/catlass-validation.sh",
        profile="bz-a3-1", remote_workspace="/owned", physical_device=0,
        state_directory=tmp_path / "state",
    )


def test_generic_bundle_retains_shapes_types_source_and_explicit_toolchain(tmp_path):
    candidate = plan()
    bundle = backend(tmp_path).build_bundle(candidate, tmp_path / "bundle.tar")
    with tarfile.open(bundle.path) as archive:
        payload = json.load(archive.extractfile("input.json"))
        build = json.load(archive.extractfile("build.json"))
        wrapper = archive.extractfile("host_wrapper.inc").read().decode()
    assert payload["inputs"][0]["shape"] == [2, 3]
    assert payload["inputs"][0]["dtype"] == "float16"
    assert payload["output"]["shape"] == [2]
    assert build["arch"] == "discovered-arch"
    assert build["target"] == "discovered-a3"
    assert "::compute<<<" in wrapper
    assert "vector_add" not in wrapper


def test_identity_changes_with_input_dtype_shape_and_target():
    base = plan()
    assert base.execution_id != plan(target="other-a3").execution_id
    assert base.execution_id != plan(inputs=(NativeTensor("x", (6,), "bfloat16", tuple(range(6))),)).execution_id
    assert "GM_ADDR x, GM_ADDR y" in base.signature


@pytest.mark.parametrize("kwargs", [
    {"shape": (0,)}, {"dtype": "float8"}, {"values": (1.0,)},
    {"values": (float("nan"),) * 6},
])
def test_invalid_host_tensor_is_rejected(kwargs):
    with pytest.raises(ValueError):
        NativeTensor(**(dict(name="x", shape=(2, 3), dtype="float32", values=tuple(range(6))) | kwargs))


def test_compile_source_signature_is_declared_by_host():
    with pytest.raises(ValueError, match="signature"):
        plan(source='extern "C" __global__ __aicore__ void wrong(GM_ADDR x, GM_ADDR y) {}')


def test_generic_compile_is_compile_only_then_returns_ungraded_output(tmp_path):
    transport = backend(tmp_path)
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[0].endswith("cpl-remote"):
            text = "\n".join(("REMOTE_TARGET=bz-a3-1", "REMOTE_BACKEND=ssh", "REMOTE_OPERATION=upload", "REMOTE_STATE=completed", "REMOTE_EXIT=0"))
        else:
            compiling = "a3-candidate-" in " ".join(argv)
            body = "A3CANDIDATE_COMPILED=" + "a" * 64 if compiling else "A3KERNEL_OUTPUT=[123.0,456.0]"
            text = "\n".join((body, "CATLASS_VALIDATION_PROFILE=bz-a3-1", "CATLASS_VALIDATION_STATE=completed", "CATLASS_VALIDATION_HANDLE=bz-a3-1:test", "CATLASS_VALIDATION_EXIT=0"))
        return subprocess.CompletedProcess(argv, 0, text, "")

    transport._run = command
    compiled = transport.compile(plan(), tmp_path / "bundle")
    assert not isinstance(compiled, FailedEvidence)
    assert len(calls) == 2
    assert "python host_driver.py --compile-only" in " ".join(calls[-1])
    receipt = transport.execute_receipt(compiled)
    assert isinstance(receipt, ExecutionReceipt)
    assert receipt.output == (123.0, 456.0)  # Not an a+b or correctness claim.
    with pytest.raises(TypeError, match="host oracle"):
        transport.execute(compiled)


@pytest.mark.parametrize("dtype,cpp_type", [("float16", "kFloat16"), ("bfloat16", "kBFloat16"), ("float32", "kFloat32")])
def test_output_dtype_is_explicitly_owned_by_host(dtype, cpp_type):
    candidate = plan(output=NativeTensor("y", (2,), dtype))
    wrapper = next(f.content for f in candidate.files if f.relative_path == "host_wrapper.inc")
    assert f"torch::{cpp_type}" in wrapper


def test_compile_only_driver_never_initializes_device_or_launches(tmp_path, monkeypatch, capsys):
    driver_path = files("benchmarks.a3kernels.candidate_runtime").joinpath("native_driver.py")
    spec = importlib.util.spec_from_file_location("native_driver_under_test", driver_path)
    module = importlib.util.module_from_spec(spec)
    library = tmp_path / "a3_candidate.so"
    library.write_bytes(b"compiled-library")
    calls = []

    def compile_only(*args):
        calls.append(args)
        return library

    monkeypatch.setitem(sys.modules, "legacy_driver", SimpleNamespace(_compile=compile_only))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    spec.loader.exec_module(module)
    module.__file__ = str(tmp_path / "host_driver.py")
    (tmp_path / "build.json").write_text('{"arch":"discovered-arch"}')
    monkeypatch.setattr(sys, "argv", ["host_driver.py", "--compile-only"])
    module.main()
    assert len(calls) == 1
    assert "A3CANDIDATE_COMPILED=" in capsys.readouterr().out
