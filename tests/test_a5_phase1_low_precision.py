from dataclasses import replace
import json
import math

import pytest

from benchmarks.a5kernels.phase1_curriculum import Capability, DType, ProjectSpec
from benchmarks.a5kernels.phase1_low_precision import mx_quantize, quantize_fp8
from benchmarks.a5kernels.phase1_project import (
    HostTensor, ProjectGraph, evaluate_reference, materialize_inputs,
)
from benchmarks.a5kernels.phase1_project_bz import _PROJECT_DRIVER
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from tests.test_a5_phase1_project_bz import FakeExecutor, _backend
from tests.test_a5_phase1_project_executor import _spec
from benchmarks.a5kernels.evidence import EvidenceLedger


@pytest.mark.parametrize("dtype,step,tiny", [
    ("float8_e4m3fn", 0.125, 2**-9), ("float8_e5m2", 0.25, 2**-16),
])
def test_fp8_nearest_even_subnormal_and_signed_zero(dtype, step, tiny):
    assert quantize_fp8(1 + step / 2, dtype) == 1
    assert quantize_fp8(1 + step * 1.5, dtype) == 1 + 2 * step
    assert quantize_fp8(tiny / 2, dtype) == 0
    assert quantize_fp8(tiny * 1.5, dtype) == tiny * 2
    assert math.copysign(1, quantize_fp8(-tiny / 2, dtype)) == -1


def test_fp8_format_overflow():
    assert quantize_fp8(464, "float8_e4m3fn") == 448
    assert math.isnan(quantize_fp8(465, "float8_e4m3fn"))
    assert math.isinf(quantize_fp8(61440, "float8_e5m2"))


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_matches_torch_cast_for_encoding_midpoints(dtype):
    torch = pytest.importorskip("torch")
    typ = getattr(torch, dtype)
    # All finite positive encodings and their exact midpoints exercise RNE.
    decoded = torch.arange(128, dtype=torch.uint8).view(typ).float()
    finite = decoded[torch.isfinite(decoded)]
    samples = torch.cat((finite, (finite[:-1] + finite[1:]) / 2))
    actual = samples.to(typ).float().tolist()
    for value, want in zip(samples.tolist(), actual):
        got = quantize_fp8(value, dtype)
        assert (math.isnan(got) and math.isnan(want)) or got == want


@pytest.mark.parametrize("dtype", ["mxfp8_e4m3fn", "mxfp8_e5m2"])
def test_mx_group32_scale_padding_and_b_transpose(dtype):
    # K=65 exercises three groups padded to four; scale differs in each group.
    source = [1.0] * 32 + [16.0] * 32 + [0.0]
    q, scale, dq = mx_quantize((1, 65), source, dtype, 0)
    assert scale["shape"] == [1, 4]
    assert scale["storage"][1] - scale["storage"][0] == 4
    assert scale["storage"][-2:] == [127, 127]
    assert dq == source
    assert q[:32] != source[:32]
    _, scale_b, dq_b = mx_quantize((65, 1), source, dtype, 1)
    assert scale_b == scale and dq_b == source


@pytest.mark.parametrize("dtype", [DType.FLOAT8_E4M3FN, DType.FLOAT8_E5M2,
                                   DType.MXFP8_E4M3FN, DType.MXFP8_E5M2])
def test_low_precision_cube_oracle_has_fp32_output_and_exact_scale_abi(tmp_path, dtype):
    spec = replace(_spec("cube"), dtype=dtype)
    inputs = materialize_inputs(spec)
    expected = evaluate_reference(spec, inputs)
    assert expected.dtype == "float32"
    project = ProjectExecutor(FakeExecutor([]), EvidenceLedger(tmp_path / "ledger"))
    prepared = project.prepare(spec)
    mx = dtype.value.startswith("mx")
    assert prepared.abi.input_arguments == (("gm_input0", "gm_input1", "gm_scale0", "gm_scale1")
                                             if mx else ("gm_input0", "gm_input1"))
    request = project.request(prepared, "# candidate")
    _, files = _backend(tmp_path, FakeExecutor([]))._bundle(request)
    manifest = json.loads(next(item.content for item in files if item.relative_path == "manifest.json"))
    assert manifest["output_dtype"] == "float32"
    assert len(manifest["scales"]) == (2 if mx else 0)
    if mx:
        assert manifest["scales"][1]["shape"] == [4, 2]
        assert manifest["inputs"][0]["storage"] != list(inputs[0].storage)
    compile(_PROJECT_DRIVER, "host_driver.py", "exec")
    assert ProjectSpec.from_mapping(spec.as_dict()) == spec


def test_cube_uses_quantized_operands_not_original_values():
    for dtype in (DType.FLOAT8_E4M3FN, DType.MXFP8_E4M3FN):
        spec = replace(_spec("cube"), dtype=dtype)
        a = HostTensor((2, 3), (2, 3), (1.1,) * 6)
        b = HostTensor((3, 4), (3, 4), (1.1,) * 12)
        out = evaluate_reference(spec, (a, b))
        assert out.values == (3 * 1.125**2,) * 8
        assert out.values != (3 * 1.1**2,) * 8


def test_mx_rejects_non_cube_graph():
    with pytest.raises(ValueError, match="low-precision Cube"):
        replace(_spec("movement"), dtype=DType.MXFP8_E4M3FN)


@pytest.mark.parametrize("dtype", [DType.FLOAT8_E4M3FN, DType.FLOAT8_E5M2])
def test_fp8_copy_output_is_rounded_storage_dtype(dtype):
    original = _spec("movement")
    graph = ProjectGraph(original.graph.inputs, original.graph.nodes[:1], "copy")
    spec = replace(original, dtype=dtype, graph=graph,
                   capabilities=tuple(c for c in original.capabilities if c is not Capability.SIMD_VECTOR))
    out = evaluate_reference(spec, materialize_inputs(spec))
    assert out.dtype == dtype.value
    assert all(value == quantize_fp8(value, dtype.value) for value in out.storage)


@pytest.mark.parametrize("dtype", [DType.FLOAT8_E4M3FN, DType.FLOAT8_E5M2])
def test_fp8_two_squares_widen_before_arithmetic_and_serialize(tmp_path, dtype):
    graph = ProjectGraph.from_mapping({
        "inputs": [{"name": "x", "shape": [1]}],
        "nodes": [
            {"name": "square", "op": "multiply", "operands": ["x", "x"], "axis": None},
            {"name": "out", "op": "multiply", "operands": ["square", "square"], "axis": None},
        ],
        "output": "out",
    })
    spec = replace(_spec("movement"), dtype=dtype, shape=(1,), graph=graph,
                   capabilities=(Capability.SIMD_VECTOR,))
    out = evaluate_reference(spec, (HostTensor((1,), (1,), (8.0,)),))
    assert out.dtype == "float32" and out.values == (4096.0,)
    project = ProjectExecutor(FakeExecutor([]), EvidenceLedger(tmp_path / "ledger"))
    prepared = project.prepare(spec)
    request = project.request(prepared, "# candidate")
    _, files = _backend(tmp_path, FakeExecutor([]))._bundle(request)
    manifest = json.loads(next(item.content for item in files if item.relative_path == "manifest.json"))
    assert manifest["output_dtype"] == "float32"
    assert '"elementwise_dtype":"float32"' in prepared.instruction
    assert all(math.isfinite(value) for value in prepared.expected.storage)


def test_mx_scale_entry_layout_is_dynamic_like_pinned_catlass_example():
    # Special MX GM tags cannot be static entry layouts. Match the pinned
    # MX example's dynamic descriptor ABI while retaining scale DMA tags.
    assert "tag = tla.arch.RowMajorMxScaleA if index == 0 else tla.arch.ColMajorMxScaleB" in _PROJECT_DRIVER
    assert "origin_shape=shape, element_type=tla.Float8E8M0).mark_layout_dynamic()" in _PROJECT_DRIVER
