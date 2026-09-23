"""Host-owned hello/vector-add fixtures for supported A5 languages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from benchmarks.a5kernels.protocol import SourceFile


class Language(str, Enum):
    CATLASS_DSL = "catlass-dsl"
    ASCEND_C = "ascend-c"
    TRITON_ASCEND = "triton-ascend"


@dataclass(frozen=True)
class Fixture:
    language: Language
    files: tuple[SourceFile, ...]
    # Concrete runtime drivers are supplied by language-specific integrations.
    # A missing argv is an explicit, fail-closed support boundary.
    argv: tuple[str, ...] | None = None
    max_length: int | None = None


_CATLASS_SOURCE = '''\
import catlass.tla as tla

VECTOR_ELE = 400
VL_ELE = 64

@tla.kernel
def vector_add(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor) -> None:
    n_ele = gm_a.origin_shape[0]
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    ptr_a = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ptr_b = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ptr_c = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_a = tla.make_tensor_like(ptr_a, gm_a, tla.arch.RowMajor)
    ub_b = tla.make_tensor_like(ptr_b, gm_b, tla.arch.RowMajor)
    ub_c = tla.make_tensor_like(ptr_c, gm_c, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(ub_a, gm_a)
        tla.copy(ub_b, gm_b)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range((n_ele + VL_ELE - 1) // VL_ELE):
                tile_a = tla.tile_view(ub_a, tla.make_shape(VL_ELE), tla.make_coord(i))
                tile_b = tla.tile_view(ub_b, tla.make_shape(VL_ELE), tla.make_coord(i))
                tile_c = tla.tile_view(ub_c, tla.make_shape(VL_ELE), tla.make_coord(i))
                tile_c.store(tla.add(tile_a.load(), tile_b.load()))
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(gm_c, ub_c)
        tla.pipe_barrier(tla.pipes.ALL)


def run(input_a, input_b):
    import torch
    import torch_npu
    from catlass.tla.runtime import from_dlpack

    if not 0 < len(input_a) <= VECTOR_ELE or len(input_a) != len(input_b):
        raise ValueError(f"vector length must be in [1, {VECTOR_ELE}]")
    torch.npu.set_device(0)
    a = torch.tensor(input_a, dtype=torch.float32, device="npu")
    b = torch.tensor(input_b, dtype=torch.float32, device="npu")
    out = torch.empty_like(a)

    def as_tla(tensor):
        return from_dlpack(
            tensor.contiguous(), layout_tag=tla.arch.RowMajor
        ).mark_compact_shape_dynamic(0)

    tla_a, tla_b, tla_out = (as_tla(item) for item in (a, b, out))
    artifact = tla.compile(
        vector_add, tla_a, tla_b, tla_out, options="--npu-arch 3510"
    )
    artifact(tla_a, tla_b, tla_out, block_num=1)
    torch.npu.synchronize()
    return out.cpu().tolist()
'''

_CATLASS_DRIVER = '''\
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

OUTPUT_MARKER = "A5KERNEL_OUTPUT="


def _load_kernel(path: str):
    source = Path(path)
    spec = importlib.util.spec_from_file_location("a5kernel_fixture", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load kernel source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _numbers(value, name: str):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON list")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{name} must contain only numbers")
    return [float(item) for item in value]


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: host_driver.py KERNEL.py INPUT.json")
    with Path(sys.argv[2]).open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict) or set(payload) != {"input_a", "input_b"}:
        raise ValueError("input must contain exactly input_a and input_b")
    input_a = _numbers(payload["input_a"], "input_a")
    input_b = _numbers(payload["input_b"], "input_b")
    if len(input_a) != len(input_b):
        raise ValueError("input vectors must have equal length")
    output = _numbers(_load_kernel(sys.argv[1]).run(input_a, input_b), "output")
    print(OUTPUT_MARKER + json.dumps(output, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_ASCEND_C_SOURCE = '''\
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void vector_add(
    GM_ADDR a, GM_ADDR b, GM_ADDR out, uint32_t n) {
  GlobalTensor<float> ga, gb, gout;
  ga.SetGlobalBuffer((__gm__ float*)a, n);
  gb.SetGlobalBuffer((__gm__ float*)b, n);
  gout.SetGlobalBuffer((__gm__ float*)out, n);
  for (uint32_t i = GetBlockIdx(); i < n; i += GetBlockNum()) {
    gout.SetValue(i, ga.GetValue(i) + gb.GetValue(i));
  }
}
'''

_TRITON_SOURCE = '''\
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add(a, b, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, tl.load(a + offsets, mask=mask) + tl.load(b + offsets, mask=mask), mask=mask)

def run(a, b):
    out = torch.empty_like(a)
    vector_add[(triton.cdiv(a.numel(), 256),)](a, b, out, a.numel(), BLOCK=256)
    return out
'''


_FIXTURES = {
    Language.CATLASS_DSL: Fixture(
        Language.CATLASS_DSL,
        (
            SourceFile("kernel.py", _CATLASS_SOURCE),
            SourceFile("host_driver.py", _CATLASS_DRIVER),
        ),
        ("python", "host_driver.py", "kernel.py", "input.json"),
        400,
    ),
    Language.ASCEND_C: Fixture(
        Language.ASCEND_C,
        (SourceFile("kernel.cpp", _ASCEND_C_SOURCE),),
    ),
    Language.TRITON_ASCEND: Fixture(
        Language.TRITON_ASCEND,
        (SourceFile("kernel.py", _TRITON_SOURCE),),
    ),
}


def fixture_for(language: Language | str) -> Fixture:
    try:
        return _FIXTURES[Language(language)]
    except ValueError as exc:
        supported = ", ".join(item.value for item in Language)
        raise ValueError(f"unsupported language {language!r}; choose {supported}") from exc
