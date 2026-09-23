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


_CATLASS_SOURCE = '''\
import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

@tla.kernel
def vector_add(a: tla.Tensor, b: tla.Tensor, out: tla.Tensor) -> None:
    n = a.origin_shape[0]
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=256):
            tid, _, _ = tla.arch.thread_idx()
            width, _, _ = tla.arch.thread_block_dim()
            for i in tla.range(tid, n, width):
                out[i] = a[i] + b[i]
        tla.pipe_barrier(tla.pipes.ALL)

def run(a, b):
    out = torch.empty_like(a)
    tensors = tuple(
        from_dlpack(item.contiguous(), layout_tag=tla.arch.RowMajor)
        for item in (a, b, out)
    )
    artifact = tla.compile(vector_add, *tensors, options="--npu-arch 3510")
    artifact(*tensors, block_num=1)
    torch.npu.synchronize()
    return out
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
        (SourceFile("kernel.py", _CATLASS_SOURCE),),
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
