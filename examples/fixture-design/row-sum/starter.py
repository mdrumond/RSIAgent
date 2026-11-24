import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


@tla.kernel
def row_sum(values: tla.Tensor, out: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=32):
            lane, _, _ = tla.arch.thread_idx()
            if lane < 3:
                acc = tla.Float32(0.0)
                for column in tla.range(0, 7, 1):
                    acc = acc + values[lane * 7 + column]
                out[lane] = acc


def run(values, out):
    inputs = [from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in (values, out)]
    fn = tla.compile(row_sum, *inputs, options="--npu-arch 3510")
    fn(*inputs, block_num=1)
    torch.npu.synchronize()
    return out
