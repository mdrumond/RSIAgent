"""Single-tile plain FP8 fixture, (M,K,N)=(16,64,16), one block.

Derived from Catlass698379e examples/end_to_end/mx_mmad/mx_matmul.py.
Host binds row-major operands and FP32 output.
"""
import catlass.tla as tla


@tla.kernel
def project_kernel(gm_input0: tla.Tensor, gm_input1: tla.Tensor,
                   gm_output: tla.Tensor) -> None:
    pa = tla.allocate(16 * 64, DTYPE, tla.AddressSpace.l1, 512)
    pb = tla.allocate(64 * 16, DTYPE, tla.AddressSpace.l1, 512)
    pla = tla.allocate(16 * 64, DTYPE, tla.AddressSpace.l0a, 512)
    plb = tla.allocate(64 * 16, DTYPE, tla.AddressSpace.l0b, 512)
    plc = tla.allocate(16 * 16, tla.Float32, tla.AddressSpace.l0c, 512)
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.MTE1)
    ready = tla.flag("ready", tla.arch.MTE1, tla.arch.CUBE)
    with tla.cube():
        a = tla.make_tensor_like(pa, gm_input0, tla.arch.zN)
        b = tla.make_tensor_like(pb, gm_input1, tla.arch.zN)
        la = tla.make_tensor_like(pla, a, tla.arch.zN)
        lb = tla.make_tensor_like(plb, b, tla.arch.nZ)
        c = tla.make_tensor_like(plc, gm_output)
        tla.copy(a, gm_input0)
        tla.copy(b, gm_input1)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        tla.copy(la, a)
        tla.copy(lb, b)
        tla.set_flag(ready)
        tla.wait_flag(ready)
        tla.mmad(c, la, lb, init_c=True, unit_flag=0b11)
        tla.copy(gm_output, c, tla.params.CopyL0C2DstParams(unit_flag=0b11))
