"""FP8 movement fixture: one block, 256 logical elements."""
import catlass.tla as tla


@tla.kernel
def project_kernel(gm_input0: tla.Tensor, gm_output: tla.Tensor) -> None:
    ready = tla.flag("ready", tla.arch.MTE2, tla.arch.MTE3)
    source = tla.allocate(256, DTYPE, tla.AddressSpace.ub, 256)
    src = tla.make_tensor_like(source, gm_input0, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(src, gm_input0)
        tla.set_flag(ready)
        tla.wait_flag(ready)
        tla.copy(gm_output, src)
        tla.pipe_barrier(tla.pipes.ALL)
