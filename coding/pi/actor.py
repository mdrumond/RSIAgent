"""Pi action driver, retaining the existing noncoding terminal distillation."""

from benchmarks.a5kernels.phase1_provider import (
    DirectActorDriver,
)
from benchmarks.a5kernels.trial import actor_system_prompt
from .session import runtime_preflight



def task_contract(executor):
    """Expose layout and ABI metadata without oracle values or tensor storage."""
    prepared = getattr(executor, "prepared", None)
    if prepared is None:
        return None
    spec, abi = prepared.spec, prepared.abi
    input_names = abi.input_arguments[:len(prepared.inputs)]
    auxiliary = [{"argument": name, "kind": "auxiliary"}
                 for name in abi.input_arguments[len(prepared.inputs):]]
    if spec.dtype.value in {"mxfp8_e4m3fn", "mxfp8_e5m2"}:
        for index, item in enumerate(auxiliary):
            if index >= len(prepared.inputs):
                break
            rows, cols = prepared.inputs[index].shape
            k, outer = (cols, rows) if index == 0 else (rows, cols)
            groups = ((k + 31) // 32 + 1) // 2 * 2
            item.update(kind="mx-scale", dtype="E8M0", group_size=32,
                logical_shape=[outer, groups] if index == 0 else [groups, outer],
                physical_shape=[outer, groups],
                layout="RowMajorMxScaleA" if index == 0 else "ColMajorMxScaleB")
    return {
        "shape": list(spec.shape), "dtype": spec.dtype.value,
        "padding_multiple": spec.padding_multiple, "block_count": spec.block_count,
        "abi": {"kernel_name": abi.kernel_name, "input_arguments": list(abi.input_arguments),
                "output_argument": abi.output_argument},
        "graph": spec.graph.as_dict(),
        "inputs": [{"argument": name, "logical_shape": list(tensor.shape),
                    "physical_shape": list(tensor.physical_shape)}
                   for name, tensor in zip(input_names, prepared.inputs, strict=True)],
        "auxiliary_inputs": auxiliary,
        "output": {"argument": abi.output_argument, "logical_shape": list(prepared.expected.shape),
                   "physical_shape": list(prepared.expected.physical_shape),
                   "dtype": prepared.expected.dtype},
    }


class PiActorDriver(DirectActorDriver):
    """Public programming driver; canonical adoption owns native tool execution."""
    def __init__(self, *args, max_seconds=3600, system_prompt=None, **kwargs):
        runtime_preflight()
        kwargs.pop('max_turns', None)
        super().__init__(*args, max_seconds=max_seconds,
                         system_prompt=system_prompt or actor_system_prompt(typed=True), **kwargs)
