"""Closed project grammar and host-owned reference semantics for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Mapping, Sequence, TYPE_CHECKING

from benchmarks.a5kernels.evidence import canonical_digest

if TYPE_CHECKING:
    from benchmarks.a5kernels.phase1_curriculum import ProjectSpec


_NAME = re.compile(r"[a-z][a-z0-9]{0,15}\Z")


class ProjectOp(str, Enum):
    COPY = "copy"
    ADD = "add"
    MULTIPLY = "multiply"
    RELU = "relu"
    MASK_POSITIVE = "mask-positive"
    REDUCE_SUM = "reduce-sum"
    MATMUL = "matmul"


def _shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 3:
        raise ValueError("tensor shape must have one to three dimensions")
    result = tuple(value)
    if any(type(item) is not int or not 1 <= item <= 4096 for item in result):
        raise ValueError("tensor shape dimensions must be integers in [1, 4096]")
    if math.prod(result) > 65536:
        raise ValueError("tensor shape may contain at most 65536 logical elements")
    return result


@dataclass(frozen=True)
class GraphInput:
    name: str
    shape: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "GraphInput":
        if not isinstance(value, Mapping) or set(value) != {"name", "shape"}:
            raise ValueError("graph input fields must be exactly name, shape")
        return cls(value["name"], _shape(value["shape"]))

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError("tensor names must be short lowercase identifiers")
        if self.shape != _shape(self.shape):
            raise ValueError("tensor shape must be canonical")

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "shape": list(self.shape)}


@dataclass(frozen=True)
class GraphNode:
    name: str
    op: ProjectOp
    operands: tuple[str, ...]
    axis: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "GraphNode":
        if not isinstance(value, Mapping) or set(value) != {
            "name", "op", "operands", "axis",
        }:
            raise ValueError("graph node fields must be exactly axis, name, op, operands")
        operands = value["operands"]
        if not isinstance(operands, (list, tuple)):
            raise ValueError("node operands must be a list")
        try:
            op = ProjectOp(value["op"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unregistered project operation") from exc
        return cls(value["name"], op, tuple(operands), value["axis"])

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError("tensor names must be short lowercase identifiers")
        if not isinstance(self.op, ProjectOp):
            raise ValueError("operation must be registered")
        if any(not isinstance(item, str) or not _NAME.fullmatch(item) for item in self.operands):
            raise ValueError("operands must name tensors")
        if self.axis is not None and type(self.axis) is not int:
            raise ValueError("axis must be an integer or null")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "op": self.op.value,
            "operands": list(self.operands), "axis": self.axis,
        }


def _infer(node: GraphNode, shapes: Mapping[str, tuple[int, ...]]) -> tuple[int, ...]:
    try:
        operands = tuple(shapes[name] for name in node.operands)
    except KeyError as exc:
        raise ValueError("operands must reference previously defined tensors") from exc
    unary = {ProjectOp.COPY, ProjectOp.RELU, ProjectOp.MASK_POSITIVE}
    if node.op in unary:
        if len(operands) != 1 or node.axis is not None:
            raise ValueError(f"{node.op.value} requires one operand and no axis")
        return operands[0]
    if node.op in {ProjectOp.ADD, ProjectOp.MULTIPLY}:
        if len(operands) != 2:
            raise ValueError(f"{node.op.value} requires two operands")
        if operands[0] != operands[1] or node.axis is not None:
            raise ValueError(f"{node.op.value} operands must have equal shapes and no axis")
        return operands[0]
    if node.op is ProjectOp.REDUCE_SUM:
        if len(operands) != 1 or node.axis is None or not 0 <= node.axis < len(operands[0]):
            raise ValueError("reduce-sum requires one operand and an in-range axis")
        return operands[0][:node.axis] + operands[0][node.axis + 1:] or (1,)
    if len(operands) != 2 or node.axis is not None:
        raise ValueError("matmul requires two operands and no axis")
    left, right = operands
    if len(left) != 2 or len(right) != 2 or left[1] != right[0]:
        raise ValueError("matmul requires compatible rank-two tensors")
    if left[0] * left[1] * right[1] > 2_000_000:
        raise ValueError("matmul host reference work exceeds the bound")
    return left[0], right[1]


@dataclass(frozen=True)
class ProjectGraph:
    inputs: tuple[GraphInput, ...]
    nodes: tuple[GraphNode, ...]
    output: str
    output_shape: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not 1 <= len(self.inputs) <= 3 or any(
            not isinstance(item, GraphInput) for item in self.inputs
        ):
            raise ValueError("graph must have one to three inputs")
        if not 1 <= len(self.nodes) <= 12 or any(
            not isinstance(item, GraphNode) for item in self.nodes
        ):
            raise ValueError("graph must have one to at most 12 nodes")
        names = [item.name for item in self.inputs]
        if len(set(names)) != len(names):
            raise ValueError("tensor names must be unique")
        shapes = {item.name: item.shape for item in self.inputs}
        for node in self.nodes:
            if node.name in shapes:
                raise ValueError("tensor names must be unique")
            shapes[node.name] = _infer(node, shapes)
        if self.output not in {node.name for node in self.nodes}:
            raise ValueError("graph output must name a computed node")
        object.__setattr__(self, "output_shape", shapes[self.output])

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ProjectGraph":
        if not isinstance(value, Mapping) or set(value) != {"inputs", "nodes", "output"}:
            raise ValueError("graph fields must be exactly inputs, nodes, output")
        raw_inputs, raw_nodes = value["inputs"], value["nodes"]
        if not isinstance(raw_inputs, list) or not 1 <= len(raw_inputs) <= 3:
            raise ValueError("graph must have one to three inputs")
        if not isinstance(raw_nodes, list) or not 1 <= len(raw_nodes) <= 12:
            raise ValueError("graph must have one to at most 12 nodes")
        inputs = tuple(GraphInput.from_mapping(item) for item in raw_inputs)
        nodes = tuple(GraphNode.from_mapping(item) for item in raw_nodes)
        return cls(inputs, nodes, value["output"])

    def as_dict(self) -> dict[str, object]:
        return {
            "inputs": [item.as_dict() for item in self.inputs],
            "nodes": [item.as_dict() for item in self.nodes],
            "output": self.output,
        }


def _coords(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    result = []
    for extent in reversed(shape):
        result.append(flat % extent)
        flat //= extent
    return tuple(reversed(result))


def _flat(coords: Sequence[int], shape: tuple[int, ...]) -> int:
    result = 0
    for coordinate, extent in zip(coords, shape, strict=True):
        result = result * extent + coordinate
    return result


@dataclass(frozen=True)
class HostTensor:
    shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    storage: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.physical_shape) or any(
            logical > physical
            for logical, physical in zip(self.shape, self.physical_shape, strict=True)
        ) or len(self.storage) != math.prod(self.physical_shape):
            raise ValueError("host tensor storage does not match its logical layout")

    @property
    def logical_values(self) -> tuple[float, ...]:
        return tuple(
            self.storage[_flat(_coords(index, self.shape), self.physical_shape)]
            for index in range(math.prod(self.shape))
        )


@dataclass(frozen=True)
class ReferenceOutput:
    shape: tuple[int, ...]
    values: tuple[float, ...]
    physical_shape: tuple[int, ...] | None = None
    storage: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.values) != math.prod(self.shape):
            raise ValueError("reference values do not match the logical shape")
        physical = self.shape if self.physical_shape is None else self.physical_shape
        storage = self.values if self.storage is None else self.storage
        if len(physical) != len(self.shape) or any(
            logical > extent
            for logical, extent in zip(self.shape, physical, strict=True)
        ) or len(storage) != math.prod(physical):
            raise ValueError("reference storage does not match its physical shape")
        object.__setattr__(self, "physical_shape", physical)
        object.__setattr__(self, "storage", storage)


def materialize_inputs(spec: "ProjectSpec") -> tuple[HostTensor, ...]:
    """Create reproducible, small integer inputs; padding is a visible sentinel."""

    digest = bytes.fromhex(canonical_digest(spec.as_dict()))
    padded = any(item.value == "padded-tail" for item in spec.capabilities)
    result = []
    for input_index, item in enumerate(spec.graph.inputs):
        physical = item.shape
        if padded:
            extent = math.ceil(item.shape[-1] / spec.padding_multiple) * spec.padding_multiple
            physical = item.shape[:-1] + (extent,)
        storage = [97.0] * math.prod(physical)
        for index in range(math.prod(item.shape)):
            value = float((digest[(index + input_index) % len(digest)] + index * 7) % 17 - 8)
            storage[_flat(_coords(index, item.shape), physical)] = value
        result.append(HostTensor(item.shape, physical, tuple(storage)))
    return tuple(result)


def _apply(node: GraphNode, operands: tuple[ReferenceOutput, ...]) -> ReferenceOutput:
    if node.op is ProjectOp.COPY:
        return operands[0]
    if node.op is ProjectOp.RELU:
        return ReferenceOutput(operands[0].shape, tuple(max(value, 0.0) for value in operands[0].values))
    if node.op is ProjectOp.MASK_POSITIVE:
        return ReferenceOutput(operands[0].shape, tuple(float(value > 0.0) for value in operands[0].values))
    if node.op in {ProjectOp.ADD, ProjectOp.MULTIPLY}:
        fn = (lambda a, b: a + b) if node.op is ProjectOp.ADD else (lambda a, b: a * b)
        return ReferenceOutput(operands[0].shape, tuple(fn(a, b) for a, b in zip(operands[0].values, operands[1].values, strict=True)))
    if node.op is ProjectOp.REDUCE_SUM:
        source = operands[0]
        shape = source.shape[:node.axis] + source.shape[node.axis + 1:] or (1,)
        values = [0.0] * math.prod(shape)
        for index, value in enumerate(source.values):
            coords = _coords(index, source.shape)
            target = coords[:node.axis] + coords[node.axis + 1:] or (0,)
            values[_flat(target, shape)] += value
        return ReferenceOutput(shape, tuple(values))
    left, right = operands
    m, k = left.shape
    _, n = right.shape
    values = tuple(
        sum(left.values[row * k + inner] * right.values[inner * n + col] for inner in range(k))
        for row in range(m) for col in range(n)
    )
    return ReferenceOutput((m, n), values)


def evaluate_reference(spec: "ProjectSpec", inputs: Sequence[HostTensor]) -> ReferenceOutput:
    if len(inputs) != len(spec.graph.inputs):
        raise ValueError("host inputs do not match the graph")
    values: dict[str, ReferenceOutput] = {}
    for contract, tensor in zip(spec.graph.inputs, inputs, strict=True):
        if tensor.shape != contract.shape:
            raise ValueError("host input shape does not match the graph")
        values[contract.name] = ReferenceOutput(tensor.shape, tensor.logical_values)
    for node in spec.graph.nodes:
        values[node.name] = _apply(node, tuple(values[name] for name in node.operands))
    output = values[spec.graph.output]
    if not any(item.value == "padded-tail" for item in spec.capabilities):
        return output
    extent = math.ceil(output.shape[-1] / spec.padding_multiple) * spec.padding_multiple
    physical = output.shape[:-1] + (extent,)
    storage = [0.0] * math.prod(physical)
    for index, value in enumerate(output.values):
        storage[_flat(_coords(index, output.shape), physical)] = value
    return ReferenceOutput(output.shape, output.values, physical, tuple(storage))


class EvidenceClassification(str, Enum):
    COMPILE_FAILURE = "compile-failure"
    RUNTIME_FAILURE = "runtime-failure"
    HOST_MISMATCH = "host-mismatch"
    HOST_VERIFIED = "host-verified"


@dataclass(frozen=True)
class ExecutionEvidence:
    classification: EvidenceClassification
    detail: str

    @property
    def verified(self) -> bool:
        return self.classification is EvidenceClassification.HOST_VERIFIED


def classify_execution(
    expected: ReferenceOutput,
    *,
    compile_succeeded: bool,
    runtime_error: str | None = None,
    actual: Sequence[float] | None = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> ExecutionEvidence:
    """Classify observed evidence; source or compilation alone cannot verify output."""

    if type(compile_succeeded) is not bool:
        raise ValueError("compile_succeeded must be boolean")
    if not compile_succeeded:
        return ExecutionEvidence(EvidenceClassification.COMPILE_FAILURE, "compilation failed")
    if runtime_error is not None:
        return ExecutionEvidence(EvidenceClassification.RUNTIME_FAILURE, runtime_error)
    if actual is None:
        return ExecutionEvidence(EvidenceClassification.RUNTIME_FAILURE, "no host output was observed")
    try:
        observed = tuple(float(value) for value in actual)
    except (TypeError, ValueError) as exc:
        raise ValueError("actual output must be numeric") from exc
    if len(observed) != len(expected.storage) or any(
        not math.isclose(got, want, abs_tol=atol, rel_tol=rtol)
        for got, want in zip(observed, expected.storage)
    ):
        return ExecutionEvidence(EvidenceClassification.HOST_MISMATCH, "host output differs from the oracle")
    return ExecutionEvidence(EvidenceClassification.HOST_VERIFIED, "host output matches the oracle")


__all__ = [
    "EvidenceClassification", "ExecutionEvidence", "GraphInput", "GraphNode",
    "HostTensor", "ProjectGraph", "ProjectOp", "ReferenceOutput",
    "classify_execution", "evaluate_reference", "materialize_inputs",
]
