"""Execute generic Phase-1 projects through host-authoritative backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Mapping, Protocol

from benchmarks.a5kernels.evidence import (
    EvidenceKind,
    EvidenceLedger,
    canonical_bytes,
    canonical_digest,
)
from benchmarks.a5kernels.phase1_curriculum import FailureMode, ProjectSpec
from benchmarks.a5kernels.phase1_project import (
    ExecutionEvidence,
    HostTensor,
    ProjectOp,
    ReferenceOutput,
    classify_execution,
    evaluate_reference,
    materialize_inputs,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ProjectExecutionFamily(str, Enum):
    """Backend techniques selected from the closed project graph."""

    MOVEMENT_ELEMENTWISE = "movement-elementwise-padded-tail"
    SIMT_REDUCTION_MASK = "simt-reduction-mask"
    CUBE_MATMUL = "cube-matmul"


@dataclass(frozen=True)
class ProjectKernelABI:
    kernel_name: str
    input_arguments: tuple[str, ...]
    output_argument: str

    @property
    def signature(self) -> str:
        arguments = self.input_arguments + (self.output_argument,)
        typed = ", ".join(f"{name}: tla.Tensor" for name in arguments)
        return f"{self.kernel_name}({typed}) -> None"

    def as_dict(self) -> dict[str, object]:
        return {
            "kernel_name": self.kernel_name,
            "input_arguments": list(self.input_arguments),
            "output_argument": self.output_argument,
            "signature": self.signature,
        }


def _require_digest(value: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("evidence_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CompileObservation:
    """An actual compiler observation supplied by the execution backend."""

    passed: bool
    evidence_sha256: str
    diagnostic: str
    artifact_id: str | None

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise ValueError("compile passed must be boolean")
        _require_digest(self.evidence_sha256)
        if not isinstance(self.diagnostic, str) or not self.diagnostic.strip():
            raise ValueError("compile diagnostic must be non-empty")
        if self.passed and (
            not isinstance(self.artifact_id, str) or not self.artifact_id.strip()
        ):
            raise ValueError("artifact_id is required after successful compilation")
        if not self.passed and self.artifact_id is not None:
            raise ValueError("artifact_id is invalid after failed compilation")


@dataclass(frozen=True)
class RuntimeObservation:
    """A synchronized runtime observation, including the complete host output."""

    passed: bool
    evidence_sha256: str
    diagnostic: str
    output: tuple[float, ...] | None

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise ValueError("runtime passed must be boolean")
        _require_digest(self.evidence_sha256)
        if not isinstance(self.diagnostic, str) or not self.diagnostic.strip():
            raise ValueError("runtime diagnostic must be non-empty")
        if self.passed != (self.output is not None):
            raise ValueError("output is required exactly when runtime passed")
        if self.output is not None and (
            not isinstance(self.output, tuple)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                for value in self.output
            )
        ):
            raise ValueError("runtime output must be a numeric tuple")


@dataclass(frozen=True)
class ProjectExecutionRequest:
    """Candidate and public project material passed to the backend."""

    spec: ProjectSpec
    family: ProjectExecutionFamily
    abi: ProjectKernelABI
    instruction: str
    candidate_source: str
    inputs: tuple[HostTensor, ...]
    inputs_sha256: str


class ProjectExecutionBackend(Protocol):
    """Explicit compiler/runtime seam implemented by a checked A5 adapter."""

    def compile(self, request: ProjectExecutionRequest) -> CompileObservation: ...

    def run(
        self, request: ProjectExecutionRequest, compilation: CompileObservation
    ) -> RuntimeObservation: ...


@dataclass(frozen=True)
class PreparedProject:
    spec: ProjectSpec
    family: ProjectExecutionFamily
    abi: ProjectKernelABI
    instruction: str
    inputs: tuple[HostTensor, ...]
    expected: ReferenceOutput


@dataclass(frozen=True)
class ProjectExecutionResult:
    prepared: PreparedProject
    evidence: ExecutionEvidence
    artifacts: Mapping[str, Mapping[str, object]]
    artifact_sha256: Mapping[str, str]


def _family(spec: ProjectSpec) -> ProjectExecutionFamily:
    operations = {node.op for node in spec.graph.nodes}
    if ProjectOp.MATMUL in operations:
        return ProjectExecutionFamily.CUBE_MATMUL
    if operations & {ProjectOp.REDUCE_SUM, ProjectOp.MASK_POSITIVE}:
        return ProjectExecutionFamily.SIMT_REDUCTION_MASK
    return ProjectExecutionFamily.MOVEMENT_ELEMENTWISE


def _tensor_payload(tensor: HostTensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "physical_shape": list(tensor.physical_shape),
        "storage": list(tensor.storage),
    }


def _evidence_number(value: float) -> float | str:
    """Encode IEEE-754 specials without weakening canonical JSON evidence."""

    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Infinity" if value > 0 else "-Infinity"
    return value


def _expected_payload(expected: ReferenceOutput) -> dict[str, object]:
    return {
        "dtype": expected.dtype,
        "shape": list(expected.shape),
        "physical_shape": list(expected.physical_shape),
        "storage": list(expected.storage),
    }


def _instruction(
    spec: ProjectSpec,
    family: ProjectExecutionFamily,
    abi: ProjectKernelABI,
    inputs_sha256: str,
) -> str:
    contract = {
        "project": spec.as_dict(),
        "execution_family": family.value,
        "kernel_abi": abi.as_dict(),
        "inputs_sha256": inputs_sha256,
        "output_shape": list(spec.graph.output_shape),
        "reference_semantics": {
            "elementwise_dtype": spec.dtype.value,
            "accumulator_dtype": "float32",
            "output_dtype": spec.dtype.value,
        },
    }
    return (
        f"Implement `@tla.kernel def {abi.signature}` as the complete Catlass DSL "
        "candidate for this generic learning "
        "project. Preserve logical shapes and padded physical storage. The host "
        "owns compilation, launch, synchronization, and the host-owned oracle; "
        "do not claim correctness from source or compilation. Project contract: "
        f"{canonical_bytes(contract).decode('ascii')}"
    )


class ProjectExecutor:
    """Prepare and execute a project without inspecting candidate semantics."""

    def __init__(self, backend: ProjectExecutionBackend, ledger: EvidenceLedger):
        self._backend = backend
        self._ledger = ledger

    def prepare(self, spec: ProjectSpec) -> PreparedProject:
        if not isinstance(spec, ProjectSpec):
            raise TypeError("project must be a validated ProjectSpec")
        if spec.failure_mode is not FailureMode.NONE:
            raise NotImplementedError(
                "failure-repair projects require a repair-loop coordinator"
            )
        inputs = materialize_inputs(spec)
        expected = evaluate_reference(spec, inputs)
        family = _family(spec)
        abi = ProjectKernelABI(
            "project_kernel",
            tuple(f"gm_input{index}" for index in range(len(inputs))),
            "gm_output",
        )
        inputs_sha256 = canonical_digest({
            "tensors": [_tensor_payload(tensor) for tensor in inputs]
        })
        return PreparedProject(
            spec,
            family,
            abi,
            _instruction(spec, family, abi, inputs_sha256),
            inputs,
            expected,
        )

    def execute(self, spec: ProjectSpec, candidate_source: str) -> ProjectExecutionResult:
        if not isinstance(candidate_source, str) or not candidate_source.strip():
            raise ValueError("candidate source must be non-empty")
        prepared = self.prepare(spec)
        inputs_payload = {
            "tensors": [_tensor_payload(tensor) for tensor in prepared.inputs]
        }
        expected_payload = _expected_payload(prepared.expected)
        request = ProjectExecutionRequest(
            spec=spec,
            family=prepared.family,
            abi=prepared.abi,
            instruction=prepared.instruction,
            candidate_source=candidate_source,
            inputs=prepared.inputs,
            inputs_sha256=canonical_digest(inputs_payload),
        )
        compilation = self._backend.compile(request)
        if not isinstance(compilation, CompileObservation):
            raise TypeError("backend compile must return CompileObservation")
        runtime = None
        if compilation.passed:
            runtime = self._backend.run(request, compilation)
            if not isinstance(runtime, RuntimeObservation):
                raise TypeError("backend run must return RuntimeObservation")
        evidence = self._classify(prepared.expected, compilation, runtime)
        output = None
        if runtime is not None and runtime.output is not None:
            output = [_evidence_number(value) for value in runtime.output]
        observed_payload = {
            "compile_passed": compilation.passed,
            "compile_diagnostic": compilation.diagnostic,
            "compile_evidence_sha256": compilation.evidence_sha256,
            "compiled_artifact_id": compilation.artifact_id,
            "runtime_passed": None if runtime is None else runtime.passed,
            "runtime_diagnostic": None if runtime is None else runtime.diagnostic,
            "runtime_evidence_sha256": (
                None if runtime is None else runtime.evidence_sha256
            ),
            "output_encoding": "json-number-or-ieee754-symbol",
            "output": output,
        }
        artifacts: dict[str, Mapping[str, object]] = {
            "inputs": inputs_payload,
            "expected": expected_payload,
            "observed": observed_payload,
        }
        hashes = {name: canonical_digest(payload) for name, payload in artifacts.items()}
        artifacts["classification"] = {
            "classification": evidence.classification.value,
            "detail": evidence.detail,
            "inputs_sha256": hashes["inputs"],
            "expected_sha256": hashes["expected"],
            "observed_sha256": hashes["observed"],
        }
        hashes["classification"] = canonical_digest(artifacts["classification"])
        for name, payload in artifacts.items():
            self._ledger.append(EvidenceKind.ARTIFACT, {
                "event": "phase1-project-artifact",
                "project_id": spec.project_id,
                "artifact_kind": name,
                "artifact_sha256": hashes[name],
                "payload": payload,
            })
        return ProjectExecutionResult(prepared, evidence, artifacts, hashes)

    @staticmethod
    def _classify(
        expected: ReferenceOutput,
        compilation: CompileObservation,
        runtime: RuntimeObservation | None,
    ) -> ExecutionEvidence:
        if not compilation.passed:
            result = classify_execution(expected, compile_succeeded=False)
            return ExecutionEvidence(result.classification, compilation.diagnostic)
        if runtime is None:
            raise RuntimeError("successful compilation omitted a runtime observation")
        if not runtime.passed:
            return classify_execution(
                expected,
                compile_succeeded=True,
                runtime_error=runtime.diagnostic,
            )
        return classify_execution(
            expected,
            compile_succeeded=True,
            actual=runtime.output,
        )


__all__ = [
    "CompileObservation",
    "PreparedProject",
    "ProjectExecutionBackend",
    "ProjectExecutionFamily",
    "ProjectKernelABI",
    "ProjectExecutionRequest",
    "ProjectExecutionResult",
    "ProjectExecutor",
    "RuntimeObservation",
]
