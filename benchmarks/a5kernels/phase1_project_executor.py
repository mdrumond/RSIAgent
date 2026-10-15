"""Execute generic Phase-1 projects through host-authoritative backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Protocol

from benchmarks.ascend.contracts import CompileObservation, RuntimeObservation

from benchmarks.a5kernels.evidence import (
    EvidenceKind,
    EvidenceLedger,
    canonical_bytes,
    canonical_digest,
)
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_low_precision import MX_DTYPES, storage_dtype
from benchmarks.a5kernels.phase1_project import (
    ExecutionEvidence,
    HostTensor,
    ProjectOp,
    ReferenceOutput,
    classify_execution,
    evaluate_reference,
    materialize_inputs,
    project_output_dtype,
)


class ProjectExecutionFamily(str, Enum):
    """Backend techniques selected from the closed project graph."""

    MOVEMENT_ELEMENTWISE = "movement-elementwise-padded-tail"
    SIMT_REDUCTION_MASK = "simt-reduction-mask"
    CUBE_MATMUL = "cube-matmul"
    VECTOR_REDUCTION_MASK = "vector-reduction-mask"


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




@dataclass(frozen=True)
class ProjectExecutionRequest:
    """Candidate and public project material passed to the backend."""

    spec: ProjectSpec
    family: ProjectExecutionFamily
    abi: ProjectKernelABI
    instruction: str
    candidate_source: str
    inputs: tuple[HostTensor, ...]
    expected: ReferenceOutput
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


@dataclass(frozen=True)
class ProjectCompileResult:
    """Compile-only observation that carries no runtime correctness authority."""

    prepared: PreparedProject
    observation: CompileObservation
    artifact: Mapping[str, object]
    artifact_sha256: str


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
            "elementwise_dtype": project_output_dtype(spec),
            "accumulator_dtype": "float32",
            "output_dtype": project_output_dtype(spec),
            "input_storage_dtype": storage_dtype(spec.dtype.value),
            "mx_scale_group": 32 if spec.dtype.value in MX_DTYPES else None,
        },
    }
    return (
        f"Implement `@tla.kernel def {abi.signature}` as the complete Catlass DSL "
        "candidate for this generic learning "
        "project. Preserve logical shapes and padded physical storage. The host binds "
        "LOGICAL_INPUT_SHAPES, PHYSICAL_INPUT_SHAPES, LOGICAL_OUTPUT_SHAPE, "
        "PHYSICAL_OUTPUT_SHAPE, DTYPE (input storage), and OUTPUT_DTYPE as read-only globals. "
        "For MXFP8, gm_scale0 and gm_scale1 precede the output; scales are E8M0, "
        "one per32 K elements, with the group count padded even. A is RowMajor, "
        "B is ColumnMajor; scale tensors have logical shapes (M,G)/(G,N), "
        "RowMajorMxScaleA/ColMajorMxScaleB tags, and physical storage (M,G)/(N,G). "
        "Load scales with operands using tla.copy(..., scale=...) and tla.mmad_mx; "
        "Plain FP8 copy-only graphs preserve FP8 storage; graphs with arithmetic, "
        "masking, reduction, or matrix multiplication widen quantized inputs to "
        "Float32 for all intermediate computation and output. MXFP8 output is Float32. "
        "Conversion exercises must account for packed register lanes. "
        "Tensor arguments expose full physical padded storage: tensor.origin_shape "
        "is the PHYSICAL shape here, not the logical data extent. Use "
        "LOGICAL_INPUT_SHAPES and LOGICAL_OUTPUT_SHAPE for computation bounds; "
        "use PHYSICAL_INPUT_SHAPES and PHYSICAL_OUTPUT_SHAPE for allocation and "
        "storage bounds. Input padding contains nonzero sentinels, not data or "
        "implicit zeros. Every physical output element must be written; output "
        "padding must be zero. For logical length 65 padded to 128, compute only "
        "elements 0..64 and explicitly zero elements 65..127. The host "
        "owns compilation, launch, synchronization, and the host-owned oracle; "
        "do not claim correctness from source or compilation. Project contract: "
        f"{canonical_bytes(contract).decode('ascii')}"
    )


class ProjectTargetBinding(Protocol):
    """Language-specific ABI and instructions, independent of host verification.

    Binding implementations must describe the supplied graph, not change its
    generated inputs or numerical oracle. Existing callers default to Catlass.
    """

    def abi(
        self, spec: ProjectSpec, inputs: tuple[HostTensor, ...],
    ) -> ProjectKernelABI: ...

    def instruction(
        self, spec: ProjectSpec, family: ProjectExecutionFamily,
        abi: ProjectKernelABI, inputs_sha256: str,
    ) -> str: ...


class CatlassProjectBinding:
    """Compatibility binding preserving the established A5 public contract."""

    def abi(
        self, spec: ProjectSpec, inputs: tuple[HostTensor, ...],
    ) -> ProjectKernelABI:
        return ProjectKernelABI(
            "project_kernel",
            tuple(f"gm_input{index}" for index in range(len(inputs)))
            + (("gm_scale0", "gm_scale1") if spec.dtype.value in MX_DTYPES else ()),
            "gm_output",
        )

    @staticmethod
    def instruction(
        spec: ProjectSpec, family: ProjectExecutionFamily,
        abi: ProjectKernelABI, inputs_sha256: str,
    ) -> str:
        return _instruction(spec, family, abi, inputs_sha256)


class ProjectExecutor:
    """Prepare and execute a project without inspecting candidate semantics."""

    def __init__(
        self, backend: ProjectExecutionBackend, ledger: EvidenceLedger,
        *, binding: ProjectTargetBinding | None = None,
    ):
        self._backend = backend
        self._ledger = ledger
        self._binding = binding if binding is not None else CatlassProjectBinding()

    def prepare(self, spec: ProjectSpec) -> PreparedProject:
        if not isinstance(spec, ProjectSpec):
            raise TypeError("project must be a validated ProjectSpec")
        inputs = materialize_inputs(spec)
        expected = evaluate_reference(spec, inputs)
        family_selector = getattr(self._binding, "family", _family)
        family = family_selector(spec)
        abi = self._binding.abi(spec, inputs)
        inputs_sha256 = canonical_digest({
            "tensors": [_tensor_payload(tensor) for tensor in inputs]
        })
        return PreparedProject(
            spec,
            family,
            abi,
            self._binding.instruction(spec, family, abi, inputs_sha256),
            inputs,
            expected,
        )

    def compile(self, spec: ProjectSpec, candidate_source: str) -> ProjectCompileResult:
        """Compile a candidate without launching it or classifying correctness."""

        if not isinstance(candidate_source, str) or not candidate_source.strip():
            raise ValueError("candidate source must be non-empty")
        prepared = self.prepare(spec)
        compilation = self._backend.compile(self._request(prepared, candidate_source))
        if not isinstance(compilation, CompileObservation):
            raise TypeError("backend compile must return CompileObservation")
        artifact = {
            "compile_passed": compilation.passed,
            "compile_diagnostic": compilation.diagnostic,
            "compile_evidence_sha256": compilation.evidence_sha256,
            "compiled_artifact_id": compilation.artifact_id,
        }
        digest = canonical_digest(artifact)
        self._ledger.append(EvidenceKind.ARTIFACT, {
            "event": "phase1-project-compile-artifact",
            "project_id": spec.project_id,
            "artifact_kind": "compile",
            "artifact_sha256": digest,
            "payload": artifact,
        })
        return ProjectCompileResult(prepared, compilation, artifact, digest)

    def execute(self, spec: ProjectSpec, candidate_source: str) -> ProjectExecutionResult:
        if not isinstance(candidate_source, str) or not candidate_source.strip():
            raise ValueError("candidate source must be non-empty")
        prepared = self.prepare(spec)
        inputs_payload = {"tensors": [_tensor_payload(tensor) for tensor in prepared.inputs]}
        expected_payload = _expected_payload(prepared.expected)
        request = self._request(prepared, candidate_source)
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
    def request(
        prepared: PreparedProject, candidate_source: str,
    ) -> ProjectExecutionRequest:
        """Bind a candidate to already prepared host inputs and oracle."""
        return ProjectExecutor._request(prepared, candidate_source)

    @staticmethod
    def _request(
        prepared: PreparedProject, candidate_source: str
    ) -> ProjectExecutionRequest:
        inputs_payload = {"tensors": [_tensor_payload(item) for item in prepared.inputs]}
        return ProjectExecutionRequest(
            spec=prepared.spec,
            family=prepared.family,
            abi=prepared.abi,
            instruction=prepared.instruction,
            candidate_source=candidate_source,
            inputs=prepared.inputs,
            expected=prepared.expected,
            inputs_sha256=canonical_digest(inputs_payload),
        )

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
    "CatlassProjectBinding",
    "CompileObservation",
    "PreparedProject",
    "ProjectCompileResult",
    "ProjectExecutionBackend",
    "ProjectExecutionFamily",
    "ProjectKernelABI",
    "ProjectExecutionRequest",
    "ProjectExecutionResult",
    "ProjectExecutor",
    "ProjectTargetBinding",
    "RuntimeObservation",
]
