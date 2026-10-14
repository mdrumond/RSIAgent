from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.a5kernels.evidence import EvidenceLedger, canonical_digest
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_project import EvidenceClassification, ProjectGraph
from benchmarks.a5kernels.phase1_project_executor import (
    CompileObservation,
    ProjectExecutionFamily,
    ProjectExecutor,
    RuntimeObservation,
)


def _spec(family: str) -> ProjectSpec:
    if family == "movement":
        graph = ProjectGraph.from_mapping({
            "inputs": [{"name": "x", "shape": [2, 5]}],
            "nodes": [
                {"name": "copy", "op": "copy", "operands": ["x"], "axis": None},
                {"name": "out", "op": "relu", "operands": ["copy"], "axis": None},
            ],
            "output": "out",
        })
        capabilities = ["gm-ub-movement", "simd-vector", "padded-tail"]
        shape, padding = [2, 5], 16
    elif family == "simt":
        graph = ProjectGraph.from_mapping({
            "inputs": [{"name": "x", "shape": [2, 5]}],
            "nodes": [
                {"name": "mask", "op": "mask-positive", "operands": ["x"], "axis": None},
                {"name": "out", "op": "reduce-sum", "operands": ["mask"], "axis": 1},
            ],
            "output": "out",
        })
        capabilities = ["simt-mask", "simt-reduction"]
        shape, padding = [2, 5], 1
    else:
        graph = ProjectGraph.from_mapping({
            "inputs": [
                {"name": "lhs", "shape": [2, 3]},
                {"name": "rhs", "shape": [3, 4]},
            ],
            "nodes": [
                {"name": "out", "op": "matmul", "operands": ["lhs", "rhs"], "axis": None},
            ],
            "output": "out",
        })
        capabilities = ["cube-tiling"]
        shape, padding = [2, 3, 4], 1
    return ProjectSpec.from_mapping({
        "project_id": f"learn-{family}",
        "capabilities": capabilities,
        "shape": shape,
        "dtype": "float32",
        "block_count": 2,
        "padding_multiple": padding,
        "failure_mode": "none",
        "timing_lengths": [],
        "profile_metric": None,
        "hypothesis": "Host observations decide what was learned.",
        "graph": graph.as_dict(),
    })


class FakeBackend:
    def __init__(self, compile_result, runtime_result=None):
        self.compile_result = compile_result
        self.runtime_result = runtime_result
        self.compile_requests = []
        self.run_requests = []

    def compile(self, request):
        self.compile_requests.append(request)
        return self.compile_result

    def run(self, request, compilation):
        self.run_requests.append((request, compilation))
        return self.runtime_result


def _compile(*, passed=True):
    return CompileObservation(
        passed=passed,
        evidence_sha256=canonical_digest({"actual-compiler-report": passed}),
        diagnostic="compiled" if passed else "frontend rejected candidate",
        artifact_id="compiled-object" if passed else None,
    )


def _runtime(values=None, *, passed=True):
    return RuntimeObservation(
        passed=passed,
        evidence_sha256=canonical_digest({"actual-runtime-report": passed}),
        diagnostic="launch synchronized" if passed else "device launch failed",
        output=None if not passed else tuple(values),
    )


@pytest.mark.parametrize(
    "name,family",
    [
        ("movement", ProjectExecutionFamily.MOVEMENT_ELEMENTWISE),
        ("simt", ProjectExecutionFamily.SIMT_REDUCTION_MASK),
        ("cube", ProjectExecutionFamily.CUBE_MATMUL),
    ],
)
def test_executor_materializes_each_generic_family_without_benchmark_adapter(
    tmp_path, name, family,
):
    spec = _spec(name)
    backend = FakeBackend(_compile(), _runtime([]))
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "evidence.jsonl"))
    prepared = executor.prepare(spec)

    assert prepared.family is family
    assert prepared.instruction == executor.prepare(spec).instruction
    assert prepared.abi.kernel_name == "project_kernel"
    assert prepared.abi.input_arguments == tuple(
        f"gm_input{index}" for index in range(len(spec.graph.inputs))
    )
    assert prepared.abi.signature in prepared.instruction
    assert spec.project_id in prepared.instruction
    assert "host-owned oracle" in prepared.instruction
    assert "expected_values" not in prepared.instruction
    assert "bsa" not in prepared.instruction.lower()
    assert "gdm" not in prepared.instruction.lower()
    assert "benchmark" not in prepared.instruction.lower()
    assert len(prepared.inputs) == len(spec.graph.inputs)
    assert prepared.expected.shape == spec.graph.output_shape


def test_compile_failure_is_a_real_backend_result_and_never_launches(tmp_path):
    backend = FakeBackend(_compile(passed=False))
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")
    result = ProjectExecutor(backend, ledger).execute(
        _spec("movement"), "source that is forwarded without source heuristics"
    )

    assert result.evidence.classification is EvidenceClassification.COMPILE_FAILURE
    assert result.evidence.verified is False
    assert backend.compile_requests[0].candidate_source == (
        "source that is forwarded without source heuristics"
    )
    assert backend.compile_requests[0].abi.signature.endswith("gm_output: tla.Tensor) -> None")
    assert backend.run_requests == []
    assert set(result.artifact_sha256) == {
        "inputs", "expected", "observed", "classification",
    }
    assert result.artifact_sha256["classification"] == canonical_digest(
        result.artifacts["classification"]
    )
    assert len(ledger.entries) == 4
    for entry in ledger.entries:
        assert entry.payload["artifact_sha256"] == canonical_digest(
            entry.payload["payload"]
        )


def test_runtime_failure_requires_an_actual_runtime_observation(tmp_path):
    backend = FakeBackend(_compile(), _runtime(passed=False))
    result = ProjectExecutor(
        backend, EvidenceLedger(tmp_path / "evidence.jsonl")
    ).execute(_spec("simt"), "candidate")

    assert result.evidence.classification is EvidenceClassification.RUNTIME_FAILURE
    assert "device launch failed" in result.evidence.detail
    assert result.artifacts["observed"]["output"] is None
    assert result.artifacts["observed"]["runtime_evidence_sha256"]
    assert len(backend.run_requests) == 1


def test_host_mismatch_and_verified_results_retain_exact_observations(tmp_path):
    spec = _spec("cube")
    probe = ProjectExecutor(
        FakeBackend(_compile()), EvidenceLedger(tmp_path / "probe.jsonl")
    ).prepare(spec)
    wrong = tuple(0.0 for _ in probe.expected.storage)
    if wrong == probe.expected.storage:
        wrong = (1.0,) + wrong[1:]
    mismatch = ProjectExecutor(
        FakeBackend(_compile(), _runtime(wrong)),
        EvidenceLedger(tmp_path / "mismatch.jsonl"),
    ).execute(spec, "cube candidate")
    verified = ProjectExecutor(
        FakeBackend(_compile(), _runtime(probe.expected.storage)),
        EvidenceLedger(tmp_path / "verified.jsonl"),
    ).execute(spec, "cube candidate")

    assert mismatch.evidence.classification is EvidenceClassification.HOST_MISMATCH
    assert mismatch.evidence.verified is False
    assert verified.evidence.classification is EvidenceClassification.HOST_VERIFIED
    assert verified.evidence.verified is True
    assert verified.artifacts["observed"]["output"] == list(probe.expected.storage)
    assert verified.artifacts["expected"]["storage"] == list(probe.expected.storage)
    assert verified.artifact_sha256["observed"] != mismatch.artifact_sha256["observed"]


def test_backend_cannot_claim_compile_or_runtime_success_without_evidence(tmp_path):
    valid = _compile()
    with pytest.raises(ValueError, match="evidence_sha256"):
        replace(valid, evidence_sha256="invented")
    with pytest.raises(ValueError, match="artifact_id"):
        replace(valid, artifact_id=None)
    with pytest.raises(ValueError, match="output"):
        RuntimeObservation(
            passed=True,
            evidence_sha256=canonical_digest("runtime"),
            diagnostic="no output",
            output=None,
        )


def test_executor_rejects_objects_outside_the_explicit_backend_protocol(tmp_path):
    backend = FakeBackend({"passed": False})
    with pytest.raises(TypeError, match="CompileObservation"):
        ProjectExecutor(
            backend, EvidenceLedger(tmp_path / "evidence.jsonl")
        ).execute(_spec("movement"), "candidate")
