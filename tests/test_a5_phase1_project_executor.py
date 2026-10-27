from __future__ import annotations

from dataclasses import replace
import math

import pytest

from benchmarks.a5kernels.evidence import EvidenceLedger, canonical_digest
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.a5kernels.phase1_project import EvidenceClassification, ProjectGraph
from benchmarks.a5kernels.phase1_project_executor import (
    CompileObservation,
    ProjectExecutionFamily,
    ProjectExecutor,
    RuntimeObservation,
    ProjectKernelABI,
)


def test_target_binding_changes_language_not_host_oracle(tmp_path):
    class NativeBinding:
        def abi(self, spec, inputs):
            return ProjectKernelABI("native_kernel", ("input",), "output")

        def instruction(self, spec, family, abi, inputs_sha256):
            return f"Ascend C {abi.kernel_name}: {inputs_sha256}"

    backend = FakeBackend(_compile())
    default = ProjectExecutor(backend, EvidenceLedger(tmp_path / "default.jsonl"))
    native = ProjectExecutor(
        backend, EvidenceLedger(tmp_path / "native.jsonl"), binding=NativeBinding(),
    )
    spec = _spec("movement")
    expected = default.prepare(spec)
    actual = native.prepare(spec)
    assert actual.inputs == expected.inputs
    assert actual.expected == expected.expected
    assert actual.family == expected.family
    assert actual.abi.kernel_name == "native_kernel"
    backend.runtime_result = _runtime(actual.expected.storage)
    good = native.execute(spec, "native candidate")
    assert good.evidence.classification is EvidenceClassification.HOST_VERIFIED
    backend.runtime_result = _runtime([999.0] * len(actual.expected.storage))
    bad = native.execute(spec, "native candidate")
    assert bad.evidence.classification is EvidenceClassification.HOST_MISMATCH


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
    assert backend.compile_requests[0].expected == result.prepared.expected
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


@pytest.mark.parametrize("passed", [False, True])
def test_compile_only_returns_exact_evidence_without_running_or_verifying(
    tmp_path, passed,
):
    backend = FakeBackend(_compile(passed=passed), _runtime((1.0,)))
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")

    result = ProjectExecutor(backend, ledger).compile(
        _spec("movement"), "compile probe source"
    )

    assert result.observation.passed is passed
    assert result.artifact == {
        "compile_passed": passed,
        "compile_diagnostic": (
            "compiled" if passed else "frontend rejected candidate"
        ),
        "compile_evidence_sha256": result.observation.evidence_sha256,
        "compiled_artifact_id": "compiled-object" if passed else None,
    }
    assert result.artifact_sha256 == canonical_digest(result.artifact)
    assert backend.compile_requests[0].candidate_source == "compile probe source"
    assert backend.run_requests == []
    assert ledger.entries[-1].payload["event"] == "phase1-project-compile-artifact"


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


@pytest.mark.parametrize(
    ("bad_value", "retained"),
    [(math.nan, "NaN"), (math.inf, "+Infinity"), (-math.inf, "-Infinity")],
)
def test_non_finite_kernel_output_is_retained_and_classified_as_mismatch(
    tmp_path, bad_value, retained,
):
    spec = _spec("movement")
    probe = ProjectExecutor(
        FakeBackend(_compile()), EvidenceLedger(tmp_path / "probe.jsonl")
    ).prepare(spec)
    output = list(probe.expected.storage)
    output[0] = bad_value
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")

    result = ProjectExecutor(
        FakeBackend(_compile(), _runtime(output)), ledger
    ).execute(spec, "candidate")

    assert result.evidence.classification is EvidenceClassification.HOST_MISMATCH
    assert result.artifacts["observed"]["output"][0] == retained
    assert result.artifacts["observed"]["output_encoding"] == (
        "json-number-or-ieee754-symbol"
    )
    assert len(ledger.entries) == 4


def test_failure_repair_project_materializes_for_the_adapter_repair_loop(tmp_path):
    mapping = _spec("movement").as_dict()
    mapping["capabilities"].append("failure-repair")
    mapping["failure_mode"] = "compile"
    repair = ProjectSpec.from_mapping(mapping)

    backend = FakeBackend(_compile(passed=False))
    executor = ProjectExecutor(backend, EvidenceLedger(tmp_path / "evidence.jsonl"))

    prepared = executor.prepare(repair)
    result = executor.execute(repair, "intentionally failing candidate")

    assert prepared.spec == repair
    assert result.prepared == prepared
    assert result.evidence.classification is EvidenceClassification.COMPILE_FAILURE


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
