from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from benchmarks.a5kernels.candidate import (
    EXACT_KERNEL_NAME,
    CandidateProfileEvaluation,
    CatlassCandidateBackend,
    validate_candidate_source,
)
from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.matrix import Workload
from benchmarks.a5kernels.protocol import ExecutionReceipt


SOURCE = '''\
import catlass.tla as tla

VECTOR_ELE = 400

@tla.kernel
def vector_add(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(gm_c, gm_a)
'''


class FakeExecution:
    runtime_provenance = (
        ("catlass_revision", "a" * 40),
        ("execution_profile", "bz-a5"),
    )

    def __init__(self, *, marker=EXACT_KERNEL_NAME, exit_code=0):
        self.marker = marker
        self.exit_code = exit_code
        self.plans = []

    def execute(self, plan):
        self.plans.append(plan)
        output = tuple(left + right for left, right in zip(plan.input_a, plan.input_b))
        marker = "" if self.marker is None else f"A5KERNEL_NAME={self.marker}\n"
        if self.exit_code:
            output = ()
        return ExecutionReceipt(self.exit_code, output, stdout=marker,
                                stderr="ordinary compiler error" if self.exit_code else "",
                                session_handle="bz-a5:fake")


@pytest.mark.parametrize("source, message", [
    ("", "must not be empty"),
    ("def nope(:", "not valid Python"),
    ("def vector_add(gm_a, gm_b, gm_c): pass", "exactly @tla.kernel"),
    ("import os\n" + SOURCE, "import only catlass.tla"),
    (SOURCE + "\ndef helper(): pass\n", "exactly one synchronous"),
    (SOURCE.replace("gm_c", "out"), "must use vector_add"),
])
def test_candidate_source_rejects_non_abi_modules(source, message):
    with pytest.raises(ValueError, match=message):
        validate_candidate_source(source)


def test_compile_invokes_real_plan_with_host_owned_compile_mode(tmp_path):
    (tmp_path / "kernel.py").write_text(SOURCE)
    execution = FakeExecution()
    backend = CatlassCandidateBackend(execution, length=7, seed=19)

    first = backend.compile(tmp_path, "catlass-dsl")
    second = backend.compile(tmp_path, "catlass-dsl")

    assert first == second
    assert first["passed"] is True
    assert first["exit_code"] == 0
    assert first["kernel_name"] == EXACT_KERNEL_NAME
    assert first["diagnostics"] == "Catlass compilation succeeded"
    plan = execution.plans[0]
    assert plan == execution.plans[1]
    assert plan.argv[:2] == ("env", "A5KERNEL_COMPILE_ONLY=1")
    assert plan.argv[2:5] == ("env", "-u", "PYTHONPYCACHEPREFIX")
    candidate = next(item for item in plan.files if item.relative_path == "kernel.py")
    assert SOURCE.rstrip() in candidate.content
    assert "tla.compile(" in candidate.content
    assert "--npu-arch 3510" in candidate.content


def test_candidate_without_launch_constants_uses_host_owned_runtime(tmp_path):
    source = SOURCE.replace("\nVECTOR_ELE = 400\n", "\n")
    (tmp_path / "kernel.py").write_text(source)
    execution = FakeExecution()

    result = CatlassCandidateBackend(execution).compile(tmp_path, "catlass-dsl")

    assert result["passed"]
    composed = next(item for item in execution.plans[0].files
                    if item.relative_path == "kernel.py").content
    assert "_HOST_VECTOR_ELE = 400" in composed
    assert "_HOST_VL_ELE = 64" in composed


def test_compile_failure_returns_bounded_diagnostics_and_allows_retry(tmp_path):
    (tmp_path / "kernel.py").write_text(SOURCE)
    execution = FakeExecution(marker=None, exit_code=1)
    backend = CatlassCandidateBackend(execution)

    failed = backend.compile(tmp_path, "catlass-dsl")
    execution.exit_code = 0
    execution.marker = EXACT_KERNEL_NAME
    retried = backend.compile(tmp_path, "catlass-dsl")

    assert failed["passed"] is False
    assert failed["kernel_name"] is None
    assert failed["diagnostics"] == "ordinary compiler error"
    assert retried["passed"] is True


def test_run_returns_exact_plan_bound_candidate(tmp_path):
    (tmp_path / "kernel.py").write_text(SOURCE)
    execution = FakeExecution()
    backend = CatlassCandidateBackend(execution, length=5)
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")

    run = backend.run(
        tmp_path, "catlass-dsl", Workload.SMOKE_VECTOR_ADD, "candidate-4", ledger
    )

    assert run.kernel_name == EXACT_KERNEL_NAME
    assert run.verified.passed
    assert run.plan is execution.plans[0]
    assert run.plan.attempt_id == "candidate-4"
    assert "A5KERNEL_COMPILE_ONLY=1" not in run.plan.argv
    assert run.verified.execution_id == run.plan.execution_id
    assert run.verified.source_fingerprint == run.plan.source_fingerprint


def test_run_rejects_missing_or_unexpected_discovered_name(tmp_path):
    (tmp_path / "kernel.py").write_text(SOURCE)
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")
    for marker in (None, "agent_claimed_name"):
        backend = CatlassCandidateBackend(FakeExecution(marker=marker))
        with pytest.raises(RuntimeError, match="one exact kernel name"):
            backend.run(
                tmp_path, "catlass-dsl", Workload.SMOKE_VECTOR_ADD, "final", ledger
            )


def test_runtime_rejects_other_language_and_workload(tmp_path):
    (tmp_path / "kernel.py").write_text(SOURCE)
    backend = CatlassCandidateBackend(FakeExecution())
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")
    with pytest.raises(ValueError, match="requires catlass-dsl"):
        backend.compile(tmp_path, "ascend-c")
    with pytest.raises(ValueError, match="only smoke-vector-add"):
        backend.run(tmp_path, "catlass-dsl", Workload.SEMANTIC_GEMM, "x", ledger)


class FakeController:
    def __init__(self, intermediate):
        self.intermediate_result = intermediate
        self.calls = []

    def run_intermediate(self, verified, request):
        self.calls.append(("intermediate", verified, request))
        return self.intermediate_result

    def run_final(self, verified, request):
        self.calls.append(("final", verified, request))
        return _FinalResult(3.25, request.expected_kernel)


@dataclass(frozen=True)
class _FinalResult:
    duration_us: float
    kernel_name: str


def _candidate(tmp_path):
    (tmp_path / "kernel.py").write_text(SOURCE)
    return CatlassCandidateBackend(FakeExecution(), length=3).run(
        tmp_path,
        "catlass-dsl",
        Workload.SMOKE_VECTOR_ADD,
        "final",
        EvidenceLedger(tmp_path / "evidence.jsonl"),
    )


def test_profile_adapter_binds_both_campaigns_to_candidate(tmp_path):
    run = _candidate(tmp_path)
    controller = FakeController(None)
    evaluation = CandidateProfileEvaluation(controller, device=6)

    assert evaluation.intermediate(run) == {"available": False}
    final = evaluation.final(run)

    assert final == {"duration_us": 3.25, "kernel_name": EXACT_KERNEL_NAME}
    assert [call[0] for call in controller.calls] == ["intermediate", "final"]
    for _, verified, request in controller.calls:
        assert verified is run.verified
        assert request.plan is run.plan
        assert request.expected_kernel == run.kernel_name
        assert request.device == 6


def test_profile_adapter_rejects_unbound_kernel_name(tmp_path):
    run = replace(_candidate(tmp_path), kernel_name="other")
    evaluation = CandidateProfileEvaluation(FakeController(None), device=0)
    with pytest.raises(ValueError, match="host-discovered"):
        evaluation.final(run)
