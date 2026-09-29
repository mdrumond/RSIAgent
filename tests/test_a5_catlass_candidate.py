from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.candidate import (
    EXACT_KERNEL_NAME,
    CandidateProfileEvaluation,
    CatlassCandidateBackend,
    validate_candidate_source,
)
from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.matrix import Workload
from benchmarks.a5kernels.protocol import ExecutionReceipt, RunRequest
from benchmarks.a5kernels.runner import A5KernelRunner
from benchmarks.a5kernels.trial import TrialAction, TrialActionExecutor


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
    backend = CatlassCandidateBackend(execution, length=7, seed=19, device=5)

    first = backend.compile(tmp_path, "catlass-dsl", "compile-1")
    second = backend.compile(tmp_path, "catlass-dsl", "compile-2")

    assert first["execution_id"] == second["execution_id"]
    assert first["source_fingerprint"] == second["source_fingerprint"]
    assert first["attestation_sha256"] != second["attestation_sha256"]
    assert first["passed"] is True
    assert first["exit_code"] == 0
    assert first["kernel_name"] == EXACT_KERNEL_NAME
    assert first["diagnostics"] == "Catlass compilation succeeded"
    plan = execution.plans[0]
    assert plan.attempt_id != execution.plans[1].attempt_id
    assert replace(plan, attempt_id=execution.plans[1].attempt_id) == execution.plans[1]
    assert plan.argv[:2] == ("python", "-B")
    assert dict(plan.environment.bindings) == {
        "A5KERNEL_BLOCK_NUM": "1",
        "A5KERNEL_COMPILE_ONLY": "1",
        "BZ_A5_PROFILE_PHYSICAL_DEVICE": "5",
    }
    assert "A5KERNEL_COMPILE_ONLY" not in plan.environment.unset
    candidate = next(item for item in plan.files if item.relative_path == "kernel.py")
    assert SOURCE.rstrip() in candidate.content
    assert "tla.compile(" in candidate.content
    assert "--npu-arch 3510" in candidate.content


def test_same_source_compiles_have_isolated_trial_and_retry_sessions(tmp_path):
    from benchmarks.a5kernels.bz import BZSessionAdapter, CommandResult

    first = tmp_path / "repeat-1"
    second = tmp_path / "repeat-2"
    for workspace in (first, second):
        workspace.mkdir()
        (workspace / "kernel.py").write_text(SOURCE)
    execution = FakeExecution()
    executors = [TrialActionExecutor(
        workspace=workspace, language="catlass-dsl", workload=Workload.SMOKE_VECTOR_ADD,
        backend=CatlassCandidateBackend(execution),
        ledger=EvidenceLedger(workspace / "evidence.jsonl"),
        knowledge=type("Knowledge", (), {"enabled": False})(),
        profiling=object(), profiling_guidance=False,
    ) for workspace in (first, second)]
    for executor in (executors[0], executors[1], executors[0]):
        executor(TrialAction("compile", {}))

    attempts = [plan.attempt_id for plan in execution.plans]
    namespaces = [attempt.split("-")[0] for attempt in attempts]
    assert namespaces[0] == namespaces[2] != namespaces[1]
    assert attempts[0].endswith("-compile-1")
    assert attempts[1].endswith("-compile-1")
    assert attempts[2].endswith("-compile-2")
    assert len(set(attempts)) == 3
    assert all(len(attempt) <= 64 for attempt in attempts)
    assert len({plan.execution_id for plan in execution.plans}) == 1
    invocations = []

    def dispatch(invocation):
        invocations.append(invocation)
        return CommandResult(1, "offline dispatch probe")

    adapter = BZSessionAdapter(SimpleNamespace(
        runtime_provenance=execution.runtime_provenance, run=dispatch,
    ), session_wrapper="profiles/bz-a5/session.sh")
    for plan in execution.plans:
        adapter.execute(plan)
    assert len({invocation.argv[2] for invocation in invocations}) == 3
    assert len({invocation.remote_directory for invocation in invocations}) == 3


def test_candidate_without_launch_constants_uses_host_owned_runtime(tmp_path):
    source = SOURCE.replace("\nVECTOR_ELE = 400\n", "\n")
    (tmp_path / "kernel.py").write_text(source)
    execution = FakeExecution()

    result = CatlassCandidateBackend(execution).compile(tmp_path, "catlass-dsl", "compile-1")

    assert result["passed"]
    composed = next(item for item in execution.plans[0].files
                    if item.relative_path == "kernel.py").content
    assert "_HOST_VECTOR_ELE = 448" in composed
    assert "_HOST_VL_ELE = 64" in composed


def test_compile_failure_returns_bounded_diagnostics_and_allows_retry(tmp_path):
    (tmp_path / "kernel.py").write_text(SOURCE)
    execution = FakeExecution(marker=None, exit_code=1)
    backend = CatlassCandidateBackend(execution)

    failed = backend.compile(tmp_path, "catlass-dsl", "compile-1")
    execution.exit_code = 0
    execution.marker = EXACT_KERNEL_NAME
    retried = backend.compile(tmp_path, "catlass-dsl", "compile-2")

    assert failed["passed"] is False
    assert failed["kernel_name"] is None
    assert failed["diagnostics"] == "ordinary compiler error"
    assert retried["passed"] is True


def test_run_returns_exact_plan_bound_candidate(tmp_path):
    (tmp_path / "kernel.py").write_text(SOURCE)
    execution = FakeExecution()
    backend = CatlassCandidateBackend(execution, length=5, device=6)
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")

    run = backend.run(
        tmp_path, "catlass-dsl", Workload.SMOKE_VECTOR_ADD, "candidate-4", ledger
    )

    assert run.kernel_name == EXACT_KERNEL_NAME
    assert run.candidate_source_sha256 == hashlib.sha256(SOURCE.encode()).hexdigest()
    assert run.verified.passed
    assert run.plan is execution.plans[0]
    assert run.plan.attempt_id == "candidate-4"
    assert "A5KERNEL_COMPILE_ONLY" not in dict(run.plan.environment.bindings)
    assert dict(run.plan.environment.bindings)[
        "BZ_A5_PROFILE_PHYSICAL_DEVICE"
    ] == "6"
    assert run.plan.argv[:2] == ("python", "-B")
    assert run.verified.execution_id == run.plan.execution_id
    assert run.verified.source_fingerprint == run.plan.source_fingerprint


def test_candidate_verifies_immutable_zero_padded_smoke_inputs(tmp_path):
    (tmp_path / "kernel.py").write_text(SOURCE)
    execution = FakeExecution()
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")
    run = CatlassCandidateBackend(execution).run(
        tmp_path, "catlass-dsl", Workload.SMOKE_VECTOR_ADD, "final", ledger,
    )
    unpadded = A5KernelRunner(FakeExecution()).prepare(RunRequest("catlass-dsl"))

    assert run.verified.passed
    assert run.plan.input_a == unpadded.input_a + (0.0,) * 32
    assert run.plan.input_b == unpadded.input_b + (0.0,) * 32
    request = ledger.entries[0].payload["request"]
    assert request["length"] == 32 and request["padded_length"] == 64
    assert RunRequest(**request).request_id == run.plan.request_id != unpadded.request_id
    kernel = next(item.content for item in run.plan.files if item.relative_path == "kernel.py")
    assert "torch.full_like(a, float('nan'))" in kernel
    assert "return out.cpu().tolist()" in kernel
    assert "return out[:original_length]" not in kernel


@pytest.mark.parametrize("padding", [(), (float("nan"),) * 32, (1.0,) * 32])
def test_half_work_candidate_output_cannot_pass(tmp_path, padding):
    class HalfWorkExecution(FakeExecution):
        def execute(self, plan):
            receipt = super().execute(plan)
            return replace(receipt, output=receipt.output[:32] + padding)

    (tmp_path / "kernel.py").write_text(SOURCE)
    run = CatlassCandidateBackend(HalfWorkExecution()).run(
        tmp_path, "catlass-dsl", Workload.SMOKE_VECTOR_ADD, "final",
        EvidenceLedger(tmp_path / "evidence.jsonl"),
    )
    assert not run.verified.passed
    assert run.kernel_name is None


def test_run_rejects_missing_or_unexpected_discovered_name(tmp_path):
    (tmp_path / "kernel.py").write_text(SOURCE)
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")
    for marker in (None, "agent_claimed_name"):
        backend = CatlassCandidateBackend(FakeExecution(marker=marker))
        with pytest.raises(RuntimeError, match="one exact kernel name"):
            backend.run(
                tmp_path, "catlass-dsl", Workload.SMOKE_VECTOR_ADD, "final", ledger
            )


def test_failed_run_without_kernel_marker_is_recoverable_action_observation(tmp_path):
    execution = FakeExecution(marker=None, exit_code=1)
    backend = CatlassCandidateBackend(execution)
    executor = TrialActionExecutor(
        workspace=tmp_path,
        language="catlass-dsl",
        workload=Workload.SMOKE_VECTOR_ADD,
        backend=backend,
        ledger=EvidenceLedger(tmp_path / "evidence.jsonl"),
        knowledge=type("Knowledge", (), {"enabled": False})(),
        profiling=object(),
        profiling_guidance=False,
    )
    executor(TrialAction("write", {"slot": "kernel", "content": SOURCE}))

    outcome = executor(TrialAction("run", {}))

    assert outcome.terminal is False
    assert json.loads(outcome.observation)["host"] == {
        "event": "run",
        "max_abs_error": None,
        "error_status": "non-finite-max-abs-error",
        "exit_code": 1,
        "passed": False,
        "status": "runtime-error",
    }
    assert executor.latest_run is not None
    assert executor.latest_run.kernel_name is None
    assert executor.latest_run.verified.exit_code == 1


def test_runtime_rejects_other_language_and_workload(tmp_path):
    (tmp_path / "kernel.py").write_text(SOURCE)
    backend = CatlassCandidateBackend(FakeExecution())
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")
    with pytest.raises(ValueError, match="requires catlass-dsl"):
        backend.compile(tmp_path, "ascend-c", "compile-1")
    with pytest.raises(ValueError, match="only smoke-vector-add"):
        backend.run(tmp_path, "catlass-dsl", Workload.SEMANTIC_GEMM, "x", ledger)


@pytest.mark.parametrize("device", [-1, True, 1.5])
def test_runtime_rejects_invalid_device(device):
    with pytest.raises(ValueError, match="non-negative integer"):
        CatlassCandidateBackend(FakeExecution(), device=device)


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


def _candidate(tmp_path, *, device=0):
    (tmp_path / "kernel.py").write_text(SOURCE)
    return CatlassCandidateBackend(FakeExecution(), length=3, device=device).run(
        tmp_path,
        "catlass-dsl",
        Workload.SMOKE_VECTOR_ADD,
        "final",
        EvidenceLedger(tmp_path / "evidence.jsonl"),
    )


def test_profile_adapter_binds_both_campaigns_to_candidate(tmp_path):
    run = _candidate(tmp_path, device=6)
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
        assert dict(request.plan.environment.bindings)[
            "BZ_A5_PROFILE_PHYSICAL_DEVICE"
        ] == "6"


def test_profile_adapter_rejects_unbound_kernel_name(tmp_path):
    run = replace(_candidate(tmp_path), kernel_name="other")
    evaluation = CandidateProfileEvaluation(FakeController(None), device=0)
    with pytest.raises(ValueError, match="host-discovered"):
        evaluation.final(run)
