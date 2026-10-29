from dataclasses import replace
from types import SimpleNamespace

import pytest

from benchmarks.ascend import attention as shared
from benchmarks.a5kernels import attention as a5
from benchmarks.a3kernels.attention import (
    AttentionTargetEvidence, AttentionTargetVerifier, NativeAttentionBackend, build_plan, TARGET_ID,
)
from benchmarks.a3kernels.phase1_protocol import FailedEvidence
from benchmarks.a3kernels.remote_candidate_bz import BZA3RemoteCandidateBackend


SOURCE = '''#include "kernel_operator.h"
extern "C" __global__ __aicore__ void dense_attention(GM_ADDR gm_q, GM_ADDR gm_k, GM_ADDR gm_v, GM_ADDR gm_output) {}
'''


def test_a5_attention_contract_is_shared_without_identity_changes():
    assert a5.ATTENTION_CASES is shared.ATTENTION_CASES
    assert a5.build_execution is shared.build_execution
    assert a5.attention_reference is shared.attention_reference
    assert a5.AttentionReceipt is shared.AttentionReceipt
    assert a5.TARGET_ID == "a5-phase2-dense-forward-attention-v1"
    assert a5.SUITE_SHA256 == shared.SUITE_SHA256


def test_native_plan_binds_real_inputs_dtype_and_case_constants():
    for case in shared.ATTENTION_CASES:
        execution = shared.build_execution(case)
        plan = build_plan(SOURCE, execution, request_id="req", attempt_id="try",
                          execution_profile="bz-a3-2", target="observed-soc", arch="observed-arch")
        assert tuple(t.values for t in plan.inputs) == (execution.q, execution.k, execution.v)
        assert all(t.dtype == case.dtype and t.shape == case.shape for t in plan.inputs)
        assert plan.output.values is None
        assert plan.output.dtype == case.dtype
        assert plan.block_count == 1
        assert plan.target == "observed-soc" and plan.arch == "observed-arch"
        assert f"#define SEQUENCE {case.sequence}" in plan.source
        assert f"#define CAUSAL {int(case.causal)}" in plan.source


class NativeRuntime(BZA3RemoteCandidateBackend):
    """Stub transport only; inherit the real public legacy execute contract."""
    def __init__(self, *, wrong=False):
        self.plans = []
        self.wrong = wrong

    def compile(self, plan, directory):
        self.plans.append(plan)
        return plan

    def execute_receipt(self, plan):
        case = next(c for c in shared.ATTENTION_CASES if c.case_id == plan.project_id)
        output = shared.attention_reference(shared.build_execution(case))
        if self.wrong:
            output = tuple(42.0 for _ in output)
        return SimpleNamespace(exit_code=0, output=output, stdout="", stderr="",
                               job_handle="bz-a3-1:fixture", metadata=())


@pytest.mark.parametrize("wrong", [False, True])
def test_a3_suite_uses_host_grading_and_separate_identity(tmp_path, wrong):
    runtime = NativeRuntime(wrong=wrong)
    backend = NativeAttentionBackend(runtime, request_id="req", attempt_id="try",
                                     execution_profile="bz-a3-1", target="soc", arch="arch")
    result = AttentionTargetVerifier(backend).verify(
        SOURCE, request_id="req", attempt_id="try", execution_profile="bz-a3-1", workdir=tmp_path)
    assert result.target_id == TARGET_ID != a5.TARGET_ID
    assert result.suite_sha256 == a5.SUITE_SHA256
    assert result.passed is (not wrong)
    assert len(runtime.plans) == len(a5.ATTENTION_CASES)
    assert all(t.case_id == c.case_id for t, c in zip(result.cases, a5.ATTENTION_CASES))
    result.write(tmp_path / "evidence.json")
    assert AttentionTargetEvidence.read(tmp_path / "evidence.json") == result
    with pytest.raises(ValueError):
        replace(result, passed=not result.passed)


def test_a3_backend_rejects_profile_mismatch_before_running(tmp_path):
    backend = NativeAttentionBackend(NativeRuntime(), request_id="req", attempt_id="try",
                                     execution_profile="bz-a3-1", target="soc", arch="arch")
    with pytest.raises(ValueError):
        AttentionTargetVerifier(backend).verify(SOURCE, request_id="req", attempt_id="try",
                                               execution_profile="bz-a5", workdir=tmp_path)


@pytest.mark.parametrize("stage,error,infrastructure", [
    ("compile", "CompilerError", False),
    ("prepare", "RuntimeError", True),
    ("execute", "AmbiguousRemoteOutcome", True),
])
def test_native_failure_does_not_invent_a_success(tmp_path, stage, error, infrastructure):
    class FailingRuntime:
        def compile(self, plan, directory):
            return FailedEvidence.create(plan, stage=stage, error_type=error, detail="failed host action")

    backend = NativeAttentionBackend(FailingRuntime(), request_id="req", attempt_id="try",
                                     execution_profile="bz-a3-1", target="soc", arch="arch")
    if infrastructure:
        with pytest.raises(shared.AttentionInfrastructureError):
            backend.execute(SOURCE, shared.build_execution(shared.ATTENTION_CASES[0]), tmp_path)
    else:
        receipt = backend.execute(SOURCE, shared.build_execution(shared.ATTENTION_CASES[0]), tmp_path)
        assert receipt.exit_code != 0 and not receipt.output


def test_bounded_stage_and_host_grade_are_separate(tmp_path, capsys):
    import json
    from benchmarks.a3kernels.attention_smoke import main

    case = shared.ATTENTION_CASES[0]
    source = tmp_path / "source.cpp"
    source.write_text(SOURCE)
    stage = tmp_path / "stage"
    assert main(["stage", "--source", str(source), "--stage", str(stage),
                 "--target", "soc", "--arch", "arch", "--case", case.case_id]) == 0
    payload = json.loads((stage / "input.json").read_text())
    assert payload["output"]["values"] is None
    stdout = tmp_path / "stdout.log"
    expected = shared.attention_reference(shared.build_execution(case))
    stdout.write_text("A3KERNEL_OUTPUT=" + json.dumps(expected))
    assert main(["grade", "--stdout", str(stdout), "--case", case.case_id]) == 0
    stdout.write_text("A3KERNEL_OUTPUT=" + json.dumps([0] * case.output_elements))
    assert main(["grade", "--stdout", str(stdout), "--case", case.case_id]) == 1
