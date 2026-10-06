from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.a3kernels.candidate import A3CandidateBackend, CandidateCompilation
from benchmarks.a3kernels.phase1_protocol import (
    ExecutionReceipt,
    FailedEvidence,
    VerifiedResult,
    attest,
)
from benchmarks.a3kernels.phase2_target import (
    PHASE2_TARGET_CASES,
    PHASE2_TARGET_ID,
    Phase2TargetEvidence,
    Phase2TargetVerifier,
)


SOURCE = r'''#include "kernel_operator.h"
extern "C" __global__ __aicore__ void vector_add(
    GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output,
    uint32_t count, uint32_t buffer_bytes) {}
'''


class FakeBackend:
    def __init__(self, *, fail_case: str | None = None, wrong_case: str | None = None):
        self.fail_case = fail_case
        self.wrong_case = wrong_case
        self.compiled = []
        self.executed = []

    def compile(self, plan, workdir: Path):
        self.compiled.append((plan, workdir))
        case_id = plan.project_id.rsplit("/", 1)[-1]
        if case_id == self.fail_case:
            return FailedEvidence.create(
                plan, stage="compile", error_type="CompileError", detail="bad source"
            )
        body = {
            "plan": plan,
            "library_sha256": "a" * 64,
            "stdout": "compiled",
            "stderr": "",
        }
        return CandidateCompilation(
            plan, "a" * 64, "compiled", "", attest(body)
        )

    def execute(self, compilation):
        plan = compilation.plan
        self.executed.append(plan)
        case_id = plan.project_id.rsplit("/", 1)[-1]
        output = [a + b for a, b in zip(plan.input_a, plan.input_b)]
        if case_id == self.wrong_case:
            output[plan.logical_length - 1] += 1.0
        maximum = max(
            abs(output[index] - plan.input_a[index] - plan.input_b[index])
            for index in range(plan.logical_length)
        )
        return VerifiedResult.from_receipt(
            plan,
            ExecutionReceipt(
                exit_code=0,
                output=tuple(output),
                stdout="host verified",
                job_handle=f"bz-a3-1:{case_id}",
                metadata=(("library_sha256", compilation.library_sha256),),
            ),
            max_abs_error=maximum,
        )


def _verifier(backend):
    planner = A3CandidateBackend(lambda *args, **kwargs: None)
    return Phase2TargetVerifier(backend, plan_builder=planner.plan)


def test_target_cases_are_fixed_boundary_padding_and_multiblock_suite():
    assert PHASE2_TARGET_ID == "a3-phase2-tiled-vector-add-v1"
    assert [
        (case.logical_length, case.padded_length, case.block_count)
        for case in PHASE2_TARGET_CASES
    ] == [
        (1, 64, 1),
        (31, 64, 1),
        (33, 64, 1),
        (65, 128, 2),
        (255, 256, 4),
        (400, 448, 4),
        (4096, 4096, 8),
    ]
    assert len({case.case_id for case in PHASE2_TARGET_CASES}) == 7
    assert len({case.seed for case in PHASE2_TARGET_CASES}) == 7


def test_complete_fake_runtime_proves_every_case_and_persists_evidence(tmp_path):
    backend = FakeBackend()
    evidence = _verifier(backend).verify(
        SOURCE,
        request_id="lineage-1",
        attempt_id="target-1",
        execution_profile="bz-a3-1",
        workdir=tmp_path / "runs",
    )

    assert evidence.passed is True
    assert len(evidence.cases) == len(PHASE2_TARGET_CASES)
    assert len(backend.compiled) == len(backend.executed) == 7
    assert len({item.source_fingerprint for item in evidence.cases}) == 1
    assert len({item.execution_id for item in evidence.cases}) == 7
    assert [item.job_handle for item in evidence.cases] == [
        f"bz-a3-1:{case.case_id}" for case in PHASE2_TARGET_CASES
    ]
    assert all(item.passed and item.compile_attestation_sha256 for item in evidence.cases)
    assert evidence.candidate_sha256 == hashlib.sha256(SOURCE.encode()).hexdigest()

    destination = tmp_path / "evidence.json"
    evidence.write(destination)
    assert Phase2TargetEvidence.read(destination) == evidence
    first_bytes = destination.read_bytes()
    evidence.write(destination)
    assert destination.read_bytes() == first_bytes


@pytest.mark.parametrize("mode", ["compile", "verify"])
def test_aggregate_fails_if_any_case_fails_but_retains_all_case_evidence(tmp_path, mode):
    failed_case = PHASE2_TARGET_CASES[3].case_id
    backend = FakeBackend(
        fail_case=failed_case if mode == "compile" else None,
        wrong_case=failed_case if mode == "verify" else None,
    )

    evidence = _verifier(backend).verify(
        SOURCE,
        request_id="lineage-1",
        attempt_id="target-2",
        execution_profile="bz-a3-1",
        workdir=tmp_path,
    )

    assert evidence.passed is False
    assert len(evidence.cases) == 7
    failed = next(item for item in evidence.cases if item.case.case_id == failed_case)
    assert failed.passed is False
    assert failed.failure_stage == mode
    assert failed.result_attestation_sha256
    assert len(backend.compiled) == 7
    assert len(backend.executed) == (6 if mode == "compile" else 7)


def test_aggregate_contract_rejects_forged_pass_and_mixed_source_identity(tmp_path):
    evidence = _verifier(FakeBackend()).verify(
        SOURCE,
        request_id="lineage",
        attempt_id="attempt",
        execution_profile="bz-a3-1",
        workdir=tmp_path,
    )
    failed_case = replace(evidence.cases[0], passed=False, failure_stage="verify")
    with pytest.raises(ValueError, match="aggregate pass"):
        replace(evidence, cases=(failed_case, *evidence.cases[1:]))

    foreign = replace(evidence.cases[0], source_fingerprint="f" * 64)
    with pytest.raises(ValueError, match="source fingerprint"):
        replace(evidence, cases=(foreign, *evidence.cases[1:]), passed=False)


def test_persisted_evidence_rejects_tampering_and_conflicting_overwrite(tmp_path):
    evidence = _verifier(FakeBackend()).verify(
        SOURCE,
        request_id="lineage",
        attempt_id="attempt",
        execution_profile="bz-a3-1",
        workdir=tmp_path / "run",
    )
    destination = tmp_path / "evidence.json"
    evidence.write(destination)

    value = json.loads(destination.read_text())
    value["passed"] = False
    destination.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="aggregate pass|attestation"):
        Phase2TargetEvidence.read(destination)
    with pytest.raises(RuntimeError, match="conflicts"):
        evidence.write(destination)
