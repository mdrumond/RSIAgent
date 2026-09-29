from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from benchmarks.a5kernels.candidate import (
    EXACT_KERNEL_NAME,
    CandidateProfileEvaluation,
    CatlassCandidateBackend,
    validate_candidate_source,
)
from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.matrix import Workload
from benchmarks.a5kernels.phase1_registry import DEFAULT_PROPOSALS, CurriculumProposal
from benchmarks.a5kernels.phase1_runtime import (
    COMPILE_FAILURE_STARTER,
    HOST_FAILURE_STARTER,
    Phase1ProjectRuntime,
    RecoveryEvidence,
    RecoveryStarter,
)
from benchmarks.a5kernels.protocol import ExecutionReceipt


class FakeExecution:
    runtime_provenance = (("execution_profile", "bz-a5"),)

    def __init__(self):
        self.plans = []

    def execute(self, plan):
        self.plans.append(plan)
        output = tuple(left + right for left, right in zip(plan.input_a, plan.input_b))
        return ExecutionReceipt(
            0, output, stdout=f"A5KERNEL_NAME={EXACT_KERNEL_NAME}\n",
            session_handle="bz-a5:phase1",
        )


def _proposal(family):
    return next(item for item in DEFAULT_PROPOSALS if item.family.value == family)


@pytest.mark.parametrize("family,length,padded,blocks", [
    ("vector-add-baseline", 32, 64, 1),
    ("padded-multitile", 400, 448, 1),
    ("length-knee", 128, 128, 1),
    ("cross-layer-launch", 32, 64, 1),
    ("msprof-pipe", 32, 64, 1),
])
def test_registered_families_resolve_to_fixed_host_runtime(family, length, padded, blocks):
    runtime = Phase1ProjectRuntime.from_proposal(_proposal(family))
    assert runtime.logical_length == length
    assert runtime.padded_length == padded
    assert runtime.block_count == blocks
    assert runtime.recovery is None
    assert runtime.as_dict()["recovery"] is None


def test_runtime_parameters_are_bound_into_plan_and_host_evidence(tmp_path):
    proposal = CurriculumProposal.from_mapping({
        "family": "cross-layer-launch", "parameters": {"block_count": 6},
        "hypothesis": "Six host-selected blocks exercise the launch boundary.",
        "evidence_preset": "correctness-timing",
    })
    runtime = Phase1ProjectRuntime.from_proposal(proposal)
    execution = FakeExecution()
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "kernel.py").write_text(HOST_FAILURE_STARTER)
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")

    run = runtime.backend(execution, seed=9, device=3).run(
        workspace, "catlass-dsl", Workload.SMOKE_VECTOR_ADD, "phase1-7", ledger
    )

    assert run.verified.passed
    assert len(run.plan.input_a) == 64
    assert run.plan.argv[:2] == ("python", "-B")
    assert dict(run.plan.environment.bindings) == {
        "A5KERNEL_BLOCK_NUM": "6",
        "BZ_A5_PROFILE_PHYSICAL_DEVICE": "3",
    }
    assert ledger.entries[0].payload["request"] == {
        "language": "catlass-dsl", "length": 32, "seed": 9,
        "dtype": "float32", "padded_length": 64,
    }

    @dataclass(frozen=True)
    class ProfileResult:
        block_count: int

    class Controller:
        request = None

        def run_final(self, verified, request):
            self.request = request
            return ProfileResult(request.block_count)

    controller = Controller()
    profile = CandidateProfileEvaluation(controller, device=3).final(run)
    assert profile == {"block_count": 6}
    assert controller.request.plan is run.plan


@pytest.mark.parametrize("family,evidence", [
    ("compile-recovery", RecoveryEvidence.COMPILE_FAILURE),
    ("runtime-recovery", RecoveryEvidence.HOST_VERIFICATION_FAILURE),
])
def test_recovery_contracts_supply_stable_abi_valid_starters(family, evidence):
    first = Phase1ProjectRuntime.from_proposal(_proposal(family))
    second = Phase1ProjectRuntime.from_proposal(_proposal(family))
    assert first.recovery is not None
    assert first.recovery == second.recovery
    assert first.recovery.required_evidence is evidence
    assert first.recovery.fault_count == 1
    validate_candidate_source(first.recovery.source)
    encoded = json.dumps(first.as_dict(), sort_keys=True)
    assert first.recovery.source not in encoded
    assert first.recovery.source_sha256 in encoded


@pytest.mark.parametrize("family,one_prefix,two_prefix,second_fault", [
    ("compile-recovery", "tla.copy(gm_c, missing_input",
     "tla.copy(gm_c, missing_input", "missing_input_second"),
    ("runtime-recovery", "tla.copy(gm_c, gm_a)",
     "tla.copy(gm_c[", "gm_c[32:64]"),
])
def test_recovery_source_encodes_requested_fault_count(
    family, one_prefix, two_prefix, second_fault
):
    def runtime(faults):
        proposal = CurriculumProposal.from_mapping({
            "family": family, "parameters": {"faults": faults},
            "hypothesis": f"{faults} bounded faults exercise recovery.",
            "evidence_preset": "ordinary-recovery",
        })
        return Phase1ProjectRuntime.from_proposal(proposal)

    one = runtime(1).recovery
    two = runtime(2).recovery
    assert one is not None and two is not None
    assert one.fault_count == one.source.count(one_prefix) == 1
    assert two.fault_count == two.source.count(two_prefix) == 2
    assert second_fault not in one.source and second_fault in two.source
    assert one.source != two.source
    assert one.source_sha256 != two.source_sha256
    validate_candidate_source(one.source)
    validate_candidate_source(two.source)


def test_two_runtime_faults_write_disjoint_output_regions():
    proposal = CurriculumProposal.from_mapping({
        "family": "runtime-recovery", "parameters": {"faults": 2},
        "hypothesis": "Two disjoint faults exercise recovery.",
        "evidence_preset": "ordinary-recovery",
    })
    recovery = Phase1ProjectRuntime.from_proposal(proposal).recovery

    assert recovery is not None
    assert "gm_c[0:32]" in recovery.source
    assert "gm_c[32:64]" in recovery.source
    assert "tla.copy(gm_c, " not in recovery.source


@pytest.mark.parametrize("options,message", [
    ({"length": True}, "length"),
    ({"length": 0}, "length"),
    ({"length": 401}, "length"),
    ({"padded_length": 63}, "padded_length"),
    ({"padded_length": 96}, "padded_length"),
    ({"padded_length": 512}, "padded_length"),
    ({"block_count": 0}, "block_count"),
    ({"block_count": True}, "block_count"),
])
def test_candidate_backend_rejects_values_outside_host_contract(options, message):
    with pytest.raises(ValueError, match=message):
        CatlassCandidateBackend(FakeExecution(), **options)


def test_candidate_backend_accepts_maximum_executable_fixture_extent():
    execution = FakeExecution()
    backend = CatlassCandidateBackend(execution, length=400, padded_length=448)
    request = backend._request()
    assert request.length == 400 and request.padded_length == 448
    assert execution.plans == []


@pytest.mark.parametrize("options", [
    {"length": 401, "padded_length": 448},
    {"length": 400, "padded_length": 512},
])
def test_candidate_backend_rejects_unexecutable_extent_before_dispatch(options):
    execution = FakeExecution()
    with pytest.raises(ValueError):
        CatlassCandidateBackend(execution, **options)
    assert execution.plans == []


@pytest.mark.parametrize("fields", [
    (33, 64, 1, None),
    (32, 128, 1, None),
    (32, 64, 2, None),
    (32, 64, 1, RecoveryStarter(HOST_FAILURE_STARTER,
                                RecoveryEvidence.HOST_VERIFICATION_FAILURE, 1)),
])
def test_direct_baseline_construction_rejects_derived_field_mismatch(fields):
    with pytest.raises(ValueError, match="exactly match"):
        Phase1ProjectRuntime(_proposal("vector-add-baseline"), *fields)


@pytest.mark.parametrize("recovery", [
    None,
    RecoveryStarter(HOST_FAILURE_STARTER, RecoveryEvidence.COMPILE_FAILURE, 1),
    RecoveryStarter(COMPILE_FAILURE_STARTER,
                    RecoveryEvidence.HOST_VERIFICATION_FAILURE, 1),
    RecoveryStarter(COMPILE_FAILURE_STARTER, RecoveryEvidence.COMPILE_FAILURE, 2),
])
def test_direct_recovery_construction_rejects_metadata_mismatch(recovery):
    with pytest.raises(ValueError, match="exactly match"):
        Phase1ProjectRuntime(_proposal("compile-recovery"), 32, 64, 1, recovery)


def test_recovery_templates_are_distinct_and_deterministic():
    assert COMPILE_FAILURE_STARTER != HOST_FAILURE_STARTER
    assert "missing_input" in COMPILE_FAILURE_STARTER
    assert "tla.copy(gm_c, gm_a)" in HOST_FAILURE_STARTER
