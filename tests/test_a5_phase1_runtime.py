from __future__ import annotations

import json

import pytest

from benchmarks.a5kernels.candidate import (
    EXACT_KERNEL_NAME,
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
    assert run.plan.argv[:4] == (
        "env", "A5KERNEL_BLOCK_NUM=6", "env", "BZ_A5_PROFILE_PHYSICAL_DEVICE=3",
    )
    assert ledger.entries[0].payload["request"] == {
        "language": "catlass-dsl", "length": 32, "seed": 9,
        "dtype": "float32", "padded_length": 64,
    }


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


def test_recovery_fault_count_is_registry_owned():
    proposal = CurriculumProposal.from_mapping({
        "family": "compile-recovery", "parameters": {"faults": 2},
        "hypothesis": "Two bounded compiler corrections exercise recovery.",
        "evidence_preset": "ordinary-recovery",
    })
    runtime = Phase1ProjectRuntime.from_proposal(proposal)
    assert runtime.recovery is not None and runtime.recovery.fault_count == 2


@pytest.mark.parametrize("options,message", [
    ({"length": True}, "length"),
    ({"length": 401}, "length"),
    ({"padded_length": 63}, "padded_length"),
    ({"padded_length": 512}, "padded_length"),
    ({"block_count": 0}, "block_count"),
    ({"block_count": True}, "block_count"),
])
def test_candidate_backend_rejects_values_outside_host_contract(options, message):
    with pytest.raises(ValueError, match=message):
        CatlassCandidateBackend(FakeExecution(), **options)


def test_recovery_templates_are_distinct_and_deterministic():
    assert COMPILE_FAILURE_STARTER != HOST_FAILURE_STARTER
    assert "missing_input" in COMPILE_FAILURE_STARTER
    assert "tla.copy(gm_c, gm_a)" in HOST_FAILURE_STARTER
