from pathlib import Path

import pytest

from benchmarks.a3kernels.phase1_memory import HostFact
from benchmarks.a3kernels.phase1_protocol import FailedEvidence, VerifiedResult
from benchmarks.a3kernels.phase1_registry import (
    DEFAULT_PROPOSALS,
    CurriculumProposal,
)
from benchmarks.a3kernels.project_execution import (
    PerformancePreset,
    ProjectDispatcher,
    ProjectRuntimePolicy,
    RecoveryEvidence,
)
from benchmarks.a3kernels.profiling import (
    A3ProfilingSession,
    CompactProfileResult,
)


SOURCE = '''
extern "C" __global__ __aicore__ void vector_add(
    GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output,
    uint32_t count, uint32_t buffer_bytes) {}
'''


def proposal(family, occurrence=0):
    return [item for item in DEFAULT_PROPOSALS if item.family.value == family][occurrence]


@pytest.mark.parametrize(
    "family,occurrence,length,padded,blocks,preset",
    [
        ("vector-add-baseline", 0, 32, 64, 1, PerformancePreset.NONE),
        ("padded-multitile", 0, 400, 448, 1, PerformancePreset.NONE),
        ("compile-recovery", 0, 32, 64, 1, PerformancePreset.NONE),
        ("runtime-recovery", 0, 32, 64, 1, PerformancePreset.NONE),
        ("length-knee", 0, 16, 64, 1, PerformancePreset.TIMING),
        ("length-knee", 1, 400, 448, 1, PerformancePreset.TIMING),
        ("cross-layer-launch", 0, 32, 64, 1, PerformancePreset.TIMING),
        ("msprof-pipe", 0, 32, 64, 1, PerformancePreset.PIPE),
    ],
)
def test_all_eight_projects_resolve_fixed_host_policy(
    family, occurrence, length, padded, blocks, preset
):
    policy = ProjectRuntimePolicy.from_proposal(proposal(family, occurrence))
    assert (policy.logical_length, policy.padded_length) == (length, padded)
    assert policy.block_count == blocks
    assert policy.performance_preset is preset
    assert policy.target == "Ascend910B4" and policy.language == "ascend-c"


@pytest.mark.parametrize(
    "family,evidence",
    [
        ("compile-recovery", RecoveryEvidence.COMPILE_FAILURE),
        ("runtime-recovery", RecoveryEvidence.HOST_VERIFICATION_FAILURE),
    ],
)
def test_recovery_starters_are_deterministic_ascend_c(family, evidence):
    first = ProjectRuntimePolicy.from_proposal(proposal(family))
    second = ProjectRuntimePolicy.from_proposal(proposal(family))
    assert first.recovery == second.recovery
    assert first.recovery.required_evidence is evidence
    assert first.recovery.source_sha256 == second.recovery.source_sha256
    assert 'extern "C" __global__ __aicore__ void vector_add' in first.recovery.source
    assert "catlass" not in first.recovery.source.lower()
    assert first.as_dict()["recovery"]["starter_sha256"] == first.recovery.source_sha256
    assert first.recovery.source not in str(first.as_dict())


def test_direct_policy_cannot_override_registry_derived_values():
    valid = ProjectRuntimePolicy.from_proposal(proposal("vector-add-baseline"))
    with pytest.raises(ValueError, match="exactly match"):
        ProjectRuntimePolicy(
            valid.proposal,
            33,
            valid.padded_length,
            valid.block_count,
            valid.performance_preset,
            valid.recovery,
        )


class FakeCandidateBackend:
    def __init__(self, *, fail_stage=None):
        self.calls = []
        self.fail_stage = fail_stage

    def run(self, source, workdir, **options):
        self.calls.append((source, Path(workdir), options))
        plan = type("Plan", (), {
            "request_id": options["request_id"],
            "execution_id": "a" * 64,
            "attempt_id": options["attempt_id"],
            "project_id": options["project_id"],
            "source_fingerprint": "b" * 64,
        })()
        if self.fail_stage:
            return FailedEvidence.create(
                plan, stage=self.fail_stage, error_type="ExpectedFailure", detail="fixture"
            )
        return VerifiedResult(
            plan.request_id,
            plan.execution_id,
            plan.attempt_id,
            plan.project_id,
            True,
            0.0,
            1e-5,
            0,
            "c" * 64,
            plan.source_fingerprint,
            "d" * 64,
            None,
            "e" * 64,
        )


@pytest.mark.parametrize("proposal", DEFAULT_PROPOSALS)
def test_every_project_executes_its_resolved_shape_and_launch_dimensions(
    tmp_path, proposal
):
    backend = FakeCandidateBackend()
    policy = ProjectRuntimePolicy.from_proposal(proposal)
    ProjectDispatcher(backend, profiling_session([])).execute(
        proposal, SOURCE, tmp_path / proposal.project_id,
        request_id="request", attempt_id="attempt",
    )

    options = backend.calls[0][2]
    assert options["length"] == policy.logical_length
    assert options["padded_length"] == policy.padded_length
    assert options["block_count"] == policy.block_count


def profiling_session(calls):
    def timing(binding, dimensions):
        calls.append(("timing", binding, dimensions))
        return "A3TIMING_US=5\nA3TIMING_US=7\nA3TIMING_US=6\n"

    def profiler(request):
        calls.append(("profile", request.binding, request.dimensions))
        return CompactProfileResult.create(
            request,
            exported_kernels=("vector_add",),
            metric_values=(("vector_ratio", 0.75),),
            timeline=(("kernel_count", 20.0),),
            report_sha256="f" * 64,
        )

    return A3ProfilingSession(timing=timing, profiler=profiler)


def test_timing_dispatch_binds_exact_verified_candidate_and_memory_facts(tmp_path):
    backend = FakeCandidateBackend()
    calls = []
    outcome = ProjectDispatcher(backend, profiling_session(calls)).execute(
        proposal("length-knee"), SOURCE, tmp_path, request_id="request", attempt_id="attempt"
    )

    assert outcome.verified.passed
    assert outcome.timing.candidate_execution_id == outcome.verified.execution_id
    assert outcome.timing.source_fingerprint == outcome.verified.source_fingerprint
    assert outcome.profile is None
    assert calls[0][0] == "timing"
    assert all(isinstance(item, HostFact) for item in outcome.host_facts())
    assert {item.category for item in outcome.host_facts()} == {
        "host-verification",
        "profiling",
    }


def test_msprof_dispatch_uses_registered_pipe_preset(tmp_path):
    calls = []
    outcome = ProjectDispatcher(
        FakeCandidateBackend(), profiling_session(calls)
    ).execute(
        proposal("msprof-pipe"), SOURCE, tmp_path,
        request_id="request", attempt_id="attempt",
    )
    assert outcome.timing is None
    assert outcome.profile.metric.value == "PipeUtilization"
    assert calls[0][0] == "profile"


def test_correctness_project_has_no_performance_side_effect(tmp_path):
    calls = []
    outcome = ProjectDispatcher(
        FakeCandidateBackend(), profiling_session(calls)
    ).execute(
        proposal("vector-add-baseline"), SOURCE, tmp_path,
        request_id="request", attempt_id="attempt",
    )
    assert outcome.timing is outcome.profile is None
    assert calls == []


def test_failed_candidate_stops_before_performance_and_retains_failure_fact(tmp_path):
    calls = []
    outcome = ProjectDispatcher(
        FakeCandidateBackend(fail_stage="compile"), profiling_session(calls)
    ).execute(
        proposal("length-knee"), SOURCE, tmp_path,
        request_id="request", attempt_id="attempt",
    )
    assert outcome.verified is None and outcome.failure.stage == "compile"
    assert calls == []
    assert outcome.host_facts()[0].category == "compile"


def test_same_replay_id_rejects_different_candidate_binding(tmp_path):
    calls = []
    session = profiling_session(calls)
    dispatcher = ProjectDispatcher(FakeCandidateBackend(), session)
    dispatcher.execute(
        proposal("length-knee"), SOURCE, tmp_path / "one",
        request_id="one", attempt_id="one", replay_id="fixed-replay",
    )
    backend = FakeCandidateBackend()
    original = backend.run

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        return VerifiedResult(
            result.request_id, "1" * 64, result.attempt_id, result.project_id,
            result.passed, result.max_abs_error, result.tolerance, result.exit_code,
            result.output_sha256, "2" * 64, result.evidence_sha256,
            result.job_handle, result.attestation_sha256,
        )

    backend.run = changed
    with pytest.raises(ValueError, match="conflicting timing replay"):
        ProjectDispatcher(backend, session).execute(
            proposal("length-knee"), SOURCE, tmp_path / "two",
            request_id="two", attempt_id="two", replay_id="fixed-replay",
        )


def test_custom_proposal_controls_only_registered_typed_dimension():
    custom = CurriculumProposal.from_mapping({
        "target": "Ascend910B4",
        "language": "ascend-c",
        "family": "cross-layer-launch",
        "parameters": {"block_count": 6},
        "hypothesis": "Six host-selected blocks sample the launch boundary.",
        "evidence_preset": "correctness-timing",
    })
    policy = ProjectRuntimePolicy.from_proposal(custom)
    assert policy.block_count == 6
    assert policy.logical_length == 32 and policy.padded_length == 64
