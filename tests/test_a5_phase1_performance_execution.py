from dataclasses import replace

import pytest

from benchmarks.a5kernels.phase1_performance import Phase1PerformanceExecution
from benchmarks.a5kernels.phase1_registry import CurriculumProposal
from benchmarks.a5kernels.profiling import (
    ParallelismClass,
    Phase1StudyResult,
    ProfileCapture,
    ProfileMetric,
    StudyPreset,
    StudyTimingSample,
    bind_profile_parallelism,
    bind_study_dimensions,
)
from tests.test_a5_phase1_performance import SHA, variant
from tests.test_a5kernels_profiling import archive


class RecordingBackend:
    def __init__(self) -> None:
        self.commands = []

    def time_sample(self, command):
        self.commands.append(command)
        return StudyTimingSample(
            8.0,
            command.request.source_fingerprint,
            command.request.execution_id,
            command.replay_id,
            SHA,
        )

    def capture(self, command):
        self.commands.append(command)
        summary = (("metric", "value"),)
        return ProfileCapture(
            command.metric,
            command.request.source_fingerprint,
            command.request.execution_id,
            command.replay_id,
            (command.request.expected_kernel,),
            summary,
            archive(command.metric.value),
        )


def proposal(family, parameters, evidence_preset):
    return CurriculumProposal.from_mapping({
        "family": family,
        "parameters": parameters,
        "hypothesis": "Exercise the registered project.",
        "evidence_preset": evidence_preset,
    })


def block_variant(block_count):
    candidate = variant()
    dimensions = replace(
        candidate.dimensions, parallelism=ParallelismClass(block_count)
    )
    plan = replace(
        candidate.request.plan,
        runtime_provenance=tuple(
            item
            for item in candidate.request.plan.runtime_provenance
            if not item[0].startswith("phase1.")
        ),
    )
    plan = bind_study_dimensions(
        bind_profile_parallelism(plan, ParallelismClass(block_count)), dimensions
    )
    request = replace(candidate.request, plan=plan)
    correctness = replace(
        candidate.correctness,
        execution_id=request.execution_id,
        source_fingerprint=request.source_fingerprint,
    )
    return replace(
        candidate,
        request=request,
        correctness=correctness,
        dimensions=dimensions,
    )

@pytest.mark.parametrize("registered", [
    proposal("vector-add-baseline", {"length": 32}, "correctness"),
    proposal("compile-recovery", {"faults": 1}, "ordinary-recovery"),
])
def test_non_performance_evidence_has_no_profile_side_effects(registered):
    backend = RecordingBackend()

    result = Phase1PerformanceExecution(backend).run(registered, [variant()])

    assert result is None
    assert backend.commands == []


def test_padded_non_shape_length_is_noop_without_variant_validation():
    backend = RecordingBackend()
    padded = proposal("padded-multitile", {"length": 96}, "correctness")

    result = Phase1PerformanceExecution(backend).run(
        padded, ["correctness-runtime-owns-this-value"]
    )

    assert result is None
    assert backend.commands == []


def test_correctness_timing_maps_to_fixed_timing_study():
    backend = RecordingBackend()

    result = Phase1PerformanceExecution(backend).run(
        proposal("length-knee", {"length": 32}, "correctness-timing"),
        [variant()],
    )

    assert isinstance(result, Phase1StudyResult)
    assert result.preset is StudyPreset.TIMING_STUDY
    assert result.timing[0].raw_samples_us == (8.0, 8.0, 8.0)
    assert len(backend.commands) == 3
    assert all(command.request.warm_up == 5 for command in backend.commands)
    assert all(command.request.launch_count == 20 for command in backend.commands)


def test_msprof_maps_to_separate_basic_and_pipe_captures():
    backend = RecordingBackend()

    result = Phase1PerformanceExecution(backend).run(
        proposal("msprof-pipe", {"metric": "PipeUtilization"}, "msprof-guided"),
        [variant()],
    )

    assert result is not None
    assert result.preset is StudyPreset.PIPE_UTILIZATION
    assert [command.metric for command in backend.commands] == [
        ProfileMetric.BASIC_INFO,
        ProfileMetric.PIPE_UTILIZATION,
    ]
    assert all(command.request.warm_up == 0 for command in backend.commands)
    assert all(command.request.launch_count == 1 for command in backend.commands)


def test_dispatcher_rejects_unvalidated_proposal_before_backend_use():
    backend = RecordingBackend()

    with pytest.raises(ValueError, match="validated CurriculumProposal"):
        Phase1PerformanceExecution(backend).run("correctness-timing", [variant()])

    assert backend.commands == []


def test_dispatcher_preserves_study_correctness_gate():
    backend = RecordingBackend()
    candidate = variant()
    failed = replace(candidate, correctness=replace(candidate.correctness, passed=False))

    with pytest.raises(ValueError, match="correctness must pass"):
        Phase1PerformanceExecution(backend).run(
            proposal("length-knee", {"length": 32}, "correctness-timing"),
            [failed],
        )

    assert backend.commands == []


def test_length_knee_rejects_variant_for_different_length_before_dispatch():
    backend = RecordingBackend()

    with pytest.raises(ValueError, match="shape does not match proposal length"):
        Phase1PerformanceExecution(backend).run(
            proposal("length-knee", {"length": 128}, "correctness-timing"),
            [variant()],
        )

    assert backend.commands == []


def test_unsupported_knee_length_is_rejected_before_backend_dispatch():
    backend = RecordingBackend()

    with pytest.raises(ValueError, match="must be one of"):
        unsupported = proposal("length-knee", {"length": 96}, "correctness-timing")
        Phase1PerformanceExecution(backend).run(unsupported, [variant()])

    assert backend.commands == []


def test_cross_layer_rejects_variant_for_different_block_count():
    backend = RecordingBackend()

    with pytest.raises(ValueError, match="proposal block_count"):
        Phase1PerformanceExecution(backend).run(
            proposal("cross-layer-launch", {"block_count": 6}, "correctness-timing"),
            [variant()],
        )

    assert backend.commands == []


def test_cross_layer_registered_block_count_dispatches_fixed_timing():
    backend = RecordingBackend()

    result = Phase1PerformanceExecution(backend).run(
        proposal("cross-layer-launch", {"block_count": 6}, "correctness-timing"),
        [block_variant(6)],
    )

    assert result is not None
    assert result.preset is StudyPreset.TIMING_STUDY
    assert len(backend.commands) == 3
    assert all(
        command.request.plan.argv.count("A5KERNEL_BLOCK_NUM=6") == 1
        for command in backend.commands
    )
