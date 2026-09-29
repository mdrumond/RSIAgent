from dataclasses import replace

import pytest

from benchmarks.a5kernels.phase1_performance import Phase1PerformanceExecution
from benchmarks.a5kernels.phase1_registry import EvidencePreset
from benchmarks.a5kernels.profiling import (
    Phase1StudyResult,
    ProfileCapture,
    ProfileMetric,
    StudyPreset,
    StudyTimingSample,
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


@pytest.mark.parametrize(
    "preset", [EvidencePreset.CORRECTNESS, EvidencePreset.RECOVERY]
)
def test_non_performance_evidence_has_no_profile_side_effects(preset):
    backend = RecordingBackend()

    result = Phase1PerformanceExecution(backend).run(preset, [variant()])

    assert result is None
    assert backend.commands == []


def test_correctness_timing_maps_to_fixed_timing_study():
    backend = RecordingBackend()

    result = Phase1PerformanceExecution(backend).run(
        EvidencePreset.CORRECTNESS_TIMING, [variant()]
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
        EvidencePreset.MSPROF, [variant()]
    )

    assert result is not None
    assert result.preset is StudyPreset.PIPE_UTILIZATION
    assert [command.metric for command in backend.commands] == [
        ProfileMetric.BASIC_INFO,
        ProfileMetric.PIPE_UTILIZATION,
    ]
    assert all(command.request.warm_up == 0 for command in backend.commands)
    assert all(command.request.launch_count == 1 for command in backend.commands)


def test_dispatcher_rejects_unregistered_string_before_backend_use():
    backend = RecordingBackend()

    with pytest.raises(ValueError, match="host-owned EvidencePreset"):
        Phase1PerformanceExecution(backend).run("correctness-timing", [variant()])

    assert backend.commands == []


def test_dispatcher_preserves_study_correctness_gate():
    backend = RecordingBackend()
    candidate = variant()
    failed = replace(candidate, correctness=replace(candidate.correctness, passed=False))

    with pytest.raises(ValueError, match="correctness must pass"):
        Phase1PerformanceExecution(backend).run(
            EvidencePreset.CORRECTNESS_TIMING, [failed]
        )

    assert backend.commands == []
