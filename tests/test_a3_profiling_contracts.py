from dataclasses import replace
import math

import pytest

from benchmarks.a3kernels.profiling import (
    A3ProfilingSession,
    CandidateBinding,
    CompactProfileResult,
    ProfileMetric,
    ProfileRequest,
    ProfilingTreatment,
    StudyDimensions,
    TimingRequest,
    TimingResult,
    parse_timing_output,
)


BINDING = CandidateBinding("a" * 64, "b" * 64)
DIMENSIONS = StudyDimensions(length=33, block_count=1, warm_up=5, launch_count=20)


def test_study_dimensions_and_ids_are_exact_and_stable():
    assert DIMENSIONS.as_dict() == {
        "length": 33,
        "block_count": 1,
        "warm_up": 5,
        "launch_count": 20,
    }
    first = ProfileRequest(BINDING, DIMENSIONS, ProfileMetric.PIPE_UTILIZATION)
    second = ProfileRequest(BINDING, DIMENSIONS, ProfileMetric.PIPE_UTILIZATION)
    assert first.request_id == second.request_id == (
        "0621eb7e4e99ad831eb4174b124df12f19beb8e4582849f9d109d76b703faf77"
    )
    assert first.default_replay_id == "a3-profile-0621eb7e4e99ad83"
    assert first.target == "Ascend910B4"
    assert first.language == "ascend-c"
    assert first.execution_profile == "bz-a3-1"


@pytest.mark.parametrize("profile", ["bz-a3-1", "bz-a3-2", "gz-a3"])
def test_profile_request_supports_authoritative_and_compatibility_profiles(profile):
    request = ProfileRequest(
        BINDING, DIMENSIONS, ProfileMetric.PIPE_UTILIZATION,
        execution_profile=profile,
    )
    assert request.execution_profile == profile
    assert request.as_dict()["execution_profile"] == profile
    timing = TimingRequest(BINDING, DIMENSIONS, execution_profile=profile)
    assert timing.execution_profile == profile
    assert timing.as_dict()["execution_profile"] == profile


def test_execution_profile_is_part_of_profile_and_timing_identity():
    first = ProfileRequest(BINDING, DIMENSIONS, ProfileMetric.BASIC)
    second = replace(first, execution_profile="bz-a3-2")
    compatibility = replace(first, execution_profile="gz-a3")
    assert len({first.request_id, second.request_id, compatibility.request_id}) == 3

    requests = []

    def timing(request):
        requests.append(request)
        return "A3TIMING_US=1\n"

    session = A3ProfilingSession(timing=timing)
    bz1 = session.time(BINDING, DIMENSIONS, execution_profile="bz-a3-1")
    bz2 = session.time(BINDING, DIMENSIONS, execution_profile="bz-a3-2")
    assert bz1.execution_profile == "bz-a3-1"
    assert bz2.execution_profile == "bz-a3-2"
    assert bz1.request_id != bz2.request_id
    assert [request.execution_profile for request in requests] == [
        "bz-a3-1", "bz-a3-2"
    ]
    assert all(isinstance(request, TimingRequest) for request in requests)

    replay_session = A3ProfilingSession(timing=timing)
    replay_session.time(
        BINDING, DIMENSIONS, execution_profile="bz-a3-1", replay_id="same-run"
    )
    with pytest.raises(ValueError, match="conflicting timing replay"):
        replay_session.time(
            BINDING, DIMENSIONS,
            execution_profile="bz-a3-2", replay_id="same-run",
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"length": 0}, {"length": 4097}, {"block_count": 0},
        {"block_count": 33}, {"warm_up": -1}, {"warm_up": 1001},
        {"launch_count": 0}, {"launch_count": 10001},
    ],
)
def test_study_dimension_bounds(changes):
    values = dict(length=33, block_count=1, warm_up=5, launch_count=20)
    values.update(changes)
    with pytest.raises(ValueError):
        StudyDimensions(**values)


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("A3TIMING_US=10\nA3TIMING_US=14\nA3TIMING_US=12\n", (10.0, 14.0, 12.0)),
        ("banner\nA3TIMING_US=0.125\n", (0.125,)),
    ],
)
def test_timing_parser_and_host_statistics(stdout, expected):
    samples = parse_timing_output(stdout)
    result = TimingResult.from_samples(TimingRequest(BINDING, DIMENSIONS), samples)
    assert samples == expected
    assert result.minimum_us == min(expected)
    assert result.maximum_us == max(expected)
    assert result.mean_us == pytest.approx(sum(expected) / len(expected))
    assert result.median_us == pytest.approx(12 if len(expected) == 3 else 0.125)
    assert result.evidence_sha256


@pytest.mark.parametrize(
    "stdout", ["", "A3TIMING_US=-1\n", "A3TIMING_US=nan\n", "A3TIMING_US=nope\n"]
)
def test_timing_parser_rejects_missing_or_nonfinite_samples(stdout):
    with pytest.raises(ValueError, match="timing"):
        parse_timing_output(stdout)


def test_compact_profile_schema_binds_candidate_and_request():
    request = ProfileRequest(BINDING, DIMENSIONS, ProfileMetric.PIPE_UTILIZATION)
    result = CompactProfileResult.create(
        request,
        exported_kernels=("vector_add",),
        metric_values=(("vector_ratio", 0.75), ("scalar_ratio", 0.25)),
        timeline=(("kernel_count", 20.0), ("duration_us", 240.0)),
        report_sha256="c" * 64,
    )
    assert result.request_id == request.request_id
    assert result.candidate_execution_id == BINDING.execution_id
    assert result.source_fingerprint == BINDING.source_fingerprint
    assert result.metric is ProfileMetric.PIPE_UTILIZATION
    assert result.evidence_sha256
    assert not hasattr(result, "report_tree")


@pytest.mark.parametrize(
    "field,value",
    [
        ("target", "Ascend950"),
        ("language", "catlass-dsl"),
        ("execution_profile", "bz-a5"),
        ("runtime", "a5-catlass"),
    ],
)
def test_request_rejects_a5_catlass_or_wrong_target(field, value):
    with pytest.raises(ValueError, match="A3 profiling"):
        replace(
            ProfileRequest(BINDING, DIMENSIONS, ProfileMetric.PIPE_UTILIZATION),
            **{field: value},
        )


def test_treatment_off_has_no_profiler_side_effect_and_on_uses_compact_result():
    calls = []

    def profiler(request):
        calls.append(request.request_id)
        return CompactProfileResult.create(
            request, exported_kernels=("vector_add",),
            metric_values=(("vector_ratio", 0.8),),
            timeline=(("kernel_count", 20.0),), report_sha256="d" * 64,
        )

    session = A3ProfilingSession(profiler=profiler)
    off = ProfileRequest(
        BINDING, DIMENSIONS, ProfileMetric.PIPE_UTILIZATION,
        treatment=ProfilingTreatment.OFF,
    )
    on = replace(off, treatment=ProfilingTreatment.ON)
    assert session.profile(off) is None
    assert calls == []
    assert session.profile(on).metric_values == (("vector_ratio", 0.8),)
    assert calls == [on.request_id]


def test_replay_returns_identical_evidence_and_rejects_conflicting_request():
    calls = []

    def profiler(request):
        calls.append(request.request_id)
        return CompactProfileResult.create(
            request, exported_kernels=("vector_add",),
            metric_values=(("vector_ratio", 0.5),),
            timeline=(("kernel_count", 20.0),), report_sha256="e" * 64,
        )

    session = A3ProfilingSession(profiler=profiler)
    request = ProfileRequest(BINDING, DIMENSIONS, ProfileMetric.PIPE_UTILIZATION)
    first = session.profile(request, replay_id="study-replay")
    assert session.profile(request, replay_id="study-replay") is first
    assert len(calls) == 1
    conflict = replace(request, dimensions=replace(DIMENSIONS, length=64))
    with pytest.raises(ValueError, match="conflicting profiling replay"):
        session.profile(conflict, replay_id="study-replay")


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": "f" * 64},
        {"candidate_execution_id": "f" * 64},
        {"source_fingerprint": "f" * 64},
        {"metric": ProfileMetric.BASIC},
        {"exported_kernels": ("wrong_kernel",)},
        {"metric_values": ()},
        {"timeline": ()},
        {"report_sha256": "not-a-digest"},
        {"evidence_sha256": "f" * 64},
    ],
)
def test_session_rejects_directly_constructed_malformed_profile_results(changes):
    request = ProfileRequest(BINDING, DIMENSIONS, ProfileMetric.PIPE_UTILIZATION)
    valid = CompactProfileResult.create(
        request,
        exported_kernels=("vector_add",),
        metric_values=(("vector_ratio", 0.5),),
        timeline=(("kernel_count", 20.0),),
        report_sha256="e" * 64,
    )
    malformed = replace(valid, **changes)
    session = A3ProfilingSession(profiler=lambda _request: malformed)
    with pytest.raises(ValueError, match="profiler result"):
        session.profile(request)


def test_fake_timing_backend_is_deterministic_and_bound_to_candidate():
    calls = []

    def timing(request):
        calls.append(request)
        return "A3TIMING_US=5\nA3TIMING_US=7\nA3TIMING_US=6\n"

    session = A3ProfilingSession(timing=timing)
    result = session.time(BINDING, DIMENSIONS, replay_id="timing-replay")
    assert result.median_us == 6
    assert result.candidate_execution_id == BINDING.execution_id
    assert result.execution_profile == "bz-a3-1"
    assert session.time(BINDING, DIMENSIONS, replay_id="timing-replay") is result
    assert calls == [TimingRequest(BINDING, DIMENSIONS)]
    with pytest.raises(ValueError, match="conflicting timing replay"):
        session.time(
            BINDING, replace(DIMENSIONS, length=64), replay_id="timing-replay"
        )
