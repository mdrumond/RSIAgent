from dataclasses import replace

import pytest

from benchmarks.a5kernels.profiling import (
    AccessClass,
    MetricDomain,
    PaddingClass,
    ParallelismClass,
    Phase1PerformanceStudy,
    ProfileCapture,
    ProfileMetric,
    ShapeClass,
    StudyDimensions,
    StudyPreset,
    StudyTimingSample,
    StudyVariant,
    bind_study_dimensions,
    bind_profile_parallelism,
    bind_profile_device,
)
from tests.test_a5kernels_profiling import (
    CORRECT,
    REQUEST,
    SOURCE,
    archive,
)


SHA = "b" * 64


def dimensions(parallelism=ParallelismClass.ONE):
    return StudyDimensions(
        ShapeClass.N32,
        PaddingClass.ALIGN_64,
        AccessClass.CONTIGUOUS,
        parallelism,
    )


def variant(*, attempt="attempt-1", revision="baseline", device=REQUEST.device):
    parallelism = ParallelismClass.ONE
    study_dimensions = dimensions(parallelism)
    values_a = (*([1.0] * 32), *([0.0] * 32))
    values_b = (*([2.0] * 32), *([0.0] * 32))
    plan = bind_study_dimensions(
        bind_profile_device(replace(
            REQUEST.plan,
            attempt_id=attempt,
            files=(replace(REQUEST.plan.files[0], content=f"# {revision}\n"),),
            input_a=values_a,
            input_b=values_b,
        ), device),
        study_dimensions,
    )
    request = replace(
        REQUEST,
        plan=plan,
        device=device,
    )
    correctness = replace(
        CORRECT,
        attempt_id=request.attempt_id,
        execution_id=request.execution_id,
        source_fingerprint=request.source_fingerprint,
        request_id=request.request_id,
    )
    return StudyVariant(request, correctness, study_dimensions)


class FakeStudyBackend:
    def __init__(self, timings=(10.0, 12.0, 11.0)):
        self.timings = iter(timings)
        self.commands = []

    def time_sample(self, command):
        self.commands.append(command)
        return StudyTimingSample(
            next(self.timings),
            command.request.source_fingerprint,
            command.request.execution_id,
            command.replay_id,
            SHA,
        )

    def capture(self, command):
        self.commands.append(command)
        summary = (
            (("frequency_mhz", "1800"),)
            if command.metric is ProfileMetric.BASIC_INFO
            else (("vector_ratio", "0.75"),)
        )
        return ProfileCapture(
            command.metric,
            command.request.source_fingerprint,
            command.request.execution_id,
            command.replay_id,
            (command.request.expected_kernel,),
            summary,
            archive(command.metric.value),
        )


def test_timing_study_is_three_fixed_minimally_instrumented_samples():
    backend = FakeStudyBackend()
    result = Phase1PerformanceStudy(backend).run(
        StudyPreset.TIMING_STUDY, [variant()]
    )

    timing = result.timing[0]
    assert timing.raw_samples_us == (10.0, 12.0, 11.0)
    assert timing.median_us == 11.0
    assert timing.coefficient_of_variation == pytest.approx(0.0742269)
    assert timing.evidence_sha256 == (SHA, SHA, SHA)
    assert len({command.replay_id for command in backend.commands}) == 3
    assert all(command.request.warm_up == 5 for command in backend.commands)
    assert all(command.request.launch_count == 20 for command in backend.commands)
    assert all(command.profiler_enabled is False for command in backend.commands)


def test_preset_ignores_caller_launch_flags_for_identity_and_commands():
    normal = variant()
    caller_flags = replace(
        normal,
        request=replace(normal.request, warm_up=99, launch_count=999),
    )
    backend = FakeStudyBackend()

    Phase1PerformanceStudy(backend).run(StudyPreset.TIMING_STUDY, [caller_flags])

    assert caller_flags.variant_id == normal.variant_id
    assert all(command.request.warm_up == 5 for command in backend.commands)
    assert all(command.request.launch_count == 20 for command in backend.commands)


def test_basic_and_pipe_presets_keep_metric_domains_in_separate_replays():
    basic_backend = FakeStudyBackend()
    basic = Phase1PerformanceStudy(basic_backend).run(
        StudyPreset.BASIC_INFO, [variant()]
    )
    assert [item.domain for item in basic.metrics] == [MetricDomain.BASIC_INFO]
    assert basic_backend.commands[0].kernel_name is None

    pipe_backend = FakeStudyBackend()
    pipe = Phase1PerformanceStudy(pipe_backend).run(
        StudyPreset.PIPE_UTILIZATION, [variant()]
    )
    assert [item.domain for item in pipe.metrics] == [
        MetricDomain.BASIC_INFO,
        MetricDomain.PIPE_UTILIZATION,
    ]
    assert [command.metric for command in pipe_backend.commands] == [
        ProfileMetric.BASIC_INFO,
        ProfileMetric.PIPE_UTILIZATION,
    ]
    assert pipe_backend.commands[1].kernel_name == REQUEST.expected_kernel
    assert pipe.metrics[0].evidence_manifest_sha256 != ""
    assert pipe.metrics[1].evidence_manifest_sha256 != ""


def test_optimization_comparison_preserves_synthetic_parallelism_knee():
    variants = [
        variant(attempt="baseline", revision="baseline"),
        variant(attempt="optimization-1", revision="optimization-1"),
        variant(attempt="optimization-2", revision="optimization-2"),
    ]
    # Improvement is large through p4 and nearly flat at p8: retain raw results
    # and host-computed speedups rather than asking an agent for measured values.
    backend = FakeStudyBackend((100, 101, 99, 55, 54, 56, 53, 54, 52))
    result = Phase1PerformanceStudy(backend).run(
        StudyPreset.OPTIMIZATION_COMPARISON, variants
    )

    assert [item.median_us for item in result.timing] == [100, 55, 53]
    assert result.comparison is not None
    assert result.comparison.baseline_variant_id == variants[0].variant_id
    assert dict(result.comparison.speedup_by_variant) == pytest.approx({
        variants[0].variant_id: 1.0,
        variants[1].variant_id: 100 / 55,
        variants[2].variant_id: 100 / 53,
    })


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shape", "n32"),
        ("padding", "align-64"),
        ("access", "contiguous"),
        ("parallelism", 28),
    ],
)
def test_dimensions_reject_agent_selected_raw_values(field, value):
    values = {
        "shape": ShapeClass.N32,
        "padding": PaddingClass.ALIGN_64,
        "access": AccessClass.CONTIGUOUS,
        "parallelism": ParallelismClass.ONE,
    }
    values[field] = value
    with pytest.raises(ValueError, match="host-owned"):
        StudyDimensions(**values)


def test_failed_correctness_stops_before_study_side_effects():
    backend = FakeStudyBackend()
    failed = replace(variant(), correctness=replace(CORRECT, passed=False))

    with pytest.raises(ValueError, match="correctness must pass"):
        Phase1PerformanceStudy(backend).run(StudyPreset.TIMING_STUDY, [failed])
    assert backend.commands == []


@pytest.mark.parametrize("preset", list(StudyPreset))
def test_presets_reject_unbounded_or_ambiguous_variant_counts(preset):
    study = Phase1PerformanceStudy(FakeStudyBackend())
    with pytest.raises(ValueError):
        study.run(preset, [])
    if preset is StudyPreset.OPTIMIZATION_COMPARISON:
        with pytest.raises(ValueError, match="at least two"):
            study.run(preset, [variant()])
    else:
        with pytest.raises(ValueError, match="exactly one"):
            study.run(preset, [variant(attempt="one"), variant(attempt="two")])


def test_backend_cannot_return_stale_identity_or_unhashed_timing():
    class StaleBackend(FakeStudyBackend):
        def time_sample(self, command):
            return replace(super().time_sample(command), execution_id="stale", evidence_sha256="")

    with pytest.raises(ValueError, match="exact study replay"):
        Phase1PerformanceStudy(StaleBackend()).run(
            StudyPreset.TIMING_STUDY, [variant()]
        )


def test_comparison_rejects_mixed_devices_before_measurement():
    backend = FakeStudyBackend()
    first = variant(attempt="device-3")
    second = variant(attempt="device-4", revision="device-4", device=4)

    with pytest.raises(ValueError, match="share device, kernel, and implementation"):
        Phase1PerformanceStudy(backend).run(
            StudyPreset.OPTIMIZATION_COMPARISON, [first, second]
        )
    assert backend.commands == []


@pytest.mark.parametrize("encoded", ["4", "03"])
def test_profile_request_rejects_mismatched_or_noncanonical_plan_device(encoded):
    plan = replace(
        REQUEST.plan,
        environment=REQUEST.plan.environment.with_binding(
            "BZ_A5_PROFILE_PHYSICAL_DEVICE", encoded
        ),
    )
    with pytest.raises(ValueError, match="device bound in the plan environment"):
        replace(REQUEST, plan=plan)


def test_phase1_variant_rejects_missing_plan_device_binding():
    bound = variant()
    unbound_plan = replace(
        bound.request.plan,
        environment=bound.request.plan.environment.with_unset(
            "BZ_A5_PROFILE_PHYSICAL_DEVICE"
        ),
    )
    request = replace(bound.request, plan=unbound_plan)
    correctness = replace(bound.correctness, execution_id=request.execution_id)
    with pytest.raises(ValueError, match="device-bound execution plan"):
        StudyVariant(request, correctness, bound.dimensions)


@pytest.mark.parametrize(
    "assignment",
    [
        ("A5KERNEL_EMIT_TIMING", "0"),
        ("A5KERNEL_WARM_UP", "0"),
        ("A5KERNEL_LAUNCH_COUNT", "1"),
    ],
)
def test_profile_request_rejects_plan_timing_policy_overrides(assignment):
    plan = replace(
        REQUEST.plan,
        environment=REQUEST.plan.environment.with_binding(*assignment),
    )
    with pytest.raises(ValueError, match="host-owned timing policy"):
        replace(REQUEST, plan=plan)


@pytest.mark.parametrize("encoded", ["0", "9", "01", "+1", " 1"])
def test_profile_request_rejects_noncanonical_plan_block_binding(encoded):
    plan = replace(
        REQUEST.plan,
        environment=REQUEST.plan.environment.with_binding(
            "A5KERNEL_BLOCK_NUM", encoded
        ),
    )
    with pytest.raises(ValueError, match="canonical integer"):
        replace(REQUEST, plan=plan)


def test_profile_request_does_not_parse_assignment_shaped_positional_arguments():
    plan = replace(
        REQUEST.plan,
        argv=("python", "tool.py", "A5KERNEL_BLOCK_NUM=not-environment"),
        environment=REQUEST.plan.environment.with_binding(
            "A5KERNEL_BLOCK_NUM", "1"
        ),
    )

    request = replace(REQUEST, plan=plan)

    assert request.workload_argv[-1] == "A5KERNEL_BLOCK_NUM=not-environment"
    assert request.block_count == 1


def test_phase1_variant_rejects_block_count_mismatched_with_parallelism():
    bound = variant()
    plan = replace(
        bound.request.plan,
        environment=bound.request.plan.environment.with_binding(
            "A5KERNEL_BLOCK_NUM", "2"
        ),
    )
    with pytest.raises(ValueError, match="Phase 1 dimensions"):
        replace(bound.request, plan=plan)


@pytest.mark.parametrize("encoded", ["0", "9", "06"])
def test_profile_request_rejects_noncanonical_block_bindings(encoded):
    plan = replace(
        REQUEST.plan,
        environment=REQUEST.plan.environment.with_binding(
            "A5KERNEL_BLOCK_NUM", encoded
        ),
    )
    with pytest.raises(ValueError, match="invalid host-owned block binding"):
        replace(REQUEST, plan=plan)


def test_registered_n400_align64_variant_uses_physical_448_inputs():
    dimensions = StudyDimensions(
        ShapeClass.N400,
        PaddingClass.ALIGN_64,
        AccessClass.CONTIGUOUS,
        ParallelismClass.ONE,
    )
    plan = bind_study_dimensions(
        bind_profile_device(
            replace(
                REQUEST.plan,
                input_a=(*([1.0] * 400), *([0.0] * 48)),
                input_b=(*([2.0] * 400), *([0.0] * 48)),
            ),
            REQUEST.device,
        ),
        dimensions,
    )
    request = replace(REQUEST, plan=plan)
    correctness = replace(
        CORRECT,
        execution_id=request.execution_id,
        source_fingerprint=request.source_fingerprint,
    )

    registered = StudyVariant(request, correctness, dimensions)

    assert registered.dimensions.shape is ShapeClass.N400
    assert len(registered.request.plan.input_a) == 448


def test_registered_block_count_six_is_bound_to_variant_plan():
    dimensions = StudyDimensions(
        ShapeClass.N64,
        PaddingClass.NONE,
        AccessClass.CONTIGUOUS,
        ParallelismClass.SIX,
    )
    plan = bind_study_dimensions(
        bind_profile_parallelism(
            bind_profile_device(
                replace(
                    REQUEST.plan,
                    input_a=tuple([1.0] * 64),
                    input_b=tuple([2.0] * 64),
                ),
                REQUEST.device,
            ),
            ParallelismClass.SIX,
        ),
        dimensions,
    )
    request = replace(REQUEST, plan=plan)
    correctness = replace(
        CORRECT,
        execution_id=request.execution_id,
        source_fingerprint=request.source_fingerprint,
    )

    registered = StudyVariant(request, correctness, dimensions)

    assert (
        dict(registered.request.plan.environment.bindings)["A5KERNEL_BLOCK_NUM"]
        == "6"
    )


def test_bind_study_dimensions_sets_matching_block_binding():
    dimensions = StudyDimensions(
        ShapeClass.N64,
        PaddingClass.NONE,
        AccessClass.CONTIGUOUS,
        ParallelismClass.SIX,
    )
    plan = bind_study_dimensions(
        bind_profile_device(
            replace(
                REQUEST.plan,
                input_a=tuple([1.0] * 64),
                input_b=tuple([2.0] * 64),
            ),
            REQUEST.device,
        ),
        dimensions,
    )
    assert dict(plan.environment.bindings)["A5KERNEL_BLOCK_NUM"] == "6"
    replace(REQUEST, plan=plan)


def test_profile_request_treats_assignment_shaped_positional_argument_as_argv():
    dimensions = StudyDimensions(
        ShapeClass.N64,
        PaddingClass.NONE,
        AccessClass.CONTIGUOUS,
        ParallelismClass.SIX,
    )
    plan = bind_study_dimensions(
        bind_profile_device(
            replace(
                REQUEST.plan,
                argv=(*REQUEST.plan.argv, "A5KERNEL_BLOCK_NUM=6"),
                input_a=tuple([1.0] * 64),
                input_b=tuple([2.0] * 64),
            ),
            REQUEST.device,
        ),
        dimensions,
    )
    assert plan.argv[-1] == "A5KERNEL_BLOCK_NUM=6"
    assert dict(plan.environment.bindings)["A5KERNEL_BLOCK_NUM"] == "6"
    replace(REQUEST, plan=plan)


def test_optimization_comparison_caps_variants_before_measurement():
    backend = FakeStudyBackend()
    variants = [
        variant(attempt=f"variant-{index}", revision=f"revision-{index}")
        for index in range(9)
    ]
    with pytest.raises(ValueError, match="at most 8 variants"):
        Phase1PerformanceStudy(backend).run(
            StudyPreset.OPTIMIZATION_COMPARISON, variants
        )
    assert backend.commands == []


def test_variant_rejects_dimensions_not_bound_to_execution_plan():
    bound = variant()
    with pytest.raises(ValueError, match="execution-plan provenance"):
        StudyVariant(
            bound.request,
            bound.correctness,
            StudyDimensions(
                ShapeClass.N64,
                PaddingClass.NONE,
                AccessClass.CONTIGUOUS,
                ParallelismClass.ONE,
            ),
        )


def test_comparison_rejects_mismatched_workloads_before_measurement():
    backend = FakeStudyBackend()
    first = variant(attempt="n32")
    other_dimensions = StudyDimensions(
        ShapeClass.N64,
        PaddingClass.NONE,
        AccessClass.CONTIGUOUS,
        ParallelismClass.ONE,
    )
    other_plan = bind_study_dimensions(
        replace(
            bind_profile_device(REQUEST.plan, REQUEST.device),
            attempt_id="n64",
            input_a=tuple([1.0] * 64),
            input_b=tuple([2.0] * 64),
        ),
        other_dimensions,
    )
    other_request = replace(first.request, plan=other_plan)
    other_correctness = replace(
        first.correctness,
        attempt_id=other_request.attempt_id,
        execution_id=other_request.execution_id,
        source_fingerprint=other_request.source_fingerprint,
        request_id=other_request.request_id,
    )
    second = StudyVariant(other_request, other_correctness, other_dimensions)

    with pytest.raises(ValueError, match="share shape, padding, and access"):
        Phase1PerformanceStudy(backend).run(
            StudyPreset.OPTIMIZATION_COMPARISON, [first, second]
        )
    assert backend.commands == []


def test_comparison_rejects_relabelled_duplicate_execution():
    backend = FakeStudyBackend()
    first = variant(attempt="first")
    retry_request = replace(
        first.request,
        plan=replace(first.request.plan, attempt_id="retry"),
    )
    retry_correctness = replace(first.correctness, attempt_id="retry")
    retry = StudyVariant(retry_request, retry_correctness, first.dimensions)

    with pytest.raises(ValueError, match="distinct execution identities"):
        Phase1PerformanceStudy(backend).run(
            StudyPreset.OPTIMIZATION_COMPARISON, [first, retry]
        )
    assert backend.commands == []


def test_comparison_rejects_different_input_values():
    backend = FakeStudyBackend()
    first = variant(attempt="input-a", revision="input-a")
    second = variant(attempt="input-b", revision="input-b")
    changed_plan = replace(
        second.request.plan,
        input_a=(3.0, *second.request.plan.input_a[1:]),
    )
    changed_request = replace(second.request, plan=changed_plan)
    changed_correctness = replace(
        second.correctness,
        execution_id=changed_request.execution_id,
        source_fingerprint=changed_request.source_fingerprint,
    )
    second = StudyVariant(changed_request, changed_correctness, second.dimensions)

    with pytest.raises(ValueError, match="exact request and inputs"):
        Phase1PerformanceStudy(backend).run(
            StudyPreset.OPTIMIZATION_COMPARISON, [first, second]
        )
    assert backend.commands == []


@pytest.mark.parametrize(
    "bad_dimensions",
    [
        StudyDimensions(
            ShapeClass.N128,
            PaddingClass.NONE,
            AccessClass.CONTIGUOUS,
            ParallelismClass.ONE,
        ),
        StudyDimensions(
            ShapeClass.N32,
            PaddingClass.NONE,
            AccessClass.CONTIGUOUS,
            ParallelismClass.ONE,
        ),
        StudyDimensions(
            ShapeClass.N32,
            PaddingClass.ALIGN_64,
            AccessClass.STRIDED_2,
            ParallelismClass.ONE,
        ),
    ],
)
def test_binding_rejects_mislabeled_shape_padding_or_access(bad_dimensions):
    padded_n32 = replace(
        REQUEST.plan,
        input_a=(*([1.0] * 32), *([0.0] * 32)),
        input_b=(*([2.0] * 32), *([0.0] * 32)),
    )
    with pytest.raises(ValueError):
        bind_study_dimensions(padded_n32, bad_dimensions)


def test_binding_rejects_n32_none_because_runtime_would_pad_it():
    unpadded_n32 = replace(
        REQUEST.plan,
        input_a=tuple([1.0] * 32),
        input_b=tuple([2.0] * 32),
    )
    dimensions = StudyDimensions(
        ShapeClass.N32,
        PaddingClass.NONE,
        AccessClass.CONTIGUOUS,
        ParallelismClass.ONE,
    )
    with pytest.raises(ValueError, match="vector alignment"):
        bind_study_dimensions(unpadded_n32, dimensions)


def test_variant_rejects_duplicate_phase1_provenance_keys():
    valid = variant()
    duplicate = replace(
        valid.request.plan,
        runtime_provenance=(
            *valid.request.plan.runtime_provenance,
            ("phase1.parallelism", "1"),
        ),
    )
    request = replace(valid.request, plan=duplicate)
    correctness = replace(valid.correctness, execution_id=request.execution_id)

    with pytest.raises(ValueError, match="complete Phase 1 dimension manifest"):
        StudyVariant(request, correctness, valid.dimensions)


@pytest.mark.parametrize("mutation", ["reordered", "integer-spelling", "not-suffix"])
def test_variant_rejects_noncanonical_phase1_provenance(mutation):
    valid = variant()
    runtime = list(valid.request.plan.runtime_provenance)
    if mutation == "reordered":
        runtime[-2], runtime[-1] = runtime[-1], runtime[-2]
    elif mutation == "integer-spelling":
        runtime[-1] = ("phase1.parallelism", "01")
    else:
        runtime.insert(-2, ("runtime-marker", "value"))
    plan = replace(valid.request.plan, runtime_provenance=tuple(runtime))
    request = replace(valid.request, plan=plan)
    correctness = replace(valid.correctness, execution_id=request.execution_id)

    with pytest.raises(ValueError, match="canonical Phase 1 dimension encoding"):
        StudyVariant(request, correctness, valid.dimensions)
