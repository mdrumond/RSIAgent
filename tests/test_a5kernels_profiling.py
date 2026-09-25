from dataclasses import replace

import pytest

from benchmarks.a5kernels.profiling import (
    CampaignKind,
    CaptureCommand,
    EvidenceArchive,
    EvidenceEntry,
    ProfileCapture,
    ProfileMetric,
    ProfileRequest,
    ProfilingTreatmentController,
    TimingCommand,
    TimingResult,
)
from benchmarks.a5kernels import A5KernelRunner, Language, RunRequest
from benchmarks.a5kernels.protocol import ExecutionReceipt, VerifiedResult


SHA = "a" * 64
SOURCE = "b" * 64


def archive(metric: str) -> EvidenceArchive:
    return EvidenceArchive(
        f"{metric}.tar.zst",
        SHA,
        100,
        f"/retained/msprof/{metric}",
        True,
        (EvidenceEntry(f"summary/{metric}.json", SHA, 10),),
    )


REQUEST = ProfileRequest(
    "request-1",
    "attempt-1",
    "execution-1",
    "catlass",
    "vector_add__kernel0",
    3,
    ("python", "run.py"),
    SOURCE,
)
CORRECT = VerifiedResult(
    request_id="request-1",
    execution_id="execution-1",
    attempt_id="attempt-1",
    language="catlass",
    runtime_provenance=(),
    passed=True,
    max_abs_error=0.0,
    exit_code=0,
    output_sha256=SHA,
    source_fingerprint=SOURCE,
    evidence_sha256=SHA,
    session_handle="session",
    attestation_sha256=SHA,
)


class PreparingBackend:
    def execute(self, plan):
        return ExecutionReceipt(0, ())


def test_profile_request_binds_to_host_prepared_attempt() -> None:
    plan = A5KernelRunner(PreparingBackend()).prepare(
        RunRequest(Language.CATLASS_DSL.value), attempt_id="host-attempt-7"
    )
    plan = replace(plan, argv=("python", "driver.py"))

    request = ProfileRequest.from_execution_plan(
        plan,
        implementation="catlass",
        expected_kernel="vector_add__kernel0",
        device=3,
    )

    assert request.request_id == plan.request_id
    assert request.attempt_id == "host-attempt-7"
    assert request.execution_id == plan.execution_id
    assert request.source_fingerprint == plan.source_fingerprint


def test_profile_request_rejects_foundational_plan_without_runtime() -> None:
    plan = A5KernelRunner(PreparingBackend()).prepare(
        RunRequest(Language.TRITON_ASCEND.value), attempt_id="host-attempt-7"
    )

    with pytest.raises(ValueError, match="concrete executable"):
        ProfileRequest.from_execution_plan(
            plan,
            implementation="catlass",
            expected_kernel="vector_add__kernel0",
            device=3,
        )


class FakeBackend:
    def __init__(self) -> None:
        self.commands: list[TimingCommand | CaptureCommand] = []

    def time(self, command: TimingCommand) -> TimingResult:
        self.commands.append(command)
        return TimingResult(7.5, SOURCE)

    def capture(self, command: CaptureCommand) -> ProfileCapture:
        self.commands.append(command)
        if command.metric is ProfileMetric.BASIC_INFO:
            return ProfileCapture(
                command.metric,
                SOURCE,
                ("vector_add__kernel0",),
                (("frequency_mhz", "1800"),),
                archive("basic"),
            )
        return ProfileCapture(
            command.metric,
            SOURCE,
            ("vector_add__kernel0",),
            (("vector_ratio", "0.75"),),
            archive("pipe"),
        )


def test_treatment_off_withholds_feedback_without_profiling() -> None:
    backend = FakeBackend()
    controller = ProfilingTreatmentController(backend, treatment_enabled=False)

    assert controller.run_intermediate(CORRECT, REQUEST) is None
    assert backend.commands == []


def test_treatment_runs_basic_then_exact_bound_pipe_replay() -> None:
    backend = FakeBackend()
    controller = ProfilingTreatmentController(backend, treatment_enabled=True)

    feedback = controller.run_intermediate(CORRECT, REQUEST)

    assert feedback is not None
    assert feedback.kernel_name == "vector_add__kernel0"
    assert [command.metric for command in backend.commands] == [
        ProfileMetric.BASIC_INFO,
        ProfileMetric.PIPE_UTILIZATION,
    ]
    assert backend.commands[0].kernel_name is None
    assert backend.commands[1].kernel_name == "vector_add__kernel0"
    assert backend.commands[0].replay_id != backend.commands[1].replay_id


@pytest.mark.parametrize("treatment_enabled", [False, True])
def test_final_contract_is_identical_for_both_arms(treatment_enabled: bool) -> None:
    backend = FakeBackend()
    controller = ProfilingTreatmentController(
        backend, treatment_enabled=treatment_enabled
    )

    result = controller.run_final(CORRECT, REQUEST)

    assert result.duration_us == 7.5
    assert isinstance(backend.commands[0], TimingCommand)
    assert backend.commands[0].profiler_enabled is False
    assert [command.campaign for command in backend.commands] == [
        CampaignKind.FINAL,
        CampaignKind.FINAL,
        CampaignKind.FINAL,
    ]
    assert [getattr(command, "metric", None) for command in backend.commands] == [
        None,
        ProfileMetric.BASIC_INFO,
        ProfileMetric.PIPE_UTILIZATION,
    ]


def test_failed_correctness_stops_before_any_measurement() -> None:
    backend = FakeBackend()
    failed = replace(CORRECT, passed=False)

    with pytest.raises(ValueError, match="correctness must pass"):
        ProfilingTreatmentController(backend, treatment_enabled=True).run_final(
            failed, REQUEST
        )

    assert backend.commands == []


def test_correctness_must_belong_to_same_host_attempt() -> None:
    backend = FakeBackend()

    with pytest.raises(ValueError, match="different attempt"):
        ProfilingTreatmentController(backend, treatment_enabled=True).run_final(
            replace(CORRECT, attempt_id="other-attempt"), REQUEST
        )

    assert backend.commands == []


def test_correctness_must_belong_to_same_execution() -> None:
    backend = FakeBackend()

    with pytest.raises(ValueError, match="different execution"):
        ProfilingTreatmentController(backend, treatment_enabled=True).run_final(
            replace(CORRECT, execution_id="other-execution"), REQUEST
        )

    assert backend.commands == []


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), float("-inf")])
def test_final_rejects_non_finite_timing(duration: float) -> None:
    class NonFiniteBackend(FakeBackend):
        def time(self, command: TimingCommand) -> TimingResult:
            self.commands.append(command)
            return TimingResult(duration, SOURCE)

    backend = NonFiniteBackend()
    with pytest.raises(ValueError, match="timing result"):
        ProfilingTreatmentController(backend, treatment_enabled=True).run_final(
            CORRECT, REQUEST
        )

    assert len(backend.commands) == 1


def test_basic_info_must_export_expected_kernel_exactly_once() -> None:
    class WrongKernelBackend(FakeBackend):
        def capture(self, command: CaptureCommand) -> ProfileCapture:
            result = super().capture(command)
            if command.metric is ProfileMetric.BASIC_INFO:
                return replace(result, exported_kernels=("vector_add",))
            return result

    backend = WrongKernelBackend()
    with pytest.raises(ValueError, match="exact expected kernel"):
        ProfilingTreatmentController(backend, treatment_enabled=True).run_intermediate(
            CORRECT, REQUEST
        )

    assert len(backend.commands) == 1


@pytest.mark.parametrize(
    "replacement, message",
    [
        ({"exported_kernels": ()}, "exact expected kernel"),
        ({"exported_kernels": ("vector_add",)}, "exact expected kernel"),
        ({"exported_kernels": ("vector_add__kernel0",) * 2}, "exact expected kernel"),
        ({"summary": ()}, "meaningful summary"),
        ({"summary": (("", "0.75"),)}, "meaningful summary"),
        ({"summary": (("vector_ratio", ""),)}, "meaningful summary"),
    ],
)
def test_pipe_utilization_must_match_kernel_and_include_summary(
    replacement, message
) -> None:
    class InvalidPipeBackend(FakeBackend):
        def capture(self, command: CaptureCommand) -> ProfileCapture:
            result = super().capture(command)
            if command.metric is ProfileMetric.PIPE_UTILIZATION:
                return replace(result, **replacement)
            return result

    backend = InvalidPipeBackend()
    with pytest.raises(ValueError, match=message):
        ProfilingTreatmentController(backend, treatment_enabled=True).run_intermediate(
            CORRECT, REQUEST
        )

    assert len(backend.commands) == 2


@pytest.mark.parametrize(
    "bad_archive, message",
    [
        (replace(archive("basic"), remote_tree_retained=False), "retained remotely"),
        (replace(archive("basic"), archive_size_bytes=201), "compact transfer"),
        (replace(archive("basic"), archive_sha256="unhashed"), "archive sha256"),
        (
            replace(
                archive("basic"),
                entries=(
                    EvidenceEntry("z.json", SHA, 1),
                    EvidenceEntry("a.json", SHA, 1),
                ),
            ),
            "unique, and sorted",
        ),
    ],
)
def test_evidence_archive_contract(bad_archive: EvidenceArchive, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        bad_archive.validate(200)


def test_capture_must_match_source_and_metric() -> None:
    class MismatchBackend(FakeBackend):
        def capture(self, command: CaptureCommand) -> ProfileCapture:
            return replace(super().capture(command), source_fingerprint="c" * 64)

    with pytest.raises(ValueError, match="submitted source"):
        ProfilingTreatmentController(
            MismatchBackend(), treatment_enabled=True
        ).run_intermediate(CORRECT, REQUEST)
