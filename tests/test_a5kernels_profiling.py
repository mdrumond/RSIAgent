from dataclasses import replace
import gzip
import io
from pathlib import Path
import subprocess
import tarfile

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
from benchmarks.a5kernels.profiling_bz import BZProfileBackend
from benchmarks.a5kernels import A5KernelRunner, Language, RunRequest
from benchmarks.a5kernels.protocol import (
    ExecutionPlan,
    ExecutionReceipt,
    SourceFile,
    VerifiedResult,
)


SHA = "a" * 64
PLAN = ExecutionPlan(
    request_id="request-1",
    attempt_id="attempt-1",
    language="catlass-dsl",
    files=(SourceFile("kernel.py", "def kernel(): pass\n"),),
    argv=("python", "run.py"),
    input_a=(1.0,),
    input_b=(2.0,),
)
SOURCE = PLAN.source_fingerprint


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
    PLAN,
    "catlass",
    "vector_add__kernel0",
    3,
)
CORRECT = VerifiedResult(
    request_id="request-1",
    execution_id=PLAN.execution_id,
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


def test_changed_argv_cannot_reuse_correctness_from_original_plan() -> None:
    changed = ProfileRequest.from_execution_plan(
        replace(PLAN, argv=("python", "other-driver.py")),
        implementation="catlass",
        expected_kernel="vector_add__kernel0",
        device=3,
    )

    assert changed.workload_argv == ("python", "other-driver.py")
    assert changed.execution_id != REQUEST.execution_id
    with pytest.raises(ValueError, match="different execution"):
        ProfilingTreatmentController(
            FakeBackend(), treatment_enabled=True
        ).run_final(CORRECT, changed)


class FakeBackend:
    def __init__(self) -> None:
        self.commands: list[TimingCommand | CaptureCommand] = []

    def time(self, command: TimingCommand) -> TimingResult:
        self.commands.append(command)
        return TimingResult(
            7.5, SOURCE, command.request.execution_id, command.replay_id
        )

    def capture(self, command: CaptureCommand) -> ProfileCapture:
        self.commands.append(command)
        if command.metric is ProfileMetric.BASIC_INFO:
            return ProfileCapture(
                command.metric,
                SOURCE,
                command.request.execution_id,
                command.replay_id,
                (command.request.expected_kernel,),
                (("frequency_mhz", "1800"),),
                archive("basic"),
            )
        return ProfileCapture(
            command.metric,
            SOURCE,
            command.request.execution_id,
            command.replay_id,
            (command.request.expected_kernel,),
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


def _correctness_for(request: ProfileRequest) -> VerifiedResult:
    return replace(
        CORRECT,
        request_id=request.request_id,
        execution_id=request.execution_id,
        attempt_id=request.attempt_id,
        source_fingerprint=request.source_fingerprint,
    )


def _final_replay_ids(request: ProfileRequest) -> tuple[str, ...]:
    backend = FakeBackend()
    ProfilingTreatmentController(backend, treatment_enabled=True).run_final(
        _correctness_for(request), request
    )
    return tuple(command.replay_id for command in backend.commands)


@pytest.mark.parametrize(
    "changed",
    [
        replace(REQUEST, implementation="ascendc"),
        replace(REQUEST, expected_kernel="vector_add__kernel1"),
        replace(REQUEST, device=4),
        replace(REQUEST, warm_up=1),
        replace(REQUEST, launch_count=2),
        replace(REQUEST, plan=replace(PLAN, argv=("python", "other-driver.py"))),
    ],
)
def test_every_profiling_setting_changes_all_replay_ids(changed) -> None:
    assert changed.configuration_id != REQUEST.configuration_id
    assert _final_replay_ids(changed) != _final_replay_ids(REQUEST)
    assert all(
        changed_id != original_id
        for changed_id, original_id in zip(
            _final_replay_ids(changed), _final_replay_ids(REQUEST)
        )
    )


@pytest.mark.parametrize("stale_kind", ["timing", "capture"])
def test_packets_from_another_profiling_configuration_are_rejected(stale_kind) -> None:
    baseline = _final_replay_ids(REQUEST)
    changed = replace(REQUEST, device=4)

    class StaleConfigurationBackend(FakeBackend):
        def time(self, command: TimingCommand) -> TimingResult:
            result = super().time(command)
            return replace(result, replay_id=baseline[0]) if stale_kind == "timing" else result

        def capture(self, command: CaptureCommand) -> ProfileCapture:
            result = super().capture(command)
            if stale_kind == "capture" and command.metric is ProfileMetric.BASIC_INFO:
                return replace(result, replay_id=baseline[1])
            return result

    with pytest.raises(ValueError, match="submitted replay"):
        ProfilingTreatmentController(
            StaleConfigurationBackend(), treatment_enabled=True
        ).run_final(_correctness_for(changed), changed)


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
            return TimingResult(
                duration, SOURCE, command.request.execution_id, command.replay_id
            )

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
    "summary",
    [
        (),
        (("", "1800"),),
        (("frequency_mhz", ""),),
        (("  ", "1800"),),
        (("frequency_mhz", "  "),),
    ],
)
def test_basic_info_must_include_meaningful_summary(summary) -> None:
    class InvalidBasicInfoBackend(FakeBackend):
        def capture(self, command: CaptureCommand) -> ProfileCapture:
            result = super().capture(command)
            if command.metric is ProfileMetric.BASIC_INFO:
                return replace(result, summary=summary)
            return result

    backend = InvalidBasicInfoBackend()
    with pytest.raises(ValueError, match="BasicInfo.*meaningful summary"):
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
        (
            {"exported_kernels": ("vector_add__kernel0", "helper_kernel")},
            "exact expected kernel",
        ),
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
        (
            replace(
                archive("basic"), entries=(EvidenceEntry("", SHA, 1),)
            ),
            "normalized non-empty archive-relative",
        ),
        (
            replace(
                archive("basic"), entries=(EvidenceEntry(".", SHA, 1),)
            ),
            "normalized non-empty archive-relative",
        ),
        (
            replace(
                archive("basic"),
                entries=(EvidenceEntry("summary//basic.json", SHA, 1),),
            ),
            "normalized non-empty archive-relative",
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


@pytest.mark.parametrize("result_kind", ["timing", "capture"])
@pytest.mark.parametrize("identity", ["execution", "replay"])
def test_measurements_must_match_execution_and_replay(result_kind, identity) -> None:
    class MisdirectedBackend(FakeBackend):
        def time(self, command: TimingCommand) -> TimingResult:
            result = super().time(command)
            if result_kind != "timing":
                return result
            field = "execution_id" if identity == "execution" else "replay_id"
            return replace(result, **{field: f"wrong-{identity}"})

        def capture(self, command: CaptureCommand) -> ProfileCapture:
            result = super().capture(command)
            if result_kind != "capture":
                return result
            field = "execution_id" if identity == "execution" else "replay_id"
            return replace(result, **{field: f"wrong-{identity}"})

    with pytest.raises(ValueError, match=f"submitted {identity}"):
        ProfilingTreatmentController(
            MisdirectedBackend(), treatment_enabled=True
        ).run_final(CORRECT, REQUEST)


def _write_compact_archive(destination: Path) -> None:
    basic_data = gzip.compress(
        b"Kernel Name,Vector Ratio\nvector_add__kernel0,0.75\n", mtime=0
    )
    pipe_data = gzip.compress(
        b"block_id,sub_block_id,aiv_vec_ratio\n0,vector0,0.75\n", mtime=0
    )
    archive_path = destination / "ascend-profile-summary.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive_file:
        info = tarfile.TarInfo("metrics/OpBasicInfo.csv.gz")
        info.size = len(basic_data)
        archive_file.addfile(info, io.BytesIO(basic_data))
        info = tarfile.TarInfo("metrics/PipeUtilization.csv.gz")
        info.size = len(pipe_data)
        archive_file.addfile(info, io.BytesIO(pipe_data))


def test_concrete_backend_routes_exact_bound_separate_replays(tmp_path: Path) -> None:
    calls = []

    def run(argv, **kwargs):
        calls.append(tuple(argv))
        if Path(argv[0]).name == "collect_profile.sh":
            destination = Path(argv[argv.index("--output") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            _write_compact_archive(destination)
            return subprocess.CompletedProcess(argv, 0, "REMOTE_RAW=/remote/raw\n", "")
        action = argv[argv.index("--operation") + 2]
        if action == "run":
            return subprocess.CompletedProcess(
                argv, 0, "A5KERNEL_TIMING_US=7.5\n", ""
            )
        replay = argv[argv.index("--operation") + 1]
        return subprocess.CompletedProcess(
            argv, 0, f"MSPROF_PROFILE_REMOTE_DIR=/remote/{replay}\n", ""
        )

    backend = BZProfileBackend(
        validation_wrapper="/profiles/catlass-validation.sh",
        collection_wrapper="/skills/collect_profile.sh",
        catlass_source="/remote/catlass",
        evidence_directory=str(tmp_path),
        process_runner=run,
    )
    profiled_plan = replace(
        PLAN,
        argv=("python", "kernel.py", "input.json"),
        runtime_provenance=(
            ("ascendnpu_ir_gitlink", "gitlink"),
            ("ascendnpu_ir_install_commit", "install"),
            ("bridge_sha256", "bridge"),
            ("cann_version", "9.1"),
            ("catlass_revision", "revision"),
            ("catlass_source", "/remote/catlass"),
            ("execution_profile", "bz-a5"),
            ("manifest_sha256", "manifest"),
        ),
    )
    profiled_request = replace(REQUEST, plan=profiled_plan)
    correctness = _correctness_for(profiled_request)
    result = ProfilingTreatmentController(
        backend, treatment_enabled=False
    ).run_final(correctness, profiled_request)

    assert result.duration_us == 7.5
    assert result.pipe_utilization
    assert all(
        key.startswith("PipeUtilization.csv.gz:")
        for key, _value in result.pipe_utilization
    )
    timing_call = next(
        call
        for call in calls
        if "--operation" in call and call[call.index("--operation") + 2] == "run"
    )
    separator = timing_call.index("--")
    assert timing_call[separator + 1 : separator + 4] == (
        "env",
        f"BZ_A5_PROFILE_PHYSICAL_DEVICE={profiled_request.device}",
        "A5KERNEL_EMIT_TIMING=1",
    )
    profile_calls = [
        call
        for call in calls
        if "--operation" in call and call[call.index("--operation") + 2] == "profile"
    ]
    assert [call[call.index("--metric") + 1] for call in profile_calls] == [
        "BasicInfo",
        "PipeUtilization",
    ]
    assert "--kernel-name" not in profile_calls[0]
    assert (
        profile_calls[1][profile_calls[1].index("--kernel-name") + 1]
        == profiled_request.expected_kernel
    )
    assert len(
        [call for call in calls if Path(call[0]).name == "collect_profile.sh"]
    ) == 2
    collection_calls = [
        call for call in calls if Path(call[0]).name == "collect_profile.sh"
    ]
    assert "--kernel-name" in collection_calls[0]
    assert "--kernel-name" not in collection_calls[1]
    assert all(call[call.index("--operation") + 1].startswith("profile-") for call in profile_calls)


def test_concrete_backend_rejects_relative_remote_tree(tmp_path: Path) -> None:
    def run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 0, "MSPROF_PROFILE_REMOTE_DIR=relative/tree\n", ""
        )

    backend = BZProfileBackend(
        validation_wrapper="/profiles/catlass-validation.sh",
        collection_wrapper="/skills/collect_profile.sh",
        catlass_source="/remote/catlass",
        evidence_directory=str(tmp_path),
        process_runner=run,
    )
    profiled_request = replace(
        REQUEST,
        plan=replace(
            PLAN,
            runtime_provenance=(
                ("ascendnpu_ir_gitlink", "gitlink"),
                ("ascendnpu_ir_install_commit", "install"),
                ("bridge_sha256", "bridge"),
                ("cann_version", "9.1"),
                ("catlass_revision", "revision"),
                ("catlass_source", "/remote/catlass"),
                ("execution_profile", "bz-a5"),
                ("manifest_sha256", "manifest"),
            ),
        ),
    )
    command = CaptureCommand(
        CampaignKind.FINAL,
        profiled_request,
        ProfileMetric.BASIC_INFO,
        "profile-" + "a" * 64,
    )

    with pytest.raises(RuntimeError, match="non-absolute retained tree"):
        backend.capture(command)


def test_concrete_backend_rejects_non_catlass_plan(tmp_path: Path) -> None:
    backend = BZProfileBackend(
        validation_wrapper="/profiles/catlass-validation.sh",
        collection_wrapper="/skills/collect_profile.sh",
        catlass_source="/remote/catlass",
        evidence_directory=str(tmp_path),
    )
    request = replace(REQUEST, plan=replace(PLAN, language="ascend-c"))
    command = TimingCommand(CampaignKind.FINAL, request, "profile-" + "a" * 64)

    with pytest.raises(ValueError, match="catlass-dsl execution plan"):
        backend.time(command)


@pytest.mark.parametrize("execution_profile", [None, "gz-a3"])
def test_concrete_backend_rejects_non_bz_execution_provenance(
    tmp_path: Path, execution_profile: str | None
) -> None:
    calls = []

    def run(argv, **kwargs):
        calls.append(tuple(argv))
        raise AssertionError("invalid provenance must fail before dispatch")

    backend = BZProfileBackend(
        validation_wrapper="/profiles/catlass-validation.sh",
        collection_wrapper="/skills/collect_profile.sh",
        catlass_source="/remote/catlass",
        evidence_directory=str(tmp_path),
        process_runner=run,
    )
    provenance = {
        "ascendnpu_ir_gitlink": "gitlink",
        "ascendnpu_ir_install_commit": "install",
        "bridge_sha256": "bridge",
        "cann_version": "9.1",
        "catlass_revision": "revision",
        "catlass_source": "/remote/catlass",
        "manifest_sha256": "manifest",
    }
    if execution_profile is not None:
        provenance["execution_profile"] = execution_profile
    request = replace(
        REQUEST,
        plan=replace(PLAN, runtime_provenance=tuple(sorted(provenance.items()))),
    )
    timing = TimingCommand(CampaignKind.FINAL, request, "profile-" + "a" * 64)
    capture = CaptureCommand(
        CampaignKind.FINAL,
        request,
        ProfileMetric.BASIC_INFO,
        "profile-" + "b" * 64,
    )

    with pytest.raises(ValueError, match="bz-a5 execution provenance"):
        backend.time(timing)
    with pytest.raises(ValueError, match="bz-a5 execution provenance"):
        backend.capture(capture)
    assert calls == []


def test_real_catlass_fixture_exposes_host_owned_timing_and_device_contract() -> None:
    plan = A5KernelRunner(PreparingBackend()).prepare(
        RunRequest(Language.CATLASS_DSL.value), attempt_id="timed-fixture"
    )
    kernel_source = next(
        source.content for source in plan.files if source.relative_path == "kernel.py"
    )

    assert 'os.environ.get("BZ_A5_PROFILE_PHYSICAL_DEVICE", "0")' in kernel_source
    assert 'os.environ.get("A5KERNEL_EMIT_TIMING") == "1"' in kernel_source
    assert "torch.npu.synchronize()" in kernel_source
    assert 'print(f"A5KERNEL_TIMING_US={duration_us:.6f}")' in kernel_source
