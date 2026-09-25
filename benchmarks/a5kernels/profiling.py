"""Host-owned profiling campaigns for A5 kernel experiments.

The controller intentionally knows nothing about SSH or msprof output layouts.
A backend executes immutable command specifications and returns compact,
curated captures.  That keeps treatment assignment and validation on the host
side instead of trusting an exploring agent to describe its own performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import PurePosixPath
import re
from typing import Protocol

from benchmarks.a5kernels.protocol import ExecutionPlan, VerifiedResult, canonical_hash


_SHA256 = re.compile(r"[0-9a-f]{64}")


class ProfileMetric(str, Enum):
    BASIC_INFO = "BasicInfo"
    PIPE_UTILIZATION = "PipeUtilization"


class CampaignKind(str, Enum):
    INTERMEDIATE = "intermediate"
    FINAL = "final"


@dataclass(frozen=True)
class ProfileRequest:
    """Host-declared identity and launch settings for a kernel campaign."""

    plan: ExecutionPlan
    implementation: str
    expected_kernel: str
    device: int
    warm_up: int = 0
    launch_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ExecutionPlan):
            raise TypeError("plan must be an immutable ExecutionPlan")
        if self.plan.argv is None:
            raise ValueError("profiling requires a concrete executable runtime plan")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.attempt_id):
            raise ValueError("attempt_id must be a safe host-owned identifier")
        if not self.expected_kernel.strip():
            raise ValueError("expected_kernel must be an exact non-empty name")
        if self.device < 0:
            raise ValueError("device must be non-negative")
        if not self.plan.argv:
            raise ValueError("workload_argv must not be empty")
        if self.warm_up < 0 or self.launch_count < 1:
            raise ValueError("invalid warm-up or launch count")

    @property
    def request_id(self) -> str:
        return self.plan.request_id

    @property
    def attempt_id(self) -> str:
        return self.plan.attempt_id

    @property
    def execution_id(self) -> str:
        return self.plan.execution_id

    @property
    def workload_argv(self) -> tuple[str, ...]:
        assert self.plan.argv is not None
        return self.plan.argv

    @property
    def source_fingerprint(self) -> str:
        return self.plan.source_fingerprint

    @property
    def configuration_id(self) -> str:
        """Identity of the executable and every profiling launch setting."""

        return canonical_hash(
            {
                "device": self.device,
                "execution_id": self.execution_id,
                "expected_kernel": self.expected_kernel,
                "implementation": self.implementation,
                "launch_count": self.launch_count,
                "warm_up": self.warm_up,
            }
        )

    @classmethod
    def from_execution_plan(
        cls,
        plan: ExecutionPlan,
        *,
        implementation: str,
        expected_kernel: str,
        device: int,
        warm_up: int = 0,
        launch_count: int = 1,
    ) -> ProfileRequest:
        """Bind profiling to the exact host-prepared executable attempt."""

        return cls(
            plan=plan,
            implementation=implementation,
            expected_kernel=expected_kernel,
            device=device,
            warm_up=warm_up,
            launch_count=launch_count,
        )


@dataclass(frozen=True)
class TimingCommand:
    """Canonical minimally instrumented timing run."""

    campaign: CampaignKind
    request: ProfileRequest
    replay_id: str
    profiler_enabled: bool = field(default=False, init=False)


@dataclass(frozen=True)
class CaptureCommand:
    """One profiler replay; metric groups are never combined."""

    campaign: CampaignKind
    request: ProfileRequest
    metric: ProfileMetric
    replay_id: str
    kernel_name: str | None = None

    def __post_init__(self) -> None:
        if self.metric is ProfileMetric.BASIC_INFO and self.kernel_name is not None:
            raise ValueError("BasicInfo discovers the exported kernel name")
        if self.metric is ProfileMetric.PIPE_UTILIZATION and not self.kernel_name:
            raise ValueError("PipeUtilization requires an exact kernel name")


@dataclass(frozen=True)
class EvidenceEntry:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class EvidenceArchive:
    """Compact transfer metadata; ``remote_tree`` remains on BZ-A5."""

    archive_name: str
    archive_sha256: str
    archive_size_bytes: int
    remote_tree: str
    remote_tree_retained: bool
    entries: tuple[EvidenceEntry, ...]

    def validate(self, max_archive_bytes: int) -> None:
        if not self.remote_tree.startswith("/") or not self.remote_tree_retained:
            raise ValueError("the full vendor tree must be retained remotely")
        if not (0 < self.archive_size_bytes <= max_archive_bytes):
            raise ValueError("evidence archive exceeds compact transfer limit")
        if not _SHA256.fullmatch(self.archive_sha256):
            raise ValueError("invalid archive sha256")
        paths = [entry.relative_path for entry in self.entries]
        if not paths or paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("evidence entries must be non-empty, unique, and sorted")
        for entry in self.entries:
            path = PurePosixPath(entry.relative_path)
            normalized = path.as_posix()
            if (
                not entry.relative_path
                or normalized == "."
                or normalized != entry.relative_path
                or path.is_absolute()
                or ".." in path.parts
            ):
                raise ValueError(
                    "evidence paths must be normalized non-empty archive-relative paths"
                )
            if not _SHA256.fullmatch(entry.sha256) or entry.size_bytes < 0:
                raise ValueError("invalid evidence entry metadata")

    @property
    def manifest_sha256(self) -> str:
        return canonical_hash(
            {
                "archive": self.archive_name,
                "archive_sha256": self.archive_sha256,
                "archive_size_bytes": self.archive_size_bytes,
                "remote_tree": self.remote_tree,
                "remote_tree_retained": self.remote_tree_retained,
                "entries": [entry.__dict__ for entry in self.entries],
            }
        )


@dataclass(frozen=True)
class TimingResult:
    duration_us: float
    source_fingerprint: str
    execution_id: str
    replay_id: str


@dataclass(frozen=True)
class ProfileCapture:
    metric: ProfileMetric
    source_fingerprint: str
    execution_id: str
    replay_id: str
    exported_kernels: tuple[str, ...]
    summary: tuple[tuple[str, str], ...]
    evidence: EvidenceArchive


@dataclass(frozen=True)
class ProfilingFeedback:
    kernel_name: str
    basic_info: tuple[tuple[str, str], ...]
    pipe_utilization: tuple[tuple[str, str], ...]
    evidence_manifest_sha256: tuple[str, str]


@dataclass(frozen=True)
class FinalProfileResult:
    duration_us: float
    kernel_name: str
    basic_info: tuple[tuple[str, str], ...]
    pipe_utilization: tuple[tuple[str, str], ...]
    evidence_manifest_sha256: tuple[str, str]


class ProfileBackend(Protocol):
    def time(self, command: TimingCommand) -> TimingResult: ...

    def capture(self, command: CaptureCommand) -> ProfileCapture: ...


def _replay_id(
    request: ProfileRequest, campaign: CampaignKind, kind: str
) -> str:
    return "profile-" + canonical_hash(
        {
            "campaign": campaign.value,
            "configuration_id": request.configuration_id,
            "kind": kind,
        }
    )


class ProfilingTreatmentController:
    """Run treatment feedback and treatment-independent final evaluation."""

    def __init__(
        self,
        backend: ProfileBackend,
        *,
        treatment_enabled: bool,
        max_archive_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        self._backend = backend
        self._enabled = treatment_enabled
        self._max_archive_bytes = max_archive_bytes

    def run_intermediate(
        self, correctness: VerifiedResult, request: ProfileRequest
    ) -> ProfilingFeedback | None:
        self._validate_correctness(correctness, request)
        if not self._enabled:
            return None
        basic, pipe = self._captures(CampaignKind.INTERMEDIATE, request)
        return self._feedback(request.expected_kernel, basic, pipe)

    def run_final(
        self, correctness: VerifiedResult, request: ProfileRequest
    ) -> FinalProfileResult:
        """Run the same host evaluation regardless of treatment assignment."""

        self._validate_correctness(correctness, request)
        timing_command = TimingCommand(
            CampaignKind.FINAL,
            request,
            _replay_id(request, CampaignKind.FINAL, "timing"),
        )
        timing = self._backend.time(timing_command)
        if (
            not math.isfinite(timing.duration_us)
            or timing.duration_us <= 0
            or timing.source_fingerprint != request.source_fingerprint
        ):
            raise ValueError("timing result does not match the submitted source")
        if timing.execution_id != request.execution_id:
            raise ValueError("timing result does not match the submitted execution")
        if timing.replay_id != timing_command.replay_id:
            raise ValueError("timing result does not match the submitted replay")
        basic, pipe = self._captures(CampaignKind.FINAL, request)
        feedback = self._feedback(request.expected_kernel, basic, pipe)
        return FinalProfileResult(
            timing.duration_us,
            feedback.kernel_name,
            feedback.basic_info,
            feedback.pipe_utilization,
            feedback.evidence_manifest_sha256,
        )

    def _captures(
        self, campaign: CampaignKind, request: ProfileRequest
    ) -> tuple[ProfileCapture, ProfileCapture]:
        basic_command = CaptureCommand(
            campaign,
            request,
            ProfileMetric.BASIC_INFO,
            _replay_id(request, campaign, "basic"),
        )
        basic = self._backend.capture(basic_command)
        self._validate_capture(basic, basic_command)
        if basic.exported_kernels.count(request.expected_kernel) != 1:
            raise ValueError("BasicInfo did not identify the exact expected kernel once")
        self._validate_summary(basic, "BasicInfo")
        pipe_command = CaptureCommand(
            campaign,
            request,
            ProfileMetric.PIPE_UTILIZATION,
            _replay_id(request, campaign, "pipe"),
            kernel_name=request.expected_kernel,
        )
        pipe = self._backend.capture(pipe_command)
        self._validate_capture(pipe, pipe_command)
        if pipe.exported_kernels != (request.expected_kernel,):
            raise ValueError(
                "PipeUtilization did not identify the exact expected kernel once"
            )
        self._validate_summary(pipe, "PipeUtilization")
        return basic, pipe

    def _validate_capture(
        self, capture: ProfileCapture, command: CaptureCommand
    ) -> None:
        request = command.request
        if capture.metric is not command.metric:
            raise ValueError("backend returned the wrong metric replay")
        if capture.source_fingerprint != request.source_fingerprint:
            raise ValueError("profile capture does not match the submitted source")
        if capture.execution_id != request.execution_id:
            raise ValueError("profile capture does not match the submitted execution")
        if capture.replay_id != command.replay_id:
            raise ValueError("profile capture does not match the submitted replay")
        capture.evidence.validate(self._max_archive_bytes)

    @staticmethod
    def _validate_summary(capture: ProfileCapture, metric_name: str) -> None:
        if not capture.summary or any(
            not key.strip() or not value.strip() for key, value in capture.summary
        ):
            raise ValueError(f"{metric_name} did not return a meaningful summary")

    @staticmethod
    def _validate_correctness(correctness: VerifiedResult, request: ProfileRequest) -> None:
        if not correctness.passed or correctness.exit_code != 0:
            raise ValueError("correctness must pass before timing or profiling")
        if correctness.request_id != request.request_id:
            raise ValueError("correctness result belongs to a different request")
        if correctness.attempt_id != request.attempt_id:
            raise ValueError("correctness result belongs to a different attempt")
        if correctness.execution_id != request.execution_id:
            raise ValueError("correctness result belongs to a different execution")
        if correctness.source_fingerprint != request.source_fingerprint:
            raise ValueError("correctness result belongs to different source")

    @staticmethod
    def _feedback(
        kernel_name: str, basic: ProfileCapture, pipe: ProfileCapture
    ) -> ProfilingFeedback:
        return ProfilingFeedback(
            kernel_name,
            basic.summary,
            pipe.summary,
            (basic.evidence.manifest_sha256, pipe.evidence.manifest_sha256),
        )
