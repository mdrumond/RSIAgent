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

    request_id: str
    attempt_id: str
    execution_id: str
    implementation: str
    expected_kernel: str
    device: int
    workload_argv: tuple[str, ...]
    source_fingerprint: str
    warm_up: int = 0
    launch_count: int = 1

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.attempt_id):
            raise ValueError("attempt_id must be a safe host-owned identifier")
        if not self.expected_kernel.strip():
            raise ValueError("expected_kernel must be an exact non-empty name")
        if self.device < 0:
            raise ValueError("device must be non-negative")
        if not self.workload_argv:
            raise ValueError("workload_argv must not be empty")
        if self.warm_up < 0 or self.launch_count < 1:
            raise ValueError("invalid warm-up or launch count")

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

        if plan.argv is None:
            raise ValueError("profiling requires a concrete executable runtime plan")
        return cls(
            request_id=plan.request_id,
            attempt_id=plan.attempt_id,
            execution_id=plan.execution_id,
            implementation=implementation,
            expected_kernel=expected_kernel,
            device=device,
            workload_argv=plan.argv,
            source_fingerprint=plan.source_fingerprint,
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
            if entry.relative_path.startswith("/") or ".." in entry.relative_path.split("/"):
                raise ValueError("evidence paths must be archive-relative")
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


@dataclass(frozen=True)
class ProfileCapture:
    metric: ProfileMetric
    source_fingerprint: str
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
        timing = self._backend.time(
            TimingCommand(
                CampaignKind.FINAL,
                request,
                f"{request.attempt_id}-final-timing",
            )
        )
        if (
            not math.isfinite(timing.duration_us)
            or timing.duration_us <= 0
            or timing.source_fingerprint != request.source_fingerprint
        ):
            raise ValueError("timing result does not match the submitted source")
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
        basic = self._backend.capture(
            CaptureCommand(
                campaign,
                request,
                ProfileMetric.BASIC_INFO,
                f"{request.attempt_id}-{campaign.value}-basic",
            )
        )
        self._validate_capture(basic, request, ProfileMetric.BASIC_INFO)
        if basic.exported_kernels.count(request.expected_kernel) != 1:
            raise ValueError("BasicInfo did not identify the exact expected kernel once")
        pipe = self._backend.capture(
            CaptureCommand(
                campaign,
                request,
                ProfileMetric.PIPE_UTILIZATION,
                f"{request.attempt_id}-{campaign.value}-pipe",
                kernel_name=request.expected_kernel,
            )
        )
        self._validate_capture(pipe, request, ProfileMetric.PIPE_UTILIZATION)
        if pipe.exported_kernels.count(request.expected_kernel) != 1:
            raise ValueError(
                "PipeUtilization did not identify the exact expected kernel once"
            )
        if not pipe.summary or any(
            not key.strip() or not value.strip() for key, value in pipe.summary
        ):
            raise ValueError("PipeUtilization did not return a meaningful summary")
        return basic, pipe

    def _validate_capture(
        self, capture: ProfileCapture, request: ProfileRequest, metric: ProfileMetric
    ) -> None:
        if capture.metric is not metric:
            raise ValueError("backend returned the wrong metric replay")
        if capture.source_fingerprint != request.source_fingerprint:
            raise ValueError("profile capture does not match the submitted source")
        capture.evidence.validate(self._max_archive_bytes)

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
