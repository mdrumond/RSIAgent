"""Host-owned profiling campaigns for A5 kernel experiments.

The controller intentionally knows nothing about SSH or msprof output layouts.
A backend executes immutable command specifications and returns compact,
curated captures.  That keeps treatment assignment and validation on the host
side instead of trusting an exploring agent to describe its own performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from pathlib import PurePosixPath
import re
from statistics import median, pstdev
from typing import Protocol, Sequence

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
    block_count: int | None = None

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
        bindings = dict(self.plan.environment.bindings)
        device_environment = bindings.get("BZ_A5_PROFILE_PHYSICAL_DEVICE")
        if device_environment is not None and device_environment != str(self.device):
            raise ValueError(
                "profile device must match the canonical device bound in the plan environment"
            )
        reserved_timing = {
            "A5KERNEL_EMIT_TIMING",
            "A5KERNEL_WARM_UP",
            "A5KERNEL_LAUNCH_COUNT",
        }
        if reserved_timing.intersection(bindings):
            raise ValueError("plan environment must not override host-owned timing policy")
        if self.block_count is not None:
            if type(self.block_count) is not int or not 1 <= self.block_count <= 8:
                raise ValueError("block_count must be an integer from 1 through 8")
            if bindings.get("A5KERNEL_BLOCK_NUM") != str(self.block_count):
                raise ValueError(
                    "plan block binding must match host-owned block metadata"
                )
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
                "block_count": self.block_count,
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
        block_count: int | None = None,
    ) -> ProfileRequest:
        """Bind profiling to the exact host-prepared executable attempt."""

        return cls(
            plan=plan,
            implementation=implementation,
            expected_kernel=expected_kernel,
            device=device,
            warm_up=warm_up,
            launch_count=launch_count,
            block_count=block_count,
        )


def bind_profile_device(plan: ExecutionPlan, device: int) -> ExecutionPlan:
    """Bind the physical device before correctness execution and profiling."""

    if not isinstance(plan, ExecutionPlan):
        raise TypeError("plan must be an ExecutionPlan")
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise ValueError("device must be a non-negative integer")
    return replace(
        plan,
        environment=plan.environment.with_binding(
            "BZ_A5_PROFILE_PHYSICAL_DEVICE", str(device)
        ),
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
            "attempt_id": request.attempt_id,
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


class StudyPreset(str, Enum):
    TIMING_STUDY = "timing-study"
    BASIC_INFO = "basic-info"
    PIPE_UTILIZATION = "pipe-utilization"
    OPTIMIZATION_COMPARISON = "optimization-comparison"


class ShapeClass(str, Enum):
    N32 = "n32"
    N64 = "n64"
    N128 = "n128"
    N256 = "n256"


class PaddingClass(str, Enum):
    NONE = "none"
    ALIGN_64 = "align-64"
    ALIGN_256 = "align-256"


class AccessClass(str, Enum):
    CONTIGUOUS = "contiguous"
    STRIDED_2 = "strided-2"
    TILED = "tiled"


class ParallelismClass(int, Enum):
    ONE = 1


class MetricDomain(str, Enum):
    TIMING = "timing"
    BASIC_INFO = "basic-info"
    PIPE_UTILIZATION = "pipe-utilization"


@dataclass(frozen=True)
class StudyDimensions:
    """One point in the preregistered, bounded Phase 1 design space."""

    shape: ShapeClass
    padding: PaddingClass
    access: AccessClass
    parallelism: ParallelismClass

    def __post_init__(self) -> None:
        for name, expected in (
            ("shape", ShapeClass),
            ("padding", PaddingClass),
            ("access", AccessClass),
            ("parallelism", ParallelismClass),
        ):
            if not isinstance(getattr(self, name), expected):
                raise ValueError(f"{name} must use the host-owned {expected.__name__}")

    def as_dict(self) -> dict[str, str | int]:
        return {
            "shape": self.shape.value,
            "padding": self.padding.value,
            "access": self.access.value,
            "parallelism": self.parallelism.value,
        }

    def as_provenance(self) -> tuple[tuple[str, str], ...]:
        """Canonical host-owned plan metadata binding this study point."""

        return tuple(
            (f"phase1.{name}", str(value))
            for name, value in self.as_dict().items()
        )


_STUDY_PROVENANCE_KEYS = {
    "phase1.shape", "phase1.padding", "phase1.access", "phase1.parallelism",
}


def study_dimensions_from_plan(plan: ExecutionPlan) -> StudyDimensions | None:
    """Decode and validate the study configuration executed by ``plan``."""

    entries = tuple(
        (key, value)
        for key, value in plan.runtime_provenance
        if key.startswith("phase1.")
    )
    if not entries:
        return None
    if (
        len(entries) != len(_STUDY_PROVENANCE_KEYS)
        or {key for key, _value in entries} != _STUDY_PROVENANCE_KEYS
    ):
        raise ValueError("execution plan requires a complete Phase 1 dimension manifest")
    values = dict(entries)
    try:
        dimensions = StudyDimensions(
            ShapeClass(values["phase1.shape"]),
            PaddingClass(values["phase1.padding"]),
            AccessClass(values["phase1.access"]),
            ParallelismClass(int(values["phase1.parallelism"])),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("execution plan has invalid Phase 1 dimensions") from exc
    if (
        entries != dimensions.as_provenance()
        or tuple(plan.runtime_provenance[-len(entries):]) != entries
    ):
        raise ValueError("execution plan requires canonical Phase 1 dimension encoding")
    _validate_dimensions_against_plan(plan, dimensions)
    return dimensions


def bind_study_dimensions(
    plan: ExecutionPlan, dimensions: StudyDimensions
) -> ExecutionPlan:
    """Atomically bind typed dimensions after checking the executable workload."""

    if not isinstance(plan, ExecutionPlan) or not isinstance(dimensions, StudyDimensions):
        raise TypeError("a host ExecutionPlan and StudyDimensions are required")
    if any(key.startswith("phase1.") for key, _value in plan.runtime_provenance):
        raise ValueError("execution plan already contains Phase 1 dimensions")
    bound = replace(
        plan,
        environment=plan.environment.with_binding(
            "A5KERNEL_BLOCK_NUM", str(dimensions.parallelism.value)
        ),
        runtime_provenance=(*plan.runtime_provenance, *dimensions.as_provenance()),
    )
    _validate_dimensions_against_plan(bound, dimensions)
    return bound


def _validate_dimensions_against_plan(
    plan: ExecutionPlan, dimensions: StudyDimensions
) -> None:
    if plan.language != "catlass-dsl":
        raise ValueError("Phase 1 dimensions require a Catlass DSL execution plan")
    if dimensions.access is not AccessClass.CONTIGUOUS:
        raise ValueError("the current Catlass fixture supports only contiguous access")
    block_environment = dict(plan.environment.bindings).get(
        "A5KERNEL_BLOCK_NUM"
    )
    expected_block = str(dimensions.parallelism.value)
    if block_environment != expected_block:
        raise ValueError(
            "execution plan block count must match the host-owned study block dimension"
        )
    logical_length = int(dimensions.shape.value[1:])
    if dimensions.padding is PaddingClass.NONE:
        if logical_length % 64:
            raise ValueError("unpadded shapes must match the fixture vector alignment")
        physical_length = logical_length
    else:
        alignment = 64 if dimensions.padding is PaddingClass.ALIGN_64 else 256
        physical_length = ((logical_length + alignment - 1) // alignment) * alignment
        if physical_length == logical_length:
            raise ValueError("padding dimensions must add a real padded tail")
    if len(plan.input_a) != physical_length or len(plan.input_b) != physical_length:
        raise ValueError("Phase 1 shape and padding do not match the execution inputs")
    if any(plan.input_a[logical_length:]) or any(plan.input_b[logical_length:]):
        raise ValueError("Phase 1 padded input tails must contain host-generated zeros")


@dataclass(frozen=True)
class StudyVariant:
    request: ProfileRequest
    correctness: VerifiedResult
    dimensions: StudyDimensions

    def __post_init__(self) -> None:
        if not isinstance(self.request, ProfileRequest):
            raise TypeError("request must be a ProfileRequest")
        if not isinstance(self.correctness, VerifiedResult):
            raise TypeError("correctness must be a VerifiedResult")
        if not isinstance(self.dimensions, StudyDimensions):
            raise TypeError("dimensions must be StudyDimensions")
        bindings = dict(self.request.plan.environment.bindings)
        if bindings.get("BZ_A5_PROFILE_PHYSICAL_DEVICE") != str(
            self.request.device
        ):
            raise ValueError(
                "Phase 1 variants require one canonical device-bound execution plan"
            )
        if bindings.get("A5KERNEL_BLOCK_NUM") != str(
            self.dimensions.parallelism.value
        ):
            raise ValueError(
                "Phase 1 variants require block count to match typed parallelism"
            )
        if study_dimensions_from_plan(self.request.plan) != self.dimensions:
            raise ValueError(
                "study dimensions must match the host-owned execution-plan provenance"
            )

    @property
    def variant_id(self) -> str:
        return "variant-" + canonical_hash({
            "attempt_id": self.request.attempt_id,
            # Presets, not callers, own warmup and launch flags.  Bind only the
            # invariant executable/profile identity here.
            "device": self.request.device,
            "dimensions": self.dimensions.as_dict(),
            "execution_id": self.request.execution_id,
            "implementation": self.request.implementation,
            "kernel": self.request.expected_kernel,
            "source_fingerprint": self.request.source_fingerprint,
        })[:24]


@dataclass(frozen=True)
class StudyTimingSample:
    duration_us: float
    source_fingerprint: str
    execution_id: str
    replay_id: str
    evidence_sha256: str


class Phase1StudyBackend(Protocol):
    """Measurement authority; callers never provide durations, summaries, or flags."""

    def time_sample(self, command: TimingCommand) -> StudyTimingSample: ...

    def capture(self, command: CaptureCommand) -> ProfileCapture: ...


@dataclass(frozen=True)
class TimingStudyResult:
    variant_id: str
    raw_samples_us: tuple[float, ...]
    median_us: float
    coefficient_of_variation: float
    evidence_sha256: tuple[str, ...]


@dataclass(frozen=True)
class MetricStudyResult:
    variant_id: str
    domain: MetricDomain
    summary: tuple[tuple[str, str], ...]
    evidence_manifest_sha256: str


@dataclass(frozen=True)
class OptimizationComparison:
    baseline_variant_id: str
    speedup_by_variant: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Phase1StudyResult:
    preset: StudyPreset
    timing: tuple[TimingStudyResult, ...] = ()
    metrics: tuple[MetricStudyResult, ...] = ()
    comparison: OptimizationComparison | None = None


class Phase1PerformanceStudy:
    """Execute fixed Phase 1 presets against exact, correctness-approved variants."""

    TIMING_WARM_UP = 5
    TIMING_LAUNCH_COUNT = 20
    TIMING_SAMPLES = 3
    MAX_OPTIMIZATION_VARIANTS = 8

    def __init__(
        self,
        backend: Phase1StudyBackend,
        *,
        max_archive_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        self._backend = backend
        self._max_archive_bytes = max_archive_bytes

    def run(
        self, preset: StudyPreset, variants: Sequence[StudyVariant]
    ) -> Phase1StudyResult:
        if not isinstance(preset, StudyPreset):
            raise ValueError("preset must be a host-owned StudyPreset")
        variants = tuple(variants)
        if not variants:
            raise ValueError("a Phase 1 study requires at least one variant")
        if any(not isinstance(item, StudyVariant) for item in variants):
            raise TypeError("variants must contain only StudyVariant values")
        if preset is not StudyPreset.OPTIMIZATION_COMPARISON and len(variants) != 1:
            raise ValueError(f"{preset.value} requires exactly one variant")
        if preset is StudyPreset.OPTIMIZATION_COMPARISON and len(variants) < 2:
            raise ValueError("optimization-comparison requires at least two variants")
        if (
            preset is StudyPreset.OPTIMIZATION_COMPARISON
            and len(variants) > self.MAX_OPTIMIZATION_VARIANTS
        ):
            raise ValueError(
                "optimization-comparison accepts at most "
                f"{self.MAX_OPTIMIZATION_VARIANTS} variants"
            )
        if len({item.variant_id for item in variants}) != len(variants):
            raise ValueError("study variants must have distinct exact identities")
        if preset is StudyPreset.OPTIMIZATION_COMPARISON:
            if len({item.request.execution_id for item in variants}) != len(variants):
                raise ValueError(
                    "optimization variants must have distinct execution identities"
                )
            comparison_route = {
                (
                    item.request.device,
                    item.request.expected_kernel,
                    item.request.implementation,
                )
                for item in variants
            }
            if len(comparison_route) != 1:
                raise ValueError(
                    "optimization variants must share device, kernel, and implementation"
                )
            workload = {
                (
                    item.dimensions.shape,
                    item.dimensions.padding,
                    item.dimensions.access,
                )
                for item in variants
            }
            if len(workload) != 1:
                raise ValueError(
                    "optimization variants must share shape, padding, and access"
                )
            inputs = {
                (
                    item.request.request_id,
                    item.request.plan.input_a,
                    item.request.plan.input_b,
                )
                for item in variants
            }
            if len(inputs) != 1:
                raise ValueError(
                    "optimization variants must share the exact request and inputs"
                )
        for variant in variants:
            ProfilingTreatmentController._validate_correctness(
                variant.correctness, variant.request
            )

        if preset in (StudyPreset.TIMING_STUDY, StudyPreset.OPTIMIZATION_COMPARISON):
            timing = tuple(self._timing(preset, variant) for variant in variants)
            comparison = None
            if preset is StudyPreset.OPTIMIZATION_COMPARISON:
                baseline = timing[0]
                comparison = OptimizationComparison(
                    baseline.variant_id,
                    tuple(
                        (result.variant_id, baseline.median_us / result.median_us)
                        for result in timing
                    ),
                )
            return Phase1StudyResult(preset, timing=timing, comparison=comparison)

        variant = variants[0]
        basic = self._capture(preset, variant, ProfileMetric.BASIC_INFO)
        metrics = [self._metric_result(variant, basic)]
        if preset is StudyPreset.PIPE_UTILIZATION:
            if basic.exported_kernels.count(variant.request.expected_kernel) != 1:
                raise ValueError("BasicInfo did not identify the exact expected kernel once")
            pipe = self._capture(preset, variant, ProfileMetric.PIPE_UTILIZATION)
            metrics.append(self._metric_result(variant, pipe))
        return Phase1StudyResult(preset, metrics=tuple(metrics))

    def _timing(self, preset: StudyPreset, variant: StudyVariant) -> TimingStudyResult:
        request = replace(
            variant.request,
            warm_up=self.TIMING_WARM_UP,
            launch_count=self.TIMING_LAUNCH_COUNT,
        )
        samples = []
        for index in range(1, self.TIMING_SAMPLES + 1):
            command = TimingCommand(
                CampaignKind.FINAL,
                request,
                self._study_replay_id(preset, variant, MetricDomain.TIMING, index),
            )
            sample = self._backend.time_sample(command)
            if (
                not math.isfinite(sample.duration_us)
                or sample.duration_us <= 0
                or sample.source_fingerprint != request.source_fingerprint
                or sample.execution_id != request.execution_id
                or sample.replay_id != command.replay_id
                or _SHA256.fullmatch(sample.evidence_sha256) is None
            ):
                raise ValueError("timing sample does not match the exact study replay")
            samples.append(sample)
        values = tuple(sample.duration_us for sample in samples)
        mean = sum(values) / len(values)
        return TimingStudyResult(
            variant.variant_id,
            values,
            median(values),
            pstdev(values) / mean,
            tuple(sample.evidence_sha256 for sample in samples),
        )

    def _capture(
        self, preset: StudyPreset, variant: StudyVariant, metric: ProfileMetric
    ) -> ProfileCapture:
        request = replace(variant.request, warm_up=0, launch_count=1)
        command = CaptureCommand(
            CampaignKind.FINAL,
            request,
            metric,
            self._study_replay_id(
                preset,
                variant,
                MetricDomain.BASIC_INFO if metric is ProfileMetric.BASIC_INFO
                else MetricDomain.PIPE_UTILIZATION,
                1,
            ),
            kernel_name=(
                None if metric is ProfileMetric.BASIC_INFO else request.expected_kernel
            ),
        )
        capture = self._backend.capture(command)
        ProfilingTreatmentController._validate_capture(self, capture, command)
        ProfilingTreatmentController._validate_summary(capture, metric.value)
        if metric is ProfileMetric.BASIC_INFO:
            if capture.exported_kernels.count(request.expected_kernel) != 1:
                raise ValueError("BasicInfo did not identify the exact expected kernel once")
        elif capture.exported_kernels != (request.expected_kernel,):
            raise ValueError("PipeUtilization did not identify the exact expected kernel once")
        return capture

    @staticmethod
    def _metric_result(variant: StudyVariant, capture: ProfileCapture) -> MetricStudyResult:
        domain = (
            MetricDomain.BASIC_INFO
            if capture.metric is ProfileMetric.BASIC_INFO
            else MetricDomain.PIPE_UTILIZATION
        )
        return MetricStudyResult(
            variant.variant_id,
            domain,
            capture.summary,
            capture.evidence.manifest_sha256,
        )

    @staticmethod
    def _study_replay_id(
        preset: StudyPreset,
        variant: StudyVariant,
        domain: MetricDomain,
        ordinal: int,
    ) -> str:
        return "study-" + canonical_hash({
            "domain": domain.value,
            "ordinal": ordinal,
            "preset": preset.value,
            "variant_id": variant.variant_id,
        })
