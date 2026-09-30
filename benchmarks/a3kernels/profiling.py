"""Deterministic A3 timing and compact msprof evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
import statistics
from typing import Callable

from benchmarks.a3kernels.phase1_evidence import canonical_digest


_SHA256 = re.compile(r"[0-9a-f]{64}")
_REPLAY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_TIMING = re.compile(r"^A3TIMING_US=(.+)$")


def _digest(value: object, label: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256 digest")


class ProfilingTreatment(str, Enum):
    OFF = "without-profiling-guidance"
    ON = "with-profiling-guidance"


class ProfileMetric(str, Enum):
    BASIC = "Basic"
    PIPE_UTILIZATION = "PipeUtilization"


@dataclass(frozen=True)
class CandidateBinding:
    execution_id: str
    source_fingerprint: str

    def __post_init__(self) -> None:
        _digest(self.execution_id, "candidate execution_id")
        _digest(self.source_fingerprint, "candidate source_fingerprint")

    def as_dict(self) -> dict[str, str]:
        return {
            "execution_id": self.execution_id,
            "source_fingerprint": self.source_fingerprint,
        }


@dataclass(frozen=True)
class StudyDimensions:
    length: int
    block_count: int
    warm_up: int
    launch_count: int

    def __post_init__(self) -> None:
        values = (self.length, self.block_count, self.warm_up, self.launch_count)
        if any(type(value) is not int for value in values):
            raise ValueError("A3 study dimensions must be integers")
        if not 1 <= self.length <= 4096:
            raise ValueError("length must be in [1, 4096]")
        if not 1 <= self.block_count <= 32:
            raise ValueError("block_count must be in [1, 32]")
        if not 0 <= self.warm_up <= 1000:
            raise ValueError("warm_up must be in [0, 1000]")
        if not 1 <= self.launch_count <= 10000:
            raise ValueError("launch_count must be in [1, 10000]")

    def as_dict(self) -> dict[str, int]:
        return {
            "length": self.length,
            "block_count": self.block_count,
            "warm_up": self.warm_up,
            "launch_count": self.launch_count,
        }


@dataclass(frozen=True)
class ProfileRequest:
    binding: CandidateBinding
    dimensions: StudyDimensions
    metric: ProfileMetric
    treatment: ProfilingTreatment = ProfilingTreatment.ON
    expected_kernel: str = "vector_add"
    target: str = "Ascend910B4"
    language: str = "ascend-c"
    execution_profile: str = "gz-a3"
    runtime: str = "native-ascend-c"

    def __post_init__(self) -> None:
        if not isinstance(self.binding, CandidateBinding):
            raise TypeError("binding must be CandidateBinding")
        if not isinstance(self.dimensions, StudyDimensions):
            raise TypeError("dimensions must be StudyDimensions")
        if not isinstance(self.metric, ProfileMetric) or not isinstance(
            self.treatment, ProfilingTreatment
        ):
            raise TypeError("profiling request enums must be registered values")
        if self.expected_kernel != "vector_add":
            raise ValueError("A3 profiling requires the registered vector_add kernel")
        if (
            self.target != "Ascend910B4"
            or self.language != "ascend-c"
            or self.execution_profile != "gz-a3"
            or self.runtime != "native-ascend-c"
        ):
            raise ValueError("A3 profiling identity must use native Ascend C on gz-a3")

    def as_dict(self) -> dict[str, object]:
        return {
            "binding": self.binding.as_dict(),
            "dimensions": self.dimensions.as_dict(),
            "metric": self.metric.value,
            "treatment": self.treatment.value,
            "expected_kernel": self.expected_kernel,
            "target": self.target,
            "language": self.language,
            "execution_profile": self.execution_profile,
            "runtime": self.runtime,
        }

    @property
    def request_id(self) -> str:
        return canonical_digest(self.as_dict())

    @property
    def default_replay_id(self) -> str:
        return f"a3-profile-{self.request_id[:16]}"


def parse_timing_output(stdout: str) -> tuple[float, ...]:
    values = []
    for line in stdout.splitlines():
        match = _TIMING.fullmatch(line)
        if match is None:
            continue
        try:
            value = float(match.group(1))
        except ValueError as exc:
            raise ValueError("timing output contains a non-numeric sample") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("timing samples must be finite and positive")
        values.append(value)
    if not values:
        raise ValueError("timing output contains no A3TIMING_US samples")
    return tuple(values)


@dataclass(frozen=True)
class TimingResult:
    request_id: str
    candidate_execution_id: str
    source_fingerprint: str
    dimensions: StudyDimensions
    samples_us: tuple[float, ...]
    minimum_us: float
    maximum_us: float
    mean_us: float
    median_us: float
    evidence_sha256: str

    @classmethod
    def from_samples(
        cls,
        binding: CandidateBinding,
        dimensions: StudyDimensions,
        samples: tuple[float, ...],
    ) -> "TimingResult":
        if not samples or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in samples
        ):
            raise ValueError("timing samples must be finite and positive")
        values = tuple(float(value) for value in samples)
        body = {
            "request_id": timing_request_id(binding, dimensions),
            "candidate_execution_id": binding.execution_id,
            "source_fingerprint": binding.source_fingerprint,
            "dimensions": dimensions.as_dict(),
            "samples_us": values,
            "minimum_us": min(values),
            "maximum_us": max(values),
            "mean_us": statistics.fmean(values),
            "median_us": statistics.median(values),
        }
        return cls(**body, evidence_sha256=canonical_digest(body))


def timing_request_id(
    binding: CandidateBinding, dimensions: StudyDimensions
) -> str:
    return canonical_digest(
        {
            "kind": "a3-timing",
            "binding": binding.as_dict(),
            "dimensions": dimensions.as_dict(),
            "target": "Ascend910B4",
            "language": "ascend-c",
            "execution_profile": "gz-a3",
            "runtime": "native-ascend-c",
        }
    )


@dataclass(frozen=True)
class CompactProfileResult:
    request_id: str
    candidate_execution_id: str
    source_fingerprint: str
    metric: ProfileMetric
    exported_kernels: tuple[str, ...]
    metric_values: tuple[tuple[str, float], ...]
    timeline: tuple[tuple[str, float], ...]
    report_sha256: str
    evidence_sha256: str

    @classmethod
    def create(
        cls,
        request: ProfileRequest,
        *,
        exported_kernels: tuple[str, ...],
        metric_values: tuple[tuple[str, float], ...],
        timeline: tuple[tuple[str, float], ...],
        report_sha256: str,
    ) -> "CompactProfileResult":
        _digest(report_sha256, "msprof report")
        kernels = tuple(exported_kernels)
        metrics = _pairs(metric_values, "metric")
        timeline_values = _pairs(timeline, "timeline")
        if request.expected_kernel not in kernels or len(kernels) != len(set(kernels)):
            raise ValueError("compact profile must contain the exact expected kernel")
        body = {
            "request_id": request.request_id,
            "candidate_execution_id": request.binding.execution_id,
            "source_fingerprint": request.binding.source_fingerprint,
            "metric": request.metric,
            "exported_kernels": kernels,
            "metric_values": metrics,
            "timeline": timeline_values,
            "report_sha256": report_sha256,
        }
        return cls(**body, evidence_sha256=canonical_digest(body))


def _pairs(values: tuple[tuple[str, float], ...], label: str) -> tuple[tuple[str, float], ...]:
    result = tuple((name, float(value)) for name, value in values)
    if (
        not result
        or any(type(name) is not str or not name or not math.isfinite(value) for name, value in result)
        or len({name for name, _ in result}) != len(result)
    ):
        raise ValueError(f"compact {label} values must be unique finite named pairs")
    return tuple(sorted(result))


TimingBackend = Callable[[CandidateBinding, StudyDimensions], str]
ProfilerBackend = Callable[[ProfileRequest], CompactProfileResult]


class A3ProfilingSession:
    def __init__(
        self,
        *,
        timing: TimingBackend | None = None,
        profiler: ProfilerBackend | None = None,
    ) -> None:
        self._timing = timing
        self._profiler = profiler
        self._timing_replays: dict[str, tuple[str, TimingResult]] = {}
        self._replays: dict[str, tuple[str, CompactProfileResult]] = {}

    def time(
        self,
        binding: CandidateBinding,
        dimensions: StudyDimensions,
        *,
        replay_id: str | None = None,
    ) -> TimingResult:
        request_id = timing_request_id(binding, dimensions)
        selected_replay = f"a3-timing-{request_id[:16]}" if replay_id is None else replay_id
        if _REPLAY_ID.fullmatch(selected_replay) is None:
            raise ValueError("timing replay_id is unsafe")
        prior = self._timing_replays.get(selected_replay)
        if prior is not None:
            if prior[0] != request_id:
                raise ValueError("conflicting timing replay request")
            return prior[1]
        if self._timing is None:
            raise RuntimeError("A3 timing backend is unavailable")
        result = TimingResult.from_samples(
            binding, dimensions, parse_timing_output(self._timing(binding, dimensions))
        )
        self._timing_replays[selected_replay] = (request_id, result)
        return result

    def profile(
        self, request: ProfileRequest, *, replay_id: str | None = None
    ) -> CompactProfileResult | None:
        if request.treatment is ProfilingTreatment.OFF:
            return None
        selected_replay = request.default_replay_id if replay_id is None else replay_id
        if _REPLAY_ID.fullmatch(selected_replay) is None:
            raise ValueError("profiling replay_id is unsafe")
        prior = self._replays.get(selected_replay)
        if prior is not None:
            if prior[0] != request.request_id:
                raise ValueError("conflicting profiling replay request")
            return prior[1]
        if self._profiler is None:
            raise RuntimeError("A3 profiler backend is unavailable")
        result = self._profiler(request)
        if (
            not isinstance(result, CompactProfileResult)
            or result.request_id != request.request_id
            or result.candidate_execution_id != request.binding.execution_id
            or result.source_fingerprint != request.binding.source_fingerprint
        ):
            raise ValueError("profiler result does not match the A3 request")
        self._replays[selected_replay] = (request.request_id, result)
        return result


__all__ = [
    "A3ProfilingSession", "CandidateBinding", "CompactProfileResult",
    "ProfileMetric", "ProfileRequest", "ProfilingTreatment", "StudyDimensions",
    "TimingResult", "parse_timing_output", "timing_request_id",
]
