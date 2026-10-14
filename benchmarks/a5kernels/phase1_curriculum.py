"""Model-neutral contracts for agent-authored A5 Phase-1 curriculum waves.

The contract deliberately describes generic Catlass capabilities rather than
benchmark programs.  A later execution layer may translate admitted projects
into host-owned inputs and oracles; no command, source, or benchmark payload is
accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Mapping, Sequence

from benchmarks.a5kernels.evidence import canonical_digest


_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROJECT_FIELDS = {
    "project_id", "capabilities", "shape", "dtype", "block_count",
    "padding_multiple", "failure_mode", "timing_lengths", "profile_metric",
    "hypothesis",
}


class Capability(str, Enum):
    """Bounded generic skills from which Curriculum may compose a project."""

    GM_UB_MOVEMENT = "gm-ub-movement"
    SIMD_VECTOR = "simd-vector"
    PADDED_TAIL = "padded-tail"
    SIMT_REDUCTION = "simt-reduction"
    SIMT_MASK = "simt-mask"
    CUBE_TILING = "cube-tiling"
    FAILURE_REPAIR = "failure-repair"
    TIMING_SWEEP = "timing-sweep"
    PROFILING = "profiling"


class DType(str, Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class FailureMode(str, Enum):
    NONE = "none"
    COMPILE = "compile"
    HOST_MISMATCH = "host-mismatch"


class ProfileMetric(str, Enum):
    PIPE_UTILIZATION = "pipe-utilization"
    MEMORY_THROUGHPUT = "memory-throughput"
    TIMELINE = "timeline"


def _enum(value: object, enum: type[Enum], field: str):
    try:
        return enum(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unregistered {field}") from exc


def _integers(
    value: object, field: str, *, minimum: int, maximum: int, lengths: range,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) not in lengths:
        raise ValueError(f"{field} has an invalid number of values")
    result = tuple(value)
    if any(type(item) is not int or not minimum <= item <= maximum for item in result):
        raise ValueError(f"{field} values must be integers in [{minimum}, {maximum}]")
    return result


@dataclass(frozen=True)
class ProjectSpec:
    """A command-free project selected from host-supported Catlass primitives."""

    project_id: str
    capabilities: tuple[Capability, ...]
    shape: tuple[int, ...]
    dtype: DType
    block_count: int
    padding_multiple: int
    failure_mode: FailureMode
    timing_lengths: tuple[int, ...]
    profile_metric: ProfileMetric | None
    hypothesis: str

    def __post_init__(self) -> None:
        if not isinstance(self.project_id, str) or not _ID.fullmatch(self.project_id):
            raise ValueError("project_id must be a lowercase kebab identifier")
        if (
            not self.capabilities
            or any(not isinstance(item, Capability) for item in self.capabilities)
            or len(set(self.capabilities)) != len(self.capabilities)
        ):
            raise ValueError("capabilities must be non-empty, registered, and unique")
        if not 1 <= len(self.shape) <= 3 or any(
            type(item) is not int or not 1 <= item <= 16384 for item in self.shape
        ):
            raise ValueError("shape must contain one to three bounded dimensions")
        if not isinstance(self.dtype, DType):
            raise ValueError("dtype must be registered")
        if type(self.block_count) is not int or not 1 <= self.block_count <= 32:
            raise ValueError("block_count must be in [1, 32]")
        if self.padding_multiple not in {1, 16, 32, 64, 128, 256}:
            raise ValueError("padding_multiple must be a registered alignment")
        if not isinstance(self.failure_mode, FailureMode):
            raise ValueError("failure mode must be registered")
        if any(
            type(item) is not int or not 1 <= item <= 16384
            for item in self.timing_lengths
        ) or tuple(sorted(set(self.timing_lengths))) != self.timing_lengths:
            raise ValueError("timing lengths must be unique ascending bounded integers")
        if self.profile_metric is not None and not isinstance(
            self.profile_metric, ProfileMetric
        ):
            raise ValueError("profile metric must be registered")
        if (
            not isinstance(self.hypothesis, str)
            or not self.hypothesis.strip()
            or self.hypothesis != self.hypothesis.strip()
            or len(self.hypothesis) > 500
        ):
            raise ValueError("hypothesis must be a trimmed 1-500 character string")
        self._validate_composition()

    def _validate_composition(self) -> None:
        capabilities = set(self.capabilities)
        repairs = Capability.FAILURE_REPAIR in capabilities
        if repairs != (self.failure_mode is not FailureMode.NONE):
            raise ValueError("failure-repair capability and failure mode must agree")
        timing = Capability.TIMING_SWEEP in capabilities
        if timing and len(self.timing_lengths) < 2:
            raise ValueError("timing-sweep requires at least two timing lengths")
        if not timing and self.timing_lengths:
            raise ValueError("timing lengths require timing-sweep capability")
        profiling = Capability.PROFILING in capabilities
        if profiling != (self.profile_metric is not None):
            raise ValueError("profiling capability and profile metric must agree")
        if Capability.CUBE_TILING in capabilities and len(self.shape) < 2:
            raise ValueError("cube-tiling requires a matrix-shaped project")
        if (
            Capability.PADDED_TAIL in capabilities
            and self.shape[-1] % self.padding_multiple == 0
        ):
            raise ValueError("padded-tail requires a non-aligned logical extent")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ProjectSpec":
        if not isinstance(value, Mapping) or set(value) != _PROJECT_FIELDS:
            raise ValueError(
                "project fields must be exactly " + ", ".join(sorted(_PROJECT_FIELDS))
            )
        raw_capabilities = value["capabilities"]
        if not isinstance(raw_capabilities, (list, tuple)):
            raise ValueError("capabilities must be a list")
        capabilities = tuple(
            _enum(item, Capability, "capability") for item in raw_capabilities
        )
        shape = _integers(
            value["shape"], "shape", minimum=1, maximum=16384, lengths=range(1, 4)
        )
        timing = _integers(
            value["timing_lengths"], "timing lengths",
            minimum=1, maximum=16384, lengths=range(0, 9),
        )
        metric = value["profile_metric"]
        return cls(
            project_id=value["project_id"],
            capabilities=capabilities,
            shape=shape,
            dtype=_enum(value["dtype"], DType, "dtype"),
            block_count=value["block_count"],
            padding_multiple=value["padding_multiple"],
            failure_mode=_enum(value["failure_mode"], FailureMode, "failure mode"),
            timing_lengths=timing,
            profile_metric=(
                None if metric is None else _enum(metric, ProfileMetric, "profile metric")
            ),
            hypothesis=value["hypothesis"],
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "capabilities": [item.value for item in self.capabilities],
            "shape": list(self.shape),
            "dtype": self.dtype.value,
            "block_count": self.block_count,
            "padding_multiple": self.padding_multiple,
            "failure_mode": self.failure_mode.value,
            "timing_lengths": list(self.timing_lengths),
            "profile_metric": (
                None if self.profile_metric is None else self.profile_metric.value
            ),
            "hypothesis": self.hypothesis,
        }


@dataclass(frozen=True)
class WaveProject:
    """One authored project whose dependencies precede the entire wave."""

    spec: ProjectSpec
    requires: tuple[Capability, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ProjectSpec):
            raise ValueError("wave projects require a ProjectSpec")
        if any(not isinstance(item, Capability) for item in self.requires) or len(
            set(self.requires)
        ) != len(self.requires):
            raise ValueError("requires must contain unique registered capabilities")

    def as_dict(self) -> dict[str, object]:
        return {
            "spec": self.spec.as_dict(),
            "requires": [item.value for item in self.requires],
        }


@dataclass(frozen=True)
class ProjectAssignment:
    project_id: str
    authored_ordinal: int
    pre_wave_memory_sha256: str
    spec: ProjectSpec


@dataclass(frozen=True)
class WaveSpec:
    """An immutable semantic wave; host capacity does not change membership."""

    wave_id: str
    pre_wave_memory_sha256: str
    pre_wave_coverage: tuple[Capability, ...]
    projects: tuple[WaveProject, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.wave_id, str) or not _ID.fullmatch(self.wave_id):
            raise ValueError("wave_id must be a lowercase kebab identifier")
        if not isinstance(self.pre_wave_memory_sha256, str) or not _SHA256.fullmatch(
            self.pre_wave_memory_sha256
        ):
            raise ValueError("pre-wave memory identity must be a SHA-256 digest")
        if any(
            not isinstance(item, Capability) for item in self.pre_wave_coverage
        ) or len(set(self.pre_wave_coverage)) != len(self.pre_wave_coverage):
            raise ValueError("pre-wave coverage must contain unique capabilities")
        if not self.projects or any(
            not isinstance(item, WaveProject) for item in self.projects
        ):
            raise ValueError("wave must contain authored projects")
        ids = tuple(item.spec.project_id for item in self.projects)
        if len(set(ids)) != len(ids):
            raise ValueError("project IDs must be unique within a wave")
        available = set(self.pre_wave_coverage)
        for project in self.projects:
            if not set(project.requires) <= available:
                raise ValueError("project dependencies must exist in pre-wave coverage")

    @property
    def semantic_width(self) -> int:
        return len(self.projects)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(self.as_dict())

    def assignments(self) -> tuple[ProjectAssignment, ...]:
        return tuple(
            ProjectAssignment(
                item.spec.project_id, ordinal, self.pre_wave_memory_sha256, item.spec
            )
            for ordinal, item in enumerate(self.projects, 1)
        )

    def execution_batches(
        self, max_parallel: int,
    ) -> tuple[tuple[ProjectAssignment, ...], ...]:
        """Apply host capacity without changing semantic wave membership."""

        if type(max_parallel) is not int or max_parallel < 1:
            raise ValueError("max_parallel must be a positive integer")
        assignments = self.assignments()
        return tuple(
            assignments[start:start + max_parallel]
            for start in range(0, len(assignments), max_parallel)
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "WaveSpec":
        expected = {
            "wave_id", "pre_wave_memory_sha256", "pre_wave_coverage", "projects",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("wave fields must be exactly " + ", ".join(sorted(expected)))
        raw_coverage = value["pre_wave_coverage"]
        raw_projects = value["projects"]
        if not isinstance(raw_coverage, list) or not isinstance(raw_projects, list):
            raise ValueError("wave coverage and projects must be lists")
        projects = []
        for item in raw_projects:
            if not isinstance(item, Mapping) or set(item) != {"spec", "requires"}:
                raise ValueError("wave project fields must be exactly requires, spec")
            if not isinstance(item["spec"], Mapping) or not isinstance(
                item["requires"], list
            ):
                raise ValueError("wave project spec and requirements are invalid")
            projects.append(WaveProject(
                ProjectSpec.from_mapping(item["spec"]),
                tuple(_enum(tag, Capability, "requirement") for tag in item["requires"]),
            ))
        return cls(
            wave_id=value["wave_id"],
            pre_wave_memory_sha256=value["pre_wave_memory_sha256"],
            pre_wave_coverage=tuple(
                _enum(item, Capability, "coverage") for item in raw_coverage
            ),
            projects=tuple(projects),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "wave_id": self.wave_id,
            "pre_wave_memory_sha256": self.pre_wave_memory_sha256,
            "pre_wave_coverage": [item.value for item in self.pre_wave_coverage],
            "projects": [item.as_dict() for item in self.projects],
        }


@dataclass(frozen=True)
class ProjectOutcome:
    project_id: str
    memory_delta_sha256: str
    verified: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.project_id, str) or not _ID.fullmatch(self.project_id):
            raise ValueError("outcome project_id is invalid")
        if not isinstance(self.memory_delta_sha256, str) or not _SHA256.fullmatch(
            self.memory_delta_sha256
        ):
            raise ValueError("memory delta identity must be a SHA-256 digest")
        if type(self.verified) is not bool:
            raise ValueError("verified must be boolean")


@dataclass(frozen=True)
class ConsolidationCommit:
    project_id: str
    authored_ordinal: int
    commit_ordinal: int
    memory_delta_sha256: str


@dataclass(frozen=True)
class WaveConsolidation:
    wave_id: str
    pre_wave_memory_sha256: str
    commits: tuple[ConsolidationCommit, ...]


def consolidate_wave(
    wave: WaveSpec, outcomes: Sequence[ProjectOutcome],
) -> WaveConsolidation:
    """Validate a complete barrier, then describe atomic authored-order commits."""

    if not isinstance(wave, WaveSpec):
        raise ValueError("consolidation requires a WaveSpec")
    rows = tuple(outcomes)
    expected = tuple(item.spec.project_id for item in wave.projects)
    if (
        len(rows) != len(expected)
        or any(not isinstance(item, ProjectOutcome) for item in rows)
        or len({item.project_id for item in rows}) != len(rows)
        or {item.project_id for item in rows} != set(expected)
    ):
        raise ValueError("memory publication requires a complete wave")
    by_id = {item.project_id: item for item in rows}
    if any(not by_id[project_id].verified for project_id in expected):
        raise ValueError("every project must be verified before memory publication")
    commits = tuple(
        ConsolidationCommit(
            project_id, authored_ordinal, commit_ordinal,
            by_id[project_id].memory_delta_sha256,
        )
        for commit_ordinal, (authored_ordinal, project_id) in enumerate(
            enumerate(expected, 1), 1
        )
    )
    return WaveConsolidation(wave.wave_id, wave.pre_wave_memory_sha256, commits)


__all__ = [
    "Capability", "ConsolidationCommit", "DType", "FailureMode",
    "ProfileMetric", "ProjectAssignment", "ProjectOutcome", "ProjectSpec",
    "WaveConsolidation", "WaveProject", "WaveSpec", "consolidate_wave",
]
