"""Engine-neutral frozen inputs and evidence references for the canonical workflow.

References are opaque IDs/paths/hashes owned by the existing artifact, KDB and
skill stores. These types do not execute stages or establish evidence validity.
JSON loaders reject unknown fields so misspelled limits cannot silently disable
controls. ``None`` disables an individual cap; zero is not a disabled cap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
import math
from types import UnionType
from typing import get_args, get_origin, get_type_hints


class Stage(str, Enum):
    PREFLIGHT = "preflight"
    TASK_CONSTRUCTION = "task_construction"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TEST = "test"
    DEBUG = "debug"
    OPTIMISATION = "optimisation"
    VALIDATION = "validation"
    ACCEPTANCE = "acceptance"


class Mode(str, Enum):
    PHASE1 = "phase1"
    PHASE2 = "phase2"
    INFERENCE = "inference"


def _decode(kind, value):
    if get_origin(kind) is UnionType:
        if value is None and type(None) in get_args(kind):
            return None
        for option in get_args(kind):
            if option is type(None):
                continue
            try:
                return _decode(option, value)
            except ValueError:
                pass
        raise ValueError(f"invalid value for {kind}")
    if get_origin(kind) is tuple:
        if not isinstance(value, (tuple, list)):
            raise ValueError("expected an array")
        return tuple(_decode(get_args(kind)[0], item) for item in value)
    if isinstance(kind, type) and issubclass(kind, Enum):
        return kind(value)
    if is_dataclass(kind):
        return value if isinstance(value, kind) else kind.from_dict(value)
    if kind is float and type(value) in (int, float):
        return float(value)
    if type(value) is not kind:
        raise ValueError(f"expected {kind.__name__}")
    return value


class Spec:
    """Strict JSON conversion, also normalising direct constructor inputs."""

    def __post_init__(self):
        for name, kind in get_type_hints(type(self)).items():
            try:
                object.__setattr__(self, name, _decode(kind, getattr(self, name)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name}: {exc}") from exc

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError(f"{cls.__name__}: expected an object")
        unknown = value.keys() - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc

    def to_dict(self) -> dict:
        """Return a detached JSON-compatible representation."""
        def encode(value):
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [encode(item) for item in value]
            return value
        return encode(asdict(self))


def _required(**values):
    for name, value in values.items():
        if not value.strip():
            raise ValueError(f"{name} must not be blank")


@dataclass(frozen=True)
class BudgetLimits(Spec):
    """Actor time excludes setup/transfers; wall time includes all elapsed time.

    Runtime interval accounting/enforcement belongs to the workflow controller.
    Actor time includes thinking, retrieval, compilation, execution and profiling,
    including failures. Nested work must use the remaining parent allocation.
    """

    design_iterations: int | None = 3
    implementation_iterations: int | None = 5
    execution_tries: int | None = 20
    actor_seconds: float | None = 1800.0
    wall_seconds: float | None = None

    def __post_init__(self):
        super().__post_init__()
        for name, value in vars(self).items():
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite, or null")


@dataclass(frozen=True)
class ProjectSpec(Spec):
    """A task goal and argv executed relative to its snapshotted starting folder.

    Command success alone does not establish acceptance. Verification and
    performance goals remain part of the frozen task contract in artifact refs.
    """

    project_id: str
    goal: str
    command: tuple[str, ...]
    budgets: BudgetLimits = BudgetLimits()
    starting_folder: str | None = None
    repository_ref: str | None = None
    correctness_ref: str | None = None
    performance_goal: str | None = None

    def __post_init__(self):
        super().__post_init__()
        _required(project_id=self.project_id, goal=self.goal)
        if not self.command or any(not item.strip() for item in self.command):
            raise ValueError("command must contain nonblank argv entries")


@dataclass(frozen=True)
class CycleSpec(Spec):
    """Version-one cycle input; baseline refs identify immutable snapshots.

    A skill map ref points to a separately versioned stage-to-skill mapping.
    Provider/runtime settings remain external, referenced configuration.
    """

    cycle_id: str
    mode: Mode
    projects: tuple[ProjectSpec, ...]
    baseline_kdb: str
    baseline_skills: str
    skill_map_ref: str | None = None
    configuration_ref: str | None = None
    schema_version: int = 1

    def __post_init__(self):
        super().__post_init__()
        _required(cycle_id=self.cycle_id, baseline_kdb=self.baseline_kdb,
                  baseline_skills=self.baseline_skills)
        if self.schema_version != 1:
            raise ValueError("unsupported schema_version")
        if not self.projects:
            raise ValueError("cycle requires at least one project")
        if len({p.project_id for p in self.projects}) != len(self.projects):
            raise ValueError("project IDs must be unique within a cycle")


@dataclass(frozen=True)
class StepEvidence(Spec):
    """References to host-verifiable records, not agent assertions of success.

    Retrieval refs identify query receipts, including genuine empty results;
    the KDB retains its own citation schema and validation authority.
    """

    stage: Stage
    operation_id: str
    artifact_refs: tuple[str, ...] = ()
    skill_refs: tuple[str, ...] = ()
    retrieval_refs: tuple[str, ...] = ()
    retrospective_ref: str | None = None
    source_hash: str | None = None

    def __post_init__(self):
        super().__post_init__()
        _required(operation_id=self.operation_id)
