"""Bounded, deterministic A3 Ascend C Phase 1 curriculum registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from benchmarks.a3kernels.phase1_evidence import canonical_digest


A3_TARGET = "Ascend910B4"
A3_LANGUAGE = "ascend-c"
PLAN_SCHEMA = "a3-ascendc-phase1-dry-plan-v1"
MAX_PROJECTS = 8
WAVES = (1, 2, 3, 4)
CHECKPOINTS = (0, 4, 8)


class Coverage(str, Enum):
    FUNCTIONAL_CORRECTNESS = "functional-correctness"
    PADDED_MULTITILE = "padded-multitile"
    ORDINARY_RECOVERY = "ordinary-recovery"
    PERFORMANCE_KNEE = "performance-knee"
    CROSS_LAYER = "cross-layer"
    MSPROF_GUIDED = "msprof-guided"


class EvidencePreset(str, Enum):
    CORRECTNESS = "correctness"
    CORRECTNESS_TIMING = "correctness-timing"
    RECOVERY = "ordinary-recovery"
    MSPROF = "msprof-guided"


class ProjectFamily(str, Enum):
    VECTOR_ADD_BASELINE = "vector-add-baseline"
    PADDED_MULTITILE = "padded-multitile"
    COMPILE_RECOVERY = "compile-recovery"
    RUNTIME_RECOVERY = "runtime-recovery"
    LENGTH_KNEE = "length-knee"
    CROSS_LAYER_LAUNCH = "cross-layer-launch"
    MSPROF_PIPE = "msprof-pipe"


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    minimum: int | None = None
    maximum: int | None = None
    choices: tuple[int | str, ...] = ()

    def validate(self, value: object) -> int | str:
        if self.choices:
            if type(value) not in (int, str) or value not in self.choices:
                raise ValueError(f"parameter {self.name} must be one of {self.choices}")
            return value
        if type(value) is not int:
            raise ValueError(f"parameter {self.name} must be an integer")
        assert self.minimum is not None and self.maximum is not None
        if not self.minimum <= value <= self.maximum:
            raise ValueError(
                f"parameter {self.name} must be in [{self.minimum}, {self.maximum}]"
            )
        return value


@dataclass(frozen=True)
class RegisteredProject:
    family: ProjectFamily
    wave: int
    parameters: tuple[ParameterSpec, ...]
    evidence_presets: frozenset[EvidencePreset]
    coverage: frozenset[Coverage]

    def __post_init__(self) -> None:
        if self.wave not in WAVES:
            raise ValueError("registered project wave must be in 1..4")
        names = tuple(item.name for item in self.parameters)
        if not names or len(names) != len(set(names)):
            raise ValueError("registered parameters must be non-empty and unique")
        if not self.evidence_presets or not self.coverage:
            raise ValueError("registered projects require evidence and coverage")


PROJECT_REGISTRY = (
    RegisteredProject(
        ProjectFamily.VECTOR_ADD_BASELINE,
        1,
        (ParameterSpec("length", 32, 32),),
        frozenset({EvidencePreset.CORRECTNESS}),
        frozenset({Coverage.FUNCTIONAL_CORRECTNESS}),
    ),
    RegisteredProject(
        ProjectFamily.PADDED_MULTITILE,
        1,
        (ParameterSpec("length", 65, 400),),
        frozenset({EvidencePreset.CORRECTNESS}),
        frozenset({Coverage.PADDED_MULTITILE}),
    ),
    RegisteredProject(
        ProjectFamily.COMPILE_RECOVERY,
        2,
        (ParameterSpec("faults", 1, 2),),
        frozenset({EvidencePreset.RECOVERY}),
        frozenset({Coverage.ORDINARY_RECOVERY}),
    ),
    RegisteredProject(
        ProjectFamily.RUNTIME_RECOVERY,
        2,
        (ParameterSpec("faults", 1, 2),),
        frozenset({EvidencePreset.RECOVERY}),
        frozenset({Coverage.ORDINARY_RECOVERY}),
    ),
    RegisteredProject(
        ProjectFamily.LENGTH_KNEE,
        3,
        (ParameterSpec("length", choices=(1, 8, 16, 32, 64, 128, 256, 400)),),
        frozenset({EvidencePreset.CORRECTNESS_TIMING}),
        frozenset({Coverage.PERFORMANCE_KNEE}),
    ),
    RegisteredProject(
        ProjectFamily.CROSS_LAYER_LAUNCH,
        4,
        (ParameterSpec("block_count", 1, 8),),
        frozenset({EvidencePreset.CORRECTNESS_TIMING}),
        frozenset({Coverage.CROSS_LAYER}),
    ),
    RegisteredProject(
        ProjectFamily.MSPROF_PIPE,
        4,
        (ParameterSpec("metric", choices=("PipeUtilization",)),),
        frozenset({EvidencePreset.MSPROF}),
        frozenset({Coverage.MSPROF_GUIDED}),
    ),
)
_REGISTRY_BY_FAMILY = {item.family: item for item in PROJECT_REGISTRY}
_FAMILY_RANK = {item.family: rank for rank, item in enumerate(PROJECT_REGISTRY)}


@dataclass(frozen=True)
class Phase1Brief:
    max_projects: int = MAX_PROJECTS
    waves: tuple[int, ...] = WAVES
    checkpoints: tuple[int, ...] = CHECKPOINTS
    required_coverage: frozenset[Coverage] = frozenset(Coverage)
    performance_knee_minimum: int = 2


PHASE1_BRIEF = Phase1Brief()


@dataclass(frozen=True)
class CurriculumProposal:
    family: ProjectFamily
    parameters: tuple[tuple[str, int | str], ...]
    hypothesis: str
    evidence_preset: EvidencePreset
    target: str = A3_TARGET
    language: str = A3_LANGUAGE

    def __post_init__(self) -> None:
        if self.target != A3_TARGET:
            raise ValueError(f"A3 target must be {A3_TARGET}")
        if self.language != A3_LANGUAGE:
            raise ValueError("A3 projects support only Ascend C")
        if not isinstance(self.family, ProjectFamily):
            raise ValueError("family must be a registered ProjectFamily")
        registered = _REGISTRY_BY_FAMILY[self.family]
        if (
            not isinstance(self.evidence_preset, EvidencePreset)
            or self.evidence_preset not in registered.evidence_presets
        ):
            raise ValueError("evidence preset is not registered for this family")
        specs = {item.name: item for item in registered.parameters}
        if (
            not isinstance(self.parameters, tuple)
            or tuple(name for name, _ in self.parameters) != tuple(sorted(specs))
        ):
            raise ValueError("parameters must be the canonical registered parameter tuple")
        for name, value in self.parameters:
            specs[name].validate(value)
        if (
            type(self.hypothesis) is not str
            or not self.hypothesis.strip()
            or self.hypothesis != self.hypothesis.strip()
            or len(self.hypothesis) > 500
        ):
            raise ValueError("hypothesis must be a trimmed 1-500 character string")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CurriculumProposal":
        expected = {
            "target",
            "language",
            "family",
            "parameters",
            "hypothesis",
            "evidence_preset",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("proposal fields must be exactly " + ", ".join(sorted(expected)))
        if value["target"] != A3_TARGET:
            raise ValueError(f"A3 target must be {A3_TARGET}")
        if value["language"] != A3_LANGUAGE:
            raise ValueError("A3 projects support only Ascend C")
        try:
            family = ProjectFamily(value["family"])
            evidence = EvidencePreset(value["evidence_preset"])
        except (TypeError, ValueError) as exc:
            raise ValueError("proposal must select registered family and evidence preset") from exc
        registered = _REGISTRY_BY_FAMILY[family]
        if evidence not in registered.evidence_presets:
            raise ValueError("evidence preset is not registered for this family")
        raw = value["parameters"]
        if not isinstance(raw, Mapping):
            raise ValueError("parameters must be a JSON object")
        specs = {item.name: item for item in registered.parameters}
        if set(raw) != set(specs):
            raise ValueError("parameters must exactly match the registered family schema")
        parameters = tuple((name, specs[name].validate(raw[name])) for name in sorted(specs))
        return cls(family, parameters, value["hypothesis"], evidence)

    @property
    def registered(self) -> RegisteredProject:
        return _REGISTRY_BY_FAMILY[self.family]

    @property
    def identity(self) -> tuple[ProjectFamily, tuple[tuple[str, int | str], ...]]:
        return self.family, self.parameters

    @property
    def project_id(self) -> str:
        return canonical_digest(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "language": self.language,
            "family": self.family.value,
            "parameters": dict(self.parameters),
            "hypothesis": self.hypothesis,
            "evidence_preset": self.evidence_preset.value,
        }


def _proposal(
    family: str, parameters: Mapping[str, int | str], hypothesis: str, evidence: str
) -> CurriculumProposal:
    return CurriculumProposal.from_mapping(
        {
            "target": A3_TARGET,
            "language": A3_LANGUAGE,
            "family": family,
            "parameters": parameters,
            "hypothesis": hypothesis,
            "evidence_preset": evidence,
        }
    )


DEFAULT_PROPOSALS = (
    _proposal("vector-add-baseline", {"length": 32}, "Establish native Ascend C functional correctness.", "correctness"),
    _proposal("padded-multitile", {"length": 400}, "Expose tail and multi-tile mistakes at a padded extent.", "correctness"),
    _proposal("compile-recovery", {"faults": 1}, "Recover from one ordinary Ascend C compiler failure.", "ordinary-recovery"),
    _proposal("runtime-recovery", {"faults": 1}, "Diagnose and recover from one ordinary device failure.", "ordinary-recovery"),
    _proposal("length-knee", {"length": 16}, "Sample the sub-32 native execution performance knee.", "correctness-timing"),
    _proposal("length-knee", {"length": 400}, "Sample the upper native execution performance knee.", "correctness-timing"),
    _proposal("cross-layer-launch", {"block_count": 1}, "Measure the host-to-kernel launch boundary.", "correctness-timing"),
    _proposal("msprof-pipe", {"metric": "PipeUtilization"}, "Use one bounded msprof observation to guide revision.", "msprof-guided"),
)


def _validated(
    proposals: Sequence[CurriculumProposal],
) -> tuple[CurriculumProposal, ...]:
    rows = tuple(proposals)
    if len(rows) > MAX_PROJECTS:
        raise ValueError(f"Phase 1 accepts at most {MAX_PROJECTS} projects")
    if any(not isinstance(item, CurriculumProposal) for item in rows):
        raise ValueError("proposals must contain only CurriculumProposal values")
    identities = [item.identity for item in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate curriculum proposals are not allowed")
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                item.registered.wave,
                _FAMILY_RANK[item.family],
                item.parameters,
            ),
        )
    )


def saturation_status(proposals: Sequence[CurriculumProposal]) -> str:
    rows = _validated(proposals)
    if len(rows) not in CHECKPOINTS:
        return "IN_PROGRESS"
    coverage = {tag for row in rows for tag in row.registered.coverage}
    knees = sum(Coverage.PERFORMANCE_KNEE in row.registered.coverage for row in rows)
    if PHASE1_BRIEF.required_coverage <= coverage and knees >= 2:
        return "SATURATED"
    return "CONTINUE"


def dry_run_plan(
    proposals: Sequence[CurriculumProposal] = DEFAULT_PROPOSALS,
) -> dict[str, object]:
    """Build the complete plan without model, profiler, or device side effects."""

    rows = _validated(proposals)
    projects = [
        {
            "ordinal": ordinal,
            "wave": row.registered.wave,
            "project_id": row.project_id,
            **row.as_dict(),
            "coverage": sorted(tag.value for tag in row.registered.coverage),
        }
        for ordinal, row in enumerate(rows, 1)
    ]
    body = {
        "schema": PLAN_SCHEMA,
        "mode": "dry-run",
        "target": A3_TARGET,
        "language": A3_LANGUAGE,
        "budget": {"maximum": MAX_PROJECTS, "planned": len(rows)},
        "waves": list(WAVES),
        "checkpoints": list(CHECKPOINTS),
        "projects": projects,
        "checkpoint_status": {
            str(point): saturation_status(rows[:point])
            for point in CHECKPOINTS
            if point <= len(rows)
        },
        "terminal_status": saturation_status(rows),
    }
    return {**body, "plan_id": canonical_digest(body)}
