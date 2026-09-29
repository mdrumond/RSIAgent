"""Immutable, host-owned curriculum registry for the bounded A5 Phase1 pilot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Mapping, Sequence


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
    choices: tuple[str, ...] = ()

    def validate(self, value: object) -> int | str:
        if self.choices:
            if not isinstance(value, str) or value not in self.choices:
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
        if not self.parameters or len({item.name for item in self.parameters}) != len(
            self.parameters
        ):
            raise ValueError("registered project parameters must be non-empty and unique")
        if not self.evidence_presets or not self.coverage:
            raise ValueError("registered projects require evidence and coverage")


PROJECT_REGISTRY = (
    RegisteredProject(
        ProjectFamily.VECTOR_ADD_BASELINE, 1,
        (ParameterSpec("length", 32, 32),),
        frozenset({EvidencePreset.CORRECTNESS}),
        frozenset({Coverage.FUNCTIONAL_CORRECTNESS}),
    ),
    RegisteredProject(
        ProjectFamily.PADDED_MULTITILE, 1,
        (ParameterSpec("length", 65, 400),),
        frozenset({EvidencePreset.CORRECTNESS}),
        frozenset({Coverage.PADDED_MULTITILE}),
    ),
    RegisteredProject(
        ProjectFamily.COMPILE_RECOVERY, 2,
        (ParameterSpec("faults", 1, 2),),
        frozenset({EvidencePreset.RECOVERY}),
        frozenset({Coverage.ORDINARY_RECOVERY}),
    ),
    RegisteredProject(
        ProjectFamily.RUNTIME_RECOVERY, 2,
        (ParameterSpec("faults", 1, 2),),
        frozenset({EvidencePreset.RECOVERY}),
        frozenset({Coverage.ORDINARY_RECOVERY}),
    ),
    RegisteredProject(
        ProjectFamily.LENGTH_KNEE, 3,
        (ParameterSpec("length", 32, 400),),
        frozenset({EvidencePreset.CORRECTNESS_TIMING}),
        frozenset({Coverage.PERFORMANCE_KNEE}),
    ),
    RegisteredProject(
        ProjectFamily.CROSS_LAYER_LAUNCH, 4,
        (ParameterSpec("block_count", 1, 8),),
        frozenset({EvidencePreset.CORRECTNESS_TIMING}),
        frozenset({Coverage.CROSS_LAYER}),
    ),
    RegisteredProject(
        ProjectFamily.MSPROF_PIPE, 4,
        (ParameterSpec("metric", choices=("PipeUtilization",)),),
        frozenset({EvidencePreset.MSPROF}),
        frozenset({Coverage.MSPROF_GUIDED}),
    ),
)
_REGISTRY_BY_FAMILY = {item.family: item for item in PROJECT_REGISTRY}


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

    def __post_init__(self) -> None:
        if not isinstance(self.family, ProjectFamily):
            raise ValueError("family must be a registered ProjectFamily")
        registered = _REGISTRY_BY_FAMILY[self.family]
        if not isinstance(self.evidence_preset, EvidencePreset):
            raise ValueError("evidence_preset must be an EvidencePreset")
        if self.evidence_preset not in registered.evidence_presets:
            raise ValueError("evidence preset is not registered for this family")
        specs = {item.name: item for item in registered.parameters}
        if (
            not isinstance(self.parameters, tuple)
            or tuple(name for name, _value in self.parameters) != tuple(sorted(specs))
        ):
            raise ValueError("parameters must be the canonical registered parameter tuple")
        for name, value in self.parameters:
            specs[name].validate(value)
        if (
            not isinstance(self.hypothesis, str)
            or not self.hypothesis.strip()
            or self.hypothesis != self.hypothesis.strip()
            or len(self.hypothesis) > 500
        ):
            raise ValueError("hypothesis must be a trimmed 1-500 character string")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CurriculumProposal":
        if not isinstance(value, Mapping):
            raise ValueError("each proposal must be a JSON object")
        expected = {"family", "parameters", "hypothesis", "evidence_preset"}
        if set(value) != expected:
            raise ValueError("proposal fields must be exactly " + ", ".join(sorted(expected)))
        try:
            family = ProjectFamily(value["family"])
            evidence = EvidencePreset(value["evidence_preset"])
        except (TypeError, ValueError) as exc:
            raise ValueError("proposal must select registered family and evidence preset") from exc
        registered = _REGISTRY_BY_FAMILY[family]
        if evidence not in registered.evidence_presets:
            raise ValueError("evidence preset is not registered for this family")
        raw_parameters = value["parameters"]
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("parameters must be a JSON object")
        specs = {item.name: item for item in registered.parameters}
        if set(raw_parameters) != set(specs):
            raise ValueError("parameters must exactly match the registered family schema")
        parameters = tuple(
            (name, specs[name].validate(raw_parameters[name])) for name in sorted(specs)
        )
        return cls(family, parameters, value["hypothesis"], evidence)

    @property
    def registered(self) -> RegisteredProject:
        return _REGISTRY_BY_FAMILY[self.family]

    @property
    def identity(self) -> tuple[ProjectFamily, tuple[tuple[str, int | str], ...]]:
        return self.family, self.parameters

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "parameters": dict(self.parameters),
            "hypothesis": self.hypothesis,
            "evidence_preset": self.evidence_preset.value,
        }


DEFAULT_PROPOSALS = (
    CurriculumProposal.from_mapping({
        "family": "vector-add-baseline", "parameters": {"length": 32},
        "hypothesis": "The registered baseline establishes functional correctness.",
        "evidence_preset": "correctness",
    }),
    CurriculumProposal.from_mapping({
        "family": "padded-multitile", "parameters": {"length": 400},
        "hypothesis": "A padded multi-tile extent exposes tail and tiling mistakes.",
        "evidence_preset": "correctness",
    }),
    CurriculumProposal.from_mapping({
        "family": "compile-recovery", "parameters": {"faults": 1},
        "hypothesis": "One ordinary compiler failure should permit a corrected retry.",
        "evidence_preset": "ordinary-recovery",
    }),
    CurriculumProposal.from_mapping({
        "family": "runtime-recovery", "parameters": {"faults": 1},
        "hypothesis": "One ordinary runtime failure should remain diagnosable and recoverable.",
        "evidence_preset": "ordinary-recovery",
    }),
    CurriculumProposal.from_mapping({
        "family": "length-knee", "parameters": {"length": 128},
        "hypothesis": "The first registered size samples the lower performance knee.",
        "evidence_preset": "correctness-timing",
    }),
    CurriculumProposal.from_mapping({
        "family": "length-knee", "parameters": {"length": 400},
        "hypothesis": "The second registered size samples the upper performance knee.",
        "evidence_preset": "correctness-timing",
    }),
    CurriculumProposal.from_mapping({
        "family": "cross-layer-launch", "parameters": {"block_count": 1},
        "hypothesis": "A host-to-kernel launch check covers the cross-layer boundary.",
        "evidence_preset": "correctness-timing",
    }),
    CurriculumProposal.from_mapping({
        "family": "msprof-pipe", "parameters": {"metric": "PipeUtilization"},
        "hypothesis": "A registered msprof metric can guide one bounded revision.",
        "evidence_preset": "msprof-guided",
    }),
)


def _validated_proposals(
    proposals: Sequence[CurriculumProposal],
) -> tuple[CurriculumProposal, ...]:
    rows = tuple(proposals)
    if len(rows) > MAX_PROJECTS:
        raise ValueError(f"Phase1 accepts at most {MAX_PROJECTS} projects")
    if any(not isinstance(item, CurriculumProposal) for item in rows):
        raise ValueError("proposals must contain only CurriculumProposal values")
    identities = [item.identity for item in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate curriculum proposals are not allowed")
    family_rank = {
        registered.family: rank for rank, registered in enumerate(PROJECT_REGISTRY)
    }
    return tuple(sorted(rows, key=lambda item: (
        item.registered.wave, family_rank[item.family], item.parameters,
    )))


def saturation_status(proposals: Sequence[CurriculumProposal]) -> str:
    rows = _validated_proposals(proposals)
    if len(rows) not in CHECKPOINTS:
        return "IN_PROGRESS"
    coverage = {tag for item in rows for tag in item.registered.coverage}
    knee_count = sum(Coverage.PERFORMANCE_KNEE in item.registered.coverage for item in rows)
    if PHASE1_BRIEF.required_coverage <= coverage and knee_count >= 2:
        return "SATURATED"
    return "CONTINUE"


def dry_run_plan(
    proposals: Sequence[CurriculumProposal] = DEFAULT_PROPOSALS,
) -> dict[str, object]:
    """Return a deterministic plan; this function has no model or BZ execution path."""

    rows = _validated_proposals(proposals)
    projects = [
        {
            "ordinal": ordinal,
            "wave": item.registered.wave,
            **item.as_dict(),
            "coverage": sorted(tag.value for tag in item.registered.coverage),
        }
        for ordinal, item in enumerate(rows, 1)
    ]
    return {
        "mode": "dry-run",
        "budget": {"maximum": MAX_PROJECTS, "planned": len(rows)},
        "waves": list(WAVES),
        "checkpoints": list(CHECKPOINTS),
        "projects": projects,
        "checkpoint_status": {
            str(point): saturation_status(rows[:point])
            for point in CHECKPOINTS if point <= len(rows)
        },
        "terminal_status": saturation_status(rows),
    }


def load_proposals(path: Path) -> tuple[CurriculumProposal, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("proposal input must be a JSON list")
    return tuple(CurriculumProposal.from_mapping(item) for item in payload)
