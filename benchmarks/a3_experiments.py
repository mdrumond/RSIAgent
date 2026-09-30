"""Deterministic A3/Ascend-C experiment planning without A5 runtime types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Mapping, Sequence


PLAN_SCHEMA = "a3-ascendc-experiment-plan-v1"
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class A3Target(str, Enum):
    A3 = "a3"


class A3Language(str, Enum):
    ASCEND_C = "ascend-c"


class BackendModel(str, Enum):
    GPT_5_6_SOL = "openai/gpt-5.6-sol"
    DEEPSEEK_FLASH = "deepseek-flash"


class KnowledgeMode(str, Enum):
    WITHOUT_KDB = "without-kdb"
    WITH_KDB = "with-kdb"


class ProfilingGuidance(str, Enum):
    WITHOUT_GUIDANCE = "without-profiling-guidance"
    WITH_GUIDANCE = "with-profiling-guidance"


class ProgrammingLevel(str, Enum):
    FOUNDATION = "foundation"
    TILED = "tiled"
    OPTIMIZED = "optimized"


_BACKEND_MODELS = tuple(BackendModel)
_KNOWLEDGE_MODES = tuple(KnowledgeMode)
_PROFILING_MODES = tuple(ProfilingGuidance)
_PROGRAMMING_LEVELS = tuple(ProgrammingLevel)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _cell_id(dimensions: Mapping[str, str]) -> str:
    digest = hashlib.sha256(_canonical_json(dimensions).encode("utf-8")).hexdigest()
    return f"a3-cell-{digest[:16]}"


@dataclass(frozen=True)
class A3EvidenceRef:
    """One immutable reference admitted to active A3 knowledge."""

    evidence_id: str
    sha256: str
    target: A3Target = A3Target.A3
    language: A3Language = A3Language.ASCEND_C

    def __post_init__(self) -> None:
        if not isinstance(self.target, A3Target) or not isinstance(
            self.language, A3Language
        ):
            raise ValueError("active A3 evidence must target a3 and ascend-c")
        if not isinstance(self.evidence_id, str) or _SAFE_ID.fullmatch(
            self.evidence_id
        ) is None:
            raise ValueError("evidence_id must be a safe stable identifier")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("evidence sha256 must be a lowercase digest")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "A3EvidenceRef":
        expected = {"evidence_id", "target", "language", "sha256"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("active A3 evidence requires its exact schema")
        if value["target"] != A3Target.A3.value or value[
            "language"
        ] != A3Language.ASCEND_C.value:
            raise ValueError("active A3 evidence must target a3 and ascend-c")
        return cls(value["evidence_id"], value["sha256"])

    def as_dict(self) -> dict[str, str]:
        return {
            "evidence_id": self.evidence_id,
            "target": self.target.value,
            "language": self.language.value,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class CurriculumStage:
    ordinal: int
    programming_level: ProgrammingLevel
    objective: str

    def as_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "programming_level": self.programming_level.value,
            "objective": self.objective,
        }


CURRICULUM = (
    CurriculumStage(1, ProgrammingLevel.FOUNDATION, "correct scalar Ascend C"),
    CurriculumStage(2, ProgrammingLevel.TILED, "correct tiled Ascend C"),
    CurriculumStage(3, ProgrammingLevel.OPTIMIZED, "profile-guided Ascend C"),
)


@dataclass(frozen=True)
class A3ExperimentCell:
    cell_id: str
    target: A3Target
    language: A3Language
    backend_model: BackendModel
    knowledge: KnowledgeMode
    profiling: ProfilingGuidance
    programming_level: ProgrammingLevel

    def __post_init__(self) -> None:
        typed = (
            (self.target, A3Target),
            (self.language, A3Language),
            (self.backend_model, BackendModel),
            (self.knowledge, KnowledgeMode),
            (self.profiling, ProfilingGuidance),
            (self.programming_level, ProgrammingLevel),
        )
        if any(not isinstance(value, expected) for value, expected in typed):
            raise ValueError("cell dimensions must use registered A3 enum values")
        if self.cell_id != _cell_id(self.dimension_values()):
            raise ValueError("cell must be a registered A3 experiment cell")

    def dimension_values(self) -> dict[str, str]:
        return {
            "target": self.target.value,
            "language": self.language.value,
            "backend_model": self.backend_model.value,
            "knowledge": self.knowledge.value,
            "profiling": self.profiling.value,
            "programming_level": self.programming_level.value,
        }

    def as_dict(self) -> dict[str, str]:
        return {"cell_id": self.cell_id, **self.dimension_values()}


@dataclass(frozen=True)
class A3ExperimentPlan:
    cells: tuple[A3ExperimentCell, ...]
    curriculum: tuple[CurriculumStage, ...]
    active_knowledge: tuple[A3EvidenceRef, ...] = ()
    target: A3Target = A3Target.A3
    language: A3Language = A3Language.ASCEND_C
    schema: str = PLAN_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != PLAN_SCHEMA
            or self.target is not A3Target.A3
            or self.language is not A3Language.ASCEND_C
        ):
            raise ValueError("active plan must be the A3 Ascend C schema")
        if any(not isinstance(cell, A3ExperimentCell) for cell in self.cells):
            raise ValueError("A3 plan cells must be A3ExperimentCell values")
        if len(self.cells) != 24 or len({cell.cell_id for cell in self.cells}) != 24:
            raise ValueError("A3 plan requires exactly 24 unique registered cells")
        if self.curriculum != CURRICULUM:
            raise ValueError("A3 plan requires the registered curriculum")
        if any(not isinstance(item, A3EvidenceRef) for item in self.active_knowledge):
            raise ValueError("active knowledge must contain only A3EvidenceRef values")
        ids = [item.evidence_id for item in self.active_knowledge]
        if len(ids) != len(set(ids)):
            raise ValueError("active knowledge evidence IDs must be unique")
        object.__setattr__(self, "cells", tuple(sorted(self.cells, key=_cell_order)))
        object.__setattr__(
            self,
            "active_knowledge",
            tuple(
                sorted(
                    self.active_knowledge,
                    key=lambda item: (item.evidence_id, item.sha256),
                )
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "target": self.target.value,
            "language": self.language.value,
            "cells": [cell.as_dict() for cell in self.cells],
            "curriculum": [stage.as_dict() for stage in self.curriculum],
            "active_knowledge": [item.as_dict() for item in self.active_knowledge],
        }

    def to_json(self) -> str:
        return _canonical_json(self.as_dict()) + "\n"


def _cell_order(cell: A3ExperimentCell) -> tuple[int, int, int, int]:
    return (
        _BACKEND_MODELS.index(cell.backend_model),
        _KNOWLEDGE_MODES.index(cell.knowledge),
        _PROFILING_MODES.index(cell.profiling),
        _PROGRAMMING_LEVELS.index(cell.programming_level),
    )


def build_a3_experiment_plan(
    *, active_knowledge: Sequence[A3EvidenceRef] = ()
) -> A3ExperimentPlan:
    cells = []
    for model in _BACKEND_MODELS:
        for knowledge in _KNOWLEDGE_MODES:
            for profiling in _PROFILING_MODES:
                for level in _PROGRAMMING_LEVELS:
                    dimensions = {
                        "target": A3Target.A3.value,
                        "language": A3Language.ASCEND_C.value,
                        "backend_model": model.value,
                        "knowledge": knowledge.value,
                        "profiling": profiling.value,
                        "programming_level": level.value,
                    }
                    cells.append(
                        A3ExperimentCell(
                            _cell_id(dimensions),
                            A3Target.A3,
                            A3Language.ASCEND_C,
                            model,
                            knowledge,
                            profiling,
                            level,
                        )
                    )
    return A3ExperimentPlan(tuple(cells), CURRICULUM, tuple(active_knowledge))


class CapabilityUnavailableError(RuntimeError):
    """Raised when the active A3 runtime cannot honor a registered cell."""


@dataclass(frozen=True)
class A3RuntimeCapabilities:
    """Small A5-type-independent integration surface for the future A3 runtime."""

    target: A3Target
    language: A3Language
    backend_models: frozenset[BackendModel]
    programming_levels: frozenset[ProgrammingLevel]
    kdb: bool = False
    profiling_guidance: bool = False

    def __post_init__(self) -> None:
        if self.target is not A3Target.A3 or self.language is not A3Language.ASCEND_C:
            raise ValueError("runtime capabilities must target A3 Ascend C")
        if (
            not isinstance(self.backend_models, frozenset)
            or not self.backend_models
            or any(not isinstance(item, BackendModel) for item in self.backend_models)
        ):
            raise ValueError("backend_models must be a non-empty typed frozenset")
        if (
            not isinstance(self.programming_levels, frozenset)
            or not self.programming_levels
            or any(
                not isinstance(item, ProgrammingLevel)
                for item in self.programming_levels
            )
        ):
            raise ValueError("programming_levels must be a non-empty typed frozenset")
        if type(self.kdb) is not bool or type(self.profiling_guidance) is not bool:
            raise ValueError("treatment capabilities must be booleans")


@dataclass(frozen=True)
class A3ScheduledExperiment:
    cell: A3ExperimentCell
    target: A3Target = A3Target.A3
    language: A3Language = A3Language.ASCEND_C


class A3ExperimentScheduler:
    def __init__(
        self,
        capabilities: A3RuntimeCapabilities,
        plan: A3ExperimentPlan | None = None,
    ) -> None:
        if not isinstance(capabilities, A3RuntimeCapabilities):
            raise ValueError("capabilities must be A3RuntimeCapabilities")
        if plan is not None and not isinstance(plan, A3ExperimentPlan):
            raise ValueError("plan must be an A3ExperimentPlan")
        self._capabilities = capabilities
        self._plan = build_a3_experiment_plan() if plan is None else plan

    def schedule(self, cell: A3ExperimentCell) -> A3ScheduledExperiment:
        if not isinstance(cell, A3ExperimentCell) or cell not in self._plan.cells:
            raise ValueError("schedule requires a registered A3 experiment cell")
        missing = []
        if cell.backend_model not in self._capabilities.backend_models:
            missing.append(f"backend-model:{cell.backend_model.value}")
        if cell.programming_level not in self._capabilities.programming_levels:
            missing.append(f"programming-level:{cell.programming_level.value}")
        if cell.knowledge is KnowledgeMode.WITH_KDB and not self._capabilities.kdb:
            missing.append("kdb")
        if (
            cell.profiling is ProfilingGuidance.WITH_GUIDANCE
            and not self._capabilities.profiling_guidance
        ):
            missing.append("profiling-guidance")
        if missing:
            raise CapabilityUnavailableError(
                "unavailable A3 capabilities: " + ", ".join(missing)
            )
        return A3ScheduledExperiment(cell)


__all__ = [
    "A3EvidenceRef",
    "A3ExperimentCell",
    "A3ExperimentPlan",
    "A3ExperimentScheduler",
    "A3Language",
    "A3RuntimeCapabilities",
    "A3ScheduledExperiment",
    "A3Target",
    "BackendModel",
    "CapabilityUnavailableError",
    "CurriculumStage",
    "KnowledgeMode",
    "PLAN_SCHEMA",
    "ProfilingGuidance",
    "ProgrammingLevel",
    "build_a3_experiment_plan",
]
