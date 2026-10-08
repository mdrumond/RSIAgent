"""Deterministic A5 Catlass DSL Phase 1 experiment identities.

This registry intentionally does not import the A3 experiment model.  A guide
path is an admission-time input only; persisted plans contain content and
revision identities, never a machine-local path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping


PLAN_SCHEMA = "a5-catlass-phase1-experiment-plan-v1"
GUIDE_SCHEMA = "catlass-dsl-programming-guide-v1"
PINNED_CATLASS_REVISION = "9a6ac627b5f4078060287844189730cf0d184800"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")


class A5Target(str, Enum):
    A5 = "a5"


class A5Language(str, Enum):
    CATLASS_DSL = "catlass-dsl"


class A5BackendModel(str, Enum):
    GPT_5_6_SOL = "openai/gpt-5.6-sol"
    DEEPSEEK_FLASH = "deepseek-flash"


class KnowledgeMode(str, Enum):
    WITHOUT_KDB = "without-kdb"
    WITH_KDB = "with-kdb"


class ProfilingGuidance(str, Enum):
    WITHOUT_GUIDANCE = "without-profiling-guidance"
    WITH_GUIDANCE = "with-profiling-guidance"


@dataclass(frozen=True)
class A5ModelIdentity:
    """Stable provider profile identity; no credentials or fallback aliases."""

    backend_model: A5BackendModel
    profile_id: str
    provider: str
    route: str

    def __post_init__(self) -> None:
        if not isinstance(self.backend_model, A5BackendModel):
            raise ValueError("backend_model must be a registered A5 model")
        expected = _MODEL_VALUES[self.backend_model]
        if self.as_dict() != expected:
            raise ValueError("model identity must match its registered A5 profile")

    def as_dict(self) -> dict[str, str]:
        return {
            "backend_model": self.backend_model.value,
            "profile_id": self.profile_id,
            "provider": self.provider,
            "route": self.route,
        }


_MODEL_VALUES = {
    A5BackendModel.GPT_5_6_SOL: {
        "backend_model": "openai/gpt-5.6-sol",
        "profile_id": "openai-gpt-5.6-sol-v1",
        "provider": "OpenAI",
        "route": "openrouter:OpenAI",
    },
    A5BackendModel.DEEPSEEK_FLASH: {
        "backend_model": "deepseek-flash",
        "profile_id": "deepseek-flash-native-v1",
        "provider": "DeepSeek",
        "route": "direct:https://api.deepseek.com",
    },
}

MODEL_IDENTITIES = tuple(
    A5ModelIdentity(model, **{
        key: value for key, value in values.items() if key != "backend_model"
    })
    for model, values in _MODEL_VALUES.items()
)
_MODEL_BY_BACKEND = {item.backend_model: item for item in MODEL_IDENTITIES}


@dataclass(frozen=True)
class ProgrammingGuideIdentity:
    schema: str
    guide_sha256: str
    cpl_skills_revision: str
    catlass_revision: str

    def __post_init__(self) -> None:
        if self.schema != GUIDE_SCHEMA:
            raise ValueError(f"guide schema must be {GUIDE_SCHEMA}")
        if not isinstance(self.guide_sha256, str) or _SHA256.fullmatch(
            self.guide_sha256
        ) is None:
            raise ValueError("guide_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.cpl_skills_revision, str) or _GIT_REVISION.fullmatch(
            self.cpl_skills_revision
        ) is None:
            raise ValueError("cpl_skills_revision must be a full lowercase Git revision")
        if self.catlass_revision != PINNED_CATLASS_REVISION:
            raise ValueError("Catlass revision does not match the Phase 1 pin")

    def as_dict(self) -> dict[str, str]:
        return {
            "schema": self.schema,
            "guide_sha256": self.guide_sha256,
            "cpl_skills_revision": self.cpl_skills_revision,
            "catlass_revision": self.catlass_revision,
        }


def admit_programming_guide(
    path: str | Path,
    *,
    cpl_skills_revision: str,
    catlass_revision: str = PINNED_CATLASS_REVISION,
) -> ProgrammingGuideIdentity:
    """Admit a UTF-8 guide while discarding its machine-local path."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ValueError("unable to read Catlass DSL programming guide") from exc
    if not payload:
        raise ValueError("Catlass DSL programming guide must not be empty")
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Catlass DSL programming guide must be UTF-8") from exc
    return ProgrammingGuideIdentity(
        schema=GUIDE_SCHEMA,
        guide_sha256=hashlib.sha256(payload).hexdigest(),
        cpl_skills_revision=cpl_skills_revision,
        catlass_revision=catlass_revision,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _cell_id(dimensions: Mapping[str, object]) -> str:
    digest = hashlib.sha256(_canonical_json(dimensions).encode("utf-8")).hexdigest()
    return f"a5-cell-{digest[:16]}"


@dataclass(frozen=True)
class A5Phase1Cell:
    cell_id: str
    model: A5ModelIdentity
    knowledge: KnowledgeMode
    profiling: ProfilingGuidance
    guide: ProgrammingGuideIdentity
    target: A5Target = A5Target.A5
    language: A5Language = A5Language.CATLASS_DSL

    def __post_init__(self) -> None:
        typed = (
            (self.target, A5Target),
            (self.language, A5Language),
            (self.model, A5ModelIdentity),
            (self.knowledge, KnowledgeMode),
            (self.profiling, ProfilingGuidance),
            (self.guide, ProgrammingGuideIdentity),
        )
        if any(not isinstance(value, expected) for value, expected in typed):
            raise ValueError("cell dimensions must use registered A5 types")
        if self.model != _MODEL_BY_BACKEND.get(self.model.backend_model):
            raise ValueError("cell model must use a registered A5 profile")
        if self.cell_id != _cell_id(self.identity_values()):
            raise ValueError("cell_id does not match its A5 Phase 1 dimensions")

    def identity_values(self) -> dict[str, object]:
        return {
            "target": self.target.value,
            "language": self.language.value,
            "model": self.model.as_dict(),
            "knowledge": self.knowledge.value,
            "profiling": self.profiling.value,
            "guide": self.guide.as_dict(),
        }

    def as_dict(self) -> dict[str, object]:
        return {"cell_id": self.cell_id, **self.identity_values()}


@dataclass(frozen=True)
class A5Phase1Plan:
    cells: tuple[A5Phase1Cell, ...]
    guide: ProgrammingGuideIdentity
    schema: str = PLAN_SCHEMA
    target: A5Target = A5Target.A5
    language: A5Language = A5Language.CATLASS_DSL

    def __post_init__(self) -> None:
        if (
            self.schema != PLAN_SCHEMA
            or self.target is not A5Target.A5
            or self.language is not A5Language.CATLASS_DSL
        ):
            raise ValueError("plan must use the registered A5 Catlass Phase 1 schema")
        if len(self.cells) != 8 or len({cell.cell_id for cell in self.cells}) != 8:
            raise ValueError("A5 Phase 1 requires eight unique cells")
        if any(not isinstance(cell, A5Phase1Cell) for cell in self.cells):
            raise ValueError("plan cells must be A5Phase1Cell values")
        if any(cell.guide != self.guide for cell in self.cells):
            raise ValueError("every Phase 1 cell must bind the admitted guide identity")
        expected = build_cells(self.guide)
        if self.cells != expected:
            raise ValueError("plan must contain the complete ordered treatment registry")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "target": self.target.value,
            "language": self.language.value,
            "guide": self.guide.as_dict(),
            "cells": [cell.as_dict() for cell in self.cells],
        }

    def to_json(self) -> str:
        return _canonical_json(self.as_dict()) + "\n"


def build_cells(guide: ProgrammingGuideIdentity) -> tuple[A5Phase1Cell, ...]:
    if not isinstance(guide, ProgrammingGuideIdentity):
        raise ValueError("guide must be an admitted ProgrammingGuideIdentity")
    cells = []
    for model in MODEL_IDENTITIES:
        for knowledge in KnowledgeMode:
            for profiling in ProfilingGuidance:
                dimensions = {
                    "target": A5Target.A5.value,
                    "language": A5Language.CATLASS_DSL.value,
                    "model": model.as_dict(),
                    "knowledge": knowledge.value,
                    "profiling": profiling.value,
                    "guide": guide.as_dict(),
                }
                cells.append(
                    A5Phase1Cell(
                        _cell_id(dimensions), model, knowledge, profiling, guide
                    )
                )
    return tuple(cells)


def build_a5_phase1_plan(
    guide_path: str | Path,
    *,
    cpl_skills_revision: str,
    catlass_revision: str = PINNED_CATLASS_REVISION,
) -> A5Phase1Plan:
    guide = admit_programming_guide(
        guide_path,
        cpl_skills_revision=cpl_skills_revision,
        catlass_revision=catlass_revision,
    )
    return A5Phase1Plan(build_cells(guide), guide)


__all__ = [
    "A5BackendModel",
    "A5Language",
    "A5ModelIdentity",
    "A5Phase1Cell",
    "A5Phase1Plan",
    "A5Target",
    "GUIDE_SCHEMA",
    "KnowledgeMode",
    "MODEL_IDENTITIES",
    "PINNED_CATLASS_REVISION",
    "PLAN_SCHEMA",
    "ProfilingGuidance",
    "ProgrammingGuideIdentity",
    "admit_programming_guide",
    "build_a5_phase1_plan",
    "build_cells",
]
