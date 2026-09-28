"""Immutable experiment planning and correctness-first aggregate reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from benchmarks.a5kernels.fixtures import Language
from benchmarks.a5kernels.model_profile import (
    CANONICAL_MODEL,
    CANONICAL_PROFILE_ID,
    CANONICAL_PROVIDER,
)


class KnowledgeMode(str, Enum):
    WITHOUT_KDB = "without-kdb"
    WITH_KDB = "with-kdb"


class ProfilingMode(str, Enum):
    WITHOUT_GUIDANCE = "without-profiling-guidance"
    WITH_GUIDANCE = "with-profiling-guidance"


class Workload(str, Enum):
    SMOKE_VECTOR_ADD = "smoke-vector-add"
    SEMANTIC_GEMM = "semantic-gemm"


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_id: str
    provider: str
    allow_fallback: bool = False

    def __post_init__(self) -> None:
        if (self.name, self.model_id, self.provider) != (
            CANONICAL_PROFILE_ID,
            CANONICAL_MODEL,
            CANONICAL_PROVIDER,
        ):
            raise ValueError("experiment matrix requires the canonical A5 model route")
        if self.allow_fallback is not False:
            raise ValueError("experiment model profiles must disable provider fallback")


GPT_5_6_SOL = ModelProfile(
    name=CANONICAL_PROFILE_ID,
    model_id=CANONICAL_MODEL,
    provider=CANONICAL_PROVIDER,
)


@dataclass(frozen=True)
class ExperimentCell:
    cell_id: str
    model: ModelProfile
    language: Language
    knowledge: KnowledgeMode
    profiling: ProfilingMode
    context_id: str
    memory_id: str
    workspace_id: str


@dataclass(frozen=True)
class DeterminismTrial:
    trial_id: str
    cell_id: str
    repeat: int
    context_id: str
    memory_id: str
    workspace_id: str
    workload: Workload = Workload.SMOKE_VECTOR_ADD


@dataclass(frozen=True)
class MatrixPlan:
    cells: tuple[ExperimentCell, ...]
    workloads: tuple[Workload, ...]
    determinism_trials: tuple[DeterminismTrial, ...]

    def __post_init__(self) -> None:
        if len(self.cells) != 12:
            raise ValueError("the initial experiment matrix must contain exactly 12 cells")
        for attribute in ("cell_id", "context_id", "memory_id", "workspace_id"):
            values = [getattr(cell, attribute) for cell in self.cells]
            if len(values) != len(set(values)):
                raise ValueError(f"{attribute} must be unique for every cell")

    def as_dict(self) -> dict[str, object]:
        return {
            "cells": [asdict(cell) for cell in self.cells],
            "workloads": [item.value for item in self.workloads],
            "determinism_trials": [asdict(trial) for trial in self.determinism_trials],
        }


def _identity(prefix: str, dimensions: str) -> str:
    digest = hashlib.sha256(dimensions.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def initial_matrix(model: ModelProfile = GPT_5_6_SOL) -> MatrixPlan:
    """Return the fixed 2 x 2 x 3 initial experiment matrix."""

    cells: list[ExperimentCell] = []
    for language in Language:
        for knowledge in KnowledgeMode:
            for profiling in ProfilingMode:
                dimensions = ":".join(
                    (model.model_id, language.value, knowledge.value, profiling.value)
                )
                cell_id = _identity("cell", dimensions)
                cells.append(
                    ExperimentCell(
                        cell_id=cell_id,
                        model=model,
                        language=language,
                        knowledge=knowledge,
                        profiling=profiling,
                        context_id=_identity("context", dimensions),
                        memory_id=_identity("memory", dimensions),
                        workspace_id=_identity("workspace", dimensions),
                    )
                )

    baseline = next(
        cell
        for cell in cells
        if cell.language is Language.CATLASS_DSL
        and cell.knowledge is KnowledgeMode.WITHOUT_KDB
        and cell.profiling is ProfilingMode.WITHOUT_GUIDANCE
    )
    trials = tuple(
        DeterminismTrial(
            trial_id=f"{baseline.cell_id}-repeat-{repeat}",
            cell_id=baseline.cell_id,
            repeat=repeat,
            context_id=_identity("context", f"{baseline.cell_id}:repeat:{repeat}"),
            memory_id=_identity("memory", f"{baseline.cell_id}:repeat:{repeat}"),
            workspace_id=_identity("workspace", f"{baseline.cell_id}:repeat:{repeat}"),
        )
        for repeat in range(1, 4)
    )
    return MatrixPlan(
        cells=tuple(cells),
        workloads=(Workload.SMOKE_VECTOR_ADD, Workload.SEMANTIC_GEMM),
        determinism_trials=trials,
    )


class CapabilityUnavailableError(RuntimeError):
    """Raised before execution when a requested treatment cannot be honored."""


@dataclass(frozen=True)
class RuntimeCapabilities:
    languages: frozenset[Language]
    model_ids: frozenset[str]
    kdb: bool = False
    profiling_guidance: bool = False


@dataclass(frozen=True)
class ScheduledRun:
    cell: ExperimentCell
    workload: Workload


class ExperimentOrchestrator:
    """Host-owned lifecycle gate; returned runs are plans, not success claims."""

    def __init__(self, capabilities: RuntimeCapabilities) -> None:
        self._capabilities = capabilities

    def schedule(self, cell: ExperimentCell, workload: Workload) -> ScheduledRun:
        missing: list[str] = []
        if cell.language not in self._capabilities.languages:
            missing.append(f"runtime:{cell.language.value}")
        if cell.model.model_id not in self._capabilities.model_ids:
            missing.append(f"model:{cell.model.model_id}")
        if cell.knowledge is KnowledgeMode.WITH_KDB and not self._capabilities.kdb:
            missing.append("kdb")
        if (
            cell.profiling is ProfilingMode.WITH_GUIDANCE
            and not self._capabilities.profiling_guidance
        ):
            missing.append("profiling-guidance")
        if missing:
            raise CapabilityUnavailableError("unavailable capabilities: " + ", ".join(missing))
        return ScheduledRun(cell=cell, workload=workload)


@dataclass(frozen=True)
class RunMetrics:
    cell_id: str
    workload: Workload
    correct: bool
    kernel_time_us: float | None
    exploration_succeeded: bool
    iterations: int
    tokens: int
    wall_time_s: float
    reproducible: bool | None = None

    @staticmethod
    def _boolean(value: object, field: str) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be a JSON boolean")
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RunMetrics":
        return cls(
            cell_id=str(value["cell_id"]),
            workload=Workload(str(value["workload"])),
            correct=cls._boolean(value["correct"], "correct"),
            kernel_time_us=(
                None if value.get("kernel_time_us") is None else float(value["kernel_time_us"])
            ),
            exploration_succeeded=cls._boolean(
                value["exploration_succeeded"], "exploration_succeeded"
            ),
            iterations=int(value["iterations"]),
            tokens=int(value["tokens"]),
            wall_time_s=float(value["wall_time_s"]),
            reproducible=(
                None
                if value.get("reproducible") is None
                else cls._boolean(value["reproducible"], "reproducible")
            ),
        )


def aggregate_report(results: Iterable[RunMetrics]) -> dict[str, object]:
    """Aggregate results in the declared, correctness-first metric order."""

    rows = tuple(results)
    correct = sum(row.correct for row in rows)
    valid_times = [row.kernel_time_us for row in rows if row.correct and row.kernel_time_us is not None]
    reproducible = [row.reproducible for row in rows if row.reproducible is not None]
    return {
        "correctness": {"passed": correct, "total": len(rows)},
        "kernel_performance": {
            "mean_us_correct_runs": (
                sum(valid_times) / len(valid_times) if valid_times else None
            )
        },
        "exploration_success": sum(row.exploration_succeeded for row in rows),
        "iterations": sum(row.iterations for row in rows),
        "tokens": sum(row.tokens for row in rows),
        "wall_time_s": sum(row.wall_time_s for row in rows),
        "reproducibility": {
            "reproducible": sum(value is True for value in reproducible),
            "measured": len(reproducible),
        },
    }


def load_metrics(path: Path) -> tuple[RunMetrics, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("report input must be a JSON list")
    return tuple(RunMetrics.from_mapping(item) for item in payload)
