"""Immutable experiment planning and correctness-first aggregate reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
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


# These tuples are the preregistered initial experiment dimensions.  Keep them
# explicit: extending one of the public enums must not silently alter this plan.
_INITIAL_LANGUAGES = (
    Language.CATLASS_DSL,
    Language.ASCEND_C,
    Language.TRITON_ASCEND,
)
_INITIAL_KNOWLEDGE_MODES = (
    KnowledgeMode.WITHOUT_KDB,
    KnowledgeMode.WITH_KDB,
)
_INITIAL_PROFILING_MODES = (
    ProfilingMode.WITHOUT_GUIDANCE,
    ProfilingMode.WITH_GUIDANCE,
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
    for language in _INITIAL_LANGUAGES:
        for knowledge in _INITIAL_KNOWLEDGE_MODES:
            for profiling in _INITIAL_PROFILING_MODES:
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
    workloads: frozenset[Workload]
    kdb: bool = False
    profiling_guidance: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.workloads, frozenset)
            or not self.workloads
            or any(not isinstance(workload, Workload) for workload in self.workloads)
        ):
            raise ValueError("workloads must be a non-empty frozenset of Workload")


@dataclass(frozen=True)
class ScheduledRun:
    cell: ExperimentCell
    workload: Workload


class ExperimentOrchestrator:
    """Host-owned lifecycle gate; returned runs are plans, not success claims."""

    def __init__(self, capabilities: RuntimeCapabilities) -> None:
        self._capabilities = capabilities

    def schedule(self, cell: ExperimentCell, workload: Workload) -> ScheduledRun:
        if not isinstance(workload, Workload):
            raise ValueError("workload must be a Workload")
        missing: list[str] = []
        if cell.language not in self._capabilities.languages:
            missing.append(f"runtime:{cell.language.value}")
        if cell.model.model_id not in self._capabilities.model_ids:
            missing.append(f"model:{cell.model.model_id}")
        if workload not in self._capabilities.workloads:
            missing.append(f"workload:{workload.value}")
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
    tokens: int | None
    wall_time_s: float
    reproducible: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id:
            raise ValueError("cell_id must be a non-empty string")
        if not isinstance(self.workload, Workload):
            raise ValueError("workload must be a Workload")
        self._boolean(self.correct, "correct")
        self._boolean(self.exploration_succeeded, "exploration_succeeded")
        if self.reproducible is not None:
            self._boolean(self.reproducible, "reproducible")
        if self.kernel_time_us is not None:
            self._nonnegative_number(self.kernel_time_us, "kernel_time_us")
        self._counter(self.iterations, "iterations")
        if self.tokens is not None:
            self._counter(self.tokens, "tokens")
        self._nonnegative_number(self.wall_time_s, "wall_time_s")

    @staticmethod
    def _boolean(value: object, field: str) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be a JSON boolean")
        return value

    @staticmethod
    def _nonnegative_number(value: object, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a JSON number")
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"{field} must be finite and non-negative")
        return result

    @staticmethod
    def _counter(value: object, field: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{field} must be a non-negative JSON integer")
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RunMetrics":
        if not isinstance(value, Mapping):
            raise ValueError("each report row must be a JSON object")
        return cls(
            cell_id=value["cell_id"],
            workload=Workload(str(value["workload"])),
            correct=cls._boolean(value["correct"], "correct"),
            kernel_time_us=value.get("kernel_time_us"),
            exploration_succeeded=cls._boolean(
                value["exploration_succeeded"], "exploration_succeeded"
            ),
            iterations=value["iterations"],
            tokens=value.get("tokens"),
            wall_time_s=value["wall_time_s"],
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
    timing_groups: dict[tuple[str, Workload], list[float]] = {}
    for row in rows:
        if row.correct and row.kernel_time_us is not None:
            timing_groups.setdefault((row.cell_id, row.workload), []).append(
                row.kernel_time_us
            )
    performance_groups = [
        {
            "cell_id": cell_id,
            "workload": workload.value,
            "mean_us_correct_runs": sum(times) / len(times),
            "measured": len(times),
        }
        for (cell_id, workload), times in sorted(
            timing_groups.items(), key=lambda item: (item[0][0], item[0][1].value)
        )
    ]
    reproducible = [row.reproducible for row in rows if row.reproducible is not None]
    known_tokens = [row.tokens for row in rows if row.tokens is not None]
    return {
        "correctness": {"passed": correct, "total": len(rows)},
        "kernel_performance": {
            "mean_us_correct_runs": (
                performance_groups[0]["mean_us_correct_runs"]
                if len(performance_groups) == 1
                else None
            ),
            "by_cell_workload": performance_groups,
        },
        "exploration_success": sum(row.exploration_succeeded for row in rows),
        "iterations": sum(row.iterations for row in rows),
        # Preserve the original scalar for fully measured reports. A partial
        # total would be indistinguishable from zero usage for unknown runs.
        "tokens": sum(known_tokens) if len(known_tokens) == len(rows) else None,
        "token_usage": {
            "known_total": sum(known_tokens),
            "measured": len(known_tokens),
            "total": len(rows),
        },
        "wall_time_s": sum(row.wall_time_s for row in rows),
        "reproducibility": {
            "reproducible": sum(value is True for value in reproducible),
            "measured": len(reproducible),
        },
    }


def load_metrics(path: Path) -> tuple[RunMetrics, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("report input must be a JSON object or list")
    return tuple(RunMetrics.from_mapping(item) for item in payload)
