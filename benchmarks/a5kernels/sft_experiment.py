"""Standalone, frozen Base/H/W/W+H experiment contracts.

This module plans and reports experiments.  Provider inference, memory retrieval,
benchmark execution, and public source downloads are supplied by callers so the
host can preserve exact treatment boundaries and retain its own evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
from typing import Callable, Iterable


class Arm(str, Enum):
    BASE = "base"
    H = "h"
    W = "w"
    W_H = "w+h"


class BenchmarkMode(str, Enum):
    SPEC_GENERATION = "spec-generation"
    OPTIMIZE_PUBLIC_BASELINE = "optimize-public-baseline"


class ContractTrack(str, Enum):
    ORIGINAL = "original"
    CONTRACT_PRESERVING = "contract-preserving"
    CHANGED_CONTRACT = "changed-contract"


def _model_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _content_id(value: object, field: str) -> str:
    result = _model_id(value, field)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", result) is None:
        raise ValueError(
            f"{field} must be immutable (use sha256: plus exactly 64 lowercase hex chars)"
        )
    return result


def _finite(value: object, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


@dataclass(frozen=True)
class StudyConfig:
    base_model_id: str
    adapter_model_id: str
    frozen_memory_id: str
    seeds: tuple[int, int, int]
    max_tokens: int
    max_tool_calls: int
    temperature: float
    top_p: float

    def __post_init__(self) -> None:
        # The frozen dataclass makes the exact provider-issued IDs immutable;
        # provider syntax is intentionally opaque and must not be guessed here.
        _model_id(self.base_model_id, "base_model_id")
        _model_id(self.adapter_model_id, "adapter_model_id")
        _content_id(self.frozen_memory_id, "frozen_memory_id")
        if (
            not isinstance(self.seeds, tuple)
            or len(self.seeds) != 3
            or len(set(self.seeds)) != 3
            or any(type(seed) is not int or seed < 0 for seed in self.seeds)
        ):
            raise ValueError("seeds must contain exactly three distinct seeds")
        for field in ("max_tokens", "max_tool_calls"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        _finite(self.temperature, "temperature")
        top_p = _finite(self.top_p, "top_p")
        if top_p > 1:
            raise ValueError("top_p must be <= 1")

    @property
    def fingerprint(self) -> str:
        payload = {
            field: getattr(self, field)
            for field in (
                "base_model_id", "adapter_model_id", "frozen_memory_id", "seeds",
                "max_tokens", "max_tool_calls", "temperature", "top_p",
            )
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StudyCell:
    cell_id: str
    study_fingerprint: str
    study_seeds: tuple[int, int, int]
    arm: Arm
    mode: BenchmarkMode
    track: ContractTrack
    seed: int
    base_model_id: str
    adapter_model_id: str | None
    memory_id: str | None
    max_tokens: int
    max_tool_calls: int
    temperature: float
    top_p: float

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.study_fingerprint) is None:
            raise ValueError("study_fingerprint must be 64 lowercase hex chars")
        if (
            not isinstance(self.study_seeds, tuple)
            or len(self.study_seeds) != 3
            or len(set(self.study_seeds)) != 3
            or self.seed not in self.study_seeds
        ):
            raise ValueError("study cell must use one of exactly three study seeds")
        expected = {
            Arm.BASE: (False, False),
            Arm.H: (False, True),
            Arm.W: (True, False),
            Arm.W_H: (True, True),
        }[self.arm]
        actual = (self.adapter_model_id is not None, self.memory_id is not None)
        if self.arm is Arm.BASE and actual != (False, False):
            raise ValueError("base arm must not use adapter or memory treatments")
        if actual != expected:
            raise ValueError(
                f"{self.arm.value} arm must not use unassigned treatments and must use assigned treatments"
            )


def _cell_id(fingerprint: str, arm: Arm, mode: BenchmarkMode, track: ContractTrack, seed: int) -> str:
    dimensions = f"{fingerprint}:{arm.value}:{mode.value}:{track.value}:{seed}"
    return "sft-" + hashlib.sha256(dimensions.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class StudyMatrix:
    config: StudyConfig
    cells: tuple[StudyCell, ...]


def build_matrix(config: StudyConfig) -> StudyMatrix:
    """Build the fixed four-arm, two-mode, three-track, three-seed matrix."""

    cells: list[StudyCell] = []
    for arm in Arm:
        for mode in BenchmarkMode:
            for track in ContractTrack:
                for seed in config.seeds:
                    cells.append(
                        StudyCell(
                            cell_id=_cell_id(config.fingerprint, arm, mode, track, seed),
                            study_fingerprint=config.fingerprint,
                            study_seeds=config.seeds,
                            arm=arm,
                            mode=mode,
                            track=track,
                            seed=seed,
                            base_model_id=config.base_model_id,
                            adapter_model_id=(
                                config.adapter_model_id if arm in {Arm.W, Arm.W_H} else None
                            ),
                            memory_id=(
                                config.frozen_memory_id if arm in {Arm.H, Arm.W_H} else None
                            ),
                            max_tokens=config.max_tokens,
                            max_tool_calls=config.max_tool_calls,
                            temperature=config.temperature,
                            top_p=config.top_p,
                        )
                    )
    return StudyMatrix(config=config, cells=tuple(cells))


@dataclass(frozen=True)
class LeaderboardEntry:
    rank: int
    submission_id: str

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not isinstance(self.submission_id, str) or not self.submission_id:
            raise ValueError("submission_id must be non-empty")


@dataclass(frozen=True)
class PublicSourceBundle:
    repository_url: str
    revision: str
    files: tuple[str, ...]
    public_access_verified: bool
    reproduction_command: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.repository_url, str) or not self.repository_url.startswith(
            "https://"
        ):
            raise ValueError("repository_url must be public HTTPS")
        if not isinstance(self.revision, str) or len(self.revision) < 7:
            raise ValueError("revision must identify a frozen source revision")
        if not isinstance(self.files, tuple) or not self.files:
            raise ValueError("source bundle has no files")
        if any(not isinstance(path, str) or not path or path.startswith("/") for path in self.files):
            raise ValueError("source bundle files must be relative paths")
        if self.public_access_verified is not True:
            raise ValueError("source bundle public access was not verified")
        if (
            not isinstance(self.reproduction_command, tuple)
            or not self.reproduction_command
            or any(not isinstance(arg, str) or not arg for arg in self.reproduction_command)
        ):
            raise ValueError("source bundle requires a reproduction command")


@dataclass(frozen=True)
class AcquiredBaseline:
    entry: LeaderboardEntry
    source: PublicSourceBundle


@dataclass(frozen=True)
class SkippedBaseline:
    entry: LeaderboardEntry
    reason: str


@dataclass(frozen=True)
class AcquisitionManifest:
    accepted: tuple[AcquiredBaseline, ...]
    skipped: tuple[SkippedBaseline, ...]
    examined: int
    required: int


class AcquisitionIncomplete(RuntimeError):
    def __init__(self, manifest: AcquisitionManifest) -> None:
        self.manifest = manifest
        super().__init__(
            f"required {manifest.required} public reproducible baselines; "
            f"acquired {len(manifest.accepted)} after examining {manifest.examined}"
        )


def acquire_public_baselines(
    entries: Iterable[LeaderboardEntry],
    acquire: Callable[[LeaderboardEntry], PublicSourceBundle],
    *,
    minimum: int = 3,
) -> AcquisitionManifest:
    """Walk leaderboard order and retain only caller-verified source bundles."""

    if type(minimum) is not int or minimum <= 0:
        raise ValueError("minimum must be a positive integer")
    accepted: list[AcquiredBaseline] = []
    skipped: list[SkippedBaseline] = []
    examined = 0
    for entry in entries:
        if len(accepted) == minimum:
            break
        examined += 1
        try:
            source = acquire(entry)
            if not isinstance(source, PublicSourceBundle):
                raise ValueError("acquirer returned no public source bundle")
        except (ValueError, OSError) as error:
            skipped.append(SkippedBaseline(entry, str(error)))
        else:
            accepted.append(AcquiredBaseline(entry, source))
    manifest = AcquisitionManifest(tuple(accepted), tuple(skipped), examined, minimum)
    if len(accepted) < minimum:
        raise AcquisitionIncomplete(manifest)
    return manifest


@dataclass(frozen=True)
class RunMetric:
    cell: StudyCell
    task_id: str
    compiled: bool
    correct: bool
    hap: float
    latency_ms: float
    cost_usd: float
    regressed: bool
    repairs: int
    public_baseline_id: str | None = None
    public_baseline_hap: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be non-empty")
        for field in ("compiled", "correct", "regressed"):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be a bool")
        if self.correct and not self.compiled:
            raise ValueError("correct run must compile")
        for field in ("hap", "latency_ms", "cost_usd"):
            _finite(getattr(self, field), field)
        if type(self.repairs) is not int or self.repairs < 0:
            raise ValueError("repairs must be a non-negative integer")
        if self.cell.mode is BenchmarkMode.OPTIMIZE_PUBLIC_BASELINE:
            if not isinstance(self.public_baseline_id, str) or not self.public_baseline_id:
                raise ValueError("optimization runs require public_baseline_id")
            if self.public_baseline_hap is None:
                raise ValueError("optimization runs require public_baseline_hap")
            _finite(self.public_baseline_hap, "public_baseline_hap")
        elif self.public_baseline_id is not None or self.public_baseline_hap is not None:
            raise ValueError("spec-generation runs cannot have a public baseline")


def _run_key(row: RunMetric) -> tuple[str, str, str, int, str]:
    return (
        row.cell.study_fingerprint,
        row.cell.mode.value,
        row.cell.track.value,
        row.cell.seed,
        row.task_id,
    )


def _validate_study(rows: tuple[RunMetric, ...]) -> None:
    if not rows:
        raise ValueError("matrix requires rows")
    fields = (
        "study_fingerprint", "study_seeds", "base_model_id", "max_tokens",
        "max_tool_calls", "temperature", "top_p",
    )
    if any(len({getattr(row.cell, field) for row in rows}) != 1 for field in fields):
        raise ValueError("study configuration drift across matrix cells")
    adapters = {row.cell.adapter_model_id for row in rows if row.cell.adapter_model_id}
    memories = {row.cell.memory_id for row in rows if row.cell.memory_id}
    if len(adapters) != 1 or len(memories) != 1:
        raise ValueError("study configuration drift across adapter or memory treatments")

    first = rows[0].cell
    config = StudyConfig(
        base_model_id=first.base_model_id,
        adapter_model_id=next(iter(adapters)),
        frozen_memory_id=next(iter(memories)),
        seeds=first.study_seeds,
        max_tokens=first.max_tokens,
        max_tool_calls=first.max_tool_calls,
        temperature=first.temperature,
        top_p=first.top_p,
    )
    if config.fingerprint != first.study_fingerprint:
        raise ValueError("study configuration drift from declared fingerprint")
    for row in rows:
        expected_id = _cell_id(
            config.fingerprint, row.cell.arm, row.cell.mode, row.cell.track, row.cell.seed
        )
        if row.cell.cell_id != expected_id:
            raise ValueError("study cell identity does not match its configuration")

    expected = {
        (arm, mode, track, seed)
        for arm in Arm
        for mode in BenchmarkMode
        for track in ContractTrack
        for seed in config.seeds
    }
    for task_id in {row.task_id for row in rows}:
        task_rows = tuple(row for row in rows if row.task_id == task_id)
        actual = {
            (row.cell.arm, row.cell.mode, row.cell.track, row.cell.seed)
            for row in task_rows
        }
        if actual != expected or len(task_rows) != len(expected):
            raise ValueError("every task requires the complete factorial matrix")

    optimization_groups: dict[tuple[str, str, int, str], set[tuple[str, float]]] = {}
    for row in rows:
        if row.cell.mode is BenchmarkMode.OPTIMIZE_PUBLIC_BASELINE:
            key = (row.cell.mode.value, row.cell.track.value, row.cell.seed, row.task_id)
            optimization_groups.setdefault(key, set()).add(
                (str(row.public_baseline_id), float(row.public_baseline_hap))
            )
    if any(len(values) != 1 for values in optimization_groups.values()):
        raise ValueError("matched optimization arms require the same public baseline identity and value")


def _mean(rows: tuple[RunMetric, ...], field: str) -> float:
    return sum(float(getattr(row, field)) for row in rows) / len(rows)


_METRIC_DIRECTIONS = {
    "compile_rate": "higher-is-better",
    "correctness_rate": "higher-is-better",
    "hap": "higher-is-better",
    "latency_ms": "lower-is-better",
    "cost_usd": "lower-is-better",
    "regression_rate": "lower-is-better",
    "mean_repairs": "lower-is-better",
    "public_baseline_hap_improvement": "higher-is-better",
}


def _aggregate_matched(rows: tuple[RunMetric, ...]) -> dict[str, object]:
    by_arm = {arm: tuple(row for row in rows if row.cell.arm is arm) for arm in Arm}
    keys = [{_run_key(row) for row in arm_rows} for arm_rows in by_arm.values()]
    if not rows or any(len(arm_rows) != len(set(_run_key(r) for r in arm_rows)) for arm_rows in by_arm.values()):
        raise ValueError("matrix requires unique matched run keys in every arm")
    if any(key_set != keys[0] for key_set in keys[1:]):
        raise ValueError("matrix requires identical matched run keys across all arms")

    arms: dict[str, dict[str, float]] = {}
    for arm, arm_rows in by_arm.items():
        optimized = tuple(
            row for row in arm_rows
            if row.cell.mode is BenchmarkMode.OPTIMIZE_PUBLIC_BASELINE
        )
        summary: dict[str, float] = {
            "compile_rate": _mean(arm_rows, "compiled"),
            "correctness_rate": _mean(arm_rows, "correct"),
            "hap": _mean(arm_rows, "hap"),
            "latency_ms": _mean(arm_rows, "latency_ms"),
            "cost_usd": _mean(arm_rows, "cost_usd"),
            "regression_rate": _mean(arm_rows, "regressed"),
            "mean_repairs": _mean(arm_rows, "repairs"),
        }
        if optimized:
            summary["public_baseline_hap_improvement"] = sum(
                row.hap - float(row.public_baseline_hap) for row in optimized
            ) / len(optimized)
        arms[arm.value] = summary

    effect_metrics = tuple(arms[Arm.BASE.value])
    base, h, w, wh = (arms[arm.value] for arm in (Arm.BASE, Arm.H, Arm.W, Arm.W_H))
    effects = {
        "H-Base": {metric: h[metric] - base[metric] for metric in effect_metrics},
        "W-Base": {metric: w[metric] - base[metric] for metric in effect_metrics},
        "W+H-max(W,H)": {
            metric: wh[metric] - (
                max(w[metric], h[metric])
                if _METRIC_DIRECTIONS[metric] == "higher-is-better"
                else min(w[metric], h[metric])
            )
            for metric in effect_metrics
        },
    }
    return {
        "run_count": len(rows),
        "metric_directions": {
            metric: _METRIC_DIRECTIONS[metric] for metric in effect_metrics
        },
        "arms": arms,
        "effects": effects,
    }


def aggregate_matrix(metrics: Iterable[RunMetric]) -> dict[str, object]:
    """Aggregate matched arms overall and within every mode and track."""

    rows = tuple(metrics)
    _validate_study(rows)
    report = _aggregate_matched(rows)
    report["by_mode"] = {
        mode.value: _aggregate_matched(tuple(row for row in rows if row.cell.mode is mode))
        for mode in BenchmarkMode
    }
    report["by_track"] = {
        track.value: _aggregate_matched(tuple(row for row in rows if row.cell.track is track))
        for track in ContractTrack
    }
    return report


def synthetic_demo_metrics(matrix: StudyMatrix, *, task_id: str) -> tuple[RunMetric, ...]:
    """Return synthetic numbers for plumbing demos, never execution evidence."""

    hap_by_arm = {Arm.BASE: 10.0, Arm.H: 11.0, Arm.W: 12.0, Arm.W_H: 13.0}
    return tuple(
        RunMetric(
            cell=cell,
            task_id=task_id,
            compiled=True,
            correct=True,
            hap=hap_by_arm[cell.arm],
            latency_ms=100.0 - hap_by_arm[cell.arm],
            cost_usd=0.01 * (1 + list(Arm).index(cell.arm)),
            regressed=False,
            repairs=list(Arm).index(cell.arm),
            public_baseline_id=(
                "synthetic-public-baseline"
                if cell.mode is BenchmarkMode.OPTIMIZE_PUBLIC_BASELINE else None
            ),
            public_baseline_hap=(
                9.0 if cell.mode is BenchmarkMode.OPTIMIZE_PUBLIC_BASELINE else None
            ),
        )
        for cell in matrix.cells
    )
