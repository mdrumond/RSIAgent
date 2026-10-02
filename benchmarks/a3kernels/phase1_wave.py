"""Deterministic, interruption-safe state for the first A3 Phase 1 wave."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Mapping, Sequence

from benchmarks.a3_experiments import A3ExperimentCell, ProgrammingLevel, build_a3_experiment_plan
from benchmarks.a3_model_profiles import load_a3_model_profile
from benchmarks.a3kernels.candidate import profile_driver_asset
from benchmarks.a3kernels.phase1_evidence import canonical_digest
from benchmarks.a3kernels.phase1_registry import dry_run_plan


_SHA = re.compile(r"[0-9a-f]{64}")
_HOST_ASSETS = ("build.json", "host_driver.py", "host_wrapper.inc")


def full_dry_run() -> dict[str, object]:
    return dry_run_plan()


def foundation_cells() -> tuple[A3ExperimentCell, ...]:
    return tuple(
        cell for cell in build_a3_experiment_plan().cells
        if cell.programming_level is ProgrammingLevel.FOUNDATION
    )


def select_foundation_cells(
    *,
    cell_ids: Sequence[str] = (),
    shard_count: int | None = None,
    shard_index: int | None = None,
) -> tuple[A3ExperimentCell, ...]:
    """Select a canonical, optionally sharded subset of foundation cells."""
    cells = foundation_cells()
    available = {cell.cell_id: cell for cell in cells}
    requested = tuple(cell_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("duplicate cell ID")
    unknown = [cell_id for cell_id in requested if cell_id not in available]
    if unknown:
        raise ValueError("unknown foundation cell ID: " + ", ".join(map(str, unknown)))
    if requested:
        requested_ids = set(requested)
        cells = tuple(cell for cell in cells if cell.cell_id in requested_ids)

    if (shard_count is None) != (shard_index is None):
        raise ValueError("shard count and index must be provided together")
    if shard_count is None:
        return cells
    if type(shard_count) is not int or shard_count < 1:
        raise ValueError("shard count must be a positive integer")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must be an integer in the shard range")
    return tuple(
        cell for ordinal, cell in enumerate(cells)
        if ordinal % shard_count == shard_index
    )


def _research_identity() -> dict[str, str]:
    source = files("benchmarks.a3kernels.candidate_runtime")
    fixture = {
        name: hashlib.sha256(source.joinpath(name).read_bytes()).hexdigest()
        for name in _HOST_ASSETS
    }
    return {
        "plan_fingerprint": canonical_digest(full_dry_run()),
        "profile_driver_sha256": profile_driver_asset().sha256,
        "host_fixture_sha256": canonical_digest(fixture),
    }


@dataclass(frozen=True)
class CellPaths:
    root: Path
    workspace: Path
    memory: Path
    evidence: Path
    terminal: Path
    profile_driver: Path


@dataclass(frozen=True)
class Phase1Config:
    state_root: Path
    validation_wrapper: Path
    embedding_cache: Path
    corpus_artifacts: Path
    knowledge_database: Path
    knowledge_manifest: Path

    def preflight(self, environ: Mapping[str, str]) -> dict[str, object]:
        driver = profile_driver_asset()
        credential_names = {
            load_a3_model_profile(cell.backend_model).credential_env
            for cell in foundation_cells()
        }
        paths = {
            "validation_wrapper": self.validation_wrapper.is_file()
            and os.access(self.validation_wrapper, os.X_OK)
            and self.validation_wrapper.name == "catlass-validation.sh",
            "embedding_cache": self.embedding_cache.is_dir(),
            "corpus_artifacts": self.corpus_artifacts.is_dir(),
            "knowledge_database": self.knowledge_database.is_file(),
            "knowledge_manifest": self.knowledge_manifest.is_file(),
            "profile_driver": (
                driver.relative_path == "a3_profile_driver.py"
                and _SHA.fullmatch(driver.sha256) is not None
            ),
            **{
                credential_name: bool(environ.get(credential_name))
                for credential_name in sorted(credential_names)
            },
        }
        missing = [name for name, present in paths.items() if not present]
        if missing:
            raise ValueError("preflight missing: " + ", ".join(missing))
        profile_text = driver.content.lower()
        if any(token in profile_text for token in ("catlass", "bz-a5", "a5kernel")):
            raise ValueError("profile driver contains foreign A5/Catlass evidence")
        return {
            "ready": True,
            "checks": paths,
            **_research_identity(),
        }


Executor = Callable[[A3ExperimentCell, CellPaths], Mapping[str, object]]


class Phase1Wave:
    def __init__(
        self,
        config: Phase1Config,
        executor: Executor,
        *,
        cells: Sequence[A3ExperimentCell] | None = None,
    ) -> None:
        self.config = config
        self.executor = executor
        self.cells = (
            foundation_cells()
            if cells is None
            else select_foundation_cells(cell_ids=tuple(cell.cell_id for cell in cells))
        )

    def paths(self, cell: A3ExperimentCell) -> CellPaths:
        root = self.config.state_root / "cells" / cell.cell_id
        workspace = root / "workspace"
        return CellPaths(
            root, workspace, root / "memory.jsonl", root / "evidence.jsonl",
            root / "terminal.json", workspace / "a3_profile_driver.py",
        )

    def run(self) -> tuple[dict[str, object], ...]:
        records = []
        for cell in self.cells:
            paths = self.paths(cell)
            retained = self._read_terminal(cell, paths.terminal)
            if retained is not None:
                records.append(retained)
                continue
            self._stage(paths)
            outcome = dict(self.executor(cell, paths))
            record = self._terminal(cell, outcome)
            self._publish(paths.terminal, record)
            records.append(record)
        return tuple(records)

    def resume(self) -> tuple[dict[str, object], ...]:
        return self.run()

    def report(self) -> dict[str, object]:
        records = []
        for cell in foundation_cells():
            record = self._read_terminal(cell, self.paths(cell).terminal)
            if record is not None:
                records.append(record)
        counts: dict[str, int] = {}
        for record in records:
            status = str(record["status"])
            counts[status] = counts.get(status, 0) + 1
        return {"schema": "a3-phase1-foundation-report-v1", "counts": counts, "records": records}

    def _stage(self, paths: CellPaths) -> None:
        paths.workspace.mkdir(parents=True, exist_ok=True)
        source = files("benchmarks.a3kernels.candidate_runtime")
        for name in _HOST_ASSETS:
            (paths.workspace / name).write_text(
                source.joinpath(name).read_text(encoding="utf-8"), encoding="utf-8"
            )
        driver = profile_driver_asset()
        paths.profile_driver.write_text(driver.content, encoding="utf-8")

    @staticmethod
    def _terminal(cell: A3ExperimentCell, outcome: Mapping[str, object]) -> dict[str, object]:
        if set(outcome) != {"status", "evidence_sha256"}:
            raise ValueError("executor terminal outcome has an invalid schema")
        if outcome["status"] not in {"passed", "failed"}:
            raise ValueError("executor terminal status is invalid")
        if type(outcome["evidence_sha256"]) is not str or _SHA.fullmatch(outcome["evidence_sha256"]) is None:
            raise ValueError("executor terminal evidence digest is invalid")
        return {
            "schema": "a3-phase1-cell-terminal-v2",
            "cell_id": cell.cell_id,
            "target": "a3",
            "language": "ascend-c",
            "model": cell.backend_model.value,
            "knowledge": cell.knowledge.value,
            "profiling": cell.profiling.value,
            **_research_identity(),
            **outcome,
        }

    @staticmethod
    def _publish(path: Path, record: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != data:
                raise ValueError("conflicting terminal cell record") from None
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _read_terminal(cls, cell: A3ExperimentCell, path: Path) -> dict[str, object] | None:
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("terminal cell record is corrupt") from exc
        expected = cls._terminal(
            cell, {"status": value.get("status"), "evidence_sha256": value.get("evidence_sha256")}
        )
        if value != expected:
            raise ValueError("terminal cell record has foreign or conflicting evidence")
        return value


__all__ = [
    "CellPaths",
    "Phase1Config",
    "Phase1Wave",
    "foundation_cells",
    "full_dry_run",
    "select_foundation_cells",
]
