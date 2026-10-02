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

from benchmarks.a3_experiments import (
    A3ExperimentCell,
    KnowledgeMode,
    ProgrammingLevel,
    build_a3_experiment_plan,
)
from benchmarks.a3_model_profiles import load_a3_model_profile
from benchmarks.a3kernels.candidate import profile_driver_asset
from benchmarks.a3kernels.phase1_evidence import (
    A3_EXECUTION_PROFILES,
    canonical_digest,
)
from benchmarks.a3kernels.embeddings import (
    PinnedBGEEmbeddings,
    authenticated_snapshot_identity,
)
from benchmarks.a3kernels.knowledge import CollectionManifest, KnowledgeDB
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS, CurriculumProposal, dry_run_plan


_SHA = re.compile(r"[0-9a-f]{64}")
_HOST_ASSETS = ("build.json", "host_driver.py", "host_wrapper.inc")
_KNOWLEDGE_IDENTITY_FIELDS = {
    "embedding_snapshot_sha256", "knowledge_database_sha256",
    "knowledge_manifest_sha256", "knowledge_collection_sha256",
    "knowledge_probe_sha256",
}


def full_dry_run() -> dict[str, object]:
    return dry_run_plan()


SMOKE_PROPOSALS = (DEFAULT_PROPOSALS[0],)
_SMOKE_QUERY = "A3 Ascend C vector addition tensor movement"


def smoke_dry_run() -> dict[str, object]:
    project = SMOKE_PROPOSALS[0]
    body = {
        "schema": "a3-ascendc-foundation-smoke-plan-v1",
        "mode": "smoke",
        "target": "Ascend910B4",
        "language": "ascend-c",
        "projects": [project.as_dict()],
    }
    return {**body, "plan_id": canonical_digest(body)}


def foundation_cells() -> tuple[A3ExperimentCell, ...]:
    return tuple(
        cell for cell in build_a3_experiment_plan().cells
        if cell.programming_level is ProgrammingLevel.FOUNDATION
    )


def registered_credential_envs(
    cells: Sequence[A3ExperimentCell] | None = None,
) -> tuple[str, ...]:
    """Return the deterministic credential boundary for the Phase 1 matrix."""
    selected = foundation_cells() if cells is None else tuple(cells)
    return tuple(sorted({
        load_a3_model_profile(cell.backend_model).credential_env
        for cell in selected
    }))


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_smoke_knowledge(config: "Phase1Config") -> dict[str, str]:
    """Authenticate and query the exact local KDB before provider inference."""
    snapshot_sha256 = authenticated_snapshot_identity(config.embedding_cache)
    manifest_bytes = config.knowledge_manifest.read_bytes()
    manifest = CollectionManifest.from_json(manifest_bytes.decode("utf-8"))
    embeddings = PinnedBGEEmbeddings(cache_dir=config.embedding_cache)
    with KnowledgeDB.open_read_only(config.knowledge_database, embeddings) as database:
        if database.manifest(manifest.collection) != manifest:
            raise ValueError("local KDB does not match the pinned knowledge manifest")
        hits = database.query(manifest.collection, _SMOKE_QUERY, limit=1)
    if len(hits) != 1:
        raise ValueError("smoke KDB probe returned no authenticated result")
    probe = {
        "query": _SMOKE_QUERY,
        "collection": manifest.collection,
        "chunk_id": hits[0].chunk_id,
        "content_sha256": hits[0].content_sha256,
    }
    return {
        "embedding_snapshot_sha256": snapshot_sha256,
        "knowledge_database_sha256": _sha256_file(config.knowledge_database),
        "knowledge_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "knowledge_collection_sha256": manifest.fingerprint,
        "knowledge_probe_sha256": canonical_digest(probe),
    }


def _validated_knowledge_identity(
    value: Mapping[str, str] | None,
) -> dict[str, str] | None:
    if value is None:
        return None
    if set(value) != _KNOWLEDGE_IDENTITY_FIELDS or any(
        type(item) is not str or _SHA.fullmatch(item) is None
        for item in value.values()
    ):
        raise ValueError("smoke knowledge identity is invalid")
    return dict(value)


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

    def preflight(
        self,
        environ: Mapping[str, str],
        *,
        cells: Sequence[A3ExperimentCell] | None = None,
    ) -> dict[str, object]:
        driver = profile_driver_asset()
        selected = foundation_cells() if cells is None else tuple(cells)
        paths = {
            "validation_wrapper": self.validation_wrapper.is_file()
            and os.access(self.validation_wrapper, os.X_OK)
            and self.validation_wrapper.name == "catlass-validation.sh",
            "profile_driver": (
                driver.relative_path == "a3_profile_driver.py"
                and _SHA.fullmatch(driver.sha256) is not None
            ),
            **{
                credential_name: bool(environ.get(credential_name))
                for credential_name in registered_credential_envs(cells)
            },
        }
        if any(cell.knowledge is KnowledgeMode.WITH_KDB for cell in selected):
            paths.update({
                "embedding_cache": self.embedding_cache.is_dir(),
                "corpus_artifacts": self.corpus_artifacts.is_dir(),
                "knowledge_database": self.knowledge_database.is_file(),
                "knowledge_manifest": self.knowledge_manifest.is_file(),
            })
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

    def smoke_preflight(
        self,
        environ: Mapping[str, str],
        *,
        cells: Sequence[A3ExperimentCell] | None = None,
    ) -> dict[str, object]:
        selected = foundation_cells() if cells is None else tuple(cells)
        report = self.preflight(environ, cells=selected)
        knowledge_identity = None
        if any(cell.knowledge is KnowledgeMode.WITH_KDB for cell in selected):
            knowledge_identity = authenticate_smoke_knowledge(self)
        return {
            **report,
            "smoke_plan_fingerprint": canonical_digest(smoke_dry_run()),
            "knowledge_identity": knowledge_identity,
        }


Executor = Callable[[A3ExperimentCell, CellPaths], Mapping[str, object]]


class Phase1Wave:
    def __init__(
        self,
        config: Phase1Config,
        executor: Executor,
        *,
        cells: Sequence[A3ExperimentCell] | None = None,
        execution_profile: str = "gz-a3",
    ) -> None:
        self.config = config
        self.executor = executor
        self.cells = (
            foundation_cells()
            if cells is None
            else select_foundation_cells(cell_ids=tuple(cell.cell_id for cell in cells))
        )
        if execution_profile not in A3_EXECUTION_PROFILES:
            raise ValueError("Phase 1 wave execution profile is invalid")
        self.execution_profile = execution_profile

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
            retained = self._read_terminal(
                cell, paths.terminal, execution_profile=self.execution_profile
            )
            if retained is not None:
                records.append(retained)
                continue
            self._stage(paths)
            outcome = dict(self.executor(cell, paths))
            record = self._terminal(cell, outcome, self.execution_profile)
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
    def _terminal(
        cell: A3ExperimentCell,
        outcome: Mapping[str, object],
        execution_profile: str,
    ) -> dict[str, object]:
        if set(outcome) != {"status", "evidence_sha256"}:
            raise ValueError("executor terminal outcome has an invalid schema")
        if outcome["status"] not in {"passed", "failed"}:
            raise ValueError("executor terminal status is invalid")
        if type(outcome["evidence_sha256"]) is not str or _SHA.fullmatch(outcome["evidence_sha256"]) is None:
            raise ValueError("executor terminal evidence digest is invalid")
        return {
            "schema": "a3-phase1-cell-terminal-v3",
            "cell_id": cell.cell_id,
            "target": "a3",
            "language": "ascend-c",
            "model": cell.backend_model.value,
            "knowledge": cell.knowledge.value,
            "profiling": cell.profiling.value,
            "execution_profile": execution_profile,
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
    def _read_terminal(
        cls,
        cell: A3ExperimentCell,
        path: Path,
        *,
        execution_profile: str | None = None,
    ) -> dict[str, object] | None:
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("terminal cell record is corrupt") from exc
        retained_profile = value.get("execution_profile")
        if execution_profile is None:
            if retained_profile not in A3_EXECUTION_PROFILES:
                raise ValueError("terminal cell record has an invalid execution profile")
            execution_profile = retained_profile
        expected = cls._terminal(
            cell,
            {"status": value.get("status"), "evidence_sha256": value.get("evidence_sha256")},
            execution_profile,
        )
        if value != expected:
            raise ValueError("terminal cell record has foreign or conflicting evidence")
        return value


class SmokeWave(Phase1Wave):
    """One-project, state-isolated gate for all selected foundation cells."""

    def __init__(
        self,
        config: Phase1Config,
        executor: Executor,
        *,
        cells: Sequence[A3ExperimentCell] | None = None,
        execution_profile: str,
        knowledge_identity: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            config, executor, cells=cells, execution_profile=execution_profile
        )
        self._knowledge_identity = _validated_knowledge_identity(knowledge_identity)

    def paths(self, cell: A3ExperimentCell) -> CellPaths:
        root = self.config.state_root / "smoke-cells" / cell.cell_id
        workspace = root / "workspace"
        return CellPaths(
            root, workspace, root / "memory.jsonl", root / "evidence.jsonl",
            root / "terminal.json", workspace / "a3_profile_driver.py",
        )

    def _identity(self, cell: A3ExperimentCell) -> dict[str, object]:
        knowledge = None
        if cell.knowledge is KnowledgeMode.WITH_KDB:
            if self._knowledge_identity is None:
                self._knowledge_identity = authenticate_smoke_knowledge(self.config)
            knowledge = self._knowledge_identity
        profile = load_a3_model_profile(cell.backend_model)
        return {
            "plan_fingerprint": canonical_digest(smoke_dry_run()),
            "proposal_set_sha256": canonical_digest(
                [proposal.as_dict() for proposal in SMOKE_PROPOSALS]
            ),
            "model_profile_sha256": profile.fingerprint,
            "profile_driver_sha256": _research_identity()["profile_driver_sha256"],
            "host_fixture_sha256": _research_identity()["host_fixture_sha256"],
            "knowledge_identity": knowledge,
        }

    def run(self) -> tuple[dict[str, object], ...]:
        records = []
        for cell in self.cells:
            identity = self._identity(cell)
            paths = self.paths(cell)
            retained = self._read_smoke_terminal(cell, paths.terminal, identity)
            if retained is not None:
                records.append(retained)
                continue
            self._stage(paths)
            outcome = dict(self.executor(cell, paths))
            record = self._smoke_terminal(cell, outcome, identity)
            self._publish(paths.terminal, record)
            records.append(record)
        return tuple(records)

    def resume(self) -> tuple[dict[str, object], ...]:
        return self.run()

    def report(self) -> dict[str, object]:
        records = []
        for cell in foundation_cells():
            path = self.paths(cell).terminal
            if path.exists():
                records.append(self._read_smoke_report_terminal(cell, path))
        counts: dict[str, int] = {}
        for record in records:
            status = str(record["status"])
            counts[status] = counts.get(status, 0) + 1
        return {
            "schema": "a3-phase1-foundation-smoke-report-v1",
            "counts": counts,
            "records": records,
        }

    def _read_smoke_report_terminal(
        self, cell: A3ExperimentCell, path: Path,
    ) -> dict[str, object]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("smoke terminal cell record is corrupt") from exc
        knowledge = value.get("knowledge_identity")
        if cell.knowledge is KnowledgeMode.WITHOUT_KDB:
            valid_knowledge = knowledge is None
        else:
            valid_knowledge = (
                isinstance(knowledge, dict)
                and set(knowledge) == _KNOWLEDGE_IDENTITY_FIELDS
                and all(
                    isinstance(item, str) and _SHA.fullmatch(item)
                    for item in knowledge.values()
                )
            )
        static = _research_identity()
        profile = load_a3_model_profile(cell.backend_model)
        expected = {
            "schema": "a3-phase1-smoke-cell-terminal-v1",
            "cell_id": cell.cell_id,
            "target": "a3",
            "language": "ascend-c",
            "model": cell.backend_model.value,
            "knowledge": cell.knowledge.value,
            "profiling": cell.profiling.value,
            "execution_profile": value.get("execution_profile"),
            "plan_fingerprint": canonical_digest(smoke_dry_run()),
            "proposal_set_sha256": canonical_digest(
                [proposal.as_dict() for proposal in SMOKE_PROPOSALS]
            ),
            "model_profile_sha256": profile.fingerprint,
            "profile_driver_sha256": static["profile_driver_sha256"],
            "host_fixture_sha256": static["host_fixture_sha256"],
            "knowledge_identity": knowledge,
            "status": value.get("status"),
            "evidence_sha256": value.get("evidence_sha256"),
        }
        try:
            Phase1Wave._terminal(
                cell,
                {
                    "status": value.get("status"),
                    "evidence_sha256": value.get("evidence_sha256"),
                },
                str(value.get("execution_profile")),
            )
        except ValueError as exc:
            raise ValueError("smoke terminal cell record is corrupt") from exc
        if not valid_knowledge or value != expected:
            raise ValueError("smoke terminal cell record has foreign or conflicting evidence")
        return value

    def _smoke_terminal(
        self,
        cell: A3ExperimentCell,
        outcome: Mapping[str, object],
        identity: Mapping[str, object],
    ) -> dict[str, object]:
        Phase1Wave._terminal(cell, outcome, self.execution_profile)
        return {
            "schema": "a3-phase1-smoke-cell-terminal-v1",
            "cell_id": cell.cell_id,
            "target": "a3",
            "language": "ascend-c",
            "model": cell.backend_model.value,
            "knowledge": cell.knowledge.value,
            "profiling": cell.profiling.value,
            "execution_profile": self.execution_profile,
            **identity,
            **outcome,
        }

    def _read_smoke_terminal(
        self,
        cell: A3ExperimentCell,
        path: Path,
        identity: Mapping[str, object],
    ) -> dict[str, object] | None:
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("smoke terminal cell record is corrupt") from exc
        expected = self._smoke_terminal(
            cell,
            {
                "status": value.get("status"),
                "evidence_sha256": value.get("evidence_sha256"),
            },
            identity,
        )
        if value != expected:
            raise ValueError("smoke terminal cell record has foreign or conflicting evidence")
        return value


__all__ = [
    "CellPaths",
    "Phase1Config",
    "Phase1Wave",
    "SMOKE_PROPOSALS",
    "SmokeWave",
    "authenticate_smoke_knowledge",
    "foundation_cells",
    "full_dry_run",
    "smoke_dry_run",
    "select_foundation_cells",
]
