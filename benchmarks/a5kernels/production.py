"""Production composition for the preregistered Catlass A5 pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from typing import Callable, Mapping

from config.runtime_paths import resolve_env_file


PILOT_CELL_ID = "cell-97d03f4e20e35d21"
PILOT_WORKLOAD = "smoke-vector-add"


def _transformers_cache(path: Path) -> Path:
    """Accept either a Hugging Face home or its direct ``hub`` cache."""

    hub = path / "hub"
    return hub if hub.is_dir() else path


@dataclass(frozen=True)
class ProductionPaths:
    """Explicit machine-local inputs; none are persisted in experiment source."""

    tla_root: Path
    profiling_skill_root: Path
    catlass_source: str
    catlass_revision: str
    bge_cache: Path
    kdb: Path
    collection: str
    results_root: Path
    device: int

    def __post_init__(self) -> None:
        if not self.catlass_source.startswith("/"):
            raise ValueError("catlass_source must be an absolute retained BZ path")
        if re.fullmatch(r"[0-9a-f]{40}", self.catlass_revision) is None:
            raise ValueError("catlass_revision must be a lowercase 40-character SHA")
        if not self.collection.strip():
            raise ValueError("collection must be non-empty")
        if self.device < 0:
            raise ValueError("device must be non-negative")

    @property
    def validation_wrapper(self) -> Path:
        return self.tla_root / "execution-profiles/catlass-validation.sh"

    @property
    def session_wrapper(self) -> Path:
        return self.tla_root / "execution-profiles/bz-a5/session.sh"

    @property
    def upload_wrapper(self) -> Path:
        return self.tla_root / "execution-profiles/bz-a5/upload.sh"

    @property
    def collection_wrapper(self) -> Path:
        return self.profiling_skill_root / "scripts/collect_profile.sh"


def _credential_present(environment: Mapping[str, str], env_file: Path) -> bool:
    if environment.get("OPENROUTER_API_KEY", "").strip():
        return True
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "OPENROUTER_API_KEY" and value.strip():
                return True
    except FileNotFoundError:
        pass
    return False


def preflight(
    paths: ProductionPaths,
    *,
    environment: Mapping[str, str] | None = None,
    embedding_probe: Callable[[object], None] | None = None,
    runtime_probe: Callable[[ProductionPaths], None] | None = None,
) -> dict[str, object]:
    """Validate every local input before an Actor or BZ workload can start."""

    from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings
    from benchmarks.a5kernels.knowledge import KnowledgeDB

    values = os.environ if environment is None else environment
    checks: dict[str, bool] = {
        "openrouter_credential": _credential_present(
            values, resolve_env_file(values)
        ),
        "results_parent": paths.results_root.parent.is_dir(),
        "results_absent": not paths.results_root.exists(),
    }
    for name, wrapper in {
        "catlass_validation_wrapper": paths.validation_wrapper,
        "bz_session_wrapper": paths.session_wrapper,
        "bz_upload_wrapper": paths.upload_wrapper,
        "profile_collection_wrapper": paths.collection_wrapper,
    }.items():
        checks[name] = wrapper.is_file() and os.access(wrapper, os.X_OK)
    checks["bge_cache"] = paths.bge_cache.is_dir()
    checks["kdb"] = paths.kdb.is_file()

    if all(
        checks[name]
        for name in (
            "catlass_validation_wrapper",
            "bz_session_wrapper",
            "bz_upload_wrapper",
        )
    ):
        try:
            (runtime_probe or _probe_catlass_runtime)(paths)
            checks["catlass_runtime"] = True
        except Exception:
            # Preflight output is deliberately bounded and never includes remote
            # paths, wrapper logs, credentials, or transport diagnostics.
            checks["catlass_runtime"] = False
    else:
        checks["catlass_runtime"] = False

    if checks["bge_cache"] and checks["kdb"]:
        embeddings = PinnedBGEEmbeddings(cache_dir=_transformers_cache(paths.bge_cache))
        try:
            # Loading and one CPU inference prove the pinned revision is complete,
            # local-only, dimensionally valid, and usable by this process.
            (embedding_probe or (lambda backend: backend.embed_query(["A5 preflight"])))(
                embeddings
            )
            database = KnowledgeDB.open_read_only(paths.kdb, embeddings)
            try:
                manifest = database.manifest(paths.collection)
                checks["kdb_manifest"] = (
                    manifest.embedding_model == embeddings.model
                    and manifest.embedding_revision == embeddings.revision
                )
            finally:
                database.close()
        except Exception:
            checks["kdb_manifest"] = False
    else:
        checks["kdb_manifest"] = False

    return {
        "ready": all(checks.values()),
        "cell_id": PILOT_CELL_ID,
        "workload": PILOT_WORKLOAD,
        "checks": checks,
    }


def _probe_catlass_runtime(paths: ProductionPaths) -> None:
    """Use the checked profile's provenance probe for the exact retained runtime."""

    from benchmarks.a5kernels.bz import CatlassValidationExecutor

    executor = CatlassValidationExecutor(
        upload_wrapper=str(paths.upload_wrapper),
        validation_wrapper=str(paths.validation_wrapper),
        catlass_source=paths.catlass_source,
        catlass_revision=paths.catlass_revision,
    )
    # This checked wrapper probe validates the configured BZ route and binds the
    # retained source to its exact revision without submitting a workload.
    if not executor.runtime_provenance:
        raise RuntimeError("Catlass runtime provenance is unavailable")


def build_production_trial(paths: ProductionPaths):
    """Compose the fixed model, KDB, BZ runtime, and profiling authorities."""

    from benchmarks.a5kernels.bz import (
        BZSessionAdapter,
        CatlassValidationExecutor,
    )
    from benchmarks.a5kernels.candidate import (
        CandidateProfileEvaluation,
        CatlassCandidateBackend,
    )
    from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings
    from benchmarks.a5kernels.fixtures import Language
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    from benchmarks.a5kernels.knowledge_agent import (
        KnowledgeAgent,
        ProgressiveMemoryJournal,
    )
    from benchmarks.a5kernels.matrix import (
        KnowledgeMode,
        RuntimeCapabilities,
    )
    from benchmarks.a5kernels.model_profile import CANONICAL_MODEL
    from benchmarks.a5kernels.profiling import ProfilingTreatmentController
    from benchmarks.a5kernels.profiling_bz import BZProfileBackend
    from benchmarks.a5kernels.trial import CoreAttemptDriver, TrialOrchestrator
    from config.settings import Config
    from core.trace import ArtifactSink

    command = CatlassValidationExecutor(
        upload_wrapper=str(paths.upload_wrapper),
        validation_wrapper=str(paths.validation_wrapper),
        catlass_source=paths.catlass_source,
        catlass_revision=paths.catlass_revision,
    )
    candidate = CatlassCandidateBackend(
        BZSessionAdapter(command, session_wrapper=str(paths.session_wrapper))
    )
    embeddings = PinnedBGEEmbeddings(cache_dir=_transformers_cache(paths.bge_cache))
    database = KnowledgeDB.open_read_only(paths.kdb, embeddings)

    profile_backend = BZProfileBackend(
        validation_wrapper=str(paths.validation_wrapper),
        collection_wrapper=str(paths.collection_wrapper),
        catlass_source=paths.catlass_source,
        evidence_directory=str(paths.results_root / "profile-evidence"),
    )

    def knowledge_factory(memory, cell):
        if cell.knowledge is not KnowledgeMode.WITH_KDB:
            return KnowledgeAgent(
                enabled=False,
                journal=ProgressiveMemoryJournal(memory / "knowledge.jsonl"),
            )
        return KnowledgeAgent(
            enabled=True,
            database=database,
            collection=paths.collection,
            journal=ProgressiveMemoryJournal(memory / "knowledge.jsonl"),
        )

    def profiling_factory(cell):
        controller = ProfilingTreatmentController(
            profile_backend,
            treatment_enabled=cell.profiling.value == "with-profiling-guidance",
        )
        return CandidateProfileEvaluation(controller, device=paths.device)

    capabilities = RuntimeCapabilities(
        languages=frozenset({Language.CATLASS_DSL}),
        model_ids=frozenset({CANONICAL_MODEL}),
        kdb=True,
        profiling_guidance=True,
    )
    actor = CoreAttemptDriver(
        object(), Config(agent_decided_stop=True),
        lambda context: ArtifactSink(str(context)),
    )
    return TrialOrchestrator(
        paths.results_root,
        capabilities,
        candidate,
        actor,
        knowledge_factory=knowledge_factory,
        profiling_factory=profiling_factory,
    )


def run_pilot(paths: ProductionPaths) -> dict[str, object]:
    """Run only the preregistered Catlass/KDB/no-guidance pilot cell."""

    report = preflight(paths)
    if not report["ready"]:
        raise RuntimeError("A5 pilot preflight failed: " + json.dumps(report["checks"]))
    from benchmarks.a5kernels.matrix import Workload, initial_matrix

    cell = next(cell for cell in initial_matrix().cells if cell.cell_id == PILOT_CELL_ID)
    result = build_production_trial(paths).run(cell, Workload.SMOKE_VECTOR_ADD)
    return {
        "cell_id": cell.cell_id,
        "workload": PILOT_WORKLOAD,
        "passed": result.verified.passed,
        "iterations": result.outcome.iterations,
        "tokens": result.outcome.tokens,
        "wall_time_s": result.outcome.wall_time_s,
        "attestation_sha256": result.verified.attestation_sha256,
        "evidence_sha256": result.evidence_sha256,
        "final_profile": dict(result.final_profile),
        "workspace": str(result.workspace),
    }
