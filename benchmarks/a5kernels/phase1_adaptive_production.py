"""Thin production composition for adaptive A5 Phase-1 campaigns."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Mapping

from benchmarks.a5kernels.bz import (
    BZCommandExecutor,
    CatlassValidationExecutor,
)
from benchmarks.a5kernels.evidence import EvidenceLedger, canonical_bytes
from benchmarks.a5kernels.knowledge import KnowledgeDB
from benchmarks.a5kernels.knowledge_agent import (
    KnowledgeAgent,
    KnowledgeQuery,
    ProgressiveMemoryJournal,
)
from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
from benchmarks.a5kernels.phase1_adaptive_actor import (
    ActionDependency,
    AdaptiveActionContext,
    AdaptiveActionResult,
    AdaptiveActorProjectExecutor,
    disabled_knowledge_action,
    disabled_profile_action,
)
from benchmarks.a5kernels.phase1_composition import A5_PROFILING_GUIDANCE
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from benchmarks.a5kernels.phase1_experiments import (
    A5Phase1Cell,
    KnowledgeMode,
    ProfilingGuidance,
    admit_runnable_cell,
)
from benchmarks.a5kernels.phase1_project_bz import CatlassProjectBackend
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from benchmarks.a5kernels.phase1_provider import (
    DirectActorDriver,
    DirectCompletionProvider,
    DirectModelProfile,
    Transport,
    direct_profile,
    openai_compatible_transport,
)


_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_INITIAL_CONTEXT_SCHEMA = "a5-phase1-production-context-v1"


@dataclass(frozen=True)
class AdaptiveProductionPaths:
    """Machine-local paths kept outside persisted experiment state."""

    root: Path
    tla_root: Path
    catlass_source: str
    catlass_revision: str
    device: int = 0
    kdb: Path | None = None
    collection: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "tla_root", Path(self.tla_root))
        if self.kdb is not None:
            object.__setattr__(self, "kdb", Path(self.kdb))
        if not self.catlass_source.startswith("/"):
            raise ValueError("catlass_source must be an absolute retained BZ path")
        if _REVISION.fullmatch(self.catlass_revision) is None:
            raise ValueError("catlass_revision must be a full lowercase Git revision")
        if type(self.device) is not int or self.device < 0:
            raise ValueError("device must be a non-negative integer")

    @property
    def validation_wrapper(self) -> Path:
        return self.tla_root / "execution-profiles/catlass-validation.sh"

    @property
    def upload_wrapper(self) -> Path:
        return self.tla_root / "execution-profiles/bz-a5/upload.sh"

    @property
    def session_wrapper(self) -> Path:
        return self.tla_root / "execution-profiles/bz-a5/session.sh"


@dataclass(frozen=True)
class AdaptivePhase1Production:
    """Inspectable production graph with one retained coordinator."""

    cell: A5Phase1Cell
    profile: DirectModelProfile
    actor: DirectActorDriver
    backend: CatlassProjectBackend
    query_knowledge: ActionDependency
    request_profile: ActionDependency
    project_actor: AdaptiveActorProjectExecutor
    coordinator: AdaptivePhase1Coordinator

    def run(self) -> dict[str, object]:
        return self.coordinator.run()

    def resume(self) -> dict[str, object]:
        return self.coordinator.resume()

    def report(self) -> dict[str, object]:
        return self.coordinator.report()


def _knowledge_action(paths: AdaptiveProductionPaths, embeddings) -> ActionDependency:
    if paths.kdb is None or not paths.kdb.is_file() or not paths.collection:
        raise ValueError("KDB-on treatment requires a readable KDB and collection")
    if embeddings is None:
        raise ValueError("KDB-on treatment requires a pinned embedding backend")
    # Opening once here validates the selected manifest before any provider call.
    with KnowledgeDB.open_read_only(paths.kdb, embeddings) as database:
        database.manifest(paths.collection)

    def query(
        payload: Mapping[str, object], context: AdaptiveActionContext,
    ) -> AdaptiveActionResult:
        if set(payload) - {"query", "limit"}:
            raise ValueError("knowledge action fields must be query and optional limit")
        action = KnowledgeQuery(payload.get("query"), payload.get("limit", 5))
        database = KnowledgeDB.open_read_only(paths.kdb, embeddings)
        agent = KnowledgeAgent(
            enabled=True,
            database=database,
            collection=paths.collection,
            journal=ProgressiveMemoryJournal(
                Path(context.request.branch_root) / "knowledge.jsonl"
            ),
        )
        database.close()
        try:
            results = agent.query(action)
        finally:
            agent.close()
        rows = []
        admitted = {}
        for result in results:
            digest = context.artifacts.put_bytes(result.text.encode("utf-8"))
            admitted[digest] = "knowledge"
            rows.append({
                "citation": asdict(result.citation),
                "text": result.text,
                "score": result.score,
                "artifact_sha256": digest,
            })
        return AdaptiveActionResult(
            {"event": "knowledge-query", "available": True, "results": rows},
            citations=tuple(result.citation for result in results),
            admitted_artifacts=admitted,
        )

    return query


def _guidance_action(
    _payload: Mapping[str, object], _context: AdaptiveActionContext,
) -> AdaptiveActionResult:
    return AdaptiveActionResult({
        "event": "profiling-guidance",
        "available": True,
        "guidance": A5_PROFILING_GUIDANCE,
        "measurement_available": False,
    })


def _checked_command(paths: AdaptiveProductionPaths) -> BZCommandExecutor:
    wrappers = (paths.validation_wrapper, paths.upload_wrapper, paths.session_wrapper)
    if any(not path.is_file() or not os.access(path, os.X_OK) for path in wrappers):
        raise ValueError("checked BZ-A5 wrappers are unavailable")
    command = CatlassValidationExecutor(
        upload_wrapper=str(paths.upload_wrapper),
        validation_wrapper=str(paths.validation_wrapper),
        catlass_source=paths.catlass_source,
        catlass_revision=paths.catlass_revision,
    )
    provenance = tuple(command.runtime_provenance)
    if (
        any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(value, str) and value for value in item)
            for item in provenance
        )
        or len(dict(provenance)) != len(provenance)
    ):
        raise RuntimeError("Catlass runtime provenance is invalid")
    values = dict(provenance)
    if (
        values.get("catlass_revision") != paths.catlass_revision
        or values.get("catlass_source") != paths.catlass_source
        or values.get("execution_profile") != "bz-a5"
    ):
        raise RuntimeError("Catlass runtime provenance does not match configuration")
    command.probe_device(paths.device)
    return command


def _initial_context(
    cell: A5Phase1Cell, guide_bytes: bytes, user_memory: bytes,
) -> bytes:
    if not isinstance(guide_bytes, bytes) or not guide_bytes:
        raise TypeError("programming guide must be non-empty immutable bytes")
    if hashlib.sha256(guide_bytes).hexdigest() != cell.guide.guide_sha256:
        raise ValueError("programming guide bytes do not match the admitted identity")
    try:
        guide_text = guide_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("programming guide must be UTF-8") from exc
    if not isinstance(user_memory, bytes):
        raise TypeError("initial memory must be immutable bytes")
    guidance = (
        A5_PROFILING_GUIDANCE
        if cell.profiling is ProfilingGuidance.WITH_GUIDANCE
        else None
    )
    return canonical_bytes({
        "schema": _INITIAL_CONTEXT_SCHEMA,
        "programming_guide": {
            "sha256": cell.guide.guide_sha256,
            "base64": base64.b64encode(guide_bytes).decode("ascii"),
            "text": guide_text,
        },
        "user_initial_memory": {
            "sha256": hashlib.sha256(user_memory).hexdigest(),
            "base64": base64.b64encode(user_memory).decode("ascii"),
        },
        "profiling_guidance": guidance,
    })


def build_adaptive_phase1(
    *,
    cell: A5Phase1Cell,
    paths: AdaptiveProductionPaths,
    objective: str,
    initial_memory: bytes,
    guide_bytes: bytes,
    environment: Mapping[str, str] | None = None,
    embeddings=None,
    provider_transport: Transport = openai_compatible_transport,
    command_executor: BZCommandExecutor | None = None,
    max_parallel: int = 1,
) -> AdaptivePhase1Production:
    """Bind one registered treatment cell without fallback routes."""

    admitted = admit_runnable_cell(cell)
    composed_memory = _initial_context(admitted, guide_bytes, initial_memory)
    if admitted.knowledge is KnowledgeMode.WITHOUT_KDB:
        if paths.kdb is not None or paths.collection is not None or embeddings is not None:
            raise ValueError("KDB-off treatment must not receive KDB dependencies")
        query_knowledge = disabled_knowledge_action
    else:
        query_knowledge = _knowledge_action(paths, embeddings)
    request_profile = (
        _guidance_action
        if admitted.profiling is ProfilingGuidance.WITH_GUIDANCE
        else disabled_profile_action
    )
    command = _checked_command(paths) if command_executor is None else command_executor
    provider = DirectCompletionProvider(
        os.environ if environment is None else environment,
        transport=provider_transport,
    )
    profile = direct_profile(admitted.model)
    actor = DirectActorDriver(provider)
    backend = CatlassProjectBackend(
        command,
        session_wrapper=str(paths.session_wrapper),
        evidence_root=paths.root / "project-bundles",
        device=paths.device,
    )
    executor = ProjectExecutor(
        backend, EvidenceLedger(paths.root / "project-evidence.jsonl")
    )
    project_actor = AdaptiveActorProjectExecutor(
        executor,
        actor,
        profile,
        query_knowledge=query_knowledge,
        request_profile=request_profile,
        immutable_context=composed_memory,
    )
    curriculum = CurriculumAgent(
        lambda messages: provider.complete(profile, messages).text
    )
    coordinator = AdaptivePhase1Coordinator(
        paths.root,
        objective,
        composed_memory,
        curriculum,
        project_actor,
        max_parallel=max_parallel,
    )
    return AdaptivePhase1Production(
        admitted,
        profile,
        actor,
        backend,
        query_knowledge,
        request_profile,
        project_actor,
        coordinator,
    )


__all__ = [
    "A5_PROFILING_GUIDANCE",
    "AdaptivePhase1Production",
    "AdaptiveProductionPaths",
    "build_adaptive_phase1",
]
