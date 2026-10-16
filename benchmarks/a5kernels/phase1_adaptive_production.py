"""Thin production composition for adaptive A5 Phase-1 campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
)
from benchmarks.a5kernels.phase1_composition import A5_PROFILING_GUIDANCE
from benchmarks.a5kernels.phase1_memory import HostFact
from benchmarks.a5kernels.phase1_live import InfrastructureFailure
from benchmarks.a5kernels.phase1_experiments import (
    A5Phase1Cell,
    KnowledgeMode,
    ProfilingGuidance,
    admit_runnable_cell,
)
from benchmarks.a5kernels.phase1_project_bz import CatlassProjectBackend, ProjectInfrastructureError
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
from benchmarks.a5kernels.profiling_bz import BZProfileBackend
from benchmarks.a5kernels.phase1_provider import (
    DirectActorDriver,
    DirectCompletionProvider,
    DirectModelProfile,
    Transport,
    direct_profile,
    openai_compatible_transport,
)
from benchmarks.a5kernels.profiling_guidance import verify_profiling_guidance


_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_INITIAL_CONTEXT_SCHEMA = "a5-phase1-production-context-v1"
_CAMPAIGN_IDENTITY_SCHEMA = "a5-phase1-production-identity-v1"


@dataclass(frozen=True)
class AdaptiveProductionPaths:
    """Machine-local paths kept outside persisted experiment state."""

    root: Path
    tla_root: Path
    catlass_source: str
    catlass_revision: str
    device: int = 0
    kdb: Path | None = None
    collection: str | tuple[str, ...] | list[str] | None = None
    profiling_skill_root: Path = field(
        default_factory=lambda: Path.home() / ".agents/skills/ascend-profiling",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "tla_root", Path(self.tla_root))
        object.__setattr__(self, "profiling_skill_root", Path(self.profiling_skill_root))
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

    @property
    def collection_wrapper(self) -> Path:
        return self.profiling_skill_root / "scripts/collect_profile.sh"


@dataclass(frozen=True)
class AdaptivePhase1Production:
    """Inspectable production graph with one retained coordinator."""

    cell: A5Phase1Cell
    profile: DirectModelProfile
    actor: DirectActorDriver
    backend: CatlassProjectBackend
    query_knowledge: ActionDependency
    request_profile: ActionDependency
    request_timing: ActionDependency
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
        if any(manifest.embedding_model != embeddings.model
               or manifest.embedding_revision != embeddings.revision
               for manifest in database.selected_manifests(paths.collection)):
            raise ValueError("KDB manifest does not match the embedding backend")

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


def _measurement_action(measurements, *, profiling: bool) -> ActionDependency:
    role = "profiling" if profiling else "performance-study"

    def measure(payload, context):
        if payload:
            return AdaptiveActionResult({"error": "measurement action accepts an empty object"})
        if context.source is None or context.latest_execution is None or (
            not context.latest_execution.evidence.verified
        ):
            return AdaptiveActionResult({
                "error": "measurement requires HOST_VERIFIED current source; run first",
            })
        request = ProjectExecutor.request(context.prepared, context.source)
        try:
            if profiling:
                metric = context.request.spec.profile_metric
                if metric is None:
                    return AdaptiveActionResult({"error": "project has no profiling metric"})
                result = measurements.profile(request, metric=metric)
            else:
                result = measurements.timing(request)
        except (InfrastructureFailure, ProjectInfrastructureError):
            raise
        except (ValueError, RuntimeError) as exc:
            return AdaptiveActionResult({
                "event": role, "passed": False, "diagnostic": str(exc),
                "next_action": "inspect diagnostics, repair source if needed, run, then retry measurement",
            })
        retained = {}
        for digest, artifact in result["artifact_payloads"].items():
            if context.artifacts.put_json(artifact) != digest:
                raise ValueError("measurement artifact hash does not match payload")
            retained[digest] = role
        evidence = tuple(result["evidence_sha256"])
        if not evidence or set(evidence) - set(retained):
            raise ValueError("measurement result omitted retained evidence")
        summary = result["summary"]
        return AdaptiveActionResult(
            {"event": role, "passed": True, "summary": summary,
             "evidence_sha256": evidence},
            host_facts=tuple(HostFact(
                role, canonical_bytes(summary).decode("ascii"), digest,
            ) for digest in evidence),
            admitted_artifacts=retained,
        )

    return measure


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


def _actor_context(cell: A5Phase1Cell, guide_bytes: bytes, profiling_guidance_text: str | None = None) -> bytes:
    if not isinstance(guide_bytes, bytes) or not guide_bytes:
        raise TypeError("programming guide must be non-empty immutable bytes")
    if hashlib.sha256(guide_bytes).hexdigest() != cell.guide.guide_sha256:
        raise ValueError("programming guide bytes do not match the admitted identity")
    try:
        guide_text = guide_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("programming guide must be UTF-8") from exc
    guidance = (
        A5_PROFILING_GUIDANCE
        if cell.profiling is ProfilingGuidance.WITH_GUIDANCE
        else None
    )
    if cell.profiling is ProfilingGuidance.NEW:
        verify_profiling_guidance(cell.profiling_guide, profiling_guidance_text)
        guidance = profiling_guidance_text
    elif profiling_guidance_text is not None:
        raise ValueError("explicit new guidance cannot change a legacy treatment")
    return canonical_bytes({
        "schema": _INITIAL_CONTEXT_SCHEMA,
        "programming_guide": {
            "sha256": cell.guide.guide_sha256,
            "text": guide_text,
        },
        "profiling_guidance": guidance,
    })


def _bind_campaign_identity(root: Path, cell: A5Phase1Cell, reviewed_snapshot=None) -> None:
    root = Path(root)
    path = root / "campaign-identity.json"
    identity = {
        "schema": _CAMPAIGN_IDENTITY_SCHEMA,
        "cell": cell.as_dict(),
    }
    if reviewed_snapshot is not None:
        reviewed_snapshot.verify()
        identity.update(schema="ascend-catlass-adaptive-phase2-identity-v1",
                        phase="adaptive-phase2", snapshot_id=reviewed_snapshot.snapshot_id)
    expected = canonical_bytes(identity) + b"\n"
    try:
        root.mkdir(parents=True, exist_ok=True)
        if not path.exists() and (root / "state.json").exists():
            raise ValueError("legacy adaptive state has no campaign identity")
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if path.read_bytes() != expected:
                raise ValueError("campaign root belongs to a different treatment cell")
            return
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("campaign identity could not be retained") from exc


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
    project_target: int | None = None,
    max_projects: int | None = None,
    profiling_guidance_text: str | None = None,
    reviewed_snapshot=None, trace_hook=None, trace_recorder_factory=None,
) -> AdaptivePhase1Production:
    """Bind one registered treatment cell without fallback routes."""

    if max_parallel != 1:
        raise ValueError("measurement-enabled production requires max_parallel=1 on its single device")
    admitted = admit_runnable_cell(cell)
    if paths.catlass_revision != admitted.guide.catlass_revision:
        raise ValueError("runtime Catlass revision does not match the admitted guide")
    if not isinstance(initial_memory, bytes):
        raise TypeError("initial memory must be immutable bytes")
    actor_context = _actor_context(admitted, guide_bytes, profiling_guidance_text)
    _bind_campaign_identity(paths.root, admitted, reviewed_snapshot)
    if admitted.knowledge is KnowledgeMode.WITHOUT_KDB:
        if paths.kdb is not None or paths.collection is not None or embeddings is not None:
            raise ValueError("KDB-off treatment must not receive KDB dependencies")
        query_knowledge = disabled_knowledge_action
    else:
        query_knowledge = _knowledge_action(paths, embeddings)
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
    from benchmarks.a5kernels.phase1_project_measurements import ProjectMeasurementBackend
    measurements = ProjectMeasurementBackend(
        backend,
        BZProfileBackend(
            validation_wrapper=str(paths.validation_wrapper),
            session_wrapper=str(paths.session_wrapper),
            collection_wrapper=str(paths.collection_wrapper),
            catlass_source=paths.catlass_source,
            evidence_directory=str(paths.root / "measurement-evidence"),
        ),
    )
    request_profile = _measurement_action(measurements, profiling=True)
    request_timing = _measurement_action(measurements, profiling=False)
    from benchmarks.a5kernels.phase1_project_diagnostics import diagnostic_action
    from benchmarks.ascend.production import compose_adaptive_controller
    project_actor, coordinator = compose_adaptive_controller(
        root=paths.root, objective=objective, initial_memory=initial_memory,
        executor=executor, actor=actor, profile=profile, provider=provider,
        query_knowledge=query_knowledge,
        request_profile=request_profile,
        request_timing=request_timing,
        request_diagnostic=diagnostic_action(backend),
        immutable_context=actor_context,
        profiling_guidance=profiling_guidance_text,
        trace_recorder_factory=trace_recorder_factory,
        max_parallel=max_parallel,
        project_target=project_target,
        max_projects=max_projects,
        reviewed_snapshot=reviewed_snapshot, trace_hook=trace_hook,
        trace_recorder_factory=trace_recorder_factory,
    )
    return AdaptivePhase1Production(
        admitted,
        profile,
        actor,
        backend,
        query_knowledge,
        request_profile,
        request_timing,
        project_actor,
        coordinator,
    )


__all__ = [
    "A5_PROFILING_GUIDANCE",
    "AdaptivePhase1Production",
    "AdaptiveProductionPaths",
    "build_adaptive_phase1",
]
