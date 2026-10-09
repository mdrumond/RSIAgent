"""Fail-closed production binding for qualified A5 Phase 1 cells."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Mapping

from benchmarks.a5kernels.bz import BZSessionAdapter, CatlassValidationExecutor
from benchmarks.a5kernels.candidate import (
    CandidateProfileEvaluation,
)
from benchmarks.a5kernels.evidence import EvidenceKind, EvidenceLedger, canonical_digest
from benchmarks.a5kernels.knowledge import KnowledgeDB
from benchmarks.a5kernels.knowledge_agent import (
    Citation,
    KnowledgeAgent,
    ProgressiveMemoryJournal,
)
from benchmarks.a5kernels.matrix import Workload
from benchmarks.a5kernels.phase1_composition import (
    LiveDependencies,
    LiveProjectRequest,
)
from benchmarks.a5kernels.phase1_experiments import PINNED_CATLASS_REVISION
from benchmarks.a5kernels.phase1_live import (
    BACKOFF_SECONDS,
    MAX_INFRASTRUCTURE_RETRIES,
    InfrastructureFailure,
)
from benchmarks.a5kernels.phase1_memory import HostFact, Phase1ProjectMemory
from benchmarks.a5kernels.phase1_performance import Phase1PerformanceExecution
from benchmarks.a5kernels.phase1_provider import (
    DirectActorDriver,
    DirectCompletionProvider,
    Transport,
    direct_profile,
    openai_compatible_transport,
)
from benchmarks.a5kernels.phase1_runtime import (
    Phase1ProjectRuntime,
    RecoveryEvidence,
)
from benchmarks.a5kernels.production import (
    ProductionPaths,
    _probe_knowledge_query,
    _transformers_cache,
)
from benchmarks.a5kernels.profiling import ProfilingTreatmentController
from benchmarks.a5kernels.profiling_bz import BZProfileBackend
from benchmarks.a5kernels.trial import TrialOrchestrator, _trial_instruction
from config.runtime_paths import resolve_env_file


def _require_host_verification(verified) -> None:
    if verified.passed:
        return
    if verified.exit_code == 255 and verified.session_handle is None:
        raise InfrastructureFailure(
            "A5 candidate verification transport was unavailable"
        )
    raise RuntimeError("submitted A5 candidate did not pass host verification")


def load_direct_environment(
    environment: Mapping[str, str] | None = None,
    *,
    env_file: Path | None = None,
) -> dict[str, str]:
    """Load only the two direct-provider credentials without exposing values."""

    source = os.environ if environment is None else environment
    result = dict(source)
    path = resolve_env_file(result) if env_file is None else Path(env_file)
    missing = {
        name for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY")
        if not result.get(name, "").strip()
    }
    if missing:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                name, separator, value = line.partition("=")
                if separator and name in missing and value.strip():
                    result[name] = value.strip()
        except FileNotFoundError:
            pass
    return result


class ProductionProjectExecutor:
    """Bind one registered project to direct model and checked BZ-A5 authorities."""

    def __init__(self, execution_backend, profile_backend, provider, *, device: int):
        self.execution_backend = execution_backend
        self.profile_backend = profile_backend
        self.actor = DirectActorDriver(provider)
        self.device = device

    def __call__(
        self,
        request: LiveProjectRequest,
        runtime: Phase1ProjectRuntime,
        knowledge: KnowledgeAgent,
        profiling_guidance: str | None,
        ledger: EvidenceLedger,
    ) -> Phase1ProjectMemory:
        backend = runtime.backend(self.execution_backend, device=self.device)
        recovery = self._run_recovery_starter(request, runtime, backend, ledger)
        profiling = CandidateProfileEvaluation(
            ProfilingTreatmentController(
                self.profile_backend,
                treatment_enabled=profiling_guidance is not None,
            ),
            device=self.device,
        )
        attempt = self._next_attempt(request.memory_path)
        instruction = self._instruction(request, profiling_guidance, recovery)
        trial = TrialOrchestrator.run_bound(
            trial_root=attempt,
            request_payload={
                "schema": "a5-phase1-bound-project-v1",
                "cell": request.cell.as_dict(),
                "proposal": request.proposal.as_dict(),
                "guide_sha256": request.guide_sha256,
                "qualification_sha256": request.qualification_sha256,
            },
            language="catlass-dsl",
            workload=Workload.SMOKE_VECTOR_ADD,
            instruction=instruction,
            profile=direct_profile(request.cell.model),
            backend=backend,
            actor=self.actor,
            knowledge=knowledge,
            profiling=profiling,
            profiling_guidance=profiling_guidance is not None,
            ledger=ledger,
        )
        _require_host_verification(trial.verified)

        facts = [HostFact(
            "host-verification",
            "submitted Catlass DSL candidate passed host verification",
            trial.verified.evidence_sha256,
        )]
        manifests = trial.final_profile.get("evidence_manifest_sha256", ())
        if not isinstance(manifests, (list, tuple)):
            raise RuntimeError("final profile omitted its evidence manifests")
        facts.extend(
            HostFact(
                "profiling",
                "treatment-independent final profile captured",
                manifest,
            )
            for manifest in manifests
        )
        facts.extend(self._performance_facts(request, runtime, trial, ledger))
        source = (trial.workspace / "kernel.py").read_bytes()
        actions = tuple(
            entry.payload["action"] for entry in ledger.entries
            if entry.kind == EvidenceKind.ACTION.value
            and isinstance(entry.payload.get("action"), str)
        )
        citations = self._citations(knowledge)
        return Phase1ProjectMemory(
            request.proposal,
            source_revision=hashlib.sha256(source).hexdigest(),
            actions=actions,
            host_facts=tuple(facts),
            kdb_citations=citations,
        )

    def _run_recovery_starter(self, request, runtime, backend, ledger) -> str | None:
        recovery = runtime.recovery
        if recovery is None:
            return None
        root = request.memory_path / "recovery-starter"
        root.mkdir(parents=True, exist_ok=True)
        (root / "kernel.py").write_text(recovery.source, encoding="utf-8")
        attempt_id = f"{request.cell.cell_id[:24]}-p{request.ordinal}-recovery"
        if recovery.required_evidence is RecoveryEvidence.COMPILE_FAILURE:
            result = dict(backend.compile(root, "catlass-dsl", attempt_id))
            if result.get("passed") is not False:
                raise RuntimeError("registered compile-recovery starter did not fail")
            evidence = result.get("attestation_sha256")
            stage = "compile"
        else:
            candidate = backend.run(
                root, "catlass-dsl", Workload.SMOKE_VECTOR_ADD, attempt_id, ledger,
            )
            if candidate.verified.passed:
                raise RuntimeError("registered runtime-recovery starter did not fail")
            evidence = candidate.verified.evidence_sha256
            stage = "host-verification"
        ledger.append(EvidenceKind.RESULT, {
            "event": "registered-recovery-starter-observed",
            "stage": stage,
            "source_sha256": recovery.source_sha256,
            "evidence_sha256": evidence,
        })
        return json.dumps({
            "registered_recovery": {
                "stage": stage,
                "fault_count": recovery.fault_count,
                "source_sha256": recovery.source_sha256,
                "evidence_sha256": evidence,
            }
        }, sort_keys=True, separators=(",", ":"))

    def _performance_facts(self, request, runtime, trial, ledger):
        if request.proposal.evidence_preset.value not in {
            "correctness-timing", "msprof-guided",
        }:
            return []
        variant = runtime.run_study(
            self.execution_backend,
            trial.workspace,
            f"{request.cell.cell_id[:20]}-p{request.ordinal}-study",
            ledger,
            device=self.device,
        )
        result = Phase1PerformanceExecution(self.profile_backend).run(
            request.proposal, (variant,)
        )
        if result is None:
            raise RuntimeError("registered performance project produced no study")
        payload = asdict(result)
        ledger.append(EvidenceKind.RESULT, {
            "event": "registered-performance-study",
            "proposal": request.proposal.as_dict(),
            "result": payload,
        })
        evidence = []
        for timing in result.timing:
            evidence.extend(timing.evidence_sha256)
        evidence.extend(metric.evidence_manifest_sha256 for metric in result.metrics)
        return [
            HostFact("profiling", "registered performance study completed", digest)
            for digest in evidence
        ] or [HostFact(
            "profiling", "registered performance study completed",
            canonical_digest(payload),
        )]

    @staticmethod
    def _instruction(request, profiling_guidance, recovery):
        sections = [
            _trial_instruction("catlass-dsl", Workload.SMOKE_VECTOR_ADD),
            "Pinned programming guide:\n" + request.guide_text,
            "Registered project:\n" + json.dumps(
                request.proposal.as_dict(), sort_keys=True, separators=(",", ":")
            ),
            "Progressive memory:\n" + request.memory_context,
        ]
        if recovery is not None:
            sections.append("Host-observed recovery starter:\n" + recovery)
        if profiling_guidance is not None:
            sections.append("Profiling guidance:\n" + profiling_guidance)
        return "\n\n".join(sections)

    @staticmethod
    def _citations(knowledge: KnowledgeAgent) -> tuple[Citation, ...]:
        citations = []
        for entry in knowledge.journal.read():
            citations.extend(Citation(**item) for item in entry["citations"])
        return tuple(citations)

    @staticmethod
    def _next_attempt(memory_path: Path) -> Path:
        root = memory_path / "provider-attempts"
        root.mkdir(parents=True, exist_ok=True)
        ordinal = sum(path.is_dir() for path in root.glob("attempt-*")) + 1
        return root / f"attempt-{ordinal:02d}"


def build_phase1_live_dependencies(
    paths: ProductionPaths,
    *,
    environment: Mapping[str, str] | None = None,
    env_file: Path | None = None,
    transport: Transport = openai_compatible_transport,
    execution_backend=None,
    profile_backend=None,
    embeddings=None,
) -> LiveDependencies:
    """Validate all dependencies, then construct the real eight-cell binding."""

    if paths.catlass_revision != PINNED_CATLASS_REVISION:
        raise ValueError("production Catlass revision does not match the Phase 1 pin")
    values = load_direct_environment(environment, env_file=env_file)
    provider = DirectCompletionProvider(values, transport=transport)
    if transport is openai_compatible_transport and importlib.util.find_spec("openai") is None:
        raise RuntimeError("the openai package is required for direct providers")
    if not paths.results_root.parent.is_dir():
        raise ValueError("results root parent does not exist")
    if not paths.kdb.is_file() or not paths.bge_cache.is_dir():
        raise ValueError("pinned embeddings and KDB are required before dispatch")
    execution_wrappers = (
        paths.validation_wrapper,
        paths.session_wrapper,
        paths.upload_wrapper,
        paths.provenance_wrapper,
    )
    if execution_backend is None and any(
        not path.is_file() or not os.access(path, os.X_OK)
        for path in execution_wrappers
    ):
        raise ValueError("checked BZ-A5 execution wrappers are unavailable")
    if profile_backend is None and (
        not paths.validation_wrapper.is_file()
        or not os.access(paths.validation_wrapper, os.X_OK)
        or not paths.collection_wrapper.is_file()
        or not os.access(paths.collection_wrapper, os.X_OK)
    ):
        raise ValueError("checked BZ-A5 profiling wrappers are unavailable")

    if embeddings is None:
        from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings

        embeddings = PinnedBGEEmbeddings(
            cache_dir=_transformers_cache(paths.bge_cache)
        )
        embeddings.embed_query(["A5 Phase 1 preflight"])
    database = KnowledgeDB.open_read_only(paths.kdb, embeddings)
    try:
        manifest = database.manifest(paths.collection)
        if (
            manifest.embedding_model != embeddings.model
            or manifest.embedding_revision != embeddings.revision
        ):
            raise ValueError("KDB manifest does not match the pinned embeddings")
        _probe_knowledge_query(database, paths.collection)
    finally:
        database.close()

    if execution_backend is None:
        command = CatlassValidationExecutor(
            upload_wrapper=str(paths.upload_wrapper),
            validation_wrapper=str(paths.validation_wrapper),
            catlass_source=paths.catlass_source,
            catlass_revision=paths.catlass_revision,
        )
        if not command.runtime_provenance:
            raise RuntimeError("Catlass runtime provenance is unavailable")
        command.probe_device(paths.device)
        execution_backend = BZSessionAdapter(
            command, session_wrapper=str(paths.session_wrapper)
        )
    if profile_backend is None:
        profile_backend = BZProfileBackend(
            validation_wrapper=str(paths.validation_wrapper),
            session_wrapper=str(paths.session_wrapper),
            collection_wrapper=str(paths.collection_wrapper),
            catlass_source=paths.catlass_source,
            evidence_directory=str(paths.results_root / "profile-evidence"),
            observation_retries=MAX_INFRASTRUCTURE_RETRIES,
            observation_backoff_seconds=BACKOFF_SECONDS,
        )

    def knowledge_factory(_cell, memory_path):
        selected = KnowledgeDB.open_read_only(paths.kdb, embeddings)
        agent = KnowledgeAgent(
            enabled=True,
            database=selected,
            collection=paths.collection,
            journal=ProgressiveMemoryJournal(memory_path / "knowledge.jsonl"),
        )
        selected.close()
        return agent

    executor = ProductionProjectExecutor(
        execution_backend, profile_backend, provider, device=paths.device
    )
    return LiveDependencies(executor, knowledge_factory)


__all__ = [
    "ProductionProjectExecutor",
    "build_phase1_live_dependencies",
    "load_direct_environment",
]
