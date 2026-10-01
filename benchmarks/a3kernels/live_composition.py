"""Concrete, dependency-injected composition for an A3 Phase 1 wave."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import re
from typing import Callable, Mapping, Protocol

from benchmarks.a3_experiments import A3ExperimentCell
from benchmarks.a3_model_profiles import A3Completion, A3ModelProfile, complete_a3, load_a3_model_profile
from benchmarks.a3kernels.candidate import A3CandidateBackend, CandidateCompilation
from benchmarks.a3kernels.embeddings import PinnedBGEEmbeddings
from benchmarks.a3kernels.knowledge import CollectionManifest, KnowledgeDB
from benchmarks.a3kernels.knowledge_agent import KnowledgeAgent, QueryJournal
from benchmarks.a3kernels.phase1_evidence import EvidenceLedger, canonical_bytes, canonical_digest
from benchmarks.a3kernels.phase1_memory import AuthoritativeEvidence, Phase1LearningJournal
from benchmarks.a3kernels.phase1_protocol import FailedEvidence, VerifiedResult
from benchmarks.a3kernels.phase1_registry import CurriculumProposal, DEFAULT_PROPOSALS
from benchmarks.a3kernels.phase1_wave import CellPaths, Phase1Config
from benchmarks.a3kernels.project_execution import (
    PerformancePreset, ProjectRuntimePolicy, RecoveryEvidence,
)
from benchmarks.a3kernels.profiling import (
    CompactProfileResult, ProfilingTreatment, TimingResult,
)
from benchmarks.a3kernels.profiling_gz import GZA3ProfilingBackend
from benchmarks.a3kernels.profiling_bz import BZA3ProfilingBackend
from benchmarks.a3kernels.remote_candidate import GZA3RemoteCandidateBackend
from benchmarks.a3kernels.remote_candidate_bz import BZA3RemoteCandidateBackend
from benchmarks.a3kernels.trial import A3TrialLoop


_SHA = re.compile(r"[0-9a-f]{64}")
_SYSTEM = """You are running one bounded A3 Ascend C learning project. Reply with
exactly one JSON action from the supplied schema. Treat host compile, verification,
and profiling observations as authoritative. Never claim correctness yourself."""


class CandidateFactory(Protocol):
    def __call__(self, cell: A3ExperimentCell, proposal: CurriculumProposal, paths: CellPaths): ...


class ProfilerFactory(Protocol):
    def __call__(
        self, cell: A3ExperimentCell, proposal: CurriculumProposal, paths: CellPaths,
        candidate: object,
    ): ...


@dataclass(frozen=True)
class LiveDependencies:
    actor_factory: Callable[[A3ExperimentCell, CurriculumProposal], Callable[[A3ModelProfile, str], A3Completion]]
    candidate_factory: CandidateFactory
    knowledge_factory: Callable[[A3ExperimentCell, CellPaths], object]
    profiler_factory: ProfilerFactory
    execution_profile: str = "gz-a3"


class AuthoritativeResultStore:
    """Small durable resolver populated only from typed backend results."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._records: dict[str, AuthoritativeEvidence] = {}
        if self.path.exists():
            for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    record = AuthoritativeEvidence(**json.loads(line))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid authoritative evidence at line {number}") from exc
                if record.evidence_sha256 in self._records:
                    raise ValueError("duplicate authoritative evidence")
                self._records[record.evidence_sha256] = record

    def register(
        self, kind: str, evidence_sha256: str, project_id: str,
        candidate_sha256: str, success: bool,
    ) -> AuthoritativeEvidence:
        record = AuthoritativeEvidence(
            kind, evidence_sha256, project_id, candidate_sha256, success
        )
        prior = self._records.get(evidence_sha256)
        if prior is not None:
            if prior != record:
                raise ValueError("conflicting authoritative evidence")
            return prior
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as stream:
            stream.write(canonical_bytes(asdict(record)) + b"\n")
            stream.flush(); os.fsync(stream.fileno())
        self._records[evidence_sha256] = record
        return record

    def resolve(self, evidence_sha256: str) -> AuthoritativeEvidence | None:
        return self._records.get(evidence_sha256)


class _RecordingCandidate:
    def __init__(self, backend, store: AuthoritativeResultStore) -> None:
        self.backend, self.store = backend, store
        self.verified: dict[str, VerifiedResult] = {}

    def compile(self, source, workdir, **options):
        result = self.backend.compile(source, workdir, **options)
        if isinstance(result, CandidateCompilation):
            self.store.register(
                "compile", result.attestation_sha256, result.plan.project_id,
                result.plan.source_fingerprint, True,
            )
        return result

    def run(self, source, workdir, **options):
        result = self.backend.run(source, workdir, **options)
        if isinstance(result, VerifiedResult):
            self.store.register(
                "host-verification", result.evidence_sha256, result.project_id,
                result.source_fingerprint, result.passed,
            )
            if result.passed:
                self.verified[result.execution_id] = result
        return result


class _RecordingProfiler:
    def __init__(self, backend, store: AuthoritativeResultStore, project_id: str) -> None:
        self.backend, self.store, self.project_id = backend, store, project_id

    def profile(self, request):
        result = self.backend.profile(request)
        if isinstance(result, CompactProfileResult):
            self.store.register(
                "profiling", result.evidence_sha256, self.project_id,
                result.source_fingerprint, True,
            )
        return result

    def time(self, binding, dimensions):
        result = self.backend.time(binding, dimensions)
        if not isinstance(result, TimingResult):
            raise TypeError("timing backend returned an invalid result")
        self.store.register(
            "profiling", result.evidence_sha256, self.project_id,
            result.source_fingerprint, True,
        )
        return result


class _RecoveryCandidate:
    """Require the registered starter failure before accepting a repair."""

    def __init__(self, backend, policy: ProjectRuntimePolicy) -> None:
        self.backend, self.policy = backend, policy
        self.observed = False

    def compile(self, source, workdir, **options):
        recovery = self.policy.recovery
        if recovery is not None and not self.observed:
            if source != recovery.source:
                raise ValueError("registered recovery starter must be attempted first")
            result = self.backend.compile(source, workdir, **options)
            if recovery.required_evidence is RecoveryEvidence.COMPILE_FAILURE:
                if not isinstance(result, FailedEvidence) or result.stage != "compile":
                    raise ValueError("compile recovery starter did not produce its registered failure")
                self.observed = True
            return result
        return self.backend.compile(source, workdir, **options)

    def run(self, source, workdir, **options):
        recovery = self.policy.recovery
        if (
            recovery is not None
            and recovery.required_evidence is RecoveryEvidence.HOST_VERIFICATION_FAILURE
            and not self.observed
        ):
            if source != recovery.source:
                raise ValueError("registered runtime recovery starter must run first")
            result = self.backend.run(source, workdir, **options)
            if not isinstance(result, FailedEvidence) or result.stage not in {"execute", "verify"}:
                raise ValueError("runtime recovery starter did not produce its registered failure")
            self.observed = True
            return result
        return self.backend.run(source, workdir, **options)


def _policy_actor(actor, profile: A3ModelProfile, policy: ProjectRuntimePolicy):
    recovery = policy.recovery
    if recovery is None:
        return actor
    forced = [
        {"action": "write_source", "source": recovery.source},
        {"action": "compile"},
    ]
    if recovery.required_evidence is RecoveryEvidence.HOST_VERIFICATION_FAILURE:
        forced.append({"action": "run"})
    actions = iter(forced)

    def wrapped(selected_profile, context):
        try:
            action = next(actions)
        except StopIteration:
            return actor(selected_profile, context)
        return A3Completion(
            json.dumps(action, sort_keys=True, separators=(",", ":")),
            1,
            {"profile_sha256": profile.fingerprint},
        )
    return wrapped


class LiveComposition:
    """Execute every registered project for one isolated experiment cell."""

    def __init__(self, config: Phase1Config, dependencies: LiveDependencies) -> None:
        self.config, self.dependencies = config, dependencies
        self.cell_digests: dict[str, tuple[str, ...]] = {}

    def execute(self, cell: A3ExperimentCell, paths: CellPaths) -> dict[str, object]:
        store = AuthoritativeResultStore(paths.root / "authority.jsonl")
        memory = Phase1LearningJournal(
            paths.memory, DEFAULT_PROPOSALS, cell_id=cell.cell_id,
            lineage_id=f"phase1-{cell.cell_id}", evidence_resolver=store,
        )
        knowledge = self.dependencies.knowledge_factory(cell, paths)
        evidence = EvidenceLedger(paths.evidence)
        profile = load_a3_model_profile(cell.backend_model)
        digests = [entry["entry_sha256"] for entry in memory.read()]
        start = memory.resume_state().completed_projects
        for proposal in DEFAULT_PROPOSALS[start:]:
            policy = ProjectRuntimePolicy.from_proposal(proposal)
            candidate = _RecordingCandidate(
                _RecoveryCandidate(
                    self.dependencies.candidate_factory(cell, proposal, paths), policy
                ), store
            )
            profiler = _RecordingProfiler(
                self.dependencies.profiler_factory(
                    cell, proposal, paths, candidate
                ),
                store, proposal.project_id,
            )
            trial = A3TrialLoop(
                cell=cell, proposal=proposal, profile=profile,
                actor=_policy_actor(
                    self.dependencies.actor_factory(cell, proposal), profile, policy
                ),
                candidate=candidate, knowledge=knowledge, profiler=profiler,
                evidence=evidence, memory=memory,
                workdir=paths.workspace / proposal.project_id,
                timing_dimensions=(
                    policy.study_dimensions
                    if policy.performance_preset is PerformancePreset.TIMING else None
                ),
                execution_profile=self.dependencies.execution_profile,
            ).run()
            if trial.status != "passed" or trial.memory_entry_sha256 is None:
                failure = canonical_digest(
                    {"cell_id": cell.cell_id, "project_id": proposal.project_id,
                     "status": trial.status, "failures": trial.failures}
                )
                self.cell_digests[cell.cell_id] = tuple((*digests, failure))
                return {"status": "failed", "evidence_sha256": failure}
            digests.append(trial.memory_entry_sha256)
        self.cell_digests[cell.cell_id] = tuple(digests)
        return {"status": "passed", "evidence_sha256": canonical_digest(tuple(digests))}


def model_actor_factory(
    environ: Mapping[str, str], transport=None,
) -> Callable[[A3ExperimentCell, CurriculumProposal], Callable[[A3ModelProfile, str], A3Completion]]:
    """Build the real model actor without retaining or returning credentials."""
    environment = dict(environ)
    def factory(cell: A3ExperimentCell, _proposal: CurriculumProposal):
        def actor(_profile: A3ModelProfile, context: str) -> A3Completion:
            return complete_a3(
                _SYSTEM, context, model=cell.backend_model,
                environ=environment, transport=transport,
            )
        return actor
    return factory


class RemoteCandidateBundle:
    """Expose separate trial actions over one managed compile+verify operation."""

    def __init__(
        self, backend: GZA3RemoteCandidateBackend | BZA3RemoteCandidateBackend
    ) -> None:
        self.backend = backend
        self.planner = A3CandidateBackend(lambda *args, **kwargs: None)
        self.results: dict[str, VerifiedResult | FailedEvidence] = {}
        self.plans = {}
        self.compilations: dict[str, CandidateCompilation] = {}

    def _plan(self, source, options):
        selected = dict(options)
        requested_profile = selected.pop("execution_profile", None)
        execution_profile = getattr(self.backend, "profile", "gz-a3")
        if requested_profile is not None and requested_profile != execution_profile:
            raise ValueError("candidate execution profile does not match its backend")
        return self.planner.plan(
            source, execution_profile=execution_profile, **selected
        )

    def compile(self, source, workdir, **options):
        plan = self._plan(source, options)
        result = self.backend.compile(plan, workdir)
        if isinstance(result, CandidateCompilation):
            self.compilations[plan.execution_id] = result
            self.plans[plan.execution_id] = plan
        return result

    def run(self, source, workdir, **options):
        plan = self._plan(source, options)
        # Attempt ids intentionally differ between compile/run actions, while
        # source and host policy remain identical. Match the retained source.
        for execution_id, retained_plan in self.plans.items():
            if (
                retained_plan.source_fingerprint == plan.source_fingerprint
                and retained_plan.project_id == plan.project_id
                and retained_plan.input_a == plan.input_a
                and retained_plan.input_b == plan.input_b
            ):
                result = self.backend.execute(self.compilations[execution_id])
                self.results[execution_id] = result
                return result
        raise ValueError("successful compile is required before managed execution")


class _LazyGZProfiler:
    def __init__(
        self, *, config: Phase1Config, paths: CellPaths,
        candidate: _RecordingCandidate, physical_device: int,
    ) -> None:
        self.config, self.paths = config, paths
        self.candidate, self.physical_device = candidate, physical_device

    def profile(self, request):
        if request.treatment is ProfilingTreatment.OFF:
            return None
        remote = self.candidate.backend
        if isinstance(remote, _RecoveryCandidate):
            remote = remote.backend
        if not isinstance(remote, RemoteCandidateBundle):
            raise TypeError("GZ profiler requires the managed candidate backend")
        plan = remote.plans.get(request.binding.execution_id)
        if plan is None:
            raise ValueError("profiling candidate has no retained remote directory")
        backend = GZA3ProfilingBackend(
            validation_wrapper=str(self.config.validation_wrapper),
            remote_candidate_directory=remote.backend.remote_candidate_directory(plan).as_posix(),
            evidence_directory=self.paths.root / "profiles",
            physical_device=self.physical_device,
            verified_results=self.candidate.verified,
        )
        return backend.profile(request)

    def time(self, binding, dimensions):
        remote = self.candidate.backend
        if isinstance(remote, _RecoveryCandidate):
            remote = remote.backend
        if not isinstance(remote, RemoteCandidateBundle):
            raise TypeError("GZ timing requires the managed candidate backend")
        plan = remote.plans.get(binding.execution_id)
        if plan is None:
            raise ValueError("timing candidate has no retained remote directory")
        backend = GZA3ProfilingBackend(
            validation_wrapper=str(self.config.validation_wrapper),
            remote_candidate_directory=remote.backend.remote_candidate_directory(plan).as_posix(),
            evidence_directory=self.paths.root / "profiles",
            physical_device=self.physical_device,
            verified_results=self.candidate.verified,
        )
        return backend.time(binding, dimensions)


class _LazyBZProfiler:
    """Bind profiling to the exact candidate directory and selected BZ profile."""

    def __init__(
        self, *, config: Phase1Config, paths: CellPaths,
        candidate: _RecordingCandidate, profile: str, physical_device: int,
        cpl_remote: str,
    ) -> None:
        self.config, self.paths = config, paths
        self.candidate = candidate
        self.profile_name, self.physical_device = profile, physical_device
        self.cpl_remote = cpl_remote

    def _backend(self, execution_id: str) -> BZA3ProfilingBackend:
        remote = self.candidate.backend
        if isinstance(remote, _RecoveryCandidate):
            remote = remote.backend
        if not isinstance(remote, RemoteCandidateBundle) or not isinstance(
            remote.backend, BZA3RemoteCandidateBackend
        ):
            raise TypeError("BZ profiler requires the managed BZ candidate backend")
        plan = remote.plans.get(execution_id)
        if plan is None:
            raise ValueError("profiling candidate has no retained remote directory")
        return BZA3ProfilingBackend(
            validation_wrapper=str(self.config.validation_wrapper),
            cpl_remote=self.cpl_remote,
            profile=self.profile_name,
            remote_candidate_directory=(
                remote.backend.remote_candidate_directory(plan).as_posix()
            ),
            evidence_directory=self.paths.root / "profiles",
            physical_device=self.physical_device,
            verified_results=self.candidate.verified,
        )

    def profile(self, request):
        if request.treatment is ProfilingTreatment.OFF:
            return None
        selected = replace(request, execution_profile=self.profile_name)
        return self._backend(selected.binding.execution_id).profile(selected)

    def time(self, binding, dimensions):
        return self._backend(binding.execution_id).time(binding, dimensions)


def managed_live_dependencies(
    config: Phase1Config, *, client: str, server: str, remote: str,
    remote_workspace: str, physical_device: int,
    environ: Mapping[str, str], transport=None,
) -> LiveDependencies:
    """Build the concrete checked-wrapper/managed-transfer dependency set."""
    client_path = Path(client)
    if (
        not client_path.is_file() or not os.access(client_path, os.X_OK)
        or client_path.name != "remote_agent_client.sh"
    ):
        raise ValueError("remote client must be the checked executable remote_agent_client.sh")

    def candidate_factory(_cell, _proposal, _paths):
        return RemoteCandidateBundle(GZA3RemoteCandidateBackend(
            client=client, server=server,
            validation_wrapper=str(config.validation_wrapper), remote=remote,
            remote_workspace=remote_workspace, physical_device=physical_device,
            state_directory=_paths.root / "remote-candidate-state" / _proposal.project_id,
        ))

    def profiler_factory(_cell, _proposal, paths, candidate):
        return _LazyGZProfiler(
            config=config, paths=paths, candidate=candidate,
            physical_device=physical_device,
        )

    return LiveDependencies(
        actor_factory=model_actor_factory(environ, transport),
        candidate_factory=candidate_factory,
        knowledge_factory=local_knowledge_factory(config),
        profiler_factory=profiler_factory,
        execution_profile="gz-a3",
    )


def bz_live_dependencies(
    config: Phase1Config, *, cpl_remote: str, profile: str,
    remote_workspace: str, physical_device: int,
    environ: Mapping[str, str], transport=None,
) -> LiveDependencies:
    """Compose the user-wide transport, BZ candidate, and raw profiler."""
    remote_path = Path(cpl_remote)
    if (
        not remote_path.is_file()
        or not os.access(remote_path, os.X_OK)
        or remote_path.name != "cpl-remote"
    ):
        raise ValueError("cpl_remote must be the user-wide executable cpl-remote")
    if profile not in {"bz-a3-1", "bz-a3-2"}:
        raise ValueError("profile must be bz-a3-1 or bz-a3-2")

    def candidate_factory(_cell, proposal, paths):
        return RemoteCandidateBundle(BZA3RemoteCandidateBackend(
            cpl_remote=cpl_remote,
            validation_wrapper=str(config.validation_wrapper),
            profile=profile,
            remote_workspace=remote_workspace,
            physical_device=physical_device,
            state_directory=(
                paths.root / "remote-candidate-state" / proposal.project_id
            ),
        ))

    def profiler_factory(_cell, _proposal, paths, candidate):
        return _LazyBZProfiler(
            config=config, paths=paths, candidate=candidate,
            profile=profile, physical_device=physical_device,
            cpl_remote=cpl_remote,
        )

    return LiveDependencies(
        actor_factory=model_actor_factory(environ, transport),
        candidate_factory=candidate_factory,
        knowledge_factory=local_knowledge_factory(config),
        profiler_factory=profiler_factory,
        execution_profile=profile,
    )


def local_knowledge_factory(
    config: Phase1Config,
) -> Callable[[A3ExperimentCell, CellPaths], KnowledgeAgent]:
    """Open only the pinned local KDB and only for the enabled treatment."""
    def factory(cell: A3ExperimentCell, paths: CellPaths) -> KnowledgeAgent:
        from benchmarks.a3_experiments import KnowledgeMode

        journal = QueryJournal(paths.root / "knowledge-queries.jsonl")
        if cell.knowledge is KnowledgeMode.WITHOUT_KDB:
            return KnowledgeAgent(enabled=False, journal=journal)
        expected = CollectionManifest.from_json(
            config.knowledge_manifest.read_text(encoding="utf-8")
        )
        embeddings = PinnedBGEEmbeddings(cache_dir=config.embedding_cache)
        database = KnowledgeDB.open_read_only(config.knowledge_database, embeddings)
        actual = database.manifest(expected.collection)
        if actual != expected:
            database.connection.close()
            raise ValueError("local KDB does not match the pinned knowledge manifest")
        return KnowledgeAgent(
            enabled=True, journal=journal, database=database,
            collection=expected.collection,
        )
    return factory


__all__ = [
    "AuthoritativeResultStore", "LiveComposition", "LiveDependencies", "RemoteCandidateBundle",
    "bz_live_dependencies", "local_knowledge_factory", "managed_live_dependencies",
    "model_actor_factory",
]
