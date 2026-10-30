import json
from dataclasses import dataclass
import fcntl
from pathlib import PurePosixPath

import pytest

from benchmarks.a3_experiments import KnowledgeMode, ProfilingGuidance
from benchmarks.a3_experiments import BackendModel
from benchmarks.a3_model_profiles import (
    A3Completion, A3TransportResult, DEEPSEEK_BASE_URL, load_a3_model_profile,
)
from benchmarks.a3kernels.candidate import (
    A3CandidateBackend, CANDIDATE_SOURCE_CONTRACT, CandidateCompilation,
)
from benchmarks.a3kernels.live_composition import (
    AuthoritativeResultStore,
    LiveComposition,
    LiveDependencies,
    RemoteCandidateBundle,
    local_knowledge_factory,
    model_actor_factory,
)
from benchmarks.a3kernels.knowledge_agent import (
    KnowledgeAgent,
    KnowledgeQuery,
    QueryJournal,
)
from benchmarks.a3kernels.phase1_evidence import EvidenceLedger, canonical_digest
from benchmarks.a3kernels.phase1_protocol import (
    ExecutionReceipt, FailedEvidence, VerifiedResult, attest,
)
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a3kernels.project_execution import ProjectRuntimePolicy, RecoveryEvidence
from benchmarks.a3kernels.phase1_wave import (
    SMOKE_PROPOSALS, Phase1Config,
    Phase1Wave, SmokeWave,
    foundation_cells,
)
from benchmarks.a3kernels.profiling import (
    CandidateBinding, CompactProfileResult, ProfileMetric, ProfileRequest,
    StudyDimensions,
    TimingResult,
)
from benchmarks.a3kernels.trial import TrialResult


SOURCE = '''extern "C" __global__ __aicore__ void vector_add(
    GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output,
    uint32_t count, uint32_t buffer_bytes) {}
'''


def config(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    wrapper = tmp_path / "catlass-validation.sh"
    wrapper.write_text("#!/bin/sh\n"); wrapper.chmod(0o755)
    embedding = tmp_path / "embedding"; embedding.mkdir()
    corpus = tmp_path / "corpus"; corpus.mkdir()
    database = tmp_path / "kdb.sqlite"; database.write_bytes(b"db")
    manifest = tmp_path / "manifest.json"; manifest.write_text("{}")
    return Phase1Config(tmp_path / "state", wrapper, embedding, corpus, database, manifest)


class FakeCandidate:
    def __init__(self, proposal=None):
        self.base = A3CandidateBackend(lambda *a, **k: None)
        self.policy = ProjectRuntimePolicy.from_proposal(proposal) if proposal else None

    def compile(self, source, workdir, **options):
        plan = self.base.plan(source, **options)
        if (
            self.policy and self.policy.recovery
            and self.policy.recovery.required_evidence is RecoveryEvidence.COMPILE_FAILURE
            and source == self.policy.recovery.source
        ):
            return FailedEvidence.create(
                plan, stage="compile", error_type="CompileError", detail="registered fault"
            )
        body = {"plan": plan, "library_sha256": "1" * 64, "stdout": "", "stderr": ""}
        return CandidateCompilation(plan, "1" * 64, "", "", attest(body))

    def run(self, source, workdir, **options):
        plan = self.base.plan(source, **options)
        if (
            self.policy and self.policy.recovery
            and self.policy.recovery.required_evidence is RecoveryEvidence.HOST_VERIFICATION_FAILURE
            and source == self.policy.recovery.source
        ):
            return FailedEvidence.create(
                plan, stage="verify", error_type="Mismatch", detail="registered fault"
            )
        output = tuple(a + b for a, b in zip(plan.input_a, plan.input_b))
        return VerifiedResult.from_receipt(
            plan, ExecutionReceipt(0, output, job_handle=f"gz-a3:{plan.project_id[:12]}"),
            max_abs_error=0.0,
        )


class FakeKnowledge:
    """CPU embedding fixture backed by the real read-only A3 SQLite agent."""
    def __init__(self, enabled, calls):
        from tempfile import TemporaryDirectory
        from canonical_a3_fixtures import a3_knowledge
        self._directory = TemporaryDirectory(prefix='a3-experience-')
        self.agent = a3_knowledge(self._directory.name, enabled=enabled, collection='experience')
        self.enabled, self.calls = enabled, calls
        self.manifest = self.agent.manifest
        self.collection = self.agent.collection

    def query(self, query):
        self.calls.append(query.query)
        return self.agent.query(query)


class FakeProfiler:
    def __init__(self, enabled, calls): self.enabled, self.calls = enabled, calls
    def profile(self, request):
        if not self.enabled: return None
        self.calls.append(request.request_id)
        return CompactProfileResult.create(
            request, exported_kernels=("vector_add",), metric_values=(("pipe", 1.0),),
            timeline=(("kernel", 1.0),), report_sha256="2" * 64,
        )
    def time(self, binding, dimensions):
        self.calls.append("timing")
        return TimingResult.from_samples(binding, dimensions, (1.0, 2.0))


def actor_factory(cell, proposal):
    from canonical_a3_fixtures import A3Session
    def actor(*_):
        pytest.fail('LiveComposition entered the former actor action loop')
    actor.session = actor.canonical_session = A3Session(initial=SOURCE, repaired=SOURCE + '\n// repaired')
    return actor


def dependencies(knowledge_calls, profile_calls):
    return LiveDependencies(
        actor_factory=actor_factory,
        candidate_factory=lambda cell, proposal, paths: FakeCandidate(proposal),
        knowledge_factory=lambda cell, paths: FakeKnowledge(
            cell.knowledge is KnowledgeMode.WITH_KDB, knowledge_calls
        ),
        profiler_factory=lambda cell, proposal, paths, verified: FakeProfiler(
            cell.profiling is ProfilingGuidance.WITH_GUIDANCE, profile_calls
        ),
    )


@pytest.fixture(autouse=True)
def canonical_live_trials(monkeypatch, tmp_path):
    from canonical_a3_fixtures import install_live_trials
    return install_live_trials(monkeypatch, tmp_path / 'canonical-fixture')


def trial_protocol_sha256(
    openai_turns=24, *, mismatch_feedback=True, attempt_completion=True,
    prepare_runtime_infrastructure=True,
):
    from coding.pi import ENGINE_IDENTITY
    value = {
        **ENGINE_IDENTITY,
        "schema": "a3-trial-protocol-v6",
        "candidate_source_contract": CANDIDATE_SOURCE_CONTRACT.as_dict(),
        "infrastructure_retry": {
            "max_retries": 3, "backoff_seconds": 120,
            "retryable_status": "infrastructure-unverified",
            "journal_schema": "a3-infrastructure-retry-v1",
            "attempt_outcome_required_before_decision": True,
            "ambiguous_candidate_outcome": "infrastructure-unverified",
            "prepare_runtime_outcome": "infrastructure-unverified",
            "interrupted_started_outcome": "terminal-infrastructure-unverified",
            "cell_execution": "exclusive-nonblocking",
            "memory_commit_reconciliation": "authenticated-passed-outcome",
        },
        "verification_feedback_schema": "a3-verification-mismatch-v1",
        "candidate_validation_exception": "rewrite-required",
        "recovery_starter_ambiguous_outcome": "infrastructure-unverified",
        "attempt_completion": {
            "schema": "a3-attempt-completion-v1",
            "source_frozen": True,
            "allowed_operations": ["compile", "run", "profile", "submit"],
            "call_limits": {
                ProfilingGuidance.WITHOUT_GUIDANCE.value: 3,
                ProfilingGuidance.WITH_GUIDANCE.value: 4,
            },
            "token_ceiling": "absolute",
            "rewrite_required": "terminate",
            "rewrite_terminal_status": "attempt-rewrite-required",
        },
        "model_budgets": {
            BackendModel.GPT_5_6_SOL.value: {
                "max_turns": openai_turns, "max_tokens": 32768,
            },
            BackendModel.DEEPSEEK_FLASH.value: {
                "max_turns": 24, "max_tokens": 65536,
            },
        },
    }
    if not mismatch_feedback:
        del value["verification_feedback_schema"]
    if not prepare_runtime_infrastructure:
        del value["infrastructure_retry"]["prepare_runtime_outcome"]
    if not attempt_completion:
        value["schema"] = "a3-trial-protocol-v1"
        del value["candidate_source_contract"]
        del value["attempt_completion"]
        del value["candidate_validation_exception"]
        del value["recovery_starter_ambiguous_outcome"]
        del value["infrastructure_retry"]
    return canonical_digest(value)


def test_complete_eight_cell_local_proof_is_isolated_and_resumable(tmp_path):
    cfg = config(tmp_path); knowledge_calls, profile_calls = [], []
    composition = LiveComposition(cfg, dependencies(knowledge_calls, profile_calls))
    wave = Phase1Wave(cfg, composition.execute)
    records = wave.run()
    assert len(records) == 8 and {row["status"] for row in records} == {"passed"}
    assert len(knowledge_calls) == 8 * len(DEFAULT_PROPOSALS)
    assert profile_calls.count("timing") == 8 * 3
    assert len([value for value in profile_calls if value != "timing"]) == 4 * len(DEFAULT_PROPOSALS)
    cells = __import__("benchmarks.a3kernels.phase1_wave", fromlist=["foundation_cells"]).foundation_cells()
    assert all(len(json.loads(wave.paths(cell).memory.read_text().splitlines()[0])["entry_sha256"]) == 64 for cell in cells)
    for cell in cells:
        memories = [json.loads(line)["memory"] for line in wave.paths(cell).memory.read_text().splitlines()]
        assert sum(
            fact["category"] == "profiling" and fact["statement"] == "host timing samples captured"
            for memory in memories for fact in memory["host_facts"]
        ) == 3
        failures = [
            json.loads(line)["payload"].get("failed_evidence", {})
            for line in wave.paths(cell).evidence.read_text().splitlines()
            if json.loads(line)["kind"] == "failure"
        ]
        assert {item.get("stage") for item in failures} >= {"compile", "verify"}
    assert Phase1Wave(cfg, composition.execute).resume() == records


@pytest.mark.parametrize(
    "family,stage,error_type",
    [
        ("runtime-recovery", "compile", "AmbiguousRemoteOutcome"),
        ("compile-recovery", "prepare", "RuntimeError"),
    ],
)
def test_recovery_starter_launch_loss_terminates_without_rewrite_loop(
    tmp_path, family, stage, error_type, canonical_live_trials,
):
    proposal = next(
        item for item in DEFAULT_PROPOSALS
        if item.family.value == family
    )
    cell = next(
        item for item in foundation_cells()
        if item.backend_model is BackendModel.GPT_5_6_SOL
        and item.knowledge is KnowledgeMode.WITHOUT_KDB
        and item.profiling is ProfilingGuidance.WITHOUT_GUIDANCE
    )

    class InfrastructureStarter(FakeCandidate):
        def __init__(self):
            super().__init__(proposal)
            self.compile_calls = 0

        def compile(self, source, workdir, **options):
            self.compile_calls += 1
            plan = self.base.plan(source, **options)
            return FailedEvidence.create(
                plan, stage=stage, error_type=error_type,
                detail="retained operation completed without an observable result",
            )

    candidate = InfrastructureStarter()
    deps = LiveDependencies(
        actor_factory=actor_factory,
        candidate_factory=lambda *_: candidate,
        knowledge_factory=lambda *_: FakeKnowledge(False, []),
        profiler_factory=lambda *_: FakeProfiler(False, []),
        execution_profile="bz-a3-1",
    )
    cfg = config(tmp_path)
    sleeps = []
    composition = LiveComposition(
        cfg, deps, proposals=(proposal,), sleeper=sleeps.append,
    )
    wave = Phase1Wave(
        cfg, composition.execute, cells=(cell,), execution_profile="bz-a3-1",
        proposals=(proposal,),
    )
    record = wave.run()[0]

    assert record["status"] == "failed"
    assert record["terminal_reason"] == "infrastructure-unverified"
    assert record["completed_projects"] == 0
    expected_attempts = 1 if error_type == "AmbiguousRemoteOutcome" else 4
    assert candidate.compile_calls == expected_attempts
    assert sleeps == [120] * (expected_attempts - 1)
    assert record["infrastructure_retries_used"] == expected_attempts - 1
    assert canonical_live_trials and all(not any(
        context["stage"] == "debug" for context in session.contexts)
        for session in canonical_live_trials)
    reloaded = Phase1Wave(
        cfg,
        lambda *_: pytest.fail("persisted failure replayed the recovery starter"),
        cells=(cell,), execution_profile="bz-a3-1", proposals=(proposal,),
    )
    assert reloaded.resume() == (record,)
    assert reloaded.report() == {
            "schema": "a3-phase1-foundation-report-v2",
        "counts": {"failed": 1},
        "records": [record],
    }
    paths = wave.paths(cell)
    failures = [
        json.loads(line)["payload"].get("failed_evidence")
        for line in paths.evidence.read_text().splitlines()
        if json.loads(line)["kind"] == "failure"
    ]
    assert [(failure["stage"], failure["error_type"]) for failure in failures] == [
        (stage, error_type)
    ] * expected_attempts


def test_runtime_recovery_starter_run_loss_stops_before_model_repair(tmp_path, canonical_live_trials):
    proposal = next(
        item for item in DEFAULT_PROPOSALS
        if item.family.value == "runtime-recovery"
    )
    cell = next(
        item for item in foundation_cells()
        if item.backend_model is BackendModel.GPT_5_6_SOL
        and item.knowledge is KnowledgeMode.WITHOUT_KDB
        and item.profiling is ProfilingGuidance.WITHOUT_GUIDANCE
    )

    class AmbiguousStarterRun(FakeCandidate):
        def __init__(self):
            super().__init__(proposal)
            self.compile_calls = self.run_calls = 0

        def compile(self, source, workdir, **options):
            self.compile_calls += 1
            return super().compile(source, workdir, **options)

        def run(self, source, workdir, **options):
            self.run_calls += 1
            plan = self.base.plan(source, **options)
            return FailedEvidence.create(
                plan, stage="execute", error_type="AmbiguousRemoteOutcome",
                detail="retained runtime operation has no observable result",
            )

    candidate = AmbiguousStarterRun()
    deps = LiveDependencies(
        actor_factory=actor_factory,
        candidate_factory=lambda *_: candidate,
        knowledge_factory=lambda *_: FakeKnowledge(False, []),
        profiler_factory=lambda *_: FakeProfiler(False, []),
        execution_profile="bz-a3-1",
    )
    cfg = config(tmp_path)
    sleeps = []
    composition = LiveComposition(
        cfg, deps, proposals=(proposal,), sleeper=sleeps.append,
    )
    paths = Phase1Wave(
        cfg, composition.execute, cells=(cell,), execution_profile="bz-a3-1",
        proposals=(proposal,),
    ).paths(cell)

    result = composition.execute(cell, paths)

    assert result["terminal_reason"] == "infrastructure-unverified"
    assert candidate.run_calls == candidate.compile_calls == 1
    assert sleeps == [] and result["infrastructure_retries_used"] == 0
    assert canonical_live_trials and all(not any(
        context["stage"] == "debug" for context in session.contexts)
        for session in canonical_live_trials)
    failures = [
        json.loads(line)["payload"].get("failed_evidence")
        for line in paths.evidence.read_text().splitlines()
        if json.loads(line)["kind"] == "failure"
    ]
    assert [(failure["stage"], failure["error_type"]) for failure in failures] == [
        ("execute", "AmbiguousRemoteOutcome")
    ]


@pytest.mark.parametrize("operation", ["compile", "run"])
def test_ordinary_ambiguous_candidate_evidence_uses_infrastructure_retries(
    tmp_path, operation,
):
    proposal = DEFAULT_PROPOSALS[0]
    cell = next(
        item for item in foundation_cells()
        if item.backend_model is BackendModel.GPT_5_6_SOL
        and item.knowledge is KnowledgeMode.WITHOUT_KDB
        and item.profiling is ProfilingGuidance.WITHOUT_GUIDANCE
    )
    attempts, sleeps = [], []

    class AmbiguousCandidate(FakeCandidate):
        def _ambiguous(self, source, options, stage):
            attempts.append(options["attempt_id"])
            plan = self.base.plan(source, **options)
            return FailedEvidence.create(
                plan, stage=stage, error_type="AmbiguousRemoteOutcome",
                detail="ordinary candidate result unavailable",
            )

        def compile(self, source, workdir, **options):
            if operation == "compile":
                return self._ambiguous(source, options, "compile")
            return super().compile(source, workdir, **options)

        def run(self, source, workdir, **options):
            if operation == "run":
                return self._ambiguous(source, options, "execute")
            return super().run(source, workdir, **options)

    deps = LiveDependencies(
        actor_factory=actor_factory,
        candidate_factory=lambda *args: AmbiguousCandidate(proposal),
        knowledge_factory=lambda *_: FakeKnowledge(False, []),
        profiler_factory=lambda *_: FakeProfiler(False, []),
        execution_profile="bz-a3-1",
    )
    cfg = config(tmp_path)
    paths = Phase1Wave(
        cfg, lambda *_: {}, cells=(cell,), execution_profile="bz-a3-1",
        proposals=(proposal,),
    ).paths(cell)

    result = LiveComposition(
        cfg, deps, proposals=(proposal,), sleeper=sleeps.append,
    ).execute(cell, paths)

    assert result["terminal_reason"] == "infrastructure-unverified"
    assert result["infrastructure_retries_used"] == 0
    assert sleeps == [] and len(attempts) == 1
    failures = [
        entry.payload["failed_evidence"]
        for entry in EvidenceLedger(paths.evidence).entries
        if "failed_evidence" in entry.payload
    ]
    assert all(item["error_type"] == "AmbiguousRemoteOutcome" for item in failures)


def test_prepare_runtime_failure_uses_all_infrastructure_retries(tmp_path, canonical_live_trials):
    proposal = DEFAULT_PROPOSALS[0]
    cell = next(
        item for item in foundation_cells()
        if item.backend_model is BackendModel.GPT_5_6_SOL
        and item.knowledge is KnowledgeMode.WITHOUT_KDB
        and item.profiling is ProfilingGuidance.WITHOUT_GUIDANCE
    )
    attempts, sleeps = [], []

    class PrepareFailure(FakeCandidate):
        def compile(self, source, workdir, **options):
            attempts.append(options["attempt_id"])
            plan = self.base.plan(source, **options)
            return FailedEvidence.create(
                plan, stage="prepare", error_type="RuntimeError",
                detail="VPN route unavailable",
            )

    deps = LiveDependencies(
        actor_factory=actor_factory,
        candidate_factory=lambda *_: PrepareFailure(proposal),
        knowledge_factory=lambda *_: FakeKnowledge(False, []),
        profiler_factory=lambda *_: FakeProfiler(False, []),
        execution_profile="bz-a3-1",
    )
    cfg = config(tmp_path)
    paths = Phase1Wave(
        cfg, lambda *_: {}, cells=(cell,), execution_profile="bz-a3-1",
        proposals=(proposal,),
    ).paths(cell)

    result = LiveComposition(
        cfg, deps, proposals=(proposal,), sleeper=sleeps.append,
    ).execute(cell, paths)

    assert result["terminal_reason"] == "infrastructure-unverified"
    assert result["infrastructure_retries_used"] == 3
    assert sleeps == [120, 120, 120]
    assert len(attempts) == 4
    assert len(canonical_live_trials) == 4
    assert all(session.contexts for session in canonical_live_trials)
    assert all(not any(context["stage"] == "debug" for context in session.contexts)
               for session in canonical_live_trials)


def test_cell_execution_lock_rejects_concurrent_same_process_run(tmp_path):
    proposal = DEFAULT_PROPOSALS[0]
    cell = foundation_cells()[0]
    cfg = config(tmp_path)
    paths = Phase1Wave(
        cfg, lambda *_: {}, cells=(cell,), execution_profile="bz-a3-1",
        proposals=(proposal,),
    ).paths(cell)
    paths.root.mkdir(parents=True)
    lock_path = paths.root / "execution.lock"
    calls = []
    forbidden = lambda *_: calls.append("called") or pytest.fail(
        "locked cell reached a live dependency"
    )
    deps = LiveDependencies(
        actor_factory=forbidden, candidate_factory=forbidden,
        knowledge_factory=forbidden, profiler_factory=forbidden,
        execution_profile="bz-a3-1",
    )

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="cell execution already active"):
            LiveComposition(cfg, deps, proposals=(proposal,)).execute(cell, paths)

    assert calls == []


def test_committed_memory_repairs_missing_passed_attempt_outcome(
    tmp_path, monkeypatch,
):
    proposal = DEFAULT_PROPOSALS[0]
    cell = next(
        item for item in foundation_cells()
        if item.backend_model is BackendModel.GPT_5_6_SOL
        and item.knowledge is KnowledgeMode.WITHOUT_KDB
        and item.profiling is ProfilingGuidance.WITHOUT_GUIDANCE
    )
    cfg = config(tmp_path)
    paths = Phase1Wave(
        cfg, lambda *_: {}, cells=(cell,), execution_profile="bz-a3-1",
        proposals=(proposal,),
    ).paths(cell)
    original = LiveComposition._append_attempt_outcome.__func__

    def crash_after_memory(cls, evidence, **kwargs):
        if kwargs["trial"].status == "passed":
            raise KeyboardInterrupt()
        return original(cls, evidence, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            LiveComposition, "_append_attempt_outcome",
            classmethod(crash_after_memory),
        )
        with pytest.raises(KeyboardInterrupt):
            LiveComposition(
                cfg, dependencies([], []), proposals=(proposal,),
            ).execute(cell, paths)

    memory_entry = json.loads(paths.memory.read_text().splitlines()[0])
    calls = []
    forbidden = lambda *_: calls.append("called") or pytest.fail(
        "committed project was re-executed"
    )
    deps = LiveDependencies(
        actor_factory=forbidden, candidate_factory=forbidden,
        knowledge_factory=forbidden, profiler_factory=forbidden,
        execution_profile="gz-a3",
    )

    result = LiveComposition(cfg, deps, proposals=(proposal,)).execute(cell, paths)

    assert result["status"] == "passed"
    assert calls == []
    retry = LiveComposition._retry_entries(EvidenceLedger(paths.evidence))[-1]
    outcome = retry.payload["infrastructure_retry"]
    assert outcome["event"] == "outcome"
    assert outcome["trial_status"] == "passed"
    assert outcome["memory_entry_sha256"] == memory_entry["entry_sha256"]


@pytest.mark.parametrize("failures_before_success", [1, 2, 3])
def test_infrastructure_retry_succeeds_at_each_boundary_with_fresh_dependencies(
    tmp_path, failures_before_success, canonical_live_trials,
):
    proposal = DEFAULT_PROPOSALS[2]
    cell = next(
        item for item in foundation_cells()
        if item.backend_model is BackendModel.GPT_5_6_SOL
        and item.knowledge is KnowledgeMode.WITHOUT_KDB
        and item.profiling is ProfilingGuidance.WITHOUT_GUIDANCE
    )
    factory_calls, attempted_ids, profiler_calls, actor_calls, sleeps = [], [], [], [], []

    class AttemptCandidate(FakeCandidate):
        def __init__(self, ordinal):
            super().__init__(proposal)
            self.ordinal = ordinal

        def compile(self, source, workdir, **options):
            attempted_ids.append(options["attempt_id"])
            if self.ordinal < failures_before_success:
                plan = self.base.plan(source, **options)
                return FailedEvidence.create(
                    plan, stage="prepare", error_type="RuntimeError",
                    detail="route unavailable before any launch",
                )
            return super().compile(source, workdir, **options)

    def candidate_factory(*_args):
        ordinal = len(factory_calls)
        factory_calls.append(ordinal)
        return AttemptCandidate(ordinal)

    def retry_actor_factory(selected, selected_proposal):
        actor_calls.append(len(actor_calls))
        return actor_factory(selected, selected_proposal)

    def profiler_factory(cell, proposal, paths, candidate):
        profiler_calls.append(len(profiler_calls))
        return FakeProfiler(False, [])

    deps = LiveDependencies(
        actor_factory=retry_actor_factory,
        candidate_factory=candidate_factory,
        knowledge_factory=lambda *_: FakeKnowledge(False, []),
        profiler_factory=profiler_factory,
        execution_profile="bz-a3-1",
    )
    cfg = config(tmp_path)
    composition = LiveComposition(
        cfg, deps, proposals=(proposal,), sleeper=sleeps.append,
    )
    paths = Phase1Wave(
        cfg, composition.execute, cells=(cell,), execution_profile="bz-a3-1",
        proposals=(proposal,),
    ).paths(cell)

    result = composition.execute(cell, paths)

    assert result["status"] == "passed"
    assert result["infrastructure_retries_used"] == failures_before_success
    assert sleeps == [120] * failures_before_success
    assert factory_calls == list(range(failures_before_success + 1))
    assert profiler_calls == actor_calls == factory_calls
    assert [session.owner.infrastructure_attempt_ordinal for session in canonical_live_trials] == factory_calls
    # Actual canonical operations replace the old textual infra/turn IDs.
    assert len(attempted_ids) == len(set(attempted_ids)) == failures_before_success + 2
    assert all(attempted_ids)
    assert len(paths.memory.read_text().splitlines()) == 1


def test_interrupted_started_attempt_fails_closed_without_new_identity(tmp_path):
    proposal = DEFAULT_PROPOSALS[0]
    cell = next(
        item for item in foundation_cells()
        if item.backend_model is BackendModel.GPT_5_6_SOL
        and item.knowledge is KnowledgeMode.WITHOUT_KDB
        and item.profiling is ProfilingGuidance.WITHOUT_GUIDANCE
    )
    cfg = config(tmp_path)
    paths = Phase1Wave(
        cfg, lambda *_: {}, cells=(cell,), execution_profile="bz-a3-1",
        proposals=(proposal,),
    ).paths(cell)
    evidence = EvidenceLedger(paths.evidence)
    LiveComposition._append_retry_event(
        evidence, cell=cell, proposal=proposal, execution_profile="bz-a3-1",
        event="started", ordinal=0,
    )
    calls, sleeps = [], []
    forbidden = lambda *_: calls.append("called") or pytest.fail(
        "interrupted started attempt launched a dependency"
    )
    deps = LiveDependencies(
        actor_factory=forbidden, candidate_factory=forbidden,
        knowledge_factory=forbidden, profiler_factory=forbidden,
        execution_profile="bz-a3-1",
    )
    result = LiveComposition(
        cfg, deps, proposals=(proposal,), sleeper=sleeps.append,
    ).execute(cell, paths)

    assert result["terminal_reason"] == "infrastructure-unverified"
    assert result["infrastructure_retries_used"] == 0
    assert calls == sleeps == []
    retry_events = [
        entry.payload["infrastructure_retry"]["event"]
        for entry in EvidenceLedger(paths.evidence).entries
        if "infrastructure_retry" in entry.payload
    ]
    assert retry_events == ["started", "outcome"]


def test_retained_noninfrastructure_outcome_stops_without_retry(tmp_path):
    proposal = DEFAULT_PROPOSALS[0]
    cell = next(
        item for item in foundation_cells()
        if item.backend_model is BackendModel.GPT_5_6_SOL
        and item.knowledge is KnowledgeMode.WITHOUT_KDB
        and item.profiling is ProfilingGuidance.WITHOUT_GUIDANCE
    )
    cfg = config(tmp_path)
    paths = Phase1Wave(
        cfg, lambda *_: {}, cells=(cell,), execution_profile="bz-a3-1",
        proposals=(proposal,),
    ).paths(cell)
    evidence = EvidenceLedger(paths.evidence)
    LiveComposition._append_retry_event(
        evidence, cell=cell, proposal=proposal, execution_profile="bz-a3-1",
        event="started", ordinal=0,
    )
    LiveComposition._append_attempt_outcome(
        evidence, cell=cell, proposal=proposal, execution_profile="bz-a3-1",
        ordinal=0, trial=TrialResult(
            "budget-exhausted", 24, 32768, None, None, 0,
            ("token budget exhausted",),
        ),
    )
    calls, sleeps = [], []
    forbidden = lambda *_: calls.append("called") or pytest.fail(
        "retained non-infrastructure outcome launched a dependency"
    )
    deps = LiveDependencies(
        actor_factory=forbidden, candidate_factory=forbidden,
        knowledge_factory=lambda *_: FakeKnowledge(False, []),
        profiler_factory=forbidden, execution_profile="bz-a3-1",
    )

    result = LiveComposition(
        cfg, deps, proposals=(proposal,), sleeper=sleeps.append,
    ).execute(cell, paths)

    assert result["terminal_reason"] == "token-budget-exhausted"
    assert result["infrastructure_retries_used"] == 0
    assert calls == sleeps == []


def test_retained_infrastructure_outcome_schedules_next_attempt(tmp_path, canonical_live_trials):
    proposal = DEFAULT_PROPOSALS[0]
    cell = next(
        item for item in foundation_cells()
        if item.backend_model is BackendModel.GPT_5_6_SOL
        and item.knowledge is KnowledgeMode.WITHOUT_KDB
        and item.profiling is ProfilingGuidance.WITHOUT_GUIDANCE
    )
    cfg = config(tmp_path)
    paths = Phase1Wave(
        cfg, lambda *_: {}, cells=(cell,), execution_profile="bz-a3-1",
        proposals=(proposal,),
    ).paths(cell)
    evidence = EvidenceLedger(paths.evidence)
    LiveComposition._append_retry_event(
        evidence, cell=cell, proposal=proposal, execution_profile="bz-a3-1",
        event="started", ordinal=0,
    )
    LiveComposition._append_attempt_outcome(
        evidence, cell=cell, proposal=proposal, execution_profile="bz-a3-1",
        ordinal=0, trial=TrialResult(
            "infrastructure-unverified", 0, 0, None, None, 0,
            ("route unavailable before any actor/candidate dispatch",),
        ),
    )
    attempt_ids, sleeps = [], []

    class RecordingCandidate(FakeCandidate):
        def compile(self, source, workdir, **options):
            attempt_ids.append(options["attempt_id"])
            return super().compile(source, workdir, **options)

    deps = LiveDependencies(
        actor_factory=actor_factory,
        candidate_factory=lambda *args: RecordingCandidate(proposal),
        knowledge_factory=lambda *_: FakeKnowledge(False, []),
        profiler_factory=lambda *_: FakeProfiler(False, []),
        execution_profile="bz-a3-1",
    )

    result = LiveComposition(
        cfg, deps, proposals=(proposal,), sleeper=sleeps.append,
    ).execute(cell, paths)

    assert result["status"] == "passed"
    assert result["infrastructure_retries_used"] == 1
    assert sleeps == [120]
    assert attempt_ids and len(attempt_ids) == len(set(attempt_ids))
    assert all(session.owner.infrastructure_attempt_ordinal == 1 for session in canonical_live_trials)


def test_pending_retained_handle_does_not_consume_infrastructure_retry(tmp_path, canonical_live_trials):
    proposal = DEFAULT_PROPOSALS[0]
    cell = next(
        item for item in foundation_cells()
        if item.backend_model is BackendModel.GPT_5_6_SOL
        and item.knowledge is KnowledgeMode.WITHOUT_KDB
        and item.profiling is ProfilingGuidance.WITHOUT_GUIDANCE
    )
    attempts, sleeps = [], []

    class UnobservablePending(FakeCandidate):
        def compile(self, source, workdir, **options):
            attempts.append(options["attempt_id"])
            if len(attempts) > 1:
                pytest.fail('unobserved original compile was replayed')
            plan = self.base.plan(source, **options)
            return FailedEvidence.create(plan, stage="prepare", error_type="TransferPending",
                detail="no terminal observation is available for the original transfer")


    deps = LiveDependencies(
        actor_factory=actor_factory,
        candidate_factory=lambda *args: UnobservablePending(proposal),
        knowledge_factory=lambda *_: FakeKnowledge(False, []),
        profiler_factory=lambda *_: FakeProfiler(False, []),
        execution_profile="bz-a3-1",
    )
    cfg = config(tmp_path)
    paths = Phase1Wave(
        cfg, lambda *_: {}, cells=(cell,), execution_profile="bz-a3-1",
        proposals=(proposal,),
    ).paths(cell)
    composition = LiveComposition(cfg, deps, proposals=(proposal,), sleeper=sleeps.append)
    result = composition.execute(cell, paths)
    assert result['status'] == 'failed' and result['terminal_reason'] == 'infrastructure-unverified'
    assert result['infrastructure_retries_used'] == 0 and sleeps == []
    assert len(attempts) == 1
    session, = canonical_live_trials
    owner = session.owner
    pending = dict(owner.canonical.build(owner).workflow.pending)
    assert pending['id'] == attempts[0]
    retained = paths.evidence.read_bytes()
    assert composition.execute(cell, paths) == result
    assert attempts == [pending['id']] and paths.evidence.read_bytes() == retained
    assert owner.canonical.build(owner).workflow.pending == pending


def test_smoke_composition_executes_exactly_one_baseline_project(tmp_path):
    cfg = config(tmp_path)
    cell = next(
        item for item in foundation_cells()
        if item.knowledge is KnowledgeMode.WITHOUT_KDB
        and item.profiling is ProfilingGuidance.WITHOUT_GUIDANCE
    )
    composition = LiveComposition(
        cfg, dependencies([], []), proposals=SMOKE_PROPOSALS,
        lineage_prefix="smoke",
    )
    paths = Phase1Wave(cfg, composition.execute, cells=(cell,)).paths(cell)
    assert composition.execute(cell, paths)["status"] == "passed"
    entries = [json.loads(line) for line in paths.memory.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["memory"]["proposal"]["family"] == "vector-add-baseline"
    assert entries[0]["memory"]["lineage_id"].startswith("smoke-")


@pytest.mark.parametrize("smoke", [False, True], ids=["full", "smoke"])
@pytest.mark.parametrize(
    "old_protocol",
    [
        "twelve-turns", "no-mismatch-feedback", "no-attempt-completion",
        "no-prepare-runtime-infrastructure",
    ],
)
def test_composition_rejects_old_trial_protocol_memory_before_dependencies(
    tmp_path, monkeypatch, smoke, old_protocol,
):
    import benchmarks.a3kernels.live_composition as composition_module

    cfg = config(tmp_path)
    cell = next(
        item for item in foundation_cells()
        if item.backend_model is BackendModel.GPT_5_6_SOL
        and item.knowledge is KnowledgeMode.WITHOUT_KDB
        and item.profiling is ProfilingGuidance.WITHOUT_GUIDANCE
    )
    proposals = SMOKE_PROPOSALS if smoke else DEFAULT_PROPOSALS
    lineage_prefix = "smoke" if smoke else "phase1"
    if smoke:
        paths = SmokeWave(
            cfg, lambda *_: {}, cells=(cell,), execution_profile="bz-a3-1",
        ).paths(cell)
    else:
        paths = Phase1Wave(cfg, lambda *_: {}, cells=(cell,)).paths(cell)

    old_dependencies = dependencies([], [])
    if not smoke:
        from canonical_a3_fixtures import A3Session
        class InterruptSession(A3Session):
            def next_action(self, context, allowed, **kwargs):
                raise KeyboardInterrupt()
        def interrupted_actor_factory(selected_cell, proposal):
            actor = actor_factory(selected_cell, proposal)
            if proposal == DEFAULT_PROPOSALS[1]:
                actor.session = actor.canonical_session = InterruptSession(initial=SOURCE)
            return actor
        old_dependencies = LiveDependencies(
            actor_factory=interrupted_actor_factory,
            candidate_factory=old_dependencies.candidate_factory,
            knowledge_factory=old_dependencies.knowledge_factory,
            profiler_factory=old_dependencies.profiler_factory,
        )

    monkeypatch.setattr(
        composition_module, "trial_protocol_sha256",
        lambda: {
            "twelve-turns": trial_protocol_sha256(12),
            "no-mismatch-feedback": trial_protocol_sha256(
                mismatch_feedback=False
            ),
            "no-attempt-completion": trial_protocol_sha256(
                attempt_completion=False
            ),
            "no-prepare-runtime-infrastructure": trial_protocol_sha256(
                prepare_runtime_infrastructure=False
            ),
        }[old_protocol],
    )
    old = LiveComposition(
        cfg, old_dependencies, proposals=proposals,
        lineage_prefix=lineage_prefix,
    )
    if smoke:
        assert old.execute(cell, paths)["status"] == "passed"
    else:
        with pytest.raises(KeyboardInterrupt):
            old.execute(cell, paths)
    assert len(paths.memory.read_text().splitlines()) == 1

    dependency_calls = []
    def forbidden(*_args, **_kwargs):
        dependency_calls.append("called")
        pytest.fail("retained memory mismatch reached a live dependency")
    blocked_dependencies = LiveDependencies(
        actor_factory=forbidden,
        candidate_factory=forbidden,
        knowledge_factory=forbidden,
        profiler_factory=forbidden,
    )
    monkeypatch.setattr(
        composition_module, "trial_protocol_sha256",
        lambda: trial_protocol_sha256(),
    )

    with pytest.raises(ValueError, match="resume validation"):
        LiveComposition(
            cfg, blocked_dependencies, proposals=proposals,
            lineage_prefix=lineage_prefix,
        ).execute(cell, paths)
    assert dependency_calls == []




def test_malformed_provider_result_publishes_terminal_and_later_cell_runs(tmp_path, monkeypatch, canonical_live_trials):
    from canonical_a3_fixtures import CanonicalSession
    from coding.pi.session import ActionError
    from test_a3_trial_loop import FakeClock
    from benchmarks.a3kernels.trial import TrialBudgets
    cfg = config(tmp_path)
    first, later = foundation_cells()[4:6]
    invalid_calls = []
    clock = FakeClock()
    class InvalidSession(CanonicalSession):
        def next_action(self, context, allowed, **kwargs):
            invalid_calls.append(first.cell_id)
            clock.advance(1)
            raise ActionError('scripted malformed provider action')
    def scripted_factory(cell, proposal):
        actor = actor_factory(cell, proposal)
        if cell == first:
            actor.session = actor.canonical_session = InvalidSession()
            actor.canonical_clock = clock
            actor.canonical_budgets = TrialBudgets(max_seconds=10)
        return actor
    deps = LiveDependencies(actor_factory=scripted_factory,
        candidate_factory=lambda cell, proposal, paths: FakeCandidate(proposal),
        knowledge_factory=lambda cell, paths: FakeKnowledge(False, []),
        profiler_factory=lambda cell, proposal, paths, verified: FakeProfiler(
            cell.profiling is ProfilingGuidance.WITH_GUIDANCE, []), execution_profile='bz-a3-1')
    composition = LiveComposition(cfg, deps, proposals=SMOKE_PROPOSALS, lineage_prefix='smoke')
    wave = SmokeWave(cfg, composition.execute, cells=(first, later), execution_profile='bz-a3-1')
    records = wave.run()
    assert [record['status'] for record in records] == ['failed', 'passed']
    assert records[0]['terminal_reason'] == 'time-budget-exhausted'
    assert records[0]['completed_projects'] == 0
    assert records[0]['failed_project_id'] == SMOKE_PROPOSALS[0].project_id
    assert len(invalid_calls) == 10
    owner = canonical_live_trials[0].owner
    events = owner.canonical.build(owner).events
    assert len([row for row in events if row['type'] == 'actor-invalid-action']) == 9
    first_paths = wave.paths(first)
    retained = first_paths.terminal.read_bytes(), first_paths.evidence.read_bytes()
    assert wave.paths(later).terminal.is_file() and wave.resume() == records
    assert len(invalid_calls) == 10
    assert retained == (first_paths.terminal.read_bytes(), first_paths.evidence.read_bytes())


def test_terminal_reports_time_exhaustion_independent_of_token_usage(tmp_path, monkeypatch, canonical_live_trials):
    from canonical_a3_fixtures import CanonicalSession
    from coding.pi.session import ActionError
    from test_a3_trial_loop import FakeClock
    from benchmarks.a3kernels.trial import TrialBudgets
    clock = FakeClock()
    class ExpensiveSession(CanonicalSession):
        completion_tokens = 0
        def next_action(self, context, allowed, **kwargs):
            self.completion_tokens += 40000
            clock.advance(1)
            raise ActionError('scripted malformed expensive action')
    def expensive_factory(cell, proposal):
        actor = actor_factory(cell, proposal)
        actor.session = actor.canonical_session = ExpensiveSession()
        actor.canonical_clock = clock
        actor.canonical_budgets = TrialBudgets(max_seconds=10)
        return actor
    deps = LiveDependencies(actor_factory=expensive_factory,
        candidate_factory=lambda *_: object(), knowledge_factory=lambda *_: FakeKnowledge(False, []),
        profiler_factory=lambda *_: object(), execution_profile='bz-a3-1')
    cfg = config(tmp_path)
    composition = LiveComposition(cfg, deps, proposals=SMOKE_PROPOSALS, lineage_prefix='smoke')
    cell = foundation_cells()[0]
    wave = SmokeWave(cfg, composition.execute, cells=(cell,), execution_profile='bz-a3-1')
    record = wave.run()[0]
    assert record['status'] == 'failed' and record['terminal_reason'] == 'time-budget-exhausted'
    assert record['completed_projects'] == 0
    session, = canonical_live_trials
    assert session.completion_tokens == 400000
    outcome, = [entry.payload['infrastructure_retry'] for entry in session.owner.evidence.entries
                if entry.payload.get('infrastructure_retry', {}).get('event') == 'outcome']
    assert outcome['tokens'] == session.completion_tokens
    retained = wave.paths(cell).evidence.read_bytes()
    resumed = composition.execute(cell, wave.paths(cell))
    assert resumed['terminal_reason'] == 'time-budget-exhausted'
    assert len(canonical_live_trials) == 1 and session.completion_tokens == 400000
    assert wave.paths(cell).evidence.read_bytes() == retained
    assert not session.owner.memory.read()
    assert not [row for row in session.owner.canonical.build(session.owner).events if row['type'] == 'domain-dispatch']


def test_late_host_mismatch_can_be_repaired_before_deadline(tmp_path, monkeypatch, canonical_live_trials):
    from canonical_a3_fixtures import CanonicalSession
    from coding.pi.session import ActionError
    from test_a3_trial_loop import FakeClock
    from benchmarks.a3kernels.trial import TrialBudgets
    clock = FakeClock()
    class LateSession(CanonicalSession):
        invalid = 23
        def next_action(self, context, allowed, **kwargs):
            if self.invalid:
                self.invalid -= 1
                clock.advance(1)
                raise ActionError('scripted invalid decision before candidate')
            clock.advance(.1)
            return super().next_action(context, allowed, **kwargs)
    def late_factory(cell, proposal):
        actor = actor_factory(cell, proposal)
        actor.session = actor.canonical_session = LateSession(initial=SOURCE, repaired=SOURCE + '\n// repair')
        actor.canonical_clock = clock
        actor.canonical_budgets = TrialBudgets(max_seconds=40)
        return actor
    class HostMismatch(FakeCandidate):
        mismatched = False
        def run(self, source, workdir, **options):
            if self.mismatched:
                return super().run(source, workdir, **options)
            self.mismatched = True
            plan = self.base.plan(source, **options)
            output = (0.0,) * plan.padded_length
            return VerifiedResult.from_receipt(plan, ExecutionReceipt(0, output),
                max_abs_error=max(abs(a+b) for a,b in zip(plan.input_a, plan.input_b)))
    deps = LiveDependencies(actor_factory=late_factory,
        candidate_factory=lambda cell, proposal, paths: HostMismatch(proposal),
        knowledge_factory=lambda *_: FakeKnowledge(False, []),
        profiler_factory=lambda *_: FakeProfiler(False, []), execution_profile='bz-a3-1')
    cfg = config(tmp_path)
    cell = next(cell for cell in foundation_cells() if cell.backend_model is BackendModel.GPT_5_6_SOL
        and cell.knowledge is KnowledgeMode.WITHOUT_KDB and cell.profiling is ProfilingGuidance.WITHOUT_GUIDANCE)
    composition = LiveComposition(cfg, deps, proposals=SMOKE_PROPOSALS, lineage_prefix='smoke')
    record = SmokeWave(cfg, composition.execute, cells=(cell,), execution_profile='bz-a3-1').run()[0]
    assert record['status'] == 'passed' and record['terminal_reason'] == 'completed'
    session, = canonical_live_trials
    assert session.invalid == 0 and 23 < clock() < 40
    assert any(context['stage'] == 'debug' for context in session.contexts)
    assert len(session.owner.memory.read()) == 1


def test_composition_rejects_noncanonical_proposals_before_dependencies(tmp_path):
    calls = []
    deps = LiveDependencies(
        actor_factory=lambda *args: calls.append(("actor", args)),
        candidate_factory=lambda *args: calls.append(("candidate", args)),
        knowledge_factory=lambda *args: calls.append(("knowledge", args)),
        profiler_factory=lambda *args: calls.append(("profiler", args)),
    )
    with pytest.raises(ValueError, match="canonical registry order"):
        LiveComposition(
            config(tmp_path), deps,
            proposals=tuple(reversed(DEFAULT_PROPOSALS[:2])),
        )
    assert calls == []


def test_smoke_guidance_cell_requires_one_candidate_bound_pipe_profile(tmp_path):
    cfg = config(tmp_path)
    profile_calls = []
    cell = next(
        item for item in foundation_cells()
        if item.knowledge is KnowledgeMode.WITHOUT_KDB
        and item.profiling is ProfilingGuidance.WITH_GUIDANCE
    )
    composition = LiveComposition(
        cfg, dependencies([], profile_calls), proposals=SMOKE_PROPOSALS,
        lineage_prefix="smoke",
    )
    paths = Phase1Wave(cfg, composition.execute, cells=(cell,)).paths(cell)
    assert composition.execute(cell, paths)["status"] == "passed"
    assert len(profile_calls) == 1
    memory = json.loads(paths.memory.read_text().splitlines()[0])["memory"]
    facts = [fact for fact in memory["host_facts"] if fact["category"] == "profiling"]
    assert len(facts) == 1
    assert facts[0]["statement"].startswith("raw PipeUtilization:")


def test_live_composition_recovers_torn_evidence_before_resume(tmp_path):
    cfg = config(tmp_path)
    composition = LiveComposition(cfg, dependencies([], []))
    cell = foundation_cells()[0]
    paths = Phase1Wave(cfg, composition.execute).paths(cell)
    first = composition.execute(cell, paths)
    committed = paths.evidence.read_bytes()
    with paths.evidence.open("ab") as stream:
        stream.write(b'{"sequence":999')

    resumed = composition.execute(cell, paths)

    assert resumed == first
    assert paths.evidence.read_bytes() == committed


def test_local_knowledge_factory_recovers_torn_query_before_continuation(tmp_path):
    cfg = config(tmp_path)
    cell = next(
        item for item in foundation_cells()
        if item.knowledge is KnowledgeMode.WITHOUT_KDB
    )
    paths = Phase1Wave(cfg, lambda *_: {}).paths(cell)
    journal_path = paths.root / "knowledge-queries.jsonl"
    reference_path = tmp_path / "reference-queries.jsonl"
    for target in (journal_path, reference_path):
        KnowledgeAgent(enabled=False, journal=QueryJournal(target)).query(
            KnowledgeQuery("first")
        )
    with journal_path.open("ab") as stream:
        stream.write(b'{"sequence":2')

    resumed = local_knowledge_factory(cfg)(cell, paths)
    resumed.query(KnowledgeQuery("second"))
    KnowledgeAgent(enabled=False, journal=QueryJournal(reference_path)).query(
        KnowledgeQuery("second")
    )

    assert journal_path.read_bytes() == reference_path.read_bytes()


def test_authority_store_is_durable_exact_and_conflict_rejecting(tmp_path):
    path = tmp_path / "authority.jsonl"; store = AuthoritativeResultStore(path)
    record = store.register("compile", "a" * 64, "b" * 64, "c" * 64, True)
    assert AuthoritativeResultStore(path).resolve("a" * 64) == record
    try:
        store.register("compile", "a" * 64, "b" * 64, "d" * 64, True)
    except ValueError as exc:
        assert "conflicting" in str(exc)
    else: raise AssertionError("conflicting evidence was accepted")


def test_authority_store_recovers_only_an_unterminated_final_record(tmp_path):
    path = tmp_path / "authority.jsonl"
    record = AuthoritativeResultStore(path).register(
        "compile", "a" * 64, "b" * 64, "c" * 64, True
    )
    committed = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"kind":"host-verification"')

    recovered = AuthoritativeResultStore(path)

    assert recovered.resolve("a" * 64) == record
    assert path.read_bytes() == committed


def test_authority_store_restores_delimiter_for_complete_valid_tail(tmp_path):
    path = tmp_path / "authority.jsonl"
    record = AuthoritativeResultStore(path).register(
        "compile", "a" * 64, "b" * 64, "c" * 64, True
    )
    committed = path.read_bytes()
    path.write_bytes(committed.removesuffix(b"\n"))

    recovered = AuthoritativeResultStore(path)

    assert recovered.resolve("a" * 64) == record
    assert path.read_bytes() == committed


def test_authority_store_rejects_malformed_committed_record(tmp_path):
    path = tmp_path / "authority.jsonl"
    path.write_bytes(b"not-json\n")

    with pytest.raises(ValueError, match="line 1"):
        AuthoritativeResultStore(path)


def test_completion_provenance_and_no_secret_in_terminal_output(tmp_path):
    cfg = config(tmp_path); secret = "never-print-this"
    deps = dependencies([], [])
    composition = LiveComposition(cfg, deps)
    record = Phase1Wave(cfg, composition.execute).run()[0]
    assert secret not in json.dumps(record)
    assert record["evidence_sha256"] == canonical_digest(composition.cell_digests[record["cell_id"]])


def test_deepseek_actor_uses_pi_without_exposing_secret():
    seen = {}
    class Session:
        completion_tokens = 0
        def __init__(self, **kwargs): seen.update(kwargs)
        def next_action(self, context, allowed, **kwargs):
            assert allowed == ["compile"]
            self.completion_tokens += 3
            return "compile", {}
        def close(self): pass
    factory = model_actor_factory({"DEEPSEEK_API_KEY": "private-value"}, session_factory=Session)
    cell = next(cell for cell in foundation_cells() if cell.backend_model is BackendModel.DEEPSEEK_FLASH)
    completion = factory(cell, DEFAULT_PROPOSALS[0])(
        load_a3_model_profile(cell.backend_model), '{"allowed_actions":{"compile":[]}}')
    assert seen["model"] == BackendModel.DEEPSEEK_FLASH
    assert completion.provenance["coding_engine"] == "pi"
    assert "private-value" not in json.dumps(completion.as_dict())


def test_remote_bundle_compile_then_run_reuses_one_managed_execution(tmp_path):
    class Managed:
        def __init__(self): self.compile_calls = self.execute_calls = 0
        def compile(self, plan, workdir):
            self.compile_calls += 1
            body = {"plan": plan, "library_sha256": "3" * 64, "stdout": "", "stderr": ""}
            return CandidateCompilation(plan, "3" * 64, "", "", attest(body))
        def execute(self, compilation):
            self.execute_calls += 1
            plan = compilation.plan
            output = tuple(a + b for a, b in zip(plan.input_a, plan.input_b))
            return VerifiedResult.from_receipt(
                plan, ExecutionReceipt(0, output, job_handle="gz-a3:managed"),
                max_abs_error=0.0,
            )
    managed = Managed(); bundle = RemoteCandidateBundle(managed)
    options = dict(request_id="cell", project_id="project", length=32,
                   padded_length=64, block_count=1, seed=0)
    compiled = bundle.compile(SOURCE, tmp_path, attempt_id="turn-2", **options)
    verified = bundle.run(SOURCE, tmp_path, attempt_id="turn-3", **options)
    assert isinstance(compiled, CandidateCompilation)
    assert isinstance(verified, VerifiedResult) and verified.passed
    assert verified.execution_id == compiled.plan.execution_id
    assert verified.attempt_id == "turn-2"
    assert (managed.compile_calls, managed.execute_calls) == (1, 1)


def test_remote_bundle_run_uses_latest_matching_successful_compile(tmp_path):
    class Managed:
        def __init__(self):
            self.compile_calls = 0
            self.executed = []

        def compile(self, plan, workdir):
            self.compile_calls += 1
            library = f"{self.compile_calls}" * 64
            body = {
                "plan": plan, "library_sha256": library,
                "stdout": "", "stderr": "",
            }
            return CandidateCompilation(plan, library, "", "", attest(body))

        def execute(self, compilation):
            self.executed.append(compilation)
            plan = compilation.plan
            output = tuple(a + b for a, b in zip(plan.input_a, plan.input_b))
            return VerifiedResult.from_receipt(
                plan,
                ExecutionReceipt(
                    0, output, job_handle="gz-a3:latest",
                    metadata=(("library_sha256", compilation.library_sha256),),
                ),
                max_abs_error=0.0,
            )

    managed = Managed()
    bundle = RemoteCandidateBundle(managed)
    options = dict(
        request_id="cell", project_id="project", length=32,
        padded_length=64, block_count=1, seed=0,
    )
    first = bundle.compile(SOURCE, tmp_path, attempt_id="turn-2", **options)
    latest = bundle.compile(SOURCE, tmp_path, attempt_id="turn-3", **options)
    verified = bundle.run(SOURCE, tmp_path, attempt_id="turn-4", **options)

    assert first.plan.execution_id != latest.plan.execution_id
    assert managed.executed == [latest]
    assert verified.execution_id == latest.plan.execution_id
    expected = VerifiedResult.from_receipt(
        latest.plan,
        ExecutionReceipt(
            0, tuple(a + b for a, b in zip(latest.plan.input_a, latest.plan.input_b)),
            job_handle="gz-a3:latest",
            metadata=(("library_sha256", latest.library_sha256),),
        ),
        max_abs_error=0.0,
    )
    assert verified.evidence_sha256 == expected.evidence_sha256


def test_profile_uses_exact_retained_compile_binding_and_directory(tmp_path, monkeypatch):
    import benchmarks.a3kernels.live_composition as live
    class Managed:
        def compile(self, plan, workdir):
            body = {"plan": plan, "library_sha256": "4" * 64, "stdout": "", "stderr": ""}
            return CandidateCompilation(plan, "4" * 64, "", "", attest(body))
        def execute(self, compilation):
            plan = compilation.plan
            output = tuple(a + b for a, b in zip(plan.input_a, plan.input_b))
            return VerifiedResult.from_receipt(
                plan, ExecutionReceipt(0, output, job_handle="gz-a3:exact"),
                max_abs_error=0.0,
            )
        def remote_candidate_directory(self, plan):
            return PurePosixPath("/remote/candidates") / plan.execution_id
    managed = Managed()
    store = AuthoritativeResultStore(tmp_path / "authority.jsonl")
    candidate = live._RecordingCandidate(RemoteCandidateBundle(managed), store)
    options = dict(request_id="cell", project_id="a" * 64, length=32,
                   padded_length=64, block_count=1, seed=0)
    candidate.compile(SOURCE, tmp_path, attempt_id="turn-2", **options)
    verified = candidate.run(SOURCE, tmp_path, attempt_id="turn-3", **options)
    request = ProfileRequest(
        CandidateBinding(verified.execution_id, verified.source_fingerprint),
        StudyDimensions(32, 1, 5, 20), ProfileMetric.PIPE_UTILIZATION,
    )
    seen = {}
    class ProfileBackend:
        def __init__(self, **kwargs): seen.update(kwargs)
        def profile(self, selected):
            return CompactProfileResult.create(
                selected, exported_kernels=("vector_add",),
                metric_values=(("pipe", 1.0),), timeline=(("kernel", 1.0),),
                report_sha256="5" * 64,
            )
    monkeypatch.setattr(live, "GZA3ProfilingBackend", ProfileBackend)
    cfg = config(tmp_path / "config")
    paths = Phase1Wave(cfg, lambda *_: {}).paths(
        __import__("benchmarks.a3kernels.phase1_wave", fromlist=["foundation_cells"]).foundation_cells()[0]
    )
    result = live._LazyGZProfiler(
        config=cfg, paths=paths, candidate=candidate, physical_device=0,
    ).profile(request)
    assert result.candidate_execution_id == verified.execution_id
    assert seen["remote_candidate_directory"].endswith(verified.execution_id)
    assert seen["verified_results"] == {verified.execution_id: verified}


def test_cli_run_and_resume_route_bz_and_compose_all_eight_cells(
    tmp_path, monkeypatch, capsys,
):
    import run_a3_phase1
    cfg = config(tmp_path)
    deps = dependencies([], [])
    routed = {}
    preflights = []
    monkeypatch.setattr(
        run_a3_phase1,
        "bz_live_dependencies",
        lambda *a, **k: routed.update(k) or deps,
    )
    monkeypatch.setattr(
        run_a3_phase1, "_bz_preflight",
        lambda *args: preflights.append(args) or {"state": "completed"},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    common = [
        "--state-root", str(cfg.state_root),
        "--validation-wrapper", str(cfg.validation_wrapper),
        "--embedding-cache", str(cfg.embedding_cache),
        "--corpus-artifacts", str(cfg.corpus_artifacts),
        "--knowledge-database", str(cfg.knowledge_database),
        "--knowledge-manifest", str(cfg.knowledge_manifest),
        "--profile", "bz-a3-2", "--cpl-remote", "/checked/cpl-remote",
        "--remote-workspace", "/data2/research", "--physical-device", "0",
    ]
    for command in ("run", "resume"):
        routed.clear()
        assert run_a3_phase1.main([command, *common]) == 0
        output = capsys.readouterr().out
        assert len(json.loads(output)) == 8
        assert "openrouter-secret" not in output and "deepseek-secret" not in output
        environment = routed.pop("environ")
        assert environment is __import__("os").environ
        assert routed == {
            "cpl_remote": "/checked/cpl-remote",
            "profile": "bz-a3-2",
            "remote_workspace": "/data2/research",
            "physical_device": 0,
        }
    assert preflights == [
        (cfg.validation_wrapper, "bz-a3-2", "/checked/cpl-remote"),
        (cfg.validation_wrapper, "bz-a3-2", "/checked/cpl-remote"),
    ]


def test_cli_selected_bz_transport_reaches_candidate_and_profiler(
    tmp_path, monkeypatch, capsys,
):
    import run_a3_phase1

    cfg = config(tmp_path)
    cpl_remote = tmp_path / "cpl-remote"
    cpl_remote.write_text("#!/bin/sh\n")
    cpl_remote.chmod(0o755)
    selected = {}

    class TransportSelected(Exception):
        pass

    class InspectingComposition:
        def __init__(self, _config, dependencies):
            self.dependencies = dependencies

        def execute(self, cell, paths):
            proposal = DEFAULT_PROPOSALS[0]
            candidate = self.dependencies.candidate_factory(cell, proposal, paths)
            profiler = self.dependencies.profiler_factory(
                cell, proposal, paths, object(),
            )
            selected.update(
                dependency_profile=self.dependencies.execution_profile,
                candidate_profile=candidate.backend.profile,
                candidate_cpl_remote=candidate.backend._cpl_remote,
                candidate_workspace=candidate.backend._workspace.as_posix(),
                candidate_device=candidate.backend._device,
                profiler_profile=profiler.profile_name,
                profiler_cpl_remote=profiler.cpl_remote,
                profiler_device=profiler.physical_device,
            )
            raise TransportSelected


    monkeypatch.setattr(run_a3_phase1, "LiveComposition", InspectingComposition)
    monkeypatch.setattr("benchmarks.a3kernels.live_composition.model_actor_factory",
                        lambda *_args, **_kwargs: actor_factory)
    monkeypatch.setattr(
        run_a3_phase1, "_bz_preflight", lambda *args: {"state": "completed"},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    cell = __import__(
        "benchmarks.a3kernels.phase1_wave", fromlist=["foundation_cells"]
    ).foundation_cells()[0]
    argv = [
        "run", "--state-root", str(cfg.state_root),
        "--validation-wrapper", str(cfg.validation_wrapper),
        "--embedding-cache", str(cfg.embedding_cache),
        "--corpus-artifacts", str(cfg.corpus_artifacts),
        "--knowledge-database", str(cfg.knowledge_database),
        "--knowledge-manifest", str(cfg.knowledge_manifest),
        "--profile", "bz-a3-2", "--cpl-remote", str(cpl_remote),
        "--remote-workspace", "/remote/rsi", "--physical-device", "4",
        "--cell-id", cell.cell_id,
    ]

    with pytest.raises(TransportSelected):
        run_a3_phase1.main(argv)
    assert not tuple(cfg.state_root.rglob("terminal.json"))
    assert selected == {
        "dependency_profile": "bz-a3-2",
        "candidate_profile": "bz-a3-2",
        "candidate_cpl_remote": str(cpl_remote),
        "candidate_workspace": "/remote/rsi",
        "candidate_device": 4,
        "profiler_profile": "bz-a3-2",
        "profiler_cpl_remote": str(cpl_remote),
        "profiler_device": 4,
    }
