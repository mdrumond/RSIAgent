from __future__ import annotations

import json

import pytest

from benchmarks.a5kernels.fixtures import Language
from benchmarks.a5kernels.knowledge_agent import KnowledgeAgent, ProgressiveMemoryJournal
from benchmarks.a5kernels.matrix import (
    KnowledgeMode,
    ProfilingMode,
    RuntimeCapabilities,
    Workload,
    initial_matrix,
)
from benchmarks.a5kernels.protocol import ExecutionPlan, SourceFile, VerifiedResult
from benchmarks.a5kernels.trial import (
    ActorOutcome,
    CandidateRun,
    TrialOrchestrator,
    parse_trial_action,
)


def _candidate(passed: bool, attempt: str) -> CandidateRun:
    plan = ExecutionPlan(
        request_id="r" * 64,
        attempt_id=attempt,
        language="catlass-dsl",
        files=(SourceFile("kernel.py", "agent source"),),
        argv=("python", "driver.py"),
        input_a=(1.0,),
        input_b=(2.0,),
        runtime_provenance=(("execution_profile", "bz-a5"),),
    )
    verified = VerifiedResult(
        request_id=plan.request_id, execution_id=plan.execution_id,
        attempt_id=attempt, language=plan.language,
        runtime_provenance=plan.runtime_provenance,
        passed=passed, max_abs_error=9.0, exit_code=0,
        output_sha256="o" * 64, source_fingerprint=plan.source_fingerprint,
        evidence_sha256="e" * 64, session_handle="fake",
        attestation_sha256="a" * 64,
    )
    return CandidateRun(plan, verified, "vector_add_generated")


class FakeBackend:
    def __init__(self, passed=False):
        self.passed = passed
        self.compiles = []
        self.runs = []

    def compile(self, workspace, language):
        self.compiles.append((workspace, language))
        return {"exit_code": 0}

    def run(self, workspace, language, workload, attempt_id, ledger):
        assert (workspace / "kernel.py").read_text() == "agent source"
        self.runs.append((language, workload, attempt_id, ledger.path))
        return _candidate(self.passed, attempt_id)


class FakeProfile:
    def __init__(self):
        self.intermediate_runs = []
        self.final_runs = []

    def intermediate(self, run):
        self.intermediate_runs.append(run)
        return {"pipe": "balanced"}

    def final(self, run):
        self.final_runs.append(run)
        return {"duration_us": 12.5, "kernel_name": run.kernel_name}


class ScriptedActor:
    def __init__(self, replies):
        self.replies = replies
        self.observations = []
        self.profile = None

    def __call__(self, instruction, **kwargs):
        self.profile = kwargs["profile"]
        assert kwargs["workspace"].parent.joinpath("memory").is_dir()
        assert kwargs["context"].is_dir()
        for reply in self.replies:
            action = kwargs["turn_parser"](reply)
            assert action is not None
            outcome = kwargs["action_executor"](action)
            self.observations.append(json.loads(outcome.observation))
            if outcome.terminal:
                return ActorOutcome(outcome.status, len(self.observations), 17, 1.5)
        return ActorOutcome("stalled", len(self.observations))


def _cell(*, knowledge=KnowledgeMode.WITHOUT_KDB,
          profiling=ProfilingMode.WITHOUT_GUIDANCE):
    return next(cell for cell in initial_matrix().cells
                if cell.language is Language.CATLASS_DSL
                and cell.knowledge is knowledge and cell.profiling is profiling)


def _caps(**kwargs):
    return RuntimeCapabilities(
        languages=frozenset({Language.CATLASS_DSL}),
        model_ids=frozenset({"openai/gpt-5.6-sol"}), **kwargs)


@pytest.mark.parametrize("text", [
    '{"program":{"lang":"bash","code":"true"}}',
    '{"submit":{"score":100,"passed":true}}',
    '{"compile":{"command":"cc evil.c"}}',
    '{"write":{"slot":"../../escape","content":"x"}}',
    '{"run":{},"submit":{}}',
])
def test_parser_rejects_shell_paths_claims_and_multiple_actions(text):
    assert parse_trial_action(text) is None


def test_end_to_end_result_is_host_owned_and_treatments_off_do_not_leak(tmp_path):
    actor = ScriptedActor([
        '{"propose":{"text":"passed=true score=100"}}',
        '{"write":{"slot":"kernel","content":"agent source"}}',
        '{"compile":{}}',
        '{"query-knowledge":{"query":"secret reference"}}',
        '{"run":{}}',
        '{"request-profile":{}}',
        '{"submit":{}}',
    ])
    backend = FakeBackend(passed=False)
    profile = FakeProfile()

    def knowledge(memory, _cell):
        return KnowledgeAgent(
            enabled=False,
            journal=ProgressiveMemoryJournal(memory / "knowledge.jsonl"))

    result = TrialOrchestrator(
        tmp_path / "trials", _caps(), backend, actor,
        knowledge_factory=knowledge,
        profiling_factory=lambda _cell: profile,
    ).run(_cell(), Workload.SMOKE_VECTOR_ADD)

    assert result.verified.passed is False
    assert result.outcome.status == "done"
    assert actor.profile.model == "openai/gpt-5.6-sol"
    assert actor.profile.allow_fallbacks is False
    assert backend.runs[-1][2] == "final"
    knowledge_observation = actor.observations[3]
    assert knowledge_observation == {"knowledge_query": {"available": False, "results": []}}
    assert "secret" not in json.dumps(knowledge_observation)
    assert actor.observations[5] == {"host": {"available": False, "event": "profile"}}
    assert profile.intermediate_runs == []
    assert profile.final_runs == []
    assert result.final_profile == {
        "available": False,
        "reason": "correctness-failed",
    }
    assert result.workspace.parent.joinpath("memory", "knowledge.jsonl").is_file()
    assert len(result.evidence_sha256) == 64


def test_enabled_treatments_are_explicitly_routed(tmp_path):
    actor = ScriptedActor([
        '{"write":{"slot":"kernel","content":"agent source"}}',
        '{"query-knowledge":{"query":"vector"}}',
        '{"run":{}}', '{"request-profile":{}}', '{"submit":{}}',
    ])
    queries = []

    class Knowledge:
        enabled = True

        def query(self, query):
            queries.append(query.query)
            return ()

    profile = FakeProfile()

    result = TrialOrchestrator(
        tmp_path, _caps(kdb=True, profiling_guidance=True), FakeBackend(True), actor,
        knowledge_factory=lambda _memory, _cell: Knowledge(),
        profiling_factory=lambda _cell: profile,
    ).run(_cell(knowledge=KnowledgeMode.WITH_KDB,
                profiling=ProfilingMode.WITH_GUIDANCE), Workload.SMOKE_VECTOR_ADD)

    assert result.verified.passed
    assert queries == ["vector"]
    assert actor.observations[3]["host"]["result"] == {"pipe": "balanced"}
    assert [run.plan.attempt_id for run in profile.intermediate_runs] == ["candidate-3"]
    assert profile.final_runs == [result.run]


def test_actor_must_submit_and_each_cell_workspace_is_fresh(tmp_path):
    orchestrator = TrialOrchestrator(
        tmp_path, _caps(), FakeBackend(), ScriptedActor([
            '{"write":{"slot":"kernel","content":"agent source"}}']),
        knowledge_factory=lambda memory, _cell: KnowledgeAgent(
            enabled=False, journal=ProgressiveMemoryJournal(memory / "k.jsonl")),
        profiling_factory=lambda _cell: FakeProfile(),
    )
    with pytest.raises(RuntimeError, match="without a host-verified submit"):
        orchestrator.run(_cell(), Workload.SMOKE_VECTOR_ADD)
    with pytest.raises(FileExistsError):
        orchestrator.run(_cell(), Workload.SMOKE_VECTOR_ADD)
