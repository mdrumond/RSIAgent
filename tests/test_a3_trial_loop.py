import json
from pathlib import Path

import pytest

from benchmarks.a3_experiments import (
    BackendModel, KnowledgeMode, ProfilingGuidance, ProgrammingLevel,
    build_a3_experiment_plan,
)
from benchmarks.a3_model_profiles import A3Completion, load_a3_model_profile
from benchmarks.a3kernels.candidate import CandidateCompilation
from benchmarks.a3kernels.knowledge_agent import KnowledgeQuery
from benchmarks.a3kernels.phase1_evidence import EvidenceLedger
from benchmarks.a3kernels.phase1_memory import Phase1LearningJournal
from benchmarks.a3kernels.phase1_protocol import (
    ExecutionPlan, ExecutionReceipt, FailedEvidence, SourceFile, VerifiedResult, attest,
)
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a3kernels.profiling import CompactProfileResult, ProfileMetric
from benchmarks.a3kernels.trial import A3TrialLoop, Action, TrialBudgets, parse_action


SOURCE = 'extern "C" __global__ __aicore__ void vector_add(GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output, uint32_t count, uint32_t buffer_bytes) {}'


def _cell(*, knowledge=KnowledgeMode.WITHOUT_KDB, profiling=ProfilingGuidance.WITHOUT_GUIDANCE):
    return next(
        cell for cell in build_a3_experiment_plan().cells
        if cell.backend_model is BackendModel.GPT_5_6_SOL
        and cell.knowledge is knowledge and cell.profiling is profiling
        and cell.programming_level is ProgrammingLevel.FOUNDATION
    )


@pytest.mark.parametrize(
    "payload",
    [
        {}, {"action": "write_source"}, {"action": "compile", "argv": ["sh"]},
        {"action": "query", "query": ""}, {"action": "profile", "metric": "Basic"},
        {"action": "submit", "interpretation": "", "supports": []},
    ],
)
def test_action_schema_rejects_missing_extra_or_invalid_fields(payload):
    with pytest.raises(ValueError, match="action"):
        parse_action(json.dumps(payload))


def test_action_schema_has_one_fixed_source_slot_and_no_argv():
    action = parse_action(json.dumps({"action": "write_source", "source": SOURCE}))
    assert action == Action("write_source", source=SOURCE)
    with pytest.raises(ValueError):
        parse_action(json.dumps({"action": "write_source", "path": "other.cpp", "source": SOURCE}))


class FakeCandidate:
    def __init__(self): self.source = None
    def _plan(
        self, source, request_id, attempt_id, length, project_id,
        padded_length=None, block_count=1,
    ):
        padded_length = padded_length or length
        return ExecutionPlan(request_id, attempt_id, project_id, (SourceFile("candidate.cpp", source),),
                             ("python", "host_driver.py", "input.json"),
                             (1.0,) * padded_length, (2.0,) * padded_length,
                             logical_length=length, padded_length=padded_length,
                             block_count=block_count)
    def compile(self, source, _workdir, **kw):
        self.source = source
        plan = self._plan(source, kw["request_id"], kw["attempt_id"], kw["length"], kw["project_id"], kw.get("padded_length"), kw.get("block_count", 1))
        body = {"plan": plan, "library_sha256": "c" * 64, "stdout": "ok", "stderr": ""}
        return CandidateCompilation(plan, "c" * 64, "ok", "", attest(body))
    def run(self, source, _workdir, **kw):
        assert source == self.source
        plan = self._plan(source, kw["request_id"], kw["attempt_id"], kw["length"], kw["project_id"], kw.get("padded_length"), kw.get("block_count", 1))
        return VerifiedResult.from_receipt(
            plan, ExecutionReceipt(0, (3.0,) * plan.padded_length), max_abs_error=0.0
        )


class FakeKnowledge:
    def __init__(self, enabled): self.enabled, self.queries = enabled, []
    def query(self, query):
        assert isinstance(query, KnowledgeQuery)
        self.queries.append(query.query)
        return ()


class FakeProfiler:
    def __init__(self): self.requests = []
    def profile(self, request):
        self.requests.append(request)
        if request.treatment.value.startswith("without"):
            return None
        return CompactProfileResult.create(
            request, exported_kernels=("vector_add",),
            metric_values=(("vector_ratio", .75),), timeline=(("kernel_count", 20.0),),
            report_sha256="d" * 64,
        )


def _run(
    tmp_path, actions, *, cell=None, knowledge=None, profiler=None,
    candidate=None, budgets=None,
):
    selected = cell or _cell()
    profile = load_a3_model_profile(selected.backend_model)
    queue = list(actions)
    prompts = []
    def actor(actual_profile, context):
        assert actual_profile == profile
        prompts.append(context)
        return A3Completion(queue.pop(0), {"profile_sha256": profile.fingerprint})
    journal = Phase1LearningJournal(
        tmp_path / "memory.jsonl", (DEFAULT_PROPOSALS[0],),
        cell_id=selected.cell_id, lineage_id="isolated-lineage",
    )
    loop = A3TrialLoop(
        cell=selected, proposal=DEFAULT_PROPOSALS[0], profile=profile, actor=actor,
        candidate=candidate or FakeCandidate(), knowledge=knowledge or FakeKnowledge(False),
        profiler=profiler or FakeProfiler(), evidence=EvidenceLedger(tmp_path / "evidence.jsonl"),
        memory=journal, workdir=tmp_path / "work", budgets=budgets or TrialBudgets(10, 2000),
    )
    return loop.run(), prompts, journal


def test_fake_actor_end_to_end_disabled_treatments_and_submit(tmp_path):
    actions = [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}', '{"action":"run"}',
        '{"action":"query","query":"padding","limit":3}',
        '{"action":"profile","metric":"PipeUtilization"}',
        json.dumps({"action": "submit", "interpretation": "host result passed", "supports": []}),
    ]
    result, prompts, journal = _run(tmp_path, actions)
    assert result.status == "passed" and result.verified.passed
    assert result.knowledge_queries == 1 and result.profile_result is None
    assert len(journal.read()) == 1
    assert all(_cell().cell_id in prompt and "isolated-lineage" in prompt for prompt in prompts)
    memory = journal.read()[0]["memory"]
    assert memory["agent_interpretations"][0]["supports"] == []
    first, after_write = map(json.loads, prompts[:2])
    assert first["proposal"] == DEFAULT_PROPOSALS[0].as_dict()
    assert first["policy"] == {
        "block_count": 1, "logical_length": 32, "padded_length": 64,
        "profiling_treatment": "without-profiling-guidance",
    }
    assert first["allowed_actions"]["write_source"] == ["action", "source"]
    assert first["current_candidate_source"] is None
    assert after_write["current_candidate_source"] == SOURCE


def test_candidate_failure_retains_structured_attestation_in_context_and_ledger(tmp_path):
    class CompileFailure(FakeCandidate):
        def compile(self, source, _workdir, **kw):
            plan = self._plan(
                source, kw["request_id"], kw["attempt_id"], kw["length"],
                kw["project_id"], kw.get("padded_length"), kw.get("block_count", 1),
            )
            return FailedEvidence.create(
                plan, stage="compile", error_type="CompileError", detail="fixture",
            )

    result, prompts, _ = _run(
        tmp_path,
        [json.dumps({"action": "write_source", "source": SOURCE}),
         '{"action":"compile"}', '{"action":"compile"}'],
        candidate=CompileFailure(), budgets=TrialBudgets(3, 2000),
    )
    observation = json.loads(prompts[2])["observations"][-1]["failed_evidence"]
    assert observation["stage"] == "compile"
    assert len(observation["attestation_sha256"]) == 64
    failure_entries = EvidenceLedger(tmp_path / "evidence.jsonl").entries
    assert observation["attestation_sha256"] in {
        entry.payload["failed_evidence"]["attestation_sha256"]
        for entry in failure_entries if "failed_evidence" in entry.payload
    }
    assert result.failures[-1] == "compile failed: fixture"


def test_enabled_kdb_and_profile_are_exactly_cell_bound(tmp_path):
    cell = _cell(knowledge=KnowledgeMode.WITH_KDB, profiling=ProfilingGuidance.WITH_GUIDANCE)
    knowledge, profiler = FakeKnowledge(True), FakeProfiler()
    actions = [
        json.dumps({"action": "write_source", "source": SOURCE}), '{"action":"compile"}',
        '{"action":"run"}', '{"action":"query","query":"DataCopyPad","limit":2}',
        '{"action":"profile","metric":"PipeUtilization"}',
        json.dumps({"action": "submit", "interpretation": "profile supports result", "supports": []}),
    ]
    result, _, _ = _run(tmp_path, actions, cell=cell, knowledge=knowledge, profiler=profiler)
    assert result.status == "passed"
    assert knowledge.queries == ["DataCopyPad"]
    assert profiler.requests[0].treatment.value == cell.profiling.value
    assert result.profile_result.candidate_execution_id == result.verified.execution_id


def test_submit_gates_source_identity_verification_and_required_profile(tmp_path):
    cell = _cell(profiling=ProfilingGuidance.WITH_GUIDANCE)
    result, _, journal = _run(tmp_path, [
        json.dumps({"action": "write_source", "source": SOURCE}),
        '{"action":"compile"}', '{"action":"run"}',
        json.dumps({"action": "submit", "interpretation": "too early", "supports": []}),
    ], cell=cell, budgets=TrialBudgets(4, 2000))
    assert result.status == "budget-exhausted"
    assert "profile" in result.failures[-1]
    assert journal.read() == ()


def test_iteration_and_token_budgets_are_terminal_and_isolated(tmp_path):
    result, _, journal = _run(
        tmp_path, ['{"action":"compile"}'] * 3, budgets=TrialBudgets(2, 30)
    )
    assert result.status == "budget-exhausted" and result.turns <= 2
    assert journal.read() == ()
    token_result, _, token_journal = _run(
        tmp_path / "tokens", ['{"action": "compile"}'],
        budgets=TrialBudgets(3, 1),
    )
    assert token_result.status == "budget-exhausted" and token_result.turns == 1
    assert token_result.failures == ("token budget exhausted",)
    assert token_journal.read() == ()


def test_model_and_treatment_authorities_must_match_cell(tmp_path):
    cell = _cell(knowledge=KnowledgeMode.WITH_KDB)
    with pytest.raises(ValueError, match="Knowledge Agent"):
        _run(tmp_path, [], cell=cell, knowledge=FakeKnowledge(False))


def test_completion_provenance_cannot_switch_the_cell_model(tmp_path):
    cell = _cell()
    profile = load_a3_model_profile(cell.backend_model)
    journal = Phase1LearningJournal(
        tmp_path / "memory.jsonl", (DEFAULT_PROPOSALS[0],),
        cell_id=cell.cell_id, lineage_id="isolated-lineage",
    )
    loop = A3TrialLoop(
        cell=cell, proposal=DEFAULT_PROPOSALS[0], profile=profile,
        actor=lambda *_: A3Completion('{"action":"compile"}', {"profile_sha256": "0" * 64}),
        candidate=FakeCandidate(), knowledge=FakeKnowledge(False), profiler=FakeProfiler(),
        evidence=EvidenceLedger(tmp_path / "evidence.jsonl"), memory=journal,
        workdir=tmp_path / "work", budgets=TrialBudgets(1, 100),
    )
    result = loop.run()
    assert result.status == "budget-exhausted"
    assert result.failures == ("completion provenance does not match the cell model",)
