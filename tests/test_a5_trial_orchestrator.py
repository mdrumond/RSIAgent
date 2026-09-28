from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.evidence import EvidenceKind, EvidenceLedger
from benchmarks.a5kernels.fixtures import Language
from benchmarks.a5kernels.knowledge_agent import KnowledgeAgent, ProgressiveMemoryJournal
from benchmarks.a5kernels.matrix import (
    CapabilityUnavailableError,
    KnowledgeMode,
    ProfilingMode,
    RuntimeCapabilities,
    Workload,
    initial_matrix,
)
from benchmarks.a5kernels.model_profile import load_a5_model_profile
from benchmarks.a5kernels.protocol import ExecutionPlan, SourceFile, VerifiedResult
from benchmarks.a5kernels.trial import (
    ActorOutcome,
    ActionOutcome,
    CandidateRun,
    CoreAttemptDriver,
    TrialActionExecutor,
    TrialOrchestrator,
    _trial_instruction,
    parse_trial_action,
)


def _candidate(passed: bool, attempt: str, *,
               workload=Workload.SMOKE_VECTOR_ADD,
               language="catlass-dsl", max_abs_error=9.0) -> CandidateRun:
    plan = ExecutionPlan(
        request_id="r" * 64,
        attempt_id=attempt,
        language=language,
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
        passed=passed, max_abs_error=max_abs_error, exit_code=0,
        output_sha256="o" * 64, source_fingerprint=plan.source_fingerprint,
        evidence_sha256="e" * 64, session_handle="fake",
        attestation_sha256="a" * 64,
    )
    return CandidateRun(plan, verified, "vector_add_generated", workload,
                        hashlib.sha256(b"agent source").hexdigest())


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
        return _candidate(self.passed, attempt_id, workload=workload,
                          language=language)


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
        self.instruction = None

    def __call__(self, instruction, **kwargs):
        self.instruction = instruction
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
        model_ids=frozenset({"openai/gpt-5.6-sol"}),
        workloads=frozenset(Workload),
        **kwargs)


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
    assert "catlass-dsl" in actor.instruction
    assert "smoke-vector-add" in actor.instruction
    assert backend.runs[-1][2].endswith("-final")
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
    assert len(profile.intermediate_runs) == 1
    assert profile.intermediate_runs[0].plan.attempt_id.endswith("-candidate-3")
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


def test_future_workload_contract_fails_before_actor_or_workspace_creation(tmp_path):
    def orchestrator(actor):
        return TrialOrchestrator(
            tmp_path, _caps(), FakeBackend(True), actor,
            knowledge_factory=lambda memory, _cell: KnowledgeAgent(
                enabled=False, journal=ProgressiveMemoryJournal(memory / "k.jsonl")),
            profiling_factory=lambda _cell: FakeProfile(),
        )

    smoke = orchestrator(ScriptedActor([
        '{"write":{"slot":"kernel","content":"agent source"}}',
        '{"submit":{}}',
    ])).run(_cell(), Workload.SMOKE_VECTOR_ADD)
    gemm_actor = ScriptedActor([
        '{"write":{"slot":"kernel","content":"agent source"}}',
        '{"submit":{}}',
    ])
    with pytest.raises(CapabilityUnavailableError, match="preregistered for future"):
        orchestrator(gemm_actor).run(_cell(), Workload.SEMANTIC_GEMM)
    assert not smoke.workspace.parent.parent.joinpath("semantic-gemm").exists()
    assert gemm_actor.instruction is None


def test_explicit_run_identity_isolates_preregistered_repeats(tmp_path):
    from benchmarks.a5kernels.bz import BZSessionAdapter, CommandResult

    backend = FakeBackend(True)

    def run(identity):
        return TrialOrchestrator(
            tmp_path, _caps(), backend, ScriptedActor([
                '{"write":{"slot":"kernel","content":"agent source"}}',
                '{"run":{}}',
                '{"submit":{}}',
            ]),
            knowledge_factory=lambda memory, _cell: KnowledgeAgent(
                enabled=False, journal=ProgressiveMemoryJournal(memory / "k.jsonl")),
            profiling_factory=lambda _cell: FakeProfile(),
        ).run(_cell(), Workload.SMOKE_VECTOR_ADD, run_identity=identity)

    first = run("repeat-1")
    second = run("repeat-2")
    assert first.workspace.parent.name == "repeat-1"
    assert second.workspace.parent.name == "repeat-2"
    assert len({attempt for _, _, attempt, _ in backend.runs}) == 4
    assert all(len(attempt) <= 64 for _, _, attempt, _ in backend.runs)
    # Identical source/inputs have the same execution identity, but the BZ
    # adapter must still dispatch them into distinct retained sessions.
    assert first.run.plan.execution_id == second.run.plan.execution_id
    sessions = []

    def dispatch(invocation):
        sessions.append(invocation.argv[2])
        return CommandResult(1, "offline dispatch probe")

    adapter = BZSessionAdapter(SimpleNamespace(
        runtime_provenance=first.run.plan.runtime_provenance, run=dispatch,
    ), session_wrapper="profiles/bz-a5/session.sh")
    adapter.execute(first.run.plan)
    adapter.execute(second.run.plan)
    assert len(set(sessions)) == 2

    with pytest.raises(ValueError, match="safe host-owned"):
        run("../repeat-3")


@pytest.mark.parametrize("action", ["compile", "run", "submit"])
def test_actions_before_write_return_recoverable_observations(tmp_path, action):
    actor = ScriptedActor([
        json.dumps({action: {}}),
        '{"write":{"slot":"kernel","content":"agent source"}}',
        '{"submit":{}}',
    ])
    result = TrialOrchestrator(
        tmp_path, _caps(), FakeBackend(True), actor,
        knowledge_factory=lambda memory, _cell: KnowledgeAgent(
            enabled=False, journal=ProgressiveMemoryJournal(memory / "k.jsonl")),
        profiling_factory=lambda _cell: FakeProfile(),
    ).run(_cell(), Workload.SMOKE_VECTOR_ADD)

    assert actor.observations[0]["host"] == {
        "event": action,
        "error": "write kernel source first",
    }
    assert result.verified.passed


def test_core_driver_freezes_generation_and_restricted_nudges(monkeypatch, tmp_path):
    captured = {}

    class Config:
        primary_temperature = 1.0

    class Result:
        status = "done"
        iters = 2
        wall_secs = 3.5

    def run_attempt(*args, **kwargs):
        captured["cfg"] = args[2]
        captured["kwargs"] = kwargs
        return Result(), []

    monkeypatch.setattr("core.loop.run_attempt", run_attempt)
    driver = CoreAttemptDriver(object(), Config(), lambda path: path)
    outcome = driver(
        "instruction", profile=load_a5_model_profile(),
        workspace=tmp_path, context=tmp_path,
        turn_parser=parse_trial_action, action_executor=lambda _action: None,
    )

    assert captured["cfg"].primary_temperature == 0.0
    assert captured["cfg"].retry_temperature == 0.0
    assert captured["cfg"].temperature == 0.0
    assert captured["cfg"].allow_truncation_retry is False
    assert captured["kwargs"]["action_nudge"] == captured["kwargs"]["strict_action_nudge"]
    for action in ("propose", "write", "compile", "run", "query-knowledge",
                   "request-profile", "submit"):
        assert action in captured["kwargs"]["action_nudge"]
    assert outcome.wall_time_s == 3.5
    assert outcome.tokens is None


def test_core_driver_disables_inherited_compaction_without_mutating_config(monkeypatch, tmp_path):
    import core.loop as loop
    from config.settings import load
    from core.trace import ArtifactSink

    cfg = load(None)
    cfg.max_iters = 4
    cfg.history_keep_pairs = 1
    cfg.keep_chars = 0
    cfg.fold_batch = 1
    cfg.ctx_high_water = 0
    requests = []
    replies = iter([
        '{"propose":{"text":"first"}}',
        '{"propose":{"text":"second"}}',
        '{"submit":{}}',
    ])

    def chat(_model, system, _user, **kwargs):
        assert system != loop.SUMMARIZER_SYSTEM
        requests.append(kwargs)
        return next(replies)

    monkeypatch.setattr(loop, "chat", chat)
    outcome = CoreAttemptDriver(object(), cfg, lambda path: ArtifactSink(str(path)))(
        "task", profile=load_a5_model_profile(), workspace=tmp_path,
        context=tmp_path / "context", turn_parser=parse_trial_action,
        action_executor=lambda action: ActionOutcome("ok", action.kind == "submit"),
    )

    assert outcome.status == "done"
    assert len(requests) == 3
    assert all(request["allow_truncation_retry"] is False for request in requests)
    assert all(request["top_p"] == 1.0 for request in requests)
    assert cfg.history_keep_pairs == 1


def test_failed_candidate_can_omit_undiscovered_kernel_name(tmp_path):
    failed = replace(_candidate(False, "failed"), kernel_name=None)
    assert failed.kernel_name is None
    with pytest.raises(ValueError, match="passed candidates require"):
        replace(_candidate(True, "passed"), kernel_name=None)
    with pytest.raises(ValueError, match="exact non-empty"):
        replace(_candidate(True, "passed"), kernel_name="  kernel  ")

    class FailedBackend(FakeBackend):
        def run(self, workspace, language, workload, attempt_id, ledger):
            return replace(_candidate(False, attempt_id), kernel_name=None)

    executor = _executor(tmp_path, FailedBackend())
    executor(parse_trial_action('{"write":{"slot":"kernel","content":"agent source"}}'))
    outcome = executor(parse_trial_action('{"submit":{}}'))
    assert outcome.terminal
    assert json.loads(outcome.observation)["host"]["passed"] is False
    assert executor.final_profile == {"available": False, "reason": "correctness-failed"}


def test_write_invalidates_previous_run_before_profile(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backend = FakeBackend(True)
    profile = FakeProfile()
    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")
    executor = TrialActionExecutor(
        workspace=workspace, language="catlass-dsl",
        workload=Workload.SMOKE_VECTOR_ADD, backend=backend, ledger=ledger,
        knowledge=type("Knowledge", (), {"enabled": False})(),
        profiling=profile, profiling_guidance=True,
    )

    for text in (
        '{"write":{"slot":"kernel","content":"agent source"}}',
        '{"run":{}}',
        '{"write":{"slot":"kernel","content":"revised source"}}',
    ):
        executor(parse_trial_action(text))

    outcome = executor(parse_trial_action('{"request-profile":{}}'))
    assert json.loads(outcome.observation)["host"] == {
        "available": True,
        "error": "run a correct candidate first",
        "event": "profile",
    }
    assert profile.intermediate_runs == []


def test_nonfinite_error_is_a_valid_explicit_json_observation(tmp_path):
    class NonfiniteBackend(FakeBackend):
        def run(self, workspace, language, workload, attempt_id, ledger):
            return _candidate(False, attempt_id, workload=workload,
                              language=language, max_abs_error=float("inf"))

    executor = _executor(tmp_path, NonfiniteBackend())
    executor(parse_trial_action(
        '{"write":{"slot":"kernel","content":"agent source"}}'))
    outcome = executor(parse_trial_action('{"run":{}}'))
    assert json.loads(outcome.observation)["host"] == {
        "event": "run", "passed": False, "max_abs_error": None,
        "exit_code": 0, "status": "incorrect-output",
        "error_status": "non-finite-max-abs-error",
    }
    assert "Infinity" not in outcome.observation


@pytest.mark.parametrize("passed, exit_code, status", [
    (False, 1, "runtime-error"),
    (False, 124, "runtime-error"),
    (False, 0, "incorrect-output"),
    (True, 0, "passed"),
])
def test_run_feedback_distinguishes_runtime_failure_from_incorrect_output(
    tmp_path, passed, exit_code, status,
):
    class ResultBackend(FakeBackend):
        def run(self, workspace, language, workload, attempt_id, ledger):
            error = float("inf") if exit_code else 0.0 if passed else 9.0
            candidate = _candidate(passed, attempt_id, max_abs_error=error)
            return replace(candidate, verified=replace(candidate.verified, exit_code=exit_code))

    executor = _executor(tmp_path, ResultBackend())
    executor(parse_trial_action('{"write":{"slot":"kernel","content":"agent source"}}'))
    outcome = executor(parse_trial_action('{"run":{}}'))

    assert not outcome.terminal
    assert json.loads(outcome.observation)["host"] == {
        "event": "run", "passed": passed, "exit_code": exit_code, "status": status,
        "max_abs_error": None if exit_code else 0.0 if passed else 9.0,
        "error_status": "non-finite-max-abs-error" if exit_code else None,
    }


def _executor(tmp_path, backend):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return TrialActionExecutor(
        workspace=workspace, language="catlass-dsl",
        workload=Workload.SMOKE_VECTOR_ADD, backend=backend,
        ledger=EvidenceLedger(tmp_path / "evidence.jsonl"),
        knowledge=type("Knowledge", (), {"enabled": False})(),
        profiling=FakeProfile(), profiling_guidance=False,
    )


@pytest.mark.parametrize("action", ["compile", "run", "submit"])
def test_source_validation_errors_are_recoverable(tmp_path, action):
    class RejectingBackend(FakeBackend):
        def compile(self, workspace, language):
            raise ValueError("invalid candidate source")

        def run(self, workspace, language, workload, attempt_id, ledger):
            raise ValueError("invalid candidate source")

    executor = _executor(tmp_path, RejectingBackend())
    executor(parse_trial_action(
        '{"write":{"slot":"kernel","content":"agent source"}}'))
    outcome = executor(parse_trial_action(json.dumps({action: {}})))
    assert not outcome.terminal
    assert json.loads(outcome.observation)["host"] == {
        "event": action, "status": "source-validation-error",
        "error": "invalid candidate source",
    }


@pytest.mark.parametrize("mismatch", ["language", "workload"])
def test_candidate_must_match_scheduled_language_and_workload(tmp_path, mismatch):
    class MismatchBackend(FakeBackend):
        def run(self, workspace, language, workload, attempt_id, ledger):
            return _candidate(
                True, attempt_id,
                language="ascend-c" if mismatch == "language" else language,
                workload=(Workload.SEMANTIC_GEMM
                          if mismatch == "workload" else workload),
            )

    executor = _executor(tmp_path, MismatchBackend())
    executor(parse_trial_action(
        '{"write":{"slot":"kernel","content":"agent source"}}'))
    outcome = executor(parse_trial_action('{"run":{}}'))
    observation = json.loads(outcome.observation)["host"]
    assert observation["status"] == "source-validation-error"
    assert mismatch in observation["error"]


@pytest.mark.parametrize("action", ["run", "submit"])
@pytest.mark.parametrize("fail", [False, True])
def test_accepted_action_precedes_backend_evidence_and_survives_failure(tmp_path, action, fail):
    class RecordingBackend(FakeBackend):
        def run(self, workspace, language, workload, attempt_id, ledger):
            assert ledger.entries[-1].payload["action"] == action
            ledger.append(EvidenceKind.REQUEST, {"attempt_id": attempt_id})
            if fail:
                raise RuntimeError("backend unavailable")
            ledger.append(EvidenceKind.ARTIFACT, {"attempt_id": attempt_id})
            ledger.append(EvidenceKind.RESULT, {"attempt_id": attempt_id})
            return _candidate(True, attempt_id)

    executor = _executor(tmp_path, RecordingBackend())
    executor(parse_trial_action('{"write":{"slot":"kernel","content":"agent source"}}'))
    parsed = parse_trial_action(json.dumps({action: {}}))
    if fail:
        with pytest.raises(RuntimeError, match="backend unavailable"):
            executor(parsed)
    else:
        executor(parsed)
    entries = EvidenceLedger(tmp_path / "evidence.jsonl").entries
    expected = ["action", "action", "request"]
    assert [entry.kind for entry in entries] == expected + ([] if fail else ["artifact", "result"])
    assert entries[1].payload["action"] == action


@pytest.mark.parametrize("action", ["run", "submit"])
def test_candidate_from_previous_source_is_rejected_before_profiling(tmp_path, action):
    class StaleBackend(FakeBackend):
        def run(self, workspace, language, workload, attempt_id, ledger):
            return _candidate(True, attempt_id)

    executor = _executor(tmp_path, StaleBackend())
    executor(parse_trial_action('{"write":{"slot":"kernel","content":"revised source"}}'))
    outcome = executor(parse_trial_action(json.dumps({action: {}})))
    assert not outcome.terminal
    assert json.loads(outcome.observation)["host"] == {
        "event": action, "status": "source-validation-error",
        "error": "candidate source does not match current workspace kernel",
    }
    assert executor.latest_run is None
    assert executor.final_run is None
    assert executor.profiling.intermediate_runs == executor.profiling.final_runs == []


def test_candidate_requires_an_exact_source_digest():
    with pytest.raises(ValueError, match="candidate_source_sha256"):
        replace(_candidate(True, "candidate"), candidate_source_sha256="unknown")


@pytest.mark.parametrize("action", ["run", "submit"])
def test_cached_same_source_candidate_cannot_satisfy_a_new_attempt(tmp_path, action):
    class CachedBackend(FakeBackend):
        cached = None

        def run(self, workspace, language, workload, attempt_id, ledger):
            if self.cached is None:
                self.cached = super().run(workspace, language, workload, attempt_id, ledger)
            else:
                assert self.cached.plan.attempt_id != attempt_id
            return self.cached

    executor = _executor(tmp_path, CachedBackend(passed=True))
    executor(parse_trial_action('{"write":{"slot":"kernel","content":"agent source"}}'))
    executor(parse_trial_action('{"run":{}}'))
    assert executor.latest_run.verified.passed

    outcome = executor(parse_trial_action(json.dumps({action: {}})))

    assert not outcome.terminal
    assert json.loads(outcome.observation)["host"] == {
        "event": action, "status": "source-validation-error",
        "error": "candidate attempt_id does not match requested attempt",
    }
    if action == "run":
        assert executor.latest_run is None
    assert executor.final_run is None
    assert executor.final_profile is None
    assert executor.profiling.intermediate_runs == executor.profiling.final_runs == []


def test_no_kdb_catlass_opening_specifies_executable_contract():
    instruction = _trial_instruction("catlass-dsl", Workload.SMOKE_VECTOR_ADD)
    contract = json.loads(instruction.split("Executable contract:\n", 1)[1])
    assert "vector_add(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor)" in contract["entry_point"]
    assert "float32" in contract["arguments"] and "one-dimensional" in contract["arguments"]
    assert "N = 32" in contract["shape"] and "P = 64" in contract["shape"]
    assert "gm_a.origin_shape[0]" in contract["shape"]
    assert "does not establish correctness for other logical lengths" in contract["shape"]
    assert "[1, 400]" not in instruction and "448" not in instruction
    assert "gm_a[i] + gm_b[i]" in contract["output"]
    assert "1e-5" in contract["correctness"]
    assert "import catlass.tla as tla" in contract["source"]
    assert "block_num=1" in contract["runtime"]


@pytest.mark.parametrize("language, workload", [
    (language.value, workload) for language in Language for workload in Workload
    if (language, workload) != (Language.CATLASS_DSL, Workload.SMOKE_VECTOR_ADD)
])
def test_unimplemented_trial_contracts_are_explicit(language, workload):
    with pytest.raises(CapabilityUnavailableError, match="no executable trial contract"):
        _trial_instruction(language, workload)
