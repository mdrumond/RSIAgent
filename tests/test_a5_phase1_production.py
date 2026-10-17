from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import run_a5_phase1
from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import EvidenceLedger
from benchmarks.a5kernels.knowledge_agent import KnowledgeAgent, KnowledgeQuery, ProgressiveMemoryJournal
from benchmarks.a5kernels.phase1_composition import A5_PROFILING_GUIDANCE, LiveProjectRequest
from benchmarks.a5kernels.phase1_experiments import (
    MODEL_IDENTITIES,
    PINNED_CATLASS_REVISION,
    KnowledgeMode,
    ProfilingGuidance,
    build_cells,
)
from benchmarks.a5kernels.phase1_production import (
    ProductionProjectExecutor,
    _require_host_verification,
    _require_semantic_recovery_failure,
    build_phase1_live_dependencies,
    load_direct_environment,
)
from benchmarks.a5kernels.phase1_live import InfrastructureFailure
from benchmarks.a5kernels.phase1_provider import DirectCompletion, DirectCompletionProvider
from benchmarks.a5kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a5kernels.phase1_runtime import Phase1ProjectRuntime
from benchmarks.a5kernels.production import ProductionPaths
from benchmarks.a5kernels.profiling import TimingResult
from benchmarks.a5kernels.protocol import ExecutionReceipt
from tests.test_a5_catlass_candidate import FakeExecution, SOURCE
from tests.test_a5_phase1_composition import (
    FakeEmbeddings,
    _dependencies,
    _guide_and_report,
    _knowledge_database,
)
from tests.test_a5_phase1_performance_execution import RecordingBackend


ENVIRONMENT = {
    "OPENAI_API_KEY": "openai-secret",
    "DEEPSEEK_API_KEY": "deepseek-secret",
}


@pytest.mark.parametrize("exit_code", [75, 255])
@pytest.mark.parametrize("session_handle", [None, "bz-a5:retained"])
def test_transport_failure_during_final_verification_is_retryable(
    exit_code, session_handle,
):
    with pytest.raises(InfrastructureFailure, match="transport"):
        _require_host_verification(SimpleNamespace(
            passed=False, exit_code=exit_code, session_handle=session_handle,
        ))

    with pytest.raises(RuntimeError, match="did not pass host"):
        _require_host_verification(SimpleNamespace(
            passed=False, exit_code=1, session_handle="bz-a5:completed",
        ))


@pytest.mark.parametrize("exit_code", [75, 255])
def test_recovery_starters_reject_unavailable_transport(exit_code):
    with pytest.raises(InfrastructureFailure, match="recovery observation transport"):
        _require_semantic_recovery_failure(
            passed=False, exit_code=exit_code, label="compile"
        )


def test_recovery_starters_accept_only_semantic_failures():
    _require_semantic_recovery_failure(
        passed=False, exit_code=1, label="compile"
    )
    _require_semantic_recovery_failure(
        passed=False, exit_code=0, label="runtime"
    )
    with pytest.raises(RuntimeError, match="did not fail"):
        _require_semantic_recovery_failure(
            passed=True, exit_code=0, label="runtime"
        )
    with pytest.raises(RuntimeError, match="did not fail"):
        _require_semantic_recovery_failure(
            passed=False, exit_code=0, label="compile"
        )
    with pytest.raises(RuntimeError, match="did not fail"):
        _require_semantic_recovery_failure(
            passed=False, exit_code=1, label="runtime"
        )
    with pytest.raises(RuntimeError, match="was not observed"):
        _require_semantic_recovery_failure(
            passed=False, exit_code=None, label="compile"
        )


def test_direct_credentials_load_from_the_explicit_secret_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=openai-secret\nDEEPSEEK_API_KEY=deepseek-secret\n"
    )
    loaded = load_direct_environment({}, env_file=env_file)
    assert {name: loaded[name] for name in ENVIRONMENT} == ENVIRONMENT


class TrialProfileBackend(RecordingBackend):
    def time(self, command):
        self.commands.append(command)
        return TimingResult(
            8.0,
            command.request.source_fingerprint,
            command.request.execution_id,
            command.replay_id,
        )


class ProjectExecution(FakeExecution):
    def execute(self, plan):
        self.plans.append(plan)
        source = next(
            item.content for item in plan.files if item.relative_path == "kernel.py"
        )
        expected = tuple(left + right for left, right in zip(plan.input_a, plan.input_b))
        if "missing_input" in source:
            return ExecutionReceipt(1, (), stderr="registered compile failure")
        if "fault_c_0.store(fault_a_0.load())" in source:
            expected = (0.0,) * len(expected)
        return ExecutionReceipt(
            0,
            expected,
            stdout="A5KERNEL_NAME=vector_add\n",
            session_handle="bz-a5:project-fake",
        )


def _paths(tmp_path, database_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    return ProductionPaths(
        tla_root=tmp_path / "tla",
        profiling_skill_root=tmp_path / "profiling",
        catlass_source="/remote/catlass",
        catlass_revision=PINNED_CATLASS_REVISION,
        bge_cache=cache,
        kdb=database_path,
        collection="catlass",
        results_root=tmp_path / "results",
        device=0,
    )


def test_dependency_builder_fails_closed_then_wires_read_only_kdb(tmp_path):
    database_path = _knowledge_database(tmp_path / "knowledge")
    paths = _paths(tmp_path, database_path)
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        return DirectCompletion('{"submit":{}}', 1)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        build_phase1_live_dependencies(
            paths,
            environment={"OPENAI_API_KEY": "present"},
            transport=transport,
            execution_backend=ProjectExecution(),
            profile_backend=TrialProfileBackend(),
            embeddings=FakeEmbeddings(),
        )

    dependencies = build_phase1_live_dependencies(
        paths,
        environment=ENVIRONMENT,
        transport=transport,
        execution_backend=ProjectExecution(),
        profile_backend=TrialProfileBackend(),
        embeddings=FakeEmbeddings(),
    )
    memory = tmp_path / "memory"
    memory.mkdir()
    gate_root = tmp_path / "gate"
    gate_root.mkdir()
    cell = next(cell for cell in build_cells(_guide_and_report(gate_root)[1])
                if cell.knowledge is KnowledgeMode.WITH_KDB)
    agent = dependencies.knowledge_factory(cell, memory)
    try:
        assert agent.query(KnowledgeQuery("vector", 1))
    finally:
        agent.close()
    assert calls == []


@pytest.mark.parametrize(
    "model,profiling,guidance,proposal,expected_commands",
    [
        (MODEL_IDENTITIES[0], ProfilingGuidance.WITHOUT_GUIDANCE, None,
         DEFAULT_PROPOSALS[0], 3),
        (MODEL_IDENTITIES[1], ProfilingGuidance.WITH_GUIDANCE,
         A5_PROFILING_GUIDANCE,
         next(item for item in DEFAULT_PROPOSALS
              if item.evidence_preset.value == "correctness-timing"), 8),
    ],
)
def test_executor_binds_direct_model_and_profiling_treatment(
    tmp_path, model, profiling, guidance, proposal, expected_commands,
):
    gate_root = tmp_path / "gate"
    gate_root.mkdir()
    guide_path, guide, report = _guide_and_report(gate_root)
    del guide_path
    cell = next(
        cell for cell in build_cells(guide)
        if cell.model == model
        and cell.knowledge is KnowledgeMode.WITHOUT_KDB
        and cell.profiling is profiling
    )
    replies = iter([
        json.dumps({"write": {"slot": "kernel", "content": SOURCE}}),
        '{"run":{}}',
        '{"request-profile":{}}',
        '{"submit":{}}',
    ])
    provider_calls = []

    def transport(**kwargs):
        provider_calls.append(kwargs)
        try:
            response = next(replies)
        except StopIteration:
            prompt = kwargs["request"]["messages"][-1]["content"]
            context = json.loads(prompt.rsplit("\n", 1)[-1])
            admitted = context["admitted_artifacts"]
            performance = [digest for digest, role in admitted.items()
                           if role == "performance-study"]
            category = "performance" if performance else "correctness"
            evidence = performance[0] if performance else next(
                digest for digest, role in admitted.items()
                if role == "host-verification")
            response = json.dumps({"distillation": {
                "no_new_knowledge": False,
                "lessons": [{
                    "category": category,
                    "claim": "The retained study supports this bounded lesson.",
                    "applicability": "The exact registered vector-add dimensions.",
                    "preconditions": "Use the retained source and host contract.",
                    "limitations": "This does not generalize to other shapes.",
                    "evidence_artifact_sha256": [evidence],
                    "failure_signature": None,
                    "repair": None,
                    "confidence": "high",
                }],
            }})
        return DirectCompletion(response, 2)

    profile_backend = TrialProfileBackend()
    executor = ProductionProjectExecutor(
        ProjectExecution(),
        profile_backend,
        DirectCompletionProvider(ENVIRONMENT, transport=transport),
        device=0,
    )
    memory_path = tmp_path / "memory"
    memory_path.mkdir()
    request = LiveProjectRequest(
        cell, proposal, 1, "programming guide", guide.guide_sha256,
        report.report_sha256, json.dumps({"completed_projects": 0}), memory_path,
    )
    knowledge = KnowledgeAgent(
        enabled=False,
        journal=ProgressiveMemoryJournal(memory_path / "knowledge.jsonl"),
    )

    ledger = EvidenceLedger(tmp_path / "evidence.jsonl")
    ledger.append("action", {"action": "previous-project-action"})
    memory = executor(
        request,
        Phase1ProjectRuntime.from_proposal(proposal),
        knowledge,
        guidance,
        ledger,
    )

    assert memory.host_facts[0].category == "host-verification"
    assert memory.no_new_knowledge is False
    expected_category = (
        "performance" if proposal.evidence_preset.value == "correctness-timing"
        else "correctness"
    )
    assert memory.lessons[0].category == expected_category
    assert memory.lessons[0].evidence_artifact_sha256[0] in (
        memory.admitted_artifact_sha256
    )
    assert memory.artifact_store.startswith("provider-attempts/attempt-")
    traces = list((memory_path / "sft-traces").glob("*.jsonl"))
    assert len(traces) == 1
    trace_rows = [json.loads(line) for line in traces[0].read_text().splitlines()]
    trace_messages = [row["payload"] for row in trace_rows if row["kind"] == "message"]
    assert trace_messages[-2]["role"] == "user"
    assert "verified candidate" in trace_messages[-2]["content"]
    assert trace_messages[-1] == {"role": "assistant", "content": SOURCE}
    assert any(row["kind"] == "tool" for row in trace_rows)
    assert trace_rows[-2]["kind"] == "validation"
    assert trace_rows[-2]["payload"]["status"] == "passed"
    assert trace_rows[-2]["payload"]["evidence_sha256"] == memory.host_facts[0].evidence_sha256
    assert (memory_path / memory.artifact_store / "objects").is_dir()
    assert len(profile_backend.commands) == expected_commands
    assert {call["base_url"] for call in provider_calls} == {model.route.removeprefix("direct:")}
    instruction = provider_calls[0]["request"]["messages"][1]["content"]
    assert ("Profiling guidance:" in instruction) == (guidance is not None)
    assert "previous-project-action" not in memory.actions
    distillation_messages = provider_calls[-1]["request"]["messages"]
    prior_messages = provider_calls[-2]["request"]["messages"]
    assert distillation_messages[:len(prior_messages)] == prior_messages
    assert distillation_messages[len(prior_messages)]["content"] == '{"submit":{}}'
    assert json.loads(
        distillation_messages[len(prior_messages) + 1]["content"]
    )["host"]["event"] == "submitted"
    if expected_category == "performance":
        study = next(
            digest for digest in memory.admitted_artifact_sha256
            if digest in distillation_messages[-1]["content"]
            and memory.lessons[0].evidence_artifact_sha256 == (digest,)
        )
        retained = ContentAddressedArtifacts(
            next(memory_path.glob("provider-attempts/attempt-*/artifacts"))
        )
        study_payload = retained.read_json(study)
        assert study_payload["timing"]
        context = json.loads(distillation_messages[-1]["content"].rsplit("\n", 1)[-1])
        assert context["artifact_payloads"][study]["payload"] == study_payload
        assert context["host_summaries"][0]["category"] == "host-verification"


def test_project_instruction_uses_registered_runtime_dimensions():
    proposal = next(
        item for item in DEFAULT_PROPOSALS
        if item.family.value == "padded-multitile"
    )
    runtime = Phase1ProjectRuntime.from_proposal(proposal)
    request = SimpleNamespace(
        guide_text="guide", proposal=proposal, memory_context="memory"
    )

    instruction = ProductionProjectExecutor._instruction(
        request, runtime, None, None
    )

    assert "logical length N = 400" in instruction
    assert "zero-pads inputs to P = 448" in instruction
    assert "first 400 elements" in instruction
    assert "all 448 output elements" in instruction


def test_study_attempt_identity_is_retry_scoped_not_scientific():
    proposal = next(
        item for item in DEFAULT_PROPOSALS
        if item.evidence_preset.value == "msprof-guided"
    )
    cell = SimpleNamespace(cell_id="a5-cell-" + "a" * 54)
    first = SimpleNamespace(
        cell=cell, ordinal=8, proposal=proposal,
    )
    retry = SimpleNamespace(
        cell=cell, ordinal=8, proposal=proposal,
    )
    first_trial = SimpleNamespace(
        verified=SimpleNamespace(attempt_id="candidate-1-final"),
    )
    retry_trial = SimpleNamespace(
        verified=SimpleNamespace(attempt_id="candidate-2-final"),
    )

    first_id = ProductionProjectExecutor._study_attempt_id(first, first_trial)
    retry_id = ProductionProjectExecutor._study_attempt_id(retry, retry_trial)

    assert first_id != retry_id
    assert first.proposal == retry.proposal
    assert first_id.rsplit("-", 1)[0] == retry_id.rsplit("-", 1)[0]


def test_cli_plan_run_resume_and_report_use_the_bound_dependencies(tmp_path, capsys):
    gate_root = tmp_path / "gate"
    gate_root.mkdir()
    guide_path, guide, report = _guide_and_report(gate_root)
    state = tmp_path / "gate.json"
    from benchmarks.a5kernels.phase1_live import write_gate_report
    write_gate_report(report, state)
    root = tmp_path / "run"
    shared = ["--gate", str(state), "--guide", str(guide_path), "--root", str(root)]

    assert run_a5_phase1.main(["plan", *shared]) == 0
    assert json.loads(capsys.readouterr().out)["guide"] == guide.as_dict()
    calls = []
    dependencies = _dependencies(tmp_path / "dependencies", calls)

    def builder(_paths, **_kwargs):
        return dependencies

    runtime = [
        *shared,
        "--tla-root", str(tmp_path),
        "--profiling-skill-root", str(tmp_path),
        "--catlass-source", "/remote/catlass",
        "--bge-cache", str(tmp_path),
        "--kdb", str(tmp_path / "kdb.sqlite"),
        "--collection", "catlass",
        "--device", "0",
    ]
    assert run_a5_phase1.main(["run", *runtime], dependency_builder=builder) == 0
    completed = json.loads(capsys.readouterr().out)
    assert completed["status"] == "complete-with-pending-treatments"
    assert len(completed["pending_treatment_cells"]) == 4
    assert len(completed["records"]) == 8
    assert len(calls) == 64

    smoke_root = tmp_path / "smoke"
    smoke_shared = [
        "--gate", str(state), "--guide", str(guide_path),
        "--root", str(smoke_root), "--one-project-smoke",
    ]
    smoke_runtime = [*smoke_shared, *runtime[len(shared):]]
    smoke_calls = []
    smoke_dependencies = _dependencies(tmp_path / "smoke-dependencies", smoke_calls)

    def smoke_builder(_paths, **_kwargs):
        return smoke_dependencies

    assert run_a5_phase1.main(
        ["run", *smoke_runtime], dependency_builder=smoke_builder
    ) == 0
    smoke = json.loads(capsys.readouterr().out)
    assert smoke["status"] == "complete-with-pending-treatments"
    assert len(smoke["pending_treatment_cells"]) == 4
    assert len(smoke["records"]) == len(smoke_calls) == 8
    assert {call["ordinal"] for call in smoke_calls} == {1}

    assert run_a5_phase1.main(
        ["resume", *smoke_runtime], dependency_builder=smoke_builder
    ) == 0
    assert json.loads(capsys.readouterr().out) == smoke
    assert len(smoke_calls) == 8

    assert run_a5_phase1.main(["resume", *runtime], dependency_builder=builder) == 0
    assert json.loads(capsys.readouterr().out) == completed
    assert len(calls) == 64
    assert run_a5_phase1.main(["report", *shared]) == 0
    assert json.loads(capsys.readouterr().out) == completed


def test_cli_retries_live_dependency_preflight(tmp_path, capsys):
    gate_root = tmp_path / "gate"
    gate_root.mkdir()
    guide_path, _guide, report = _guide_and_report(gate_root)
    state = tmp_path / "gate.json"
    from benchmarks.a5kernels.phase1_live import write_gate_report
    write_gate_report(report, state)
    calls = []
    sleeps = []
    dependencies = _dependencies(tmp_path / "dependencies", [])

    def builder(_paths, **_kwargs):
        calls.append(1)
        if len(calls) < 3:
            from benchmarks.a5kernels.bz import RuntimeUnavailableError
            raise RuntimeUnavailableError("temporary BZ outage")
        return dependencies

    runtime = [
        "--gate", str(state), "--guide", str(guide_path),
        "--root", str(tmp_path / "run"), "--one-project-smoke",
        "--tla-root", str(tmp_path),
        "--profiling-skill-root", str(tmp_path),
        "--catlass-source", "/remote/catlass",
        "--bge-cache", str(tmp_path),
        "--kdb", str(tmp_path / "kdb.sqlite"),
        "--collection", "catlass", "--device", "0",
    ]

    assert run_a5_phase1.main(
        ["run", *runtime], dependency_builder=builder, sleeper=sleeps.append,
    ) == 0
    assert len(calls) == 3
    assert sleeps == [120, 120]
    assert json.loads(capsys.readouterr().out)["status"] == (
        "complete-with-pending-treatments"
    )
