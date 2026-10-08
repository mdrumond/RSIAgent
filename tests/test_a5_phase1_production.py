from __future__ import annotations

import json

import pytest

import run_a5_phase1
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
    build_phase1_live_dependencies,
    load_direct_environment,
)
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
            stdout="A5KERNEL_NAME=vector_add__kernel0\n",
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
    "model,profiling,guidance,expected_commands",
    [
        (MODEL_IDENTITIES[0], ProfilingGuidance.WITHOUT_GUIDANCE, None, 3),
        (MODEL_IDENTITIES[1], ProfilingGuidance.WITH_GUIDANCE, A5_PROFILING_GUIDANCE, 5),
    ],
)
def test_executor_binds_direct_model_and_profiling_treatment(
    tmp_path, model, profiling, guidance, expected_commands,
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
        return DirectCompletion(next(replies), 2)

    profile_backend = TrialProfileBackend()
    executor = ProductionProjectExecutor(
        ProjectExecution(),
        profile_backend,
        DirectCompletionProvider(ENVIRONMENT, transport=transport),
        device=0,
    )
    proposal = DEFAULT_PROPOSALS[0]
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

    memory = executor(
        request,
        Phase1ProjectRuntime.from_proposal(proposal),
        knowledge,
        guidance,
        EvidenceLedger(tmp_path / "evidence.jsonl"),
    )

    assert memory.host_facts[0].category == "host-verification"
    assert len(profile_backend.commands) == expected_commands
    assert {call["base_url"] for call in provider_calls} == {model.route.removeprefix("direct:")}
    instruction = provider_calls[0]["request"]["messages"][1]["content"]
    assert ("Profiling guidance:" in instruction) == (guidance is not None)


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
    assert completed["status"] == "complete"
    assert len(completed["records"]) == 8
    assert len(calls) == 64

    assert run_a5_phase1.main(["resume", *runtime], dependency_builder=builder) == 0
    assert json.loads(capsys.readouterr().out) == completed
    assert len(calls) == 64
    assert run_a5_phase1.main(["report", *shared]) == 0
    assert json.loads(capsys.readouterr().out) == completed
