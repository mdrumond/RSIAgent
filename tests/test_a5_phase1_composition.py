from __future__ import annotations

import hashlib
import json

import pytest

from benchmarks.a5kernels.catlass_harness import HarnessResult
from benchmarks.a5kernels.evidence import canonical_digest
from benchmarks.a5kernels.knowledge import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    KnowledgeDB,
)
from benchmarks.a5kernels.knowledge_agent import (
    KnowledgeAgent,
    KnowledgeQuery,
    ProgressiveMemoryJournal,
)
from benchmarks.a5kernels.phase1_composition import (
    A5_PROFILING_GUIDANCE,
    CellDisposition,
    LiveDependencies,
    Phase1CellComposition,
)
from benchmarks.a5kernels.phase1_experiments import (
    PINNED_CATLASS_REVISION,
    admit_programming_guide,
)
from benchmarks.a5kernels.phase1_live import InfrastructureFailure, Phase1GateRunner
from benchmarks.a5kernels.phase1_memory import HostFact, Phase1ProjectMemory
from benchmarks.a5kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a5kernels.protocol import canonical_hash


class FakeEmbeddings:
    model = DEFAULT_EMBEDDING_MODEL
    revision = DEFAULT_EMBEDDING_REVISION

    def embed(self, texts):
        return [[float("vector" in text.lower()), 1.0] for text in texts]


def _knowledge_database(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir(parents=True)
    guide = sources / "guide.py"
    guide.write_text("vector add uses host verification\n" * 3, encoding="utf-8")
    database = KnowledgeDB(tmp_path / "knowledge.sqlite", FakeEmbeddings())
    database.index(sources, [guide], collection="catlass", language="catlass-dsl")
    database.close()
    return tmp_path / "knowledge.sqlite"


def _guide_and_report(tmp_path):
    path = tmp_path / "agent-guide.md"
    path.write_text("Pinned Catlass DSL programming guide.\n", encoding="utf-8")
    guide = admit_programming_guide(
        path, cpl_skills_revision="b" * 40,
        catlass_revision=PINNED_CATLASS_REVISION,
    )

    class Harness:
        def run(self, source, contract, attempt_id):
            fields = {
                "schema": "a5-catlass-harness-result-v1",
                "contract": contract.value,
                "attempt_id": attempt_id,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "source_fingerprint": "c" * 64,
                "catlass_revision": PINNED_CATLASS_REVISION,
                "execution_profile": "bz-a5",
                "device": 0,
                "session_handle": f"bz-a5:{attempt_id}",
                "host_verdict": "PASS",
                "exit_code": 0,
                "max_abs_error": 0.0,
                "mismatch_index": None,
                "mismatch_expected": None,
                "mismatch_actual": None,
                "output_sha256": "d" * 64,
                "retained_evidence_sha256": "e" * 64,
            }
            return HarnessResult(**fields, record_sha256=canonical_hash(fields))

    report = Phase1GateRunner(
        guide,
        Harness(),
        lambda agent, contract, revision, history: (
            f"# {agent.agent_id} {contract.value} {revision}\n"
        ),
        sleep=lambda _seconds: None,
    ).run()
    return path, guide, report


def _dependencies(tmp_path, calls, *, execute=None):
    database_path = _knowledge_database(tmp_path / "kdb")

    def knowledge_factory(cell, memory_path):
        database = KnowledgeDB(database_path, FakeEmbeddings())
        agent = KnowledgeAgent(
            enabled=True,
            database=database,
            collection="catlass",
            journal=ProgressiveMemoryJournal(memory_path / "knowledge.jsonl"),
        )
        database.close()
        return agent

    def default_execute(request, runtime, knowledge, profiling_guidance, ledger):
        results = knowledge.query(KnowledgeQuery("vector", limit=1))
        calls.append({
            "cell": request.cell.cell_id,
            "ordinal": request.ordinal,
            "model": request.cell.model.as_dict(),
            "knowledge": knowledge.enabled,
            "profiling_guidance": profiling_guidance,
            "completed_context": json.loads(request.memory_context)["completed_projects"],
            "guide_sha256": request.guide_sha256,
            "qualification_sha256": request.qualification_sha256,
            "runtime": runtime.as_dict(),
        })
        evidence = canonical_digest({
            "cell": request.cell.cell_id,
            "ordinal": request.ordinal,
            "proposal": request.proposal.as_dict(),
        })
        return Phase1ProjectMemory(
            request.proposal,
            source_revision=f"source-{request.ordinal}",
            actions=("write", "compile", "run", "submit"),
            host_facts=(HostFact("host-verification", "host PASS", evidence),),
            kdb_citations=tuple(result.citation for result in results),
        )

    return LiveDependencies(execute or default_execute, knowledge_factory)


def test_plan_is_deterministic_and_binds_gate_guide_profiles_and_retry_policy(tmp_path):
    guide_path, guide, report = _guide_and_report(tmp_path)
    composition = Phase1CellComposition(
        tmp_path / "run", guide_path, guide, report,
        _dependencies(tmp_path / "deps", []), sleeper=lambda _seconds: None,
    )

    first = composition.plan()
    assert first == composition.plan()
    assert first["guide"] == guide.as_dict()
    assert first["qualification_sha256"] == report.report_sha256
    assert len(first["cells"]) == 8
    assert len(first["pending_treatment_cells"]) == 4
    assert {row["profiling"] for row in first["pending_treatment_cells"]} == {"new"}
    assert first["infrastructure_retry"] == {
        "max_retries": 3, "backoff_seconds": 120,
    }
    assert {cell["model"]["route"] for cell in first["cells"]} == {
        "direct:https://api.openai.com/v1",
        "direct:https://api.deepseek.com",
    }
    assert {cell["model"]["credential_env"] for cell in first["cells"]} == {
        "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
    }

    with pytest.raises(ValueError, match="qualification"):
        Phase1CellComposition(
            tmp_path / "missing", guide_path, guide, None,
            _dependencies(tmp_path / "deps-missing", []),
        )


def test_all_eight_cells_isolate_treatments_checkpoint_and_resume(tmp_path):
    guide_path, guide, report = _guide_and_report(tmp_path)
    calls = []
    composition = Phase1CellComposition(
        tmp_path / "run", guide_path, guide, report,
        _dependencies(tmp_path / "deps", calls), sleeper=lambda _seconds: None,
    )

    records = composition.run()
    assert len(records) == 8
    assert {record["status"] for record in records} == {"passed"}
    assert len(calls) == 8 * len(DEFAULT_PROPOSALS)
    assert sum(call["knowledge"] for call in calls) == 4 * len(DEFAULT_PROPOSALS)
    assert sum(call["profiling_guidance"] is not None for call in calls) == (
        4 * len(DEFAULT_PROPOSALS)
    )
    assert {
        call["profiling_guidance"] for call in calls
        if call["profiling_guidance"] is not None
    } == {A5_PROFILING_GUIDANCE}
    assert {call["completed_context"] for call in calls} == set(range(8))
    assert {call["guide_sha256"] for call in calls} == {guide.guide_sha256}
    assert {call["qualification_sha256"] for call in calls} == {
        report.report_sha256
    }

    call_count = len(calls)
    assert composition.resume() == records
    assert len(calls) == call_count
    summary = composition.report()
    assert summary["status"] == "complete-with-pending-treatments"
    assert len(summary["pending_treatment_cells"]) == 4
    assert summary["qualification_sha256"] == report.report_sha256
    assert [record["cell_id"] for record in summary["records"]] == [
        cell["cell_id"] for cell in composition.plan()["cells"]
    ]


def test_one_project_smoke_runs_only_first_registered_project_and_resumes(tmp_path):
    guide_path, guide, report = _guide_and_report(tmp_path)
    calls = []
    composition = Phase1CellComposition.one_project_smoke(
        tmp_path / "smoke", guide_path, guide, report,
        _dependencies(tmp_path / "deps", calls), sleeper=lambda _seconds: None,
    )

    projects = composition.plan()["projects"]
    assert len(projects) == 1
    assert projects[0]["family"] == DEFAULT_PROPOSALS[0].family.value
    records = composition.run()
    assert len(records) == 8
    assert len(calls) == 8
    assert {call["ordinal"] for call in calls} == {1}
    assert {call["runtime"]["family"] for call in calls} == {
        DEFAULT_PROPOSALS[0].family.value
    }

    assert composition.resume() == records
    assert len(calls) == 8


def test_pending_cells_are_reported_before_any_treatment_dependency_dispatch(tmp_path):
    guide_path, guide, report = _guide_and_report(tmp_path)
    calls = []
    base = _dependencies(tmp_path / "deps", calls)
    knowledge_calls = []

    def knowledge_factory(cell, memory_path):
        knowledge_calls.append(cell.cell_id)
        return base.knowledge_factory(cell, memory_path)

    def disposition(cell):
        if cell.profiling.value == "with-profiling-guidance":
            return CellDisposition.pending("new profiler is not ready")
        return CellDisposition.runnable()

    composition = Phase1CellComposition.one_project_smoke(
        tmp_path / "smoke", guide_path, guide, report,
        LiveDependencies(base.execute, knowledge_factory),
        cell_disposition=disposition,
        sleeper=lambda _seconds: None,
    )

    records = composition.run()
    summary = composition.report()
    pending = summary["pending_treatment_cells"]
    assert len(records) == len(calls) == 4
    assert len(knowledge_calls) == 2
    assert summary["status"] == "complete-with-pending-treatments"
    assert len(pending) == 8
    assert {row["reason"] for row in pending} == {"new profiler is not ready"}
    assert {row["cell_id"] for row in pending}.isdisjoint(
        {call["cell"] for call in calls}
    )
    assert summary["pending_cell_ids"] == []


def test_registered_new_profiler_cells_are_pending_by_default(tmp_path):
    guide_path, guide, report = _guide_and_report(tmp_path)
    calls = []
    composition = Phase1CellComposition.one_project_smoke(
        tmp_path / "run",
        guide_path,
        guide,
        report,
        _dependencies(tmp_path / "deps", calls),
        sleeper=lambda _seconds: None,
    )

    records = composition.run()
    summary = composition.report()

    assert len(records) == 8
    assert len(calls) == 8
    assert sum(call["knowledge"] for call in calls) == 4
    assert len(summary["pending_treatment_cells"]) == 4
    assert {
        row["profiling"] for row in summary["pending_treatment_cells"]
    } == {"new"}


def test_project_boundary_resume_and_three_120_second_infrastructure_retries(tmp_path):
    guide_path, guide, report = _guide_and_report(tmp_path)
    calls, sleeps = [], []
    base = _dependencies(tmp_path / "deps", calls)
    failures = 0

    def interrupted(request, runtime, knowledge, profiling_guidance, ledger):
        nonlocal failures
        if request.ordinal == 3:
            failures += 1
            raise InfrastructureFailure("transient provider outage")
        return base.execute(request, runtime, knowledge, profiling_guidance, ledger)

    composition = Phase1CellComposition(
        tmp_path / "run", guide_path, guide, report,
        LiveDependencies(interrupted, base.knowledge_factory),
        sleeper=sleeps.append,
    )
    with pytest.raises(InfrastructureFailure):
        composition.run()
    assert failures == 4
    assert sleeps == [120, 120, 120]
    assert [call["ordinal"] for call in calls] == [1, 2]

    resumed_calls = []
    resumed = Phase1CellComposition(
        tmp_path / "run", guide_path, guide, report,
        _dependencies(tmp_path / "resumed-deps", resumed_calls),
        sleeper=lambda _seconds: None,
    )
    assert len(resumed.resume()) == 8
    assert resumed_calls[0]["ordinal"] == 3
    first_cell = resumed.plan()["cells"][0]["cell_id"]
    retry_rows = [
        row for row in resumed.evidence(first_cell).entries
        if row.payload.get("event") == "infrastructure-retry"
    ]
    assert [row.payload["retry"] for row in retry_rows] == [1, 2, 3]


def test_guide_drift_and_treatment_boundary_violations_fail_closed(tmp_path):
    guide_path, guide, report = _guide_and_report(tmp_path)
    guide_path.write_text("drifted guide\n", encoding="utf-8")
    with pytest.raises(ValueError, match="guide content"):
        Phase1CellComposition(
            tmp_path / "drift", guide_path, guide, report,
            _dependencies(tmp_path / "drift-deps", []),
        )

    guide_path.write_text("Pinned Catlass DSL programming guide.\n", encoding="utf-8")
    base = _dependencies(tmp_path / "deps", [])

    def leaking_execute(request, runtime, knowledge, profiling_guidance, ledger):
        memory = base.execute(request, runtime, knowledge, profiling_guidance, ledger)
        if not knowledge.enabled:
            enabled_agent = base.knowledge_factory(request.cell, request.memory_path)
            try:
                citation = enabled_agent.query(KnowledgeQuery("vector", 1))[0].citation
            finally:
                enabled_agent.close()
            return Phase1ProjectMemory(
                memory.proposal, memory.source_revision, memory.actions,
                memory.host_facts, kdb_citations=(citation,),
            )
        return memory

    composition = Phase1CellComposition(
        tmp_path / "leak", guide_path, guide, report,
        LiveDependencies(leaking_execute, base.knowledge_factory),
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(ValueError, match="KDB-off"):
        composition.run()
