from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

import run_a3_phase2
from benchmarks.a3_experiments import BackendModel, KnowledgeMode, ProfilingGuidance
from benchmarks.a3_model_profiles import A3Completion
from benchmarks.a3kernels.candidate import A3CandidateBackend, CandidateCompilation
from benchmarks.a3kernels.live_composition import LiveDependencies
from benchmarks.a3kernels.phase1_evidence import canonical_bytes, canonical_digest
from benchmarks.a3kernels.phase1_protocol import (
    ExecutionReceipt, FailedEvidence, VerifiedResult, attest,
)
from benchmarks.a3kernels.phase2_protocol import (
    A3Phase2Identity, A3Phase2InfrastructureError,
)
from benchmarks.a3kernels.remote_candidate import _PendingObservation
from benchmarks.a3kernels.phase2_composition import Phase2Composition, qualification_cells
from benchmarks.a3kernels.phase2_live import Phase2LiveRunner, phase2_cells
from benchmarks.a3kernels.phase2_memory import Phase1Snapshot, phase2_cell_pairs


SOURCE = '''#include "kernel_operator.h"
extern "C" __global__ __aicore__ void vector_add(
 GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output,
 uint32_t count, uint32_t buffer_bytes) {}
'''


class Backend:
    def compile(self, plan, _workdir):
        return CandidateCompilation(
            plan, "a" * 64, "compiled", "",
            attest({"plan": plan, "library_sha256": "a" * 64,
                    "stdout": "compiled", "stderr": ""}),
        )

    def execute(self, compilation):
        plan = compilation.plan
        output = tuple(
            plan.input_a[i] + plan.input_b[i] if i < plan.logical_length else 0.0
            for i in range(plan.padded_length)
        )
        return VerifiedResult.from_receipt(
            plan,
            ExecutionReceipt(0, output, "verified", "",
                             f"bz-a3-1:{plan.execution_id}",
                             (("library_sha256", "a" * 64),)),
            max_abs_error=0.0,
        )


class Knowledge:
    def __init__(self, enabled, calls):
        self.enabled, self.calls = enabled, calls

    def query(self, query):
        self.calls.append((self.enabled, query.query))
        if not self.enabled:
            return ()
        citation = SimpleNamespace(
            collection="a3", path="guide.md", start_line=1, end_line=2,
            chunk_id="c" * 64, source_revision="d" * 64,
            content_sha256="e" * 64,
        )
        return (SimpleNamespace(citation=citation, text="Use tiled DataCopyPad."),)


def snapshot(tmp_path, cell):
    root = tmp_path / cell.cell_id
    root.mkdir(parents=True)
    memory = canonical_bytes({"memory": {"actions": ["phase1"]}}) + b"\n"
    terminal = b'{"status":"passed"}\n'
    (root / "phase1-memory.jsonl").write_bytes(memory)
    (root / "phase1-terminal.json").write_bytes(terminal)
    source = next(src for src, dest in phase2_cell_pairs() if dest == cell)
    body = {
        "schema": "a3-phase2-phase1-snapshot-v1", "target": "Ascend910B4",
        "language": "ascend-c", "phase1_root_sha256": "1" * 64,
        "source_cell_id": source.cell_id, "destination_cell_id": cell.cell_id,
        "model": cell.backend_model.value, "knowledge": cell.knowledge.value,
        "profiling": cell.profiling.value, "phase1_plan_fingerprint": "2" * 64,
        "trial_protocol_sha256": "3" * 64,
        "phase1_evidence_head_sha256": "4" * 64,
        "phase1_terminal_sha256": hashlib.sha256(terminal).hexdigest(),
        "phase1_memory_sha256": hashlib.sha256(memory).hexdigest(),
        "phase1_memory_entries": 1, "phase1_memory_head_sha256": "5" * 64,
        "phase1_lineage_id": "phase1-test",
    }
    manifest = {**body, "snapshot_id": canonical_digest(body)}
    (root / "snapshot.json").write_bytes(canonical_bytes(manifest) + b"\n")
    return Phase1Snapshot.open(root)


def live_dependencies(
    calls, knowledge_calls, *, query_kdb=False, bad_ready=False, backend=None,
):
    def actor_factory(cell, _proposal):
        profile_sha = __import__(
            "benchmarks.a3_model_profiles", fromlist=["load_a3_model_profile"]
        ).load_a3_model_profile(cell.backend_model).fingerprint
        count = 0

        def actor(_profile, prompt):
            nonlocal count
            count += 1
            calls.append((cell.backend_model, cell.knowledge, cell.profiling, prompt))
            if '"role":"curriculum"' in prompt:
                value = {"action": "ready", "reason": "host suite passed"}
                if bad_ready:
                    value["verdict"] = "PASS"
            elif query_kdb and cell.knowledge is KnowledgeMode.WITH_KDB and count == 1:
                value = {"action": "query", "query": "tiled A3 movement"}
            else:
                value = {"action": "source", "source": SOURCE}
            return A3Completion(json.dumps(value), 10, {"profile_sha256": profile_sha})
        return actor

    return LiveDependencies(
        actor_factory=actor_factory,
        candidate_factory=lambda *_: backend or Backend(),
        knowledge_factory=lambda cell, _paths: Knowledge(
            cell.knowledge is KnowledgeMode.WITH_KDB, knowledge_calls,
        ),
        profiler_factory=lambda *_: pytest.fail("qualification must not practice"),
        execution_profile="bz-a3-1",
    )


def test_argparse_exposes_qualification_all_cell_and_resume_commands():
    parser = run_a3_phase2._parser()
    common = [
        "--state-root", "/tmp/state", "--phase1-root", "/tmp/p1",
        "--validation-wrapper", "/tmp/catlass-validation.sh",
        "--cpl-remote", "/tmp/cpl-remote", "--profile", "bz-a3-1",
        "--remote-workspace", "/tmp/remote", "--physical-device", "4",
        "--acknowledge-execution", "I_ACCEPT_A3_PHASE2_EXECUTION",
    ]
    assert parser.parse_args(["qualify", *common]).command == "qualify"
    run = parser.parse_args(["run", *common, "--cell-id", phase2_cells()[0].cell_id])
    assert run.cell_id == [phase2_cells()[0].cell_id]
    assert parser.parse_args(["resume", *common]).command == "resume"


def test_two_provider_qualification_uses_host_verdicts(tmp_path):
    calls, knowledge_calls = [], []
    composition = Phase2Composition(
        config=SimpleNamespace(), dependencies=live_dependencies(calls, knowledge_calls),
        execution_profile="bz-a3-1",
    )
    results = []
    for cell in qualification_cells():
        snap = snapshot(tmp_path / "snapshots", cell)
        deps = composition.dependencies(cell, snap, tmp_path / "runs" / cell.cell_id)
        results.append(Phase2LiveRunner(
            root=tmp_path / "runs" / cell.cell_id, snapshot=snap,
            dependencies=deps, sleeper=lambda _seconds: None,
        ).run("write reliable tiled vector add"))

    assert [result.verdict.value for result in results] == ["PASS", "PASS"]
    assert {item[0] for item in calls} == {
        BackendModel.GPT_5_6_SOL, BackendModel.DEEPSEEK_FLASH,
    }
    assert knowledge_calls == []


def test_all_treatments_keep_kdb_and_profiling_identity_isolated(tmp_path):
    calls, knowledge_calls = [], []
    composition = Phase2Composition(
        config=SimpleNamespace(),
        dependencies=live_dependencies(calls, knowledge_calls, query_kdb=True),
        execution_profile="bz-a3-1",
    )
    for cell in phase2_cells():
        snap = snapshot(tmp_path / "snapshots", cell)
        deps = composition.dependencies(cell, snap, tmp_path / "runs" / cell.cell_id)
        source = deps.target_source(
            A3Phase2Identity.from_cell(cell),
            {"phase1": {"entries": 8}}, 1,
        )
        assert source == SOURCE

    assert len(knowledge_calls) == 4
    assert all(enabled for enabled, _query in knowledge_calls)
    for model, knowledge, profiling, prompt in calls:
        assert model in BackendModel and knowledge in KnowledgeMode
        if '"role":"target-actor"' in prompt:
            assert profiling.value in prompt


def test_curriculum_agent_cannot_author_a_semantic_verdict(tmp_path):
    cell = qualification_cells()[0]
    calls, knowledge_calls = [], []
    composition = Phase2Composition(
        config=SimpleNamespace(),
        dependencies=live_dependencies(calls, knowledge_calls, bad_ready=True),
        execution_profile="bz-a3-1",
    )
    snap = snapshot(tmp_path / "snapshots", cell)
    runner = Phase2LiveRunner(
        root=tmp_path / "run", snapshot=snap,
        dependencies=composition.dependencies(cell, snap, tmp_path / "run"),
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(ValueError, match="curriculum action"):
        runner.run("target")


@pytest.mark.parametrize("pending", [False, True])
def test_candidate_infrastructure_never_becomes_semantic_fail(tmp_path, pending):
    class InfrastructureBackend:
        def compile(self, plan, _workdir):
            if pending:
                raise _PendingObservation("observe retained handle")
            return FailedEvidence.create(
                plan, stage="prepare", error_type="RuntimeError",
                detail="transport unavailable",
            )

        def execute(self, _compilation):
            pytest.fail("failed preparation cannot execute")

    cell = qualification_cells()[0]
    composition = Phase2Composition(
        config=SimpleNamespace(),
        dependencies=live_dependencies([], [], backend=InfrastructureBackend()),
        execution_profile="bz-a3-1",
    )
    snap = snapshot(tmp_path / "snapshots", cell)
    deps = composition.dependencies(cell, snap, tmp_path / "run")
    with pytest.raises(A3Phase2InfrastructureError, match="candidate|observation"):
        deps.target_verify(SOURCE, "request", "attempt", tmp_path / "target")


def test_cli_resume_reuses_bound_terminal_without_rerunning_cell(tmp_path, monkeypatch):
    cell = qualification_cells()[0]
    snap = snapshot(tmp_path / "snapshots", cell)
    source_root = tmp_path / "phase1-source"
    source_root.mkdir()
    calls = []

    class Config:
        def __init__(self, *args):
            self.state_root, self.validation_wrapper = args[:2]

        def preflight(self, _environment, *, cells):
            assert len(cells) == 1
            return {"ready": True}

    class Runner:
        def __init__(self, **_kwargs):
            calls.append("construct")

        def run(self, direction):
            calls.append(direction)
            return SimpleNamespace(
                identity=SimpleNamespace(cell_id=cell.cell_id),
                verdict=SimpleNamespace(value="PASS"), report="host passed",
                target_attempts=1, practice_projects=0,
                termination="verifier_pass_curriculum_ready",
                evidence_sha256="f" * 64,
            )

    monkeypatch.setattr(run_a3_phase2, "Phase1Config", Config)
    monkeypatch.setattr(run_a3_phase2, "_bz_preflight", lambda *_: {})
    monkeypatch.setattr(run_a3_phase2, "bz_live_dependencies", lambda *_a, **_k: object())
    monkeypatch.setattr(
        run_a3_phase2, "Phase2Composition",
        lambda **_kwargs: SimpleNamespace(dependencies=lambda *_args: object()),
    )
    monkeypatch.setattr(run_a3_phase2, "_source_cell_root", lambda *_: source_root)
    monkeypatch.setattr(run_a3_phase2, "admit_phase1_snapshot", lambda *_: snap)
    monkeypatch.setattr(run_a3_phase2, "Phase2LiveRunner", Runner)
    args = SimpleNamespace(
        state_root=tmp_path / "state", validation_wrapper=tmp_path / "wrapper",
        embedding_cache=tmp_path, corpus_artifacts=tmp_path,
        knowledge_database=tmp_path / "db", knowledge_manifest=tmp_path / "manifest",
        cpl_remote="/tmp/cpl-remote", profile="bz-a3-1",
        remote_workspace="/tmp/remote", physical_device=4,
        phase1_root=[tmp_path / "phase1"], target_direction="target",
    )

    first = run_a3_phase2._run(args, (cell,))
    second = run_a3_phase2._run(args, (cell,))

    assert first == second
    assert calls == ["construct", "target"]
