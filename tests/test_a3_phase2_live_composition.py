from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

import run_a3_phase2
from benchmarks.a3_experiments import BackendModel, KnowledgeMode, ProfilingGuidance
from benchmarks.a3_model_profiles import A3Completion
from benchmarks.a3_model_profiles import A3ProviderResponseError
from benchmarks.a3kernels.candidate import A3CandidateBackend, CandidateCompilation
from benchmarks.a3kernels.live_composition import LiveDependencies
from benchmarks.a3kernels.phase1_evidence import canonical_bytes, canonical_digest
from benchmarks.a3kernels.phase1_protocol import (
    ExecutionReceipt, FailedEvidence, VerifiedResult, attest,
)
from benchmarks.a3kernels.phase2_protocol import (
    A3Phase2Identity, A3Phase2InfrastructureError, GroundedLearning,
    GroundedTarget,
)
from benchmarks.a3kernels.remote_candidate import _PendingObservation
from benchmarks.a3kernels.phase2_composition import Phase2Composition, qualification_cells
from benchmarks.a3kernels.phase2_live import (
    Phase2LiveRunner, PracticeExecution, phase2_cells,
)
from benchmarks.a3kernels.phase2_memory import Phase1Snapshot, phase2_cell_pairs
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a3kernels.phase2_target import (
    PHASE2_TARGET_CASES, Phase2TargetEvidence,
)
from core.self_evolving_loop import TargetVerdict


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
    actor_error=None, practice_sequence=(), candidate_sources=(),
):
    def actor_factory(cell, _proposal):
        profile_sha = __import__(
            "benchmarks.a3_model_profiles", fromlist=["load_a3_model_profile"]
        ).load_a3_model_profile(cell.backend_model).fingerprint
        count = 0
        source_count = 0

        def actor(_profile, prompt):
            nonlocal count, source_count
            count += 1
            calls.append((cell.backend_model, cell.knowledge, cell.profiling, prompt))
            if actor_error is not None:
                raise actor_error
            if '"role":"curriculum"' in prompt:
                completed = sum(
                    item.get("kind") == "practice"
                    for item in json.loads(prompt)["memory"]["phase2"]["learning"]
                )
                value = (
                    {
                        "action": "practice", "reason": "close the next evidence gap",
                        "proposal": practice_sequence[completed].as_dict(),
                    }
                    if completed < len(practice_sequence)
                    else {"action": "ready", "reason": "host suite passed"}
                )
                if bad_ready:
                    value["verdict"] = "PASS"
            elif query_kdb and cell.knowledge is KnowledgeMode.WITH_KDB and count == 1:
                value = {"action": "query", "query": "tiled A3 movement"}
            else:
                source = (
                    candidate_sources[source_count]
                    if source_count < len(candidate_sources)
                    else SOURCE
                )
                source_count += 1
                value = {"action": "source", "source": source}
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


def test_cli_persists_exact_pass_qualification_gate_and_fail_is_nonzero(
    tmp_path, monkeypatch,
):
    common = [
        "--state-root", str(tmp_path / "state"),
        "--phase1-root", str(tmp_path / "phase1"),
        "--validation-wrapper", str(tmp_path / "wrapper"),
        "--cpl-remote", "/tmp/cpl-remote", "--profile", "bz-a3-1",
        "--remote-workspace", "/tmp/remote", "--physical-device", "4",
        "--acknowledge-execution", "I_ACCEPT_A3_PHASE2_EXECUTION",
        "--target-direction", "canonical target",
    ]
    records = tuple(
        {
            "cell_id": cell.cell_id, "snapshot_id": str(index) * 64,
            "execution_profile": "bz-a3-1", "verdict": "PASS",
            "evidence_sha256": chr(96 + index) * 64,
            "target_direction_sha256": run_a3_phase2._target_direction_sha256(
                "canonical target"
            ),
        }
        for index, cell in enumerate(qualification_cells(), 1)
    )
    monkeypatch.setattr(run_a3_phase2, "_run", lambda _args, _cells: records)

    assert run_a3_phase2.main(["qualify", *common]) == 0
    manifest = json.loads((tmp_path / "state" / "qualification.json").read_text())
    assert manifest["schema"] == "a3-phase2-qualification-v1"
    assert [item["verdict"] for item in manifest["cells"]] == ["PASS", "PASS"]

    calls = []
    monkeypatch.setattr(
        run_a3_phase2, "_run", lambda _args, cells: calls.append(cells) or (),
    )
    assert run_a3_phase2.main(["run", *common]) == 0
    assert calls == [phase2_cells()]

    failed = ({**records[0], "verdict": "FAIL"}, records[1])
    monkeypatch.setattr(run_a3_phase2, "_run", lambda _args, _cells: failed)
    assert run_a3_phase2.main(["qualify", *common]) == 1
    assert not (tmp_path / "state" / "qualification.json").exists()


def test_run_rejects_missing_or_nonexact_qualification_gate(tmp_path, monkeypatch):
    args = SimpleNamespace(
        state_root=tmp_path, profile="bz-a3-1", target_direction="target",
    )
    with pytest.raises(ValueError, match="qualification"):
        run_a3_phase2._require_qualification(args)
    body = {
        "schema": "a3-phase2-qualification-v1", "execution_profile": "bz-a3-1",
        "target_direction_sha256": run_a3_phase2._target_direction_sha256("target"),
        "cells": [
            {
                "cell_id": cell.cell_id, "snapshot_id": "a" * 64,
                "verdict": "PASS", "evidence_sha256": "b" * 64,
            }
            for cell in qualification_cells()
        ],
    }
    (tmp_path / "qualification.json").write_bytes(canonical_bytes(body) + b"\n")
    run_a3_phase2._require_qualification(args)
    body["cells"][1]["verdict"] = "FAIL"
    (tmp_path / "qualification.json").write_bytes(canonical_bytes(body) + b"\n")
    with pytest.raises(ValueError, match="qualification"):
        run_a3_phase2._require_qualification(args)


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


def test_curriculum_exposes_remaining_catalog_for_distinct_practices(tmp_path):
    cell = qualification_cells()[0]
    calls = []
    sequence = DEFAULT_PROPOSALS[:2]
    composition = Phase2Composition(
        config=SimpleNamespace(),
        dependencies=live_dependencies(calls, [], practice_sequence=sequence),
        execution_profile="bz-a3-1",
    )
    deps = composition.dependencies(cell, snapshot(tmp_path, cell), tmp_path / "run")
    identity = A3Phase2Identity.from_cell(cell)
    target = GroundedTarget(identity, TargetVerdict.FAIL, "host failed", "a" * 64)
    learning = GroundedLearning(identity, {}, "diagnostic", "a" * 64)
    empty = {"phase2": {"learning": []}}
    first = deps.curriculum(identity, target, learning, empty, 0)
    after_first = {"phase2": {"learning": [
        {"kind": "practice", "project_id": first.project.project_id}
    ]}}
    second = deps.curriculum(identity, target, learning, after_first, 1)

    assert first.project == sequence[0]
    assert second.project == sequence[1]
    payloads = [json.loads(item[3]) for item in calls]
    assert [row["project_id"] for row in payloads[0]["remaining_proposals"]] == [
        proposal.project_id for proposal in DEFAULT_PROPOSALS
    ]
    assert first.project.project_id not in {
        row["project_id"] for row in payloads[1]["remaining_proposals"]
    }


def test_practice_returns_bounded_evidence_linked_lesson(tmp_path, monkeypatch):
    cell = qualification_cells()[0]
    proposal = DEFAULT_PROPOSALS[4]
    evidence = "e" * 64
    candidate = "c" * 64

    class Wave:
        def __init__(self, *_args):
            pass

        def _stage(self, paths):
            paths.memory.parent.mkdir(parents=True, exist_ok=True)

    class Composition:
        def __init__(self, *_args, **_kwargs):
            pass

        def execute(self, _cell, paths):
            memory = {
                "source_revision": candidate,
                "actions": ["compile", "run", "time", "submit"],
                "host_facts": [
                    {
                        "category": "host-verification",
                        "statement": "candidate output passed host verification",
                        "evidence_sha256": evidence, "success": True,
                    },
                    {
                        "category": "profiling",
                        "statement": "host timing samples captured",
                        "evidence_sha256": "f" * 64, "success": True,
                    },
                ],
                "agent_interpretations": [{
                    "statement": "the upper-length knee needs a larger tile",
                    "supports": [evidence],
                }],
            }
            paths.memory.write_bytes(canonical_bytes({"memory": memory}) + b"\n")
            return {"status": "passed", "terminal_reason": "completed"}

    monkeypatch.setattr("benchmarks.a3kernels.phase2_composition.Phase1Wave", Wave)
    monkeypatch.setattr("benchmarks.a3kernels.phase2_composition.LiveComposition", Composition)
    composition = Phase2Composition(
        config=SimpleNamespace(), dependencies=live_dependencies([], []),
        execution_profile="bz-a3-1",
    )
    deps = composition.dependencies(cell, snapshot(tmp_path / "snap", cell), tmp_path / "run")
    completed = deps.practice(A3Phase2Identity.from_cell(cell), proposal, {}, 1)
    lesson = json.loads(completed.statement)

    assert completed.evidence_sha256 == evidence
    assert lesson["schema"] == "a3-phase2-practice-lesson-v1"
    assert lesson["primary_evidence_sha256"] == evidence
    assert lesson["evidence"][0]["evidence_sha256"] == evidence
    assert lesson["evidence"][1]["category"] == "profiling"
    assert "knee" in lesson["interpretations"][0]["statement"]
    assert len(completed.statement) <= 4096


@pytest.mark.parametrize("error", [TimeoutError("timeout"), ConnectionError("reset")])
def test_provider_transport_failures_receive_engine_retry_budget(tmp_path, error):
    cell = qualification_cells()[0]
    calls, sleeps = [], []
    composition = Phase2Composition(
        config=SimpleNamespace(),
        dependencies=live_dependencies(calls, [], actor_error=error),
        execution_profile="bz-a3-1",
    )
    snap = snapshot(tmp_path / "snap", cell)
    runner = Phase2LiveRunner(
        root=tmp_path / "run", snapshot=snap,
        dependencies=composition.dependencies(cell, snap, tmp_path / "run"),
        sleeper=sleeps.append,
    )
    with pytest.raises(A3Phase2InfrastructureError, match="provider"):
        runner.run("target")
    assert len(calls) == 4
    assert sleeps == [120, 120, 120]


def test_provider_schema_failure_is_not_infrastructure(tmp_path):
    cell = qualification_cells()[0]
    composition = Phase2Composition(
        config=SimpleNamespace(),
        dependencies=live_dependencies(
            [], [], actor_error=A3ProviderResponseError("missing-response-fields"),
        ),
        execution_profile="bz-a3-1",
    )
    snap = snapshot(tmp_path / "snap", cell)
    deps = composition.dependencies(cell, snap, tmp_path / "run")
    with pytest.raises(A3ProviderResponseError):
        deps.target_source(A3Phase2Identity.from_cell(cell), {}, 1)


def test_composed_target_source_still_rejects_non_string_source(tmp_path):
    cell = qualification_cells()[0]
    composition = Phase2Composition(
        config=SimpleNamespace(),
        dependencies=live_dependencies([], [], candidate_sources=(None,)),
        execution_profile="bz-a3-1",
    )
    snap = snapshot(tmp_path / "snapshot", cell)
    deps = composition.dependencies(cell, snap, tmp_path / "run")

    with pytest.raises(ValueError, match="must be a string"):
        deps.target_source(A3Phase2Identity.from_cell(cell), {}, 1)


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


@pytest.mark.parametrize(
    ("invalid", "detail"),
    [
        (
            SOURCE.replace("vector_add", "add_custom"),
            "candidate must contain the exact exported vector_add signature once",
        ),
        ("   \n", "candidate must contain the exact exported vector_add signature"),
    ],
)
def test_composed_invalid_candidate_is_grounded_before_corrected_retry(
    tmp_path, invalid, detail,
):
    class CountingBackend(Backend):
        def __init__(self):
            self.compiled = 0

        def compile(self, plan, workdir):
            self.compiled += 1
            return super().compile(plan, workdir)

    backend = CountingBackend()
    cell = qualification_cells()[0]
    calls = []
    composition = Phase2Composition(
        config=SimpleNamespace(),
        dependencies=live_dependencies(
            calls, [], backend=backend,
            practice_sequence=DEFAULT_PROPOSALS[:1],
            candidate_sources=(invalid, SOURCE),
        ),
        execution_profile="bz-a3-1",
    )
    snap = snapshot(tmp_path / "snapshot", cell)
    deps = composition.dependencies(cell, snap, tmp_path / "run")
    deps = replace(
        deps,
        practice=lambda _identity, proposal, *_: PracticeExecution(
            proposal.project_id, "c" * 64, "d" * 64, True,
            "reviewed the exported source contract",
        ),
    )

    result = Phase2LiveRunner(
        root=tmp_path / "run", snapshot=snap, dependencies=deps,
        sleeper=lambda _seconds: None,
    ).run("target")

    assert result.verdict is TargetVerdict.PASS
    assert result.target_attempts == 2
    assert backend.compiled == len(PHASE2_TARGET_CASES)
    learning = [
        json.loads(line)["learning"]
        for line in (tmp_path / "run" / "learning.jsonl").read_text().splitlines()
    ]
    admission = json.loads(learning[0]["statement"])
    assert admission["verdict"] == "FAIL"
    assert admission["failed_cases"] == [{
        "detail": detail,
        "error_type": "CandidateSourceAdmissionError",
        "stage": "admission",
    }]
    target_prompts = [
        json.loads(prompt) for *_identity, prompt in calls
        if '"role":"target-actor"' in prompt
    ]
    assert admission == json.loads(
        target_prompts[1]["memory"]["phase2"]["learning"][0]["statement"]
    )


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
    args.target_direction = "different target"
    with pytest.raises(ValueError, match="target direction"):
        run_a3_phase2._run(args, (cell,))


def test_cli_resume_reconciles_interrupted_evidence_without_abandonment(
    tmp_path, monkeypatch, capsys,
):
    cell = qualification_cells()[0]
    snap = snapshot(tmp_path / "snapshots", cell)
    source_root = tmp_path / "phase1-source"
    source_root.mkdir()
    calls = []

    class CountingBackend(Backend):
        def execute(self, compilation):
            calls.append("execute")
            return super().execute(compilation)

    class Config:
        def __init__(self, *args):
            self.state_root, self.validation_wrapper = args[:2]

        def preflight(self, _environment, *, cells):
            assert cells == (next(
                source for source, destination in phase2_cell_pairs()
                if destination == cell
            ),)
            return {"ready": True}

    live = live_dependencies(calls, [], backend=CountingBackend())
    monkeypatch.setattr(run_a3_phase2, "Phase1Config", Config)
    monkeypatch.setattr(run_a3_phase2, "_bz_preflight", lambda *_: {})
    monkeypatch.setattr(run_a3_phase2, "bz_live_dependencies", lambda *_a, **_k: live)
    monkeypatch.setattr(run_a3_phase2, "_source_cell_root", lambda *_: source_root)
    monkeypatch.setattr(run_a3_phase2, "admit_phase1_snapshot", lambda *_: snap)

    state_root = tmp_path / "state"
    direction = "resume exact interrupted target"
    gate = {
        "schema": "a3-phase2-qualification-v1",
        "execution_profile": "bz-a3-1",
        "target_direction_sha256": run_a3_phase2._target_direction_sha256(direction),
        "cells": [
            {
                "cell_id": item.cell_id, "snapshot_id": "a" * 64,
                "verdict": "PASS", "evidence_sha256": "b" * 64,
            }
            for item in qualification_cells()
        ],
    }
    state_root.mkdir()
    (state_root / "qualification.json").write_bytes(canonical_bytes(gate) + b"\n")
    argv = [
        "resume", "--state-root", str(state_root),
        "--phase1-root", str(tmp_path / "phase1"),
        "--validation-wrapper", str(tmp_path / "wrapper"),
        "--cpl-remote", "/tmp/cpl-remote", "--profile", "bz-a3-1",
        "--remote-workspace", "/tmp/remote", "--physical-device", "4",
        "--acknowledge-execution", "I_ACCEPT_A3_PHASE2_EXECUTION",
        "--target-direction", direction, "--cell-id", cell.cell_id,
    ]
    original_write = Phase2TargetEvidence.write
    interrupted = False

    def crash_after_write(self, path):
        nonlocal interrupted
        original_write(self, path)
        if not interrupted:
            interrupted = True
            raise RuntimeError("CLI interrupted after target evidence")

    monkeypatch.setattr(Phase2TargetEvidence, "write", crash_after_write)
    with pytest.raises(RuntimeError, match="CLI interrupted"):
        run_a3_phase2.main(argv)
    assert not (state_root / "cells" / cell.cell_id / "terminal.json").exists()
    executions_after_interrupt = calls.count("execute")

    assert run_a3_phase2.main(argv) == 0
    capsys.readouterr()
    root = state_root / "cells" / cell.cell_id
    targets = tuple((root / "targets").iterdir())
    evidence = [json.loads(line) for line in (root / "evidence.jsonl").read_text().splitlines()]

    assert len(targets) == 1
    assert (targets[0] / "candidate.ascendc").is_file()
    assert (targets[0] / "target-evidence.json").is_file()
    assert executions_after_interrupt == 7
    assert calls.count("execute") == executions_after_interrupt
    assert len([item for item in calls if isinstance(item, tuple)]) == 2
    assert len((root / "learning.jsonl").read_text().splitlines()) == 1
    assert len((root / "authority.jsonl").read_text().splitlines()) == 1
    assert len({item["entry_sha256"] for item in evidence}) == len(evidence)
    assert "abandon" not in json.dumps(evidence).lower()
    assert json.loads((root / "run-state.json").read_text())["phase"] == "complete"
    assert json.loads((root / "terminal.json").read_text())["verdict"] == "PASS"
