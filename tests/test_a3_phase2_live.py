from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import pytest

from benchmarks.a3_experiments import BackendModel, KnowledgeMode, ProfilingGuidance
from benchmarks.a3kernels.candidate import A3CandidateBackend, CandidateCompilation
from benchmarks.a3kernels.live_composition import AuthoritativeResultStore
from benchmarks.a3kernels.phase1_evidence import canonical_bytes, canonical_digest
from benchmarks.a3kernels.phase1_protocol import ExecutionReceipt, VerifiedResult, attest
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a3kernels.phase2_live import (
    Phase2LiveDependencies, Phase2LiveRunner, PracticeExecution,
)
from benchmarks.a3kernels.phase2_memory import Phase1Snapshot, phase2_cell_pairs
from benchmarks.a3kernels.phase2_protocol import (
    A3Phase2InfrastructureError, CurriculumDecision,
)
from benchmarks.a3kernels.phase2_target import Phase2TargetEvidence, Phase2TargetVerifier


SOURCE = '''#include "kernel_operator.h"
extern "C" __global__ __aicore__ void vector_add(
 GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output,
 uint32_t count, uint32_t buffer_bytes) {}
'''


class _Backend:
    def __init__(self, passed=True):
        self.passed = passed

    def compile(self, plan, _workdir):
        body = {"plan": plan, "library_sha256": "a" * 64,
                "stdout": "compiled", "stderr": ""}
        return CandidateCompilation(plan, "a" * 64, "compiled", "", attest(body))

    def execute(self, compilation):
        plan = compilation.plan
        output = tuple(
            plan.input_a[i] + plan.input_b[i] if i < plan.logical_length else 0.0
            for i in range(plan.padded_length)
        )
        if not self.passed:
            output = (output[0] + 1.0, *output[1:])
        maximum = max(
            abs(output[i] - plan.input_a[i] - plan.input_b[i])
            for i in range(plan.logical_length)
        )
        return VerifiedResult.from_receipt(
            plan, ExecutionReceipt(
                0, output, "verified", "",
                f"bz-a3-1:{plan.execution_id}",
                (("library_sha256", "a" * 64),),
            ),
            max_abs_error=maximum,
        )


def _snapshot(tmp_path, cell):
    root = tmp_path / "snapshot"
    root.mkdir()
    memory = canonical_bytes({"memory": {"actions": ["phase1"]}}) + b"\n"
    terminal = b'{"status":"passed"}\n'
    (root / "phase1-memory.jsonl").write_bytes(memory)
    (root / "phase1-terminal.json").write_bytes(terminal)
    body = {
        "schema": "a3-phase2-phase1-snapshot-v1", "target": "Ascend910B4",
        "language": "ascend-c", "phase1_root_sha256": "1" * 64,
        "source_cell_id": next(source.cell_id for source, dest in phase2_cell_pairs()
                               if dest == cell),
        "destination_cell_id": cell.cell_id, "model": cell.backend_model.value,
        "knowledge": cell.knowledge.value, "profiling": cell.profiling.value,
        "phase1_plan_fingerprint": "2" * 64, "trial_protocol_sha256": "3" * 64,
        "phase1_evidence_head_sha256": "4" * 64,
        "phase1_terminal_sha256": hashlib.sha256(terminal).hexdigest(),
        "phase1_memory_sha256": hashlib.sha256(memory).hexdigest(),
        "phase1_memory_entries": 1, "phase1_memory_head_sha256": "5" * 64,
        "phase1_lineage_id": "phase1-test",
    }
    manifest = {**body, "snapshot_id": canonical_digest(body)}
    (root / "snapshot.json").write_bytes(canonical_bytes(manifest) + b"\n")
    return Phase1Snapshot.open(root)


def _cell(model):
    return next(dest for _source, dest in phase2_cell_pairs()
                if dest.backend_model is model
                and dest.knowledge is KnowledgeMode.WITHOUT_KDB
                and dest.profiling is ProfilingGuidance.WITHOUT_GUIDANCE)


@pytest.mark.parametrize("model", [BackendModel.GPT_5_6_SOL, BackendModel.DEEPSEEK_FLASH])
def test_qualification_cell_is_grounded_and_preserves_phase1_snapshot(tmp_path, model):
    cell = _cell(model)
    snapshot = _snapshot(tmp_path, cell)
    original = snapshot.memory_path.read_bytes()
    planner = A3CandidateBackend(lambda *_args, **_kwargs: None)
    verifier = Phase2TargetVerifier(_Backend(), plan_builder=planner.plan)
    calls = []

    dependencies = Phase2LiveDependencies(
        cell=cell, execution_profile="bz-a3-1",
        target_source=lambda identity, memory, attempt: (
            calls.append((identity.backend_model, memory["phase1"]["entries"], attempt))
            or SOURCE
        ),
        target_verify=lambda source, request, attempt, workdir: verifier.verify(
            source, request_id=request, attempt_id=attempt,
            execution_profile="bz-a3-1", workdir=workdir,
        ),
        curriculum=lambda identity, target, learning, memory, evolution:
            CurriculumDecision.ready(identity, "held-out suite passed"),
        practice=lambda *_args: pytest.fail("qualification must not practice"),
    )
    result = Phase2LiveRunner(
        root=tmp_path / "run", snapshot=snapshot, dependencies=dependencies,
        sleeper=lambda _seconds: None,
    ).run("write a tiled vector add")

    assert result.verdict.value == "PASS"
    assert result.target_attempts == 1 and result.practice_projects == 0
    assert calls == [(model, 1, 1)]
    assert snapshot.memory_path.read_bytes() == original
    records = (tmp_path / "run" / "learning.jsonl").read_text().splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["learning"]["kind"] == "target-verdict"


def test_treatments_must_exactly_match_snapshot_and_cell(tmp_path):
    cell = _cell(BackendModel.GPT_5_6_SOL)
    snapshot = _snapshot(tmp_path, cell)
    wrong = next(dest for _source, dest in phase2_cell_pairs()
                 if dest.backend_model is cell.backend_model
                 and dest.knowledge is KnowledgeMode.WITH_KDB
                 and dest.profiling is cell.profiling)
    with pytest.raises(ValueError, match="treatment"):
        Phase2LiveRunner(
            root=tmp_path / "run", snapshot=snapshot,
            dependencies=Phase2LiveDependencies(
                wrong, "bz-a3-1", lambda *_: SOURCE, lambda *_: None,
                lambda *_: None, lambda *_: None,
            ),
        )


def test_target_evidence_cannot_change_execution_profile(tmp_path):
    cell = _cell(BackendModel.GPT_5_6_SOL)
    snapshot = _snapshot(tmp_path, cell)
    planner = A3CandidateBackend(lambda *_args, **_kwargs: None)

    def verify(source, request, attempt, workdir):
        evidence = Phase2TargetVerifier(_Backend(), plan_builder=planner.plan).verify(
            source, request_id=request, attempt_id=attempt,
            execution_profile="bz-a3-1", workdir=workdir,
        )
        return Phase2TargetEvidence.create(
            request_id=evidence.request_id, attempt_id=evidence.attempt_id,
            execution_profile="bz-a3-2",
            candidate_sha256=evidence.candidate_sha256,
            source_fingerprint=evidence.source_fingerprint, cases=evidence.cases,
        )

    runner = Phase2LiveRunner(
        root=tmp_path / "run", snapshot=snapshot,
        dependencies=Phase2LiveDependencies(
            cell, "bz-a3-1", lambda *_: SOURCE, verify,
            lambda identity, *_: CurriculumDecision.ready(identity, "ready"),
            lambda *_: None,
        ),
    )
    with pytest.raises(ValueError, match="execution treatment"):
        runner.run("target")


def test_infrastructure_is_retried_without_becoming_a_target_verdict(tmp_path):
    cell = _cell(BackendModel.GPT_5_6_SOL)
    snapshot = _snapshot(tmp_path, cell)
    planner = A3CandidateBackend(lambda *_args, **_kwargs: None)
    verifier = Phase2TargetVerifier(_Backend(), plan_builder=planner.plan)
    attempts = []

    def verify(source, request, attempt, workdir):
        attempts.append((request, attempt))
        if len(attempts) < 3:
            raise A3Phase2InfrastructureError("retained operation unavailable")
        return verifier.verify(source, request_id=request, attempt_id=attempt,
                               execution_profile="bz-a3-1", workdir=workdir)

    deps = Phase2LiveDependencies(
        cell, "bz-a3-1", lambda *_: SOURCE, verify,
        lambda identity, *_: CurriculumDecision.ready(identity, "ready"),
        lambda *_: PracticeExecution(DEFAULT_PROPOSALS[0].project_id,
                                     "a" * 64, "b" * 64, True, "unused"),
    )
    waits = []
    result = Phase2LiveRunner(
        root=tmp_path / "run", snapshot=snapshot, dependencies=deps,
        sleeper=waits.append,
    ).run("target")

    assert result.verdict.value == "PASS"
    assert waits == [120, 120]
    assert len(set(attempts)) == 3
    learning = (tmp_path / "run" / "learning.jsonl").read_text().splitlines()
    assert len(learning) == 1


def test_failed_target_practices_once_then_uses_fresh_target(tmp_path):
    cell = _cell(BackendModel.GPT_5_6_SOL)
    snapshot = _snapshot(tmp_path, cell)
    planner = A3CandidateBackend(lambda *_args, **_kwargs: None)
    attempts = []
    practices = []

    def verify(source, request, attempt, workdir):
        attempts.append(attempt)
        backend = _Backend(passed=len(attempts) > 1)
        return Phase2TargetVerifier(backend, plan_builder=planner.plan).verify(
            source, request_id=request, attempt_id=attempt,
            execution_profile="bz-a3-1", workdir=workdir,
        )

    def curriculum(identity, target, _learning, _memory, _evolution):
        if target.verdict.value == "FAIL":
            return CurriculumDecision.practice(
                identity, DEFAULT_PROPOSALS[0], "repair tiled movement",
            )
        return CurriculumDecision.ready(identity, "host suite passed")

    def practice(_identity, proposal, _memory, _evolution):
        practices.append(proposal.project_id)
        return PracticeExecution(
            proposal.project_id, "c" * 64, "d" * 64, True,
            "host-verified focused practice",
        )

    result = Phase2LiveRunner(
        root=tmp_path / "run", snapshot=snapshot,
        dependencies=Phase2LiveDependencies(
            cell, "bz-a3-1", lambda *_: SOURCE, verify, curriculum, practice,
        ),
        sleeper=lambda _seconds: None,
    ).run("target")

    assert result.verdict.value == "PASS"
    assert result.target_attempts == 2 and result.practice_projects == 1
    assert len(set(attempts)) == 2
    assert practices == [DEFAULT_PROPOSALS[0].project_id]
    assert len((tmp_path / "run" / "learning.jsonl").read_text().splitlines()) == 3
