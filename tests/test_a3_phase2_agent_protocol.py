from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.a3_experiments import ProgrammingLevel, build_a3_experiment_plan
from benchmarks.a3kernels.phase1_evidence import EvidenceLedger, canonical_digest
from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
from benchmarks.a3kernels.phase2_protocol import (
    A3Phase2Adapter,
    A3Phase2BudgetStop,
    A3Phase2Hooks,
    A3Phase2Identity,
    A3Phase2InfrastructureError,
    A3Phase2ProtocolError,
    CurriculumDecision,
    GroundedLearning,
    GroundedTarget,
    OwnedAgent,
    PracticeResult,
)
from core.self_evolving_loop import TargetVerdict


def _identity() -> A3Phase2Identity:
    cell = next(
        cell for cell in build_a3_experiment_plan().cells
        if cell.programming_level is ProgrammingLevel.TILED
    )
    return A3Phase2Identity.from_cell(cell)


class FakeAgents:
    def __init__(self, identity, verdicts, decisions, *, fail_at=None):
        self.identity = identity
        self.verdicts = iter(verdicts)
        self.decisions = iter(decisions)
        self.fail_at = fail_at
        self.targets = []
        self.released = []
        self.learned = []
        self.practiced = []

    def owned(self, role, attempt, value=None):
        return OwnedAgent(self.identity, f"{role}-{attempt}", value or {})

    def environment(self, identity, attempt):
        assert identity == self.identity
        item = self.owned("environment", attempt, {"attempt": attempt})
        self.targets.append(item.instance_id)
        return item

    def verifier(self, identity, environment, attempt):
        assert environment.value["attempt"] == attempt
        return self.owned("verifier", attempt)

    def actor(self, identity, memory, attempt):
        return self.owned("actor", attempt, {"memory": dict(memory)})

    def work(self, identity, actor, verifier, environment, direction):
        attempt = environment.value["attempt"]
        return {"attempt": attempt, "direction": direction}

    def verify(self, identity, verifier, environment, direction, output):
        attempt = output["attempt"]
        if self.fail_at == attempt:
            raise A3Phase2InfrastructureError("remote observation unavailable")
        verdict = next(self.verdicts)
        return GroundedTarget(
            identity, verdict, f"grounded {verdict.value}",
            canonical_digest({"attempt": attempt, "verdict": verdict.value}),
        )

    def learn(self, identity, actor, environment, direction, output,
              verification, memory):
        self.learned.append(verification.verdict)
        updated = dict(memory)
        updated[f"target-{output['attempt']}"] = verification.verdict.value
        return GroundedLearning(
            identity, updated, f"lesson {output['attempt']}",
            verification.evidence_sha256,
        )

    def curriculum(self, identity, direction, verification, learning, memory,
                   evolution):
        decision = next(self.decisions)
        if decision is None:
            return CurriculumDecision.ready(identity, "memory is ready")
        return CurriculumDecision.practice(identity, decision, "focused contrast")

    def practice(self, identity, direction, proposal, memory, evolution):
        self.practiced.append(proposal.project_id)
        updated = dict(memory)
        updated[f"practice-{evolution}"] = proposal.project_id
        return PracticeResult(
            identity, proposal.project_id, updated,
            canonical_digest({"practice": proposal.project_id, "evolution": evolution}),
        )

    def release(self, identity, environment):
        self.released.append(environment.instance_id)

    def hooks(self):
        return A3Phase2Hooks(
            fresh_target_environment=self.environment,
            fresh_target_verifier=self.verifier,
            fresh_target_actor=self.actor,
            actor_work=self.work,
            verify_target=self.verify,
            learn_target=self.learn,
            curriculum_review=self.curriculum,
            run_practice=self.practice,
            release_target_environment=self.release,
        )


def _run(tmp_path, verdicts, decisions, **kwargs):
    identity = _identity()
    agents = FakeAgents(identity, verdicts, decisions, **kwargs)
    adapter = A3Phase2Adapter(identity, agents.hooks(), tmp_path / "events.jsonl")
    return adapter, agents


def test_default_curriculum_review_pass_ready_learns_grounded_result(tmp_path):
    adapter, agents = _run(tmp_path, [TargetVerdict.PASS], [None])

    result = adapter.run("improve tiled vector add", {})

    assert result.termination == "verifier_pass_curriculum_ready"
    assert result.target_attempts == 1 and result.practice_projects == 0
    assert result.verdict is TargetVerdict.PASS
    assert agents.learned == [TargetVerdict.PASS]
    events = EvidenceLedger(tmp_path / "events.jsonl").entries
    assert events[0].payload["stop_policy"] == "curriculum_review"
    assert events[-1].payload["event"] == "PHASE2_COMPLETED"


@pytest.mark.parametrize("final_verdict", [TargetVerdict.PASS, TargetVerdict.FAIL])
def test_terminal_head_binds_target_and_learning_evidence(tmp_path, final_verdict):
    if final_verdict is TargetVerdict.FAIL:
        adapter, _agents = _run(
            tmp_path, [TargetVerdict.FAIL] * 9, list(DEFAULT_PROPOSALS)
        )
    else:
        adapter, _agents = _run(tmp_path, [TargetVerdict.PASS], [None])

    result = adapter.run("improve tiled vector add", {})

    events = EvidenceLedger(tmp_path / "events.jsonl").entries
    target = [e for e in events if e.payload["event"] == "TARGET_GROUNDED"][-1]
    learning = [
        e for e in events if e.payload["event"] == "TARGET_LEARNING_GROUNDED"
    ][-1]
    terminal = events[-1]
    assert terminal.payload["target_evidence_sha256"] == target.payload[
        "evidence_sha256"
    ]
    assert terminal.payload["learning_grounding_sha256"] == learning.payload[
        "grounding_sha256"
    ]
    assert terminal.payload["verdict"] == final_verdict.value
    assert result.evidence_sha256 == terminal.entry_sha256


@pytest.mark.parametrize("first", [TargetVerdict.FAIL, TargetVerdict.PASS])
def test_focused_practice_always_forces_a_fresh_target(tmp_path, first):
    adapter, agents = _run(
        tmp_path, [first, TargetVerdict.PASS], [DEFAULT_PROPOSALS[1], None]
    )

    result = adapter.run("improve tiled vector add", {})

    assert result.target_attempts == 2 and result.practice_projects == 1
    assert agents.targets == ["environment-1", "environment-2"]
    assert agents.released == ["environment-1"]
    assert agents.practiced == [DEFAULT_PROPOSALS[1].project_id]
    assert agents.learned == [first, TargetVerdict.PASS]


def test_eight_practices_allow_only_the_ninth_final_target(tmp_path):
    proposals = list(DEFAULT_PROPOSALS)
    adapter, agents = _run(
        tmp_path,
        [TargetVerdict.FAIL] * 8 + [TargetVerdict.PASS],
        proposals,
    )

    result = adapter.run("improve tiled vector add", {})

    assert result.target_attempts == 9 and result.practice_projects == 8
    assert result.termination == "practice_budget_final_target"
    assert len(set(agents.targets)) == 9


def test_ninth_failed_target_stops_before_a_tenth_attempt(tmp_path):
    adapter, agents = _run(
        tmp_path, [TargetVerdict.FAIL] * 9, [None] * 9
    )

    with pytest.raises(A3Phase2BudgetStop, match="nine target attempts"):
        adapter.run("improve tiled vector add", {})

    assert len(agents.targets) == 9
    assert EvidenceLedger(tmp_path / "events.jsonl").entries[-1].payload[
        "event"
    ] == "PHASE2_BUDGET_STOPPED"


def test_identity_drift_is_rejected_and_environment_is_released(tmp_path):
    adapter, agents = _run(tmp_path, [TargetVerdict.PASS], [None])
    other = replace(agents.identity, cell_id="a3-cell-drift")
    hooks = replace(
        agents.hooks(),
        verify_target=lambda *args: GroundedTarget(
            other, TargetVerdict.PASS, "grounded PASS", "a" * 64
        ),
    )
    adapter = A3Phase2Adapter(agents.identity, hooks, tmp_path / "drift.jsonl")

    with pytest.raises(A3Phase2ProtocolError, match="identity drift"):
        adapter.run("improve tiled vector add", {})
    assert agents.released == ["environment-1"]


@pytest.mark.parametrize("role", ["verifier", "actor"])
def test_target_agent_instance_is_fresh_across_attempts(tmp_path, role):
    adapter, agents = _run(
        tmp_path,
        [TargetVerdict.FAIL, TargetVerdict.PASS],
        [DEFAULT_PROPOSALS[0], None],
    )
    hooks = agents.hooks()
    constant = lambda identity, *args: OwnedAgent(identity, f"reused-{role}", {})
    hooks = replace(hooks, **{f"fresh_target_{role}": constant})
    adapter = A3Phase2Adapter(agents.identity, hooks, tmp_path / f"{role}.jsonl")

    with pytest.raises(A3Phase2ProtocolError, match=f"{role} is not fresh"):
        adapter.run("improve tiled vector add", {})
    assert agents.targets == ["environment-1", "environment-2"]


def test_actor_and_verifier_instance_ids_cannot_overlap(tmp_path):
    adapter, agents = _run(tmp_path, [TargetVerdict.PASS], [None])
    hooks = replace(
        agents.hooks(),
        fresh_target_actor=lambda identity, memory, attempt: OwnedAgent(
            identity, f"verifier-{attempt}", {}
        ),
    )
    adapter = A3Phase2Adapter(agents.identity, hooks, tmp_path / "roles.jsonl")

    with pytest.raises(A3Phase2ProtocolError, match="Actor/Verifier role overlap"):
        adapter.run("improve tiled vector add", {})


def test_infrastructure_exception_remains_a_non_verdict(tmp_path):
    adapter, agents = _run(
        tmp_path, [TargetVerdict.PASS], [None], fail_at=1
    )

    with pytest.raises(A3Phase2InfrastructureError, match="observation unavailable"):
        adapter.run("improve tiled vector add", {})

    events = EvidenceLedger(tmp_path / "events.jsonl").entries
    assert events[-1].kind == "failure"
    assert events[-1].payload["event"] == "PHASE2_INFRASTRUCTURE_EXCEPTION"
    assert all(entry.payload.get("verdict") is None for entry in events)
    assert agents.learned == []


def test_phase2_requires_tiled_a3_ascend_c_identity():
    foundation = next(
        cell for cell in build_a3_experiment_plan().cells
        if cell.programming_level is ProgrammingLevel.FOUNDATION
    )
    with pytest.raises(ValueError, match="tiled"):
        A3Phase2Identity.from_cell(foundation)
