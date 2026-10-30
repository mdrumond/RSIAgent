"""Host-enforced behavior, including restart and clock accounting."""
from dataclasses import replace

import pytest

from rsi.contracts import BudgetLimits, Stage, StepEvidence
from rsi.runtime import ActorClock, Workflow, WorkflowError


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


def resolver(kind, ref, stage, operation):
    return ref == f"{kind}:{stage.value}:{operation}"


def evidence(stage, operation, source="v1"):
    ref = lambda kind: f"{kind}:{stage.value}:{operation}"
    return StepEvidence(stage, operation, skill_refs=(ref("skill"),),
                        retrieval_refs=(ref("retrieval"),),
                        retrospective_ref=ref("retrospective"), source_hash=source)


def finish(flow, passed=True, changed=None):
    stage = flow.stage
    operation = flow.begin()
    flow.dispatched(operation, "host-handle")
    flow.complete(evidence(stage, operation, flow.source_hash), passed=passed,
                  new_source_hash=changed)


def make(tmp_path, **kwargs):
    return Workflow(tmp_path / "state.json", resolver, source_hash="v1", **kwargs)


def test_order_failed_test_and_optimisation_retest(tmp_path):
    flow = make(tmp_path)
    for _ in range(4):
        finish(flow)
    assert flow.stage == Stage.TEST
    finish(flow, False)
    assert flow.stage == Stage.DEBUG
    finish(flow)
    assert flow.stage == Stage.IMPLEMENTATION
    finish(flow)
    finish(flow)
    finish(flow)
    assert flow.stage == Stage.OPTIMISATION
    finish(flow, changed="v2")
    assert flow.stage == Stage.TEST
    finish(flow)
    finish(flow)
    finish(flow)
    assert flow.stage == Stage.VALIDATION
    finish(flow)
    finish(flow)
    assert flow.outcome == "accepted"
    with pytest.raises(WorkflowError):
        flow.begin()


def test_forged_evidence_cannot_advance(tmp_path):
    flow = make(tmp_path)
    finish(flow)
    finish(flow)
    operation = flow.begin()
    good = evidence(flow.stage, operation)
    for bad in (replace(good, retrieval_refs=("invented",)),
                replace(good, retrospective_ref=None), replace(good, skill_refs=()),
                replace(good, stage=Stage.ACCEPTANCE)):
        with pytest.raises(WorkflowError):
            flow.complete(bad)
    assert flow.stage == Stage.DESIGN
    flow.complete(good)
    assert flow.stage == Stage.IMPLEMENTATION


def test_pending_persisted_before_dispatch_stop_and_reconcile(tmp_path):
    flow = make(tmp_path)
    operation = flow.begin()
    seen = make(tmp_path)
    assert seen.pending["id"] == operation
    with pytest.raises(WorkflowError):
        seen.begin()
    with pytest.raises(WorkflowError):
        seen.dispatched(operation, "duplicate")
    seen.stop()
    seen.resume()
    with pytest.raises(WorkflowError):
        seen.begin()
    seen.reconcile(operation, evidence(Stage.PREFLIGHT, operation))
    assert seen.stage == Stage.TASK_CONSTRUCTION
    assert seen.begin() != operation


def test_source_change_invalidates_verification(tmp_path):
    flow = make(tmp_path)
    for _ in range(8):
        finish(flow)
    assert flow.stage == Stage.ACCEPTANCE
    flow.source_changed("v2")
    assert flow.stage == Stage.TEST
    assert flow.verified_source is None


def test_caps_and_nullable_limits(tmp_path):
    flow = make(tmp_path, limits=BudgetLimits(execution_tries=1))
    operation = flow.begin(execution=True)
    flow.complete(evidence(Stage.PREFLIGHT, operation))
    with pytest.raises(WorkflowError):
        flow.begin(execution=True)
    other = Workflow(tmp_path / "other.json", resolver, source_hash="v1",
                     limits=BudgetLimits(None, None, None, None, None))
    for _ in range(4):
        finish(other)
    for _ in range(12):
        finish(other, False)
        finish(other)
        finish(other)
    assert other.stage == Stage.TEST


def test_interval_union_persistence_and_restart_reconciliation(tmp_path):
    clock = Clock()
    flow = make(tmp_path, clock=clock)
    a = flow.accounting.start("thinking")
    clock.now = 105
    transfer = flow.accounting.start("transfer")
    b = flow.accounting.start("compilation")
    clock.now = 110
    flow.accounting.end(a)
    clock.now = 115
    flow.accounting.end(b)
    flow.accounting.end(transfer)
    assert flow.accounting.seconds() == 15
    active = flow.accounting.start("profiling")
    clock.now = 120
    restarted = make(tmp_path, clock=clock)
    with pytest.raises(WorkflowError):
        restarted.begin()
    with pytest.raises(WorkflowError):
        restarted.accounting.reconcile(active, 119 - 465)
    with pytest.raises(WorkflowError):
        restarted.begin()
    restarted.accounting.reconcile(active, 119)
    assert restarted.accounting.seconds() == 19
    assert restarted.begin()


def test_time_budget_excludes_setup_but_includes_failure(tmp_path):
    clock = Clock()
    flow = make(tmp_path, clock=clock, limits=BudgetLimits(actor_seconds=5))
    setup = flow.accounting.start("setup")
    clock.now += 100
    flow.accounting.end(setup)
    a = flow.accounting.start("execution")
    clock.now += 5
    flow.accounting.end(a)
    with pytest.raises(WorkflowError):
        flow.begin()
    assert flow.accounting.seconds() == 5


def test_clock_regression_and_invalid_intervals_fail_closed(tmp_path):
    clock = Clock()
    flow = make(tmp_path, clock=clock)
    token = flow.accounting.start("debug")
    clock.now = 99
    with pytest.raises(WorkflowError):
        flow.accounting.end(token)
    with pytest.raises(WorkflowError):
        flow.begin()
    clock.now = 103
    flow.accounting.end(token)
    assert flow.accounting.seconds() == 3


def test_iteration_cap_redesign_and_stale_evidence(tmp_path):
    flow = make(tmp_path, limits=BudgetLimits(design_iterations=1, implementation_iterations=1))
    for _ in range(4):
        finish(flow)
    finish(flow, False)
    finish(flow)
    with pytest.raises(WorkflowError):
        flow.begin()
    other = Workflow(tmp_path / "redesign.json", resolver, source_hash="v1",
                     limits=BudgetLimits(design_iterations=1))
    for _ in range(5):
        finish(other)
    operation = other.begin()
    other.complete(evidence(Stage.DEBUG, operation), redesign=True)
    with pytest.raises(WorkflowError):
        other.begin()


def test_mutation_and_failed_validation_require_fresh_test(tmp_path):
    flow = make(tmp_path)
    for _ in range(7):
        finish(flow)
    assert flow.stage == Stage.VALIDATION
    finish(flow, False)
    assert flow.stage == Stage.DEBUG
    finish(flow, changed="v2")
    assert flow.stage == Stage.IMPLEMENTATION
    operation = flow.begin()
    with pytest.raises(WorkflowError):
        flow.complete(evidence(Stage.IMPLEMENTATION, operation, "v1"))
    flow.complete(evidence(Stage.IMPLEMENTATION, operation, "v2"))
    assert flow.stage == Stage.TEST


def test_wall_budget_includes_excluded_intervals_and_limits_survive_reload(tmp_path):
    clock = Clock()
    flow = make(tmp_path, clock=clock, limits=BudgetLimits(actor_seconds=None, wall_seconds=10))
    token = flow.accounting.start("transfer")
    clock.now += 10
    flow.accounting.end(token)
    restored = make(tmp_path, clock=clock, limits=BudgetLimits())
    assert restored.remaining() == {"actor_seconds": None, "wall_seconds": 0}
    with pytest.raises(WorkflowError):
        restored.begin()


def test_stop_prevents_dispatch_but_allows_receipt(tmp_path):
    flow = make(tmp_path)
    operation = flow.begin()
    flow.stop()
    with pytest.raises(WorkflowError):
        flow.dispatched(operation, "new-handle")
    flow.reconcile(operation, evidence(Stage.PREFLIGHT, operation))
    with pytest.raises(WorkflowError):
        flow.begin()
    flow.resume()
    assert flow.begin()


def test_crash_before_dispatch_requires_confirmed_absence(tmp_path):
    flow = make(tmp_path)
    operation = flow.begin()
    restored = make(tmp_path)
    with pytest.raises(WorkflowError):
        restored.reconcile(operation)
    restored.reconcile(operation, not_dispatched=True)
    assert restored.stage == Stage.PREFLIGHT
    assert restored.begin() != operation


def test_offline_source_change_cannot_reuse_acceptance_verification(tmp_path):
    flow = make(tmp_path)
    for _ in range(8):
        finish(flow)
    assert flow.stage == Stage.ACCEPTANCE
    assert flow.verified_source == "v1"
    with pytest.raises(WorkflowError):
        Workflow(tmp_path / "state.json", resolver, source_hash="v2")
    restored = make(tmp_path)
    restored.source_changed("v2")
    assert restored.stage == Stage.TEST
    assert restored.verified_source is None


def test_failed_debug_after_passing_test_returns_to_implementation(tmp_path):
    flow = make(tmp_path)
    for _ in range(5):
        finish(flow)
    assert flow.stage == Stage.DEBUG
    finish(flow, False)
    assert flow.stage == Stage.IMPLEMENTATION
    assert flow.state["tested"] is None
    finish(flow)
    assert flow.stage == Stage.TEST


@pytest.mark.parametrize("changed", [None, "v2"])
def test_failed_optimisation_returns_through_debug_to_implementation(tmp_path, changed):
    flow = make(tmp_path)
    for _ in range(6):
        finish(flow)
    assert flow.stage == Stage.OPTIMISATION
    finish(flow, False, changed=changed)
    assert flow.stage == Stage.DEBUG
    assert flow.state["tested"] is None
    finish(flow)
    assert flow.stage == Stage.IMPLEMENTATION
    finish(flow)
    assert flow.stage == Stage.TEST


@pytest.mark.parametrize("target", [Stage.DESIGN, Stage.IMPLEMENTATION, Stage.TEST])
def test_confirmed_absence_restores_cap_one_reservation(tmp_path, target):
    flow = make(tmp_path, limits=BudgetLimits(design_iterations=1,
                     implementation_iterations=1, execution_tries=1))
    while flow.stage != target:
        finish(flow)
    before = {key: flow.state[key] for key in ("design", "implementation", "executions")}
    operation = flow.begin(execution=True)
    restored = make(tmp_path)
    restored.reconcile(operation, not_dispatched=True)
    assert {key: restored.state[key] for key in before} == before
    assert restored.begin(execution=True) != operation


def test_absent_redesign_restores_previous_implementation_count(tmp_path):
    flow = make(tmp_path)
    for _ in range(5):
        finish(flow)
    operation = flow.begin()
    flow.complete(evidence(Stage.DEBUG, operation), redesign=True)
    assert flow.state["implementation"] == 1
    operation = flow.begin(execution=True)
    assert flow.state["implementation"] == 0
    restored = make(tmp_path)
    restored.reconcile(operation, not_dispatched=True)
    assert restored.state["implementation"] == 1
    assert restored.state["design"] == 1
    assert restored.state["executions"] == 0
    assert restored.begin(execution=True)
