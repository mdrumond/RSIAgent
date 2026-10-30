"""Real canonical runner trials; no second development loop in test or production."""
from dataclasses import asdict, replace
import json

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.ascend.subject_memory import SubjectMemory
from rsi.contracts import BudgetLimits, CycleSpec, Mode, ProjectSpec, Stage
from rsi.cycles import CycleScheduler, HostTaskResult, SnapshotUpdate
from rsi.project_runner import ProjectRunner
from rsi.python_execution import atomic_json
from rsi.recovery import CycleController, PairedRecovery, fork_inference
from rsi.runtime import WorkflowError
from rsi.skills import SkillBundle, SkillDocument
from test_rsi_project_runner import Session, make
from test_rsi_learning import env
from rsi.learning import Validation


def setup_pair(tmp_path, *, repairs=(False, True), limits=None, remaining=None, wrap=None):
    original, injected, database = make(tmp_path)
    store = ContentAddressedArtifacts(tmp_path / 'shared')
    incumbent = store.put_json(original.bundle.to_dict())
    documents = list(original.bundle.documents)
    documents[0] = SkillDocument(documents[0].name, documents[0].content + '\n## Learned guidance\nUse stable summation.\n')
    candidate = store.put_json(SkillBundle(tuple(documents), original.bundle.mapping).to_dict())
    calls = []

    def factory(request, *, reopen):
        calls.append((request.arm, reopen, request))
        kwargs = dict(injected, backend_identity=request.backend_identity,
                      session_factory=lambda: Session(repair=repairs[request.arm == 'candidate']))
        if wrap:
            return wrap(request, reopen, kwargs)
        if reopen:
            return ProjectRunner(request.root, **kwargs)
        return ProjectRunner.create(request.root, request.spec, request.bundle, asdict(request.config), **kwargs)

    pair = PairedRecovery.create(tmp_path / 'pair.json', original, artifacts=store,
        memory_ref=store.put_bytes(b'memory'), kdb_ref=store.put_bytes(b'kdb'),
        model_config={'model': 'TEST', 'temperature': 0}, incumbent=incumbent, candidate=candidate,
        joint_limits=limits or BudgetLimits(design_iterations=6, implementation_iterations=4,
                                            execution_tries=40, actor_seconds=60),
        remaining_parent=remaining or {'actor_seconds': 40, 'recovery_seconds': 30}, runner_factory=factory)
    return pair, original, store, calls, database


def run_pair(pair):
    pair.step()
    pair.step()
    return pair.result()


def test_equal_frozen_inputs_new_allocation_and_source_isolation(tmp_path):
    pair, original, store, calls, db = setup_pair(tmp_path)
    try:
        before = (original.backend.project / 'solution.py').read_bytes()
        spent = original.workflow.accounting.seconds()
        old, new = pair.request('incumbent'), pair.request('candidate')
        assert old.spec == new.spec
        assert old.spec.budgets.actor_seconds == 15
        assert old.spec.budgets.implementation_iterations == 2
        assert old.memory_ref == new.memory_ref and old.kdb_ref == new.kdb_ref
        assert old.backend_identity == new.backend_identity
        assert old.operation_id != new.operation_id
        assert old.root != new.root
        result = run_pair(pair)
        assert result['decision'] == 'benefit', result
        assert result['arms']['candidate']['correctness'] is True
        assert result['arms']['incumbent']['outcome'] == 'budget_exhausted'
        assert result['arms']['candidate']['source_hash'] != result['arms']['incumbent']['source_hash']
        assert (original.backend.project / 'solution.py').read_bytes() == before
        assert original.workflow.accounting.seconds() == spent
        for row in result['arms'].values():
            history = store.read_json(row['workflow_receipt'])['history']
            assert history[0]['evidence']['stage'] == Stage.PREFLIGHT.value
        assert pair.step() == result
        assert len(calls) == 2
    finally:
        db.close()


@pytest.mark.parametrize('repairs,decision', [((True, False), 'regression'),
    ((True, True), 'inconclusive'), ((False, False), 'inconclusive')])
def test_regression_and_ties_retain_raw_metrics(tmp_path, repairs, decision):
    pair, _, _, _, db = setup_pair(tmp_path, repairs=repairs)
    try:
        result = run_pair(pair)
        assert result['decision'] == decision
        assert set(result['arms']) == {'candidate', 'incumbent'}
        assert result['actor_seconds'] == sum(r['actor_seconds'] for r in result['arms'].values())
    finally:
        db.close()


def test_unknown_arm_never_resubmitted_after_reopen(tmp_path):
    def interrupted(request, reopen, kwargs):
        if reopen:
            return ProjectRunner(request.root, **kwargs)
        ProjectRunner.create(request.root, request.spec, request.bundle, asdict(request.config), **kwargs)
        raise ConnectionError('lost connection before run')
    pair, _, store, calls, db = setup_pair(tmp_path, wrap=interrupted)
    try:
        with pytest.raises(ConnectionError):
            pair.step()
        retained = pair.state['arms']['incumbent']['operation_id']
        reopened = PairedRecovery(pair.path, artifacts=store, runner_factory=pair.factory)
        with pytest.raises(WorkflowError):
            reopened.step()
        with pytest.raises(WorkflowError):
            reopened.reconcile('incumbent')
        assert reopened.state['arms']['incumbent']['operation_id'] == retained
        assert [reopen for _, reopen, _ in calls] == [False, True]
    finally:
        db.close()


@pytest.mark.parametrize('change', ['budget', 'backend', 'source'])
def test_factory_cannot_change_comparison_inputs(tmp_path, change):
    def wrong(request, reopen, kwargs):
        spec = replace(request.spec, budgets=replace(request.spec.budgets, actor_seconds=100)) if change == 'budget' else request.spec
        if change == 'backend':
            kwargs['backend_identity'] = 'different-model'
        runner = ProjectRunner.create(request.root, spec, request.bundle, asdict(request.config), **kwargs)
        if change == 'source':
            (runner.backend.project / 'solution.py').write_text('changed = True\n')
        return runner
    pair, _, _, _, db = setup_pair(tmp_path, wrap=wrong)
    try:
        with pytest.raises(ValueError):
            pair.step()
        assert pair.state['arms']['candidate']['status'] == 'ready'
    finally:
        db.close()


def test_nullable_caps_remain_disabled(tmp_path):
    pair, _, _, _, db = setup_pair(tmp_path,
        limits=BudgetLimits(None, None, None, None, None), remaining={'actor_seconds': None})
    try:
        assert pair.request('candidate').spec.budgets == BudgetLimits(None, None, None, None, None)
    finally:
        db.close()


def learning_state(path, skills):
    atomic_json(path, dict(active=skills, versions=[skills], candidates={}, projects={}, operations={},
                           stopped=False, policy={'mode': 'phase2'}))


def test_inference_forks_same_cas_baseline_but_independent_mutable_heads(tmp_path):
    parent = SubjectMemory(tmp_path / 'parent', b'initial')
    parent.accept('one', dict(base_versions=[], kind='failure', evidence=[]), {'raw': 'retained'})
    store = ContentAddressedArtifacts(tmp_path / 'cas')
    baseline = parent.snapshot()
    memory = store.put_bytes(baseline)
    skills = store.put_json({'immutable': True})
    learning_path = tmp_path / 'learning.json'
    learning_state(learning_path, skills)
    parent_learning = learning_path.read_bytes()
    receipt = fork_inference(tmp_path / 'fork', parent, learning_path, artifacts=store,
        memory_ref=memory, skills_ref=skills, namespace='inference:test')
    receipt.verify()
    branch = SubjectMemory(tmp_path / 'fork/memory', b'initial')
    assert branch.snapshot() == baseline
    branch.accept('two', dict(base_versions=[], kind='correctness', evidence=[]), {})
    assert parent.snapshot() == baseline
    assert learning_path.read_bytes() == parent_learning
    assert branch.snapshot() != baseline
    # Scheduler accepts proof of separate mutable ownership with identical hashes.
    spec = CycleSpec('test', Mode.INFERENCE, (ProjectSpec('p', 'goal', ('python',)),), memory, skills)
    scheduler = CycleScheduler(tmp_path / 'cycle', spec, None, None, skill_update=lambda *a: None,
        isolate=lambda m, s, ctx: fork_inference(tmp_path / 'second', parent, learning_path,
            artifacts=store, memory_ref=m, skills_ref=s, namespace=ctx['namespace']))
    try:
        scheduler.step()
        assert scheduler.state['skills'] == skills and scheduler.state['memory'] == memory
        assert scheduler.state['isolation']['branch_root'] != str(parent.root)
    finally:
        scheduler.close()


class LearningStub:
    def __init__(self, active):
        self.active, self.calls = active, []

    def evaluate(self, candidate, cases, operation):
        self.calls.append((candidate, cases, operation))
        self.active = candidate
        return self.active


@pytest.mark.parametrize('decision', ['benefit', 'regression', 'inconclusive'])
def test_controller_drains_before_next_original_and_only_promotes_validated(tmp_path, decision, env):
    store, _, incumbent, _, baseline, _, make_learning, project = env
    learner = make_learning(validate=lambda *a: Validation(True, True, a[2][0]))
    row, context = project()
    candidate = learner.skill_update(incumbent, baseline, row, context).reference
    calls = []
    spec = CycleSpec('phase2', Mode.PHASE2, tuple(ProjectSpec(str(i), 'same goal', ('python',))
                     for i in range(2)), baseline, incumbent)
    scheduler = CycleScheduler(tmp_path / 'cycle', spec,
        lambda project, memory, skills, ctx: (calls.append((project.goal, skills)) or HostTaskResult('failed', 'r', 1)),
        lambda *a: SnapshotUpdate('memory+'), skill_update=lambda *a: SnapshotUpdate(candidate))
    learning = learner
    class Pair:
        def result(self):
            return dict(decision=decision, actor_seconds=2., evidence_ref=store.put_json({'raw': 'metrics'}))
    controller = CycleController(tmp_path / 'controller', scheduler, learning,
        artifacts=store, pair_factory=lambda *a: Pair())
    try:
        for _ in range(3):
            controller.step()
        assert calls == [('same goal', incumbent)]
        controller.step()  # Must drain pair before original #2.
        assert calls == [('same goal', incumbent)]
        controller.step()
        assert calls[-1][1] == (candidate if decision == 'benefit' else incumbent)
        validations = [op for op in learner.state['operations'].values() if op['request']['kind'] == 'validation']
        assert len(validations) == (1 if decision == 'benefit' else 0)
        assert scheduler.state['spent'] == 4.
        assert len(scheduler.state['extension_receipts']) == 1
    finally:
        scheduler.close()


def test_disabled_recovery_leaves_candidate_pending_without_pair(tmp_path):
    spec = CycleSpec('phase2', Mode.PHASE2, (ProjectSpec('p', 'goal', ('python',)),), 'm', 'old')
    scheduler = CycleScheduler(tmp_path / 'cycle', spec,
        lambda *a: HostTaskResult('failed', 'r', 1), lambda *a: SnapshotUpdate('m+'),
        skill_update=lambda *a: SnapshotUpdate('candidate'), recover=False)
    learning = LearningStub('old')
    controller = CycleController(tmp_path / 'controller', scheduler, learning,
        artifacts=ContentAddressedArtifacts(tmp_path / 'cas'), pair_factory=lambda *a: pytest.fail('disabled'))
    try:
        for _ in range(4):
            controller.step()
        assert scheduler.state['skills'] == 'old'
        assert not learning.calls
        assert not any(e['kind'] == 'paired_recovery_due' for e in scheduler.extension_events())
        assert scheduler.state['candidates'][0]['reference'] == 'candidate'
    finally:
        scheduler.close()


def test_completed_arm_lost_return_is_observed_without_reexecution(tmp_path):
    def disconnected(request, reopen, kwargs):
        if reopen:
            return ProjectRunner(request.root, **kwargs)
        runner = ProjectRunner.create(request.root, request.spec, request.bundle, asdict(request.config), **kwargs)
        execute = runner.run
        def lost_return():
            execute()
            raise ConnectionError('lost completed return')
        runner.run = lost_return
        return runner
    pair, _, store, calls, db = setup_pair(tmp_path, repairs=(True, True), wrap=disconnected)
    try:
        with pytest.raises(ConnectionError):
            pair.step()
        reopened = PairedRecovery(pair.path, artifacts=store, runner_factory=pair.factory)
        receipt = reopened.reconcile('incumbent')
        assert receipt['outcome'] == 'accepted'
        assert reopened.reconcile('incumbent') == receipt
        assert [reopen for _, reopen, _ in calls] == [False, True]
    finally:
        db.close()


def test_admission_failure_history_is_not_a_completed_stage(tmp_path):
    pair, _, _, _, db = setup_pair(tmp_path, repairs=(True, True))
    original_factory = pair.factory
    def with_closed_admission(request, *, reopen):
        runner = original_factory(request, reopen=reopen)
        original_run = runner.run
        def execute():
            result = original_run()
            runner.workflow.state['history'].insert(0, dict(
                operation='closed-admission', stage='design', passed=False,
                rejected_operation=True, host_failure_ref='retained-controller-proof'))
            runner.workflow._save()
            return result
        runner.run = execute
        return runner
    pair.factory = with_closed_admission
    try:
        result = run_pair(pair)
        assert result['decision'] == 'inconclusive'
        assert all(r['workflow'] is True and r['correctness'] is True for r in result['arms'].values())
    finally:
        db.close()
