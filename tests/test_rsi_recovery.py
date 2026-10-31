"""Real canonical runner trials; no second development loop in test or production."""
from dataclasses import asdict, replace
import json
from rsi.event_journal import read_event_refs
from core.artifacts import canonical_bytes

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
from test_rsi_project_runner import Session, Clock, make
from test_rsi_learning import env
from rsi.learning import Validation


def setup_pair(tmp_path, *, repairs=(False, True), limits=None, remaining=None, wrap=None,
               original_session=None, clock=None, checkpoint_hook=None, config_overrides=None):
    original, injected, database = make(tmp_path, original_session, config_overrides=config_overrides,
                                        **({'clock': clock} if clock else {}))
    store = ContentAddressedArtifacts(tmp_path / 'shared')
    incumbent = store.put_json(original.bundle.to_dict())
    documents = list(original.bundle.documents)
    documents[0] = SkillDocument(documents[0].name, documents[0].content + '\nUse stable summation.\n')
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

    failure_ref = checkpoint_hook(original, injected, store) if checkpoint_hook else None
    pair = PairedRecovery.create(tmp_path / 'pair.json', original, artifacts=store,
        terminal_failure_ref=failure_ref,
        memory_ref=store.put_bytes(b'memory'), kdb_ref=store.put_bytes(b'kdb'),
        model_config={'model': 'TEST', 'temperature': 0}, incumbent=incumbent, candidate=candidate,
        joint_limits=limits or BudgetLimits(design_iterations=6, implementation_iterations=4,
                                            execution_tries=40, actor_seconds=1200),
        remaining_parent=remaining or {'actor_seconds': 1200, 'recovery_seconds': 1200}, runner_factory=factory)
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
        assert old.spec.budgets.actor_seconds == 600
        assert old.spec.budgets.implementation_iterations == 2
        assert old.memory_ref == new.memory_ref and old.kdb_ref == new.kdb_ref
        assert old.backend_identity == new.backend_identity
        assert old.operation_id != new.operation_id
        assert old.root != new.root
        result = run_pair(pair)
        assert result['decision'] == 'benefit', result
        assert result['arms']['candidate']['correctness'] is True
        assert result['arms']['incumbent']['outcome'] == 'failed'
        assert result['arms']['incumbent']['classification'] == 'budget_exhausted'
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


class ClosedBudget(Session):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def next_action(self, *args, **kwargs):
        from coding.pi.session import ActionError
        self.calls += 1
        self.clock.value += 4000
        raise ActionError('numeric strategy arguments could not be constructed')


def test_closed_budget_checkpoint_pending_preserved_and_new_arms_scope(tmp_path):
    import time
    class ElapsedClock(Clock):
        def __init__(self):
            self.started = time.monotonic()

        def __call__(self):
            return self.value + time.monotonic() - self.started

    clock = ElapsedClock()
    captured = {}
    def freeze_failure(original, injections, store):
        result = original.run()
        assert result['status'] == 'budget_exhausted' and result['pending']
        proof = original.terminal_failure_proof(closure_ref=result['actor_closure_ref'])
        captured.update(proof=proof, state=(original.root / 'workflow.json').read_bytes(),
            events=canonical_bytes(read_event_refs(original.root)), files=(original.backend.project / 'solution.py').read_bytes())
        return store.put_json(proof)
    def trial(request, reopen, injections):
        if request.arm == 'incumbent':
            injections['session_factory'] = lambda: ClosedBudget(clock)
        return ProjectRunner(request.root, **injections) if reopen else ProjectRunner.create(
            request.root, request.spec, request.bundle, asdict(request.config), **injections)
    pair, original, store, _, db = setup_pair(tmp_path, original_session=ClosedBudget(clock),
        clock=clock, checkpoint_hook=freeze_failure, wrap=trial)
    try:
        frozen = pair.frozen
        assert store.read_json(frozen['terminal_failure_ref']) == captured['proof']
        assert frozen['original_spent'] == captured['proof']['actor_seconds_total']
        left, right = pair.request('incumbent'), pair.request('candidate')
        assert left.spec == right.spec and left.memory_ref == right.memory_ref and left.kdb_ref == right.kdb_ref
        assert left.model_config == right.model_config and left.backend_identity == right.backend_identity
        assert left.operation_id != right.operation_id != original.workflow.pending['id']
        assert (left.root.parent / 'inputs/solution.py').read_bytes() == captured['files']
        result = run_pair(pair)
        assert result['decision'] == 'benefit'
        failed = result['arms']['incumbent']
        proof = store.read_json(failed['terminal_failure_ref'])
        state = store.read_json(failed['workflow_receipt'])
        assert failed['workflow'] is False and failed['classification'] == 'budget_exhausted'
        assert state['pending'] == proof['pending'] and proof['stage_completion_claim'] is False
        assert store.read_json(proof['accounting_snapshot']) == state
        assert result['arms']['candidate']['outcome'] == 'accepted'
        assert captured['state'] == (original.root / 'workflow.json').read_bytes()
        assert captured['events'] == canonical_bytes(read_event_refs(original.root))
        assert original.workflow.pending == captured['proof']['pending']
    finally:
        db.close()


@pytest.mark.parametrize('fault', ['forged_proof', 'unknown_query', 'unknown_command'])
def test_checkpoint_failure_proof_veto_blocks_pair_creation(tmp_path, fault):
    from coding.pi.rpc import PiDeadlineError
    class Failed(Session):
        def next_action(self, context, allowed, **kwargs):
            if context['stage'] == 'design' and any(e['name'] == 'query_kdb' for e in context['evidence']):
                raise PiDeadlineError('provider callback closed')
            return super().next_action(context, allowed, **kwargs)
    def hook(original, injections, store):
        result = original.run()
        proof = original.terminal_failure_proof(closure_ref=result['actor_closure_ref'])
        if fault == 'forged_proof':
            proof['actor_seconds_total'] += 1
        elif fault == 'unknown_query':
            original.append(dict(type='query-dispatch', call_id='unknown', operation_id=original.workflow.pending['id']))
        else:
            original.workflow.reserve_execution('unsettled-command')
        return store.put_json(proof)
    with pytest.raises((WorkflowError, ValueError)):
        setup_pair(tmp_path, original_session=Failed(), checkpoint_hook=hook)
    assert not (tmp_path / 'pair.json').exists()


@pytest.mark.parametrize('provider_arm', ['incumbent', 'candidate', 'both'])
def test_provider_only_arm_difference_remains_inconclusive(tmp_path, provider_arm):
    from coding.pi.rpc import PiDeadlineError
    class Failed(Session):
        def next_action(self, context, allowed, **kwargs):
            if context['stage'] == 'design' and any(e['name'] == 'query_kdb' for e in context['evidence']):
                raise PiDeadlineError('observed closed provider deadline')
            return super().next_action(context, allowed, **kwargs)
    def trial(request, reopen, injections):
        if provider_arm in (request.arm, 'both'):
            injections['session_factory'] = lambda: Failed()
        return ProjectRunner(request.root, **injections) if reopen else ProjectRunner.create(
            request.root, request.spec, request.bundle, asdict(request.config), **injections)
    pair, _, store, _, db = setup_pair(tmp_path, repairs=(True, True), wrap=trial)
    try:
        result = run_pair(pair)
        assert result['decision'] == 'inconclusive'
        for arm, metrics in result['arms'].items():
            if provider_arm in (arm, 'both'):
                assert metrics['classification'] == 'provider_timeout'
                assert metrics['task_failure'] is False and metrics['workflow'] is False
                proof = store.read_json(metrics['terminal_failure_ref'])
                assert proof['pending'] and proof['retrieval_receipts']
                assert all(store.read_json(ref)['kind'] == 'retrieval' for ref in proof['retrieval_receipts'])
    finally:
        db.close()


def test_unknown_arm_query_never_becomes_a_completed_failure(tmp_path):
    def trial(request, reopen, injections):
        def unavailable(query):
            raise OSError('backend completion not observed')
        injections['query'] = unavailable
        return ProjectRunner(request.root, **injections) if reopen else ProjectRunner.create(
            request.root, request.spec, request.bundle, asdict(request.config), **injections)
    pair, _, _, calls, db = setup_pair(tmp_path, wrap=trial)
    try:
        with pytest.raises(WorkflowError):
            pair.step()
        assert pair.state['arms']['incumbent']['status'] == 'pending'
        assert pair.result() is None
        with pytest.raises(WorkflowError):
            pair.step()
        assert len(calls) == 1
    finally:
        db.close()


def test_failed_command_changed_source_is_frozen_from_public_proof(tmp_path):
    import sys
    captured = {}
    def checkpoint(original, injections, store):
        result = original.run()
        proof = original.terminal_failure_proof(closure_ref=result['actor_closure_ref'])
        assert proof['source_hash'] != proof['stage_initial_source']
        captured.update(proof=proof, state=(original.root / 'workflow.json').read_bytes())
        return store.put_json(proof)
    commands = {'compile': [sys.executable, '-c',
        "from pathlib import Path; Path('new_input.json').write_text('{}')"],
        'test': [sys.executable, '{verifier}'], 'validation': [sys.executable, '{verifier}']}
    pair, original, store, _, db = setup_pair(tmp_path, checkpoint_hook=checkpoint,
        config_overrides={'commands': commands})
    try:
        proof = captured['proof']
        assert pair.frozen['original_source'] == proof['source_hash']
        assert pair.frozen['stage_initial_source'] == proof['stage_initial_source']
        assert store.read_json(pair.frozen['original_workflow_ref'])['pending'] == original.workflow.pending
        for arm in ('incumbent', 'candidate'):
            request = pair.request(arm)
            assert (request.root.parent / 'inputs/new_input.json').read_bytes() == b'{}'
            assert request.spec.goal == original.spec.goal
        assert captured['state'] == (original.root / 'workflow.json').read_bytes()
    finally:
        db.close()


def test_parent_deadline_persists_without_rewriting_runner_contract(tmp_path):
    clock = Clock()
    runner, injections, db = make(tmp_path, clock=clock)
    try:
        limits = runner.spec.budgets.to_dict()
        created = runner.workflow.state['created']
        runner.workflow.constrain_wall_deadline('parent-cycle', clock() + 100)
        clock.value += 25
        reopened = ProjectRunner(runner.root, **injections)
        assert reopened.workflow.remaining()['wall_seconds'] == 75
        reopened.workflow.constrain_wall_deadline('parent-cycle', clock() + 1000)
        assert reopened.workflow.remaining()['wall_seconds'] == 75
        assert reopened.workflow.state['created'] == created
        assert reopened.workflow.limits.to_dict() == limits == reopened.spec.budgets.to_dict()
        assert runner.store.read_json(runner.contract_ref)['spec']['budgets'] == limits
    finally:
        db.close()


def test_first_arm_setup_wall_expiry_blocks_next_dispatch_and_resume(tmp_path):
    clock = Clock()
    prepared = []
    class SlowSetup(Session):
        def prepare(self, timeout, *, schemas):
            prepared.append(timeout)
            clock.value += 50
            return super().prepare(timeout, schemas=schemas)
    def trial(request, reopen, injections):
        injections['session_factory'] = SlowSetup
        return ProjectRunner(request.root, **injections) if reopen else ProjectRunner.create(
            request.root, request.spec, request.bundle, asdict(request.config), **injections)
    pair, _, store, calls, db = setup_pair(tmp_path, clock=clock, wrap=trial,
        limits=BudgetLimits(actor_seconds=1200, wall_seconds=40),
        remaining={'actor_seconds': 1200, 'wall_seconds': 40})
    incumbent, candidate = pair.frozen['incumbent'], pair.frozen['candidate']
    spec = CycleSpec('wall', Mode.PHASE2, (ProjectSpec('p', 'original goal', ('python',)),),
                     pair.frozen['memory'], incumbent)
    scheduler = CycleScheduler(tmp_path / 'wall-cycle', spec,
        lambda *a: HostTaskResult('failed', 'r', 0), lambda *a: SnapshotUpdate(pair.frozen['memory']),
        skill_update=lambda *a: SnapshotUpdate(candidate), wall_seconds=40, clock=clock)
    learning = LearningStub(incumbent)
    def open_pair(*args):
        return PairedRecovery(pair.path, artifacts=store, runner_factory=pair.factory, clock=clock)
    try:
        for _ in range(3):
            scheduler.step()
        controller = CycleController(tmp_path / 'wall-controller', scheduler, learning,
                                     artifacts=store, pair_factory=open_pair)
        with pytest.raises(WorkflowError, match='parent wall'):
            controller.step()
        retained = open_pair()
        first = store.read_json(retained.state['arms']['incumbent']['receipt'])
        assert first['wall_limited'] and first['outcome'] == 'failed'
        assert prepared == [20] and len(calls) == 1
        assert retained.state['arms']['candidate']['status'] == 'ready'
        frozen = retained.frozen
        clock.value += 100
        resumed = CycleController(controller.path, scheduler, learning, artifacts=store, pair_factory=open_pair)
        with pytest.raises(WorkflowError, match='parent wall'):
            resumed.reconcile(dispatch=True)
        with pytest.raises(WorkflowError, match='observe/reconcile'):
            resumed.reconcile()
        assert open_pair().frozen == frozen and len(calls) == 1
        assert not learning.calls
    finally:
        scheduler.close()
        db.close()


def test_current_arm_thinking_gets_remaining_wall_and_expiry_is_retained(tmp_path):
    clock = Clock()
    thinking_limits = []
    class Bounded(Session):
        def next_action(self, context, allowed, *, seconds, **kwargs):
            thinking_limits.append(seconds)
            clock.value += seconds + 1
            return super().next_action(context, allowed, seconds=seconds, **kwargs)
    def trial(request, reopen, injections):
        injections['session_factory'] = Bounded
        return ProjectRunner(request.root, **injections) if reopen else ProjectRunner.create(
            request.root, request.spec, request.bundle, asdict(request.config), **injections)
    pair, _, store, calls, db = setup_pair(tmp_path, clock=clock, wrap=trial,
        limits=BudgetLimits(actor_seconds=1200, wall_seconds=40),
        remaining={'actor_seconds': 1200, 'wall_seconds': 40})
    try:
        first = pair.step()
        assert thinking_limits == [20] and first['wall_limited']
        proof = store.read_json(first['terminal_failure_ref'])
        state = store.read_json(proof['accounting_snapshot'])
        assert state['pending'] and state['wall_deadlines']
        assert proof['limits']['wall_seconds'] == 20
        assert first['actor_seconds'] == 21
        with pytest.raises(WorkflowError, match='parent wall'):
            pair.step()
        assert len(calls) == 1
    finally:
        db.close()


def test_factory_copy_time_counts_against_persisted_arm_deadline(tmp_path):
    clock = Clock()
    sessions = []
    def trial(request, reopen, injections):
        clock.value += 50
        injections['session_factory'] = lambda: sessions.append(Session()) or sessions[-1]
        return ProjectRunner.create(request.root, request.spec, request.bundle, asdict(request.config), **injections)
    pair, _, store, calls, db = setup_pair(tmp_path, clock=clock, wrap=trial,
        limits=BudgetLimits(actor_seconds=1200, wall_seconds=40),
        remaining={'actor_seconds': 1200, 'wall_seconds': 40})
    try:
        first = pair.step()
        assert first['wall_limited'] and not sessions
        state = store.read_json(first['workflow_receipt'])
        assert state['history'] == [] and state['limits']['wall_seconds'] == 20
        assert min(state['wall_deadlines'].values()) < state['created']
        with pytest.raises(WorkflowError, match='parent wall'):
            pair.step()
        assert len(calls) == 1
    finally:
        db.close()


def test_parent_remaining_wall_bounds_backend_command_timeout(tmp_path, monkeypatch):
    import subprocess
    import rsi.python_execution as execution
    runner, _, db = make(tmp_path)
    timeouts, killed = [], []
    class Process:
        pid, returncode = 123456, -9
        def communicate(self, timeout=None):
            if timeout is not None:
                timeouts.append(timeout)
                raise subprocess.TimeoutExpired('frozen-compile', timeout)
            return b'', b''
    try:
        runner.workflow.constrain_wall_deadline('parent-cycle', runner.workflow.clock() + 100)
        remaining = runner.workflow.remaining()['wall_seconds']
        runner.workflow.begin()
        monkeypatch.setattr(execution.subprocess, 'Popen', lambda *a, **kw: Process())
        monkeypatch.setattr(execution.os, 'killpg', lambda pid, sig: killed.append(pid))
        receipt = runner.backend.run('bounded-compile', 'compile')
        assert receipt['timed_out'] and receipt['classification'] == 'timeout'
        assert len(timeouts) == 1 and 0 < timeouts[0] <= remaining
        assert killed == [Process.pid]
        assert runner.spec.budgets.wall_seconds is None
    finally:
        db.close()
