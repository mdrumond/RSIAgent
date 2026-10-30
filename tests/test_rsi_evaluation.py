"""Trusted comparisons execute canonical runners, never fabricate arm receipts."""
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from rsi.contracts import BudgetLimits
from rsi.evaluation import SkillEvaluation, TuningController, case_identity
from rsi.project_runner import ProjectRunner
from rsi.runtime import WorkflowError
from test_rsi_project_runner import Session, make
from test_rsi_learning import env


def checkpoint(path, name, marker, **overrides):
    path.mkdir()
    template, injected, database = make(path, **overrides)
    source = template.config.source
    solution = Path(source) / 'solution.py'
    solution.write_text(solution.read_text() + '\n# ' + marker + '\n')
    runner = ProjectRunner.create(path / 'checkpoint',
        replace(template.spec, project_id=name), template.bundle, asdict(template.config), **injected)
    return runner, injected, database


@pytest.fixture
def comparison(tmp_path, env, request):
    store, _, incumbent, _, memory, _, make_learning, project = env
    learner = make_learning()
    row, ctx = project()
    candidate = learner.skill_update(incumbent, memory, row, ctx).reference
    training, _, training_db = checkpoint(tmp_path / 'training', 'training', 'training')
    heldout, injected, heldout_db = checkpoint(tmp_path / 'heldout', 'heldout', 'independent heldout',
        **getattr(request, 'param', {}))
    calls = []
    def factory(request, *, reopen, session_factory=None, clock=None):
        calls.append((request.arm, reopen))
        kwargs = dict(injected, backend_identity=request.backend_identity,
                      session_factory=session_factory or (lambda: Session(repair=request.arm == 'candidate')))
        if clock is not None:
            kwargs['clock'] = clock
        if reopen:
            return ProjectRunner(request.root, **kwargs)
        return ProjectRunner.create(request.root, request.spec, request.bundle, asdict(request.config), **kwargs)
    def create(**kwargs):
        remaining = kwargs.pop('remaining_parent', {'actor_seconds': 30})
        repetitions = kwargs.pop('repetitions', 1)
        return SkillEvaluation.create(tmp_path / 'evaluation.json', checkpoints=[heldout],
            training=[case_identity(training)], artifacts=store, memory_ref=memory,
            kdb_ref=store.put_bytes(b'baseline knowledge'), model_config={'model': 'TEST'},
            incumbent=incumbent, candidate=candidate, runner_factory=factory,
            joint_limits=BudgetLimits(6, 4, 40, 30), remaining_parent=remaining,
            case_count=1, repetitions=repetitions, **kwargs)
    yield create, learner, calls, store, heldout, training, factory
    heldout_db.close()
    training_db.close()


def finish(evaluation):
    while evaluation.result() is None:
        evaluation.step()
    return evaluation.result()


def test_real_pairs_promote_only_after_entire_frozen_suite(comparison):
    create, learner, calls, store, _, _, _ = comparison
    evaluation = create(steering={'user_guidance': 'Prioritise numeric recovery', 'project_ref': 'p'})
    learner.validate = evaluation.validation
    assert learner.active == evaluation.plan['incumbent']
    evaluation.step()
    assert learner.active == evaluation.plan['incumbent']
    assert evaluation.result() is None
    result = finish(evaluation)
    assert result['decision'] == 'benefit'
    assert result['actor_seconds'] > 0
    pair = store.read_json(result['pairs'][0])
    assert pair['decision'] == 'benefit'
    assert pair['arms']['incumbent']['task_failure'] is True
    proof = store.read_json(pair['arms']['incumbent']['terminal_failure_ref'])
    assert any(store.read_json(ref)['classification'] == 'numeric_failure'
               for ref in proof['command_receipts'])
    assert calls == [('incumbent', False), ('candidate', False)]
    assert evaluation.promote(learner) == evaluation.plan['candidate']
    assert learner.active == evaluation.plan['candidate']
    assert store.read_json(result['evidence_ref'])['steering']['user_guidance'] == 'Prioritise numeric recovery'
    assert evaluation.promote(learner) == learner.active


def test_renamed_training_copy_is_rejected_before_any_dispatch(comparison):
    create, _, calls, _, heldout, _, _ = comparison
    with pytest.raises(ValueError, match='training'):
        SkillEvaluation.create('unused', checkpoints=[heldout], training=[dict(case_identity(heldout), project_id='renamed')],
            artifacts=None, memory_ref='m', kdb_ref='k', model_config={}, incumbent='old', candidate='new',
            runner_factory=None, joint_limits=BudgetLimits(), remaining_parent={}, case_count=1, repetitions=1)
    assert not calls


def test_unknown_arm_requires_explicit_observation_after_reopen(comparison):
    create, _, calls, store, _, _, factory = comparison
    evaluation = create()
    original_factory = evaluation.factory
    def interrupted(request, *, reopen):
        if not reopen:
            raise ConnectionError('dispatch uncertain')
        return original_factory(request, reopen=reopen)
    evaluation.factory = interrupted
    with pytest.raises(ConnectionError):
        evaluation.step()
    reopened = SkillEvaluation(evaluation.path, artifacts=store, runner_factory=factory)
    with pytest.raises(WorkflowError):
        reopened.step()
    assert calls == []
    # Explicit absence after observing no retained run is safe, not an automatic retry.
    reopened.reconcile(not_dispatched=True)
    finish(reopened)
    assert len(calls) == 2


def test_validation_cannot_admit_unfinished_or_other_candidate(comparison):
    create, _, _, _, _, _, _ = comparison
    evaluation = create()
    with pytest.raises(WorkflowError):
        evaluation.validation('old', 'new', [], 'validation')
    finish(evaluation)
    with pytest.raises(ValueError):
        evaluation.validation(evaluation.plan['incumbent'], 'other', [evaluation.result()['evidence_ref']], 'v')


def test_parent_budget_is_split_and_pending_cannot_be_replaced(comparison):
    create, _, _, store, _, _, _ = comparison
    evaluation = create()
    pair = evaluation.pair(0)
    assert pair.request('candidate').spec.budgets.actor_seconds == 15
    assert pair.request('incumbent').spec.budgets == pair.request('candidate').spec.budgets
    with pytest.raises(ValueError):
        create()
    finish(evaluation)
    spent = evaluation.result()['actor_seconds']
    assert spent <= 30
    assert store.read_json(evaluation.result()['evidence_ref'])['pairs']


@pytest.mark.parametrize('mode,decision', [('tie', 'inconclusive'), ('regression', 'regression')])
def test_ties_and_regressions_do_not_call_learning(comparison, mode, decision):
    create, learner, _, _, _, _, factory = comparison
    evaluation = create()
    def both_repair(request, *, reopen):
        return factory(request, reopen=reopen,
                       session_factory=lambda: Session(repair=mode == 'tie' or request.arm == 'incumbent'))
    # Change the scripted provider, not trusted comparison receipts.
    # Factory remains a real canonical runner; its fixed model identity is unchanged.
    evaluation.factory = both_repair
    result = finish(evaluation)
    assert result['decision'] == decision
    before = dict(learner.state['operations'])
    assert evaluation.promote(learner) == evaluation.plan['incumbent']
    assert learner.state['operations'] == before


def test_default_five_cases_twice_are_frozen_before_dispatch(tmp_path, comparison):
    _, _, calls, store, original, training, factory = comparison
    cases, databases = [], []
    try:
        for index in range(5):
            runner, _, database = checkpoint(tmp_path / f'case-{index}', f'case-{index}', f'case {index}')
            cases.append(runner)
            databases.append(database)
        evaluation = SkillEvaluation.create(tmp_path / 'default.json', checkpoints=cases,
            training=[case_identity(training)], artifacts=store,
            memory_ref=store.put_bytes(b'memory'), kdb_ref=store.put_bytes(b'knowledge'),
            model_config={'model': 'TEST'}, incumbent=store.put_json(original.bundle.to_dict()),
            candidate=store.put_json(original.bundle.to_dict()), runner_factory=factory,
            joint_limits=BudgetLimits(6, 4, 40, 30), remaining_parent={'actor_seconds': 100})
        assert len(evaluation.plan['pairs']) == 10
        assert not calls
        allocations = []
        for index in range(10):
            pair = evaluation.pair(index)
            allocations.extend(pair.request(arm).spec.budgets.actor_seconds for arm in ('incumbent', 'candidate'))
            assert pair.state['arms']['candidate']['status'] == 'ready'
        assert sum(allocations) == 100
    finally:
        for database in databases:
            database.close()


def test_elapsed_wall_limit_blocks_new_arms_without_replay(comparison):
    create, _, calls, _, _, _, _ = comparison
    evaluation = create(clock=lambda: 10., remaining_parent={'actor_seconds': 30, 'wall_seconds': 2.})
    evaluation.clock = lambda: 12.
    with pytest.raises(WorkflowError, match='wall allocation'):
        evaluation.step()
    assert not calls


def test_periodic_and_manual_requests_use_one_scheduler_ledger(tmp_path, comparison):
    from rsi.contracts import CycleSpec, Mode, ProjectSpec
    from rsi.cycles import CycleScheduler, HostTaskResult, SnapshotUpdate
    from rsi.recovery import CycleController
    create, learner, _, store, _, _, _ = comparison
    evaluation = create(steering={'project_ref': 'original', 'user_guidance': 'Numeric recovery first'})
    learner.validate = evaluation.validation
    originals = []
    scheduler = CycleScheduler(tmp_path / 'cycle', CycleSpec('cycle', Mode.PHASE2,
        (ProjectSpec('original', 'goal', ('python',)),), 'memory', learner.active),
        lambda *a: originals.append(a) or HostTaskResult('accepted', 'receipt', 0),
        lambda *a: SnapshotUpdate('memory'), skill_update=lambda *a: SnapshotUpdate(learner.active))
    controller = CycleController(tmp_path / 'extensions', scheduler, learner, artifacts=store, pair_factory=None)
    tuning = TuningController(tmp_path / 'tuning', controller,
        evaluation_factory=lambda event, operation, remaining: evaluation)
    try:
        event_id = tuning.request({'project_ref': 'original', 'user_guidance': 'Numeric recovery first'})
        assert scheduler.extension_events()[0]['steering']['project_ref'] == 'original'
        assert tuning.step() is None
        assert not originals
        result = tuning.step()
        assert result['decision'] == 'benefit'
        spent = scheduler.state['spent']
        assert spent == result['actor_seconds']
        assert scheduler.state['skills'] == evaluation.plan['candidate']
        assert scheduler.state['extension_receipts'][event_id]['actor_seconds'] == spent
        # Receipt acknowledgement is idempotent after a disconnect.
        scheduler.complete_extension(event_id, result['evidence_ref'], actor_seconds=spent)
        assert scheduler.state['spent'] == spent
    finally:
        scheduler.close()


def test_wave_end_coalesces_with_every_five_originals(tmp_path):
    from rsi.contracts import CycleSpec, Mode, ProjectSpec
    from rsi.cycles import CycleScheduler, HostTaskResult, SnapshotUpdate
    scheduler = CycleScheduler(tmp_path / 'cycle', CycleSpec('five', Mode.PHASE2,
        tuple(ProjectSpec(str(i), 'goal', ('python',)) for i in range(5)), 'memory', 'skills'),
        lambda *a: HostTaskResult('accepted', 'receipt', 0), lambda *a: SnapshotUpdate('memory'),
        skill_update=lambda *a: SnapshotUpdate('skills'))
    try:
        for _ in range(15):
            scheduler.step()
        events = [e for e in scheduler.extension_events() if e['kind'] == 'tuning_due']
        assert len(events) == 1
        assert events[0]['original_count'] == 5
        assert set(events[0]['reasons']) == {'cadence', 'wave_end'}
    finally:
        scheduler.close()


def test_real_admission_closure_remains_in_evidence_and_cost(comparison):
    create, learner, _, store, _, _, _ = comparison
    evaluation = create()
    learner.validate = evaluation.validation
    factory = evaluation.factory
    charged = []
    def with_closed_admission(request, *, reopen):
        runner = factory(request, reopen=reopen)
        if reopen:
            return runner
        normal_session = runner.actor.session_factory
        execute = runner.run
        def run():
            runner.actor.session_factory = lambda: Session(missing_query=True)
            interrupted = execute()
            assert interrupted['status'] == 'suspended'
            pending = runner.workflow.pending
            proof = runner.record_admission_failure(pending['id'], controller_proof=dict(
                handle=pending['handle'], actor_closed=True, no_host_effects=True,
                observation='Controller observed closed scripted session before host dispatch'),
                observed_source_hash=runner.workflow.source_hash)
            runner.workflow.reject_operation(pending['id'], proof, runner.workflow.source_hash)
            charged.append(runner.workflow.accounting.seconds())
            runner.actor.session_factory = normal_session
            return execute()
        runner.run = run
        return runner
    evaluation.factory = with_closed_admission
    result = finish(evaluation)
    assert result['decision'] == 'benefit'
    assert result['actor_seconds'] >= sum(charged) > 0
    for pair_ref in result['pairs']:
        for arm in store.read_json(pair_ref)['arms'].values():
            history = store.read_json(arm['workflow_receipt'])['history']
            closure = next(h for h in history if h.get('rejected_operation'))
            assert 'evidence' not in closure
            arm_store = ContentAddressedArtifacts(Path(arm['root']) / 'artifacts')
            assert arm_store.read_json(closure['host_failure_ref'])['controller_proof']['actor_closed'] is True
    assert evaluation.promote(learner) == evaluation.plan['candidate']


@pytest.mark.parametrize('comparison', [dict(config_overrides=dict(
    performance_seconds=1e-9, commands={
        'compile': [sys.executable, '-m', 'py_compile', 'solution.py'],
        'test': [sys.executable, '{verifier}'],
        'validation': [sys.executable, '{verifier}'],
        'timing': [sys.executable, 'solution.py']}))], indirect=True)
def test_actual_correctness_pass_cannot_promote_performance_shortfall(comparison):
    create, learner, _, store, _, _, _ = comparison
    evaluation = create()
    learner.validate = evaluation.validation
    result = finish(evaluation)
    candidate = store.read_json(result['pairs'][0])['arms']['candidate']
    assert candidate['correctness'] is True
    assert candidate['performance'] == 'shortfall'
    assert result['decision'] == 'inconclusive'
    assert learner.evaluate(evaluation.plan['candidate'], [result['evidence_ref']], 'forced-validation') == evaluation.plan['incumbent']
    assert learner.active == evaluation.plan['incumbent']


@pytest.mark.parametrize('repetitions', [1, 2])
def test_provider_only_variance_cannot_promote_even_alongside_task_benefit(comparison, repetitions):
    from coding.pi.rpc import PiDeadlineError
    create, learner, _, store, _, _, _ = comparison
    evaluation = create(repetitions=repetitions)
    learner.validate = evaluation.validation
    factory = evaluation.factory
    failed_once = False
    class ClosedProvider(Session):
        def next_action(self, context, allowed, **kwargs):
            if context['stage'] == 'design' and any(e['name'] == 'query_kdb' for e in context['evidence']):
                raise PiDeadlineError('Observed closed provider before any task command')
            return super().next_action(context, allowed, **kwargs)
    def trial(request, *, reopen):
        nonlocal failed_once
        if not reopen and request.arm == 'incumbent' and not failed_once:
            failed_once = True
            return factory(request, reopen=reopen, session_factory=lambda: ClosedProvider())
        return factory(request, reopen=reopen)
    evaluation.factory = trial
    result = finish(evaluation)
    raw = [store.read_json(ref) for ref in result['pairs']]
    assert raw[0]['decision'] == 'inconclusive'
    old, new = raw[0]['arms']['incumbent'], raw[0]['arms']['candidate']
    assert old['classification'] == 'provider_timeout' and old['task_failure'] is False
    assert new['outcome'] == 'accepted' and new['correctness'] is True
    proof = store.read_json(old['terminal_failure_ref'])
    assert proof['pending'] and proof['retrieval_receipts'] and not proof['command_receipts']
    assert result['actor_seconds'] == sum(arm['actor_seconds'] for pair in raw for arm in pair['arms'].values()) > 0
    assert result['actor_seconds'] <= evaluation.plan['parent']['actor_seconds']
    if repetitions == 2:
        assert raw[1]['decision'] == 'benefit'
        assert raw[1]['arms']['incumbent']['task_failure'] is True
    assert result['decision'] == 'inconclusive'
    assert evaluation.promote(learner) == evaluation.plan['incumbent']
    # Even direct Learning validation cannot reinterpret a provider-only failure.
    assert learner.evaluate(evaluation.plan['candidate'], [result['evidence_ref']], 'provider-validation') == evaluation.plan['incumbent']


@pytest.mark.parametrize('setup', ['factory', 'actor'])
def test_original_parent_deadline_bounds_current_setup_and_resumed_evaluation(comparison, setup):
    create, learner, calls, store, _, _, factory = comparison
    class HostClock:
        offset = 0.
        def __call__(self):
            return time.time() + self.offset
    clock = HostClock()
    prepared = []
    class SlowSetup(Session):
        def prepare(self, timeout, *, schemas):
            prepared.append(timeout)
            clock.offset += 50
            return super().prepare(timeout, schemas=schemas)
    def trial(request, *, reopen):
        if not reopen and setup == 'factory':
            clock.offset += 50
        return factory(request, reopen=reopen,
            session_factory=SlowSetup if setup == 'actor' else Session, clock=clock)
    evaluation = create(clock=clock, remaining_parent={'actor_seconds': 30, 'wall_seconds': 40})
    evaluation.factory = trial
    learner.validate = evaluation.validation
    plan_ref = evaluation.state['plan']
    deadline = evaluation.plan['created'] + 40
    frozen = evaluation.pair(0).frozen
    evaluation.step()  # Executes only the first real canonical arm.
    pair = evaluation.pair(0)
    assert pair.state['wall_deadline'] <= deadline
    old = store.read_json(pair.state['arms']['incumbent']['receipt'])
    assert old['wall_limited'] and old['outcome'] == 'failed'
    assert old['actor_seconds'] < 50  # Actual setup is excluded from actor time.
    workflow = store.read_json(old['workflow_receipt'])
    created = workflow['created']
    assert min(workflow['wall_deadlines'].values()) <= deadline
    assert workflow['limits'] == frozen['allocation']
    assert pair.state['arms']['candidate']['status'] == 'ready'
    assert len(calls) == 1
    if setup == 'actor':
        assert len(prepared) == 1 and 0 < prepared[0] <= frozen['allocation']['wall_seconds']
    else:
        assert not prepared
    resumed = SkillEvaluation(evaluation.path, artifacts=store, runner_factory=trial, clock=clock)
    clock.offset += 100
    with pytest.raises(WorkflowError, match='wall allocation'):
        resumed.step()
    retained = resumed.pair(0)
    assert resumed.state['plan'] == plan_ref and retained.frozen == frozen
    assert retained.state['wall_deadline'] <= deadline
    assert store.read_json(retained.state['arms']['incumbent']['receipt']) == old
    assert store.read_json(old['workflow_receipt'])['created'] == created
    assert len(calls) == 1 and not [op for op in learner.state['operations'].values()
                                   if op['request']['kind'] == 'validation']
