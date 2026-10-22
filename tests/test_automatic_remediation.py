import json

import pytest

from benchmarks.ascend.automatic_remediation import AutomaticRemediation
from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from test_a5_phase1_adaptive import _project, _wave, _result


@pytest.mark.parametrize("assessment_error", [False, True])
def test_failed_project_remedial_wave_then_original_retry_beyond_target(tmp_path, assessment_error):
    calls, executions = [], []
    runner = None
    automatic = None
    def complete(messages):
        calls.append(messages)
        prompt = messages[-1]['content']
        if len(calls) == 1:
            return _wave(prompt, [_project('original', 'gm-ub-movement')])
        failure = automatic.feedback.selection['selected'][0]['feedback_id']
        row = dict(feedback_id=failure, status='investigating', classification='learnable',
                   reason='isolate movement', project_id='probe', control_project_id='control',
                   hypothesis='alignment', observation='compare oracle')
        if len(calls) == 2:
            response = _wave(prompt, [_project('probe', 'padded-tail'),
                                      _project('control', 'gm-ub-movement')])
            payload = json.loads(response.split('\n', 1)[1])
            payload['failure_investigations'] = [row]
            return 'DECISION: WAVE\n' + json.dumps(payload)
        row.update(status='ready-to-retry', reason='host control and probe isolated movement')
        return 'DECISION: SATURATED\n' + json.dumps(dict(reason='ready', failure_investigations=[row]))
    def execute(request):
        executions.append(request)
        passed = request.spec.project_id != 'original' or len(executions) > 1
        return _result(request.spec, passed, 'host outcome')
    runner = AdaptivePhase1Coordinator(tmp_path / 'run', 'Repair failures.', b'{}',
        CurriculumAgent(complete), execute, max_parallel=1, project_target=1)
    automatic = AutomaticRemediation(runner, target='a3')
    from types import SimpleNamespace
    assessments = []
    def assess(memory, prior, source_id):
        linked = next(row for row in automatic.store.records() if
            row['outcome']['attempt_id'] == 'original' and row['outcome']['source_root'] == str(runner.root.resolve()))
        assert linked['lifecycle'] == 'fixed'
        assessments.append((memory, prior, source_id))
        if assessment_error:
            raise RuntimeError('optional assessment unavailable')
    runner.usefulness_verifier = SimpleNamespace(enabled=True, assess_memory=assess)
    result = runner.run()
    original = next(r for r in automatic.store.records() if
                    r['outcome']['source_root'] == str(runner.root.resolve()) and
                    r['outcome']['attempt_id'] == 'original')
    assert original['lifecycle'] == 'fixed'
    assert [r.spec.project_id for r in executions].count('original') == 2
    retries = [r for r in executions if 'original-retries' in str(r.branch_root)]
    assert retries[0].spec == executions[0].spec
    assert retries[0].pre_wave_memory != executions[0].pre_wave_memory
    assert assessments and assessments[0][1] == retries[0].pre_wave_memory
    assert assessments[0][0].project_spec == executions[0].spec
    assert assessments[0][2].startswith('original-retries/' + original['feedback_id'] + '/retry-1/')
    assert not assessments[0][2].startswith('/')
    marker = next((runner.root/'original-retries').glob('*/retry-1/usefulness-assessment.json'))
    assert json.loads(marker.read_bytes())['complete'] is not assessment_error
    assert result['project_target'] == 1
    before = len(executions)
    runner.resume()
    assert len(executions) == before
    if not assessment_error:
        assert len(assessments) == 1


def test_expired_failure_budget_does_not_extend_campaign(tmp_path):
    from test_generic_failure_feedback import phase1
    phase1(tmp_path / 'source')
    runner = AdaptivePhase1Coordinator(tmp_path / 'run', 'Bounded.', b'{}',
        CurriculumAgent(lambda _: 'DECISION: SATURATED\n{"reason":"done"}'),
        lambda request: _result(request.spec, True, 'pass'), max_parallel=1, project_target=1)
    automatic = AutomaticRemediation(runner, target='a3', sources=[dict(type='phase1', root=str(tmp_path/'source'))])
    automatic.refresh(runner._load_state())
    key = next(iter(automatic.budgets))
    path = runner.root / 'failure-budgets' / (key + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(max_seconds=1800, elapsed_seconds=1800)))
    assert not automatic.refresh(runner._load_state())
    assert automatic.store.records()[0]['lifecycle'] == 'failed'


def test_committed_memory_export_authenticates_without_review(tmp_path):
    import pytest
    from tests.test_a3_reviewed_phase2 import native_source
    from optimizer.memory import freeze_committed_phase1, verify_snapshot
    root, review, _ = native_source(tmp_path)
    state = json.loads((root / 'state.json').read_bytes())
    state['prior_outcomes'][0]['summary'] = 'host verified'
    (root / 'state.json').write_text(json.dumps(state))
    review.unlink()
    snapshot = freeze_committed_phase1(root, b'')
    assert snapshot['phase'] == 'committed-phase1'
    assert not any('review' in path for path in snapshot['files'])
    assert len(snapshot['memory_entries']) == 1
    verify_snapshot(snapshot)
    next(root.glob('waves/*/branches/*/artifacts/objects/*/*')).write_bytes(b'changed')
    with pytest.raises(ValueError):
        freeze_committed_phase1(root, b'')


def test_unrelated_projects_do_not_consume_failure_budget(tmp_path):
    from types import SimpleNamespace
    from test_generic_failure_feedback import phase1
    phase1(tmp_path/'source')
    runner = AdaptivePhase1Coordinator(tmp_path/'run', 'Bounded.', b'{}',
        CurriculumAgent(lambda _: ''), lambda _: None, max_parallel=1, project_target=1)
    automatic = AutomaticRemediation(runner, target='a3', sources=[dict(type='phase1', root=str(tmp_path/'source'))])
    automatic.refresh(runner._load_state())
    budget = next(iter(automatic.budgets.values()))
    now = [1.]
    budget.clock = lambda: now[0]
    request = SimpleNamespace(branch_root=runner.root/'waves/wave-1/branches/new',
                              assignment=SimpleNamespace(project_id='new'))
    with automatic.project_active(request):
        now[0] += 60
    assert budget.elapsed_seconds == 0


@pytest.mark.parametrize("standalone", [False, "envelope", "list", "single"])
def test_external_phase2_retry_uses_frozen_suite_with_updated_committed_memory(tmp_path, monkeypatch, standalone):
    from types import SimpleNamespace
    from tests.test_a3_reviewed_phase2 import native_source
    from tests.test_phase2_optimizer_execution import Agent, Backend, candidate, snapshot
    from benchmarks.ascend.phase2_optimizer_execution import OptimizerTaskExecutor
    from benchmarks.ascend.automatic_remediation import optimizer_retry_adapter
    from benchmarks.ascend.phase2_feedback import FeedbackStore
    from optimizer.jobs import atomic, canonical
    source, _, _ = native_source(tmp_path/'learning')
    state = json.loads((source/'state.json').read_bytes())
    state['prior_outcomes'][0]['summary'] = 'host verified'
    atomic(source/'state.json', canonical(state))
    task = dict(task_id='original', submission=dict(source='def forward(a): return a',
        entrypoint='forward', signatures=[dict(args=[dict(shape=[8], dtype='float32')])]))
    class FrozenBackend(Backend):
        def __init__(self, latency):
            super().__init__()
            self.latency = latency
        def prepare(self, spec, directory):
            value = super().prepare(spec, directory)
            value.update(suite_sha256='a'*64, manifest=dict(policy=dict(rtol=.001), submission=spec))
            value['cases'][0]['timing']['torch']['latency_us'] = self.latency
            return value
        def evaluate(self, source, directory, **kwargs):
            return dict(super().evaluate(source, directory, **kwargs), suite_sha256='a'*64)
    config = dict(target='bz-a3-1', physical_device=14, validation_wrapper=str(tmp_path/'validation.sh'), max_cases=2)
    original_dir = tmp_path/'external'
    first = OptimizerTaskExecutor(original_dir, task, snapshot(), config,
        agent=Agent([dict(action='cases', signatures=[], summary='freeze'), candidate('bad')]), backend=FrozenBackend(10))
    first.prepare()
    failed = first.attempt('failed')
    atomic(original_dir/'state.json', canonical(dict(phase='optimizer-phase2', attempts=[
        dict(attempt_id='failed', task=task, result=failed, execution_directory='')])) )
    atomic(original_dir/'campaign-identity.json', canonical(dict(target='a3', language='ascend-c')))
    store = FeedbackStore(tmp_path/'feedback')
    store.import_campaign(original_dir)
    row = store.records()[0]
    if standalone:
        submission = first.store.submission(first.job_id)
        original = {k: v for k, v in submission.items() if k not in {
            'memory_snapshot', 'max_attempts', 'max_seconds', 'name'}}
        if standalone == 'list':
            original['signature'] = original['signature']['signatures']
        elif standalone == 'single':
            original['signature'] = original['signature']['signatures'][0]
        row['outcome'].update(source_type='optimizer', original_spec=original,
            task=dict(task_id='original', submission=original))
        atomic(store.root / (row['feedback_id'] + '.json'), canonical(row))
    received = []
    def factory(directory, task, memory, bound, **kwargs):
        received.append(memory)
        assert type(bound['phase2_attempt_seconds']) is int and bound['phase2_attempt_seconds'] == 30
        assert bound['max_cases'] == 2
        return OptimizerTaskExecutor(directory, task, memory, bound,
            agent=Agent([candidate('good')]), backend=FrozenBackend(12), **kwargs)
    monkeypatch.setattr('benchmarks.ascend.phase2_optimizer_execution.OptimizerTaskExecutor', factory)
    runner = SimpleNamespace(root=source, initial_memory=b'')
    retry = optimizer_retry_adapter(runner, store, dict(config, max_cases=1))
    retry(row, tmp_path/'retried', 30.75)
    retried = next(r for r in store.records() if r['feedback_id'] != row['feedback_id'])
    assert store.link_original_retry(row['feedback_id'], retried['feedback_id'])['lifecycle'] == 'fixed'
    assert received[0]['phase'] == 'committed-phase1'
    assert len(received[0]['memory_entries']) == 1


def test_original_retry_propagates_only_remaining_failure_seconds(tmp_path):
    from types import SimpleNamespace
    from test_generic_failure_feedback import phase1
    from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
    source = tmp_path/'source'
    phase1(source, spec=_project('original', 'gm-ub-movement'))
    received = []
    def execute(request):
        received.append(request.remaining_seconds)
        return _result(request.spec, True, 'pass')
    runner = AdaptivePhase1Coordinator(tmp_path/'run', 'Bounded.', b'{}',
        CurriculumAgent(lambda _: ''), execute, max_parallel=1, project_target=1)
    automatic = AutomaticRemediation(runner, target='a3', sources=[dict(type='phase1', root=str(source))])
    automatic.refresh(runner._load_state())
    key = next(iter(automatic.budgets))
    path = runner.root/'failure-budgets'/(key+'.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(max_seconds=1800, elapsed_seconds=1780)))
    automatic.refresh(runner._load_state())
    spec = ProjectSpec.from_mapping(_project('original', 'gm-ub-movement'))
    request = SimpleNamespace(branch_root=runner.root/'original-retries'/key/'retry-1/waves/original/branches/original',
        assignment=SimpleNamespace(project_id='original', spec=spec, authored_ordinal=1),
        pre_wave_memory=b'{}', inputs=(), oracle=None)
    runner._execute(request)
    assert 0 < received[0] <= 20


@pytest.mark.parametrize('initially_expired', [True, False])
def test_retry_dispatch_stops_when_campaign_time_is_spent(tmp_path, initially_expired):
    from test_generic_failure_feedback import phase1
    for name in ('first', 'second'):
        phase1(tmp_path/name, spec=_project(name, 'gm-ub-movement'))
    runner = AdaptivePhase1Coordinator(tmp_path/'run', 'Bounded retries.', b'{}',
        CurriculumAgent(lambda _: ''), lambda _: None, max_parallel=1, project_target=1)
    automatic = AutomaticRemediation(runner, target='a3', sources=[
        dict(type='phase1', root=str(tmp_path/name)) for name in ('first', 'second')])
    state = runner._load_state()
    automatic.refresh(state)
    for record in automatic.store.records():
        automatic.store.update_investigation(record['feedback_id'], status='ready-to-retry', evidence=['probe'])
    automatic.refresh(state)
    remaining, dispatched = [0 if initially_expired else 1], []
    runner._remaining_seconds = lambda: remaining[0]
    def execute_original(row, directory, state):
        dispatched.append(directory)
        phase1(directory, verified=True, spec=row['outcome']['original_spec'])
        automatic.store.import_phase1(directory)
        remaining[0] = 0
    automatic._phase1_retry = execute_original
    automatic.retry_ready(state)
    assert len(dispatched) == (0 if initially_expired else 1)
    assert sum(len(row['retries']) for row in automatic.state['failures'].values()) == len(dispatched)


@pytest.mark.parametrize("late_decision", ["wave", "invalid"])
def test_selection_uses_failure_allowance_and_discards_late_wave(tmp_path, late_decision):
    from test_generic_failure_feedback import phase1
    phase1(tmp_path/'source', spec=_project('failed', 'gm-ub-movement'))
    now, prompts, executed = [0.0], [], []
    def complete(messages):
        prompts.append(messages)
        prompt = messages[1]['content']
        if len(prompts) == 1:
            supplied = json.loads(prompt.rsplit('\n', 1)[-1])
            assert 0 < supplied['remaining_seconds'] <= 5
            now[0] += 6
            return (_wave(prompt, [_project('must-not-run', 'gm-ub-movement')])
                    if late_decision == 'wave' else '{invalid')
        return 'DECISION: STALLED\n{"reason":"No eligible work remains."}'
    runner = AdaptivePhase1Coordinator(tmp_path/'run', 'Bounded.', b'{}', CurriculumAgent(complete, clock=lambda:now[0]),
        lambda request:executed.append(request), max_parallel=1, project_target=1,
        max_seconds=100, clock=lambda:now[0])
    automatic = AutomaticRemediation(runner, target='a3', seconds=5,
        sources=[dict(type='phase1', root=str(tmp_path/'source'))])
    assert runner.run()['status'] == 'STALLED'
    assert len(prompts) == 2 and not executed
    assert next(iter(automatic.state['failures'].values()))['remaining_seconds'] == 0
