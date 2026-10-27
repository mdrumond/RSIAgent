import json

import pytest

from benchmarks.ascend.automatic_remediation import AutomaticRemediation
from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from test_a5_phase1_adaptive import _project, _wave, _result


@pytest.mark.parametrize("assessment_error", [False, True])
@pytest.mark.parametrize("ready_wave", [False, True])
@pytest.mark.parametrize("allow_partial", [False, True])
def test_failed_project_remedial_wave_then_original_retry_beyond_target(tmp_path, assessment_error, ready_wave, allow_partial):
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
        if ready_wave:
            payload = json.loads(_wave(prompt, [_project('new-work', 'simd-vector')]).split('\n', 1)[1])
            payload['failure_investigations'] = [row]
            return 'DECISION: WAVE\n' + json.dumps(payload)
        return 'DECISION: SATURATED\n' + json.dumps(dict(reason='ready', failure_investigations=[row]))
    def execute(request):
        executions.append(request)
        passed = request.spec.project_id != 'original' or len(executions) > 1
        return _result(request.spec, passed, 'host outcome')
    runner = AdaptivePhase1Coordinator(tmp_path / 'run', 'Repair failures.', b'{}',
        CurriculumAgent(complete), execute, max_parallel=1, project_target=1, allow_partial=allow_partial)
    automatic = AutomaticRemediation(runner, target='a3')
    from types import SimpleNamespace
    assessments, ordinary_assessments = [], []
    def assess(memory, prior, source_id):
        if not source_id.startswith('original-retries/'):
            ordinary_assessments.append(source_id)
            return []
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
    assert {'wave-002/probe', 'wave-002/control'} <= set(ordinary_assessments)
    assert [r.spec.project_id for r in executions].count('original') == 2
    retries = [r for r in executions if 'original-retries' in str(r.branch_root)]
    assert retries[0].spec == executions[0].spec
    if ready_wave:
        order = [request.spec.project_id for request in executions]
        assert order.index('original', 1) < order.index('new-work')
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
@pytest.mark.parametrize("failed_retry", [False, True])
@pytest.mark.parametrize("retry_latency", [5, 11])
def test_external_phase2_retry_uses_frozen_suite_with_updated_committed_memory(tmp_path, monkeypatch, standalone, failed_retry, retry_latency):
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
        def __init__(self, latency, failed=False, native_latency=5):
            super().__init__()
            self.latency, self.failed, self.native_latency = latency, failed, native_latency
        def prepare(self, spec, directory):
            value = super().prepare(spec, directory)
            value.update(suite_sha256='a'*64, manifest=dict(policy=dict(rtol=.001), submission=spec))
            value['cases'][0]['timing']['torch']['latency_us'] = self.latency
            return value
        def evaluate(self, source, directory, **kwargs):
            result = dict(super().evaluate(source, directory, **kwargs), suite_sha256='a'*64)
            for case in result.get('cases', []):
                case['timing']['ascendc']['latency_us'] = self.native_latency
            if self.failed:
                result.update(status='incorrect', failure_case=dict(schema='failure-case-v1',
                    classification='host-mismatch', description='retry returned zero',
                    inputs={'x': [1]}, expected={'output': [1]}, observed={'output': [0]},
                    diagnostics={'index': 0}, policy={'exact': True}))
            return result
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
    received, children, now = [], [], [0]
    def factory(directory, task, memory, bound, **kwargs):
        received.append(memory)
        assert type(bound['phase2_attempt_seconds']) is int and bound['phase2_attempt_seconds'] == 30
        assert bound['max_cases'] == 2
        backend = FrozenBackend(12, failed=failed_retry, native_latency=retry_latency)
        backend.interrupt = not children
        child = OptimizerTaskExecutor(directory, task, memory, bound,
            agent=Agent([candidate('good')]), backend=backend, **kwargs)
        prepare = child.prepare
        def charged_prepare():
            value = prepare()
            now[0] += 2
            return value
        child.prepare = charged_prepare
        children.append(child)
        return child
    monkeypatch.setattr('benchmarks.ascend.phase2_optimizer_execution.OptimizerTaskExecutor', factory)
    runner = SimpleNamespace(root=source, initial_memory=b'', clock=lambda: now[0])
    retry = optimizer_retry_adapter(runner, store, dict(config, max_cases=1))
    from optimizer.control import ObservationUnavailable, WorkerStopped
    with pytest.raises(ObservationUnavailable):
        retry(row, tmp_path/'retried', 30.75)
    # Older retained children have no allocation receipt; recover their immutable submission.
    (tmp_path/'retried/retry-allocation.json').unlink()
    retry(row, tmp_path/'retried', 20.5)
    assert children[1].job_id == children[0].job_id
    assert children[1]._backend.prepares == 0
    assert children[1]._agent.contexts == []
    assert len(children[1]._backend.evaluations) == 1
    import time
    assert 0 < children[1].deadline - time.monotonic() <= 18.5
    retried = next(r for r in store.records() if r['feedback_id'] != row['feedback_id'])
    linked = store.link_original_retry(row['feedback_id'], retried['feedback_id'])
    # Remeasuring the reference at 12us cannot relax original optimizer acceptance at 10us.
    assert linked['lifecycle'] == ('failed' if failed_retry or standalone and retry_latency > 10 else 'fixed')
    if standalone and not failed_retry:
        assert retried['outcome']['result']['speedup'] > 1
        assert retried['outcome']['evidence']['result_artifact']
    if failed_retry:
        from benchmarks.ascend.failure_curriculum import FailureCurriculum
        from test_curriculum_context_files import read_all
        case, = store.failure_cases(retried['feedback_id'])
        assert case['availability']['complete']
        feedback = FailureCurriculum(source, store.root, dict(selected=[linked], controls=[]))
        index = feedback.context({})[-1]
        retry_manifest, = read_all(feedback.read, index['selected'][0]['linked_retries_file'])
        retained_case, = read_all(feedback.read, retry_manifest['failure_cases_file'])
        assert retained_case['artifacts'] == case['artifacts']
        for role, expected_value in dict(source='good', inputs={'x': [1]}, expected={'output': [1]},
                                         observed={'output': [0]}, diagnostics={'index': 0}).items():
            assert json.loads(feedback.read(case['artifacts'][role]['sha256'])['text']) == expected_value
    assert received[0]['phase'] == 'committed-phase1'
    assert len(received[0]['memory_entries']) == 1
    # The same pending child survives a lower parent allowance and changed memory.
    monkeypatch.setattr('optimizer.memory.freeze_committed_phase1', lambda *args: pytest.fail('snapshot refrozen'))
    retry = optimizer_retry_adapter(runner, store, dict(config, max_cases=1))
    retry(row, tmp_path/'retried', 12.5)
    assert children[1].job_id == children[0].job_id
    assert received[1] == received[0]
    assert children[1]._backend.prepares == 0
    assert children[2].job_id == children[0].job_id
    assert children[2]._backend.evaluations == []
    assert children[2]._agent.contexts == []
    # Export time belongs to this retry, so exhausted setup cannot dispatch.
    now = [0]
    runner.clock = lambda: now[0]
    def slow_export(*args):
        now[0] += 40
        return received[0]
    monkeypatch.setattr('optimizer.memory.freeze_committed_phase1', slow_export)
    slow_retry = optimizer_retry_adapter(runner, store, dict(config, max_cases=1))
    with pytest.raises(WorkerStopped, match='elapsed-time'):
        slow_retry(row, tmp_path/'setup-expired', 30.75)
    assert len(children) == 3


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


def test_interrupted_failure_clock_excludes_expired_obligation(tmp_path, monkeypatch):
    from test_generic_failure_feedback import phase1
    phase1(tmp_path/'source')
    runner = AdaptivePhase1Coordinator(tmp_path/'run', 'Bounded.', b'{}',
        CurriculumAgent(lambda _: ''), lambda _: pytest.fail('expired dispatch'), max_parallel=1, project_target=1)
    automatic = AutomaticRemediation(runner, target='a3', seconds=10,
        sources=[dict(type='phase1', root=str(tmp_path/'source'))])
    state = runner._load_state()
    automatic.refresh(state)
    budget = next(iter(automatic.budgets.values()))
    budget.path.parent.mkdir(parents=True, exist_ok=True)
    budget.path.write_text(json.dumps(dict(max_seconds=10, elapsed_seconds=2, active_since=100)))
    monkeypatch.setattr('time.time', lambda: 109)
    assert not automatic.refresh(state)
    assert next(iter(automatic.state['failures'].values()))['remaining_seconds'] == 0


def test_retry_rechecks_recovered_clock_inside_active(tmp_path, monkeypatch):
    from test_generic_failure_feedback import phase1
    phase1(tmp_path/'source')
    runner = AdaptivePhase1Coordinator(tmp_path/'run', 'Bounded.', b'{}',
        CurriculumAgent(lambda _: ''), lambda _: None, max_parallel=1, project_target=1)
    automatic = AutomaticRemediation(runner, target='a3', seconds=10,
        sources=[dict(type='phase1', root=str(tmp_path/'source'))])
    state = runner._load_state()
    automatic.refresh(state)
    key = next(iter(automatic.budgets))
    automatic.store.update_investigation(key, status='ready-to-retry', evidence=['probe'])
    automatic.refresh(state)
    budget = automatic.budgets[key]
    budget.path.parent.mkdir(parents=True, exist_ok=True)
    budget.path.write_text(json.dumps(dict(max_seconds=10, elapsed_seconds=2, active_since=100)))
    wall = iter([107, 109])
    monkeypatch.setattr('time.time', lambda: next(wall, 109))
    automatic._phase1_retry = lambda *args: pytest.fail('expired dispatch')
    automatic.retry_ready(state)
    assert automatic.state['failures'][key]['retries'] == []


@pytest.mark.parametrize('reviewed', [False, True])
def test_nested_original_retry_reads_parent_memory_artifacts_and_reviewed_seed(tmp_path, reviewed):
    import base64
    import shutil
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.evidence import canonical_bytes
    from benchmarks.a5kernels.phase2_adaptive_memory import admit_adaptive_snapshot
    from benchmarks.ascend.agent_presentation import MemoryArtifactReader
    from benchmarks.ascend.phase2_feedback import FeedbackStore
    from tests.test_a5_phase2_adaptive_memory import _source
    source_root, source, review, memory = _source(tmp_path/'history')
    parent = tmp_path/'parent'
    initial = b'{}'
    entries = [base64.b64encode(canonical_bytes(memory)).decode()]
    if reviewed:
        snapshot = admit_adaptive_snapshot(source_root, b'', review, source, parent)
        initial = canonical_bytes(dict(schema='ascend-phase2-curriculum-seed-v1',
            snapshot_id=snapshot.snapshot_id, phase1_memory=[memory]))
        entries = []
    requests = []
    def execute(request):
        requests.append(request)
        assert request.memory_root == parent
        assert request.branch_root.is_relative_to(parent/'original-retries')
        current = ContentAddressedArtifacts(request.branch_root/'artifacts')
        reader = MemoryArtifactReader(request, current)
        assert reader.read_bytes(memory['source_revision']) == b'actual candidate source'
        evidence = memory['host_facts'][0]['evidence_sha256']
        assert json.loads(reader.read_bytes(evidence))['host'] == 'verified'
        assert not current._path(evidence).exists()
        with pytest.raises(FileNotFoundError):
            reader.read_bytes('f'*64)
        return _result(request.spec, True, 'host pass')
    runner = AdaptivePhase1Coordinator(parent, 'Retry original.', initial,
        CurriculumAgent(lambda _: ''), execute, max_parallel=1, project_target=1)
    state = runner._load_state()
    state['memory_bytes_base64'] = entries
    if not reviewed:
        shutil.copytree(source_root/'waves', parent/'waves', dirs_exist_ok=True)
    retry = object.__new__(AutomaticRemediation)
    retry.runner, retry.target, retry.store = runner, 'a5', FeedbackStore(tmp_path/'feedback')
    row = dict(outcome=dict(original_spec=_project('original', 'gm-ub-movement')))
    directory = parent/'original-retries'/'failure'/'retry-1'
    retry._phase1_retry(row, directory, state)
    assert len(requests) == 1
    # Retained retry replay observes its committed result without another actor.
    retry._phase1_retry(row, directory, state)
    assert len(requests) == 1


@pytest.mark.parametrize('ready_wave', [False, True])
def test_failed_original_retry_checkpoint_survives_restart_until_new_ready_recommendation(tmp_path, ready_wave):
    from optimizer.jobs import atomic, canonical
    executions, stages = [], []
    automatic = None
    def complete(messages):
        stages.append(messages)
        if len(stages) == 1:
            return _wave(messages[-1]['content'], [_project('original', 'gm-ub-movement')])
        key = next(iter(automatic.feedback.ids))
        row = dict(feedback_id=key, classification='learnable', reason='isolate movement',
                   project_id='probe', control_project_id='control', hypothesis='alignment', observation='compare oracle')
        if len(stages) == 2:
            row['status'] = 'investigating'
            proposal = json.loads(_wave(messages[-1]['content'], [_project('probe', 'padded-tail'),
                _project('control', 'gm-ub-movement')]).split('\n', 1)[1])
            proposal['failure_investigations'] = [row]
            return 'DECISION: WAVE\n' + json.dumps(proposal)
        if len(stages) == 3:
            row['status'] = 'ready-to-retry'
            if ready_wave:
                proposal = json.loads(_wave(messages[-1]['content'], [_project('new-work', 'simd-vector')]).split('\n', 1)[1])
                proposal['failure_investigations'] = [row]
                return 'DECISION: WAVE\n' + json.dumps(proposal)
            return 'DECISION: SATURATED\n' + json.dumps(dict(reason='retry original', failure_investigations=[row]))
        raise RuntimeError('restart after failed original retry')
    def execute(request):
        executions.append(request.spec.project_id)
        passed = request.spec.project_id != 'original' or executions.count('original') >= 3
        return _result(request.spec, passed, 'host outcome')
    def build(provider):
        return AdaptivePhase1Coordinator(tmp_path/'run', 'Repair failures.', b'{}',
            CurriculumAgent(provider), execute, max_parallel=1, project_target=1)
    runner = build(complete)
    automatic = AutomaticRemediation(runner, target='a3')
    with pytest.raises(RuntimeError, match='restart after failed original retry'):
        runner.run()
    assert executions.count('original') == 2
    state_path = runner.root/'state.json'
    checkpoint = json.loads(state_path.read_bytes())
    update, = checkpoint['failure_investigations']
    assert update['status'] == 'investigating'
    key = update['feedback_id']
    # A fresh process replays the campaign checkpoint before selecting new work.
    def defer(messages):
        assert executions.count('original') == 2
        return 'DECISION: SATURATED\n' + json.dumps(dict(reason='await another investigation',
            failure_investigations=[dict(update, status='deferred', reason='await new evidence')]))
    resumed = build(defer)
    automatic = AutomaticRemediation(resumed, target='a3')
    resumed.resume()
    assert executions.count('original') == 2
    # Explicitly publishing a new recommendation permits exactly one next retry.
    state = json.loads(state_path.read_bytes())
    state['terminal'] = None
    state['failure_investigations'] = [dict(update, status='ready-to-retry', reason='new reviewed evidence')]
    from benchmarks.ascend.failure_curriculum import FailureCurriculum, select_feedback
    review = FailureCurriculum(resumed.root, automatic.store.root,
        select_feedback(automatic.store.root, 'a3', [key]))
    review.record(state['failure_investigations'], state)
    atomic(state_path, canonical(state))
    resumed.resume()
    assert executions.count('original') == 3
    assert next(row for row in automatic.store.records() if row['feedback_id'] == key)['lifecycle'] == 'fixed'
