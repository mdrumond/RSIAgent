import json

import pytest

from benchmarks.ascend.optimizer_execution import OptimizerTaskExecutor
from optimizer.control import ObservationUnavailable, WorkerStopped
from optimizer.jobs import canonical, sha


class Agent:
    identity = {'model': 'fixed-test'}

    def __init__(self, actions):
        self.actions = iter(actions)
        self.contexts = []

    def complete(self, stage, context, record):
        self.contexts.append((stage, context))
        return next(self.actions)


class Backend:
    def __init__(self):
        self.prepares, self.evaluations, self.compilations = 0, [], []
        self.interrupt = False

    def prepare(self, spec, directory):
        self.prepares += 1
        return {'suite_sha256': 'suite', 'cases': [
            {'case': 'case-000', 'contract': {}, 'timing': {'torch': {'latency_us': 10}}}]}

    def evaluate(self, source, directory, *, block_count):
        if self.interrupt:
            self.interrupt = False
            raise ObservationUnavailable('retained handle unavailable')
        self.evaluations.append((source, str(directory)))
        return {'status': 'verified' if source != 'bad' else 'incorrect',
            'candidate_sha256': sha(source.encode()), 'suite_sha256': 'suite', 'cases': [
                {'case': 'case-000', 'timing': {'ascendc': {'latency_us': 5}}}]}

    def compile(self, source, directory, *, block_count):
        self.compilations.append(source)
        return {'status': 'compile-error' if source == 'bad' else 'compiled'}

    def profile(self, source, directory, *, block_count):
        return {'status': 'unavailable', 'reason': 'test'}


def snapshot(entries=()):
    body = {'files': {}, 'memory_entries': list(entries)}
    return {**body, 'snapshot_id': sha(canonical(body))}


def candidate(source, action='candidate'):
    return {'action': action, 'source': source, 'block_count': 1, 'summary': source}


def executor(tmp_path, actions):
    task = {'task_id': 'matmul-residual', 'submission': {
        'source': 'def forward(a,b,r): return (a @ b + r).relu()', 'entrypoint': 'forward',
        'signatures': [{'args': [{'shape': shape, 'dtype': 'float32'} for shape in ([2, 3], [3, 4], [2, 4])]}]},
        'rationale': 'composed graph', 'citations': []}
    config = {'target': 'bz-a3-1', 'physical_device': 14,
              'validation_wrapper': str(tmp_path / 'validation.sh'), 'max_cases': 1}
    agent = Agent([{'action': 'cases', 'signatures': [], 'summary': 'representative'}, *actions])
    backend = Backend()
    return OptimizerTaskExecutor(tmp_path / 'task', task, snapshot(), config, agent=agent, backend=backend), agent, backend


def test_prepare_and_single_attempts_reuse_baseline_and_authenticated_results(tmp_path):
    execution, agent, backend = executor(tmp_path, [candidate('bad'), candidate('good')])
    first_baseline = execution.prepare()
    assert execution.prepare() == first_baseline
    failed = execution.attempt('target-001')
    assert not failed['verified']
    assert execution.attempt('target-001') == failed
    learned = snapshot([{'lesson': 'repair strides'}])
    passed = execution.attempt('target-002', memory_snapshot=learned, feedback={'failure': 'strides'})
    assert passed['verified'] and passed['speedup'] == 2
    assert backend.prepares == 1
    assert [source for source, _ in backend.evaluations] == ['bad', 'good']
    assert execution.store.get(execution.job_id)['attempts'] == 2
    assert agent.contexts[-1][1]['phase2']['feedback'] == {'failure': 'strides'}
    assert 'repair strides' in str(agent.contexts[-1][1]['memory_summary'])
    assert execution.store.submission(execution.job_id)['memory_snapshot'] == snapshot()
    content = json.loads(execution.store.read_artifact(execution.job_id, passed['result_artifact']['sha256']))
    assert content == {key: value for key, value in passed.items() if key != 'result_artifact'}
    source = json.loads(execution.store.read_artifact(execution.job_id, passed['candidate']['sha256']))
    assert sha(source.encode()) == passed['source_sha256']


def test_compile_repairs_do_not_charge_candidate_budget(tmp_path):
    execution, agent, backend = executor(tmp_path, [candidate('bad', 'compile'),
        candidate('good', 'compile'), candidate('good')])
    result = execution.attempt('target-001')
    assert result['verified']
    assert backend.compilations == ['bad', 'good']
    assert len(backend.evaluations) == 1
    assert execution.store.get(execution.job_id)['attempts'] == 1
    assert 'compile-error' in str(agent.contexts[-1][1]['history'])


def test_interrupted_pending_candidate_resumes_without_new_model_action(tmp_path):
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    backend.interrupt = True
    with pytest.raises(ObservationUnavailable):
        execution.attempt('target-001')
    assert execution.store.get(execution.job_id)['attempts'] == 1
    with pytest.raises(ValueError, match='pending attempt'):
        execution.attempt('target-002')
    result = execution.attempt('target-001')
    assert result['verified']
    assert backend.prepares == 1 and len(backend.evaluations) == 1
    assert len(agent.contexts) == 2
    assert execution.store.get(execution.job_id)['attempts'] == 1


def test_crash_after_evaluation_before_phase2_commit_does_not_evaluate_again(tmp_path, monkeypatch):
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    original = execution.store.add_event
    def crash(job_id, kind, payload):
        if kind == 'phase2-attempt-result':
            raise RuntimeError('process stopped before result commit')
        return original(job_id, kind, payload)
    monkeypatch.setattr(execution.store, 'add_event', crash)
    with pytest.raises(RuntimeError, match='before result commit'):
        execution.attempt('target-001')
    monkeypatch.setattr(execution.store, 'add_event', original)
    assert execution.attempt('target-001')['verified']
    assert len(backend.evaluations) == 1


def test_finish_without_candidate_is_not_verified_and_next_attempt_can_run(tmp_path):
    execution, agent, backend = executor(tmp_path, [{'action': 'finish', 'summary': 'cannot implement'}, candidate('good')])
    assert not execution.attempt('target-001')['verified']
    assert execution.attempt('target-002')['verified']
    assert len(backend.evaluations) == 1


def test_changed_task_and_corrupt_updated_memory_rejected(tmp_path):
    execution, _, _ = executor(tmp_path, [])
    bad = snapshot()
    bad['memory_entries'] = ['not authenticated']
    with pytest.raises(ValueError, match='snapshot digest'):
        execution.attempt('target-001', memory_snapshot=bad)
    with pytest.raises(ValueError, match='frozen task'):
        OptimizerTaskExecutor(execution.directory, {**execution.task, 'rationale': 'changed'},
            snapshot(), execution.config, agent=execution._agent, backend=execution._backend)


def test_compile_candidate_never_launches_and_reports_errors(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from optimizer import native
    from optimizer.reference import prepare_reference
    prepared = prepare_reference({'source': 'def f(x): return x + 1', 'entrypoint': 'f',
        'signatures': [{'args': [{'shape': [3], 'dtype': 'float32'}]}]}, max_cases=1)
    def compile_only(source, contract, directory, **kwargs):
        return lambda *args: pytest.fail('compile-only kernel launched'), {'binary_sha256': 'fake'}
    monkeypatch.setattr(native, '_compile_case', compile_only)
    assert native.compile_candidate(prepared, 'source', tmp_path / 'good', device='cpu', arch='test')['status'] == 'compiled'
    def failure(source, contract, directory, **kwargs):
        directory.mkdir()
        (directory / 'compile.log').write_text('unsupported API')
        raise RuntimeError('compile failed')
    monkeypatch.setattr(native, '_compile_case', failure)
    result = native.compile_candidate(prepared, 'source', tmp_path / 'bad', device='cpu', arch='test')
    assert result['status'] == 'compile-error'
    assert result['cases'][0]['compiler_diagnostics'] == 'unsupported API'


def test_large_memory_and_feedback_are_bounded_but_artifacts_remain_readable(tmp_path):
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    learned = snapshot([{'lesson': f'lesson-{i}-' + 'x' * 12000} for i in range(218)])
    result = execution.attempt('target-001', memory_snapshot=learned,
        feedback={'errors': 'huge compiler diagnostics' * 20000})
    context = agent.contexts[-1][1]
    assert len(canonical(context)) < 60000
    assert len(context['memory_summary']) == 218
    assert context['memory_summary'][-1]['index'] == 217
    feedback_record = context['phase2']['artifact']
    assert b'huge compiler diagnostics' in execution.store.read_artifact(execution.job_id, feedback_record['sha256'])
    assert result['verified']


def test_large_baseline_profile_is_hash_readable_not_replayed_into_actor_prompt(tmp_path):
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    original_prepare = backend.prepare
    def prepare(spec, directory):
        return {**original_prepare(spec, directory), 'profiling': {'raw_report': 'metrics' * 100000}}
    backend.prepare = prepare
    assert execution.attempt('target-001')['verified']
    context = agent.contexts[-1][1]
    assert len(canonical(context)) < 20000
    assert context['baseline']['cases'][0]['contract'] == {}
    profile = context['baseline']['profiling']
    assert profile['truncated']
    assert b'raw_report' in execution.store.read_artifact(execution.job_id, profile['artifact']['sha256'])


def test_read_artifact_pages_are_contiguous_and_visible_in_context(tmp_path):
    execution, agent, backend = executor(tmp_path, [])
    artifact = execution.store.put_artifact(execution.job_id, 'long-log', b'a' * 1500 + b'NEXT' + b'b' * 1500)
    agent.actions = iter([{'action': 'cases', 'signatures': [], 'summary': ''},
        {'action': 'read-artifact', 'sha256': artifact['sha256'], 'offset': 0},
        {'action': 'read-artifact', 'sha256': artifact['sha256'], 'offset': 1500}, candidate('good')])
    assert execution.attempt('target-001')['verified']
    preview = json.loads(agent.contexts[-1][1]['history'][-1]['content_preview'])
    assert preview['text'].startswith('NEXT') and preview['next_offset'] == 3000


def test_read_historical_source_from_authenticated_memory_snapshot(tmp_path):
    import base64
    execution, agent, backend = executor(tmp_path, [])
    source = b'// actual prior verified kernel source'
    body = {'files': {'evidence/kernel.cpp': {'sha256': sha(source),
        'base64': base64.b64encode(source).decode()}}, 'memory_entries': []}
    memory = {**body, 'snapshot_id': sha(canonical(body))}
    agent.actions = iter([{'action': 'cases', 'signatures': [], 'summary': ''},
        {'action': 'read-artifact', 'sha256': sha(source), 'offset': 0}, candidate('good')])
    assert execution.attempt('target-001', memory_snapshot=memory)['verified']
    assert source.decode() in agent.contexts[-1][1]['history'][-1]['content_preview']


def test_task_directories_have_distinct_jobs_and_remote_workspaces(tmp_path):
    from optimizer.worker import RemoteBackend
    first, _, _ = executor(tmp_path / 'first', [])
    second, _, _ = executor(tmp_path / 'second', [])
    assert first.job_id != second.job_id
    assert OptimizerTaskExecutor(first.directory, first.task, snapshot(), first.config).job_id == first.job_id
    roots = [RemoteBackend(item.store, item.job_id, {'remote_workspace': '/configured/workspace'}).remote
             for item in (first, second)]
    assert roots[0] != roots[1]


def test_completed_attempt_rejects_changed_memory_or_feedback(tmp_path):
    execution, _, _ = executor(tmp_path, [candidate('good')])
    memory = snapshot([{'lesson': 'original'}])
    result = execution.attempt('target-001', memory_snapshot=memory, feedback={'failure': 'original'})
    assert execution.attempt('target-001', memory_snapshot=memory, feedback={'failure': 'original'}) == result
    for kwargs in ({'memory_snapshot': snapshot([{'lesson': 'changed'}]), 'feedback': {'failure': 'original'}},
                   {'memory_snapshot': memory, 'feedback': {'failure': 'changed'}}):
        with pytest.raises(ValueError, match='original memory and feedback'):
            execution.attempt('target-001', **kwargs)


@pytest.mark.parametrize('key', ['cpl_remote', 'validation_wrapper'])
def test_transport_identity_cannot_change_on_resume(tmp_path, key):
    execution, _, _ = executor(tmp_path, [])
    config = {**execution.config, key: '/different/transport'}
    with pytest.raises(ValueError, match='execution configuration changed'):
        OptimizerTaskExecutor(execution.directory, execution.task, snapshot(), config)


def test_attempt_input_events_contain_only_hash_references_not_bulk_memory(tmp_path):
    execution, _, _ = executor(tmp_path, [candidate('good')])
    memory = snapshot([{'lesson': 'x' * 1000000}])
    feedback = {'error': 'y' * 1000000}
    execution.attempt('target-001', memory_snapshot=memory, feedback=feedback)
    request = execution._event('phase2-attempt-request')
    assert len(canonical(request)) < 1000
    for key, expected in [('memory_snapshot', memory), ('feedback', feedback)]:
        assert json.loads(execution.store.read_artifact(execution.job_id, request[key]['sha256'])) == expected
    assert (execution.job_directory / 'events.jsonl').stat().st_size < 250000


def test_optional_profile_deadline_does_not_lose_verified_evaluation(tmp_path):
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    def profile(*args, **kwargs):
        execution.store.account_time(execution.job_id, 7201)
        raise WorkerStopped('deadline expired during optional diagnostics')
    backend.profile = profile
    result = execution.attempt('target-001')
    assert result['verified']
    assert 'pending' not in execution._state()
    assert execution.attempt('target-001') == result
    assert len(backend.evaluations) == 1
    assert any(event['kind'] == 'profiling-unavailable' for event in execution.store.events(execution.job_id))


@pytest.mark.parametrize('failure', ['explicit-stop', 'uncertain-cancellation'])
def test_optional_profile_honors_stop_and_uncertain_observation(tmp_path, failure):
    execution, _, backend = executor(tmp_path, [candidate('good')])
    def profile(*args, **kwargs):
        if failure == 'explicit-stop':
            execution.store.stop(execution.job_id)
            raise WorkerStopped('explicit user stop')
        raise ObservationUnavailable('cancellation still uncertain')
    backend.profile = profile
    with pytest.raises(WorkerStopped if failure == 'explicit-stop' else ObservationUnavailable):
        execution.attempt('target-001')
    assert 'evaluation' in execution._state()['pending']
    assert execution._event('phase2-attempt-result') is None


def test_failed_evaluation_checkpoint_recovers_same_attempt_without_new_allocation(tmp_path, monkeypatch):
    from optimizer.loop import OptimizationLoop
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    original = OptimizationLoop._save
    def crash(loop):
        original(loop)
        pending = loop.state.get('pending', {})
        if pending.get('evaluation') and not pending.get('history_recorded'):
            raise RuntimeError('evaluation checkpoint crash')
    monkeypatch.setattr(OptimizationLoop, '_save', crash)
    with pytest.raises(RuntimeError, match='evaluation checkpoint crash'):
        execution.attempt('target-001')
    before = execution.store.get(execution.job_id)
    assert before['status'] == 'failed'
    assert 'evaluation' in execution._state()['pending']
    calls = (len(agent.contexts), len(backend.evaluations), backend.prepares)
    monkeypatch.setattr(OptimizationLoop, '_save', original)
    result = execution.attempt('target-001')
    assert result['verified']
    assert execution.attempt('target-001') == result
    assert (len(agent.contexts), len(backend.evaluations), backend.prepares) == calls
    recovered = next(event['payload']['manifest'] for event in execution.store.events(execution.job_id)
                     if event['kind'] == 'recovered')
    for key in ('max_seconds', 'elapsed_seconds', 'max_attempts', 'attempts', 'latest_attempt'):
        assert recovered[key] == before[key]
    after = execution.store.get(execution.job_id)
    assert after['max_seconds'] == before['max_seconds']
    assert after['attempts'] == before['attempts'] == 1
    assert not any(event['kind'] == 'resumed' for event in execution.store.events(execution.job_id))


def test_failed_job_recovery_preserves_elapsed_exhaustion(tmp_path):
    execution, _, _ = executor(tmp_path, [])
    execution.store.account_time(execution.job_id, 7200)
    execution.store.transition(execution.job_id, 'failed')
    before = execution.store.get(execution.job_id)
    recovered = execution.store.recover(execution.job_id)
    assert recovered == {**before, 'status': 'budget-exhausted', 'stage': 'budget-exhausted'}


def test_failed_job_recovery_honors_explicit_stop(tmp_path):
    execution, _, _ = executor(tmp_path, [])
    execution.store.transition(execution.job_id, 'failed')
    execution.store.stop(execution.job_id)
    before = execution.store.get(execution.job_id)
    with pytest.raises(ValueError, match='without a stop request'):
        execution.store.recover(execution.job_id)
    assert execution.store.get(execution.job_id) == before


@pytest.mark.parametrize('boundary', ['deadline', 'user-stop', 'client-timeout', 'failed-cancel'])
def test_real_transport_interruption_after_evaluation_preserves_only_settled_budget_result(tmp_path, boundary):
    import time
    from types import SimpleNamespace
    from optimizer.transport import CheckedTransport
    execution, _, backend = executor(tmp_path, [candidate('good')])
    stopped = False
    calls = []
    transport = None
    def run(argv, **kwargs):
        nonlocal stopped
        calls.append(argv)
        if 'cancel' in argv:
            return SimpleNamespace(returncode=1 if boundary == 'failed-cancel' else 0, stdout='', stderr='')
        if boundary == 'user-stop':
            execution.store.stop(execution.job_id)
            stopped = True
        elif boundary != 'client-timeout':
            transport.deadline = backend.deadline = time.monotonic() - 1
            stopped = True
        return SimpleNamespace(returncode=-15, interrupted=True, stderr='', stdout=(
            'BZ_A3_REMOTE_HANDLE=remote:bz-a3-1:job:profile-test\n'
            'CATLASS_VALIDATION_HANDLE=bz-a3-1:profile-test\n'))
    transport = CheckedTransport({'validation_wrapper': '/profiles/validate', 'profile': 'bz-a3-1',
        'runtime': 'py311-torch', 'physical_device': 0, 'cpl_remote': '/cpl-remote'},
        run=run, stopped=lambda: stopped)
    backend.profile = lambda source, directory, **kwargs: transport.execute(directory, ['true'])
    if boundary == 'deadline':
        result = execution.attempt('target-001')
        assert result['verified']
        assert execution.attempt('target-001') == result
        assert 'pending' not in execution._state()
    else:
        with pytest.raises(WorkerStopped if boundary == 'user-stop' else ObservationUnavailable):
            execution.attempt('target-001')
        assert 'evaluation' in execution._state()['pending']
        assert execution._event('phase2-attempt-result') is None
    assert len(backend.evaluations) == 1
