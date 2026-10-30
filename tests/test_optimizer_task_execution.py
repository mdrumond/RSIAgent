from canonical_optimizer_fixtures import canonical_optimizer_backend, evidence_results, optimizer_inputs, retained_native, scripted_optimizer_http, unresolved_authority

import json

import pytest

from benchmarks.ascend.optimizer_execution import OptimizerTaskExecutor
from optimizer.control import ObservationUnavailable, WorkerStopped
from optimizer.jobs import canonical, sha


class Agent:
    canonical_fixture = True
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
    execution, agent, backend = executor(tmp_path, [candidate('bad'),
        {'action': 'read-memory', 'index': 0}, candidate('good')])
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
    inputs = optimizer_inputs(agent.contexts[-1][1])
    assert inputs['optimizer_context']['optimizer_task']['feedback'] == {'failure': 'strides'}
    assert inputs['memory']['snapshot_id'] == learned['snapshot_id']
    memory_reads = evidence_results(agent.contexts[-1][1], 'read_memory')
    assert len(memory_reads) == 1
    assert memory_reads[0]['lesson'] == learned['memory_entries'][0]['lesson']
    assert execution.store.submission(execution.job_id)['memory_snapshot'] == snapshot()
    content = json.loads(execution.store.read_artifact(execution.job_id, passed['result_artifact']['sha256']))
    assert content == {key: value for key, value in passed.items() if key != 'result_artifact'}
    source = json.loads(execution.store.read_artifact(execution.job_id, passed['candidate']['sha256']))
    assert sha(source.encode()) == passed['source_sha256']


@pytest.mark.parametrize('capability', ['absent', None, 'not callable'])
def test_real_pi_omits_unsupported_compile_and_completes_all_stages(tmp_path, monkeypatch, capability):
    from types import SimpleNamespace
    from rsi.contracts import Stage
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    # Keep the existing CPU host effects; expose no callable compile capability.
    runtime = SimpleNamespace(prepare=backend.prepare, evaluate=backend.evaluate, profile=backend.profile)
    if capability != 'absent':
        runtime.compile = capability
    execution._backend = runtime
    http = scripted_optimizer_http(monkeypatch, agent, compile_if_offered=True)
    result = execution.attempt('target-001')
    assert result['verified'] and result['speedup'] == 2
    assert http.requests and all('compile' not in {tool['function']['name']
        for tool in wire.get('tools', [])} for wire in http.requests)
    assert backend.prepares == 1 and backend.compilations == []
    assert [source for source, _ in backend.evaluations] == ['good']
    assert execution.store.get(execution.job_id)['attempts'] == 1
    native = retained_native(execution)
    assert native.workflow['pending'] is None
    assert {row['evidence']['stage'] for row in native.workflow['history']} == {stage.value for stage in Stage}
    assert execution._state()['canonical_active']['status'] == 'accepted'
    assert native.store.verify()
    source = json.loads(execution.store.read_artifact(execution.job_id, result['candidate']['sha256']))
    assert source == 'good' and sha(source.encode()) == result['source_sha256']
    baseline = execution.prepare()
    assert baseline['cases'][0]['case'] == 'case-000'
    assert baseline['suite_sha256'] == result['result']['suite_sha256']
    runner = http.session.owner.fixture_runner
    for row in native.workflow['history']:
        operation = row['evidence']['operation_id']
        reads = [event for event in native.events if event.get('operation_id') == operation
                 and event['type'] == 'skill-usage' and event.get('kind') == 'read']
        assert {event['skill'] for event in reads} == set(runner.bundle.skills_for(Stage(row['evidence']['stage'])))
        receipts = [native.store.read_json(ref) for ref in row['evidence']['skill_refs']]
        assert {receipt['payload']['receipt_id'] for receipt in receipts} == {event['receipt_id'] for event in reads}
        assert all(receipt['context']['operation'] == operation for receipt in receipts)
    queries = [event for event in native.events if event['type'] == 'query-dispatch']
    assert queries
    finished = [event for event in native.events if event['type'] == 'query-finished']
    assert {event['call_id'] for event in queries} == {event['call_id'] for event in finished}
    assert all(event['status'] == 'complete' for event in finished)
    for event in finished:
        receipt = native.store.read_json(event['reference'])
        assert receipt['context']['operation'] == event['operation_id']
        assert receipt['payload']['response']['results']
    assert result['result']['candidate_sha256'] == result['source_sha256']
    assert [case['case'] for case in result['result']['cases']] == [case['case'] for case in baseline['cases']]


def test_compile_repairs_do_not_charge_candidate_budget(tmp_path, monkeypatch):
    execution, agent, backend = executor(tmp_path, [candidate('bad', 'compile'),
        candidate('good', 'compile'), candidate('good')])
    http = scripted_optimizer_http(monkeypatch, agent)
    result = execution.attempt('target-001')
    assert result['verified']
    assert any(tool['function']['name'] == 'compile' for wire in http.requests for tool in wire.get('tools', []))
    assert backend.compilations == ['bad', 'good']
    assert len(backend.evaluations) == 1
    assert execution.store.get(execution.job_id)['attempts'] == 1
    reports = evidence_results(agent.contexts[-1][1], 'compile')
    assert [report['status'] for report in reports] == ['compile-error', 'compiled']


def test_interrupted_pending_candidate_requires_observation_without_replay(tmp_path):
    from rsi.runtime import WorkflowError
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    backend.interrupt = True
    with pytest.raises(ObservationUnavailable, match='retained handle unavailable'):
        execution.attempt('target-001')
    before = retained_native(execution)
    budget = execution.store.get(execution.job_id)
    assert before.workflow['pending'] is not None and budget['attempts'] == 1
    assert any(row['type'] == 'domain-interrupted' for row in before.events)
    calls = (len(agent.contexts), len(backend.evaluations), backend.prepares)
    with pytest.raises(ValueError, match='pending attempt'):
        execution.attempt('target-002')
    with pytest.raises(WorkflowError, match='reconciliation'):
        execution.attempt('target-001')
    assert unresolved_authority(retained_native(execution)) == unresolved_authority(before)
    assert (len(agent.contexts), len(backend.evaluations), backend.prepares) == calls
    after = execution.store.get(execution.job_id)
    assert all(after[key] == budget[key] for key in ('max_seconds', 'max_attempts', 'attempts'))
    assert after['elapsed_seconds'] >= budget['elapsed_seconds']
    assert execution._event('task-attempt-result') is None


def test_crash_after_evaluation_before_phase2_commit_does_not_evaluate_again(tmp_path, monkeypatch):
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    original = execution.store.add_event
    def crash(job_id, kind, payload):
        if kind == 'task-attempt-result':
            raise RuntimeError('process stopped before result commit')
        return original(job_id, kind, payload)
    monkeypatch.setattr(execution.store, 'add_event', crash)
    with pytest.raises(RuntimeError, match='before result commit'):
        execution.attempt('target-001')
    monkeypatch.setattr(execution.store, 'add_event', original)
    assert execution.attempt('target-001')['verified']
    assert len(backend.evaluations) == 1


def test_unregistered_finish_is_refused_before_candidate_or_elapsed_stop(tmp_path):
    for recover in (True, False):
        execution, _, backend = executor(tmp_path / str(recover), [])
        class Clock:
            now = 0
            def __call__(self): return self.now
        clock = Clock()
        class RefusalAgent(Agent):
            feedback = []
            def complete(self, stage, context, record):
                observation = context['observation'] or {}
                if observation.get('event') == 'invalid-action':
                    self.feedback.append(observation)
                    assert observation['accepted'] is False
                    assert backend.prepares == 1 and backend.evaluations == []
                    assert execution.store.get(execution.job_id)['attempts'] == 0
                    if not recover:
                        clock.now += 7200
                return super().complete(stage, context, record)
        agent = RefusalAgent([{'action': 'cases', 'signatures': [], 'summary': ''},
            {'action': 'finish', 'summary': 'cannot implement'}, candidate('good')])
        agent.canonical_clock = clock
        execution._agent = agent
        if recover:
            assert execution.attempt('target-001')['verified']
            assert len(backend.evaluations) == 1
            assert execution._state()['canonical_active']['status'] == 'accepted'
            from rsi.contracts import Stage
            assert {row['evidence']['stage'] for row in retained_native(execution).workflow['history']} == {stage.value for stage in Stage}
        else:
            with pytest.raises(WorkerStopped, match='without a candidate result'):
                execution.attempt('target-001')
            assert backend.evaluations == []
            assert execution.store.get(execution.job_id)['status'] == 'budget-exhausted'
            assert execution.store.get(execution.job_id)['attempts'] == 0
            assert execution._event('task-attempt-result') is None
        assert len(agent.feedback) == 1 and backend.prepares == 1


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


def test_large_memory_and_feedback_are_bounded_but_artifacts_remain_readable(tmp_path, monkeypatch):
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    agent.actions = iter([{'action': 'cases', 'signatures': [], 'summary': ''},
        {'action': 'read-memory', 'index': 217}, candidate('good')])
    http = scripted_optimizer_http(monkeypatch, agent)
    learned = snapshot([{'lesson': f'lesson-{i}-' + 'x' * 12000} for i in range(218)])
    result = execution.attempt('target-001', memory_snapshot=learned,
        feedback={'errors': 'huge compiler diagnostics' * 20000})
    context = agent.contexts[-1][1]
    assert http.requests and all(len(canonical(request)) <= 49152 for request in http.requests)
    assert optimizer_inputs(context)['memory']['snapshot_id'] == learned['snapshot_id']
    reads = evidence_results(context, 'read_memory')
    assert len(reads) == 1 and reads[0]['lesson'] == learned['memory_entries'][217]['lesson']
    request = execution._event('task-attempt-request')
    feedback_record = request['feedback']
    assert json.loads(execution.store.read_artifact(execution.job_id, feedback_record['sha256'])) == {
        'errors': 'huge compiler diagnostics' * 20000}
    assert result['verified']


def test_large_baseline_profile_is_hash_readable_not_replayed_into_actor_prompt(tmp_path, monkeypatch):
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    http = scripted_optimizer_http(monkeypatch, agent)
    original_prepare = backend.prepare
    def prepare(spec, directory):
        return {**original_prepare(spec, directory), 'profiling': {'raw_report': 'metrics' * 100000}}
    backend.prepare = prepare
    assert execution.attempt('target-001')['verified']
    assert http.requests and all(len(canonical(request)) <= 49152 for request in http.requests)
    baseline = execution.prepare()
    assert baseline['cases'][0]['contract'] == {}
    assert baseline['profiling']['raw_report'] == 'metrics' * 100000
    reference = execution._state()['baseline']
    assert json.loads(execution.store.read_artifact(execution.job_id, reference['sha256'])) == baseline


def test_read_artifact_pages_are_contiguous_and_visible_in_context(tmp_path):
    execution, agent, backend = executor(tmp_path, [])
    raw = b'a' * 8192 + b'NEXT' + b'b' * 8192
    artifact = execution.store.put_artifact(execution.job_id, 'long-log', raw)
    agent.actions = iter([{'action': 'cases', 'signatures': [], 'summary': ''},
        *[{'action': 'read-artifact', 'sha256': artifact['sha256'], 'offset': offset}
          for offset in (0, 8192, 16384)], candidate('good')])
    assert execution.attempt('target-001')['verified']
    pages = evidence_results(agent.contexts[-1][1], 'read_artifact')
    assert len(pages) == 3
    assert ''.join(page['text'] for page in pages).encode() == raw
    assert [page['next_offset'] for page in pages] == [8192, 16384, None]


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
    pages = evidence_results(agent.contexts[-1][1], 'read_artifact')
    assert len(pages) == 1 and pages[0]['text'].encode() == source
    assert pages[0]['next_offset'] is None


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
    request = execution._event('task-attempt-request')
    assert len(canonical(request)) < 1000
    for key, expected in [('memory_snapshot', memory), ('feedback', feedback)]:
        assert json.loads(execution.store.read_artifact(execution.job_id, request[key]['sha256'])) == expected
    assert (execution.job_directory / 'events.jsonl').stat().st_size < 250000


def test_plain_profile_deadline_without_terminal_proof_remains_uncertain(tmp_path):
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    agent.profile = True
    def profile(*args, **kwargs):
        backend.profiles.append('unobserved')
        raise WorkerStopped('deadline expired during optional diagnostics')
    backend.profiles = []
    backend.profile = profile
    with pytest.raises(WorkerStopped, match='deadline expired during optional diagnostics'):
        execution.attempt('target-001')
    before, budget = retained_native(execution), execution.store.get(execution.job_id)
    assert before.workflow['pending'] is not None and before.workflow['stage'] == 'optimisation'
    assert execution._state()['history'][0]['content']['status'] == 'verified'
    with pytest.raises(ObservationUnavailable):
        execution.attempt('target-001')
    assert unresolved_authority(retained_native(execution)) == unresolved_authority(before)
    assert len(backend.evaluations) == 1 and backend.profiles == ['unobserved']
    assert execution._event('profiling-unavailable') is None
    assert execution._event('task-attempt-result') is None
    after = execution.store.get(execution.job_id)
    assert all(after[key] == budget[key] for key in ('max_seconds', 'max_attempts', 'attempts'))
    assert after['elapsed_seconds'] >= budget['elapsed_seconds']


@pytest.mark.parametrize('failure', ['explicit-stop', 'uncertain-cancellation'])
def test_optional_profile_honors_stop_and_uncertain_observation(tmp_path, failure):
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    agent.profile = True
    def profile(*args, **kwargs):
        if failure == 'explicit-stop':
            execution.store.stop(execution.job_id)
            raise WorkerStopped('explicit user stop')
        raise ObservationUnavailable('cancellation still uncertain')
    backend.profile = profile
    with pytest.raises(WorkerStopped if failure == 'explicit-stop' else ObservationUnavailable):
        execution.attempt('target-001')
    native = retained_native(execution)
    assert native.workflow['pending'] is not None and native.workflow['stage'] == 'optimisation'
    assert execution._state()['history'][0]['content']['status'] == 'verified'
    assert execution._event('task-attempt-result') is None


def test_failed_evaluation_checkpoint_recovers_same_attempt_without_new_allocation(tmp_path, monkeypatch):
    from rsi.domain_execution import DomainRunner
    from rsi.domain_checkpoint import decode
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    original = DomainRunner.append
    def crash(runner, event):
        original(runner, event)
        if event['type'] == 'domain-result':
            result = runner.store.read_json(event['reference'])
            dispatch = runner.store.read_json(result['dispatch_ref'])
            if dispatch['name'] == 'evaluate' and result['observed_terminal']:
                raise RuntimeError('evaluation checkpoint crash')
    monkeypatch.setattr(DomainRunner, 'append', crash)
    with pytest.raises(RuntimeError, match='evaluation checkpoint crash'):
        execution.attempt('target-001')
    before = execution.store.get(execution.job_id)
    native = retained_native(execution)
    assert native.workflow['pending'] is not None and native.workflow['stage'] == 'test'
    observations = [native.store.read_json(row['reference']) for row in native.events
                    if row['type'] == 'domain-result']
    evaluation = next(row for row in observations
                      if native.store.read_json(row['dispatch_ref'])['name'] == 'evaluate')
    assert evaluation['observed_terminal'] is True
    assert decode(evaluation['result'])['status'] == 'verified'
    calls = (len(agent.contexts), len(backend.evaluations), backend.prepares)
    monkeypatch.setattr(DomainRunner, 'append', original)
    result = execution.attempt('target-001')
    assert result['verified']
    assert execution.attempt('target-001') == result
    assert (len(agent.contexts), len(backend.evaluations), backend.prepares) == calls
    after = execution.store.get(execution.job_id)
    for key in ('max_seconds', 'max_attempts', 'attempts', 'latest_attempt'):
        assert after[key] == before[key]
    assert after['elapsed_seconds'] >= before['elapsed_seconds']
    assert after['attempts'] == 1
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
    execution, agent, backend = executor(tmp_path, [candidate('good')])
    agent.profile = True
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
        from pathlib import Path
        proof = execution._event('profiling-unavailable')['terminal_receipt']
        receipt = Path(proof['path'])
        assert receipt.resolve().is_relative_to((execution.job_directory / 'execution').resolve())
        raw = receipt.read_bytes()
        assert sha(raw) == proof['sha256']
        retained = json.loads(raw)
        assert retained['state'] == proof['state'] == 'cancelled'
        assert retained['native_handle'] == proof['handle'] == 'remote:bz-a3-1:job:profile-test'
        assert not execution.store.get(execution.job_id)['stop_requested']
        assert sum(argv[-2:] == ['cancel', proof['handle']] for argv in calls) == 1
    else:
        with pytest.raises(WorkerStopped if boundary == 'user-stop' else ObservationUnavailable):
            execution.attempt('target-001')
        native = retained_native(execution)
        assert native.workflow['pending'] is not None
        assert native.workflow['stage'] == 'optimisation'
        assert execution._state()['history'][0]['content']['status'] == 'verified'
        assert execution._event('task-attempt-result') is None
    assert len(backend.evaluations) == 1
