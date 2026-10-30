import json
import pytest

from optimizer.jobs import JobStore, canonical, sha
from optimizer.loop import OptimizationLoop


def fixture(tmp_path, attempts=3, seconds=7200):
    body = {'files': {}, 'memory_entries': []}
    snapshot = {**body, 'snapshot_id': sha(canonical(body))}
    store = JobStore(tmp_path)
    job = store.create('def forward(x): return x', 'forward',
                       {'args': [{'shape': [5], 'dtype': 'float32'}]}, snapshot,
                       max_attempts=attempts, max_seconds=seconds)
    return store, job['job_id']


class Agent:
    identity = {'model': 'fake-fixed'}

    def __init__(self, actions):
        self.actions = iter(actions)

    def complete(self, stage, context, record):
        action = next(self.actions)
        record('response', {'text': json.dumps(action), 'completion_tokens': 1})
        return action


class Backend:
    def __init__(self):
        self.prepares = 0
        self.evaluations = []
        self.profiles = []

    def prepare(self, spec, directory):
        self.prepares += 1
        return {'suite_sha256': 'suite', 'cases': [
            {'case': 'case-000', 'contract': {}, 'timing': {'torch': {'latency_us': 10}}}]}

    def evaluate(self, source, directory, *, block_count):
        self.evaluations.append(source)
        if source == 'bad':
            return {'status': 'incorrect', 'suite_sha256': 'suite', 'cases': []}
        return {'status': 'verified', 'suite_sha256': 'suite', 'cases': [
            {'case': 'case-000', 'correctness': {}, 'timing': {
                'ascendc': {'latency_us': 20 if source == 'slow' else 5}}}]}

    def profile(self, source, directory, *, block_count):
        self.profiles.append(source)
        return {'status': 'unavailable', 'reason': 'fake profiler'}


def candidate(source):
    return {'action': 'candidate', 'source': source, 'block_count': 1, 'summary': source}


@pytest.mark.parametrize('action', [
    {'action': 'read-artifact', 'sha256': '01d481b2410940ecfb8c6765cc4d70c22ef72d73bde972', 'offset': 0},
    {'action': 'read-artifact', 'sha256': 'a' * 64, 'offset': 0},
    {'action': 'read-artifact', 'sha256': 'a' * 64, 'offset': -1},
    {'action': 'read-artifact', 'offset': 0},
    {'action': 'read-memory', 'index': 99},
    {'action': 'read-memory'},
])
def test_bad_evidence_read_returns_feedback_before_candidate(tmp_path, action):
    store, key = fixture(tmp_path, attempts=1)
    backend = Backend()

    class RepairAgent(Agent):
        def complete(self, stage, context, record):
            if context.get('history'):
                assert json.loads(context['history'][-1]['content_preview'])['status'] == 'tool-error'
            return super().complete(stage, context, record)

    agent = RepairAgent([{'action': 'cases', 'signatures': [], 'summary': ''},
                         action, candidate('fast')])
    result = OptimizationLoop(store, key, agent, backend).run(single_evaluation=True)
    assert result['attempts'] == 1
    assert backend.evaluations == ['fast']
    state = OptimizationLoop(store, key, Agent([]), backend).state
    assert 'model_action' not in state
    assert state['history'][0]['tool'] == action


def test_bad_read_replay_is_cleared_only_with_persisted_feedback(tmp_path, monkeypatch):
    store, key = fixture(tmp_path, attempts=1)
    backend = Backend()
    bad = {'action': 'read-artifact', 'sha256': 'abc123', 'offset': 0}
    loop = OptimizationLoop(store, key, Agent([
        {'action': 'cases', 'signatures': [], 'summary': ''}, bad]), backend)
    def interrupted(action, result):
        # Simulate process death before the feedback checkpoint.
        raise KeyboardInterrupt

    monkeypatch.setattr(loop, '_tool_result', interrupted)
    with pytest.raises(KeyboardInterrupt):
        loop.run()
    resumed = OptimizationLoop(store, key, Agent([candidate('fast')]), backend)
    assert resumed._read(resumed.state['model_action']) == bad
    resumed.run(single_evaluation=True)
    state = OptimizationLoop(store, key, Agent([]), backend).state
    assert state['history'][0]['content']['status'] == 'tool-error'
    assert 'model_action' not in state
    assert backend.evaluations == ['fast']


def test_resume_pre_fix_failed_job_replays_bad_read_as_feedback(tmp_path):
    store, key = fixture(tmp_path, attempts=1)
    backend = Backend()
    loop = OptimizationLoop(store, key, Agent([
        {'action': 'cases', 'signatures': [], 'summary': ''}]), backend)
    loop.run(prepare_only=True)
    bad = {'action': 'read-artifact',
           'sha256': '01d481b2410940ecfb8c6765cc4d70c22ef72d73bde972', 'offset': 0}
    retained = loop._artifact('optimize', bad)
    loop.state.update(action_result=retained, model_action=retained)
    loop._save()
    store.transition(key, 'failed')
    store.resume(key, additional_attempts=1, additional_seconds=1)
    OptimizationLoop(store, key, Agent([candidate('fast')]), backend).run(single_evaluation=True)
    state = OptimizationLoop(store, key, Agent([]), backend).state
    assert state['history'][0]['tool'] == bad
    assert state['history'][0]['content']['status'] == 'tool-error'
    assert backend.prepares == 1
    assert backend.evaluations == ['fast']


def test_corrupt_retained_artifact_is_not_model_feedback(tmp_path):
    store, key = fixture(tmp_path)
    artifact = store.put_artifact(key, 'evidence', b'original')
    store.artifact_path(key, artifact['sha256']).write_bytes(b'corrupt')
    loop = OptimizationLoop(store, key, Agent([
        {'action': 'cases', 'signatures': [], 'summary': ''},
        {'action': 'read-artifact', 'sha256': artifact['sha256'], 'offset': 0}]), Backend())
    with pytest.raises(ValueError, match='artifact digest mismatch'):
        loop.run()
    assert store.get(key)['status'] == 'failed'


def test_missing_recorded_artifact_is_not_model_feedback(tmp_path):
    store, key = fixture(tmp_path)
    artifact = store.put_artifact(key, 'evidence', b'original')
    store.artifact_path(key, artifact['sha256']).unlink()
    loop = OptimizationLoop(store, key, Agent([
        {'action': 'cases', 'signatures': [], 'summary': ''},
        {'action': 'read-artifact', 'sha256': artifact['sha256'], 'offset': 0}]), Backend())
    with pytest.raises(FileNotFoundError):
        loop.run()
    assert store.get(key)['status'] == 'failed'


def test_failure_repair_slower_faster_and_time_budget(tmp_path):
    store, key = fixture(tmp_path, attempts=1, seconds=13)
    class Clock:
        now = 0
        def __call__(self): return self.now
    clock = Clock()
    class TimedBackend(Backend):
        def prepare(self, *args, **kwargs):
            clock.now += 1
            return super().prepare(*args, **kwargs)
        def evaluate(self, *args, **kwargs):
            clock.now += 2
            return super().evaluate(*args, **kwargs)
        def profile(self, *args, **kwargs):
            clock.now += 1
            return super().profile(*args, **kwargs)
    class TimedAgent(Agent):
        def complete(self, *args, **kwargs):
            clock.now += 1
            return super().complete(*args, **kwargs)
    backend = TimedBackend()
    agent = TimedAgent([{'action': 'cases', 'signatures': [], 'summary': 'tails'},
                   candidate('bad'), candidate('slow'), candidate('fast')])
    result = OptimizationLoop(store, key, agent, backend, clock=clock).run()
    assert result['status'] == 'budget-exhausted'
    state = [e['payload'] for e in store.events(key) if e['kind'] == 'optimizer-state'][-1]
    assert state['best']['speedup'] == 2
    assert state['best']['faster_than_torch'] is True
    assert backend.profiles == ['slow', 'fast']  # reserved final attempt retains diagnostics
    assert store.get(key)['attempts'] == 3
    assert backend.prepares == 1
    assert any(e['kind'] == 'model-response' for e in store.events(key))


def test_stop_resume_preserves_suite_and_candidates(tmp_path):
    store, key = fixture(tmp_path, attempts=1)
    backend = Backend()
    OptimizationLoop(store, key, Agent([{'action': 'cases', 'signatures': [], 'summary': ''},
                                       candidate('slow')]), backend).run(single_evaluation=True)
    store.stop(key)
    assert OptimizationLoop(store, key, Agent([]), backend).run()['status'] == 'stopped'
    store.resume(key, additional_attempts=2, additional_seconds=60)
    result = OptimizationLoop(store, key, Agent([candidate('slow'),
                         {'action': 'finish', 'summary': 'done'}]), backend).run()
    assert result['status'] == 'completed'
    assert backend.prepares == 1
    assert backend.evaluations == ['slow']


def test_model_time_counts_before_prepare_and_stop_boundary(tmp_path):
    store, key = fixture(tmp_path, seconds=1)
    backend = Backend()
    clock = iter([0, 2])
    result = OptimizationLoop(store, key, Agent([{'action': 'cases', 'signatures': [], 'summary': ''}]),
                              backend, clock=lambda: next(clock)).run()
    assert result['status'] == 'budget-exhausted'
    assert backend.prepares == 0
    store, key = fixture(tmp_path / 'stop')
    store.stop(key)
    assert OptimizationLoop(store, key, Agent([]), backend).run()['status'] == 'stopped'


def test_corrupt_memory_and_changed_model_rejected(tmp_path):
    import pytest
    store, key = fixture(tmp_path, attempts=1)
    backend = Backend()
    OptimizationLoop(store, key, Agent([{'action': 'cases', 'signatures': [], 'summary': ''},
                                       candidate('slow')]), backend).run(single_evaluation=True)
    store.stop(key)
    assert OptimizationLoop(store, key, Agent([]), backend).run()['status'] == 'stopped'
    store.resume(key, additional_attempts=1, additional_seconds=10)
    agent = Agent([])
    agent.identity = {'model': 'changed'}
    with pytest.raises(ValueError, match='identity'):
        OptimizationLoop(store, key, agent, backend).run()
    assert backend.prepares == 1


def test_whole_suite_score_and_missing_timing():
    from optimizer.loop import score
    base = {'suite_sha256': 's', 'cases': [
        {'case': 'a', 'timing': {'torch': {'latency_us': 10}}},
        {'case': 'b', 'timing': {'torch': {'latency_us': 10}}}]}
    result = {'suite_sha256': 's', 'status': 'verified', 'cases': [
        {'case': 'a', 'timing': {'ascendc': {'latency_us': 5}}},
        {'case': 'b', 'timing': {'ascendc': {'latency_us': 20}}}]}
    assert score(base, result) == 1
    result['cases'].pop()
    assert score(base, result) is None


def test_retained_legacy_candidate_requires_checkpoint_migration_without_replay(tmp_path):
    store, key = fixture(tmp_path)
    backend = Backend()
    loop = OptimizationLoop(store, key, Agent([]), backend)
    store.transition(key, 'running')
    source = candidate('fast')
    digest = sha(b'fast')
    loop.state.update(model=Agent.identity, spec=loop._artifact('spec', loop._spec(store.submission(key))),
        baseline=loop._artifact('base', backend.prepare({}, tmp_path)),
        candidate=loop._artifact('source', 'fast'),
        pending={**source, 'sha256': digest, 'attempt': 1})
    loop._save()
    store.record_attempt(key, candidate_sha256=digest, result='evaluating')
    from rsi.domain_execution import CanonicalConfigurationError
    before = dict(loop.state['pending'])
    with pytest.raises(CanonicalConfigurationError, match='checkpoint migration'):
        OptimizationLoop(store, key, Agent([]), backend).run()
    resumed = OptimizationLoop(store, key, Agent([]), backend)
    assert resumed.state['pending'] == before
    assert store.get(key)['attempts'] == 1
    assert backend.evaluations == []


def test_rejected_cases_visible_and_profile_failure_preserves_best(tmp_path):
    store, key = fixture(tmp_path)
    backend = Backend()
    def fail(*args, **kwargs):
        raise RuntimeError('unsupported capture')
    backend.profile = fail
    actions = [{'action': 'cases', 'signatures': [{'args': [{'shape': [-1], 'dtype': 'float32'}]}],
                'summary': 'negative rejected'}, candidate('slow'), {'action': 'finish', 'summary': ''}]
    OptimizationLoop(store, key, Agent(actions), backend).run()
    events = store.events(key)
    proposals = next(e['payload'] for e in events if e['kind'] == 'case-proposals')
    assert len(proposals['rejected']) == 1
    state = [e['payload'] for e in events if e['kind'] == 'optimizer-state'][-1]
    assert state['best']['speedup'] == .5
    assert not state['best']['faster_than_torch']
    assert any(e['kind'] == 'profiling-unavailable' for e in events)


def test_pi_agent_retains_provider_usage_and_host_action():
    from optimizer.agent import OptimizerAgent
    records = []
    class Session:
        def __init__(self, **kwargs): pass
        def next_action(self, context, allowed, record, **kwargs):
            assert context['guide'] == 'guide-text'
            record('pi-provider', {'usage': {'completion_tokens': 7}})
            return 'finish', {'summary': 'done'}
        def close(self): pass
    agent = OptimizerAgent(guide='guide-text', session_factory=Session)
    action = agent.complete('optimize', {}, lambda kind, body: records.append((kind, body)))
    assert action['action'] == 'finish'
    assert records[0][0] == 'pi-provider'
    assert records[0][1]['usage']['completion_tokens'] == 7


def test_dashboard_signature_envelope_flows_through_job_store(tmp_path):
    body = {'files': {}, 'memory_entries': []}
    snapshot = {**body, 'snapshot_id': sha(canonical(body))}
    store = JobStore(tmp_path)
    signature = {'args': [{'shape': [33], 'dtype': 'float32'}]}
    key = store.create('def forward(x): return x', 'forward',
                       {'signatures': [signature]}, snapshot)['job_id']
    class CheckingBackend(Backend):
        def prepare(self, spec, directory):
            assert spec['signatures'] == [signature]
            return super().prepare(spec, directory)
    result = OptimizationLoop(store, key, Agent([
        {'action': 'cases', 'signatures': [], 'summary': 'preserve supplied representative'},
        {'action': 'finish', 'summary': 'baseline only'}]), CheckingBackend()).run()
    assert result['status'] == 'completed'


def test_launch_geometry_is_part_of_candidate_identity(tmp_path):
    store, key = fixture(tmp_path, attempts=3)
    class GeometryBackend(Backend):
        def __init__(self):
            super().__init__()
            self.paths = []
        def evaluate(self, source, directory, *, block_count):
            self.paths.append((str(directory), block_count))
            return super().evaluate(source, directory, block_count=block_count)
    backend = GeometryBackend()
    actions = [{'action': 'cases', 'signatures': [], 'summary': ''},
               candidate('fast'), {**candidate('fast'), 'block_count': 2},
               {**candidate('fast'), 'block_count': 2}, {'action': 'finish', 'summary': ''}]
    result = OptimizationLoop(store, key, Agent(actions), backend).run()
    assert result['attempts'] == 2
    assert [count for _, count in backend.paths] == [1, 2]
    assert backend.paths[0][0] != backend.paths[1][0]


def test_final_reserved_attempt_reconnects_without_added_budget(tmp_path):
    import pytest
    from optimizer.control import ObservationUnavailable
    store, key = fixture(tmp_path, attempts=1)
    class LostBackend(Backend):
        def evaluate(self, *args, **kwargs):
            if not self.evaluations:
                self.evaluations.append('lost')
                store.account_time(key, store.get(key)['max_seconds'])
                raise ObservationUnavailable('same retained handle')
            return super().evaluate(*args, **kwargs)
    backend = LostBackend()
    with pytest.raises(ObservationUnavailable):
        OptimizationLoop(store, key, Agent([{'action': 'cases', 'signatures': [], 'summary': ''},
                                           candidate('fast')]), backend).run()
    assert store.get(key)['status'] == 'budget-exhausted'
    assert store.get(key)['stage'] == 'observation-unavailable'
    OptimizationLoop(store, key, Agent([]), backend).run()
    state = [e['payload'] for e in store.events(key) if e['kind'] == 'optimizer-state'][-1]
    assert state['best']['speedup'] == 2
    assert store.get(key)['attempts'] == store.get(key)['max_attempts'] == 1


def test_replay_persisted_cases_and_candidate_actions_without_model_calls(tmp_path):
    for stage in ('cases', 'optimize'):
        store, key = fixture(tmp_path / stage, attempts=1)
        backend = Backend()
        loop = OptimizationLoop(store, key, Agent([]), backend)
        store.transition(key, 'running', stage=stage)
        action = {'action': 'cases', 'signatures': [], 'summary': ''} if stage == 'cases' else candidate('fast')
        if stage == 'optimize':
            loop.state['spec'] = loop._artifact('spec', loop._spec(store.submission(key)))
            loop.state['baseline'] = loop._artifact('baseline', backend.prepare({}, tmp_path))
        loop.state['action_result'] = loop._artifact(stage, action)
        loop._save()  # crash after _action returned evidence, before caller checkpoint
        actions = [candidate('fast')] if stage == 'cases' else []
        OptimizationLoop(store, key, Agent(actions), backend).run(single_evaluation=True)
        assert store.get(key)['attempts'] == 1
        assert backend.evaluations == ['fast']


def test_final_reserved_profile_reconnect_does_not_repeat_evaluation(tmp_path):
    import pytest
    from optimizer.control import ObservationUnavailable
    store, key = fixture(tmp_path, attempts=1)
    class LostProfile(Backend):
        def profile(self, *args, **kwargs):
            if not self.profiles:
                self.profiles.append('lost')
                store.account_time(key, store.get(key)['max_seconds'])
                raise ObservationUnavailable('retained profile handle')
            return {'status': 'available'}
    backend = LostProfile()
    with pytest.raises(ObservationUnavailable):
        OptimizationLoop(store, key, Agent([
            {'action': 'cases', 'signatures': [], 'summary': ''}, candidate('fast')]), backend).run()
    OptimizationLoop(store, key, Agent([]), backend).run()
    state = [e['payload'] for e in store.events(key) if e['kind'] == 'optimizer-state'][-1]
    assert len(state['history']) == len(backend.evaluations) == 1
    assert state['best']['speedup'] == 2
    assert 'profiling' not in state['best']
    assert backend.profiles == ['lost']
    assert any(event['kind'] == 'profiling-unavailable' for event in store.events(key))
    assert store.get(key)['attempts'] == store.get(key)['max_attempts'] == 1


def test_finish_respects_stop_or_time_boundary_after_model_returns(tmp_path):
    for boundary in ('stop', 'time'):
        store, key = fixture(tmp_path / boundary)
        class BoundaryAgent(Agent):
            def complete(self, stage, context, record):
                action = super().complete(stage, context, record)
                if stage == 'optimize':
                    if boundary == 'stop':
                        store.stop(key)
                    else:
                        store.account_time(key, 7200)
                return action
        result = OptimizationLoop(store, key, BoundaryAgent([
            {'action': 'cases', 'signatures': [], 'summary': ''},
            {'action': 'finish', 'summary': 'done'}]), Backend()).run()
        assert result['status'] == ('stopped' if boundary == 'stop' else 'budget-exhausted')


def test_configured_case_cap_is_shared_with_model_and_filters_excess(tmp_path):
    store, key = fixture(tmp_path, attempts=1)
    extra = [{'args': [{'shape': [size], 'dtype': 'float32'}]} for size in (7, 9, 11)]
    class CappedAgent(Agent):
        def complete(self, stage, context, record):
            if stage == 'cases':
                assert context['max_cases'] == 2
                assert context['remaining_signatures'] == 1
            return super().complete(stage, context, record)
    class CappedBackend(Backend):
        def prepare(self, spec, directory):
            assert len(spec['signatures']) == 2
            assert spec['signatures'][1] == extra[0]
            return super().prepare(spec, directory)
    result = OptimizationLoop(store, key, CappedAgent([
        {'action': 'cases', 'signatures': extra, 'summary': 'corners'},
        {'action': 'finish', 'summary': ''}]), CappedBackend(), max_cases=2).run()
    assert result['status'] == 'completed'
    proposals = next(e['payload'] for e in store.events(key) if e['kind'] == 'case-proposals')
    assert len(proposals['accepted']) == 1 and len(proposals['rejected']) == 2
    assert all('2-case budget' in row['error'] for row in proposals['rejected'])


def test_case_cap_validation_preserves_all_representatives(tmp_path):
    import pytest
    store, key = fixture(tmp_path)
    for maximum in (0, 25, True):
        with pytest.raises(ValueError, match='max_cases'):
            OptimizationLoop(store, key, Agent([]), Backend(), max_cases=maximum)
    body = {'files': {}, 'memory_entries': []}
    signatures = [{'args': [{'shape': [n], 'dtype': 'float32'}]} for n in (3, 5)]
    key = store.create('def f(x): return x', 'f', {'signatures': signatures},
                       {**body, 'snapshot_id': sha(canonical(body))})['job_id']
    with pytest.raises(ValueError, match='representatives exceed'):
        OptimizationLoop(store, key, Agent([]), Backend(), max_cases=1).run()


def test_model_actions_have_no_count_ceiling_and_receive_remaining_time(tmp_path):
    store, key = fixture(tmp_path, attempts=1)
    class TimeAwareAgent(Agent):
        identity = {'model': 'fake-fixed', 'coding_engine': 'pi'}
        def complete(self, stage, context, record):
            assert 0 < context['remaining_seconds'] <= 7200
            assert 'remaining_provider_requests' not in context
            return super().complete(stage, context, record)
    actions = [{'action': 'cases', 'signatures': [], 'summary': ''}] + [
        {'action': 'read-memory', 'index': 99}] * 2 + [
        candidate('bad'), candidate('slow'), candidate('fast'),
        {'action': 'finish', 'summary': 'done'}]
    backend = Backend()
    loop = OptimizationLoop(store, key, TimeAwareAgent(actions), backend)
    loop.state['turn'] = 199  # Retained telemetry must not become a stopping limit.
    loop._save()
    result = loop.run()
    assert result['status'] == 'completed' and result['attempts'] == 3
    assert backend.evaluations == ['bad', 'slow', 'fast']
    state = next(event['payload'] for event in reversed(store.events(key))
                 if event['kind'] == 'optimizer-state')
    assert state['turn'] > 200 and state['best']['speedup'] == 2


@pytest.mark.parametrize('wrapped', [False, True])
def test_knowledge_compacts_before_checkpoints_and_sources_are_readable_after_resume(tmp_path, wrapped):
    from test_knowledge_presentation import source_rows
    store, key = fixture(tmp_path)
    store.transition(key, 'running')
    rows = source_rows(8)
    callback_result = {'results': rows} if wrapped else rows
    loop = OptimizationLoop(store, key, Agent([]), Backend(), knowledge=lambda _: callback_result)
    snapshot = {'files': {}, 'memory_entries': []}
    loop._tool({'action': 'query-knowledge', 'query': 'racecheck synchronization hazards'}, snapshot)
    result = loop.state['history'][-1]['content']
    assert len(canonical(result)) < 4096
    assert result['result_count'] == len(rows)
    index = json.loads(store.read_artifact(key, result['results_index_sha256']))
    assert len(index['results']) == len(rows)
    # The elapsed-action checkpoint must contain the compact result too.
    for event in store.events(key):
        if event['kind'] == 'optimizer-state' and 'action_result' in event['payload']:
            checkpoint = loop._read(event['payload']['action_result'])
            assert len(canonical(checkpoint)) < 4096
    resumed = OptimizationLoop(store, key, Agent([]), Backend())
    resumed._tool({'action': 'query-knowledge', 'query': 'unavailable'}, snapshot)
    assert resumed.state['history'][-1]['content'] == {'status': 'unavailable'}
    for entry, original in zip(index['results'], rows):
        assert store.read_artifact(key, entry['artifact_sha256']) == original['text'].encode()
    digest = index['results'][-1]['artifact_sha256']
    resumed._tool({'action': 'read-artifact', 'sha256': digest, 'offset': 0}, snapshot)
    page = resumed.state['history'][-1]['content']
    assert page['text'] == rows[-1]['text'][:1500]
    assert page['next_offset'] == 1500

    class InspectAgent(Agent):
        def complete(self, stage, context, record):
            earlier = context['history'][0]
            assert json.loads(earlier['content_preview']) == result
            assert earlier['content_truncated'] is False
            from coding.pi.context import ContextRecords
            visible = ContextRecords().snapshot(context)
            assert isinstance(visible['history'], dict)  # The entire history was paged.
            assert visible['latest_knowledge'] == result
            return candidate('fast')

    class UnavailableAgent(Agent):
        def complete(self, stage, context, record):
            assert 'latest_knowledge' not in context
            return candidate('fast')

    resumed.agent = UnavailableAgent([])
    resumed.context = {}
    resumed.single_evaluation, resumed.prepare_only = True, False
    resumed._model('optimize', {'history': resumed.state['history']})
    resumed.knowledge = lambda _: callback_result
    resumed._tool({'action': 'query-knowledge', 'query': 'racecheck synchronization hazards'}, snapshot)
    resumed.state['history'].extend([
        {'tool': {'action': 'compile'}, 'content': {'report': 'x' * 6000}}
        for _ in range(4)
    ])
    resumed.agent = InspectAgent([])
    resumed.context = {}
    resumed.single_evaluation, resumed.prepare_only = True, False
    resumed._model('optimize', {'history': resumed.state['history']})
