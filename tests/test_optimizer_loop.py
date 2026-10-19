import json

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


def test_failure_repair_slower_faster_and_budget(tmp_path):
    store, key = fixture(tmp_path)
    backend = Backend()
    agent = Agent([{'action': 'cases', 'signatures': [], 'summary': 'tails'},
                   candidate('bad'), candidate('slow'), candidate('fast')])
    result = OptimizationLoop(store, key, agent, backend).run()
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
                                       candidate('slow')]), backend).run()
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
                                       candidate('slow')]), backend).run()
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


def test_pending_candidate_recovery_reuses_directory_and_charge(tmp_path):
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
    result = OptimizationLoop(store, key, Agent([{'action': 'finish', 'summary': ''}]), backend).run()
    assert result['status'] == 'completed'
    assert result['attempts'] == 1
    assert backend.evaluations == ['fast']


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


def test_direct_agent_retains_exact_provider_payload_and_usage():
    from optimizer.agent import OptimizerAgent
    from benchmarks.a3_model_profiles import A3Completion
    records = []
    def complete(system, user, *, model):
        assert 'guide-text' in user
        return A3Completion(text='{"action":"finish","summary":"done"}', completion_tokens=7,
                            provenance={'test': True})
    agent = OptimizerAgent(guide='guide-text', complete=complete)
    action = agent.complete('optimize', {}, lambda kind, body: records.append((kind, body)))
    assert action['action'] == 'finish'
    assert records[0][0] == 'request'
    assert records[1][1]['completion_tokens'] == 7


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
        OptimizationLoop(store, key, Agent(actions), backend).run()
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
                raise ObservationUnavailable('retained profile handle')
            return {'status': 'available'}
    backend = LostProfile()
    with pytest.raises(ObservationUnavailable):
        OptimizationLoop(store, key, Agent([
            {'action': 'cases', 'signatures': [], 'summary': ''}, candidate('fast')]), backend).run()
    OptimizationLoop(store, key, Agent([]), backend).run()
    state = [e['payload'] for e in store.events(key) if e['kind'] == 'optimizer-state'][-1]
    assert len(state['history']) == len(backend.evaluations) == 1
    assert 'profiling' in state['best']
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
