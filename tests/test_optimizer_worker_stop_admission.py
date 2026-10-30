"""Stop recovery must finish before admission to the optimizer actor."""
import builtins
import json
from types import SimpleNamespace

import pytest

from optimizer.jobs import JobStore, canonical, sha
from optimizer.transport import CheckedTransport, ObservationUnavailable
from optimizer.worker import run_worker
from test_optimizer_worker import config


@pytest.fixture
def stopped_worker(tmp_path, monkeypatch):
    import optimizer.worker as worker
    from benchmarks.ascend.feedback import FeedbackStore
    import benchmarks.ascend.execution as execution

    store = JobStore(tmp_path / 'jobs')
    body = {'files': {}, 'memory_entries': []}
    key = store.create('def f(x): return x', 'f', {},
                       {**body, 'snapshot_id': sha(canonical(body))})['job_id']
    store.transition(key, 'running')
    store.stop(key)
    guide = tmp_path / 'guide'
    guide.write_text('guide')
    settings = {**config(tmp_path), 'guide': str(guide),
                'lock_directory': str(tmp_path / 'locks')}
    operation = store.root / key / 'execution' / 'candidate'
    operation.mkdir(parents=True)
    marker = operation / 'transport.json'
    marker.write_text(json.dumps({'state': 'running',
                                  'native_handle': 'remote:bz-a3-1:job:original'}))

    def forbidden(*args, **kwargs):
        pytest.fail('stopped worker entered optimizer/runtime setup')

    # Guard imports too: loading the actor/provider path can have side effects.
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if ((level == 1 and globals and globals.get('__package__') == 'optimizer'
             and name in {'agent', 'loop', 'provider'})
                or name in {'optimizer.agent', 'optimizer.loop', 'optimizer.provider', 'coding.pi'}):
            forbidden()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(worker, 'knowledge_query', forbidden)
    monkeypatch.setattr(execution, 'preflight_local', forbidden)
    feedback = []
    monkeypatch.setattr(FeedbackStore, 'import_optimizer',
                        lambda self, root: feedback.append(root))
    return SimpleNamespace(store=store, key=key, settings=settings,
                           operation=operation, marker=marker, feedback=feedback)


@pytest.mark.parametrize('injected_agent', [False, True])
def test_stop_retries_real_handle_cancellation_before_admission(stopped_worker, monkeypatch,
                                                               injected_agent):
    import optimizer.worker as worker
    case = stopped_worker
    calls = []

    def cancel(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0, stdout='{}', stderr='')

    monkeypatch.setattr(worker, 'CheckedTransport', lambda config, **controls:
                        CheckedTransport(config, run=cancel, **controls))
    # No model property: this agent must never be inspected on the stop path.
    agent = object() if injected_agent else None
    with pytest.raises(ObservationUnavailable):
        run_worker(case.store.root, case.key, case.settings, agent=agent)
    pending = case.store.get(case.key)
    assert pending['status'] == 'running'
    assert pending['stage'] == 'observation-unavailable'
    assert pending['stop_requested']
    assert json.loads(case.marker.read_text())['state'] == 'running'
    result = run_worker(case.store.root, case.key, case.settings, agent=agent)
    assert result['status'] == 'stopped'
    assert result['attempts'] == 0
    assert [argv[-2:] for argv in calls] == [
        ['cancel', 'remote:bz-a3-1:job:original'],
        ['cancel', 'remote:bz-a3-1:job:original']]
    assert json.loads(case.marker.read_text())['state'] == 'cancelled'
    assert case.feedback == [case.store.root, case.store.root]


def test_unknown_dispatch_remains_pending_without_cancel_or_dispatch(stopped_worker, monkeypatch):
    import optimizer.worker as worker
    case = stopped_worker
    case.marker.write_text('{"state":"dispatching"}')
    calls = []

    def invoke(argv, **kwargs):
        calls.append(argv)
        pytest.fail('unknown dispatch cannot be cancelled or redispatched')

    monkeypatch.setattr(worker, 'CheckedTransport', lambda config, **controls:
                        CheckedTransport(config, run=invoke, **controls))
    for _ in range(2):
        with pytest.raises(ObservationUnavailable):
            run_worker(case.store.root, case.key, case.settings)
        assert case.store.get(case.key)['status'] == 'running'
        assert case.store.get(case.key)['stop_requested']
        assert json.loads(case.marker.read_text()) == {'state': 'dispatching'}
    assert calls == []
    assert len(case.feedback) == 2


def test_injected_stop_pins_route_before_constructing_transport(stopped_worker):
    case = stopped_worker
    case.settings['transport_identity'] = 'stop-route-v1'
    calls = []
    factories = []

    def factory(settings, **controls):
        factories.append(settings['transport_identity'])
        return SimpleNamespace(cancel_pending=lambda path: calls.append(path))

    result = run_worker(case.store.root, case.key, case.settings,
                        agent=object(), transport_factory=factory)
    assert result['status'] == 'stopped'
    assert calls == [case.operation]
    case.settings['transport_identity'] = 'stop-route-v2'
    with pytest.raises(ValueError, match='changed on resume'):
        run_worker(case.store.root, case.key, case.settings, transport_factory=factory)
    assert factories == ['stop-route-v1']
    assert calls == [case.operation]


@pytest.mark.parametrize('field,value', [('runtime', 'different-runtime'),
                                         ('physical_device', 11),
                                         ('remote_workspace', '/different-workspace')])
def test_stop_rejects_runtime_identity_change_before_cancellation(stopped_worker, field, value):
    case = stopped_worker
    calls = []
    case.settings['transport_identity'] = 'stop-route-v1'

    def factory(settings, **controls):
        return SimpleNamespace(cancel_pending=lambda path: calls.append(path))

    run_worker(case.store.root, case.key, case.settings, transport_factory=factory)
    case.settings[field] = value
    with pytest.raises(ValueError, match='changed on resume'):
        run_worker(case.store.root, case.key, case.settings, transport_factory=factory)
    assert calls == [case.operation]


def test_local_stop_preserves_unknown_dispatch_without_runtime_probe(stopped_worker):
    case = stopped_worker
    case.settings.update(execution='local', workspace=str(case.store.root / 'workspace'))
    case.marker.write_text('{"state":"dispatching"}')
    with pytest.raises(ObservationUnavailable):
        run_worker(case.store.root, case.key, case.settings)
    assert case.store.get(case.key)['status'] == 'running'
    assert json.loads(case.marker.read_text()) == {'state': 'dispatching'}
    assert case.feedback == [case.store.root]
