import json
from types import SimpleNamespace

import pytest

from benchmarks.ascend.actor_time import ActorWorkBudget, WorkDeadline, preparation
from optimizer.transport import CheckedTransport, ObservationUnavailable
from tests.test_optimizer_worker import config


@pytest.mark.parametrize('kind', ['execute', 'upload'])
def test_queued_preparation_has_persisted_expiry_even_with_exhausted_actor(tmp_path, monkeypatch, kind):
    now, calls = [0.], []
    monkeypatch.setattr('optimizer.transport.time.monotonic', lambda: now[0])
    budget = ActorWorkBudget(tmp_path/'actor.json', 1, clock=lambda: now[0])
    settings = config(tmp_path) | {'transport_timeout_seconds': 5}
    def run(argv, **kwargs):
        calls.append(argv)
        assert len(calls) <= 2, 'queued operation outlived its preparation allowance'
        now[0] += 3
        output = (json.dumps(dict(state='queued', handle='remote:bz-a3-1:transfer:one'))
                  if kind == 'upload' else 'CATLASS_VALIDATION_HANDLE=bz-a3-1:one\n'
                  'BZ_A3_REMOTE_HANDLE=remote:bz-a3-1:job:one\nCATLASS_VALIDATION_STATE=queued\n')
        return SimpleNamespace(returncode=75, stdout=output, stderr='')
    source = tmp_path/'package'; source.write_bytes(b'package')
    def invoke(transport):
        if kind == 'upload': transport.transfer('upload', source, '/work/package')
        else: transport.execute(tmp_path/'operation', ['true'])
    with budget.actor('actor', 1) as worker:
        with worker.scope(work=True): now[0] += 1
        assert worker.remaining() == 0
        with preparation('reference'):
            transport = CheckedTransport(settings, run=run, deadline=WorkDeadline(lambda: 5))
            with pytest.raises(ObservationUnavailable, match='elapsed allocation expired'):
                invoke(transport)
            assert len(calls) == 2
            marker = (source.with_name('package.upload.json') if kind == 'upload'
                      else tmp_path/'operation/transport.json')
            retained = marker.read_bytes()
            # A new controller instance must not grant another observation window.
            resumed = CheckedTransport(settings, run=run, deadline=WorkDeadline(lambda: 5))
            with pytest.raises(ObservationUnavailable, match='elapsed allocation expired'):
                invoke(resumed)
            assert len(calls) == 2 and marker.read_bytes() == retained
    assert budget.elapsed == 1 and budget.preparation == 6


def test_independent_preparation_operations_each_get_allowance(tmp_path, monkeypatch):
    now, calls = [0.], []
    monkeypatch.setattr('optimizer.transport.time.monotonic', lambda: now[0])
    budget = ActorWorkBudget(tmp_path/'actor.json', 1, clock=lambda: now[0])
    def run(argv, **kwargs):
        calls.append(argv); now[0] += 4
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    with budget.actor('actor', 1):
        with preparation('reference'):
            transport = CheckedTransport(config(tmp_path) | {'transport_timeout_seconds': 5}, run=run)
            for name in ('first', 'second'):
                transport.execute(tmp_path/name, ['true'])
            # Completed evidence can be read after its operation expiry.
            transport.execute(tmp_path/'first', ['true'])
    assert len(calls) == 2 and budget.elapsed == 0 and budget.preparation == 8
    assert json.loads((tmp_path/'first/preparation-deadline.json').read_text())['deadline'] == 5
    assert json.loads((tmp_path/'second/preparation-deadline.json').read_text())['deadline'] == 9


def test_default_preparation_allowance_preserves_six_hundred_seconds(tmp_path, monkeypatch):
    now, calls = [0.], []
    monkeypatch.setattr('optimizer.transport.time.monotonic', lambda: now[0])
    budget = ActorWorkBudget(tmp_path/'actor.json', 1, clock=lambda: now[0])
    def run(argv, **kwargs):
        calls.append(argv); now[0] += 200
        assert len(calls) <= 3, 'default preparation allowance was refreshed'
        return SimpleNamespace(returncode=75, stdout='CATLASS_VALIDATION_HANDLE=bz-a3-1:one\n'
                               'CATLASS_VALIDATION_STATE=queued\n', stderr='')
    with budget.actor('actor', 1):
        with preparation('reference'):
            with pytest.raises(ObservationUnavailable, match='elapsed allocation expired'):
                CheckedTransport(config(tmp_path), run=run).execute(tmp_path/'reference', ['true'])
    assert len(calls) == 3 and now[0] == 600 and budget.elapsed == 0


def test_legacy_handle_gets_bounded_observation_without_redispatch(tmp_path, monkeypatch):
    now, calls = [0.], []
    monkeypatch.setattr('optimizer.transport.time.monotonic', lambda: now[0])
    directory = tmp_path/'operation'; directory.mkdir()
    state = dict(state='queued', handle='bz-a3-1:old', native_handle='remote:bz-a3-1:job:old')
    (directory/'transport.json').write_text(json.dumps(state))
    budget = ActorWorkBudget(tmp_path/'actor.json', 1, clock=lambda: now[0])
    def run(argv, **kwargs):
        calls.append(argv); now[0] += 5
        assert len(calls) == 1, 'legacy observation allowance was refreshed'
        assert '--resume' in argv and argv[argv.index('--resume')+1] == 'old'
        return SimpleNamespace(returncode=75, stdout='CATLASS_VALIDATION_STATE=queued\n', stderr='')
    with budget.actor('actor', 1):
        with preparation('reference'):
            with pytest.raises(ObservationUnavailable, match='elapsed allocation expired'):
                CheckedTransport(config(tmp_path) | {'transport_timeout_seconds': 5}, run=run).execute(directory, ['true'])
    assert len(calls) == 1 and json.loads((directory/'transport.json').read_text()) == state
