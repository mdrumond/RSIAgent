"""Cancellation and numeric persistence at Actor budget adapter boundaries."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import json
import sys

import pytest

from benchmarks.ascend.actor_control import ActorControl
from benchmarks.ascend.actor_time import ActorWorkBudget
from benchmarks.ascend.execution import LocalProcessRunner, process_deadline
from benchmarks.ascend.production import NativeProjectPool


@pytest.mark.parametrize('project_stop', [False, True])
def test_waiting_worker_is_cancellable_without_poisoning_available_worker(tmp_path, project_stop):
    waiting, cancelled = Event(), Event()

    def stopped():
        waiting.set()
        return cancelled.is_set()

    pool = NativeProjectPool([lambda request: request.branch_root], stopped=stopped)
    held = pool.actors.get_nowait()
    request = SimpleNamespace(branch_root=tmp_path / 'queued', remaining_seconds=10)
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(pool, request)
        assert waiting.wait(2)
        if project_stop:
            ActorControl(request.branch_root / 'actor').stop()
        else:
            cancelled.set()
        with pytest.raises(TimeoutError, match='operator') as raised:
            future.result(timeout=2)
    assert raised.value.interruption_reason == 'operator-stop'
    assert pool.live == 1 and pool.quarantined == []
    pool.actors.put(held)
    cancelled.clear()
    request.branch_root = tmp_path / 'next'
    assert pool(request) == request.branch_root


def test_local_process_freezes_live_actor_deadline_at_dispatch_boundary(tmp_path, monkeypatch):
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES', raising=False)
    config = dict(execution='local', workspace=str(tmp_path / 'work'),
        physical_device=0, arch='discovered-architecture', max_seconds=600,
        lock_directory=str(tmp_path / 'locks'))
    budget = ActorWorkBudget(tmp_path / 'elapsed.json', 100)
    with budget.bind(), budget.actor('writer', 10) as actor, actor.scope(work=True):
        with process_deadline(actor.deadline):
            result = LocalProcessRunner(config).run(
                [sys.executable, '-c', "print('completed')"], tmp_path / 'process')
    assert result.returncode == 0 and result.stdout == 'completed\n'
    payload = json.loads((tmp_path / 'process/process.json').read_bytes())
    assert isinstance(payload['deadline'], (float, int))
    assert json.loads((tmp_path / 'process/transport.json').read_bytes())['state'] == 'completed'
