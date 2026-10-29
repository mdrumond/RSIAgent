import multiprocessing
import time

import pytest

from optimizer.provider import BoundedAgent
from optimizer.transport import WorkerStopped


class Agent:
    identity = {'model': 'test'}

    def complete(self, stage, context, record):
        record('request', {'stage': stage})
        if context.get('hang'):
            time.sleep(10)
        record('response', {'summary': 'bounded'})
        return {'action': 'finish', 'summary': 'done'}


def test_child_streams_complete_provider_trace_and_result():
    records = []
    agent = BoundedAgent(Agent(), deadline=time.monotonic()+5, stopped=lambda: False)
    result = agent.complete('optimize', {}, lambda *row: records.append(row))
    assert result['action'] == 'finish'
    assert [row[0] for row in records] == ['request', 'response']


@pytest.mark.parametrize('stop', [True, False])
def test_hanging_provider_is_terminated_on_stop_or_remaining_budget(stop):
    before = set(multiprocessing.active_children())
    records = []
    started = time.monotonic()
    now, stop_requested = [started], [False]
    deadline = started+60
    # Exercise cancellation during a request, regardless of supervisor startup
    # latency. The real clock is only a watchdog for a broken test handshake.
    def record(kind, data):
        records.append((kind, data))
        if kind == 'request':
            if stop:
                stop_requested[0] = True
            else:
                now[0] = deadline+1
    agent = BoundedAgent(Agent(), deadline=deadline, clock=lambda: now[0],
                         stopped=lambda: stop_requested[0] or time.monotonic()-started > 5)
    with pytest.raises(WorkerStopped):
        agent.complete('optimize', {'hang': True}, record)
    assert time.monotonic() - started < 10
    assert set(multiprocessing.active_children()) == before
    assert [row[0] for row in records] == ['request']
    assert stop_requested[0] == stop


def test_child_keeps_agent_state_between_host_actions():
    class Stateful(Agent):
        calls = 0
        def complete(self, stage, context, record):
            self.calls += 1
            return {'calls': self.calls}
    agent = BoundedAgent(Stateful(), deadline=time.monotonic()+5, stopped=lambda: False)
    try:
        assert agent.complete('cases', {}, lambda *_: None) == {'calls':1}
        pid = agent.process.pid
        assert agent.complete('optimize', {}, lambda *_: None) == {'calls':2}
        assert agent.process.pid == pid
    finally:
        agent.close()


def test_stop_kills_provider_descendants(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    class Spawning(Agent):
        def complete(self, stage, context, record):
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            record('child', {'pid': child.pid})
            time.sleep(30)
    started, descendants = time.monotonic(), []
    agent = BoundedAgent(Spawning(), deadline=started+60,
                         stopped=lambda: bool(descendants) or time.monotonic()-started > 5)
    with pytest.raises(WorkerStopped):
        agent.complete('optimize', {}, lambda kind,data: descendants.append(data['pid']))
    assert time.monotonic()-started < 10
    assert descendants
    for pid in descendants:
        stat = Path(f'/proc/{pid}/stat')
        try:
            state = stat.read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            continue  # The killed descendant was reaped between observations.
        # Linux may briefly expose a dead task before it becomes a zombie or is reaped.
        assert state in {'X', 'Z'}
    assert agent.process is None
