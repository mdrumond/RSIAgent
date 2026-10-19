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
    agent = BoundedAgent(Agent(), deadline=started+.3,
                         stopped=lambda: stop and time.monotonic() > started+.1)
    with pytest.raises(WorkerStopped):
        agent.complete('optimize', {'hang': True}, lambda *row: records.append(row))
    assert time.monotonic() - started < 2
    assert set(multiprocessing.active_children()) == before
    assert records[0][0] == 'request'
