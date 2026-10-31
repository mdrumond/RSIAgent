"""Finite production capacity, explicit small stress, and native history ownership."""
import json
import time

import pytest

from coding.pi.controller import ControllerSession
from coding.pi.gateway import RequestContextLimit
from coding.pi.schema import STRING, obj
from coding.pi.session import PiCodingSession
from rsi.actor import EvidenceTool, StageResult
from rsi.contracts import Stage
from rsi.skills import builtin_bundle
from test_rsi_pi_controller import native_actor, response


@pytest.mark.parametrize('limits', [{}, dict(context_window=16384, request_bytes=49152)])
def test_adoption_preserves_capacity_deadline_and_dispatch_guard(monkeypatch, limits):
    monkeypatch.setattr('coding.pi.session.runtime_preflight', lambda: 'unused-no-process')
    deadline = time.monotonic() + 120
    session = PiCodingSession(model='deepseek-flash', schemas={}, deadline=deadline,
                              environ={'DEEPSEEK_API_KEY': 'fake'}, **limits)
    gateway = session.gateway
    controller = ControllerSession.from_existing_session(session)
    try:
        assert controller.gateway is gateway
        assert controller.deadline == controller.gateway.deadline == deadline
        assert controller.context_window == limits.get('context_window', 65536)
        payload = dict(model='deepseek-flash', messages=[dict(role='user', content='x' * 60000)])
        if limits:
            with pytest.raises(RequestContextLimit):
                gateway.prepare(payload)
        else:
            gateway.prepare(payload)
        payload['messages'][0]['content'] = 'x' * 200000
        with pytest.raises(RequestContextLimit):
            gateway.prepare(payload)
        assert gateway.calls == 0 and gateway.records == []
        assert controller.rpc is None and session.closed
    finally:
        controller.close()


@pytest.mark.parametrize('limits', [dict(context_window=True), dict(context_window=1048576),
    dict(context_window=8192), dict(request_bytes=0), dict(request_bytes=196609),
    dict(request_bytes=float('inf'))])
def test_invalid_capacity_cannot_start_provider(monkeypatch, limits):
    monkeypatch.setattr('coding.pi.session.runtime_preflight', lambda: 'unused-no-process')
    with pytest.raises(ValueError):
        ControllerSession(model='deepseek-flash', schemas={},
                          environ={'DEEPSEEK_API_KEY': 'fake'}, **limits)


def test_native_compaction_continues_with_atomic_large_tool_batch(monkeypatch, tmp_path):
    events, requests, sizes, summaries, effects = [], [], [], [], []
    names = builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)

    def provider(*_, **kwargs):
        request = json.loads(kwargs['data'])
        requests.append(request)
        sizes.append(len(kwargs['data']))
        assert sizes[-1] <= 196608
        if not request.get('tools'):
            summaries.append(request)
            return response([], len(requests), tokens=100,
                            summary='Task pending; evidence receipts remain authoritative. Continue guarded tools.')
        read = [row['skill'] for kind, row in events if kind == 'skill-usage'
                and row['kind'] == 'read' and row['stage'] == Stage.TASK_CONSTRUCTION.value]
        missing = [name for name in names if name not in read]
        if missing:
            actions = [('read_skill', {'name': name}) for name in missing]
        elif not effects:
            # Pi must keep the completed atomic exchange intact. It cannot
            # repair the former byte cap merely by summarizing older history.
            actions = [('evidence', {'sample': str(i)}) for i in range(20)]
        else:
            actions = [('perform', {'proposal': 'complete after retained native compaction'})]
        return response(actions, len(requests), tokens=60000 if not missing and not effects else 100)

    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)

    def evidence(stage, args, operation, seconds):
        effects.append(args['sample'])
        return dict(reference='retained-' + args['sample'], evidence='raw-tool-evidence ' * 140)

    actor, flow = native_actor(tmp_path, events, lambda *_: StageResult(),
        {'evidence': EvidenceTool(obj({'sample': STRING}), evidence)},
        context_window=65536, request_bytes=196608)
    try:
        actor.step({})
        actor.step({'goal': 'continue after a real mixed skill and atomic evidence batch'})
    finally:
        actor.close()
    assert max(sizes) > 49152 and summaries
    assert effects == [str(i) for i in range(20)] and flow.stage == Stage.DESIGN
    assert any(kind == 'pi-event' and row['event']['type'] == 'compaction_end'
               and row['event'].get('result') for kind, row in events)
    intents = [row['reservation_id'] for kind, row in events if kind == 'pi-provider-request']
    settled = [row['reservation_id'] for kind, row in events if kind == 'pi-provider']
    assert set(intents) == set(settled) and len(settled) == len(set(settled)) == len(requests)
