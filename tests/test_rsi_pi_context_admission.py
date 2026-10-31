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


@pytest.mark.parametrize('limits', [{}, dict(context_window=16384, request_bytes=49152),
                                  dict(context_window=65536, request_bytes=196608)])
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
        if gateway.hard_bytes < 60000:
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


def test_direct_controller_uses_production_capacity_without_widening_generic_defaults(monkeypatch):
    monkeypatch.setattr('coding.pi.session.runtime_preflight', lambda: 'unused-no-process')
    generic = PiCodingSession(model='deepseek-flash', schemas={}, environ={'DEEPSEEK_API_KEY': 'fake'})
    controller = ControllerSession(model='deepseek-flash', schemas={}, environ={'DEEPSEEK_API_KEY': 'fake'})
    payload = dict(model='deepseek-flash', messages=[dict(role='user', content='x' * 60000)])
    try:
        with pytest.raises(RequestContextLimit):
            generic.gateway.prepare(payload)
        controller.gateway.prepare(payload)
        assert controller.context_window == 65536
        assert generic.gateway.calls == controller.gateway.calls == 0
    finally:
        generic.close()
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


@pytest.mark.parametrize('oversized_guidance', [False, True])
def test_native_wire_keeps_operator_context_visible_when_feedback_is_paged(monkeypatch, tmp_path, oversized_guidance):
    events, requests = [], []
    guidance = dict(actor='Use the already frozen cases and write only candidate.py',
                    supervisor='Preserve target evidence', curator='Retain rejected memory')
    if oversized_guidance:
        guidance['supervisor_notes'] = {f'fact-{i}': 'operator observation ' * 10 for i in range(60)}
    execution = dict(physical_device=3, soc='Ascend910B4-1', runtime='py311-torch',
                     admission=dict(admitted=True, degraded=False, measurements=dict(memory_percent=5)))
    identity = dict(project_id='conversion-001', goal='Translate the frozen AscendC reference to Triton')
    task = dict(project={**identity, **{f'case-{i}': 'case metadata ' * 20 for i in range(50)}},
                user_guidance=guidance, execution_context=execution, configuration_ref='a' * 64,
                feedback={f'compiler-{i}': 'retained diagnostic ' * 20 for i in range(70)})
    pinned = dict(project=identity, user_guidance=guidance, execution_context=execution,
                  configuration_ref='a' * 64)
    names = builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    reference = None
    pages = {}

    def provider(*_, **kwargs):
        nonlocal reference
        request = json.loads(kwargs['data'])
        requests.append(request)
        content = next(row['content'] for row in request['messages'] if row['role'] == 'user')
        prompt = json.loads(content if isinstance(content, str) else
                            ''.join(part['text'] for part in content if part['type'] == 'text'))
        delivered = prompt['task']
        assert delivered['user_guidance']['actor'] == guidance['actor']
        if not oversized_guidance:
            assert delivered['user_guidance'] == guidance
        else:
            assert 'full_observation' in delivered['user_guidance']
        assert delivered['execution_context'] == execution
        assert {key: delivered['project'][key] for key in identity} == identity
        assert delivered['configuration_ref'] == 'a' * 64
        assert 'full_observation' in delivered and 'full_observation' in delivered['project']
        reference = prompt['task_context']['record']
        results = [json.loads(row['content']) for row in request['messages'] if row['role'] == 'tool']
        if len(requests) == 1:
            return response([('read_skill', {'name': name}) for name in names], len(requests))
        # Completed tools retain a readable identity handle, rather than
        # duplicating full operator guidance and execution facts each time.
        for result in results:
            assert result['state']['task_context']['record'] == reference
            assert 'user_guidance' not in result['state']
        if len(requests) == 2:
            return response([('read_context', dict(record=reference, limit=3072))], len(requests))
        page = [result['observation'] for result in results if result['observation'].get('record') == reference][-1]
        pages[page['offset']] = page['text']
        if page['next_offset'] is not None:
            return response([('read_context', dict(record=reference, offset=page['next_offset'], limit=3072))], len(requests))
        assert json.loads(''.join(pages[offset] for offset in sorted(pages))) == pinned
        return response([('perform', {'proposal': 'complete context delivery fixture'})], len(requests))

    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    actor, flow = native_actor(tmp_path, events, lambda *_: StageResult())
    try:
        actor.step({})
        actor.step(task)
    finally:
        actor.close()
    assert flow.stage == Stage.DESIGN and (len(requests) > 3 if oversized_guidance else len(requests) == 3)
    assert any(kind == 'pi-context-read' and row['record'] == reference for kind, row in events)
