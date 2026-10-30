"""Live host inputs supersede unexecuted decisions within the shared actor loop."""
from copy import deepcopy

import pytest

from coding.pi.schema import obj
from rsi.actor import EvidenceTool
from rsi.contracts import Stage
from rsi.runtime import WorkflowError
from test_rsi_actor import harness, reads


@pytest.mark.parametrize('stale', [
    ('perform', {'proposal': 'obsolete host command'}),
    ('perform', {'proposal': 'obsolete edit', 'edit': {
        'path': 'outside-admitted-source', 'sha256': 'stale', 'source': 'obsolete'}}),
    ('diagnostic', {}),
])
def test_live_input_supersedes_selection_before_host_edit_or_evidence(tmp_path, stale):
    stage = Stage.IMPLEMENTATION
    actor, flow, session, called, clock, events = harness(tmp_path,
        reads(stage) + [stale, ('perform', {'proposal': 'follow new feedback'})],
        stage=stage, limit=50)
    diagnostic = []
    actor.evidence_tools['diagnostic'] = EvidenceTool(obj({}), lambda *_: diagnostic.append(True))
    actor.schemas['diagnostic'] = obj({})
    arrived = False
    selected = session.next_action
    observed = []

    def select(context, *args, **kwargs):
        nonlocal arrived
        observed.append(deepcopy(context))
        action = selected(context, *args, **kwargs)
        if action == stale:
            arrived = True
        return action

    session.next_action = select
    def refresh(context):
        nonlocal arrived
        if arrived:
            arrived = False
            clock.value += 1
            return dict(context, feedback=['preserve source and use new constraints'])
        return None
    actor.context_refresh = refresh
    actor.step({'goal': 'implement', 'feedback': []})
    assert [row[1] for row in called] == [{'proposal': 'follow new feedback'}]
    assert not diagnostic
    assert observed[-2]['task']['feedback'] == []
    assert observed[-1]['task']['feedback'] == ['preserve source and use new constraints']
    assert observed[-2]['operation_id'] == observed[-1]['operation_id']
    assert observed[-2]['skill_receipts'] == observed[-1]['skill_receipts']
    assert observed[-1]['unread_skills'] == []
    assert len(flow.state['history']) == 1 and flow.pending is None
    assert flow.accounting.seconds('thinking') == 2 * len(observed)
    assert flow.accounting.seconds('retrieval') == 1
    assert flow.accounting.seconds() == 2 * len(observed) + 4
    superseded = [data for kind, data in events if kind == 'actor-action-superseded']
    assert len(superseded) == 1 and superseded[0]['executed'] is False


@pytest.mark.parametrize('failure', ['stop', 'error', 'budget'])
def test_refresh_failure_keeps_pending_and_never_executes_selected_action(tmp_path, failure):
    actor, flow, session, called, clock, _ = harness(tmp_path,
        reads(Stage.TASK_CONSTRUCTION) + [('perform', {'proposal': 'construct'})], limit=30)
    original = session.next_action
    def select(*args, **kwargs):
        action = original(*args, **kwargs)
        if action[0] == 'perform':
            def refresh(context):
                if failure == 'error':
                    raise RuntimeError('host input unavailable')
                if failure == 'stop':
                    flow.stop()
                else:
                    clock.value += 31
                return None
            actor.context_refresh = refresh
        return action
    session.next_action = select
    with pytest.raises(RuntimeError if failure == 'error' else WorkflowError):
        actor.step({'goal': 'construct'})
    assert flow.pending and not called and not flow.state['history']
    assert flow.accounting.seconds('thinking') > 0


def test_no_changed_input_preserves_existing_decision_path(tmp_path):
    actor, flow, _, called, _, events = harness(tmp_path,
        reads(Stage.TASK_CONSTRUCTION) + [('perform', {'proposal': 'construct'})])
    checks = []
    actor.context_refresh = lambda context: checks.append(context) or None
    actor.step({'goal': 'construct'})
    assert len(checks) == len(reads(Stage.TASK_CONSTRUCTION)) + 1
    assert len(called) == 1 and flow.stage is Stage.DESIGN
    assert not any(kind == 'actor-action-superseded' for kind, _ in events)


def test_native_pi_receives_refreshed_task_and_rejects_remaining_stale_batch(monkeypatch, tmp_path):
    import json
    from test_rsi_pi_controller import native_actor, response
    from rsi.actor import StageResult
    from rsi.skills import builtin_bundle
    events, requests, called, diagnostics = [], [], [], []
    arrived = False
    names = builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)

    def provider(*_, **kwargs):
        nonlocal arrived
        request = json.loads(kwargs['data'])
        requests.append(request)
        if len(requests) == 1:
            return response([('read_skill', {'name': name}) for name in names], 1)
        if len(requests) == 2:
            arrived = True
            return response([('perform', {'proposal': 'stale decision'}), ('diagnostic', {})], 2)
        assert len(requests) == 3
        results = []
        for message in request['messages']:
            if message['role'] == 'tool':
                try:
                    results.append(json.loads(message['content']))
                except json.JSONDecodeError:
                    pass  # Pi may also retain its own cancelled-tool observations.
        changed = [row for row in results if row.get('observation', {}).get('event') == 'action-superseded']
        assert len(changed) == 1
        assert changed[0]['task']['feedback'] == ['use the revised host constraints']
        return response([('perform', {'proposal': 'follow revised host constraints'})], 3)

    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    def perform(stage, args, operation, seconds):
        called.append(args)
        return StageResult()
    actor, flow = native_actor(tmp_path, events, perform,
        {'diagnostic': EvidenceTool(obj({}), lambda *_: diagnostics.append(True))})
    def refresh(context):
        nonlocal arrived
        if arrived:
            arrived = False
            return dict(context, feedback=['use the revised host constraints'])
        return None
    actor.context_refresh = refresh
    try:
        actor.step({'goal': 'native refresh', 'feedback': []})
        actor.step({'goal': 'native refresh', 'feedback': []})
    finally:
        actor.close()
    assert called == [{'proposal': 'follow revised host constraints'}]
    assert not diagnostics and flow.stage is Stage.DESIGN
    assert len(requests) == 3
    assert len([row for kind, row in events if kind == 'actor-action-superseded']) == 1
    assert len([row for kind, row in events if kind == 'skill-usage' and row['kind'] == 'read'
                and row['stage'] == Stage.TASK_CONSTRUCTION.value]) == len(names)
