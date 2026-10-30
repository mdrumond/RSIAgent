"""Behavior at the actor/controller boundary; no live provider calls."""
import hashlib
import json

import pytest

from rsi.actor import PiSessionAdapter, StageResult, WorkflowActor, SourceSnapshot
from rsi.contracts import BudgetLimits, Stage
from rsi.runtime import Workflow, WorkflowError
from rsi.skills import builtin_bundle


class Clock:
    value = 1000.0
    def __call__(self): return self.value


class Session:
    def __init__(self, actions, clock):
        self.actions, self.clock = iter(actions), clock
        self.contexts, self.deadlines = [], []
    def prepare(self, timeout, *, schemas): self.clock.value += 100
    def next_action(self, context, allowed, *, seconds, record, source):
        self.contexts.append(context)
        self.deadlines.append(seconds)
        self.clock.value += 2
        return next(self.actions)
    def close(self): pass


def harness(tmp_path, actions, *, stage=Stage.TASK_CONSTRUCTION, limit=20):
    clock, events = Clock(), []
    workflow = Workflow(tmp_path/'state.json', lambda *_: False, source_hash='project-v1',
                        limits=BudgetLimits(actor_seconds=limit), clock=clock)
    workflow.state['stage'] = stage.value
    workflow._save()
    session = Session(actions, clock)
    called = []
    def perform(stage, args, operation, deadline):
        assert workflow.pending['id'] == operation
        assert workflow.pending['handle'].startswith('pi:'+operation+':')
        called.append((stage, args, deadline))
        clock.value += 3
        return StageResult()
    actor = WorkflowActor(workflow, builtin_bundle(), lambda: session,
                          perform=perform, preflight=lambda *_: StageResult(),
                          emit=lambda kind, data: events.append((kind, data)))
    return actor, workflow, session, called, clock, events


def reads(stage):
    return [('read_skill', {'name': name}) for name in builtin_bundle().skills_for(stage)]


def test_reads_are_real_operation_scoped_and_setup_is_excluded(tmp_path):
    actor, flow, session, called, clock, events = harness(
        tmp_path, reads(Stage.TASK_CONSTRUCTION)+[('perform', {'proposal':'construct task'})])
    actor.step({'goal':'Python project'})
    assert flow.stage == Stage.DESIGN
    assert flow.accounting.seconds('setup') == 100
    assert flow.accounting.seconds('thinking') == 6
    assert flow.accounting.seconds() == 9
    assert session.deadlines == [20, 18, 16]
    assert called[0][2] == 14
    evidence = flow.state['history'][-1]['evidence']
    assert len(evidence['skill_refs']) == 2
    assert all(row[1]['operation_id'] == evidence['operation_id']
               for row in events if row[0] == 'skill-usage')
    assert session.contexts[-1]['observation']['receipt_id'] in evidence['skill_refs']


@pytest.mark.parametrize('skill_actions', [[], reads(Stage.TASK_CONSTRUCTION)[-1:]])
def test_model_cannot_perform_without_reads(tmp_path, skill_actions):
    actor, flow, _, called, _, _ = harness(tmp_path, [*skill_actions, ('perform', {'proposal':'skip'})])
    with pytest.raises(WorkflowError, match='read'):
        actor.step({})
    assert flow.pending and not called and not flow.state['history']


def test_unknown_skill_is_not_synthesised(tmp_path):
    actor, flow, _, called, _, events = harness(tmp_path, [('read_skill', {'name':'invented'})])
    with pytest.raises(ValueError): actor.step({})
    assert flow.pending and not called
    assert not any(row[1]['kind'] == 'read' for row in events if row[0] == 'skill-usage')


def test_preflight_is_host_only_and_does_not_start_model(tmp_path):
    actor, flow, session, called, _, events = harness(tmp_path, [], stage=Stage.PREFLIGHT)
    actor.step({})
    assert flow.stage == Stage.TASK_CONSTRUCTION
    assert not session.contexts and not called
    assert flow.accounting.seconds() == 0
    assert len([row for row in events if row[0]=='skill-usage' and row[1]['kind']=='read']) == 2


def test_design_does_not_get_fake_retrieval_or_retrospective(tmp_path):
    actor, flow, _, _, _, _ = harness(tmp_path, reads(Stage.DESIGN)+[('perform', {'proposal':'idea'})], stage=Stage.DESIGN)
    with pytest.raises(WorkflowError, match='retrieval and retrospective'): actor.step({})
    assert flow.stage == Stage.DESIGN and flow.pending


def test_pending_operation_from_restart_is_not_resubmitted(tmp_path):
    actor, flow, session, called, _, _ = harness(tmp_path, [])
    flow.begin()
    with pytest.raises(WorkflowError): actor.step({})
    assert not session.contexts and not called


def test_budget_exhaustion_after_thinking_does_not_dispatch(tmp_path):
    actor, flow, session, called, _, _ = harness(tmp_path, reads(Stage.TASK_CONSTRUCTION), limit=1)
    with pytest.raises(WorkflowError, match='budget'): actor.step({})
    assert flow.accounting.seconds() == 2 and not called
    assert session.deadlines == [1]


def test_source_snapshot_supports_python_paths_and_project_hash(tmp_path):
    snapshot = SourceSnapshot('src/solution.py', 'answer = 42\n', 'whole-project-hash')
    assert snapshot.sha256 == hashlib.sha256(snapshot.text.encode()).hexdigest()
    actor, flow, session, called, _, _ = harness(tmp_path, reads(Stage.IMPLEMENTATION)+[
        ('perform', {'proposal':'edit Python', 'edit':dict(path=snapshot.path, sha256=snapshot.sha256, source='answer = 43\n')})], stage=Stage.IMPLEMENTATION)
    flow.source_changed(snapshot.project_hash)
    actor.source_snapshot = lambda: snapshot
    actor.step({})
    assert session.contexts[-1]['source']['path'] == 'src/solution.py'
    assert called[0][1]['edit']['source'] == 'answer = 43\n'


@pytest.mark.parametrize('read_all', [True, False])
def test_real_pi_uses_existing_gateway_and_explicit_skill_reads(monkeypatch, tmp_path, read_all):
    from coding.pi.session import runtime_preflight, RuntimeUnavailable
    from test_pi_runtime import stream
    try: runtime_preflight()
    except RuntimeUnavailable: pytest.skip('Pi runtime is not installed')
    calls, executed = [], []
    actual_reads = reads(Stage.TASK_CONSTRUCTION) if read_all else reads(Stage.TASK_CONSTRUCTION)[-1:]
    actions = iter(actual_reads+[('perform', {'proposal':'construct'})])
    def provider(*_, **kwargs):
        calls.append(json.loads(kwargs['data']))
        name, args = next(actions)
        return stream(name, args, len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    flow = Workflow(tmp_path/'state.json', lambda *_: False, source_hash='v1')
    actor = WorkflowActor(flow, builtin_bundle(),
        lambda: PiSessionAdapter(model='deepseek-flash', environ={'DEEPSEEK_API_KEY':'fake-provider-secret'}),
        perform=lambda *args: executed.append(args) or StageResult(),
        preflight=lambda *_: StageResult(), emit=lambda *_: None)
    try:
        actor.step({})
        if read_all:
            actor.step({'goal':'write Python'})
        else:
            with pytest.raises(WorkflowError, match='read every mapped skill'):
                actor.step({'goal':'write Python'})
    finally: actor.close()
    assert len(calls) == len(actual_reads)+1
    if read_all:
        assert flow.stage == Stage.DESIGN and len(executed) == 1
        assert len(flow.state['history'][-1]['evidence']['skill_refs']) == 2
    else:
        assert flow.stage == Stage.TASK_CONSTRUCTION and not executed
        assert flow.pending['handle'].startswith('pi:')
    assert 'fake-provider-secret' not in json.dumps(calls)


def test_read_material_remains_available_after_later_reads(tmp_path):
    actor, _, session, _, _, _ = harness(tmp_path, reads(Stage.TASK_CONSTRUCTION)+[
        ('perform', {'proposal':'construct'})])
    actor.step({})
    material = session.contexts[-1]['read_skills']
    assert {item['skill'] for item in material} == set(builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION))
    assert all(item['content'] == builtin_bundle().document(item['skill']).content for item in material)


def test_section_read_does_not_prevent_later_full_contract_read(tmp_path):
    bundle = builtin_bundle()
    name = bundle.skills_for(Stage.TASK_CONSTRUCTION)[0]
    section = bundle.document(name).sections[0]
    actor, flow, _, _, _, _ = harness(tmp_path, [
        ('read_skill', {'name':name, 'section':section}),
        *reads(Stage.TASK_CONSTRUCTION), ('perform', {'proposal':'construct'})])
    actor.step({})
    assert flow.stage == Stage.DESIGN
    assert len(flow.state['history'][-1]['evidence']['skill_refs']) == 3


def test_cross_operation_skill_receipt_is_rejected(tmp_path):
    actor, flow, _, _, _, _ = harness(tmp_path, reads(Stage.TASK_CONSTRUCTION)+[
        ('perform', {'proposal':'construct'})])
    actor.step({})
    evidence = flow.state['history'][-1]['evidence']
    reference = evidence['skill_refs'][0]
    assert actor._resolve('skill', reference, Stage.TASK_CONSTRUCTION, evidence['operation_id'])
    assert not actor._resolve('skill', reference, Stage.TASK_CONSTRUCTION, 'another-operation')
    assert not actor._resolve('skill', reference, Stage.DESIGN, evidence['operation_id'])


def test_failed_thinking_is_charged_and_pending_preserved(tmp_path):
    actor, flow, session, called, clock, _ = harness(tmp_path, [])
    def fail(*args, **kwargs):
        clock.value += 4
        raise RuntimeError('provider failed')
    session.next_action = fail
    with pytest.raises(RuntimeError, match='provider failed'): actor.step({})
    assert flow.accounting.seconds() == 4 and flow.pending and not called


def test_stale_edit_is_not_executed(tmp_path):
    actor, flow, _, called, _, _ = harness(tmp_path, reads(Stage.IMPLEMENTATION)+[
        ('perform', {'proposal':'edit', 'edit':dict(path='src/solution.py',sha256='stale',source='x=1')})], stage=Stage.IMPLEMENTATION)
    actor.source_snapshot = lambda: SourceSnapshot('src/solution.py','x=0','project-v1')
    with pytest.raises(WorkflowError, match='stale'): actor.step({})
    assert not called and flow.pending['handle'].startswith('pi:')


def test_custom_skill_names_come_only_from_mapping(tmp_path):
    from rsi.skills import SkillBundle, SkillDocument
    names = ('project-plan', 'knowledge-guide')
    bundle = SkillBundle(tuple(SkillDocument(name, '# '+name+'\nRead me.') for name in names),
                         tuple((stage, names) for stage in Stage))
    actor, flow, session, _, _, _ = harness(tmp_path, [
        *[('read_skill', {'name':name}) for name in names], ('perform', {'proposal':'construct'})])
    actor.bundle = bundle
    actor.step({})
    assert flow.stage == Stage.DESIGN
    assert session.contexts[0]['skills'] == list(names)
    assert {item['skill'] for item in session.contexts[-1]['read_skills']} == set(names)


def test_host_evidence_tool_runs_in_current_operation_before_strategy(tmp_path):
    from coding.pi.schema import STRING, obj
    from rsi.actor import EvidenceTool
    actor, flow, session, _, _, _ = harness(tmp_path, reads(Stage.DESIGN)+[
        ('query_kdb', {'query':'stable sums'}), ('perform', {'proposal':'use retrieved idea'})], stage=Stage.DESIGN)
    retrieved = []
    def query(stage, args, operation, seconds):
        assert stage == Stage.DESIGN and flow.pending['id'] == operation
        retrieved.append((operation, args['query']))
        return {'retrieval_ref':'host-query-1', 'records':['negative cancellation result']}
    actor.evidence_tools = {'query_kdb': EvidenceTool(obj({'query':STRING}), query)}
    actor.schemas['query_kdb'] = actor.evidence_tools['query_kdb'].schema
    # PR6 owns the authoritative receipt resolver and retrospective store.
    with pytest.raises(WorkflowError, match='retrieval and retrospective'): actor.step({})
    assert retrieved and session.contexts[-1]['observation']['retrieval_ref'] == 'host-query-1'


def test_adapter_deadline_is_assigned_after_setup_and_gateway_matches(monkeypatch):
    from types import SimpleNamespace
    import rsi.actor as module
    clock = Clock()
    monkeypatch.setattr(module.time, 'monotonic', clock)
    captured = []
    raw = SimpleNamespace(gateway=SimpleNamespace(deadline=1001), deadline=1001,
        next_action=lambda *args, **kwargs: captured.append(kwargs) or ('perform', {'proposal':'edit'}))
    monkeypatch.setattr(module, 'PiCodingSession', lambda **kwargs: raw)
    adapter = PiSessionAdapter(model='deepseek-flash')
    clock.value += 100  # setup must not consume the subsequent model allocation
    source = SourceSnapshot('project.py','x=1','project-v1')
    adapter.next_action({'stage':Stage.IMPLEMENTATION.value}, ['perform'],
        seconds=7, record=lambda *_:None, source=source)
    assert raw.deadline == raw.gateway.deadline == 1107
    assert captured[-1]['write_action']('x=2') == ('perform', {
        'proposal':'Apply exact source edit', 'edit':dict(path='project.py',sha256=source.sha256,source='x=2')})
    adapter.next_action({'stage':Stage.DESIGN.value}, ['perform'],
        seconds=5, record=lambda *_:None, source=source)
    assert captured[-1]['write_action'] is None


def test_late_host_result_is_retained_without_advancing_workflow(tmp_path):
    actor, flow, _, _, clock, events = harness(tmp_path, reads(Stage.TASK_CONSTRUCTION)+[
        ('perform', {'proposal':'construct'})])
    def late(*args):
        clock.value += 30
        return StageResult(artifact_refs=('host-artifact',))
    actor.perform = late
    with pytest.raises(WorkflowError, match='budget'): actor.step({})
    assert flow.stage == Stage.TASK_CONSTRUCTION and flow.pending
    results = [data for kind, data in events if kind == 'stage-result']
    assert results[0]['result']['artifact_refs'] == ('host-artifact',)
    assert results[0]['operation_id'] == flow.pending['id']


def test_read_receipts_can_be_translated_to_strategy_cas_refs(tmp_path):
    actor, flow, session, _, _, _ = harness(tmp_path, reads(Stage.TASK_CONSTRUCTION)+[
        ('perform', {'proposal':'construct'})])
    actor.skill_reference = lambda raw: 'cas:'+raw
    actor.step({})
    refs = flow.state['history'][-1]['evidence']['skill_refs']
    assert all(ref.startswith('cas:') for ref in refs)
    assert all(item['receipt_id'] == 'cas:'+item['raw_receipt_id']
               for item in session.contexts[-1]['read_skills'])


def test_durable_pending_result_reconciles_without_new_model_or_reads(tmp_path):
    from rsi.contracts import StepEvidence
    actor, flow, _, _, clock, events = harness(tmp_path, reads(Stage.TASK_CONSTRUCTION)+[
        ('perform', {'proposal':'construct'})])
    def interrupt(*args, **kwargs): raise RuntimeError('controller interrupted')
    flow.complete = interrupt
    with pytest.raises(RuntimeError, match='interrupted'): actor.step({})
    pending = flow.pending
    trusted_reads = {data['receipt_id']:data for kind,data in events
                     if kind == 'skill-usage' and data['kind'] == 'read'}
    def authority(kind, reference, stage, operation):
        row = trusted_reads.get(reference)
        return bool(kind == 'skill' and row and row['stage'] == stage.value
                    and row['operation_id'] == operation)
    reopened = Workflow(flow.path, authority, source_hash=flow.source_hash, clock=clock)
    restored = WorkflowActor(reopened, builtin_bundle(), lambda: pytest.fail('model restarted'),
        perform=lambda *_: pytest.fail('redispatch'), preflight=lambda *_: pytest.fail('preflight'),
        emit=lambda *_: pytest.fail('new event'))
    receipt = next(data for kind,data in events if kind == 'stage-result')
    restored.reconcile_pending(StepEvidence.from_dict(receipt['evidence']), StageResult(**receipt['result']))
    assert reopened.stage == Stage.DESIGN and reopened.pending is None
    assert receipt['operation_id'] == pending['id']


def test_pi_dispatch_identity_is_durable_before_thinking(tmp_path):
    actor, flow, session, _, _, events = harness(tmp_path, [])
    def fail(*args, **kwargs):
        assert flow.pending['handle'].startswith('pi:'+flow.pending['id']+':')
        dispatch = next(data for kind,data in events if kind == 'actor-dispatch')
        request = next(data for kind,data in events if kind == 'actor-request')
        assert dispatch['handle'] == request['handle'] == flow.pending['handle']
        assert request['request_id']
        kwargs['record']('pi-provider-request', {'provider':'fake'})
        raise RuntimeError('provider failed')
    session.next_action = fail
    with pytest.raises(RuntimeError, match='provider failed'): actor.step({})
    request = next(data for kind,data in events if kind == 'actor-request')
    provider = next(data for kind,data in events if kind == 'pi-provider-request')
    assert request['request_id'] == provider['request_id']


@pytest.mark.parametrize('with_context_read', [False, True])
def test_real_pi_batched_calls_execute_only_first_host_action(monkeypatch, tmp_path, with_context_read):
    from coding.pi.session import runtime_preflight, RuntimeUnavailable
    from test_pi_runtime import stream
    try: runtime_preflight()
    except RuntimeUnavailable: pytest.skip('Pi runtime is not installed')
    bundle = builtin_bundle()
    names = bundle.skills_for(Stage.TASK_CONSTRUCTION)
    events, executed, requests = [], [], []
    guide = 'Python project context. ' * 100

    def actual_reads():
        return [data['skill'] for kind,data in events if kind == 'skill-usage'
                and data['kind'] == 'read' and data['stage'] == Stage.TASK_CONSTRUCTION.value]

    def provider(*_, **kwargs):
        request = json.loads(kwargs['data'])
        requests.append(request)
        index = len(requests)
        if with_context_read and index == 1:
            snapshot = json.loads(next(m['content'] for m in request['messages'] if m['role']=='user'))
            reference = snapshot['context']['host']['task']['guide']['record']
            actions = [('read_context', {'record':reference, 'limit':128})]
        elif index == 1 + int(with_context_read):
            actions = [('read_skill', {'name':name}) for name in names]
            actions.append(('perform', {'proposal':'phantom execution'}))
            if with_context_read:
                pages = [json.loads(m['content']) for m in request['messages'] if m['role']=='tool']
                assert pages[0]['text'] == guide[:128]
        elif index == 2 + int(with_context_read):
            assert actual_reads() == [names[0]] and not executed
            actions = [('read_skill', {'name':names[1]})]
        else:
            assert index == 3 + int(with_context_read) and actual_reads() == list(names) and not executed
            actions = [('perform', {'proposal':'verified host invocation'})]
        response = stream(*actions[0], index)
        original = response.iter_lines
        def lines():
            for line in original():
                if line.startswith(b'data: {'):
                    body = json.loads(line[6:])
                    delta = body['choices'][0]['delta']
                    if delta.get('tool_calls'):
                        delta['tool_calls'] = [dict(index=i, id=f'batch_{index}_{i}', type='function',
                            function=dict(name=name, arguments=json.dumps(args)))
                            for i,(name,args) in enumerate(actions)]
                        line = b'data: ' + json.dumps(body).encode()
                yield line
        response.iter_lines = lines
        return response

    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    flow = Workflow(tmp_path/'state.json', lambda *_:False, source_hash='v1')
    actor = WorkflowActor(flow, bundle,
        lambda:PiSessionAdapter(model='deepseek-flash', environ={'DEEPSEEK_API_KEY':'fake-provider-secret'}),
        perform=lambda stage,args,*_:executed.append(args['proposal']) or StageResult(),
        preflight=lambda *_:StageResult(), emit=lambda kind,data:events.append((kind,data)))
    try:
        actor.step({})
        actor.step({'guide':guide})
    finally:
        actor.close()
    assert flow.stage == Stage.DESIGN and len(requests) == 3 + int(with_context_read)
    assert actual_reads() == list(names)
    assert executed == ['verified host invocation']
    actions = [data for kind,data in events if kind == 'pi-action']
    assert [data['discarded_actions'] for data in actions] == [2, 0, 0]
    context_reads = [data for kind,data in events if kind == 'pi-context-read']
    assert len(context_reads) == int(with_context_read)
    if with_context_read:
        assert context_reads[0]['record'] == hashlib.sha256(guide.encode()).hexdigest()
        assert context_reads[0]['bytes'] == 128
