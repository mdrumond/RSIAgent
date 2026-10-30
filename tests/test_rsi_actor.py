"""Behavior at the actor/controller boundary; no live provider calls."""
import hashlib
import json

import pytest

# Snapshot transport compatibility; production native transport is tested separately.
from rsi.actor import SnapshotPiSessionAdapter as PiSessionAdapter
from rsi.actor import StageResult, WorkflowActor, SourceSnapshot
from rsi.contracts import BudgetLimits, Stage
from rsi.runtime import Workflow, WorkflowError
from rsi.skills import builtin_bundle


class Clock:
    value = 1000.0
    def __call__(self): return self.value


class Session:
    def __init__(self, actions, clock):
        self.actions, self.clock = iter(actions), clock
        self.contexts, self.deadlines, self.allowed = [], [], []
    def prepare(self, timeout, *, schemas): self.clock.value += 100
    def next_action(self, context, allowed, *, seconds, record, source):
        self.contexts.append(context)
        self.allowed.append(list(allowed))
        self.deadlines.append(seconds)
        self.clock.value += 2
        return next(self.actions)
    def close(self): pass


def harness(tmp_path, actions, *, stage=Stage.TASK_CONSTRUCTION, limit=20,
            execution_cap=20, managed_executions=False):
    clock, events = Clock(), []
    workflow = Workflow(tmp_path/'state.json', lambda *_: False, source_hash='project-v1',
                        limits=BudgetLimits(actor_seconds=limit, execution_tries=execution_cap), clock=clock)
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
                          emit=lambda kind, data: events.append((kind, data)),
                          managed_executions=managed_executions)
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


@pytest.mark.parametrize('section', ['', '   '])
def test_empty_optional_section_delivers_full_contract(tmp_path, section):
    actions = [(name, dict(args, section=section))
               for name, args in reads(Stage.TASK_CONSTRUCTION)]
    actor, flow, session, called, _, events = harness(
        tmp_path, actions+[('perform', {'proposal': 'construct'})])
    actor.step({})
    assert flow.stage == Stage.DESIGN and len(called) == 1
    delivered = session.contexts[-1]['read_skills']
    assert len(delivered) == len(actions)
    for row in delivered:
        assert row['section'] is None
        assert row['content'] == builtin_bundle().document(row['skill']).content


@pytest.mark.parametrize('bad_read', [
    {'name': 'invented'},
    {'name': builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)[0],
     'section': 'invented section'},
])
def test_invalid_skill_read_can_recover_without_fabricating_receipts(tmp_path, bad_read):
    actor, flow, session, called, _, events = harness(tmp_path, [
        ('read_skill', bad_read), *reads(Stage.TASK_CONSTRUCTION),
        ('perform', {'proposal': 'construct'})])
    actor.recover_invalid_actions = True
    actor.step({})
    assert flow.stage == Stage.DESIGN and len(called) == 1
    assert session.contexts[1]['observation']['accepted'] is False
    actual_reads = [data for kind, data in events
                    if kind == 'skill-usage' and data['kind'] == 'read']
    assert len(actual_reads) == len(reads(Stage.TASK_CONSTRUCTION))
    assert flow.accounting.seconds('thinking') == 8
    assert flow.accounting.seconds() == 11


def test_preflight_is_host_only_and_does_not_start_model(tmp_path):
    actor, flow, session, called, _, events = harness(tmp_path, [], stage=Stage.PREFLIGHT)
    actor.step({})
    assert flow.stage == Stage.TASK_CONSTRUCTION
    assert not session.contexts and not called
    assert flow.accounting.seconds() == 0
    assert len([row for row in events if row[0]=='skill-usage' and row[1]['kind']=='read']) == 2


def test_skill_names_are_rejected_then_real_read_references_can_be_selected(tmp_path):
    from rsi.actor import EvidenceTool
    from coding.pi.schema import obj, STRING
    names = list(builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION))
    actor, flow, session, called, _, events = harness(tmp_path, [
        *reads(Stage.TASK_CONSTRUCTION),
        ('select_strategy', {'skill_refs': names}),
        ('select_strategy', {'skill_refs': []}),
        ('perform', {'proposal': 'construct'})])
    actor.recover_invalid_actions = True
    selected = []
    tool = EvidenceTool(obj({'skill_refs': {'type': 'array', 'items': STRING}}, ['skill_refs']),
                        lambda stage,args,*_: selected.append(args) or {})
    actor.evidence_tools['select_strategy'] = tool
    actor.schemas['select_strategy'] = tool.schema
    original = session.next_action
    def next_action(context, *args, **kwargs):
        name, arguments = original(context, *args, **kwargs)
        if name == 'select_strategy' and not arguments['skill_refs']:
            arguments = {'skill_refs': [r['receipt_id'] for r in context['skill_receipts']]}
        return name, arguments
    session.next_action = next_action
    actor.step({})
    assert flow.stage == Stage.DESIGN and len(called) == 1
    assert len(selected) == 1
    assert set(selected[0]['skill_refs']) == set(actor._reads)
    assert set(selected[0]['skill_refs']).isdisjoint(names)
    assert session.contexts[-2]['observation']['accepted'] is False
    assert len([d for k,d in events if k == 'actor-invalid-action']) == 1
    assert flow.accounting.seconds('thinking') == 10


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
        action = next(actions, None)
        if action is not None:
            return stream(*action, len(calls))
        # Native Pi reports the unavailable tool and asks the provider again.
        # End that request without selecting another action; do not fake a read.
        response = stream('perform', {}, len(calls), finish='stop')
        original = response.iter_lines
        def lines():
            for line in original():
                if line.startswith(b'data: {'):
                    body = json.loads(line[6:])
                    delta = body['choices'][0]['delta']
                    if delta.pop('tool_calls', None):
                        delta['content'] = 'No further action selected.'
                    line = b'data: ' + json.dumps(body).encode()
                yield line
        response.iter_lines = lines
        return response
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
            from coding.pi.session import ActionError
            with pytest.raises(ActionError):
                actor.step({'goal':'write Python'})
    finally: actor.close()
    assert len(calls) == len(actual_reads)+(1 if read_all else 2)
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
    raw = SimpleNamespace(gateway=SimpleNamespace(deadline=1001), deadline=1001, schemas={},
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
    adapter.next_action({'stage':Stage.IMPLEMENTATION.value}, ['read_skill'],
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
    results = [data for kind, data in events if kind == 'stage-result-proposed']
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
            if with_context_read:
                pages = [json.loads(m['content']) for m in request['messages'] if m['role']=='tool']
                assert pages[0]['text'] == guide[:128]
        elif index == 2 + int(with_context_read):
            assert actual_reads() == [names[0]] and not executed
            actions = [('read_skill', {'name':names[1]})]
        else:
            assert index == 3 + int(with_context_read) and actual_reads() == list(names) and not executed
            actions = [('perform', {'proposal':'verified host invocation'}),
                       ('perform', {'proposal':'phantom execution'})]
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
    assert [data['discarded_actions'] for data in actions] == [1, 0, 1]
    context_reads = [data for kind,data in events if kind == 'pi-context-read']
    assert len(context_reads) == int(with_context_read)
    if with_context_read:
        assert context_reads[0]['record'] == hashlib.sha256(guide.encode()).hexdigest()
        assert context_reads[0]['bytes'] == 128


@pytest.mark.parametrize('stage', [Stage.TEST, Stage.VALIDATION])
@pytest.mark.parametrize('managed', [False, True])
def test_exhausted_execution_cap_refuses_stage_callback(tmp_path, stage, managed):
    actor, flow, _, called, _, events = harness(tmp_path, reads(stage)+[
        ('perform', {'proposal':'run command'})], stage=stage, execution_cap=1,
        managed_executions=managed)
    flow.state['executions'] = 1
    flow._save()
    with pytest.raises(WorkflowError, match='execution cap'): actor.step({})
    assert not called and flow.state['executions'] == 1
    assert not any(kind == 'host-dispatch' for kind,_ in events)
    assert len([data for kind,data in events if kind == 'skill-usage' and data['kind'] == 'read']) == len(reads(stage))


@pytest.mark.parametrize('stage', [Stage.TEST, Stage.VALIDATION])
def test_default_callback_reserves_one_execution(tmp_path, stage):
    actor, flow, _, called, _, _ = harness(tmp_path, reads(stage)+[
        ('perform', {'proposal':'run command'})], stage=stage, execution_cap=1)
    actor.step({})
    assert len(called) == 1 and flow.state['executions'] == 1
    operation = flow.state['history'][-1]['evidence']['operation_id']
    assert flow.state['execution_reservations'][operation]['stage_operation'] == operation


def test_reads_without_admitted_perform_do_not_reserve_execution(tmp_path):
    actor, flow, _, called, _, _ = harness(tmp_path, [reads(Stage.TEST)[-1],
        ('perform', {'proposal':'skip step skill'})], stage=Stage.TEST)
    with pytest.raises(WorkflowError, match='read every mapped skill'): actor.step({})
    assert not called and flow.state['executions'] == 0
    assert not flow.state.get('execution_reservations')


@pytest.mark.parametrize('managed', [False, True])
def test_disabled_execution_cap_allows_additional_work(tmp_path, managed):
    actor, flow, _, _, _, _ = harness(tmp_path, reads(Stage.TEST)+[
        ('perform', {'proposal':'run command'})], stage=Stage.TEST, execution_cap=None,
        managed_executions=managed)
    flow.state['executions'] = 100
    flow._save()
    if managed:
        def backend(stage, args, operation, seconds):
            assert flow.reserve_execution(operation+':command')
            return StageResult()
        actor.perform = backend
    actor.step({})
    assert flow.state['executions'] == 101


def test_managed_backend_reserves_commands_without_stage_double_count(tmp_path):
    actor, flow, _, _, _, _ = harness(tmp_path, reads(Stage.TEST)+[
        ('perform', {'proposal':'compile test profile'})], stage=Stage.TEST,
        execution_cap=3, managed_executions=True)
    commands = []
    def backend(stage, args, operation, seconds):
        for command in ('compile', 'test', 'profile'):
            command_id = operation+':'+command
            assert flow.reserve_execution(command_id)
            assert not flow.reserve_execution(command_id)  # retry cannot admit replay
            commands.append(command_id)
        return StageResult()
    actor.perform = backend
    actor.step({})
    assert len(commands) == flow.state['executions'] == 3
    assert set(flow.state['execution_reservations']) == set(commands)


def test_default_multi_command_callback_reserves_additional_ids(tmp_path):
    actor, flow, _, _, _, _ = harness(tmp_path, reads(Stage.TEST)+[
        ('perform', {'proposal':'three commands'})], stage=Stage.TEST, execution_cap=3)
    def callback(stage, args, operation, seconds):
        assert flow.state['executions'] == 1
        for suffix in ('extra-test', 'profile'):
            assert flow.reserve_execution(operation+':'+suffix)
        return StageResult()
    actor.perform = callback
    actor.step({})
    assert flow.state['executions'] == len(flow.state['execution_reservations']) == 3


def test_perform_waits_for_all_full_reads_and_host_readiness(tmp_path):
    from coding.pi.schema import obj
    from rsi.actor import EvidenceTool
    ready = []
    actor, flow, session, called, _, _ = harness(tmp_path, reads(Stage.TASK_CONSTRUCTION)+[
        ('query_kdb', {}), ('perform', {'proposal':'construct'})])
    actor.perform_ready = lambda stage,operation: bool(ready)
    actor.evidence_tools = {'query_kdb':EvidenceTool(obj(), lambda *args:ready.append(args[2]) or {})}
    actor.schemas['query_kdb'] = actor.evidence_tools['query_kdb'].schema
    actor.step({})
    assert ['perform' in names for names in session.allowed] == [False, False, False, True]
    assert session.contexts[0]['read_skill_names'] == []
    assert session.contexts[0]['unread_skills'] == list(builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION))
    assert session.contexts[-1]['unread_skills'] == []
    assert session.contexts[-1]['read_skill_names'] == list(builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION))
    assert len(called) == 1 and flow.stage == Stage.DESIGN


def test_scripted_perform_cannot_bypass_host_readiness(tmp_path):
    actor, flow, _, called, _, _ = harness(tmp_path, reads(Stage.TASK_CONSTRUCTION)+[
        ('perform', {'proposal':'skip selection'})])
    actor.perform_ready = lambda *_:False
    with pytest.raises(WorkflowError, match='unavailable'): actor.step({})
    assert not called and flow.stage == Stage.TASK_CONSTRUCTION


def test_scripted_tool_cannot_bypass_stage_scope(tmp_path):
    from coding.pi.schema import obj
    from rsi.actor import EvidenceTool
    actor, _, _, _, _, _ = harness(tmp_path, [('select_idea', {})])
    actor.evidence_tools = {'select_idea':EvidenceTool(obj(), lambda *_:pytest.fail('wrong-stage call'),
                                                    stages=(Stage.DESIGN, Stage.OPTIMISATION))}
    actor.schemas['select_idea'] = actor.evidence_tools['select_idea'].schema
    with pytest.raises(WorkflowError, match='unavailable'): actor.step({})


def test_real_pi_advertises_only_scoped_ready_tools(monkeypatch, tmp_path):
    from coding.pi.schema import obj
    from coding.pi.session import runtime_preflight, RuntimeUnavailable
    from rsi.actor import EvidenceTool
    from test_pi_runtime import stream
    try: runtime_preflight()
    except RuntimeUnavailable: pytest.skip('Pi runtime is not installed')
    from rsi.skills import SkillBundle, SkillDocument
    base = builtin_bundle()
    bundle = SkillBundle(tuple(SkillDocument(doc.name, doc.content+'\n'+('actual guidance ' * 500))
                               for doc in base.documents), base.mapping)
    ready, requests, executed, page_offsets = [], [], [], {}
    flow = Workflow(tmp_path/'state.json', lambda *_:False, source_hash='v1')
    names = bundle.skills_for(Stage.TASK_CONSTRUCTION)
    actions = iter([*[('read_skill', {'name':name}) for name in names],
                    ('query_kdb', {}), ('perform', {'proposal':'construct'})])
    def provider(*_, **kwargs):
        request = json.loads(kwargs['data'])
        requests.append(request)
        advertised = {tool['function']['name'] for tool in request['tools']}
        assert not advertised & {'select_idea', 'optimisation_decision'}
        assert 'query_kdb' in advertised
        perform = next((t['function'] for t in request['tools'] if t['function']['name']=='perform'), None)
        if perform:
            assert 'edit' not in perform['parameters']['properties']
        assert ('perform' in advertised) == bool(ready)
        if ready:
            snapshot = json.loads(next(row['content'] for row in request['messages'] if row['role']=='user'))['context']['host']
            assert snapshot['read_skills']['read_with'] == snapshot['evidence']['read_with'] == 'read_context'
            assert {row['skill'] for row in snapshot['skill_receipts']} == set(names)
            assert all(row['receipt_id'] for row in snapshot['skill_receipts'])
            assert snapshot['evidence_receipts'] == [{'name':'query_kdb','reference':'actual-query-receipt'}]
            assert snapshot['evidence_summaries']['query_kdb']['query_ref'] == 'actual-query-receipt'
        if ready and snapshot['observation'].get('event') == 'skill-delivery-required':
            for message in request['messages']:
                if message['role']=='tool':
                    page = json.loads(message['content'])
                    if 'record' in page and 'next_offset' in page:
                        page_offsets[page['record']] = page['next_offset']
            for row in snapshot['observation']['reads'].values():
                reference = row['content']['record']
                if reference not in page_offsets or page_offsets[reference] is not None:
                    return stream('read_context', dict(record=reference, offset=page_offsets.get(reference,0), limit=8192), len(requests))
            return stream('perform', {'proposal':'construct'}, len(requests))
        name,args = next(actions)
        return stream(name,args,len(requests))
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    actor = WorkflowActor(flow, bundle,
        lambda:PiSessionAdapter(model='deepseek-flash', environ={'DEEPSEEK_API_KEY':'fake-provider-secret'}),
        perform=lambda *_:executed.append(True) or StageResult(), preflight=lambda *_:StageResult(),
        emit=lambda *_:None, perform_ready=lambda *_:bool(ready), evidence_tools={
            'query_kdb':EvidenceTool(obj(), lambda *args:ready.append(args[2]) or dict(reference='actual-query-receipt',
                summary=dict(query_ref='actual-query-receipt', counts={'results':0}),
                evidence={'retained':'actual result ' * 600})),
            'select_idea':EvidenceTool(obj(), lambda *_:pytest.fail('wrong-stage selection'),
                                      stages=(Stage.DESIGN,Stage.OPTIMISATION)),
            'optimisation_decision':EvidenceTool(obj(), lambda *_:pytest.fail('wrong-stage optimisation'),
                                                stages=(Stage.OPTIMISATION,))})
    try:
        actor.step({})
        actor.step({'goal':'normal Python task construction'})
    finally: actor.close()
    expected_pages = sum((len(bundle.document(name).content.encode())+8191)//8192 for name in names)
    assert flow.stage == Stage.DESIGN and len(requests) == 5+expected_pages and executed == [True]


@pytest.mark.parametrize('first_name,first_args', [
    ('read_skill', {'name':'task-construction'}), ('perform', {'proposal':'first model action'})])
def test_ordered_batch_chooses_model_order_even_if_requests_arrive_reversed(monkeypatch, first_name, first_args):
    import coding.pi.session as module
    from rsi.actor import SCHEMAS
    monkeypatch.setattr(module, 'runtime_preflight', lambda:None)
    with module.PiCodingSession(model='deepseek-flash', schemas=SCHEMAS,
            environ={'DEEPSEEK_API_KEY':'fake-provider-secret'}, first_action_only=True) as pi:
        pi.allowed = {'read_context', 'read_skill', 'perform'}
        reference = pi.context.admit('actual context page')['record']
        calls = [dict(id='page', name='read_context', arguments={'record':reference}),
                 dict(id='first', name=first_name, arguments=first_args),
                 dict(id='later', name='perform', arguments={'proposal':'phantom'})]
        assert pi._operation('propose_batch', dict(batch=1, calls=calls))['accepted']
        with pytest.raises(ValueError, match='not executed'):
            pi._operation('execute_call', dict(batch=1, id='later'))
        assert pi.pending is None
        assert pi._operation('execute_call', dict(batch=1, id='page'))['text'] == 'actual context page'
        pi._operation('execute_call', dict(batch=1, id='first'))
        assert pi.pending == (first_name, first_args) and pi.discarded_actions == 1
        with pytest.raises(ValueError, match='already handled'):
            pi._operation('execute_call', dict(batch=1, id='first'))


def test_visible_receipt_inventory_survives_large_material_paging(tmp_path):
    from coding.pi.context import ContextRecords
    from coding.pi.schema import obj
    from rsi.actor import EvidenceTool
    from rsi.skills import SkillBundle, SkillDocument
    baseline = builtin_bundle()
    bundle = SkillBundle(tuple(SkillDocument(doc.name, '# Contract\n'+('actual guidance ' * 500))
                               for doc in baseline.documents), baseline.mapping)
    actor, _, session, _, _, events = harness(tmp_path, reads(Stage.TASK_CONSTRUCTION)+[
        ('query_kdb', {}), ('query_kdb', {}), ('perform', {'proposal':'construct'})])
    actor.bundle = bundle
    actor.skill_reference = lambda raw:'cas:'+raw
    results = []
    def query(*args):
        reference = 'actual-query-'+str(len(results))
        result = dict(reference=reference, evidence={'rows':'retained row ' * 700},
                      summary=dict(query_ref=reference, counts={'results':2}, decision_indexes={'results':[0,1]}))
        results.append(result)
        return result
    actor.evidence_tools = {'query_kdb':EvidenceTool(obj(), query)}
    actor.schemas['query_kdb'] = actor.evidence_tools['query_kdb'].schema
    actor.step({})
    payload = session.contexts[-1]
    compact = ContextRecords().snapshot({'host':payload})['host']
    assert compact['read_skills']['read_with'] == compact['evidence']['read_with'] == 'read_context'
    assert len(compact['skill_receipts']) == len(bundle.skills_for(Stage.TASK_CONSTRUCTION))
    actual_reads = {data['skill']:'cas:'+data['receipt_id'] for kind,data in events
                    if kind=='skill-usage' and data['kind']=='read'}
    assert {row['skill']:row['receipt_id'] for row in compact['skill_receipts']} == actual_reads
    assert compact['evidence_receipts'] == [{'name':'query_kdb','reference':results[-1]['reference']}]
    assert compact['evidence_summaries']['query_kdb'] == results[-1]['summary']
    assert len(payload['evidence']) == 2 and all(item['result']['evidence'] for item in payload['evidence'])


def test_optional_invalid_action_feedback_preserves_operation_and_bills_thinking(tmp_path):
    from coding.pi.session import ActionError
    actor, flow, session, called, clock, events = harness(tmp_path, reads(Stage.TASK_CONSTRUCTION)+[
        ('perform', {'proposal':'construct'})])
    actor.recover_invalid_actions = True
    next_action = session.next_action
    failed = []
    def select(*args, **kwargs):
        if not failed:
            failed.append(True)
            clock.value += 2
            raise ActionError('missing query_ref')
        return next_action(*args, **kwargs)
    session.next_action = select
    actor.step({})
    assert len(called) == len(flow.state['history']) == 1
    assert session.contexts[0]['observation']['event'] == 'invalid-action'
    assert session.contexts[0]['observation']['accepted'] is False
    assert flow.accounting.seconds('thinking') == 8
    feedback = [data for kind,data in events if kind=='actor-invalid-action']
    assert len(feedback) == 1
    assert feedback[0]['operation_id'] == flow.state['history'][0]['evidence']['operation_id']


def test_optional_action_feedback_does_not_mask_provider_failure(tmp_path):
    from coding.pi.rpc import PiProviderError
    actor, flow, session, called, clock, events = harness(tmp_path, [])
    actor.recover_invalid_actions = True
    def fail(*args, **kwargs):
        clock.value += 2
        raise PiProviderError('provider unavailable')
    session.next_action = fail
    with pytest.raises(PiProviderError): actor.step({})
    assert not called and flow.pending and flow.accounting.seconds('thinking') == 2
    assert not any(kind=='actor-invalid-action' for kind,_ in events)


def test_real_pi_recovers_invalid_arguments_without_phantom_reads(monkeypatch, tmp_path):
    from coding.pi.session import runtime_preflight, RuntimeUnavailable
    from test_pi_runtime import stream
    try: runtime_preflight()
    except RuntimeUnavailable: pytest.skip('Pi runtime is not installed')
    names = builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    events, requests, executed = [], [], []
    def actual_reads():
        return [data['skill'] for kind,data in events if kind=='skill-usage'
                and data['kind']=='read' and data['stage']==Stage.TASK_CONSTRUCTION.value]
    def provider(*_, **kwargs):
        requests.append(json.loads(kwargs['data']))
        assert len(requests) <= 6
        if len(requests) == 1:
            return stream('read_skill', {'name':123}, len(requests))
        missing = [name for name in names if name not in actual_reads()]
        return stream('read_skill', {'name':missing[0]}, len(requests)) if missing else stream(
            'perform', {'proposal':'construct'}, len(requests))
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    flow = Workflow(tmp_path/'state.json', lambda *_:False, source_hash='v1')
    actor = WorkflowActor(flow, builtin_bundle(),
        lambda:PiSessionAdapter(model='deepseek-flash', environ={'DEEPSEEK_API_KEY':'fake-provider-secret'}),
        perform=lambda *_:executed.append(True) or StageResult(), preflight=lambda *_:StageResult(),
        emit=lambda kind,data:events.append((kind,data)), recover_invalid_actions=True)
    try:
        actor.step({})
        actor.step({'goal':'construct Python project'})
    finally: actor.close()
    assert flow.stage == Stage.DESIGN and executed == [True] and actual_reads() == list(names)
    feedback = [data for kind,data in events if kind=='actor-invalid-action']
    assert feedback and all(data['accepted'] is False for data in feedback)
    assert {data['operation_id'] for kind,data in events if kind=='actor-request'} == {feedback[0]['operation_id']}


def test_canonical_context_keeps_completed_small_skill_text_inline():
    from rsi.actor import WorkflowContextRecords
    from coding.pi.context import ContextRecords
    text = 'actual skill body\n' * 80
    read = dict(skill='mapped-skill', section=None, receipt_id='actual-read', content=text)
    context = {'host':dict(observation=dict(executed=True, content=text), read_skills=[read]*5)}
    default = ContextRecords().snapshot(context)['host']
    assert default['observation']['content']['read_with'] == 'read_context'
    canonical = WorkflowContextRecords().snapshot(context)['host']
    assert canonical['observation']['content'] == text
    assert canonical['current_skill_content']['mapped-skill'] == dict(receipt_id='actual-read',content=text)
    assert canonical['read_skills']['read_with'] == 'read_context'


def test_long_skill_delivery_requires_actual_forwarded_page_coverage():
    from coding.pi.context import ContextRecords
    from rsi.actor import SkillDelivery
    text = 'actual long contract\n' * 500
    pages, events = ContextRecords(), []
    reference = pages.admit(text)['record']
    delivery = SkillDelivery()
    delivery.begin('operation')
    delivery.expect('read-receipt', text)
    record = lambda kind,data:events.append((kind,data))
    delivery.observe({'messages':[{'content':json.dumps({'preview':text[:384]})}]}, record)
    assert delivery.missing(['read-receipt'])
    offset = 0
    while offset is not None:
        page = pages.read(reference, offset, 4096)
        delivery.observe({'messages':[{'role':'tool','content':json.dumps(page)}]}, record)
        offset = page['next_offset']
        assert bool(delivery.missing(['read-receipt'])) == (offset is not None)
    assert [kind for kind,_ in events] == ['skill-delivered']
    assert events[0][1]['receipt_id'] == 'read-receipt'


def test_pre_dispatch_stage_edit_error_is_recoverable_when_enabled(tmp_path):
    source = SourceSnapshot('solution.py','x=1','project-v1')
    actor, flow, session, called, _, _ = harness(tmp_path, reads(Stage.TASK_CONSTRUCTION)+[
        ('perform', {'proposal':'inline edit in wrong stage', 'edit':dict(path=source.path,sha256=source.sha256,source='x=2')}),
        ('perform', {'proposal':'construct without editing'})])
    actor.source_snapshot = lambda:source
    actor.recover_invalid_actions = True
    actor.step({})
    assert flow.stage == Stage.DESIGN and len(called) == 1 and 'edit' not in called[0][1]
    assert session.contexts[-1]['observation']['event'] == 'invalid-action'
    assert session.contexts[-1]['observation']['accepted'] is False


@pytest.mark.parametrize('recover', [False, True])
@pytest.mark.parametrize('error_kind', ['strategy', 'io', 'value', 'query', 'workflow'])
def test_evidence_tool_only_recovers_typed_strategy_rejection(tmp_path, recover, error_kind):
    from coding.pi.schema import obj
    from rsi.actor import EvidenceTool
    from rsi.strategy import QueryUnavailable, StrategyRejected
    errors = dict(strategy=StrategyRejected, io=OSError, value=ValueError,
                  query=QueryUnavailable, workflow=WorkflowError)
    error = errors[error_kind]('actual selection failure')
    actor, flow, session, called, clock, events = harness(tmp_path,
        reads(Stage.TASK_CONSTRUCTION) + [('select_strategy', {}),
                                         ('perform', {'proposal': 'construct'})])
    actor.recover_invalid_actions = recover
    attempts = []
    def select(stage, args, operation, seconds):
        attempts.append(operation)
        clock.value += 3
        raise error
    actor.evidence_tools = {'select_strategy': EvidenceTool(obj(), select)}
    actor.schemas['select_strategy'] = actor.evidence_tools['select_strategy'].schema
    if recover and error_kind == 'strategy':
        actor.step({})
        assert len(called) == len(flow.state['history']) == 1
        context = session.contexts[-1]
        observation = context['observation']
        assert observation['event'] == 'invalid-action'
        assert observation['action'] == 'select_strategy'
        assert observation['error'] == str(error)
        assert observation['accepted'] is False
        assert context['evidence'] == context['evidence_receipts'] == []
        assert {row['receipt_id'] for row in context['skill_receipts']} == set(
            flow.state['history'][0]['evidence']['skill_refs'])
        assert attempts == [flow.state['history'][0]['evidence']['operation_id']]
        assert flow.accounting.seconds('thinking') == 8
        assert flow.accounting.seconds('retrieval') == 3
        assert session.deadlines == [20, 18, 16, 11]
        assert len([data for kind, data in events if kind == 'actor-invalid-action']) == 1
    else:
        with pytest.raises(type(error)) as raised:
            actor.step({})
        assert raised.value is error
        assert not called and flow.pending and not flow.state['history']
        assert attempts == [flow.pending['id']]
        assert flow.accounting.seconds('thinking') == 6
        assert flow.accounting.seconds('retrieval') == 3
        assert not any(kind == 'actor-invalid-action' for kind, _ in events)
    assert not any(kind == 'evidence-tool-result' for kind, _ in events)
