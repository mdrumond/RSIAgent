"""Canonical host contracts and fixture wire helpers; SDK behavior lives in test_native_pi_*."""
import hashlib
import json
import time
import pytest

def stream(tool, args, index=0, finish='tool_calls'):
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def iter_lines(self):
            delta = {'role': 'assistant', 'tool_calls': [{'index': 0, 'id': f'call_{index}',
                     'type': 'function', 'function': {'name': tool, 'arguments': json.dumps(args)}}]}
            for chunk in [{'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
                          {'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}],
                           'usage': {'prompt_tokens': 50, 'completion_tokens': 10}}]:
                chunk.update(id=f'test_{index}', object='chat.completion.chunk', created=0, model='deepseek-flash')
                yield b'data: ' + json.dumps(chunk).encode()
                yield b''
            yield b'data: [DONE]'
            yield b''
    return Response()


def responses_stream(tool, args, index=0):
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def iter_lines(self):
            reasoning = {'type':'reasoning','id':f'rs_{index}','summary':[], 'encrypted_content':'opaque'}
            call = {'type':'function_call', 'id':f'fc_{index}', 'call_id':f'call_{index}','name':tool,
                    'arguments':json.dumps(args), 'status':'completed'}
            events = [{'type':'response.created','response':{'id':f'resp_{index}'}}]
            for i, item in enumerate([reasoning, call]):
                events += [{'type':'response.output_item.added','output_index':i,'item':item},
                           {'type':'response.output_item.done','output_index':i,'item':item}]
            events.append({'type':'response.completed','response':{'id':f'resp_{index}','status':'completed',
                'output':[reasoning,call],'usage':{'input_tokens':100,'output_tokens':40,
                'input_tokens_details':{'cached_tokens':20},'output_tokens_details':{'reasoning_tokens':30}}}})
            for event in events:
                yield b'data: '+json.dumps(event).encode(); yield b''
    return Response()


@pytest.mark.parametrize('dtype_name', ['MXFP8_E4M3FN', 'MXFP8_E5M2'])
def test_task_contract_preserves_real_mxfp8_auxiliary_scale_abi(tmp_path, dtype_name):
    from dataclasses import replace
    from types import SimpleNamespace
    from coding.pi.actor import task_contract
    from benchmarks.a5kernels.phase1_curriculum import DType
    from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor
    from benchmarks.a5kernels.evidence import EvidenceLedger
    from tests.test_a5_phase1_project_bz import FakeExecutor, _backend
    from tests.test_a5_phase1_project_executor import _spec
    project = ProjectExecutor(FakeExecutor([]), EvidenceLedger(tmp_path / 'ledger'))
    prepared = project.prepare(replace(_spec('cube'), dtype=getattr(DType, dtype_name)))
    contract = task_contract(SimpleNamespace(prepared=prepared))
    assert len(contract['inputs']) == len(prepared.inputs) == 2
    assert [item['argument'] for item in contract['auxiliary_inputs']] == list(prepared.abi.input_arguments[2:])
    _, files = _backend(tmp_path, FakeExecutor([]))._bundle(project.request(prepared, '# candidate'))
    manifest = json.loads(next(item.content for item in files if item.relative_path == 'manifest.json'))
    assert [item['physical_shape'] for item in contract['auxiliary_inputs']] == [item['shape'] for item in manifest['scales']]
    assert contract['auxiliary_inputs'][0]['logical_shape'] == manifest['scales'][0]['shape']
    assert contract['auxiliary_inputs'][1]['logical_shape'] == list(reversed(manifest['scales'][1]['shape']))
    assert all(item['dtype'] == 'E8M0' and item['group_size'] == 32 for item in contract['auxiliary_inputs'])
    assert not any('storage' in item for item in contract['inputs'] + contract['auxiliary_inputs'])


def test_usage_ledger_does_not_stop_at_legacy_dollar_limits(tmp_path):
    from coding.pi.budget import Budget
    ledger = Budget(tmp_path/'usage.json', .00001, window='run', window_limit=.000001)
    first = ledger.reserve({}, output_tokens=100, rates=(1, 1), label='first')
    ledger.reserve({}, output_tokens=100, rates=(1, 1), label='second')
    ledger.settle(first, {'prompt_tokens': 100000, 'completion_tokens': 100000}, (1, 1))
    assert len(ledger.snapshot()['requests']) == 2
    assert ledger.snapshot()['requests'][0]['charged_bound_usd'] == .2


def test_actor_compile_and_host_turns_are_bounded_by_time_only(monkeypatch, tmp_path):
    state = _canonical_kernel_case(tmp_path, monkeypatch).run()
    assert state.outcome.status=='host-verified'
    assert state.calls.count('compile')>1 and state.calls.count('run')>1
    assert len(state.contexts)>1  # max_turns=1 is retired telemetry, not an elapsed limit.
    assert state.runner.workflow.accounting.seconds()>0
    assert state.runner.workflow.remaining()['actor_seconds']>0


def test_actor_expires_without_executing_another_action(monkeypatch, tmp_path):
    state = _canonical_kernel_case(tmp_path, monkeypatch, expire=True).run()
    assert state.outcome.status=='budget_exhausted' and state.calls==[]
    assert state.runner.workflow.pending['stage']=='task_construction'
    assert state.runner.workflow.accounting.seconds()>=1
    assert not any(e['type']=='host-dispatch' for e in state.runner.events)


def test_actor_inherits_executor_deadline_and_preserves_clock_on_clone(monkeypatch, tmp_path):
    # Explicit canonical parent wall allocation, not an old raw actor deadline.
    state = _canonical_kernel_case(tmp_path, monkeypatch, parent_expired=True)
    cloned = state.driver.with_trace(lambda *_:None)
    assert cloned.clock is state.driver.clock
    assert cloned.canonical_budgets == state.driver.canonical_budgets
    state.run(cloned)
    assert state.outcome.status=='budget_exhausted' and state.calls==[]
    assert state.contexts==[] and state.runner.workflow.accounting.seconds()==0


def test_actor_reports_rejected_action_for_next_turn_recovery(monkeypatch, tmp_path):
    state = _canonical_kernel_case(tmp_path, monkeypatch, reject=True).run()
    assert state.outcome.status=='host-verified'
    rejected=[c for c in state.contexts if c.get('observation',{} ) and c['observation'].get('event')=='invalid-action']
    assert rejected and rejected[0]['observation']['accepted'] is False
    assert any(e['type']=='actor-invalid-action' for e in state.runner.events)


def _native_wire_objects(wires):
    for wire in wires:
        for message in wire['messages']:
            content=message.get('content') or []
            texts=[content] if isinstance(content,str) else [part.get('text','') for part in content]
            for text in texts:
                if text.startswith('{'):
                    yield message['role'],json.loads(text)


def _canonical_kernel_case(tmp_path, monkeypatch, *, real=False, shape=None,
                           reject=False, expire=False, parent_expired=False):
    """Real canonical host/SQL evidence; scripted CPU decisions, never a device."""
    from dataclasses import replace
    from types import SimpleNamespace
    from hashlib import sha256
    from rsi.contracts import Stage, BudgetLimits
    from rsi.domain_execution import DomainRunner
    from rsi.actor import PiSessionAdapter
    from coding.pi.session import ActionError
    import rsi.domain_bindings as binding
    from rsi.strategy import knowledge_db_query
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
    from benchmarks.a5kernels.knowledge import KnowledgeDB
    from benchmarks.a5kernels.evidence import EvidenceLedger, canonical_digest
    from benchmarks.a5kernels.phase1_project import ProjectGraph
    from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
    from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor, CompileObservation, RuntimeObservation
    from benchmarks.a5kernels.phase1_adaptive_actor import _ProjectActions, AdaptiveActionResult
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider, direct_profile
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.trial import parse_trial_action
    from coding.pi.actor import PiActorDriver
    from test_a5_phase1_adaptive_actor import _spec, _request
    from test_rsi_evidence_io import FixtureEmbeddings
    from test_rsi_production_domain_bindings import CanonicalSession
    from test_rsi_pi_controller import response
    clock = SimpleNamespace(value=time.time())
    clock.now = lambda: clock.value
    state = SimpleNamespace(calls=[], contexts=[], wires=[], runner=None)
    class Backend:
        def compile(self, request):
            state.calls.append('compile'); clock.value += .05
            compile(request.candidate_source, 'cpu-candidate.py', 'exec')
            return CompileObservation(True, canonical_digest({'source':request.candidate_source}), 'CPU Python compiled', 'CPU')
        def run(self, request, compilation):
            state.calls.append('run'); clock.value += .05
            namespace = {}
            exec(compile(request.candidate_source, 'cpu-candidate.py', 'exec'), namespace)
            observed = tuple(namespace['solve'](list(request.inputs[0].storage)))
            return RuntimeObservation(True, canonical_digest({'observed':observed}), 'CPU numeric result', observed)
    executor = ProjectExecutor(Backend(), EvidenceLedger(tmp_path/'host.jsonl'))
    spec = _spec()
    if shape is not None:
        graph = ProjectGraph.from_mapping(dict(inputs=[dict(name='x',shape=list(shape))],
            nodes=[dict(name='out',op='relu',operands=['x'],axis=None)],output='out'))
        spec = ProjectSpec.from_mapping(dict(spec.as_dict(), shape=list(shape), graph=graph.as_dict(), padding_multiple=32))
    request = _request(tmp_path, executor, spec)
    artifacts = ContentAddressedArtifacts(request.branch_root/'artifacts')
    seed = tmp_path/'seed.py'; seed.write_text('def stable_relu(x): return max(0, x)\n')
    database = KnowledgeDB(tmp_path/'knowledge.sqlite3', FixtureEmbeddings())
    manifest = database.index(tmp_path,[seed],collection='cpu-baseline',language='python')
    query = knowledge_db_query(database, ('cpu-baseline',), negative_query=lambda _: [
        row for row in state.runner.workflow.state['history'] if row.get('passed') is False])
    def knowledge(payload, context):
        return AdaptiveActionResult(query(payload))
    knowledge.canonical_identity = dict(manifest=json.loads(manifest.to_json()))
    actions = _ProjectActions(executor,request,executor.prepare(request.spec),artifacts,
        max_actions=1,max_seconds=120,max_source_bytes=128000,query_knowledge=knowledge,
        request_profile=lambda *_:None,request_timing=lambda *_:None)
    original = DomainRunner.create.__func__
    def create(cls,*args,**kwargs):
        runner = original(cls,*args,**dict(kwargs,test_backend=True,clock=clock.now))
        state.runner = runner
        if parent_expired: clock.value += 2
        return runner
    monkeypatch.setattr(DomainRunner,'create',classmethod(create))
    width = request.inputs[0].physical_shape[-1]; logical = request.inputs[0].shape[-1]
    good = f'def solve(xs):\n    return [max(0,x) if i % {width} < {logical} else 0 for i,x in enumerate(xs)]\n'
    class Model(CanonicalSession):
        rejected = False
        def next_action(self, context, allowed, **kwargs):
            state.contexts.append(context); clock.value += .01
            if reject and not self.rejected:
                self.rejected = True
                raise ActionError('CPU fixture rejected malformed model action')
            action = super().next_action(context,allowed,**kwargs)
            if expire and action[0]=='perform': clock.value += 2
            return action
    model = Model(initial='def solve(xs):\n    return list(xs)\n', repaired=good)
    # Explicit factory boundary substitutes decisions only; all stages/authority
    # and numeric verification remain the real default native production binding.
    if real:
        def payload():
            runner=state.runner; stage=runner.workflow.stage; op=runner.workflow.pending['id']
            reads=[e for e in runner.events if e['type']=='skill-usage' and e.get('kind')=='read' and e['operation_id']==op]
            names=runner.bundle.skills_for(stage); read_names={e['skill'] for e in reads}
            source=runner.snapshot()
            evidence=[dict(name=e['name'],result=dict(reference=e['reference'],evidence=runner.store.read_json(e['reference'])))
                for e in runner.events if e['type']=='project-tool' and e['operation_id']==op]
            return dict(stage=stage.value,unread_skills=[n for n in names if n not in read_names],evidence=evidence,
                read_skills=[dict(receipt_id=runner.service.read_receipt(e['receipt_id'])) for e in reads],
                source=dict(path=source.path,text=source.text,sha256=source.sha256,project_hash=source.project_hash))
        def provider(*_,**kwargs):
            wire=json.loads(kwargs['data']); state.wires.append(wire)
            if not wire.get('tools'):
                return response([],len(state.wires))
            # One genuine SDK-level rejection before reads. No host completion.
            if reject and len(state.wires)==1:
                model.rejected=True
                return response([('read_context',{'record':'0'*64})],len(state.wires))
            name,args=model.next_action(payload(),[],seconds=60,record=lambda *_:None,source=None)
            return response([(name,args)],len(state.wires))
        monkeypatch.setattr('coding.pi.gateway.requests.post',provider)
    else:
        monkeypatch.setattr(binding,'PiSessionAdapter',lambda **_:model)
        monkeypatch.setattr('coding.pi.actor.runtime_preflight',lambda:None)
    profile = direct_profile(next(m for m in MODEL_IDENTITIES if m.backend_model.value=='deepseek-flash'))
    limits=BudgetLimits(actor_seconds=1 if expire else 120,wall_seconds=1 if parent_expired else None)
    driver = PiActorDriver(DirectCompletionProvider({'DEEPSEEK_API_KEY':'fake-provider-secret','OPENAI_API_KEY':'fake-openai-secret'}),
        max_turns=1,max_seconds=120,clock=clock.now,canonical_budgets=limits,system_prompt='Use the frozen CPU host contract.')
    def run(active=None):
        try:
            state.outcome=(active or driver)('CPU ReLU with zero padded tail',profile=profile,workspace=tmp_path/'actor',
                context=None,turn_parser=parse_trial_action,action_executor=actions)
            return state
        finally:
            database.close()
    state.run,state.driver,state.clock = run,driver,clock
    return state



def test_actor_provider_receives_layout_contract_without_context_reads(monkeypatch, tmp_path):
    state = _canonical_kernel_case(tmp_path, monkeypatch, real=True, shape=(2,23)).run()
    assert state.outcome.status == 'host-verified'
    first = next(value for role,value in _native_wire_objects(state.wires[:1])
                 if role=='user' and 'task' in value)
    contract = first['task']['domain']['kernel_contract']
    assert contract['shape']==[2,23] and contract['dtype']=='float32'
    assert contract['block_count']==1 and contract['padding_multiple']==32
    assert contract['abi']['input_arguments']==['gm_input0']
    assert contract['inputs']==[dict(argument='gm_input0',logical_shape=[2,23],physical_shape=[2,32])]
    assert contract['output']['logical_shape']==[2,23]
    assert contract['output']['physical_shape']==[2,32]
    assert 'storage' not in contract['inputs'][0] and 'values' not in contract['output']
    assert not any(e['type']=='pi-context-read' for e in state.runner.events
                   if e.get('stage')=='task_construction')

