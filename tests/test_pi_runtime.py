import hashlib
import json
import time

import pytest

from coding.pi.context import ContextRecords, ContextLimitError
from coding.pi.gateway import Gateway
from coding.pi.schema import STRING, obj
from coding.pi.session import PiCodingSession, RequestAccounting, runtime_preflight


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


def session(**kwargs):
    try:
        runtime_preflight()
    except RuntimeError:
        pytest.skip('npm ci --prefix integrations/pi required')
    return PiCodingSession(model='deepseek-flash', schemas={'write': obj({'source': STRING})},
                           environ={'DEEPSEEK_API_KEY': 'fake-provider-secret'},
                           deadline=kwargs.pop('deadline', time.monotonic()+30), **kwargs)


def test_real_pi_persistent_action_and_exact_edit(monkeypatch):
    captured = []
    def provider(*_, **kwargs):
        captured.append(json.loads(kwargs['data']))
        i = len(captured)-1
        return stream('write', {'source': 'old'}, i) if i == 0 else stream('edit_source', {
            'sha256': hashlib.sha256(b'old').hexdigest(), 'old_text': 'old', 'new_text': 'fixed'}, i)
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session() as pi:
        assert pi.next_action({'task': 'TEST'}, ['write']) == ('write', {'source': 'old'})
        process = pi.rpc.process.pid
        assert pi.next_action({'task': 'TEST', 'feedback': 'compile failure'}, ['write'], source='old',
                              write_action=lambda s: ('write', {'source': s})) == ('write', {'source': 'fixed'})
        assert pi.rpc.process.pid == process
        assert pi.completion_tokens == 20
        assert pi.gateway.calls == 2
        assert 'fake-provider-secret' not in json.dumps(captured)
        assert captured[0]['max_tokens'] == 8192
        assert captured[0]['thinking'] == {'type': 'disabled'}


def test_gateway_counts_tool_schema_bytes_before_http():
    gateway = Gateway('deepseek', 'fake', RequestAccounting(), 'test', hard_bytes=200)
    with pytest.raises(ValueError, match='hard byte'):
        gateway.prepare({'model': 'deepseek-flash', 'messages': [{'role':'user','content':'x'}],
                         'tools':[{'schema':'z'*300}]})
    assert gateway.calls == 0


def test_context_paging_multibyte():
    records = ContextRecords()
    ref = records.admit('é'*10000)
    data, offset = '', 0
    while offset is not None:
        page = records.read(ref['record'], offset, 8191)
        data += page['text']; offset = page['next_offset']
    assert data == 'é'*10000
    with pytest.raises(ValueError): records.read('unknown')
    with pytest.raises(ContextLimitError): records.snapshot({'a':'x'*100}, target=10)


@pytest.mark.parametrize('tool,args', [
    ('write', {'source': 123}), ('write', {'source': 'x', 'unexpected': True}),
    ('compile', {}), ('edit_source', {'sha256': 'stale', 'old_text': 'old', 'new_text': 'fixed'}),
])
def test_invalid_or_wrong_stage_tools_have_no_host_effect(monkeypatch, tool, args):
    from coding.pi.rpc import PiError
    monkeypatch.setattr('coding.pi.gateway.requests.post', lambda *a, **kw: stream(tool, args))
    with session(max_calls=1, deadline=time.monotonic()+4) as pi:
        with pytest.raises(PiError):
            pi.next_action({'task': 'TEST'}, ['write'])
        assert pi.pending is None
        assert pi.gateway.calls >= 1



def test_original_tool_validation_explains_rejection(monkeypatch):
    from coding.pi.session import ActionError
    recorded = []
    monkeypatch.setattr('coding.pi.gateway.requests.post', lambda *a, **kw: stream('write', {'source': 123}))
    with session() as pi:
        with pytest.raises(ActionError, match=r'arguments.source: expected string'):
            pi.next_action({'task': 'TEST'}, ['write'], record=lambda *args: recorded.append(args))
        assert pi.pending is None
        rejected = [payload for kind, payload in recorded if kind == 'pi-action-rejected']
        assert rejected and rejected[-1]['allowed'] == ['read_context', 'write']


def test_truncation_has_no_effect_and_same_process_can_repair(monkeypatch):
    from coding.pi.session import ActionError
    calls = []
    def provider(*_, **kwargs):
        calls.append(kwargs)
        return stream('write', {'source':'partial' if len(calls)==1 else 'fixed'}, len(calls),
                      finish='length' if len(calls)==1 else 'tool_calls')
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session(max_calls=2) as pi:
        with pytest.raises(ActionError, match='truncated'):
            pi.next_action({'task':'TEST'}, ['write'])
        pid = pi.rpc.process.pid
        assert pi.next_action({'task':'TEST', 'host':'Previous call truncated'}, ['write']) == ('write', {'source':'fixed'})
        assert pi.rpc.process.pid == pid
        assert pi.gateway.calls == 2


def test_real_pi_reads_admitted_large_guide_with_complete_exchange(monkeypatch):
    calls = []
    guide = 'Unique guidance. ' * 10000
    def provider(*_, **kwargs):
        request = json.loads(kwargs['data']); calls.append(request)
        if len(calls) == 1:
            snapshot = json.loads(next(m['content'] for m in request['messages'] if m['role']=='user'))
            ref = snapshot['context']['host']['guide']['record']
            return stream('read_context', {'record':ref,'offset':0,'limit':4096}, 0)
        call_ids = {call['id'] for m in request['messages'] for call in m.get('tool_calls', [])}
        results = {m['tool_call_id'] for m in request['messages'] if m['role']=='tool'}
        assert call_ids == results == {'call_0'}
        assert len(kwargs['data']) <= 49152
        return stream('write', {'source':'kernel using the admitted guide'}, 1)
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session(max_calls=2) as pi:
        name, args = pi.next_action({'task':'TEST', 'guide':guide}, ['write'])
        assert name == 'write' and pi.gateway.calls == 2
        assert pi.completion_tokens == 20



def test_real_pi_accumulates_three_pages_then_refreshes_host_turn(monkeypatch):
    calls, records = [], []
    guide = "a" * 2048 + "b" * 2048 + "c" * 1000
    reference = None
    def provider(*_, **kwargs):
        nonlocal reference
        request = json.loads(kwargs['data']); calls.append(request)
        messages = request['messages']
        snapshot = json.loads(next(m['content'] for m in messages if m['role'] == 'user'))
        pages = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
        index = len(calls) - 1
        if index < 3:
            reference = snapshot['context']['host']['guide']['record']
            assert ''.join(page['text'] for page in pages) == guide[:index * 2048]
            return stream('read_context', {'record': reference, 'offset': index * 2048,
                                          'limit': 2048}, index)
        if index == 3:
            assert ''.join(page['text'] for page in pages) == guide
            return stream('write', {'source': ''.join(page['text'][0] for page in pages)}, index)
        assert index == 4  # A queued host action must not trigger another provider request.
        assert snapshot['context']['host'] == {'task': 'NEW', 'feedback': 'current'}
        assert not pages and not any(m.get('tool_calls') for m in messages)
        return stream('write', {'source': 'new'}, index)
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session() as pi:
        assert pi.next_action({'task': 'OLD', 'guide': guide}, ['write'],
                              record=lambda *args: records.append(args)) == ('write', {'source': 'abc'})
        assert len(calls) == 4
        assert pi.next_action({'task': 'NEW', 'feedback': 'current'}, ['write']) == ('write', {'source': 'new'})
        assert len(calls) == 5
    reads = [payload for kind, payload in records if kind == 'pi-context-read']
    assert [row['offset'] for row in reads] == [0, 2048, 4096]
    assert not any(row['repeat'] for row in reads)



def test_compiler_errors_are_visible_and_full_diagnostic_is_readable():
    records = ContextRecords()
    diagnostic = ('warning: header warning\n' * 300 +
                  'candidate.cpp:17:4: error: undeclared identifier x\n')
    snapshot = records.snapshot({'diagnostic': diagnostic})
    assert snapshot['diagnostic']['errors'].splitlines() == ['candidate.cpp:17:4: error: undeclared identifier x']
    ref = snapshot['diagnostic']['full']
    assert records.read(ref['record'], 0, 8192)['text'] == diagnostic



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


def test_context_read_telemetry_and_actionable_validation():
    records = []
    with session() as pi:
        pi.record = lambda *args: records.append(args)
        ref = pi.context.admit('data')
        pi.allowed = {'write', 'read_context'}
        pi._operation('read_context', {'record': ref['record']})
        pi._operation('read_context', {'record': ref['record']})
        assert [payload['repeat'] for _, payload in records] == [False, True]
        with pytest.raises(ValueError, match=r'arguments.source: expected string'):
            pi._operation('write', {'source': 1})
        assert pi.pending is None and pi.mutations == 0


def test_real_pi_retains_instruction_pages_and_source_until_next_host_turn(monkeypatch):
    instruction = 'Instruction body. ' * 600
    source = 'kernel source;\n' * 160
    requests, refs = [], {}
    def pages(request):
        return [json.loads(message['content']) for message in request['messages'] if message['role'] == 'tool']
    def provider(*_, **kwargs):
        request = json.loads(kwargs['data'])
        index = len(requests)
        requests.append(request)
        assert len(kwargs['data']) <= 49152
        if index == 0:
            packet = json.loads(next(m['content'] for m in request['messages'] if m['role'] == 'user'))
            refs.update({key: packet['context']['host'][key]['record'] for key in ['instruction', 'source']})
        reads = [(refs.get('instruction'), 0), (refs.get('instruction'), 8192), (refs.get('source'), 0)]
        if index < 3:
            record, offset = reads[index]
            return stream('read_context', dict(record=record, offset=offset, limit=8192), index)
        if index in (3, 4):
            retained = pages(request)
            assert len(retained) == 3
            assert {(row['record'], row['offset']) for row in retained} == set(reads)
            assert ''.join(row['text'] for row in sorted(retained, key=lambda row: row['offset'])
                           if row['record'] == refs['instruction']) == instruction
            assert next(row['text'] for row in retained if row['record'] == refs['source']) == source
            calls = {call['id'] for m in request['messages'] for call in m.get('tool_calls', [])}
            assert calls == {m['tool_call_id'] for m in request['messages'] if m['role'] == 'tool'}
            if index == 3:
                return stream('read_context', dict(limit=8192, offset=0, record=refs['instruction']), index)
        if index == 5:
            assert not pages(request)
        return stream('write', {'source': 'kernel'}, index)
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session() as pi:
        assert pi.next_action({'instruction': instruction, 'source': source}, ['write']) == ('write', {'source': 'kernel'})
        assert pi.next_action({'task': 'New authoritative task'}, ['write']) == ('write', {'source': 'kernel'})
    assert len(requests) == 6


def test_real_pi_many_context_pages_keep_requests_bounded_and_exchanges_complete(monkeypatch):
    requests = []
    guide = 'Guide page contents. ' * 10000
    def provider(*_, **kwargs):
        request = json.loads(kwargs['data'])
        index = len(requests)
        requests.append(request)
        assert len(kwargs['data']) <= 49152
        packet = json.loads(next(m['content'] for m in request['messages'] if m['role'] == 'user'))
        calls = {call['id'] for m in request['messages'] for call in m.get('tool_calls', [])}
        results = [m for m in request['messages'] if m['role'] == 'tool']
        assert calls == {m['tool_call_id'] for m in results}
        if index:
            assert any(json.loads(m['content'])['offset'] == (index-1)*4096 for m in results)
        if index == 10:
            assert 1 < len(results) < 10
            return stream('write', {'source': 'bounded kernel'}, index)
        return stream('read_context', {'record': packet['context']['host']['guide']['record'],
                                      'offset': index*4096, 'limit': 4096}, index)
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session(request_bytes=49152) as pi:
        assert pi.next_action({'guide': guide}, ['write']) == ('write', {'source': 'bounded kernel'})
    assert len(requests) == 11


def test_credential_in_context_is_rejected_before_provider(monkeypatch):
    monkeypatch.setattr('coding.pi.gateway.requests.post', lambda *a, **kw: pytest.fail('paid request'))
    with session() as pi:
        with pytest.raises(ValueError, match='credential'):
            pi.next_action({'guide':'fake-provider-secret'}, ['write'])
        assert pi.rpc is None and pi.gateway.calls == 0


def test_gateway_refuses_full_oversized_request_even_if_pi_hook_accepts(monkeypatch):
    from coding.pi.rpc import PiError
    monkeypatch.setattr('coding.pi.gateway.requests.post', lambda *a, **kw: pytest.fail('upstream HTTP'))
    with session() as pi:
        pi.gateway.hard_bytes = 100
        with pytest.raises(PiError, match='context exceeds'):
            pi.next_action({'task':'TEST'}, ['write'])
        assert pi.gateway.calls == 0


def test_multiple_mutations_reject_whole_batch(monkeypatch):
    from coding.pi.rpc import PiError
    def provider(*_, **kwargs):
        response = stream('write', {'source':'first'})
        original = response.iter_lines
        def lines():
            for line in original():
                if line.startswith(b'data: {'):
                    body = json.loads(line[6:])
                    delta = body['choices'][0]['delta']
                    if delta.get('tool_calls'):
                        second = json.loads(json.dumps(delta['tool_calls'][0]))
                        second.update(index=1, id='second')
                        second['function']['arguments'] = json.dumps({'source':'second'})
                        delta['tool_calls'].append(second)
                        line = b'data: ' + json.dumps(body).encode()
                yield line
        response.iter_lines = lines
        return response
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session(max_calls=1, deadline=time.monotonic()+4) as pi:
        with pytest.raises(PiError): pi.next_action({'task':'TEST'}, ['write'])
        assert pi.pending is None


def test_openai_responses_uses_registered_profile_and_counts_subsets(monkeypatch):
    try: runtime_preflight()
    except RuntimeError: pytest.skip('Pi runtime unavailable')
    captured = []
    def provider(*_, **kwargs):
        captured.append(json.loads(kwargs['data'])); return responses_stream('write', {'source':'kernel'})
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with PiCodingSession(model='openai/gpt-5.6-sol', schemas={'write':obj({'source':STRING})},
                         environ={'OPENAI_API_KEY':'fake'}, deadline=time.monotonic()+30) as pi:
        assert pi.next_action({'task':'TEST'}, ['write']) == ('write', {'source':'kernel'})
        assert pi.completion_tokens == 40
        assert captured[0]['max_output_tokens'] == 32768
        from benchmarks.pi_coding.report import usage_totals
        usage = usage_totals(pi.gateway.records)
        assert usage['total_tokens'] == 140
        assert usage['cached_input_tokens'] == 20 and usage['reasoning_tokens'] == 30


def test_usage_ledger_does_not_stop_at_legacy_dollar_limits(tmp_path):
    from coding.pi.budget import Budget
    ledger = Budget(tmp_path/'usage.json', .00001, window='run', window_limit=.000001)
    first = ledger.reserve({}, output_tokens=100, rates=(1, 1), label='first')
    ledger.reserve({}, output_tokens=100, rates=(1, 1), label='second')
    ledger.settle(first, {'prompt_tokens': 100000, 'completion_tokens': 100000}, (1, 1))
    assert len(ledger.snapshot()['requests']) == 2
    assert ledger.snapshot()['requests'][0]['charged_bound_usd'] == .2


def test_requests_and_cumulative_tokens_do_not_stop_before_deadline(monkeypatch):
    monkeypatch.setattr('coding.pi.gateway.requests.post', lambda *a, **kw: stream('write', {'source': 'kernel'}))
    class LegacyBudget:
        def reserve(self, *args, **kwargs): pytest.fail('legacy dollar budget used')
    with session(max_calls=0, budget=LegacyBudget()) as pi:
        pi.gateway.completion_limit = 0  # Legacy attributes cannot cap a run.
        for _ in range(3):
            assert pi.next_action({'task': 'TEST'}, ['write'], limits={'provider_requests': 0}) == ('write', {'source': 'kernel'})
        assert pi.gateway.calls == 3 and pi.completion_tokens == 30


def test_expired_deadline_stops_before_starting_pi(monkeypatch):
    from coding.pi.rpc import PiDeadlineError
    monkeypatch.setattr('coding.pi.gateway.requests.post', lambda *a, **kw: pytest.fail('upstream HTTP'))
    with session() as pi:
        pi.deadline = time.monotonic() - 1
        with pytest.raises(PiDeadlineError):
            pi.next_action({'task': 'TEST'}, ['write'])
        assert pi.rpc is None and pi.gateway.calls == 0


def test_action_completed_after_deadline_is_not_released(monkeypatch):
    from coding.pi.rpc import PiDeadlineError
    def provider(*args, **kwargs):
        pi.deadline = time.monotonic() - 1
        return stream('write', {'source': 'late kernel'})
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session() as pi:
        with pytest.raises(PiDeadlineError):
            pi.next_action({'task': 'TEST'}, ['write'])
        assert pi.pending is None


def test_parallel_pi_sessions_keep_contexts_and_processes_isolated(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    def provider(*_, **kwargs):
        payload = json.loads(kwargs['data'])
        snapshot = json.loads(next(m['content'] for m in payload['messages'] if m['role']=='user'))
        return stream('write', {'source': snapshot['context']['host']['task']})
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    def run(task):
        with session(max_calls=1, deadline=time.monotonic()+4) as pi:
            result = pi.next_action({'task':task}, ['write'])
            return result, pi.rpc.process.pid
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(run, ['task-a','task-b']))
    assert a[0] == ('write', {'source':'task-a'})
    assert b[0] == ('write', {'source':'task-b'})
    assert a[1] != b[1]


@pytest.mark.parametrize('program', ["print('not-json', flush=True)", 'pass'])
def test_rpc_malformed_or_early_eof_closes_child(tmp_path, program):
    import os
    import sys
    from coding.pi.rpc import PiRPC, PiError
    rpc = PiRPC([sys.executable, '-c', program], cwd=tmp_path, env={'PATH':os.defpath}, timeout=2)
    try:
        with pytest.raises(PiError): rpc.prompt('test')
    finally:
        rpc.close()
    assert rpc.process.poll() is not None


def test_missing_runtime_fails_before_credentials_or_provider(monkeypatch):
    from coding.pi.session import RuntimeUnavailable
    monkeypatch.setattr('coding.pi.session.shutil.which', lambda _: None)
    monkeypatch.setattr('coding.pi.gateway.requests.post', lambda *a, **kw: pytest.fail('provider called'))
    with pytest.raises(RuntimeUnavailable, match='Node'):
        PiCodingSession(model='deepseek-flash', schemas={})


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



def test_real_pi_actor_waits_for_submit_acceptance(monkeypatch, tmp_path):
    state = _canonical_kernel_case(tmp_path, monkeypatch, real=True, reject=True).run()
    assert state.outcome.status=='host-verified'
    history=state.runner.workflow.state['history']
    assert history[-1]['evidence']['stage']=='acceptance' and history[-1]['passed']
    assert any(h['evidence']['stage']=='validation' and h['passed'] for h in history)
    assert state.runner.backend.host.actions.actions.count('submit')==1
    responses=[value for role,value in _native_wire_objects(state.wires) if role=='tool']
    assert any(r.get('isError') and r.get('observation',{}).get('executed') is False for r in responses)



def test_actor_invalid_action_feedback_reaches_next_request(monkeypatch, tmp_path):
    state = _canonical_kernel_case(tmp_path, monkeypatch, real=True, reject=True).run()
    assert state.outcome.status=='host-verified'
    denied=[value for role,value in _native_wire_objects(state.wires) if role=='tool']
    denied=[r for r in denied if r.get('isError') and r.get('observation',{}).get('executed') is False]
    assert denied and 'perform' not in denied[0]['state']['allowed']
    assert 'read_skill' in denied[0]['state']['allowed']
    assert any(e['type']=='skill-delivered' for e in state.runner.events)



def test_real_pi_escaped_snapshot_trims_older_pages_under_gateway_request_cap(monkeypatch):
    requests = []
    context = {'guide': 'Page contents. ' * 2000,
               'quoted': {f'field-{index}': '\\"' * 240 for index in range(16)}}
    def provider(*_, **kwargs):
        request = json.loads(kwargs['data'])
        requests.append(request)
        assert len(kwargs['data']) <= 49152
        snapshot = json.loads(next(row['content'] for row in request['messages'] if row['role'] == 'user'))
        assert snapshot['context']['host']['quoted'] == context['quoted']
        index = len(requests) - 1
        retained = [json.loads(row['content']) for row in request['messages'] if row['role'] == 'tool']
        if index:
            assert any(row['offset'] == (index - 1) * 7400 for row in retained)
        if index == 2:
            assert [row['offset'] for row in retained] == [7400]
            calls = {call['id'] for row in request['messages'] for call in row.get('tool_calls', [])}
            assert calls == {row['tool_call_id'] for row in request['messages'] if row['role'] == 'tool'}
            return stream('write', {'source': 'selected after bounded paging'}, index)
        return stream('read_context', dict(record=snapshot['context']['host']['guide']['record'],
                                          offset=index * 7400, limit=7400), index)
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session(request_bytes=49152) as pi:
        assert pi.next_action(context, ['write']) == ('write', {'source': 'selected after bounded paging'})
    assert len(requests) == 3


def test_real_pi_openai_large_snapshot_retains_multiple_pages_within_configured_cap(monkeypatch):
    try: runtime_preflight()
    except RuntimeError: pytest.skip('Pi runtime unavailable')
    requests = []
    context = {'guide': 'paged-content-' * 2000,
               'fields': {f'field-{index}': 'retained-' * 100 for index in range(64)}}
    def provider(*_, **kwargs):
        request = json.loads(kwargs['data'])
        requests.append(request)
        assert len(kwargs['data']) <= 196608
        user = next(row for row in request['input'] if row.get('role') == 'user')
        text = user['content'] if isinstance(user['content'], str) else user['content'][0]['text']
        snapshot = json.loads(text)
        assert len(text.encode()) > 49152
        assert snapshot['context']['host']['fields'] == context['fields']
        index = len(requests) - 1
        retained = [json.loads(row['output']) for row in request['input']
                    if row.get('type') == 'function_call_output']
        assert [row['offset'] for row in retained] == [offset * 4096 for offset in range(index)]
        calls = {row['call_id'] for row in request['input'] if row.get('type') == 'function_call'}
        assert calls == {row['call_id'] for row in request['input']
                         if row.get('type') == 'function_call_output'}
        if index == 2:
            return responses_stream('write', {'source': 'selected with both pages'}, index)
        return responses_stream('read_context', dict(record=snapshot['context']['host']['guide']['record'],
                                offset=index * 4096, limit=4096), index)
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with PiCodingSession(model='openai/gpt-5.6-sol', schemas={'write': obj({'source': STRING})},
                         environ={'OPENAI_API_KEY': 'fake-provider-secret'}, deadline=time.monotonic()+30) as pi:
        assert pi.next_action(context, ['write']) == ('write', {'source': 'selected with both pages'})
    assert len(requests) == 3


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


@pytest.mark.parametrize('kind',['callback','missing','deadline','layout-only'])
def test_incomplete_callback_requires_explicit_canonical_authority(monkeypatch,tmp_path,kind):
    from types import SimpleNamespace
    import coding.pi.actor as module
    import rsi.domain_bindings as binding
    from rsi.domain_execution import CanonicalConfigurationError
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider
    monkeypatch.setattr(module,'runtime_preflight',lambda:None)
    calls=[]
    def callback(action):
        calls.append(action)
        pytest.fail('inadmissible callback executed')
    if kind=='layout-only':callback.prepared=SimpleNamespace()
    executor={'callback':callback,'missing':None,'deadline':SimpleNamespace(deadline=9),'layout-only':callback}[kind]
    monkeypatch.setattr(binding,'PiSessionAdapter',lambda **_:pytest.fail('stage session created before admission'))
    provider=DirectCompletionProvider({'DEEPSEEK_API_KEY':'fake','OPENAI_API_KEY':'fake-openai'})
    monkeypatch.setattr(provider,'complete',lambda *a,**kw:pytest.fail('provider invoked before admission'))
    driver=module.PiActorDriver(provider)
    with pytest.raises(CanonicalConfigurationError):
        driver('task',profile=None,workspace=tmp_path,context=None,turn_parser=None,action_executor=executor)
    assert calls==[] and not (tmp_path/'canonical/domain.json').exists()
