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
        assert len(kwargs['data']) <= 196608
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
        assert len(pages) == 4  # Pi retains history; only the injected snapshot changes.
        return stream('write', {'source': 'new'}, index)
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session() as pi:
        assert pi.next_action({'task': 'OLD', 'guide': guide}, ['write'],
                              record=lambda *args: records.append(args)) == ('write', {'source': 'abc'})
        assert len(calls) == 4
        assert pi.next_action({'task': 'NEW', 'feedback': 'current'}, ['write']) == ('write', {'source': 'new'})
        assert pi.context.read(reference, 0, 2048)['text'] == guide[:2048]
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
    try: runtime_preflight()
    except RuntimeError: pytest.skip('Pi runtime unavailable')
    from types import SimpleNamespace
    from coding.pi.actor import PiActorDriver
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider, direct_profile
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_project_executor import ProjectKernelABI
    from benchmarks.a5kernels.trial import ActionOutcome, parse_trial_action
    seen = []
    def provider(*_, **kwargs):
        request = json.loads(kwargs['data']); seen.append(request)
        snapshot = json.loads(next(m['content'] for m in request['messages'] if m['role'] == 'user'))
        contract = snapshot['context']['host']['task_contract']
        assert contract['shape'] == [2, 23] and contract['dtype'] == 'float32'
        assert contract['block_count'] == 1 and contract['padding_multiple'] == 32
        assert contract['abi']['input_arguments'] == ['gm_input0']
        assert contract['inputs'] == [{'argument': 'gm_input0', 'logical_shape': [2, 23], 'physical_shape': [2, 32]}]
        assert contract['output']['logical_shape'] == [2, 23]
        assert contract['output']['physical_shape'] == [2, 32]
        return stream('write', {'slot': 'kernel', 'content': 'candidate'})
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    def execute(action):
        assert action.kind == 'write'
        return ActionOutcome('{}', terminal=True)
    execute.prepared = SimpleNamespace(
        spec=SimpleNamespace(shape=(2, 23), dtype=SimpleNamespace(value='float32'),
            padding_multiple=32, block_count=1, graph=SimpleNamespace(as_dict=lambda: {'output': 'y'})),
        abi=ProjectKernelABI('project_kernel', ('gm_input0',), 'gm_output'),
        inputs=(SimpleNamespace(shape=(2, 23), physical_shape=(2, 32), storage='private inputs'),),
        expected=SimpleNamespace(shape=(2, 23), physical_shape=(2, 32), dtype='float32', values='private oracle'))
    model = next(item for item in MODEL_IDENTITIES if item.backend_model.value == 'deepseek-flash')
    driver = PiActorDriver(DirectCompletionProvider({'DEEPSEEK_API_KEY': 'fake-provider-secret', 'OPENAI_API_KEY': 'fake-openai-secret'}), max_seconds=10)
    driver('Implement the supplied layout.', profile=direct_profile(model), workspace=tmp_path,
           context=None, turn_parser=parse_trial_action, action_executor=execute)
    assert len(seen) == 1
    assert 'private inputs' not in json.dumps(seen) and 'private oracle' not in json.dumps(seen)



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
        assert len(kwargs['data']) <= 196608
        if index == 0:
            packet = json.loads(next(m['content'] for m in request['messages'] if m['role'] == 'user'))
            refs.update({key: packet['context']['host'][key]['record'] for key in ['instruction', 'source']})
        reads = [(refs.get('instruction'), 0), (refs.get('instruction'), 8192), (refs.get('source'), 0)]
        if index < 3:
            record, offset = reads[index]
            return stream('read_context', dict(record=record, offset=offset, limit=8192), index)
        if index in (3, 4):
            retained = pages(request)
            assert len(retained) == (3 if index == 3 else 4)
            assert {(row['record'], row['offset']) for row in retained} == set(reads)
            assert ''.join(row['text'] for row in sorted(retained[:3], key=lambda row: row['offset'])
                           if row['record'] == refs['instruction']) == instruction
            assert next(row['text'] for row in retained if row['record'] == refs['source']) == source
            calls = {call['id'] for m in request['messages'] for call in m.get('tool_calls', [])}
            assert calls == {m['tool_call_id'] for m in request['messages'] if m['role'] == 'tool'}
            if index == 3:
                return stream('read_context', dict(limit=8192, offset=0, record=refs['instruction']), index)
        if index == 5:
            assert len(pages(request)) == 5
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
        assert len(kwargs['data']) <= 196608
        packet = json.loads(next(m['content'] for m in request['messages'] if m['role'] == 'user'))
        calls = {call['id'] for m in request['messages'] for call in m.get('tool_calls', [])}
        results = [m for m in request['messages'] if m['role'] == 'tool']
        assert calls == {m['tool_call_id'] for m in results}
        if index:
            assert any(json.loads(m['content'])['offset'] == (index-1)*4096 for m in results)
        if index == 10:
            assert len(results) == 10
            return stream('write', {'source': 'bounded kernel'}, index)
        return stream('read_context', {'record': packet['context']['host']['guide']['record'],
                                      'offset': index*4096, 'limit': 4096}, index)
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session() as pi:
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
    import coding.pi.actor as actor_module
    from benchmarks.a5kernels.trial import ActionOutcome, parse_trial_action
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider, direct_profile
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    monkeypatch.setattr(actor_module, 'runtime_preflight', lambda: None)
    actions = iter([('compile', {}), ('compile', {}), ('submit', {})])
    class FakeSession:
        completion_tokens = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def next_action(self, *args, **kwargs): return next(actions)
    monkeypatch.setattr(actor_module.PiActorDriver, '_new_session', lambda *args: FakeSession())
    driver = actor_module.PiActorDriver(DirectCompletionProvider({'OPENAI_API_KEY': 'fake', 'DEEPSEEK_API_KEY': 'fake'}), max_turns=1, max_seconds=10)
    seen = []
    def execute(action):
        seen.append(action.kind)
        return ActionOutcome('{}', action.kind == 'submit', 'passed', budget_exempt=action.kind == 'compile')
    outcome = driver('task', profile=direct_profile(next(iter(MODEL_IDENTITIES))), workspace=tmp_path,
                     context=None, turn_parser=parse_trial_action, action_executor=execute)
    assert outcome.status == 'passed' and seen == ['compile', 'compile', 'submit']
    assert outcome.continuation.deadline > time.monotonic()


def test_actor_expires_without_executing_another_action(monkeypatch, tmp_path):
    import coding.pi.actor as actor_module
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider
    monkeypatch.setattr(actor_module, 'runtime_preflight', lambda: None)
    class FakeSession:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def next_action(self, *args, **kwargs): pytest.fail('deadline expired')
    monkeypatch.setattr(actor_module.PiActorDriver, '_new_session', lambda *args: FakeSession())
    times = iter([0, 2])
    driver = actor_module.PiActorDriver(DirectCompletionProvider({'OPENAI_API_KEY': 'fake', 'DEEPSEEK_API_KEY': 'fake'}),
                                        max_seconds=1, clock=lambda: next(times))
    outcome = driver('task', profile=None, workspace=tmp_path, context=None,
                     turn_parser=None, action_executor=None)
    assert outcome.status == 'time-budget-exhausted' and outcome.iterations == 0


def test_actor_inherits_executor_deadline_and_preserves_clock_on_clone(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import coding.pi.actor as actor_module
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider
    monkeypatch.setattr(actor_module, 'runtime_preflight', lambda: None)
    clock = lambda: 10
    session = SimpleNamespace(deadline=4000, gateway=SimpleNamespace(deadline=4000))
    class Session:
        completion_tokens = 0
        deadline = 4000
        gateway = session.gateway
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def next_action(self, *args, **kwargs): pytest.fail('enclosing attempt expired')
    active = Session()
    monkeypatch.setattr(actor_module.PiActorDriver, '_new_session', lambda *args: active)
    driver = actor_module.PiActorDriver(DirectCompletionProvider(
        {'OPENAI_API_KEY': 'fake', 'DEEPSEEK_API_KEY': 'fake'}), max_seconds=3600, clock=clock)
    cloned = driver.with_trace(lambda *args: None)
    assert cloned.clock is clock
    outcome = cloned('task', profile=None, workspace=tmp_path, context=None,
        turn_parser=None, action_executor=SimpleNamespace(deadline=9))
    assert outcome.status == 'time-budget-exhausted'
    assert active.deadline == active.gateway.deadline == 9


def test_actor_reports_rejected_action_for_next_turn_recovery(monkeypatch, tmp_path):
    import coding.pi.actor as actor_module
    from benchmarks.a5kernels.trial import ActionOutcome, parse_trial_action
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider, direct_profile
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    monkeypatch.setattr(actor_module, 'runtime_preflight', lambda: None)
    observations = []
    class RecoveringSession:
        completion_tokens = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def next_action(self, context, *args, **kwargs):
            observations.append(context['latest_host_observation'])
            if len(observations) == 1:
                raise actor_module.ActionError('Pi returned no accepted host action')
            host = context['latest_host_observation']['host']
            assert host['event'] == 'invalid-action'
            assert host['error'] == 'Pi returned no accepted host action'
            return 'submit', {}
    monkeypatch.setattr(actor_module.PiActorDriver, '_new_session', lambda *args: RecoveringSession())
    driver = actor_module.PiActorDriver(DirectCompletionProvider({
        'OPENAI_API_KEY': 'fake', 'DEEPSEEK_API_KEY': 'fake'}), max_seconds=10)
    executed = []
    def execute(action):
        executed.append(action.kind)
        return ActionOutcome('{}', True, 'host-verified')
    outcome = driver('task', profile=direct_profile(next(iter(MODEL_IDENTITIES))),
                     workspace=tmp_path, context=None, turn_parser=parse_trial_action,
                     action_executor=execute)
    assert outcome.status == 'host-verified'
    assert executed == ['submit']
    assert len(observations) == 2


def test_real_pi_actor_waits_for_submit_acceptance(monkeypatch, tmp_path):
    from benchmarks.a5kernels.trial import ActionOutcome, parse_trial_action
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider, direct_profile
    from benchmarks.a5kernels.phase1_experiments import A5BackendModel, MODEL_IDENTITIES
    from coding.pi.actor import PiActorDriver
    try:
        runtime_preflight()
    except RuntimeError:
        pytest.skip('npm ci --prefix integrations/pi required')
    actions = [('run', {}), ('submit', {}), ('request-timing', {}), ('submit', {})]
    calls = []
    def provider(*args, **kwargs):
        calls.append(json.loads(kwargs['data']))
        name, payload = actions[len(calls)-1]
        return stream(name, payload, len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    seen = []
    verified, timed = False, False
    def execute(action):
        nonlocal verified, timed
        seen.append(action.kind)
        if action.kind == 'run':
            verified = True
            host = {'classification': 'host-verified'}
        elif action.kind == 'request-timing':
            assert verified
            timed = True
            host = {'event': 'performance-study', 'passed': True}
        else:
            assert action.kind == 'submit'
            if verified and timed:
                return ActionOutcome('{}', True, 'host-verified')
            host = {'error': 'submit requires retained project evidence', 'missing': ['performance-study']}
        return ActionOutcome(json.dumps({'host': host}), False, 'continue')
    driver = PiActorDriver(DirectCompletionProvider({'DEEPSEEK_API_KEY': 'fake'},
        models=(A5BackendModel.DEEPSEEK_FLASH,)), max_seconds=30)
    profile = direct_profile(next(i for i in MODEL_IDENTITIES if i.backend_model is A5BackendModel.DEEPSEEK_FLASH))
    outcome = driver('Synthetic task requires verified correctness and timing.', profile=profile,
                     workspace=tmp_path, context=None, turn_parser=parse_trial_action,
                     action_executor=execute)
    assert outcome.status == 'host-verified'
    assert seen == ['run', 'submit', 'request-timing', 'submit']
    assert len(calls) == 4


def test_actor_invalid_action_feedback_reaches_next_request(monkeypatch, tmp_path):
    import coding.pi.actor as actor_module
    from benchmarks.a5kernels.trial import ActionOutcome, parse_trial_action
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider, direct_profile
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    monkeypatch.setattr(actor_module, 'runtime_preflight', lambda: None)
    requests = []
    class FakeSession:
        completion_tokens = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def next_action(self, payload, allowed, **kwargs):
            requests.append((payload, set(allowed)))
            return ('not-an-action', {}) if len(requests) == 1 else ('submit', {})
    monkeypatch.setattr(actor_module.PiActorDriver, '_new_session', lambda *args: FakeSession())
    driver = actor_module.PiActorDriver(DirectCompletionProvider(
        {'OPENAI_API_KEY': 'fake', 'DEEPSEEK_API_KEY': 'fake'}), max_seconds=10)
    executed = []
    def execute(action):
        executed.append(action.kind)
        return ActionOutcome('{}', True, 'passed')
    outcome = driver('task', profile=direct_profile(next(iter(MODEL_IDENTITIES))),
                     workspace=tmp_path, context=None,
                     turn_parser=parse_trial_action, action_executor=execute)
    assert outcome.status == 'passed' and executed == ['submit']
    feedback = requests[1][0]['latest_host_observation']['host']
    assert feedback['event'] == 'invalid-action'
    assert feedback['allowed_actions'] == sorted(requests[1][1])
    assert 'submit' in feedback['allowed_actions']


def test_real_pi_escaped_snapshot_preserves_complete_pages_under_gateway_request_cap(monkeypatch):
    requests = []
    context = {'guide': 'Page contents. ' * 2000,
               'quoted': {f'field-{index}': '\\"' * 240 for index in range(16)}}
    def provider(*_, **kwargs):
        request = json.loads(kwargs['data'])
        requests.append(request)
        assert len(kwargs['data']) <= 196608
        snapshot = json.loads(next(row['content'] for row in request['messages'] if row['role'] == 'user'))
        assert snapshot['context']['host']['quoted'] == context['quoted']
        index = len(requests) - 1
        retained = [json.loads(row['content']) for row in request['messages'] if row['role'] == 'tool']
        if index:
            assert any(row['offset'] == (index - 1) * 7400 for row in retained)
        if index == 2:
            assert [row['offset'] for row in retained] == [0, 7400]
            calls = {call['id'] for row in request['messages'] for call in row.get('tool_calls', [])}
            assert calls == {row['tool_call_id'] for row in request['messages'] if row['role'] == 'tool'}
            return stream('write', {'source': 'selected after bounded paging'}, index)
        return stream('read_context', dict(record=snapshot['context']['host']['guide']['record'],
                                          offset=index * 7400, limit=7400), index)
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session() as pi:
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
