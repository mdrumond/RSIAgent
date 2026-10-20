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


def session(**kwargs):
    try:
        runtime_preflight()
    except RuntimeError:
        pytest.skip('npm ci --prefix integrations/pi required')
    return PiCodingSession(model='deepseek-flash', schemas={'write': obj({'source': STRING})},
                           environ={'DEEPSEEK_API_KEY': 'fake-provider-secret'}, deadline=time.monotonic()+30, **kwargs)


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
    with session(max_calls=1) as pi:
        with pytest.raises(PiError):
            pi.next_action({'task': 'TEST'}, ['write'])
        assert pi.pending is None
        assert pi.gateway.calls == 1


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
        with pytest.raises(PiError):
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
    with session(max_calls=1) as pi:
        with pytest.raises(PiError): pi.next_action({'task':'TEST'}, ['write'])
        assert pi.pending is None


def test_openai_responses_uses_registered_profile_and_counts_subsets(monkeypatch):
    try: runtime_preflight()
    except RuntimeError: pytest.skip('Pi runtime unavailable')
    captured = []
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def iter_lines(self):
            reasoning = {'type':'reasoning','id':'rs_0','summary':[], 'encrypted_content':'opaque'}
            call = {'type':'function_call', 'id':'fc_0', 'call_id':'call_0','name':'write',
                    'arguments':json.dumps({'source':'kernel'}), 'status':'completed'}
            events = [{'type':'response.created','response':{'id':'resp_0'}}]
            for i, item in enumerate([reasoning, call]):
                events += [{'type':'response.output_item.added','output_index':i,'item':item},
                           {'type':'response.output_item.done','output_index':i,'item':item}]
            events.append({'type':'response.completed','response':{'id':'resp_0','status':'completed',
                'output':[reasoning,call],'usage':{'input_tokens':100,'output_tokens':40,
                'input_tokens_details':{'cached_tokens':20},'output_tokens_details':{'reasoning_tokens':30}}}})
            for event in events:
                yield b'data: '+json.dumps(event).encode(); yield b''
    def provider(*_, **kwargs):
        captured.append(json.loads(kwargs['data'])); return Response()
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


def test_incremental_window_preserves_original_allocation_and_unknown_charge(tmp_path):
    from coding.pi.budget import Budget, BudgetExceeded
    path = tmp_path/'budget.json'
    old = Budget(path, 5)
    old.reserve({}, output_tokens=100, rates=(1,1), label='original')
    window = Budget(path, 5, window='production-smoke', window_limit=.01)
    window.reserve({}, output_tokens=100, rates=(1,1), label='new')
    window.reserve({}, output_tokens=100, rates=(1,1), label='new')
    with pytest.raises(BudgetExceeded, match='incremental'):
        window.reserve({}, output_tokens=100, rates=(1,1), label='new')
    assert window.snapshot()['limit_usd'] == 5
    assert len(window.snapshot()['requests']) == 3
    with pytest.raises(ValueError, match='different limit'):
        Budget(path,5,window='production-smoke',window_limit=1).reserve({},output_tokens=1,rates=(1,1),label='new')


def test_parallel_pi_sessions_keep_contexts_and_processes_isolated(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    def provider(*_, **kwargs):
        payload = json.loads(kwargs['data'])
        snapshot = json.loads(next(m['content'] for m in payload['messages'] if m['role']=='user'))
        return stream('write', {'source': snapshot['context']['host']['task']})
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    def run(task):
        with session(max_calls=1) as pi:
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
