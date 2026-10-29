"""Whole Chat/Responses packet limits, exact history paging, and pair preservation."""
import json

import pytest

from benchmarks.ascend.review_context import ReviewContextRecords
from benchmarks.ascend.review_projection import ReviewRequestProjector, exchanges
from coding.pi.context import encoded
from coding.pi.gateway import Gateway
from coding.pi.session import RequestAccounting
from test_review_context import fixture, reconstruct


@pytest.mark.parametrize('responses', [False, True])
def test_whole_request_pages_old_complete_exchanges_and_keeps_latest_batch(responses):
    records, events = ReviewContextRecords(), []
    projector = ReviewRequestProjector(records, fixture, lambda kind, value: events.append((kind, value)))
    rows = [dict(role='user', content='Review the retained project')]
    for turn in range(24):
        names = [f'call_{turn}_{n}' for n in range(3)]
        if responses:
            rows += [dict(type='function_call', call_id=name, name='read_artifact', arguments='{}') for name in names]
            rows += [dict(type='function_call_output', call_id=name,
                          output=json.dumps(dict(text='éα😀' * 250, turn=turn))) for name in names]
        else:
            rows.append(dict(role='assistant', content=None, tool_calls=[dict(id=name, type='function',
                function=dict(name='read_artifact', arguments='{}')) for name in names]))
            rows += [dict(role='tool', tool_call_id=name,
                          content=json.dumps(dict(text='éα😀' * 250, turn=turn))) for name in names]
    instruction = 'Host instructions ' * 300
    tools = [dict(type='function', name='read_artifact', description='Schema details ' * 200,
                  parameters={'type': 'object'})]
    payload = dict(model='gpt-5.6-sol' if responses else 'deepseek-flash', tools=tools,
        **({'input': rows, 'instructions': instruction} if responses else
           {'messages': [dict(role='system', content=instruction), *rows]}))
    gateway = Gateway('openai' if responses else 'deepseek', 'private-test', RequestAccounting(), 'projection',
                      request_projector=projector)
    assert len(encoded(payload)) > 49152
    packed = gateway.prepare(payload, responses=responses)
    assert len(encoded(packed)) <= 49152
    assert packed['tools'] == tools
    if responses:
        assert packed['instructions'] == instruction
        projected = packed['input']
    else:
        assert packed['messages'][0]['content'] == instruction
        projected = packed['messages'][1:]
    metadata = json.loads(projected[0]['content'])
    assert metadata['review_context']['host']['review_state'] == fixture()['host']['review_state']
    assert metadata['review_context']['host']['project']['rewrite_feedback']['known_counterevidence'] == fixture()['host']['project']['rewrite_feedback']['known_counterevidence']
    archived = json.loads(reconstruct(records, metadata['older_exchanges']))
    assert archived + projected[1:] == rows
    original_latest = exchanges(rows, responses)[-1][0]
    assert projected[-len(original_latest):] == original_latest
    assert all(batch for _, batch in exchanges(projected[1:], responses))
    assert any(kind == 'pi-raw-provider-request' for kind, _ in events)


def test_default_gateway_keeps_existing_hard_limit():
    gateway = Gateway('deepseek', 'fake', RequestAccounting(), 'actor-default')
    with pytest.raises(ValueError):
        gateway.prepare(dict(model='deepseek-flash', messages=[dict(role='user', content='x'*50000)]))


def test_impossible_latest_batch_is_held_before_provider_dispatch():
    records = ReviewContextRecords()
    projector = ReviewRequestProjector(records, fixture, lambda *a: None)
    gateway = Gateway('deepseek', 'fake', RequestAccounting(), 'capacity', request_projector=projector)
    rows = [dict(role='assistant', tool_calls=[dict(id='latest', type='function',
        function=dict(name='read_context', arguments='{}'))]),
        dict(role='tool', tool_call_id='latest', content='x'*50000)]
    with pytest.raises(ValueError):
        gateway.prepare(dict(model='deepseek-flash', messages=rows))
    assert gateway.calls == 0


def test_gateway_reports_context_exhaustion_before_dispatch(monkeypatch):
    from http.client import HTTPConnection
    from coding.pi.context import ContextLimitError
    def project(*args, **kwargs):
        raise ContextLimitError('mandatory review controls exceed target byte budget')
    monkeypatch.setattr('coding.pi.gateway.requests.post',
        lambda *a, **kw: pytest.fail('a context failure must not reach the provider'))
    with Gateway('deepseek', 'private-test', RequestAccounting(), 'context-limit',
                 request_projector=project) as gateway:
        connection = HTTPConnection('127.0.0.1', gateway.server.server_port, timeout=5)
        connection.request('POST', '/v1/chat/completions',
            body=json.dumps(dict(model='deepseek-flash', messages=[dict(role='user', content='Review')])).encode(),
            headers={'Authorization':'Bearer '+gateway.token, 'Content-Type':'application/json'})
        response = connection.getresponse()
        assert response.status == 413
        response.read();connection.close()
        assert gateway.stop_reason == 'context-limit'
        assert gateway.calls == 0
