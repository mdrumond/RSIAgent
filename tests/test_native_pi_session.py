"""Native Pi owns tool dispatch and persistent conversation, including continuation."""
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from coding.pi.native import NativeSession, NativeTool, NativeError, Cancelled


@pytest.fixture
def provider():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            message = dict(role='assistant', content='Native work completed.')
            if len(requests) == 1:
                message.update(content=None, tool_calls=[dict(id='native-call-1', type='function',
                    function=dict(name='observe', arguments='{"value":7}'))])
            payload = dict(id='fixture', object='chat.completion', model='deepseek-flash',
                choices=[dict(index=0, message=message, finish_reason='tool_calls' if len(requests)==1 else 'stop')],
                usage=dict(prompt_tokens=10, completion_tokens=2, total_tokens=12))
            # The real SDK's streaming parser consumes the provider fixture.
            delta = dict(message)
            if 'tool_calls' in delta:
                delta['tool_calls'][0]['index'] = 0
            chunk = dict(payload, object='chat.completion.chunk',
                         choices=[dict(index=0, delta=delta, finish_reason=payload['choices'][0]['finish_reason'])])
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            self.wfile.write(('data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n').encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', requests
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_real_sdk_tool_result_continues_and_retains_native_session(tmp_path, monkeypatch, provider):
    url, requests = provider
    import coding.pi.native as native
    original = native.model_profile
    monkeypatch.setattr(native, 'model_profile', lambda name: dict(original(name), base_url=url))
    calls, events = [], []
    def observe(args, context):
        calls.append((args, context))
        return dict(content=[dict(type='text', text='Observed seven.')])
    tool = NativeTool('observe', 'Observe a value.',
        dict(type='object', properties=dict(value=dict(type='integer')), required=['value']), observe)
    with NativeSession(tmp_path, tools=[tool], environ={'DEEPSEEK_API_KEY': 'fixture-secret'},
                       observer=events.append) as session:
        result = session.prompt('Use observe with value 7.', operation_id='operation', remaining=lambda: 30.)
        assert result.provider_settled and not result.pending_tools
        assert calls[0][0] == {'value': 7}
        assert calls[0][1]['tool_call_id'] == 'native-call-1'
        assert calls[0][1]['operation_id'] == 'operation'
        assert len(requests) == 2
        assert any(m['role']=='tool' for m in requests[1]['messages'])
        assert Path(result.session_file).is_file()
        assert session.completion_tokens == 4
    assert any(e['type']=='agent_settled' for e in events)
    assert 'fixture-secret' not in (tmp_path/'native-events.jsonl').read_text()
    with NativeSession(tmp_path, tools=[tool], environ={'DEEPSEEK_API_KEY': 'fixture-secret'}) as reopened:
        resumed = reopened.prompt('Recall the observed value.', operation_id='next', remaining=lambda: 30.)
        assert resumed.provider_settled and resumed.session_file == result.session_file
        assert any(m.get('tool_call_id')=='native-call-1' for m in requests[-1]['messages'])


@pytest.mark.parametrize('action', ['denied', 'abort', 'transport_loss'])
def test_native_feedback_abort_and_unknown_closure_are_distinct(tmp_path, monkeypatch, provider, action):
    import coding.pi.native as native
    url, requests = provider
    original = native.model_profile
    monkeypatch.setattr(native, 'model_profile', lambda name: dict(original(name), base_url=url))
    events = []
    def observe(args, context):
        if action == 'denied':
            raise PermissionError('Path is not admitted')
        if action == 'abort':
            session.abort()
        else:
            session.process.terminate(); session.process.wait(timeout=3)
        return dict(content=[dict(type='text', text='Host tool stopped')])
    tool = NativeTool('observe', 'Observe a value.', dict(type='object', properties={}), observe)
    with NativeSession(tmp_path, tools=[tool], environ={'DEEPSEEK_API_KEY': 'fixture-secret'},
                       observer=events.append) as session:
        if action == 'denied':
            result = session.prompt('Use observe.', operation_id='denied', remaining=lambda: 30.)
            assert result.provider_settled and len(requests) == 2
            assert any(e['type']=='host_callback_error' and e['origin']=='domain' for e in events)
        else:
            with pytest.raises(Cancelled if action=='abort' else NativeError) as error:
                session.prompt('Use observe.', operation_id=action, remaining=lambda: 30.)
            result = error.value.settlement
            if action == 'abort':
                assert result.provider_settled
                assert any(e['type']=='agent_settled' and e['aborted'] for e in events)
            else:
                assert not result.provider_settled and result.pending_tools == ('native-call-1',)
                assert not any(e['type']=='agent_settled' for e in events)
