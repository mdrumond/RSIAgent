"""Pinned Pi performs summarization and overflow recovery through the real gateway."""
import json
import time

import pytest

from test_pi_runtime import session, stream


def text_response(text):
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def iter_lines(self):
            for choice in [dict(delta={'role': 'assistant', 'content': text}, finish_reason=None),
                           dict(delta={}, finish_reason='stop')]:
                yield b'data: ' + json.dumps(dict(id='summary', object='chat.completion.chunk', created=0,
                    model='deepseek-flash', choices=[dict(index=0, **choice)],
                    usage={'prompt_tokens': 2000, 'completion_tokens': 50})).encode()
                yield b''
            yield b'data: [DONE]'; yield b''
    return Response()


def with_usage(response, tokens):
    original = response.iter_lines
    def lines():
        for line in original():
            if line.startswith(b'data: {'):
                data = json.loads(line[6:])
                if data.get('usage'):
                    data['usage']['prompt_tokens'] = tokens
                    line = b'data: '+json.dumps(data).encode()
            yield line
    response.iter_lines = lines
    return response


def test_pi_compacts_actor_history_and_keeps_current_source_authoritative(monkeypatch):
    normal, summaries, events = [], [], []
    def provider(*_, **options):
        body = json.loads(options['data'])
        if not body.get('tools'):
            summaries.append(body)
            return text_response('Earlier source was old. Recheck the current host snapshot before editing.')
        normal.append(body)
        source = 'old'*10000 if len(normal) == 1 else 'fixed'
        return with_usage(stream('write', {'source': source}, len(normal)), 20000 if len(normal) == 1 else 100)
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session(deadline=time.monotonic()+40) as pi:
        assert pi.next_action({'task': 'change'}, ['write'], record=lambda *e: events.append(e)) == ('write', {'source': 'old'*10000})
        assert pi.next_action({'task': 'change', 'host_verified': False}, ['write'], source='current',
                              record=lambda *e: events.append(e)) == ('write', {'source': 'fixed'})
        assert summaries
        assert any(kind == 'pi-compaction' for kind, _ in events)
        metadata = json.loads(next(m['content'] for m in normal[-1]['messages'] if m['role'] == 'user'))
        import hashlib
        assert metadata['context']['source_sha256'] == hashlib.sha256(b'current').hexdigest()
        assert pi.mutations == 1
        assert len(pi.gateway.records) == len(normal) + len(summaries)


@pytest.mark.parametrize("mode", ["sse", "http"])
def test_pi_recovered_provider_overflow_does_not_leak_stale_rpc_failure(monkeypatch, mode):
    normal, summaries = [], []
    class Overflow:
        status_code = 400 if mode == "http" else 200
        def json(self): return {"error": {"message": "context_length_exceeded"}}
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def iter_lines(self):
            yield b'data: '+json.dumps({'error': {'message': 'context_length_exceeded'}}).encode()
            yield b''
            yield b'data: [DONE]'; yield b''
    def provider(*_, **options):
        body = json.loads(options['data'])
        if not body.get('tools'):
            summaries.append(body); return text_response('Continue using the current host snapshot.')
        normal.append(body)
        # There must be some earlier session history to compact.
        if len(normal) == 2: return Overflow()
        return stream('write', {'source': 'retained'*2000 if len(normal) == 1 else 'retained'}, len(normal))
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session(deadline=time.monotonic()+40) as pi:
        assert pi.next_action({'task': 'first'}, ['write']) == ('write', {'source': 'retained'*2000})
        assert pi.next_action({'task': 'second'}, ['write']) == ('write', {'source': 'retained'})
        assert summaries and len(normal) == 3
        assert pi.mutations == 1


def test_compaction_failure_discards_queued_action_and_reports_cause(monkeypatch):
    from coding.pi.rpc import PiError
    normal, summaries = [], []
    class FailedSummary:
        status_code = 500
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def json(self): return {'error': {'message': 'summary backend unavailable'}}
    def provider(*_, **options):
        body = json.loads(options['data'])
        if not body.get('tools'):
            summaries.append(body); return FailedSummary()
        normal.append(body)
        return with_usage(stream('write', {'source': 'old'*10000}, len(normal)), 20000)
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session(deadline=time.monotonic()+20) as pi:
        with pytest.raises(PiError, match='summary backend unavailable'):
            pi.next_action({'task': 'first'}, ['write'])
        assert summaries and pi.pending is None
