"""Pinned Pi performs summarization and overflow recovery through the real gateway."""
import json
import time

import pytest

from test_pi_runtime import session, stream


def text_response(text, *, finish="stop", output_tokens=50):
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def iter_lines(self):
            for choice in [dict(delta={'role': 'assistant', 'content': text}, finish_reason=None),
                           dict(delta={}, finish_reason=finish)]:
                yield b'data: ' + json.dumps(dict(id='summary', object='chat.completion.chunk', created=0,
                    model='deepseek-flash', choices=[dict(index=0, **choice)],
                    usage={'prompt_tokens': 2000, 'completion_tokens': output_tokens})).encode()
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


@pytest.mark.parametrize("mode", ["sse", "http", "early-length"])
def test_pi_recovers_overflow_or_early_length_without_replaying_host_actions(monkeypatch, mode):
    normal, summaries = [], []
    class Overflow:
        status_code = 400 if mode == "http" else 200
        def json(self): return {"error": {"message": "context_length_exceeded"}}
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def iter_lines(self):
            if mode == 'early-length':
                yield from text_response('Incomplete response', finish='length').iter_lines()
                return
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


@pytest.mark.parametrize("trigger", ["threshold", "early-length"])
def test_compaction_failure_discards_queued_action_and_reports_cause(monkeypatch, trigger):
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
        if trigger == 'early-length' and len(normal) == 2:
            return text_response('Incomplete response', finish='length')
        return with_usage(stream('write', {'source': 'old'*10000}, len(normal)),
                          20000 if trigger == 'threshold' else 100)
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session(deadline=time.monotonic()+20) as pi:
        if trigger == 'early-length':
            pi.next_action({'task': 'seed history'}, ['write'])
        with pytest.raises(PiError, match='summary backend unavailable'):
            pi.next_action({'task': 'first'}, ['write'])
        assert summaries and pi.pending is None


def test_full_output_limit_still_rejects_truncated_response_without_action(monkeypatch):
    from coding.pi.session import ActionError
    calls = []
    def provider(*_, **options):
        body = json.loads(options['data']); calls.append(body)
        assert body.get('tools'), 'A full output limit must not trigger overflow compaction'
        if len(calls) == 1:
            return stream('write', {'source': 'retained'*2000})
        return text_response('Incomplete response', finish='length', output_tokens=8192)
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    with session(deadline=time.monotonic()+20) as pi:
        assert pi.next_action({'task': 'first'}, ['write']) == ('write', {'source': 'retained'*2000})
        with pytest.raises(ActionError, match='truncated'):
            pi.next_action({'task': 'second'}, ['write'])
        assert pi.pending is None and pi.mutations == 0
    assert len(calls) == 2


@pytest.mark.parametrize('failure_type', [TimeoutError, RuntimeError])
def test_successful_compaction_preserves_prior_verifier_callback_failure(failure_type):
    from benchmarks.ascend.native_review import PiReviewSession
    from benchmarks.ascend.pi_project_verifier import SCHEMAS
    from coding.pi.session import runtime_preflight
    try:
        runtime_preflight()
    except RuntimeError:
        pytest.skip('Node22 and pinned Pi required')
    failure = failure_type('retained callback failure')
    with PiReviewSession(model='deepseek-flash', schemas=SCHEMAS,
                         environ={'DEEPSEEK_API_KEY': 'fake-provider-secret'}) as pi:
        pi.failure = failure
        pi._operation('compaction', {'fromExtension': False})
        assert pi.failure is failure and pi.compaction_failure is None
