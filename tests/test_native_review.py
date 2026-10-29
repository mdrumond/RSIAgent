"""Real Pi-native reads/standalone admission with a deterministic streamed provider."""
from types import SimpleNamespace
import json
import time

import pytest

from benchmarks.ascend.pi_project_verifier import session_factory
from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider
from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from coding.pi.session import runtime_preflight
from test_pi_project_verifier import stream_many
from test_pi_runtime import stream
from test_project_verifier_agent import Harness, prerequisites, decision, memory, request, S, R, L, SOURCE


def native(h, monkeypatch, batches):
    try:
        runtime_preflight()
    except RuntimeError:
        pytest.skip('Node22 and pinned Pi required')
    selected = next(p.backend_model for p in MODEL_IDENTITIES if p.backend_model.value == 'deepseek-flash')
    provider = DirectCompletionProvider({'DEEPSEEK_API_KEY': 'private-native-test-key'}, models=[selected])
    calls, batches = [], iter(batches)
    def http(*_, **options):
        payload = json.loads(options['data'])
        calls.append(payload)
        assert len(options['data']) <= 49152
        batch = next(batches)
        return batch(payload) if callable(batch) else stream_many(batch)
    monkeypatch.setattr('coding.pi.gateway.requests.post', http)
    h.agent.session_factory = session_factory({}, provider, SimpleNamespace(model='deepseek-flash'))
    return calls


def test_native_multi_reads_have_real_results_and_only_standalone_terminal_finishes(monkeypatch):
    h = Harness([])
    def finish(payload):
        results = [json.loads(row['content']) for row in payload['messages'] if row['role'] == 'tool']
        assert {value['sha256'] for value in results if 'sha256' in value} == {S, R, L}
        assert next(value for value in results if value.get('sha256') == S)['text'] == SOURCE.decode()
        assert all('status' not in value for value in results)
        return stream_many([decision(memory())])
    calls = native(h, monkeypatch, [prerequisites(), finish])
    assert h.agent.review(request(), max_seconds=20) == decision(memory())
    assert len(calls) == 2
    assert [d for d, _ in h.reads[:3]] == [S, R, L]
    assert len(h.queries) == 1
    assert len([e for e in h.agent.last_trace if e['kind'] == 'decision']) == 1
    assert len([e for e in h.agent.last_trace if e['kind'] == 'pi-call-proposed']) == 5


def test_mixed_terminal_and_check_do_not_execute_but_reads_do(monkeypatch):
    h = Harness([])
    check = {'request_check': {'hypothesis': 'Control', 'check': {'variant': 'other'}}}
    calls = native(h, monkeypatch, [[*prerequisites(), check, decision(memory())], [decision(memory())]])
    assert h.agent.review(request(), max_seconds=20) == decision(memory())
    assert not h.checks
    assert len(calls) == 2
    proposed = [e['payload'] for e in h.agent.last_trace if e['kind'] == 'pi-call-proposed']
    blocked = [e for e in proposed if e['name'] in {'submit_review', 'request_check'} and e['disposition'] == 'rejected']
    assert {e['name'] for e in blocked} == {'submit_review', 'request_check'}
    assert len([e for e in h.agent.last_trace if e['kind'] == 'decision']) == 1


def test_original_arguments_rejected_before_coercion_and_cache_rereads_allowed(monkeypatch):
    h = Harness([])
    bad = {'read_artifact': {'sha256': S, 'offset': '0'}}
    def finish(payload):
        pages = [json.loads(m['content']) for m in payload['messages'] if m['role'] == 'tool']
        pages = [p for p in pages if p.get('sha256') == S]
        assert pages[-1]['cache_hit'] is True and pages[-1]['repeat'] is True
        assert pages[-1]['page_requests'] == 2
        return stream_many([decision(memory())])
    native(h, monkeypatch, [[bad, *prerequisites()[1:]], [prerequisites()[0], prerequisites()[0]], finish])
    assert h.agent.review(request(), max_seconds=20) == decision(memory())
    selected = [e['payload'] for e in h.agent.last_trace if e['kind'] == 'pi-call-proposed']
    assert selected[0]['arguments']['offset'] == '0'
    assert selected[0]['disposition'] == 'rejected'
    reads = [e['payload'] for e in h.agent.last_trace if e['kind'] == 'review-read' and e['payload']['sha256'] == S]
    assert len(reads) == 2
    assert reads[0]['cache_hit'] is False and reads[1]['cache_hit'] is True and reads[1]['repeat'] is True
    # One actual cache load plus the existing independent snippet validation read.
    assert [d for d, _ in h.reads].count(S) == 2


def test_invalid_terminal_returns_feedback_then_validates_and_utf8_pages_reconstruct(monkeypatch):
    from benchmarks.ascend.project_verifier import ProjectReview
    import hashlib
    text = '\n' * 4000 + 'αé😀' * 10
    raw = text.encode()
    digest = hashlib.sha256(raw).hexdigest()
    h = Harness([])
    h.artifacts[digest] = raw
    offset, chunks = [0], []
    def provider(payload):
        tools = [json.loads(m['content']) for m in payload['messages'] if m['role'] == 'tool']
        for page in tools[-1:]:
            if page.get('sha256') == digest:
                assert len(page['text'].encode()) <= 2048
                assert len(json.dumps(page, ensure_ascii=False).encode()) <= 4096
                chunks.append(page['text'])
                offset[0] = page['next_offset']
        if offset[0] is not None:
            return stream('read_artifact', {'sha256': digest, 'offset': offset[0]})
        return stream_many([decision()])
    native(h, monkeypatch, [[decision()], [*prerequisites()[1:]], *[provider] * 10])
    req = ProjectReview('utf8', digest, True, {'result': R, 'actor-log': L})
    assert h.agent.review(req, max_seconds=20)['decision'] == 'accept'
    assert ''.join(chunks) == text
    assert any(e['kind'] == 'feedback' for e in h.agent.last_trace)


def test_standalone_check_updates_shared_host_gate_and_failure_requires_repair(monkeypatch):
    from test_project_verifier_agent import repair
    h = Harness([], check={'host_verified': False, 'artifacts': {}})
    check = {'request_check': {'hypothesis': 'Control', 'check': {'variant': 'other'}}}
    native(h, monkeypatch, [prerequisites(), [check], [decision(memory())], [repair()]])
    assert h.agent.review(request(), max_seconds=20) == repair()
    assert len(h.checks) == 1
    assert any(e['kind'] == 'feedback' for e in h.agent.last_trace)


def test_model_end_without_verdict_continues_same_review_without_auto_accept(monkeypatch):
    class Text:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def iter_lines(self):
            for choice in [dict(delta={'role': 'assistant', 'content': 'I will inspect the evidence.'}, finish_reason=None),
                           dict(delta={}, finish_reason='stop')]:
                yield b'data: ' + json.dumps(dict(id='text', object='chat.completion.chunk', created=0,
                    model='deepseek-flash', choices=[dict(index=0, **choice)],
                    usage={'prompt_tokens': 20, 'completion_tokens': 10})).encode()
                yield b''
            yield b'data: [DONE]'
            yield b''
    h = Harness([])
    calls = native(h, monkeypatch, [lambda _: Text(), prerequisites(), [decision(memory())]])
    assert h.agent.review(request(), max_seconds=20) == decision(memory())
    assert len(calls) == 3
    assert len([e for e in h.agent.last_trace if e['kind'] == 'pi-review-continuation']) == 1
    assert len([e for e in h.agent.last_trace if e['kind'] == 'decision']) == 1


def test_native_long_conversation_projects_whole_packets_without_dropping_tool_pairs(monkeypatch):
    from benchmarks.ascend.review_projection import exchanges
    import hashlib
    h = Harness([], knowledge={'memories': [], 'citations': [], 'hits': [
        {'citation': {'collection': 'baseline', 'path': f'kernel{i}.cpp'}, 'text': 'α'*500}
        for i in range(5)]})
    query = {'knowledge_query': {'query': 'increment examples'}}
    calls = native(h, monkeypatch, [[query]] * 70 + [prerequisites(), [decision(memory())]])
    assert h.agent.review(request(), max_seconds=40) == decision(memory())
    projections = [e['payload'] for e in h.agent.last_trace if e['kind'] == 'pi-request-projection']
    assert any(p['raw_bytes'] > 49152 for p in projections)
    assert all(p['forwarded_bytes'] <= 49152 for p in projections)
    final = calls[-1]
    metadata = json.loads(next(m['content'] for m in final['messages'] if m['role'] == 'user'))
    reference = metadata['older_exchanges']
    stored = next(e['payload'] for e in h.agent.last_trace if e['kind'] == 'pi-context-record'
                  and e['payload']['record'] == reference['record'])
    assert hashlib.sha256(stored['text'].encode()).hexdigest() == reference['record']
    archived = json.loads(stored['text'])
    assert any(batch for _, batch in exchanges(archived))
    assert any(batch for _, batch in exchanges(final['messages']))
    assert metadata['review_context']['host']['review_state']['ready_for_decision'] is True


def test_native_callback_cancellation_keeps_typed_timeout_and_no_verdict(monkeypatch):
    h = Harness([])
    def cancelled(*args, **kwargs):
        raise TimeoutError('user stopped this review')
    h.agent.query_knowledge = cancelled
    native(h, monkeypatch, [prerequisites()])
    with pytest.raises(TimeoutError):
        h.agent.review(request(), max_seconds=20)
    assert not any(e['kind'] == 'decision' for e in h.agent.last_trace)
    assert any(e['kind'] == 'pi-call-result' and e['payload']['disposition'] == 'interrupted'
               for e in h.agent.last_trace)


def test_native_rpc_abort_is_typed_cancellation_without_verdict(monkeypatch):
    from benchmarks.ascend.native_review import PiReviewSession
    original = PiReviewSession._start
    active = []
    def start(session):
        original(session)
        active.append(session)
    monkeypatch.setattr(PiReviewSession, '_start', start)
    def cancelled(payload):
        active[0].rpc.send({'type': 'abort'})
        response = stream_many(prerequisites())
        original_lines = response.iter_lines
        def slow_lines():
            time.sleep(.2)  # Abort an active model read, not an idle pre-prompt process.
            yield from original_lines()
        response.iter_lines = slow_lines
        return response
    h = Harness([])
    native(h, monkeypatch, [cancelled])
    with pytest.raises(TimeoutError):
        h.agent.review(request(), max_seconds=10)
    assert not any(e['kind'] == 'decision' for e in h.agent.last_trace)


@pytest.mark.parametrize('rewrite_feedback', [None, {'instructions':'Preserve the actual retained original'}])
def test_native_original_source_coverage_and_before_contract_match_completion(monkeypatch, rewrite_feedback):
    from test_project_verifier_agent import B, BEFORE, read, retained_request, contrasted_memory, snippet
    h = Harness([])
    bad = memory()
    bad.update(before=snippet(),before_unavailable_reason=None)
    calls = native(h, monkeypatch, [
        [*prerequisites(),{'read_artifact':{'sha256':B,'offset':0,'limit':4}}],
        [decision(memory())],
        [{'read_artifact':{'sha256':B,'offset':4}}],
        [decision(memory())],
        [decision(bad)],
        [decision(contrasted_memory())]])
    result = h.agent.review(retained_request(),max_seconds=30,feedback=rewrite_feedback)
    assert result['memory']['before']['sha256']==B
    feedback = [e['payload'] for e in h.agent.last_trace if e['kind']=='feedback']
    assert [e['error_code'] for e in feedback]==[
        'incomplete-original-source','retained-before-required','identical-before-after']
    missing = next(row for row in feedback[0]['missing_required_artifacts'] if row['sha256']==B)
    assert missing['missing_ranges']==[{'offset':4,'end':len(BEFORE)}]
    assert missing['next_read']=={'sha256':B,'offset':4,'limit':2048}
    assert B in feedback[0]['allowed_evidence'] and B in feedback[0]['readable_artifacts']
    assert len(calls)==6
    assert len([e for e in h.agent.last_trace if e['kind']=='decision'])==1
