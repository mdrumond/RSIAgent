"""Production tool routing and real Pi argument exchanges with a fake provider."""
from contextlib import nullcontext
from types import SimpleNamespace
import json
import time

import pytest

from benchmarks.ascend.pi_project_verifier import session_factory, SCHEMAS, SYSTEM
from benchmarks.a5kernels.phase1_provider import openai_compatible_transport
from benchmarks.ascend.project_verifier import ProjectVerifierAgent
from coding.pi.session import PiCodingSession, runtime_preflight
from test_project_verifier_agent import Harness, prerequisites, decision, memory, request, L, S, R
from test_pi_runtime import stream


def test_production_defaults_pi_but_injected_transport_stays_compatible(monkeypatch):
    made = []
    monkeypatch.setattr('benchmarks.ascend.pi_project_verifier.PiReviewSession',
                        lambda **kwargs: made.append(kwargs) or nullcontext(kwargs))
    completion = SimpleNamespace(model='deepseek-flash')
    live = SimpleNamespace(_transport=openai_compatible_transport)
    assert session_factory({}, SimpleNamespace(_transport=object()), completion) is None
    assert session_factory({'verifier_engine': 'completion'}, live, completion) is None
    create = session_factory({}, live, completion)
    before = time.monotonic()
    with create(max_seconds=7):
        pass
    assert made[0]['model'] == 'deepseek-flash'
    from test_review_context import fixture
    from coding.pi.context import encoded
    packed = made[0]['context_factory']().snapshot(fixture(), target=16384)
    assert len(encoded(packed)) <= 16384
    assert packed['host']['review_state']['missing_required_artifacts']
    assert before + 7 <= made[0]['deadline'] <= time.monotonic() + 7
    with pytest.raises(ValueError):
        session_factory({'verifier_engine': 'unknown'}, live, completion)


def test_real_pi_verifier_reads_missing_log_and_admits_only_host_valid_decision(monkeypatch):
    try:
        runtime_preflight()
    except RuntimeError:
        pytest.skip('Node22 and pinned Pi required')
    h = Harness([])
    actions = iter([*prerequisites()[:2], decision(), prerequisites()[2],
                    prerequisites()[3], decision(memory())])
    calls = []
    def provider(*_, **options):
        action = next(actions)
        name, args = ('submit_review', action) if 'decision' in action else next(iter(action.items()))
        calls.append(json.loads(options['data']))
        return stream(name, args, len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    h.agent.session_factory = lambda **options: PiCodingSession(model='deepseek-flash', schemas=SCHEMAS,
        environ={'DEEPSEEK_API_KEY': 'fake-provider-secret'}, system_prompt=SYSTEM,
        deadline=time.monotonic()+options['max_seconds'])
    assert h.agent.review(request(), max_seconds=30) == decision(memory())
    assert L in [digest for digest, _ in h.reads]
    assert len(calls) == 6
    assert not h.calls


def test_subject_composition_selects_configured_verifier_model(tmp_path, monkeypatch):
    from benchmarks.ascend.subject_runtime import build_subject_campaign
    from benchmarks.ascend.subjects import Subject
    from test_subject_runtime import bound
    made = []
    monkeypatch.setattr('benchmarks.ascend.pi_project_verifier.PiReviewSession',
                        lambda **kwargs: made.append(kwargs) or nullcontext(kwargs))
    campaign = build_subject_campaign({'verifier_engine': 'pi', 'verifier_model': 'deepseek-flash',
        }, target='a3', root=tmp_path,
        subjects=[Subject('s', 'copies')], initial_memory=b'', compose=lambda *a, **k: bound(tmp_path))
    with campaign.verifier.session_factory(max_seconds=5):
        pass
    assert made[0]['model'] == 'deepseek-flash'


def test_verifier_engine_is_pinned_and_legacy_resume_keeps_completion(tmp_path):
    from benchmarks.ascend.subject_runtime import build_subject_campaign
    from benchmarks.ascend.subjects import Subject
    from test_subject_runtime import bound
    live = [False]
    def compose(*args, **options):
        result = bound(tmp_path)
        if live[0]:
            result['provider'] = SimpleNamespace(_transport=openai_compatible_transport)
        return result
    options = dict(target='a3', root=tmp_path, subjects=[Subject('s', 'copies')],
                   initial_memory=b'', compose=compose)
    campaign = build_subject_campaign({}, **options)
    assert campaign.verifier.session_factory is None
    identity = tmp_path/'project-verifier-runtime.json'
    assert json.loads(identity.read_text())['engine'] == 'completion'
    live[0] = True
    with pytest.raises(ValueError, match='project verifier engine or model changed'):
        build_subject_campaign({}, **options)
    identity.unlink()  # Simulate a checkpoint written before engine metadata existed.
    legacy = build_subject_campaign({}, **options)
    assert legacy.verifier.session_factory is None
    assert json.loads(identity.read_text())['engine'] == 'completion'


def test_new_live_composition_pins_pi_engine(tmp_path):
    from benchmarks.ascend.subject_runtime import build_subject_campaign
    from benchmarks.ascend.subjects import Subject
    from test_subject_runtime import bound
    def compose(*args, **options):
        result = bound(tmp_path)
        result['provider'] = SimpleNamespace(_transport=openai_compatible_transport)
        return result
    campaign = build_subject_campaign({}, target='a3', root=tmp_path,
        subjects=[Subject('s', 'copies')], initial_memory=b'', compose=compose)
    assert campaign.verifier.session_factory is not None
    identity = json.loads((tmp_path/'project-verifier-runtime.json').read_text())
    assert identity['engine'] == 'pi'
    assert identity['runtime']['pi_version'] == '1.3.0'
    assert identity['runtime']['context_policy'] == identity['context_policy']


def stream_many(actions):
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def iter_lines(self):
            calls = []
            for i, action in enumerate(actions):
                name, args = (('submit_review', action) if 'decision' in action else next(iter(action.items())))
                calls.append({'index': i, 'id': f'call_{i}', 'type': 'function',
                              'function': {'name': name, 'arguments': json.dumps(args)}})
            for part in [{'delta': {'role': 'assistant', 'tool_calls': calls}, 'finish_reason': None},
                         {'delta': {}, 'finish_reason': 'tool_calls'}]:
                chunk = dict(id='multi', object='chat.completion.chunk', created=0, model='deepseek-flash',
                    choices=[dict(index=0, **part)], usage={'prompt_tokens': 50, 'completion_tokens': 20})
                yield b'data: ' + json.dumps(chunk).encode()
                yield b''
            yield b'data: [DONE]'
            yield b''
    return Response()


def test_compat_first_arriving_action_cannot_bypass_host_evidence_gate(monkeypatch):
    try:
        runtime_preflight()
    except RuntimeError:
        pytest.skip('Node22 and pinned Pi required')
    h = Harness([])
    calls = []
    def provider(*_, **options):
        body = json.loads(options['data'])
        calls.append(body)
        if len(calls) == 1:
            return stream_many([*prerequisites()[:3], decision(memory())])
        # The legacy extension permits parallel HTTP arrivals. Whichever call
        # arrived first, follow the authoritative host's remaining evidence.
        snapshot = json.loads(next(m['content'] for m in body['messages'] if m['role'] == 'user'))
        state = snapshot['context']['host']['review_state']
        if state['missing_required_artifacts']:
            action = {'read_artifact': {'sha256': state['missing_required_artifacts'][0]['sha256']}}
        elif not state['knowledge_queried']:
            action = {'knowledge_query': {'query': 'increment'}}
        else:
            action = decision(memory())
        name, args = ('submit_review', action) if 'decision' in action else next(iter(action.items()))
        return stream(name, args, len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    h.agent.session_factory = lambda **options: PiCodingSession(model='deepseek-flash', schemas=SCHEMAS,
        environ={'DEEPSEEK_API_KEY': 'fake-provider-secret'}, system_prompt=SYSTEM,
        deadline=time.monotonic()+options['max_seconds'], first_action_only=True)
    assert h.agent.review(request(), max_seconds=30) == decision(memory())
    assert {digest for digest, _ in h.reads} == {S, R, L}
    selected = [e['payload'] for e in h.agent.last_trace if e['kind'] == 'pi-action']
    assert selected[0]['discarded_actions'] == 3
    assert len([e for e in h.agent.last_trace if e['kind'] == 'pi-action-rejected']) == 3
    assert len([e for e in h.agent.last_trace if e['kind'] == 'decision']) == 1


def test_actor_still_rejects_multiple_queued_actions(monkeypatch):
    from coding.pi.session import ActionError
    try:
        runtime_preflight()
    except RuntimeError:
        pytest.skip('Node22 and pinned Pi required')
    monkeypatch.setattr('coding.pi.gateway.requests.post', lambda *a, **k: stream_many(prerequisites()[:2]))
    with PiCodingSession(model='deepseek-flash', schemas=SCHEMAS,
            environ={'DEEPSEEK_API_KEY': 'fake-provider-secret'}, deadline=time.monotonic()+30) as session:
        with pytest.raises(ActionError):
            session.next_action({}, ['read_artifact'])
        assert session.pending is None


def test_real_pi_rejects_prose_evidence_then_corrects_with_native_history(monkeypatch):
    try:
        runtime_preflight()
    except RuntimeError:
        pytest.skip('Node22 and pinned Pi required')
    h = Harness([])
    bad = decision(memory())
    bad['evidence'] = [f'source {S} with observed result {R}']
    actions = iter([*prerequisites(), bad, *[prerequisites()[0]] * 15, decision(memory())])
    calls = []
    def provider(*_, **options):
        action = next(actions)
        name, args = ('submit_review', action) if 'decision' in action else next(iter(action.items()))
        calls.append(json.loads(options['data']))
        assert len(options['data']) <= 196608
        return stream(name, args, len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    h.agent.session_factory = lambda **options: PiCodingSession(model='deepseek-flash', schemas=SCHEMAS,
        environ={'DEEPSEEK_API_KEY': 'fake-provider-secret'}, system_prompt=SYSTEM,
        deadline=time.monotonic()+options['max_seconds'], first_action_only=True)
    assert h.agent.review(request(), max_seconds=30) == decision(memory())
    assert len(calls) == 21
    assert len([e for e in h.agent.last_trace if e['kind'] == 'decision']) == 1
    submitted = [e['payload'] for e in h.agent.last_trace
                 if e['kind'] == 'pi-action' and e['payload']['name'] == 'submit_review']
    assert len(submitted) == 1
    assert submitted[0]['arguments']['evidence'] == [R]


@pytest.mark.parametrize('legacy', [False, True])
def test_admitted_runtime_config_preserves_verifier_route_and_retained_review(tmp_path, legacy):
    from benchmarks.ascend.subject_runtime import build_subject_campaign, referenced_settings
    from benchmarks.ascend.project_verifier import ProjectReview
    from benchmarks.ascend.subjects import Subject
    from optimizer.jobs import canonical
    from test_subject_runtime import bound
    from test_a5_phase1_adaptive import _project
    import hashlib
    native = tmp_path/'native.json'
    native.write_text(json.dumps({'physical_device': 0, 'workspace': 'retained', 'compiler': 'pin'}))
    settings = {'phase1_arguments': ['--worker-config', str(native)]}
    root = tmp_path/'run'
    root.mkdir()
    if legacy:
        (root/'subject-runtime.json').write_bytes(canonical(dict(target='a3', max_seconds=7200,
            settings_sha256=hashlib.sha256(canonical(settings)).hexdigest(),
            referenced_json_sha256=referenced_settings(settings))))
    def compose(*args, **options):
        result = bound(root)
        result['provider'] = SimpleNamespace(_transport=openai_compatible_transport)
        result['coordinator'].curriculum.target_policy = SimpleNamespace(admit=lambda _: None)
        result['check_environment'] = lambda **_: 'environment-checked'
        return result
    options = dict(target='a3', root=root, subjects=[Subject('s', 'copies')],
                   initial_memory=b'', compose=compose)
    campaign = build_subject_campaign(settings, **options)
    assert (campaign.verifier.session_factory is None) is legacy
    assert json.loads((root/'subject-runtime.json').read_text())['schema'] == 'subject-runtime-v2'
    assert campaign.check_environment() == 'environment-checked'
    calls = []
    campaign.verifier.review = lambda request, **kwargs: calls.append(request) or {'decision': 'repair'}
    item = dict(project_id='retained', spec=_project('retained', 'gm-ub-movement'))
    request = ProjectReview('retained', None, False, {})
    assert campaign.review_retained(item, request, max_seconds=10) == {'decision': 'repair'}
    assert calls == [request]
    native.write_text(json.dumps({'physical_device': 1, 'workspace': 'retained', 'compiler': 'pin'}))
    resumed = build_subject_campaign(settings, **options)
    assert (resumed.verifier.session_factory is None) is legacy
    allocations = json.loads((root/'execution-allocations.json').read_text())
    assert allocations[-1]['allocation']['worker']['physical_device'] == 1


@pytest.mark.parametrize('ambient', [None, 'wrong-ambient-key'])
def test_real_pi_gateway_uses_private_provider_credential(monkeypatch, ambient):
    from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    try:
        runtime_preflight()
    except RuntimeError:
        pytest.skip('Node22 and pinned Pi required')
    if ambient is None:
        monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    else:
        monkeypatch.setenv('DEEPSEEK_API_KEY', ambient)
    selected = next(p.backend_model for p in MODEL_IDENTITIES if p.backend_model.value == 'deepseek-flash')
    private = 'configured-private-provider-key'
    provider = DirectCompletionProvider({'DEEPSEEK_API_KEY': private}, models=[selected])
    calls = []
    actions = iter([*prerequisites(), decision(memory())])
    def http(*_, **options):
        calls.append(options)
        action = next(actions)
        name, args = ('submit_review', action) if 'decision' in action else next(iter(action.items()))
        return stream(name, args, len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post', http)
    create = session_factory({}, provider, SimpleNamespace(model='deepseek-flash'))
    h = Harness([])
    h.agent.session_factory = create
    assert h.agent.review(request(), max_seconds=15) == decision(memory())
    assert calls[0]['headers']['Authorization'] == 'Bearer ' + private
    assert private not in calls[0]['data'].decode()


@pytest.mark.parametrize('failure,expected', [
    ('provider', 'InfrastructureFailure'), ('context', 'InfrastructureFailure'),
    ('cancelled', 'TimeoutError'), ('deadline', 'TimeoutError'),
])
def test_production_review_boundary_normalizes_pi_failures(monkeypatch, failure, expected):
    from coding.pi.context import ContextLimitError
    from coding.pi.rpc import PiProviderError, PiCancelled, PiDeadlineError
    classes = {'provider': PiProviderError, 'context': ContextLimitError,
               'cancelled': PiCancelled, 'deadline': PiDeadlineError}
    class Failing:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def next_action(self, *args, **kwargs): raise classes[failure]('synthetic failure')
    monkeypatch.setattr('benchmarks.ascend.pi_project_verifier.PiReviewSession', Failing)
    h = Harness([])
    h.agent.session_factory = session_factory({'verifier_engine': 'pi'}, object(),
                                             SimpleNamespace(model='deepseek-flash'))
    with pytest.raises(Exception) as caught:
        h.agent.review(request(), max_seconds=10)
    assert type(caught.value).__name__ == expected
    assert not any(event['kind'] == 'decision' for event in h.agent.last_trace)




def test_real_pi_pages_aggregate_feedback_then_executes_required_reads(monkeypatch):
    from benchmarks.ascend.review_context import ReviewContextRecords
    from test_review_context import fixture
    try:
        runtime_preflight()
    except RuntimeError:
        pytest.skip('Node22 and pinned Pi required')
    h = Harness([])
    feedback = fixture()['host']['project']['rewrite_feedback']
    actions = iter([*prerequisites(), decision(memory())])
    restored, reference, reading = [], [None], [True]
    calls = []
    def provider(*_, **options):
        body = json.loads(options['data'])
        calls.append(body)
        assert len(options['data']) < 49152
        if reference[0] is None:
            snapshot = json.loads(next(m['content'] for m in body['messages'] if m['role'] == 'user'))
            host = snapshot['context']['host']
            pointer = host['project']['rewrite_feedback']['prior_draft']
            reference[0] = pointer['record']
            assert host['review_state']['missing_required_artifacts']
            assert host['review_state']['host_checks_passed'] is True
            return stream('read_context', {'record': reference[0], 'offset': 0, 'limit': 4096}, len(calls))
        if reading[0]:
            message = next(m for m in reversed(body['messages']) if m['role'] == 'tool')
            content = message['content']
            if isinstance(content, list):
                content = ''.join(p.get('text', '') for p in content)
            page = json.loads(content)
            restored.append(page['text'])
            if page['next_offset'] is not None:
                return stream('read_context', {'record': reference[0], 'offset': page['next_offset'], 'limit': 4096}, len(calls))
            reading[0] = False
        action = next(actions)
        name, args = ('submit_review', action) if 'decision' in action else next(iter(action.items()))
        return stream(name, args, len(calls))
    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    h.agent.session_factory = lambda **options: PiCodingSession(model='deepseek-flash', schemas=SCHEMAS,
        environ={'DEEPSEEK_API_KEY': 'fake-provider-secret'}, system_prompt=SYSTEM,
        deadline=time.monotonic()+options['max_seconds'], first_action_only=True,
        context_factory=ReviewContextRecords)
    assert h.agent.review(request(), feedback=feedback, max_seconds=30) == decision(memory())
    assert json.loads(''.join(restored)) == feedback['prior_draft']
    assert [digest for digest, _ in h.reads[:3]] == [S, R, L]
    assert len(h.queries) == 1
