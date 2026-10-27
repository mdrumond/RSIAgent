"""Production tool routing and real Pi argument exchanges with a fake provider."""
from types import SimpleNamespace
import json
import time

import pytest

from benchmarks.ascend.pi_project_verifier import session_factory, SCHEMAS, SYSTEM
from benchmarks.a5kernels.phase1_provider import openai_compatible_transport
from benchmarks.ascend.project_verifier import ProjectVerifierAgent
from coding.pi.session import PiCodingSession, runtime_preflight
from test_project_verifier_agent import Harness, prerequisites, decision, memory, request, L
from test_pi_runtime import stream


def test_production_defaults_pi_but_injected_transport_stays_compatible(monkeypatch):
    made = []
    monkeypatch.setattr('benchmarks.ascend.pi_project_verifier.PiCodingSession',
                        lambda **kwargs: made.append(kwargs) or kwargs)
    completion = SimpleNamespace(model='deepseek-flash')
    live = SimpleNamespace(_transport=openai_compatible_transport)
    assert session_factory({}, SimpleNamespace(_transport=object()), completion) is None
    assert session_factory({'verifier_engine': 'completion'}, live, completion) is None
    create = session_factory({}, live, completion)
    before = time.monotonic()
    create(max_seconds=7)
    assert made[0]['model'] == 'deepseek-flash'
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
    monkeypatch.setattr('benchmarks.ascend.pi_project_verifier.PiCodingSession',
                        lambda **kwargs: made.append(kwargs) or kwargs)
    campaign = build_subject_campaign({'verifier_engine': 'pi', 'verifier_model': 'deepseek-flash',
        'memory_verification': False}, target='a3', root=tmp_path,
        subjects=[Subject('s', 'copies')], initial_memory=b'', compose=lambda *a, **k: bound(tmp_path))
    campaign.verifier.session_factory(max_seconds=5)
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
    campaign = build_subject_campaign({'memory_verification': False}, **options)
    assert campaign.verifier.session_factory is None
    identity = tmp_path/'project-verifier-runtime.json'
    assert json.loads(identity.read_text())['engine'] == 'completion'
    live[0] = True
    with pytest.raises(ValueError, match='project verifier engine or model changed'):
        build_subject_campaign({'memory_verification': False}, **options)
    identity.unlink()  # Simulate a checkpoint written before engine metadata existed.
    legacy = build_subject_campaign({'memory_verification': False}, **options)
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
    campaign = build_subject_campaign({'memory_verification': False}, target='a3', root=tmp_path,
        subjects=[Subject('s', 'copies')], initial_memory=b'', compose=compose)
    assert campaign.verifier.session_factory is not None
    identity = json.loads((tmp_path/'project-verifier-runtime.json').read_text())
    assert identity['engine'] == 'pi'
    assert identity['runtime']['pi_version'] == '1.3.0'
