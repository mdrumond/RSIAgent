"""Real NOOA PredictStrategy over the registered direct-provider boundary."""
import importlib.util
import json
import subprocess
import sys

import pytest

from rsi.roles import AdmittedEvidence, NoncodingRoles

@pytest.fixture(autouse=True)
def optional_framework(request):
    if (not request.node.name.startswith('test_isolated')
            and request.node.name != 'test_core_module_import_does_not_load_optional_framework'
            and importlib.util.find_spec('nooa') is None):
        pytest.skip('install rsi[roles] on Python 3.12/3.13')


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


def make_roles(responses, clock=None):
    from benchmarks.a5kernels.phase1_provider import DirectCompletion, DirectCompletionProvider, direct_profile
    from benchmarks.a5kernels.phase1_experiments import A5BackendModel, MODEL_IDENTITIES

    calls = []
    events = []
    clock = clock or Clock()

    def transport(**kwargs):
        calls.append(kwargs)
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        clock.now += 2
        return DirectCompletion(json.dumps(value), 20)

    model = A5BackendModel.DEEPSEEK_FLASH
    provider = DirectCompletionProvider({'DEEPSEEK_API_KEY': 'test-only-secret'},
                                        transport=transport, models=[model], clock=clock)
    profile = direct_profile(next(x for x in MODEL_IDENTITIES if x.backend_model == model))
    roles = NoncodingRoles(provider, profile, clock=clock,
                          observer=lambda event, payload: events.append((event, payload)))
    return roles, calls, events, clock


def test_curriculum_runs_real_predict_and_preserves_registered_route():
    roles, calls, events, _ = make_roles([{
        'rationale': 'exercise numeric repair', 'evidence_ids': ['failure'],
        'projects': [{'name': 'stable_sum', 'goal': 'accurate cancellation',
                      'acceptance': 'match decimal reference'}]}])
    result = roles.curriculum('Construct a CPU curriculum',
                              [AdmittedEvidence('failure', {'error': 1.0})], deadline=110)
    assert result.projects[0].name == 'stable_sum'
    assert result.evidence_ids == ('failure',)
    assert len(calls) == 1
    assert calls[0]['base_url'] == 'https://api.deepseek.com'
    assert calls[0]['request']['model'] == 'deepseek-flash'
    assert 0 < calls[0]['request']['timeout'] <= 10
    wire = json.dumps(calls[0]['request']['messages'])
    assert 'test-only-secret' not in wire
    assert [e for e, _ in events].count('role_finished') == 1
    assert events[-1][1]['elapsed_seconds'] == 2
    assert events[0][1]['evidence'] == [{'evidence_id': 'failure', 'data': {'error': 1.0}}]


def test_skillwriter_returns_proposal_without_applying_it():
    roles, _, _, _ = make_roles([{
        'rationale': 'detect regression', 'evidence_ids': ['run'],
        'edits': [{'skill': 'debug', 'operation': 'replace', 'content': 'Check reference.'}]}])
    proposal = roles.skillwriter('Improve detection', [AdmittedEvidence('run', {})], deadline=110)
    assert proposal.edits[0].operation == 'replace'
    assert proposal.edits[0].skill == 'debug'


def test_review_allows_no_actionable_findings():
    roles, _, _, _ = make_roles([{'rationale': 'no regression', 'evidence_ids': [], 'findings': []}])
    assert roles.review('Review', [], deadline=110).findings == ()


@pytest.mark.parametrize('bad', [
    {'rationale': 'x', 'evidence_ids': [], 'findings': [], 'approved': True},
    {'rationale': 'x', 'evidence_ids': [], 'findings': [{'severity': 'urgent', 'location': 'a:1', 'issue': 'bug'}]},
    {'rationale': 'x', 'evidence_ids': ['invented'], 'findings': []},
])
def test_invalid_proposal_does_not_retry_or_escape_host_gate(bad):
    roles, calls, events, _ = make_roles([bad])
    with pytest.raises(Exception):
        roles.review('Review', [AdmittedEvidence('real', {})], deadline=110)
    assert len(calls) == 1
    assert events[-1][1]['success'] is False


def test_deadline_exhausted_before_dispatch():
    roles, calls, _, _ = make_roles([])
    with pytest.raises(TimeoutError):
        roles.review('Review', [], deadline=100)
    assert calls == []


def test_late_provider_result_cannot_be_accepted():
    roles, calls, events, _ = make_roles([{'rationale': 'ok', 'evidence_ids': [], 'findings': []}])
    with pytest.raises(TimeoutError):
        roles.review('Review', [], deadline=101)
    assert len(calls) == 1
    assert events[-1][1]['success'] is False


def test_duplicate_evidence_rejected_before_dispatch():
    roles, calls, _, _ = make_roles([])
    with pytest.raises(ValueError):
        roles.review('Review', [AdmittedEvidence('a', {}), AdmittedEvidence('a', {})], deadline=110)
    assert calls == []


def test_core_module_import_does_not_load_optional_framework():
    subprocess.run([sys.executable, '-c',
                    "import sys; import rsi.roles; assert 'nooa' not in sys.modules; assert 'pydantic' not in sys.modules"],
                   check=True)


def test_framework_setup_does_not_consume_actor_allowance(monkeypatch):
    import rsi.roles as module
    roles, calls, events, clock = make_roles([{'rationale': 'ok', 'evidence_ids': [], 'findings': []}])
    original = module._framework

    def setup():
        framework = original()
        clock.now += 7
        return framework

    monkeypatch.setattr(module, '_framework', setup)
    roles.review('Review', [], deadline=110)
    assert calls[0]['request']['timeout'] == 10
    assert events[-1][1]['elapsed_seconds'] == 2


def test_provider_failure_is_observed_and_never_falls_back():
    roles, calls, events, _ = make_roles([RuntimeError('transport failed')])
    with pytest.raises(Exception):
        roles.review('Review', [], deadline=110)
    assert len(calls) == 1
    assert events[-1][1]['success'] is False


def test_active_async_loop_rejected_before_dispatch():
    import asyncio
    roles, calls, _, _ = make_roles([])

    async def nested():
        with pytest.raises(RuntimeError):
            roles.review('Review', [], deadline=110)

    asyncio.run(nested())
    assert calls == []


def isolated(monkeypatch, script, *, allowance=5, setup_seconds=5):
    """Use a real subprocess with scripted receipts, never the network."""
    import os
    import rsi.roles as module
    roles, _, _, _ = make_roles([])
    original = subprocess.Popen
    captured = []
    events = []

    def launch(command, **kwargs):
        captured.append((command, kwargs['env']))
        return original([sys.executable, '-c', script], **kwargs)

    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    adapter = module.IsolatedRoles(sys.executable, roles.profile,
        observer=lambda e, p: events.append((e, p)),
        environment={**os.environ, 'DEEPSEEK_API_KEY': 'test-only-secret',
                     'OPENAI_API_KEY': 'must-not-inherit'}, setup_seconds=setup_seconds)
    return adapter, events, captured


WORKER = '''
import json, sys, time
request = json.loads(sys.stdin.readline())
def emit(event, payload):
    print(json.dumps(dict(event=event, payload=payload)), flush=True)
time.sleep(0.05)
emit('role_started', {'role': 'review', 'provider_identity': IDENTITY})
time.sleep(0.02)
emit('role_finished', {'role': 'review', 'success': True, 'elapsed_seconds': 0.02})
emit('result', {'rationale': 'checked', 'evidence_ids': ['run'], 'findings': []})
'''


def worker_script():
    roles, _, _, _ = make_roles([])
    return WORKER.replace('IDENTITY', repr(roles.profile.identity.as_dict()))


def test_isolated_validates_real_process_receipts_and_controls_environment(monkeypatch):
    import time
    adapter, events, captured = isolated(monkeypatch, worker_script())
    result = adapter.review('review', [AdmittedEvidence('run', {})], deadline=time.monotonic() + 5)
    assert result.findings == ()
    env = captured[0][1]
    assert env['DEEPSEEK_API_KEY'] == 'test-only-secret'
    assert 'OPENAI_API_KEY' not in env
    assert captured[0][0][-2:] == ['-m', 'rsi.roles']
    assert [p['elapsed_seconds'] for e, p in events if e == 'role_finished'] == [0.02]


@pytest.mark.parametrize('replacement', [
    ("'evidence_ids': ['run']", "'evidence_ids': ['invented']"),
    ("'elapsed_seconds': 0.02", "'elapsed_seconds': -1"),
    ("'elapsed_seconds': 0.02", "'elapsed_seconds': float('nan')"),
    ("'success': True", "'success': False"),
])
def test_isolated_rejects_invalid_proposals_and_receipts(monkeypatch, replacement):
    import time
    script = worker_script().replace(*replacement)
    adapter, _, _ = isolated(monkeypatch, script)
    with pytest.raises((ValueError, RuntimeError)):
        adapter.review('review', [AdmittedEvidence('run', {})], deadline=time.monotonic() + 5)


def test_isolated_enforces_timeout_without_retry_and_marks_missing_accounting(monkeypatch):
    import time
    adapter, events, captured = isolated(monkeypatch,
        'import time; time.sleep(5)', setup_seconds=0.05)
    with pytest.raises(TimeoutError):
        adapter.review('review', [], deadline=time.monotonic() + 5)
    assert len(captured) == 1
    assert events[-1][0] == 'accounting_incomplete'


def test_isolated_never_publishes_secret_output(monkeypatch):
    import time
    script = "import sys; sys.stdin.readline(); print('test-only-secret', flush=True)"
    adapter, events, _ = isolated(monkeypatch, script)
    with pytest.raises(ValueError):
        adapter.review('review', [], deadline=time.monotonic() + 5)
    assert 'test-only-secret' not in json.dumps(events)
