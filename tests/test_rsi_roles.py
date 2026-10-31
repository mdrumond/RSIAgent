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
            and not request.node.name.startswith('test_wall')
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
    assert env['PYTHONNOUSERSITE'] == '1'
    from pathlib import Path
    assert not Path(env['PYTHONPATH']).exists()
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
    from pathlib import Path
    assert not Path(captured[0][1]['PYTHONPATH']).exists()


def test_isolated_never_publishes_secret_output(monkeypatch):
    import time
    script = "import sys; sys.stdin.readline(); print('test-only-secret', flush=True)"
    adapter, events, _ = isolated(monkeypatch, script)
    with pytest.raises(ValueError):
        adapter.review('review', [], deadline=time.monotonic() + 5)
    assert 'test-only-secret' not in json.dumps(events)


def test_isolated_installed_wheel_excludes_host_dependencies(tmp_path):
    """Install a real wheel outside checkout; child keeps its own OpenAI SDK."""
    import os
    import venv
    from pathlib import Path
    worker_version = '2.0.0-worker-fixture'
    worker = tmp_path / 'worker'
    venv.EnvBuilder(with_pip=False).create(worker)
    worker_python = worker / 'bin' / 'python'
    site = Path(subprocess.check_output([str(worker_python), '-c',
        'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip())
    (site / 'openai').mkdir()
    (site / 'openai' / '__init__.py').write_text('__version__ = ' + repr(worker_version) + '\n')
    source = Path(__file__).resolve().parents[1]
    wheels, host = tmp_path / 'wheels', tmp_path / 'host-site'
    wheels.mkdir()
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps',
                    '--no-build-isolation', '--wheel-dir', str(wheels), str(source)],
                   check=True, capture_output=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                    '--target', str(host), str(next(wheels.glob('rsi-*.whl')))],
                   check=True, capture_output=True)
    sentinel = host / 'openai'
    sentinel.mkdir()
    (sentinel / '__init__.py').write_text("raise RuntimeError('host SDK leaked into worker')\n")
    probe = '''
import hashlib, json, pathlib, sys, time
import openai, rsi.roles
req = json.loads(sys.stdin.readline())
assert pathlib.Path.cwd() != pathlib.Path(HOST)
assert pathlib.Path(rsi.roles.__file__).read_bytes() == pathlib.Path(ORIGINAL).read_bytes()
assert openai.__version__ == EXPECTED_VERSION
def emit(event, payload):
    print(json.dumps(dict(event=event, payload=payload)), flush=True)
emit('role_started', {'role': 'review', 'provider_identity': IDENTITY})
emit('role_finished', {'role': 'review', 'success': True, 'elapsed_seconds': 0.0})
emit('result', {'rationale': openai.__version__, 'evidence_ids': [], 'findings': []})
'''
    parent = '''
import json, os, pathlib, subprocess, sys, time
import rsi.roles as module
from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
from benchmarks.a5kernels.phase1_provider import direct_profile
identity = next(x for x in MODEL_IDENTITIES if x.backend_model.value == 'deepseek-flash')
original = subprocess.Popen
projection = []
probe = %r
probe = probe.replace('HOST', repr(%r)).replace('ORIGINAL', repr(module.__file__)).replace('IDENTITY', repr(identity.as_dict())).replace('EXPECTED_VERSION', repr(%r))
def launch(command, **kwargs):
    projection.append(pathlib.Path(kwargs['cwd']))
    assert kwargs['env']['PYTHONNOUSERSITE'] == '1'
    assert kwargs['env']['PYTHONPATH'] != %r
    return original([%r, '-c', probe], **kwargs)
module.subprocess.Popen = launch
adapter = module.IsolatedRoles(sys.executable, direct_profile(identity), observer=lambda *args: None,
                               environment={'DEEPSEEK_API_KEY': 'test-only-secret'})
result = adapter.review('review', [], deadline=time.monotonic() + 10)
assert result.rationale == %r
assert not projection[0].exists()
print(json.dumps({'worker_openai': result.rationale, 'projection_removed': True}))
''' % (probe, str(host), worker_version, str(host), str(worker_python), worker_version)
    completed = subprocess.run([sys.executable, '-c', parent], cwd=tmp_path,
                               env={**os.environ, 'PYTHONPATH': str(host), 'PYTHONNOUSERSITE': '1'},
                               check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout)['projection_removed'] is True


def test_wall_framework_setup_exhaustion_never_starts_provider(monkeypatch):
    import rsi.roles as module
    roles, calls, events, clock = make_roles([])
    class Client:
        def __init__(self, owner, deadline):
            self.deadline = deadline
    class Agent:
        def __init__(self, llm):
            pass
    def framework():
        clock.now += 100
        return Agent, Client
    monkeypatch.setattr(module, '_framework', framework)
    with pytest.raises(TimeoutError, match='wall'):
        roles.skillwriter('write', [], deadline=110, wall_deadline=105)
    assert calls == []
    assert not events  # No fabricated role_finished for setup-only work.


def test_wall_remaining_after_setup_bounds_provider_and_actor_receipt(monkeypatch):
    from types import SimpleNamespace
    import rsi.roles as module
    roles, calls, events, clock = make_roles([])
    observed = []
    class Client:
        def __init__(self, owner, deadline):
            self.owner, self.deadline = owner, deadline
    class Agent:
        def __init__(self, llm):
            self.llm = llm
        async def skillwriter(self, **kwargs):
            observed.append(self.llm.owner._remaining(self.llm.deadline))
            clock.now += 2
            return SimpleNamespace(model_dump=lambda **kwargs: dict(
                rationale='no changes', evidence_ids=[], edits=[]))
    def framework():
        clock.now += 3
        return Agent, Client
    monkeypatch.setattr(module, '_framework', framework)
    with pytest.raises(TimeoutError):
        roles.skillwriter('write', [], deadline=110, wall_deadline=105)
    assert observed == [2]
    receipts = [payload for event, payload in events if event == 'role_finished']
    assert len(receipts) == 1
    assert receipts[0]['elapsed_seconds'] == 2
    assert receipts[0]['success'] is False


@pytest.mark.parametrize('finished_before_timeout', [False, True])
def test_isolated_wall_kills_slow_worker_without_fabricated_finished(monkeypatch, tmp_path, finished_before_timeout):
    from contextlib import contextmanager
    import time
    import rsi.roles as module
    @contextmanager
    def projection(check=lambda: None):
        check()
        yield tmp_path
        check()
    monkeypatch.setattr(module, '_machinery_projection', projection)
    script = worker_script().split("time.sleep(0.02)")[0]
    if finished_before_timeout:
        script += "emit('role_finished', {'role': 'review', 'success': True, 'elapsed_seconds': 0.0})\n"
    script += 'time.sleep(5)\n'
    adapter, events, captured = isolated(monkeypatch, script)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        adapter.review('review', [], deadline=started + 10, wall_deadline=started + 0.3)
    assert time.monotonic() - started < 2
    assert len(captured) == 1
    receipts = [payload for event, payload in events if event == 'role_finished']
    assert len(receipts) == int(finished_before_timeout)
    assert any(event == 'accounting_incomplete' for event, _ in events) is not finished_before_timeout


def test_isolated_wall_projection_exhaustion_never_spawns(monkeypatch, tmp_path):
    from contextlib import contextmanager
    import rsi.roles as module
    now = [100.0]
    @contextmanager
    def projection(check=lambda: None):
        now[0] += 6
        check()
        yield tmp_path
    monkeypatch.setattr(module, '_machinery_projection', projection)
    adapter, events, captured = isolated(monkeypatch, 'raise AssertionError("must not run")')
    adapter.clock = lambda: now[0]
    with pytest.raises(TimeoutError, match='wall'):
        adapter.review('review', [], deadline=110, wall_deadline=105)
    assert captured == []
    assert events == []


def test_skillwriter_real_predict_accepts_typed_append_without_normalisation():
    roles, calls, events, _ = make_roles([{
        'rationale': 'prevent recurrence', 'evidence_ids': ['numeric-failure'],
        'edits': [{'skill': 'debug', 'operation': 'append', 'content': 'Retest cancellation.'}]}])
    proposal = roles.skillwriter('Improve detection', [AdmittedEvidence('numeric-failure', {})], deadline=110)
    assert proposal.edits[0].operation == 'append'
    assert proposal.edits[0].content == 'Retest cancellation.'
    assert len(calls) == 1
    emitted = [payload for event, payload in events if event == 'role_proposal']
    assert emitted[0]['proposal']['edits'] == [dict(skill='debug', operation='append', content='Retest cancellation.')]


def test_isolated_finished_failure_retains_detail_scope_and_known_elapsed(monkeypatch):
    import time
    from rsi.roles import RoleFailure
    detail = {'type': 'ValueError', 'message': 'negative_decisions[0].redirection is required'}
    script = worker_script().replace("'success': True", "'success': False")
    script = script.replace("'elapsed_seconds': 0.02}", "'elapsed_seconds': 0.02, 'error': " + repr(detail) + "}")
    adapter, events, _ = isolated(monkeypatch, script)
    with pytest.raises(RoleFailure) as error:
        adapter.review('review', [AdmittedEvidence('run', {})], deadline=time.monotonic() + 5)
    assert error.value.role == 'review'
    assert error.value.error_type == detail['type']
    assert error.value.message == detail['message']
    finished = [payload for event, payload in events if event == 'role_finished']
    assert len(finished) == 1 and finished[0]['elapsed_seconds'] == 0.02
    assert finished[0]['error'] == detail
    assert not any(event == 'accounting_incomplete' for event, _ in events)


def test_isolated_legacy_failure_still_retains_elapsed(monkeypatch):
    import time
    adapter, events, _ = isolated(monkeypatch, worker_script().replace("'success': True", "'success': False"))
    with pytest.raises(RuntimeError):
        adapter.review('review', [], deadline=time.monotonic() + 5)
    assert [payload['elapsed_seconds'] for event, payload in events if event == 'role_finished'] == [0.02]
    assert not any(event == 'accounting_incomplete' for event, _ in events)


def test_isolated_error_detail_is_bounded_and_selected_secret_redacted():
    from rsi.roles import _error_details
    from uuid import uuid4
    secret = uuid4().hex
    value = _error_details(ValueError('field: ' + secret + '🙂' * 3000), secret)
    assert value['type'] == 'ValueError'
    assert len(value['message'].encode()) <= 2048
    assert secret not in json.dumps(value)



def test_actual_typed_validation_emits_specific_finished_failure_detail():
    payload = dict(rationale='Use retained guidance', evidence_ids=['query'],
        proposal=dict(name='small', goal='goal', acceptance='oracle', template_id='cpu.case-subset',
                      parameters={'case_indices': [0]}),
        negative_decisions=[dict(query_ref='query', index=0, action='used', reason='Reuse')])
    roles, calls, events, _ = make_roles([payload])
    with pytest.raises(ValueError) as error:
        roles.diagnostic('Choose a diagnostic', [AdmittedEvidence('query', {})], deadline=110)
    finished = [value for event, value in events if event == 'role_finished']
    assert len(calls) == 1 and len(finished) == 1
    assert finished[0]['success'] is False and finished[0]['elapsed_seconds'] == 2
    assert finished[0]['error']['type'] == type(error.value).__name__
    assert finished[0]['error']['message'] == str(error.value)
    assert 'negative_decisions[0].redirection' in str(error.value)
