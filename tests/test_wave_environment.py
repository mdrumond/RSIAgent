import json
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumAgent
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError
from benchmarks.ascend.wave_environment import NativeWaveEnvironmentCheck
from tests.test_a5_phase1_adaptive import SEED, _project, _result, _wave


class Transport:
    def __init__(self, *, wrong_output=False, stderr='', compile_error=None, exit_code=0):
        self.calls = []
        self.exit_code = exit_code
        self.remaining_seconds = None
        self.wrong_output, self.stderr, self.compile_error = wrong_output, stderr, compile_error

    def compile(self, plan, root):
        self.calls.append(('compile', plan, self.remaining_seconds()))
        if self.compile_error:
            raise self.compile_error
        return SimpleNamespace(library_sha256='a' * 64, stdout='compiled', stderr=self.stderr)

    def execute_receipt(self, compilation):
        self.calls.append(('execute',))
        plan = self.calls[0][1]
        expected = tuple(sum(xs) for xs in zip(*(t.values for t in plan.inputs)))
        return SimpleNamespace(output=(0,) if self.wrong_output else expected, job_handle='remote:fake:job:owned', exit_code=self.exit_code, stdout='', stderr='')


def config(root):
    return SimpleNamespace(root=root, target='a3', arch='dav-2201', execution_profile='gz-a3')


def reports(root):
    return [json.loads(p.read_text()) for p in root.glob('environment-checks/*/*/report.json')]


def test_probe_checks_each_worker_and_never_reuses_previous_compilation(tmp_path):
    workers = [Transport(), Transport()]
    gate = NativeWaveEnvironmentCheck(config(tmp_path), workers)
    gate(1, lambda: 17)
    first_ids = [t.calls[0][1].execution_id for t in workers]
    for t in workers:
        assert [c[0] for c in t.calls] == ['compile', 'execute']
        assert t.calls[0][2] == 17
        assert t.remaining_seconds is None
        t.calls.clear()
    gate(1, lambda: 12)
    assert all(t.calls[0][1].execution_id != old for t, old in zip(workers, first_ids))
    assert all(r['status'] == 'passed' for r in reports(tmp_path))


@pytest.mark.parametrize('kwargs', [dict(wrong_output=True), dict(stderr='sh: 28: source: not found\n'),
                                    dict(compile_error=TimeoutError('retained remote:fake:job:owned')), dict(exit_code=1)])
def test_probe_failure_retains_diagnostics_and_stops_worker_admission(tmp_path, kwargs):
    failed, untouched = Transport(**kwargs), Transport()
    with pytest.raises(ProjectInfrastructureError):
        NativeWaveEnvironmentCheck(config(tmp_path), (failed, untouched))(1, lambda: 10)
    assert untouched.calls == []
    assert failed.remaining_seconds is None
    report, = reports(tmp_path)
    assert report['status'] == 'failed'
    assert report['error_type'] in ('RuntimeError', 'TimeoutError')
    assert len(report['workers']) == 1
    assert report['workers'][0]['execution_id']


def coordinator(root, events, clock=lambda: 0):
    def select(messages):
        events.append('select')
        return _wave(messages[-1]['content'], [_project('copy', 'gm-ub-movement')])
    def execute(request):
        events.append('execute')
        return _result(request.spec, True, 'host verified')
    return AdaptivePhase1Coordinator(root, 'learn', SEED, CurriculumAgent(select), execute,
        max_parallel=1, project_target=1, max_seconds=20, clock=clock)


def test_environment_rejection_blocks_curriculum_and_charges_elapsed(tmp_path):
    events = [];now = [0]
    c = coordinator(tmp_path, events, lambda: now[0])
    def failed(index, remaining):
        events.append(('check', index, remaining()))
        now[0] += 7
        raise ProjectInfrastructureError('runtime setup failed')
    c.environment_check = failed
    with pytest.raises(ProjectInfrastructureError):c.propose_wave()
    assert events == [('check', 1, 20)]
    assert json.loads((tmp_path/'state.json').read_text())['elapsed_seconds'] == 7
    assert not (tmp_path/'pending.json').exists()


def test_approved_resume_rechecks_before_actor_and_does_not_select_again(tmp_path):
    events = [];c = coordinator(tmp_path, events)
    def check(index, remaining):events.append('check')
    c.environment_check = check
    proposal = c.propose_wave()
    c.approve_wave(proposal, ['copy'])
    c.run_approved_wave()
    assert events == ['check', 'select', 'check', 'execute']


def test_one_check_per_wave_in_automatic_run(tmp_path):
    events = [];c = coordinator(tmp_path, events)
    c.environment_check = lambda index, remaining: events.append('check')
    c.run()
    assert events == ['check', 'select', 'execute']


@pytest.mark.parametrize('retain_publication', [True, False])
def test_terminal_branch_recovery_does_not_require_live_environment(tmp_path, retain_publication):
    events = [];c = coordinator(tmp_path, events)
    c.environment_check = lambda index, remaining: events.append('check')
    proposal = c.propose_wave()
    c.approve_wave(proposal, ['copy'])
    finish = c._finish_wave
    def interrupted_publication(*args):
        raise OSError('supervisor interrupted after durable branch results')
    c._finish_wave = interrupted_publication
    with pytest.raises(OSError):c.run_approved_wave()
    c._finish_wave = finish
    if not retain_publication:
        (tmp_path/'waves/wave-001/published.json').unlink()
    def unavailable(index, remaining):
        raise ProjectInfrastructureError('device no longer available')
    c.environment_check = unavailable
    result = c.run_approved_wave()
    assert result['successful_projects'] == 1
    assert not (tmp_path/'pending.json').exists()
    assert events == ['check', 'select', 'check', 'execute']


def test_legacy_remote_calls_and_poll_waits_share_remaining_allocation(tmp_path):
    from subprocess import CompletedProcess
    from benchmarks.a3kernels.remote_candidate_bz import BZA3RemoteCandidateBackend
    from benchmarks.ascend.wave_environment import _transport_allocation
    now = [0];seen = []
    def runner(argv, **options):
        seen.append((argv, options['timeout']))
        now[0] += 3
        return CompletedProcess(argv, 0, '', '')
    def sleeper(seconds):
        seen.append(('sleep', seconds));now[0] += seconds
    transport = BZA3RemoteCandidateBackend(cpl_remote='/tools/cpl-remote',
        validation_wrapper='/tools/catlass-validation.sh', profile='bz-a3-1', remote_workspace='/owned',
        physical_device=0, state_directory=tmp_path, process_runner=runner)
    transport._sleep = sleeper
    old_call, old_timeout, old_sleep = transport._call, transport._timeout, transport._sleep
    with _transport_allocation(transport, lambda: 8-now[0]):
        transport._call(('/tools/catlass-validation.sh', '--timeout', '1200'))
        transport._call(('/tools/catlass-validation.sh', '--timeout', '1200'))
        transport._sleep(30)
        with pytest.raises(TimeoutError):transport._call(('/tools/catlass-validation.sh',))
    assert [row[1] for row in seen] == [8, 5, 2]
    assert [row[0][-1] for row in seen[:2]] == ['8', '5']
    assert transport._call == old_call and transport._timeout == old_timeout and transport._sleep == old_sleep


def test_curriculum_receives_only_time_remaining_after_environment_probe(tmp_path):
    now = [0];budgets = []
    def select(messages):
        context = json.loads(messages[-1]['content'].rsplit('\n', 1)[-1])
        budgets.append(context['remaining_seconds'])
        return _wave(messages[-1]['content'], [_project('copy', 'gm-ub-movement')])
    c = AdaptivePhase1Coordinator(tmp_path, 'learn', SEED, CurriculumAgent(select),
        lambda request: None, max_parallel=1, project_target=1, max_seconds=100, clock=lambda: now[0])
    def check(index, remaining):
        assert remaining() == 100
        now[0] += 80
    c.environment_check = check
    c.propose_wave()
    assert budgets == [20]
    assert json.loads((tmp_path/'state.json').read_text())['elapsed_seconds'] == 80


@pytest.mark.parametrize('failed', [False, True])
def test_usefulness_live_work_is_gated_but_retained_assessment_replay_is_probe_free(tmp_path, failed):
    from benchmarks.ascend.production import compose_adaptive_controller
    from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
    from tests.test_a5_phase1_adaptive import _memory
    events, requests, now = [], [], [0.0]
    def complete(profile, messages, **kwargs):
        events.append('provider')
        try:
            stage = json.loads(messages[-1]['content']).get('stage')
        except ValueError:
            stage = None
        return SimpleNamespace(text=stage or _wave(messages[-1]['content'], [_project('paired', 'gm-ub-movement')]))
    def execute(request):
        events.append('actor')
        requests.append(request)
        return _result(request.spec, True, 'host pass')
    _, runner = compose_adaptive_controller(root=tmp_path, objective='Learn', initial_memory=b'{}',
        executor=None, actor=None, profile=None, provider=SimpleNamespace(complete=complete),
        max_parallel=1, project_target=1, max_seconds=100, project_actor=execute, usefulness_enabled=True)
    runner.clock = lambda: now[0]
    def check(index, remaining):
        events.append('check')
        assert remaining() == 100
        now[0] += 7
        if failed:
            raise ProjectInfrastructureError('environment unavailable')
    runner.environment_check = check
    memory = _memory(ProjectSpec.from_mapping(_project('source', 'gm-ub-movement')))
    def assess():
        return runner.usefulness_verifier.assess_memory(memory, b'{}', 'source')[0]
    result = runner._with_allocation(assess)
    assert result['outcome'] == ('inconclusive' if failed else 'both-succeeded')
    assert events[0] == 'check'
    if failed:
        assert events == ['check'] and requests == []
    else:
        assert len(requests) == 2 and all(0 < request.remaining_seconds <= 93 for request in requests)
    before = list(events)
    runner.environment_check = lambda *args: pytest.fail('retained assessment must not probe')
    assert runner._with_allocation(assess)['experiment_id'] == result['experiment_id']
    assert events == before


@pytest.mark.parametrize('failed', [False, True])
def test_original_retry_live_work_is_gated_but_completed_publication_replay_is_probe_free(tmp_path, failed):
    from benchmarks.ascend.automatic_remediation import AutomaticRemediation
    from tests.test_generic_failure_feedback import phase1
    phase1(tmp_path / 'source', spec=_project('original', 'gm-ub-movement'))
    events, requests, now = [], [], [0.0]
    def execute(request):
        events.append('actor')
        requests.append(request)
        return _result(request.spec, True, 'host pass')
    runner = AdaptivePhase1Coordinator(tmp_path / 'run', 'Repair', b'{}',
        CurriculumAgent(lambda _: pytest.fail('retry must not select')), execute,
        max_parallel=1, project_target=1, max_seconds=100, clock=lambda: now[0])
    automatic = AutomaticRemediation(runner, target='a3', sources=[dict(type='phase1', root=str(tmp_path / 'source'))])
    state = runner._load_state()
    automatic.refresh(state)
    original, = automatic.store.records()
    automatic.store.update_investigation(original['feedback_id'], status='ready-to-retry', evidence=['retained probe'])
    automatic.refresh(state)
    def check(index, remaining):
        events.append('check')
        assert remaining() == 100
        now[0] += 7
        if failed:
            raise ProjectInfrastructureError('environment unavailable')
    runner.environment_check = check
    def retry():
        automatic.retry_ready(state)
        return {}
    if failed:
        with pytest.raises(ProjectInfrastructureError):
            runner._with_allocation(retry)
        assert events == ['check'] and requests == []
    else:
        runner._with_allocation(retry)
        assert events == ['check', 'actor']
        assert 0 < requests[0].remaining_seconds <= 93
        directory = requests[0].branch_root.parents[3]
        row = automatic.feedback.selection['selected'][0]
        runner.environment_check = lambda *args: pytest.fail('completed retry must not probe')
        runner._with_allocation(lambda: (automatic._phase1_retry(row, directory, state) or {}))
        assert events == ['check', 'actor']
