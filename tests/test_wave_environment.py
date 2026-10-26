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
