from dataclasses import replace

import pytest

from rsi.contracts import BudgetLimits, CycleSpec, Mode, ProjectSpec
from rsi.cycles import CycleScheduler, HostTaskResult, SnapshotUpdate
from rsi.runtime import WorkflowError


def spec(mode=Mode.PHASE1, count=2):
    return CycleSpec('cycle', mode, tuple(ProjectSpec(str(i), 'research', ('python', 'main.py'))
                                         for i in range(count)), 'memory0', 'skills0')


def finish(scheduler):
    for _ in range(100):
        if scheduler.state['phase'] in ('complete', 'wave_limit'):
            return
        scheduler.step()
    pytest.fail('scheduler did not terminate')


def test_phase1_memory_before_linked_recovery_and_fixed_skills(tmp_path):
    calls = []

    def run(project, memory, skills, context):
        calls.append(('run', project.project_id, memory, skills, context['recovery']))
        return HostTaskResult('accepted' if context['recovery'] else 'failed', 'result', 2)

    def memory(previous, rows, context):
        calls.append(('memory', len(rows)))
        return SnapshotUpdate('memory1')

    def simplify(project, result, context):
        assert result['outcome'] == 'failed'
        return replace(project, project_id=project.project_id + '-simple', goal='smaller goal')

    s = CycleScheduler(tmp_path / 'state.json', spec(), run, memory, simplify=simplify,
                       skill_update=lambda *a: pytest.fail('Phase1 must not update skills'))
    finish(s)
    assert calls == [('run', '0', 'memory0', 'skills0', False),
                     ('run', '1', 'memory0', 'skills0', False), ('memory', 2),
                     ('run', '0-simple', 'memory1', 'skills0', True),
                     ('run', '1-simple', 'memory1', 'skills0', True)]
    assert [r['outcome'] for r in s.state['originals']] == ['failed', 'failed']
    assert [r['original_id'] for r in s.state['recoveries']] == ['0', '1']
    assert len([e for e in s.state['events'] if e['kind'] == 'project_revised']) == 2
    assert s.state['skills'] == 'skills0'
    s.close()


def test_phase2_required_hook_and_coalesced_due_events(tmp_path):
    with pytest.raises(ValueError):
        CycleScheduler(tmp_path / 'missing', spec(Mode.PHASE2), None, None)
    actions = []

    def run(project, memory, skills, ctx):
        actions.append(('run', memory, skills))
        return HostTaskResult('failed', 'host-result', 1)

    def memory(previous, results, ctx):
        actions.append(('memory', len(results)))
        return SnapshotUpdate(previous + '+')

    def skill(previous, memory, result, ctx):
        actions.append(('skill', memory))
        return SnapshotUpdate('candidate')

    s = CycleScheduler(tmp_path / 'state', spec(Mode.PHASE2, 5), run, memory, skill_update=skill)
    finish(s)
    assert [a[0] for a in actions] == ['run', 'memory', 'skill'] * 5
    assert actions[3] == ('run', 'memory0+', 'skills0')
    assert len(s.state['originals']) == 5 and not s.state['recoveries']
    due = [e for e in s.state['events'] if e['kind'] == 'tuning_due']
    assert len(due) == 1 and set(due[0]['reasons']) == {'cadence', 'wave_end'}
    assert len([e for e in s.state['events'] if e['kind'] == 'paired_recovery_due']) == 5
    assert s.state['skills'] == 'skills0'
    s.close()


@pytest.mark.parametrize('kind', ['original', 'memory', 'skill'])
def test_crash_pending_hook_is_reconciled_never_redispatched(tmp_path, kind):
    calls = []

    def callback(name, result):
        def invoke(*args):
            calls.append(name)
            if name == kind:
                raise RuntimeError('controller lost result')
            return result
        return invoke

    kwargs = dict(runner=callback('original', HostTaskResult('accepted', 'r', 2)),
                  memory_update=callback('memory', SnapshotUpdate('memory1')),
                  skill_update=callback('skill', SnapshotUpdate('candidate')))
    path, cycle = tmp_path / 'state', spec(Mode.PHASE2, 1)
    s = CycleScheduler(path, cycle, **kwargs)
    while s.state['phase'] != kind:
        s.step()
    with pytest.raises(RuntimeError):
        s.step()
    operation = s.stop()
    s.close()
    reopened = CycleScheduler(path, cycle, **kwargs)
    reopened.resume()
    with pytest.raises(WorkflowError):
        reopened.step()
    receipt = HostTaskResult('accepted', 'r', 2) if kind == 'original' else SnapshotUpdate('retained')
    reopened.reconcile(operation['id'], receipt)
    assert calls.count(kind) == 1
    assert reopened.state['pending'] is None
    if kind == 'original':
        assert reopened.state['spent'] == 2 and reopened.state['cursor'] == 1
    reopened.close()


def test_disabled_recovery_cap_and_parent_time_resume(tmp_path):
    attempts, limits = [], []

    def run(project, memory, skills, ctx):
        attempts.append(ctx['recovery'])
        limits.append(project.budgets.actor_seconds)
        return HostTaskResult('accepted' if len(attempts) == 4 else 'failed', 'r', 2)

    kwargs = dict(runner=run, memory_update=lambda *a: SnapshotUpdate('m1'),
                  recovery_limit=None, actor_seconds=10, recovery_seconds=8, wave_limit=None)
    path, cycle = tmp_path / 'state', spec(count=1)
    s = CycleScheduler(path, cycle, **kwargs)
    s.step()
    s.close()
    s = CycleScheduler(path, cycle, **kwargs)
    finish(s)
    assert attempts == [False, True, True, True]
    assert limits == [10, 8, 6, 4]
    assert s.remaining()['actor_seconds'] == 2
    assert len(s.state['originals']) == 1
    s.close()


def test_recovery_caps_and_disabled_recovery(tmp_path):
    for recover in (False, True):
        s = CycleScheduler(tmp_path / str(recover), spec(count=1),
                           lambda *a: HostTaskResult('failed', 'r', 1),
                           lambda *a: SnapshotUpdate('m'), recover=recover, recovery_limit=2)
        finish(s)
        assert len(s.state['recoveries']) == (2 if recover else 0)
        s.close()


def test_inference_isolation_and_recovery_disabled(tmp_path):
    seen = []

    def run(project, memory, skills, ctx):
        seen.append((memory, skills, ctx['namespace']))
        return HostTaskResult('failed', 'r', 1)

    s = CycleScheduler(tmp_path / 'state', spec(Mode.INFERENCE, 1), run,
                       lambda *a: SnapshotUpdate('isolated-memory1'),
                       skill_update=lambda *a: SnapshotUpdate('isolated-candidate'),
                       isolate=lambda m, sk, ctx: ('isolated-memory0', 'isolated-skills0'),
                       recover=False)
    finish(s)
    assert seen[0][:2] == ('isolated-memory0', 'isolated-skills0')
    assert seen[0][2].startswith('inference:')
    assert not any(e['kind'] == 'paired_recovery_due' for e in s.state['events'])
    s.close()


def test_single_writer_stop_resume_and_wave_limit(tmp_path):
    kwargs = dict(runner=lambda *a: HostTaskResult('accepted', 'r', 1),
                  memory_update=lambda *a: SnapshotUpdate('m1'), wave_size=1, wave_limit=1)
    path, cycle = tmp_path / 'state', spec()
    s = CycleScheduler(path, cycle, **kwargs)
    with pytest.raises(WorkflowError):
        CycleScheduler(path, cycle, **kwargs)
    s.stop()
    assert s.step() is None and not s.state['originals']
    s.resume()
    finish(s)
    assert s.state['phase'] == 'wave_limit' and len(s.state['originals']) == 1
    s.close()
