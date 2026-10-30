from dataclasses import replace
import hashlib
import json

import pytest

from rsi.contracts import BudgetLimits, CycleSpec, Mode, ProjectSpec
from rsi.cycles import CycleScheduler, HostTaskResult, IsolationSnapshot, SimplificationResult, SnapshotUpdate
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
        revised = replace(project, project_id=project.project_id + '-simple', goal='smaller goal')
        return SimplificationResult(revised, 0, 'host-only-revision')

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


def test_every_failed_original_is_simplified_after_prior_recovery_exhaustion(tmp_path):
    calls = []
    def run(project, memory, skills, ctx):
        calls.append(('run', project.project_id))
        return HostTaskResult('failed', 'observed-failure', 1)
    def simplify(project, result, ctx):
        calls.append(('simplify', project.project_id))
        return SimplificationResult(replace(project, project_id=project.project_id+'-small'),
                                    0, 'host-simplification-receipt')
    scheduler = CycleScheduler(tmp_path/'state', spec(), run,
        lambda *_: SnapshotUpdate('updated-memory'), simplify=simplify, recovery_limit=1)
    try:
        finish(scheduler)
        assert calls == [('run', '0'), ('run', '1'), ('simplify', '0'),
                         ('run', '0-small'), ('simplify', '1'), ('run', '1-small')]
        assert [r['original_id'] for r in scheduler.state['recoveries']] == ['0', '1']
        assert len([e for e in scheduler.state['events'] if e['kind']=='recovery_exhausted']) == 2
        assert scheduler.state['spent'] == 4
    finally:
        scheduler.close()


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


def isolation(root, memory, skills, namespace):
    (root / 'memory').mkdir(parents=True)
    (root / 'memory/active.json').write_text('{}')
    (root / 'learning.json').write_text('{}')
    data = json.dumps(dict(namespace=namespace, memory=memory, skills=skills,
                           parent=str(root.parent / 'parent-memory'))).encode()
    (root / 'isolation.json').write_bytes(data)
    return IsolationSnapshot(namespace, memory, skills, str(root), hashlib.sha256(data).hexdigest())


def test_inference_isolation_and_recovery_disabled(tmp_path):
    seen = []

    def run(project, memory, skills, ctx):
        seen.append((memory, skills, ctx['namespace']))
        return HostTaskResult('failed', 'r', 1)

    s = CycleScheduler(tmp_path / 'state', spec(Mode.INFERENCE, 1), run,
                       lambda *a: SnapshotUpdate('isolated-memory1'),
                       skill_update=lambda *a: SnapshotUpdate('isolated-candidate'),
                       isolate=lambda m, sk, ctx: isolation(tmp_path / 'fork', m, sk, ctx['namespace']),
                       recover=False)
    finish(s)
    assert seen[0][:2] == ('memory0', 'skills0')
    assert s.state['isolation']['branch_root'] == str(tmp_path / 'fork')
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


def test_isolation_requires_receipt_and_valid_namespace(tmp_path):
    s = CycleScheduler(tmp_path / 'state', spec(Mode.INFERENCE, 1), None, None,
                       skill_update=lambda *a: None,
                       isolate=lambda *a: ('different-memory', 'different-skills'))
    with pytest.raises(ValueError):
        s.step()
    operation = s.state['pending']['id']
    receipt = isolation(tmp_path / 'fork', 'memory0', 'skills0', 'wrong-namespace')
    with pytest.raises(ValueError):
        s.reconcile(operation, receipt)
    assert s.state['phase'] == 'isolate'
    assert s.state['pending']['id'] == operation
    s.close()


def test_extensions_persist_and_charge_once_without_promoting_candidates(tmp_path):
    path, cycle = tmp_path / 'state', spec(Mode.PHASE2, 1)
    kwargs = dict(runner=lambda *a: HostTaskResult('accepted', 'r', 1),
                  memory_update=lambda *a: SnapshotUpdate('m1'),
                  skill_update=lambda *a: SnapshotUpdate('candidate'), actor_seconds=20)
    s = CycleScheduler(path, cycle, **kwargs)
    finish(s)
    events = s.extension_events()
    validation = next(e for e in events if e['kind'] == 'candidate_validation_due')
    assert validation['incumbent'] == 'skills0' and validation['candidate'] == 'candidate'
    s.enqueue_tuning('manual', {'user_guidance': 'prioritise numeric diagnostics'})
    s.enqueue_tuning('manual', {'user_guidance': 'prioritise numeric diagnostics'})
    with pytest.raises(ValueError):
        s.enqueue_tuning('manual', {'project_ref': 'different'})
    s.complete_extension(validation['id'], 'retained-evaluation', actor_seconds=3, recovery=True)
    s.complete_extension(validation['id'], 'retained-evaluation', actor_seconds=3, recovery=True)
    assert s.state['skills'] == 'skills0'
    assert s.remaining()['actor_seconds'] == 16
    assert s.state['recovery_spent'] == 3
    with pytest.raises(ValueError):
        s.complete_extension(validation['id'], 'different')
    s.publish_validated_skills('skills0', 'validated')
    s.publish_validated_skills('skills0', 'validated')
    with pytest.raises(WorkflowError):
        s.publish_validated_skills('stale', 'other')
    s.close()
    s = CycleScheduler(path, cycle, **kwargs)
    assert s.state['skills'] == 'validated'
    assert s.remaining()['actor_seconds'] == 16
    assert validation['id'] not in {e['id'] for e in s.extension_events()}
    assert sum(e['id'] == 'manual' for e in s.extension_events()) == 1
    s.close()


def test_phase1_refuses_skill_extensions(tmp_path):
    s = CycleScheduler(tmp_path / 'state', spec(), None, None)
    with pytest.raises(WorkflowError):
        s.enqueue_tuning('manual', {})
    with pytest.raises(WorkflowError):
        s.publish_validated_skills('skills0', 'other')
    assert s.state['skills'] == 'skills0'
    s.close()


def test_isolation_receipt_rejects_parent_mutable_head_reuse(tmp_path):
    receipt = isolation(tmp_path / 'fork', 'memory0', 'skills0', 'namespace')
    proof_path = tmp_path / 'fork/isolation.json'
    proof = json.loads(proof_path.read_text())
    proof['parent'] = str(tmp_path / 'fork/memory')
    data = json.dumps(proof).encode()
    proof_path.write_bytes(data)
    receipt = replace(receipt, receipt_ref=hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError):
        receipt.verify()


def test_simplification_elapsed_receipt_reconciles_once_after_restart(tmp_path):
    captured, runs = [], []

    def run(project, memory, skills, ctx):
        runs.append(project.budgets.actor_seconds)
        return HostTaskResult('accepted' if ctx['recovery'] else 'failed', 'r', 1 if ctx['recovery'] else 2)

    def simplify(project, result, ctx):
        assert ctx['remaining']['actor_seconds'] == 8
        assert ctx['remaining']['recovery_seconds'] == 6
        assert project.budgets.actor_seconds == 6
        captured.append((ctx['operation_id'], SimplificationResult(
            replace(project, project_id='revised'), 2.5, 'host-timing-receipt')))
        raise RuntimeError('lost callback return; host receipt retained')

    path, cycle = tmp_path / 'state', spec(count=1)
    kwargs = dict(runner=run, memory_update=lambda *a: SnapshotUpdate('m'), simplify=simplify,
                  actor_seconds=10, recovery_seconds=6)
    s = CycleScheduler(path, cycle, **kwargs)
    s.step()
    s.step()
    with pytest.raises(RuntimeError):
        s.step()
    s.close()
    s = CycleScheduler(path, cycle, **kwargs)
    with pytest.raises(WorkflowError):
        s.step()
    operation, receipt = captured[0]
    s.reconcile(operation, receipt)
    with pytest.raises(WorkflowError):
        s.reconcile(operation, receipt)
    assert s.remaining()['actor_seconds'] == 5.5
    assert s.remaining()['recovery_seconds'] == 3.5
    revision = next(e for e in s.state['events'] if e['kind'] == 'project_revised')
    assert revision['receipt_ref'] == 'host-timing-receipt' and revision['actor_seconds'] == 2.5
    s.close()
    s = CycleScheduler(path, cycle, **kwargs)
    finish(s)
    assert len(captured) == 1 and runs == [10, 3.5]
    assert s.remaining()['actor_seconds'] == 4.5
    assert s.remaining()['recovery_seconds'] == 2.5
    s.close()


def test_simplification_exhausts_recovery_before_execution(tmp_path):
    s = CycleScheduler(tmp_path / 'state', spec(count=1),
                       lambda *a: HostTaskResult('failed', 'r', 1),
                       lambda *a: SnapshotUpdate('m'), recovery_seconds=2,
                       simplify=lambda project, result, ctx: SimplificationResult(
                           replace(project, project_id='revised'), 2, 'host-timing-receipt'))
    finish(s)
    assert not s.state['recoveries']
    assert s.state['spent'] == 3 and s.state['recovery_spent'] == 2
    assert any(e['kind'] == 'recovery_exhausted' for e in s.state['events'])
    s.close()


def test_bare_simplification_spec_is_not_silently_free(tmp_path):
    s = CycleScheduler(tmp_path / 'state', spec(count=1),
                       lambda *a: HostTaskResult('failed', 'r', 1),
                       lambda *a: SnapshotUpdate('m'),
                       simplify=lambda project, result, ctx: replace(project, project_id='revised'))
    s.step()
    s.step()
    with pytest.raises(TypeError):
        s.step()
    assert s.state['pending']['kind'] == 'simplify'
    assert s.state['phase'] == 'simplify' and not s.state['recoveries']
    s.close()
