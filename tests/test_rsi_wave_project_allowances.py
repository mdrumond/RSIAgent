"""Carried elapsed allocations survive wave admission, leases and resume."""
import json

import pytest

from rsi.configuration import set_configuration, show_configuration
from rsi.contracts import BudgetLimits, CycleSpec, Mode, ProjectSpec
from rsi.cycles import CycleScheduler
from rsi.project_wave import ProjectWave
from rsi.runtime import WorkflowError
from test_rsi_project_wave import processes
from test_rsi_services import instance
from test_rsi_service_wave import wave_instance
from test_rsi_worker_admission import profiles


@pytest.mark.parametrize('parent,expected', [(14452., [7192., 60., 7200.]), (7252., [7192., 60.]),
                                           (7191., [])])
def test_parent_reserves_actual_sum_without_rewriting_projects(tmp_path, parent, expected):
    projects = tuple(ProjectSpec(name, 'preserve budget', ('python', 'verify.py'),
        budgets=BudgetLimits(actor_seconds=seconds)) for name, seconds in
        [('carried', 7192.), ('small', 60.), ('fallback', None)])
    spec = CycleSpec('allowances', Mode.PHASE1, projects, 'memory', 'skills')
    class Reserved(Exception):
        pass
    def reserve(*args):
        scheduler.reserve_original_wave(3)
        raise Reserved()
    scheduler = CycleScheduler(tmp_path / 'cycle.json', spec, reserve, None,
                               wave_size=3, actor_seconds=parent)
    try:
        if expected:
            with pytest.raises(Reserved):
                scheduler.step()
            wave = scheduler.state['active_wave']
            assert [op['spec']['budgets']['actor_seconds'] for op in wave['operations']] == expected
            assert sum(expected) <= parent
            assert scheduler.reserve_original_wave(3) == wave
            assert all(op['id'] != op['original_id'] for op in wave['operations'])
            assert all(op['spec']['budgets'][key] is None for op in wave['operations']
                       for key in ('design_iterations', 'implementation_iterations', 'execution_tries'))
        else:
            with pytest.raises(WorkflowError):
                scheduler.step()
            assert not scheduler.state.get('active_wave')
    finally:
        scheduler.close()


def test_pool_preserves_base_elapsed_and_one_additive_recovery(tmp_path, monkeypatch):
    options = dict(max_parallel=1, actor_allowances={'operation': 7192.})
    def open_pool():
        return ProjectWave(tmp_path, ['operation'], ['cpu'], b'memory', b'skills', **options)
    pool = open_pool()
    requests, _ = processes(monkeypatch, failures=('operation',))
    pool.run(None)
    first = requests[0]
    assert first.actor_seconds == first.initial_actor_seconds == 7192.
    pool.reconcile('operation', lambda request, row: dict(closed=True,
        observation='owned harness closed', actor_seconds=17., resume_ready=True, harness_failure=True))
    pool.grant_harness_recovery('operation', observation='one authorized harness recovery')
    with pytest.raises(ValueError):
        pool.grant_harness_recovery('operation', observation='second extension forbidden')
    reopened = open_pool()
    more, _ = processes(monkeypatch, failures=('operation',))
    reopened.run(None)
    assert more[0].initial_actor_seconds == 7192.
    assert more[0].actor_seconds == 14392. and more[0].actor_elapsed == 17.
    assert more[0].reopen and more[0].recovery_receipt
    reopened.run(None)  # An unresolved dispatch must not replay or reset consumption.
    assert len(more) == 1 and reopened.status()['projects']['operation']['actor_seconds'] == 17.
    with pytest.raises(ValueError):
        ProjectWave(tmp_path, ['operation'], ['cpu'], b'memory', b'skills',
                    max_parallel=1, actor_allowances={'operation': 7200.})


def test_default_plan_reopens_without_new_allowance_metadata(tmp_path):
    pool = ProjectWave(tmp_path, ['legacy'], ['cpu'], b'memory', b'skills')
    state = json.loads(pool.path.read_text())
    assert 'actor_allowances' not in state['plan']
    reopened = ProjectWave(tmp_path, ['legacy'], ['cpu'], b'memory', b'skills',
                           actor_allowances={'legacy': None})
    assert reopened.status()['projects']['legacy']['allowance'] == 7200.


def strict_budget_bind(request, logical_admission, *, metadata, configuration, remaining):
    from test_rsi_lease_binding import cpu_bind_project
    expected = configuration['expected'][logical_admission['spec']['project_id']]
    assert logical_admission['spec']['budgets']['actor_seconds'] == expected
    assert request.actor_seconds == request.initial_actor_seconds == expected
    assert request.project_id != logical_admission['spec']['project_id']
    return cpu_bind_project(request, logical_admission, metadata=metadata,
                           configuration=configuration, remaining=remaining)


def test_real_service_children_keep_carried_spec_through_strict_host_binder(wave_instance, tmp_path):
    path = tmp_path / 'profiles.json'
    resolver = profiles(path)
    values = json.loads(path.read_text())
    expected = {'answer': 7192., 'other': 60., 'third': 7200.}
    for profile in values['profiles'].values():
        profile.update(project_binder='test_rsi_wave_project_allowances:strict_budget_bind',
            project_configuration=dict(coordination=str(tmp_path / 'order'), expected=expected))
    path.write_text(json.dumps(values))
    settings = show_configuration(wave_instance.root)['payload']
    settings['budgets'] = dict(design_iterations=None, implementation_iterations=None,
        execution_tries=None, actor_seconds=None, wall_seconds=None)
    first = settings['phase1_projects'][0]
    settings['phase1_projects'].append(dict(first, project_id='third', budgets=dict(first['budgets'])))
    settings['project_configs']['third'] = dict(settings['project_configs']['answer'])
    for spec in settings['phase1_projects']:
        spec['budgets']['actor_seconds'] = None if spec['project_id'] == 'third' else expected[spec['project_id']]
    settings['project_wave'].update(wave_size=3, devices=[
        dict(device=str(i), worker_profile=f'worker-{i}') for i in (0, 1)])
    settings['cycle_limits'].update(wave_size=3, actor_seconds=sum(expected.values()))
    set_configuration(wave_instance.root, settings)
    wave_instance.executor_factory = resolver
    cycle = wave_instance.create_cycle('carried-allowances', 'phase1')
    try:
        assert cycle.step().outcome == 'accepted'
        wave = cycle.scheduler.state['active_wave']
        rows = cycle.waves._pool(wave).status()['projects']
        workers = {}
        for op in wave['operations']:
            runner = cycle.open_project(op['id'])
            assert runner.spec.budgets.actor_seconds == expected[op['original_id']]
            assert runner.workflow.actor_allocation() == expected[op['original_id']]
            assert rows[op['id']]['receipt']['result']['status'] == 'accepted'
            workers[op['id']] = rows[op['id']]['worker']
        assert cycle.step().outcome == 'accepted'
        assert cycle.step().outcome == 'accepted'
        cycle.step()
        assert len(cycle.memory.records()) == 3
    finally:
        cycle.close()
    reopened = wave_instance.open_cycle('carried-allowances')
    try:
        retained = reopened.waves._pool(reopened.scheduler.state['active_wave'] or
                                       reopened.scheduler.state['wave_history'][0])
        retained.run(None)
        assert {op: row['worker'] for op, row in retained.status()['projects'].items()} == workers
    finally:
        reopened.close()
