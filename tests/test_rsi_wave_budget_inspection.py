"""Effective allocations come from the runner, not the scheduler ceiling."""
import json

import pytest

from rsi.contracts import BudgetLimits
from rsi.runtime import Workflow
from rsi.wave_inspection import inspect_waves
from dashboard.rsi_ledger import inspection


@pytest.fixture
def retained(tmp_path):
    cycle = tmp_path/'runs/cycle'
    wave = cycle/'waves/wave'
    runner = wave/'projects/op/runner'
    runner.mkdir(parents=True)
    (cycle/'cycle.json').write_text(json.dumps(dict(active_wave=dict(id='wave', phase='executing',
        memory='memory', skills='skills', operations=[dict(id='op', original_id='project')]))))
    (wave/'wave.json').write_text(json.dumps(dict(projects=dict(op=dict(
        status='pending', allowance=6300., actor_seconds=0., device='0')))))
    return tmp_path, cycle, runner


def project(cycle):
    return inspect_waves(cycle)[0]['projects'][0]


def workflow(runner, cap=3593):
    return Workflow(runner/'workflow.json', None, source_hash='frozen',
                    limits=BudgetLimits(actor_seconds=cap, wall_seconds=None), clock=lambda: 100.)


@pytest.mark.parametrize('cap', [3593, 3598])
def test_frozen_allocation_and_union_override_stale_wave_counters(retained, cap):
    _, cycle, runner = retained
    flow = workflow(runner, cap)
    flow.state['intervals'] = dict(thinking=['thinking', 100., 110.], compilation=['compilation', 105., 115.])
    flow._save()
    before = {p: p.read_bytes() for p in cycle.rglob('*.json')}
    row = project(cycle)
    assert row['actor_allocation'] == cap
    assert row['actor_seconds'] == 15
    assert row['remaining_actor_seconds'] == cap-15
    assert row['wave_actor_ceiling'] == 6300
    assert row['budget_source'] == 'workflow'
    assert all(p.read_bytes() == data for p, data in before.items())


def test_grants_and_exhausted_overrun_do_not_reconstruct_allocation(retained):
    _, cycle, runner = retained
    flow = workflow(runner, 10)
    flow.grant_actor_seconds('recovery', 5, 'receipt')
    flow.state['intervals'] = dict(work=['implementation', 100., 122.])
    flow._save()
    row = project(cycle)
    assert row['actor_allocation'] == 15
    assert row['actor_seconds'] == 22
    assert row['remaining_actor_seconds'] == 0


@pytest.mark.parametrize('uncertain', [False, True])
def test_unresolved_work_keeps_allocation_but_never_claims_zero_elapsed(retained, uncertain):
    _, cycle, runner = retained
    flow = workflow(runner)
    if uncertain:
        flow.state['timing_uncertainties']['remote'] = dict(category='execution')
        flow._save()
    else:
        flow.accounting.start('thinking')
    row = project(cycle)
    assert row['actor_allocation'] == 3593
    assert row['actor_seconds'] is None
    assert 'remaining_actor_seconds' not in row
    assert row['accounting_state'] == ('uncertain' if uncertain else 'active')


def test_explicit_unlimited_is_different_from_missing_runner(retained):
    _, cycle, runner = retained
    row = project(cycle)
    assert 'actor_allocation' not in row
    assert 'remaining_actor_seconds' not in row
    assert row['actor_seconds'] is None
    workflow(runner, None)
    row = project(cycle)
    assert row['actor_allocation'] is None
    assert row['remaining_actor_seconds'] is None
    assert row['actor_seconds'] == 0


def test_missing_wave_allowance_is_not_defaulted(retained):
    _, cycle, _ = retained
    path = cycle/'waves/wave/wave.json'
    state = json.loads(path.read_text())
    del state['projects']['op']['allowance']
    path.write_text(json.dumps(state))
    assert 'wave_actor_ceiling' not in project(cycle)


def test_old_pinned_cli_evidence_is_refreshed_from_frozen_local_workflow(retained):
    instance, cycle, runner = retained
    flow = workflow(runner, 3593)
    flow.state['intervals'] = dict(work=['thinking', 100., 3700.])
    flow._save()
    old = dict(job=dict(request=dict(cycle_id='cycle')), inspection=dict(tasks=[], waves=[dict(
        wave_id='wave', projects=[dict(operation_id='op', actor_allocation=6300, actor_seconds=0)])]))
    result = inspection(old, instance=instance)
    row = result['waves'][0]['projects'][0]
    assert row['actor_allocation'] == 3593
    assert row['actor_seconds'] == 3600
    assert row['remaining_actor_seconds'] == 0
    assert row['wave_actor_ceiling'] == 6300


def test_legacy_evidence_without_workflow_does_not_pretend_ceiling_is_allocation():
    value = dict(inspection=dict(tasks=[], waves=[dict(wave_id='w', projects=[dict(
        actor_allocation=6300, actor_seconds=None, remaining_actor_seconds=6300)])]))
    row = inspection(value)['waves'][0]['projects'][0]
    assert row['wave_actor_ceiling'] == 6300
    assert 'actor_allocation' not in row
    assert 'remaining_actor_seconds' not in row
    assert value['inspection']['waves'][0]['projects'][0]['actor_allocation'] == 6300


@pytest.mark.parametrize('cycle_id', ['../outside', '/tmp/outside'])
def test_local_legacy_projection_rejects_cycle_escape(tmp_path, cycle_id):
    with pytest.raises(ValueError, match='escapes'):
        inspection(dict(job=dict(request=dict(cycle_id=cycle_id))), instance=tmp_path)


def test_dashboard_job_route_refreshes_old_cli_without_dispatching_commands(retained):
    from types import SimpleNamespace
    from flask import Flask
    from dashboard.rsi_execution import register_rsi_execution
    instance, _, runner = retained
    workflow(runner, 3598)
    calls = []
    def cli(*args):
        calls.append(args)
        return dict(job=dict(job_id='job', request=dict(cycle_id='cycle')), events=[],
                    inspection=dict(tasks=[], waves=[dict(wave_id='wave', projects=[dict(
                        operation_id='op', actor_allocation=6300)])]))
    app = Flask(__name__)
    register_rsi_execution(app, SimpleNamespace(handlers={}, enabled=False), '', cli, instance=instance)
    row = app.test_client().get('/api/rsi/jobs/job').json['inspection']['waves'][0]['projects'][0]
    assert row['actor_allocation'] == 3598
    assert row['remaining_actor_seconds'] == 3598
    assert calls == [('job', 'evidence', 'job')]


def test_current_cli_budget_fields_are_preferred_to_local_fallback(retained):
    instance, _, runner = retained
    workflow(runner, 3593)
    supplied = dict(tasks=[dict(task_id='new', actor_allocation=42, actor_union_seconds=2)],
                    waves=[dict(wave_id='w', projects=[dict(budget_source='workflow', actor_allocation=42)])])
    result = inspection(dict(job=dict(request=dict(cycle_id='cycle')), inspection=supplied), instance=instance)
    assert result['tasks'][0]['actor_allocation'] == 42
    assert result['waves'][0]['projects'][0]['actor_allocation'] == 42
