"""Read-only accounting uses the canonical union, including overlapping work."""
import json

from rsi.contracts import BudgetLimits
from rsi.job_inspection import inspect_task
from rsi.runtime import Workflow


def test_exact_categories_exclusions_overlap_union_and_remaining(tmp_path):
    now = [100.]
    flow = Workflow(tmp_path / 'workflow.json', None, source_hash='frozen',
                    limits=BudgetLimits(actor_seconds=20, wall_seconds=100), clock=lambda: now[0])
    thinking = flow.accounting.start('thinking')
    now[0] = 102.
    compile = flow.accounting.start('compilation')
    now[0] = 105.
    flow.accounting.end(thinking)
    now[0] = 107.
    flow.accounting.end(compile)
    setup = flow.accounting.start('setup')
    now[0] = 111.
    flow.accounting.end(setup)
    transfer = flow.accounting.start('transfer')
    now[0] = 114.
    flow.accounting.end(transfer)
    before = (tmp_path / 'workflow.json').read_bytes()
    result = inspect_task(tmp_path, clock=lambda: now[0])
    assert result['charged_seconds']['thinking'] == 5
    assert result['charged_seconds']['compilation'] == 5
    assert result['actor_union_seconds'] == 7
    assert result['excluded_seconds']['setup'] == 4
    assert result['excluded_seconds']['transfer'] == 3
    assert result['remaining'] == {'actor_seconds': 13, 'wall_seconds': 86}
    assert result['accounting_state'] == 'complete'
    assert (tmp_path / 'workflow.json').read_bytes() == before


def test_open_interval_never_becomes_zero_or_unlimited_budget(tmp_path):
    now = [100.]
    flow = Workflow(tmp_path / 'workflow.json', None, source_hash='frozen',
                    limits=BudgetLimits(actor_seconds=20, wall_seconds=None), clock=lambda: now[0])
    flow.accounting.start('thinking')
    now[0] = 105.
    result = inspect_task(tmp_path, clock=lambda: now[0])
    assert result['accounting_state'] == 'active'
    assert result['actor_union_seconds'] is None
    assert result['charged_seconds']['thinking'] is None
    assert 'actor_seconds' not in result['remaining']
    assert result['remaining']['wall_seconds'] is None
    assert next(iter(result['intervals'].values()))[2] is None


def test_remote_uncertainty_retains_evidence_without_epoch_subtraction(tmp_path):
    flow = Workflow(tmp_path / 'workflow.json', None, source_hash='frozen', clock=lambda: 100.)
    flow.state['timing_uncertainties']['remote-operation'] = {
        'elapsed': 4, 'observed_start': 100, 'observed_end': 110, 'category': 'execution'}
    flow._save()
    result = inspect_task(tmp_path, clock=lambda: 120.)
    assert result['accounting_state'] == 'uncertain'
    assert result['actor_union_seconds'] is None
    assert result['charged_seconds']['execution'] is None
    assert result['timing_uncertainties'] == flow.state['timing_uncertainties']
