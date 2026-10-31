"""Linked diagnostic scheduling, shared elapsed allocation and child feedback."""
from dataclasses import replace

import pytest

from rsi.contracts import BudgetLimits
from rsi.cycles import CycleScheduler, HostTaskResult, SimplificationResult, SimplificationDeferred, SnapshotUpdate
from test_rsi_cycles import spec, finish


def test_each_child_updates_memory_before_the_next_decision(tmp_path):
    calls = []
    def run(project, memory, skills, context):
        calls.append(('run', project.project_id, memory, context['recovery']))
        return HostTaskResult('accepted' if context['recovery'] else 'failed', 'host-result', 2)
    def memory(previous, rows, context):
        calls.append(('memory', [row['project']['project_id'] for row in rows], context['recovery']))
        return SnapshotUpdate(previous + '+', .25)
    def simplify(parent, failure, context):
        calls.append(('simplify', parent.project_id))
        return SimplificationResult(replace(parent, project_id=parent.project_id+'-child',
            budgets=BudgetLimits(actor_seconds=2)), .5, 'observed-research')
    scheduler = CycleScheduler(tmp_path/'cycle', spec(), run, memory, simplify=simplify,
        recovery_feedback=True, recovery_limit=None, actor_seconds=30, recovery_seconds=15)
    try:
        finish(scheduler)
        assert [row for row in calls if row[0] == 'run'] == [
            ('run', '0', 'memory0', False), ('run', '1', 'memory0', False),
            ('run', '0-child', 'memory0+', True), ('run', '1-child', 'memory0++', True)]
        assert [row for row in calls if row[0] == 'memory'] == [
            ('memory', ['0', '1'], False), ('memory', ['0-child'], True), ('memory', ['1-child'], True)]
        assert [row['outcome'] for row in scheduler.state['originals']] == ['failed', 'failed']
        assert scheduler.state['skills'] == 'skills0'
        assert scheduler.state['spent'] == 9.75
        assert scheduler.state['recovery_spent'] == 5.5
    finally:
        scheduler.close()


def test_retrospective_parents_are_not_reexecuted_and_curriculum_can_select_another_parent(tmp_path):
    projects = spec()
    historical = [dict(project=p.to_dict(), original_id=p.project_id, operation_id='old-'+p.project_id,
        outcome='budget_exhausted', artifact_ref='retained-'+p.project_id, actor_seconds=7200) for p in projects.projects]
    calls = []
    def simplify(parent, failure, context):
        selected = '1' if not calls else '0'
        child = replace(projects.projects[int(selected)], project_id=selected+'-diagnostic',
                        budgets=BudgetLimits(actor_seconds=1))
        return SimplificationResult(child, .25, 'actual-research', selected)
    def run(project, memory, skills, context):
        calls.append((project.project_id, context['original_project_id'], memory))
        return HostTaskResult('accepted', 'new-child-result', 1)
    scheduler = CycleScheduler(tmp_path/'cycle', projects, run,
        lambda previous, rows, context: SnapshotUpdate(previous+'+'), simplify=simplify,
        recovery_feedback=True, recovery_limit=None, actor_seconds=10, recovery_seconds=10)
    try:
        scheduler.admit_failed_parents(historical, 'retained-catalog')
        scheduler.admit_failed_parents(historical, 'retained-catalog')
        finish(scheduler)
        assert calls == [('1-diagnostic', '1', 'memory0'), ('0-diagnostic', '0', 'memory0+')]
        assert scheduler.state['originals'] == []
        assert scheduler.state['failed_parent_admission']['rows'] == historical
        assert scheduler.state['spent'] == scheduler.state['recovery_spent'] == 2.5
        assert all(row['actor_seconds'] == 7200 for row in historical)
    finally:
        scheduler.close()


def test_shared_pool_exhaustion_queues_unfunded_memory_without_free_work(tmp_path):
    memory_calls = []
    def run(project, memory, skills, context):
        return HostTaskResult('failed', 'actual-result', 1.5 if context['recovery'] else 0)
    scheduler = CycleScheduler(tmp_path/'cycle', spec(), run,
        lambda *args: memory_calls.append(args) or SnapshotUpdate('memory1'),
        simplify=lambda parent, failure, context: SimplificationResult(replace(parent,
            project_id=parent.project_id+'-child', budgets=BudgetLimits(actor_seconds=1.5)), .5, 'research'),
        recovery_feedback=True, recovery_limit=None, actor_seconds=10, recovery_seconds=2)
    try:
        for _ in range(6):
            scheduler.step()
        assert scheduler.state['stopped'] and scheduler.state['phase'] == 'recovery_memory'
        assert scheduler.remaining()['recovery_seconds'] == 0
        assert len(memory_calls) == 1
        assert scheduler.state['recovery_feedback_rows'][0]['outcome'] == 'failed'
    finally:
        scheduler.close()


def test_known_research_exhaustion_charges_once_and_preserves_the_queue(tmp_path):
    projects = spec(count=1)
    old = dict(project=projects.projects[0].to_dict(), original_id='0', operation_id='old',
               outcome='failed', artifact_ref='proof', actor_seconds=100)
    scheduler = CycleScheduler(tmp_path/'cycle', projects, lambda *_: pytest.fail('no root retry'),
        lambda *_: pytest.fail('no free memory work'),
        simplify=lambda *args: SimplificationDeferred(2, 'observed-budget-receipt'),
        recovery_feedback=True, actor_seconds=5, recovery_seconds=5)
    try:
        scheduler.admit_failed_parents([old], 'catalog')
        scheduler.step()
        scheduler.step()
        assert scheduler.state['stopped'] and scheduler.state['spent'] == 2
        assert scheduler.state['recovery_spent'] == 2
        assert scheduler.state['recovery_queue'][0]['result'] == old
        assert scheduler.state['pending'] is None
    finally:
        scheduler.close()


def retained_rows(projects):
    return [dict(project=p.to_dict(), original_id=p.project_id, operation_id='old-'+p.project_id,
                 outcome='failed', artifact_ref='proof-'+p.project_id, actor_seconds=7200)
            for p in projects.projects]


@pytest.mark.parametrize('outcome', ['accepted', 'failed'])
def test_child_feedback_precedes_next_research_and_survives_restart(tmp_path, outcome):
    events = []
    projects = spec()
    def simplify(parent, failure, context):
        events.append(('research', parent.project_id))
        return SimplificationResult(replace(parent, project_id=parent.project_id+'-child',
            budgets=BudgetLimits(actor_seconds=1)), .25, 'research-receipt')
    def memory(previous, rows, context):
        assert context['recovery']
        events.append(('memory', rows[0]['outcome']))
        return SnapshotUpdate(previous+'+', .25)
    def run(project, memory, skills, context):
        events.append(('run', project.project_id, memory))
        return HostTaskResult(outcome, 'child-receipt', 1)
    kwargs = dict(simplify=simplify, recovery_feedback=True, actor_seconds=10, recovery_seconds=10)
    scheduler = CycleScheduler(tmp_path/'cycle', projects, run, memory, **kwargs)
    scheduler.admit_failed_parents(retained_rows(projects), 'catalog')
    scheduler.step()
    scheduler.step()
    assert scheduler.state['phase'] == 'recovery_memory'
    scheduler.close()
    scheduler = CycleScheduler(tmp_path/'cycle', projects, run, memory, **kwargs)
    try:
        finish(scheduler)
        assert events == [('research', '0'), ('run', '0-child', 'memory0'),
                          ('memory', outcome), ('research', '1'),
                          ('run', '1-child', 'memory0+'), ('memory', outcome)]
        assert scheduler.state['spent'] == scheduler.state['recovery_spent'] == 3
        assert scheduler.state['memory'] == 'memory0++'
    finally:
        scheduler.close()


@pytest.mark.parametrize('collision', ['failed', 'accepted'])
def test_child_cannot_reuse_any_imported_catalog_identity(tmp_path, collision):
    projects = spec()
    rows = retained_rows(projects)
    rows[1]['outcome'] = collision
    scheduler = CycleScheduler(tmp_path/'cycle', projects, None, None,
        simplify=lambda parent, *_: SimplificationResult(replace(parent, project_id='1',
            budgets=BudgetLimits(actor_seconds=1)), .25, 'receipt'),
        recovery_feedback=True, actor_seconds=10)
    try:
        scheduler.admit_failed_parents(rows, 'catalog')
        with pytest.raises(ValueError, match='identity already used'):
            scheduler.step()
        assert scheduler.state['spent'] == 0
        assert not scheduler.state['recoveries']
        assert scheduler.state['recovery_queue'][0]['spec']['project_id'] == '0'
    finally:
        scheduler.close()


def test_admission_rejects_changed_receipt_and_elapsed_without_mutation(tmp_path):
    projects = spec()
    scheduler = CycleScheduler(tmp_path/'cycle', projects, None, None,
        simplify=lambda *_: None, actor_seconds=10)
    try:
        rows = retained_rows(projects)
        scheduler.admit_failed_parents(rows, 'catalog')
        for changed_rows, receipt in [(rows, 'other'), ([dict(rows[0], actor_seconds=0)], 'catalog')]:
            with pytest.raises(ValueError):
                scheduler.admit_failed_parents(changed_rows, receipt)
        assert scheduler.state['spent'] == 0
        assert scheduler.state['failed_parent_admission']['rows'] == rows
    finally:
        scheduler.close()


@pytest.mark.parametrize('boundary', ['unbounded', 'progressed', 'phase2'])
def test_admission_requires_fresh_phase1_and_finite_elapsed_pool(tmp_path, boundary):
    from rsi.contracts import Mode
    from rsi.runtime import WorkflowError
    projects = spec(Mode.PHASE2 if boundary == 'phase2' else Mode.PHASE1)
    scheduler = CycleScheduler(tmp_path/'cycle', projects,
        lambda *_: HostTaskResult('accepted', 'receipt', 0), lambda *_: SnapshotUpdate('memory'),
        skill_update=lambda *_: SnapshotUpdate('candidate'), simplify=lambda *_: None,
        actor_seconds=None if boundary == 'unbounded' else 10)
    try:
        if boundary == 'progressed':
            scheduler.step()
        with pytest.raises((ValueError, WorkflowError)):
            scheduler.admit_failed_parents(retained_rows(projects), 'catalog')
        assert 'failed_parent_admission' not in scheduler.state
        assert scheduler.state['recovery_queue'] == []
    finally:
        scheduler.close()


def test_deferred_receipt_reconciles_once_and_resume_keeps_elapsed_bank(tmp_path):
    from rsi.runtime import WorkflowError
    captured = []
    projects = spec(count=1)
    def simplify(*args):
        captured.append(args[-1]['operation_id'])
        raise RuntimeError('lost return')
    kwargs = dict(simplify=simplify, recovery_feedback=True, actor_seconds=2, recovery_seconds=2)
    scheduler = CycleScheduler(tmp_path/'cycle', projects, None, None, **kwargs)
    scheduler.admit_failed_parents(retained_rows(projects), 'catalog')
    with pytest.raises(RuntimeError):
        scheduler.step()
    scheduler.close()
    scheduler = CycleScheduler(tmp_path/'cycle', projects, None, None, **kwargs)
    try:
        receipt = SimplificationDeferred(2, 'retained-elapsed')
        scheduler.reconcile(captured[0], receipt)
        with pytest.raises(WorkflowError):
            scheduler.reconcile(captured[0], receipt)
        scheduler.resume()
        scheduler.step()
        assert scheduler.state['stopped']
        assert scheduler.state['spent'] == scheduler.state['recovery_spent'] == 2
        assert scheduler.remaining()['actor_seconds'] == 0
        assert scheduler.remaining()['recovery_seconds'] == 0
        assert len(captured) == 1
        assert scheduler.state['recovery_queue'][0]['spec']['project_id'] == '0'
    finally:
        scheduler.close()


def test_invalid_selected_child_does_not_reorder_queue_or_change_lineage(tmp_path):
    projects = spec()
    scheduler = CycleScheduler(tmp_path/'cycle', projects, None, None,
        simplify=lambda *_: SimplificationResult(replace(projects.projects[1], project_id='child',
            budgets=BudgetLimits(actor_seconds=20)), .25, 'receipt', '1'),
        recovery_feedback=True, actor_seconds=10)
    try:
        scheduler.admit_failed_parents(retained_rows(projects), 'catalog')
        with pytest.raises(ValueError, match='allocation exceeds'):
            scheduler.step()
        assert [item['spec']['project_id'] for item in scheduler.state['recovery_queue']] == ['0', '1']
        assert scheduler.state['pending']['original_id'] == '0'
        assert scheduler.state['spent'] == 0
    finally:
        scheduler.close()
