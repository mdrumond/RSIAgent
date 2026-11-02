"""Recovery work joins later waves; only original acceptance resolves an original."""
from dataclasses import replace

import pytest

from rsi.contracts import BudgetLimits
from rsi.cycles import CycleScheduler, HostTaskResult, SimplificationResult, SnapshotUpdate
from test_rsi_cycles import spec, finish


def build(tmp_path, *, limit=None, child_pass=True, root_pass=False, count=1, root_cap=100):
    trace = []
    blueprint = spec(count=count)
    blueprint = replace(blueprint, projects=tuple(replace(p, budgets=BudgetLimits(
        actor_seconds=root_cap, design_iterations=None, implementation_iterations=None,
        execution_tries=None)) for p in blueprint.projects))
    def run(project, memory, skills, ctx):
        kind = 'retry' if ctx['root_retry'] else 'child' if ctx['recovery'] else 'original'
        trace.append((kind, project.project_id, ctx['wave'], memory, project.budgets.actor_seconds))
        accepted = root_pass if kind == 'retry' else child_pass if kind == 'child' else project.project_id != '0'
        return HostTaskResult('accepted' if accepted else 'failed', 'trusted-host-result', 1.)
    def memory(previous, rows, ctx):
        trace.append(('memory', tuple((r['project']['project_id'], r['outcome']) for r in rows), ctx['wave']))
        return SnapshotUpdate(previous+'+', 0.)
    def simplify(parent, failure, ctx):
        trace.append(('research', parent.project_id, ctx['wave'], ctx['remaining']))
        child = replace(parent, project_id=parent.project_id+'-child-'+ctx['operation_id'],
                        budgets=BudgetLimits(actor_seconds=2))
        return SimplificationResult(child, .1, 'retained-investigation')
    def retry(parent, latest, ctx):
        trace.append(('prepare_retry', parent.project_id, ctx['wave'], latest['operation_id']))
        return parent
    args = dict(simplify=simplify, retry_original=retry, recovery_feedback=True,
        wave_size=2, actor_seconds=1000, recovery_seconds=500)
    if limit is not None:
        args['max_failed_recovery_attempts']=limit
    scheduler = CycleScheduler(tmp_path/'state.json', blueprint, run, memory, **args)
    return scheduler, trace, (blueprint, run, memory, args)


def test_normal_siblings_finish_before_research_and_root_retry_is_a_later_wave(tmp_path):
    s, trace, _ = build(tmp_path, count=3, root_pass=True)
    try:
        finish(s)
        actions = [r for r in trace if r[0] in ('original', 'child', 'retry', 'research')]
        assert [(r[0], r[1]) for r in actions[:2]] == [('original', '0'), ('original', '1')]
        assert next(r for r in trace if r[0]=='memory')[1] == (('0','failed'),('1','accepted'))
        assert next(r for r in actions if r[0]=='research')[2] == 2
        assert next(r for r in actions if r[0]=='original' and r[1]=='2')[2] == 2
        child = next(r for r in actions if r[0]=='child')
        retry = next(r for r in actions if r[0]=='retry')
        assert child[2] == 2 and retry[2] == 3
        assert child[3] != retry[3]
        assert s.state['originals'][0]['outcome'] == 'failed'
        assert s.state['root_retries'][0]['outcome'] == 'accepted'
        item = s.state['recovery_queue'][0]
        assert item['status']=='accepted' and item['root_spent']==2 and item['failed_attempts']==0
        assert item['root_spec']==s.state['originals'][0]['project']
        assert retry[4]==100
    finally:
        s.close()


@pytest.mark.parametrize('limit', [None, 2])
def test_completed_failed_findings_retry_original_then_freeze_and_resume_never_auto_reopens(tmp_path, limit):
    s, trace, args = build(tmp_path, limit=limit, child_pass=False, count=2)
    limit=5 if limit is None else limit
    finish(s)
    item = s.state['recovery_queue'][0]
    assert item['failed_attempts']==limit and item['status']=='unrecoverable'
    assert len(s.state['recoveries'])==limit
    assert len([r for r in trace if r[0]=='memory'])==2*limit+1
    assert len(s.state['root_retries'])==limit
    assert all(row['outcome']=='failed' for row in s.state['root_retries'])
    assert s.state['originals'][1]['outcome']=='accepted'
    s.close()
    blueprint, run, memory, kwargs = args
    reopened = CycleScheduler(tmp_path/'state.json', blueprint, run, memory, **kwargs)
    try:
        before=len(trace)
        reopened.resume();reopened.step()
        assert len(trace)==before
        assert reopened.state['recovery_queue'][0]['failed_attempts']==limit
        reopened.reopen_original('0')
        finish(reopened)
        item=reopened.state['recovery_queue'][0]
        assert item['failed_attempts']==2*limit and item['status']=='unrecoverable'
        assert item['root_spent']==1+2*limit # Cumulative telemetry survives explicit reopening.
    finally:
        reopened.close()


def test_successful_children_do_not_reset_failed_root_retries_and_each_retry_is_fresh(tmp_path):
    s, trace, _=build(tmp_path, limit=None, child_pass=True, root_cap=3)
    try:
        finish(s)
        item=s.state['recovery_queue'][0]
        assert item['root_spent']==6 and item['status']=='unrecoverable'
        assert item['failed_attempts']==5
        assert [r[4] for r in trace if r[0]=='retry']==[3]*5
        assert len(s.state['recoveries'])==5
        assert all(r['outcome']=='accepted' for r in s.state['recoveries'])
        assert all(r['outcome']=='failed' for r in s.state['root_retries'])
        assert s.state['cursor']==1
    finally:
        s.close()
