"""Every diagnostic and original retry receives its own elapsed allocation."""
import pytest

from rsi.cycles import HostTaskResult, SimplificationDeferred
from test_rsi_cycles import finish
from test_rsi_deferred_original_recovery import build


def test_exhausted_original_and_shared_pools_do_not_prevent_fresh_projects(tmp_path):
    scheduler, trace, _ = build(tmp_path, root_cap=1, root_pass=True)
    try:
        scheduler.config.update(actor_seconds=1, recovery_seconds=0)
        finish(scheduler)
        assert [row[0] for row in trace if row[0] in ('original', 'child', 'retry')] == [
            'original', 'child', 'retry']
        assert next(row for row in trace if row[0] == 'retry')[4] == 1
        assert scheduler.state['recovery_queue'][0]['status'] == 'accepted'
        assert scheduler.state['spent'] == pytest.approx(3.1)
    finally:
        scheduler.close()


@pytest.mark.parametrize('phase', ['original', 'child', 'retry'])
def test_host_proven_harness_failure_abandons_without_retrying_and_retains_memory(tmp_path, phase):
    scheduler, trace, _ = build(tmp_path)
    run = scheduler.runner
    def fault(project, memory, skills, context):
        kind = 'retry' if context['root_retry'] else 'child' if context['recovery'] else 'original'
        if kind == phase:
            trace.append(('harness', project.project_id, context['wave']))
            return HostTaskResult('failed', 'canonical-result', .5,
                classification='harness_failure', terminal_failure_ref='terminal-proof', audit_ref='scoped-audit')
        return run(project, memory, skills, context)
    scheduler.runner = fault
    try:
        finish(scheduler)
        assert len([row for row in trace if row[0] == 'harness']) == 1
        assert not scheduler.state['pending']
        assert scheduler.state['memory'].endswith('+')
        if phase == 'original':
            assert not scheduler.state['recovery_queue']
        else:
            item = scheduler.state['recovery_queue'][0]
            assert item['status'] == 'abandoned'
            assert item['failed_attempts'] == 0
        event = next(row for row in scheduler.state['events'] if row['kind'] == 'project_abandoned')
        assert event['terminal_failure_ref'] == 'terminal-proof'
        assert event['audit_ref'] == 'scoped-audit'
        before = len(trace)
        scheduler.resume()
        scheduler.step()
        assert len(trace) == before
    finally:
        scheduler.close()


@pytest.mark.parametrize('refs', [{}, {'terminal_failure_ref': 'proof'}, {'audit_ref': 'audit'}])
def test_harness_classification_requires_both_host_proof_references(refs):
    with pytest.raises(ValueError):
        HostTaskResult('failed', 'result', 1., classification='harness_failure', **refs)


def test_five_failed_original_retries_each_receive_the_full_project_cap(tmp_path):
    scheduler, trace, _ = build(tmp_path, root_cap=1, child_pass=False)
    try:
        finish(scheduler)
        retries = [row for row in trace if row[0] == 'retry']
        assert len(retries) == 5
        assert [row[4] for row in retries] == [1] * 5
        item = scheduler.state['recovery_queue'][0]
        assert item['failed_attempts'] == 5
        assert item['status'] == 'unrecoverable'
        assert len(scheduler.state['recoveries']) == 5
        assert scheduler.state['originals'][0]['outcome'] == 'failed'
    finally:
        scheduler.close()


def test_deferred_investigation_does_not_spend_an_original_retry(tmp_path):
    scheduler, trace, _ = build(tmp_path, root_cap=100)
    scheduler.simplify = lambda *args: SimplificationDeferred(.25, 'retained-investigation')
    try:
        # Inspect the first durable deferred receipt; do not require an infinite
        # no-proposal curriculum to complete.
        for _ in range(12):
            scheduler.step()
            if any(event['kind'] == 'simplification_deferred' for event in scheduler.state['events']):
                break
        assert any(event['kind'] == 'simplification_deferred' for event in scheduler.state['events'])
        assert scheduler.state['recovery_queue'][0]['failed_attempts'] == 0
        assert not scheduler.state.get('root_retries')
        assert scheduler.state['spent'] == 1.25
        assert not [row for row in trace if row[0] == 'retry']
    finally:
        scheduler.close()
