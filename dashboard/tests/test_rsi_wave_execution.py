"""Wave lifecycle controls stay on the durable CLI queue."""
from pathlib import Path

from dashboard.rsi_ledger import inspection
from dashboard.tests.test_rsi_execution import execution, mutation_headers, settle


def test_recovery_uses_registered_cli_with_retained_operation_and_temporary_proof(execution):
    app, client, calls, headers = execution
    app.config['TEST_CAPABILITIES'] = {'harness_recovery': True}
    payload = dict(action='phase', mode='phase1', operation='recover', job_id='one',
        operation_id='original-admission', observation={'controller_proof': {
            'actor_closed': True, 'observation': 'original controller closure retained'}})
    response = client.post('/api/rsi/execute', json=payload, headers=mutation_headers(headers))
    assert settle(app, response)['status'] == 'succeeded'
    args = calls[-1][0]
    assert args[:2] == ['phase1', 'recover']
    assert args[args.index('--job') + 1] == 'one'
    assert args[args.index('--operation') + 1] == 'original-admission'
    assert not Path(args[args.index('--observation') + 1]).exists()


def test_unregistered_recovery_is_rejected_before_job_dispatch(execution):
    _, client, calls, headers = execution
    response = client.post('/api/rsi/execute', headers=mutation_headers(headers), json=dict(
        action='phase', mode='phase1', operation='recover', job_id='one',
        operation_id='original', observation={'controller_proof': {}}))
    assert response.status_code == 409
    assert all(args == ['capabilities'] for args, *_ in calls)


def test_projection_retains_unstarted_sibling_and_true_failed_outcome():
    waves = [{'wave_id': 'wave', 'status': 'observation_required', 'projects': [
        {'operation_id': 'done', 'status': 'completed', 'outcome': 'failed'},
        {'operation_id': 'unknown', 'status': 'observation_required', 'device': '2'},
        {'operation_id': 'future', 'status': 'ready', 'device': None}]}]
    result = inspection({'inspection': {'tasks': [], 'waves': waves}})
    assert result['available'] is True
    assert result['waves'] == waves
    assert result['tasks'] == []
