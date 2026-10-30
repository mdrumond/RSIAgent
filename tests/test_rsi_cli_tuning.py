"""Manual CLI service tuning runs only its real heldout suite, even when paused."""
from test_rsi_services import instance as bound_instance, configure_heldout, two_projects, task_boundary, Roles
from rsi.configuration import set_configuration, show_configuration
import rsi.service


def test_manual_request_drains_actual_pair_restores_pause_and_is_idempotent(bound_instance, monkeypatch):
    services = bound_instance
    configure_heldout(services)
    config = show_configuration(services.root)['payload']
    config['tuning'].update(every_projects=None, at_wave_end=False)
    set_configuration(services.root, config)
    cycle = two_projects(services)
    try:
        task_boundary(cycle)
        row = cycle.scheduler.state['originals'][0]
        incumbent = cycle.scheduler.state['skills']
        cycle.scheduler.stop()
    finally:
        cycle.close()
    monkeypatch.setattr(rsi.service, 'InstanceServices', lambda root: services)
    request = dict(action='tune', mode='phase2', cycle_id='steered', request_id='manual-owned-request',
        steering={'project_ref': row['artifact_ref'], 'user_guidance': 'Detect numeric errors before speed'})
    events = []
    result = rsi.service.run_job(services.root, request, show_configuration(services.root),
        should_stop=lambda: False, emit=events.append)
    assert result['status'] == 'tuned' and result['evidence_ref']
    assert result['result']['decision'] == 'inconclusive'
    state = services.status('steered')['cycle.json']
    assert state['cursor'] == 1 and len(state['originals']) == 1
    assert state['stopped'] is True and state['skills'] == incumbent
    assert any('Detect numeric errors before speed' in prompt for prompt in Roles.prompts)
    original_spent = state['spent']
    original_role_count = len(Roles.prompts)
    again = rsi.service.run_job(services.root, request, should_stop=lambda: False, emit=events.append)
    assert again == result
    assert services.status('steered')['cycle.json']['spent'] == original_spent
    assert len(Roles.prompts) == original_role_count
