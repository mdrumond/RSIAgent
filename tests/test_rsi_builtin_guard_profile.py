"""The ignored-profile resolver binds the shipped guard, with fake installed CLI."""
import json
from types import SimpleNamespace

import pytest

from rsi.executor_profiles import ExecutorProfiles, GuardPending
from test_device_admission import setup


def test_builtin_profile_keeps_raw_pending_handle_and_observes_it_once(setup, tmp_path, monkeypatch):
    _, configuration, metadata, state = setup
    path = tmp_path / 'profiles.json'
    path.write_text(json.dumps({'profiles': {'frozen': dict(metadata=metadata,
        guard='rsi.device_admission:guard', guard_configuration={'product': 'a3', 'chip': 0})}}))
    resolver = ExecutorProfiles(str(path))
    monkeypatch.setattr(ExecutorProfiles, '_executor', lambda *args: configuration['executor'])
    request = SimpleNamespace(project=tmp_path / 'worker/runner/project',
                              operation_id='original-command', category='execution', profile_path=None)
    state['terminal'] = 'running'
    with pytest.raises(GuardPending) as pending:
        resolver.guard('frozen', request, phase='command', remaining=lambda: 60)
    retained = pending.value.admission_receipt
    assert retained['failure']['type'] == 'probe_pending'
    assert retained['raw_measurements']['submission']['handle'] == 'remote:test-target:job:original'
    assert retained['admitted'] is False
    state['terminal'] = 'completed'
    result = resolver.guard('frozen', request, phase='command', remaining=lambda: 60)
    assert result['admitted'] is True
    assert result['raw_measurements']['logs']['content']
    assert sum(args[0] == 'run' for args in state['calls']) == 1
    assert resolver('frozen').metadata == metadata


def test_builtin_profile_passes_alarm_permission_and_keeps_timing_healthy_only(setup, tmp_path, monkeypatch):
    from rsi.executor_profiles import GuardDenied
    from test_device_admission import probe
    _, configuration, metadata, state = setup
    path = tmp_path / 'alarm-profiles.json'
    path.write_text(json.dumps({'profiles': {'frozen': dict(metadata=metadata,
        guard='rsi.device_admission:guard', guard_configuration={
            'product': 'a3', 'chip': 0, 'allow_alarm': True})}}))
    resolver = ExecutorProfiles(str(path))
    monkeypatch.setattr(ExecutorProfiles, '_executor', lambda *args: configuration['executor'])
    state['probe'] = probe(health='Alarm', error='81B18603')
    request = SimpleNamespace(project=tmp_path / 'worker/runner/project',
        operation_id='ordinary', command='test', category='execution', profile_path=None)
    result = resolver.guard('frozen', request, phase='command', remaining=lambda: 60)
    assert result['admitted'] and result['degraded']
    assert result['measurements']['health'] == 'Alarm'
    assert result['measurements']['hardware_error_code'] == '81B18603'
    request.operation_id, request.command = 'timing', 'timing'
    with pytest.raises(GuardDenied) as denied:
        resolver.guard('frozen', request, phase='command', remaining=lambda: 60)
    assert denied.value.admission_receipt['failure']['type'] == 'unhealthy'


def test_cpl_profile_constructs_without_importing_or_forwarding_lease_binder_settings(tmp_path):
    path = tmp_path / 'binding-profile.json'
    path.write_text(json.dumps({'profiles': {'frozen': dict(
        target='test-target', runtime='test-runtime', workspace='/workspace', physical_device=2,
        installed_client=str(tmp_path / 'skill/scripts/cpl-remote'), profile_identity='frozen',
        command_cwd='/workspace/commands', guard='rsi.device_admission:guard',
        project_binder='not_imported_before_lease:bind_project',
        project_configuration={'private_host_builder_setting': 'configured'})}}))
    resolver = ExecutorProfiles(str(path))
    assert resolver.metadata('frozen')['physical_device'] == 2
    assert resolver('frozen').metadata['runtime'] == 'test-runtime'
    assert len(resolver.binding_contract('frozen')) == 64
