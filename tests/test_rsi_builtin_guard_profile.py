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


@pytest.mark.parametrize('receipt_written', [False, True])
def test_post_intent_cli_failure_is_guard_pending_and_never_resubmits(
        setup, tmp_path, monkeypatch, receipt_written):
    from rsi.python_execution import atomic_json
    from rsi.runtime import WorkflowError
    _, configuration, metadata, state = setup
    path = tmp_path / 'unknown-profiles.json'
    path.write_text(json.dumps({'profiles': {'frozen': dict(metadata=metadata,
        guard='rsi.device_admission:guard', guard_configuration={'product': 'a3', 'chip': 0})}}))
    resolver = ExecutorProfiles(str(path))
    executor = configuration['executor']
    monkeypatch.setattr(ExecutorProfiles, '_executor', lambda *args: executor)
    calls = []
    def failed_cli(args, remaining, *, result_path):
        calls.append(args)
        if args[0] == 'run' and receipt_written:
            atomic_json(args[args.index('--receipt-file') + 1], dict(
                target='test-target', handle='remote:test-target:job:original',
                runtime='test-runtime', npu=dict(devices=[2])))
        raise WorkflowError('retained CLI ambiguity')
    monkeypatch.setattr(executor, '_cli', failed_cli)
    request = SimpleNamespace(project=tmp_path / 'worker/runner/project',
                              operation_id='original-command', category='execution', profile_path=None)
    receipts = []
    for _ in range(2):
        with pytest.raises(GuardPending) as pending:
            resolver.guard('frozen', request, phase='command', remaining=lambda: 60)
        receipts.append(pending.value.admission_receipt)
    expected = 'probe_unknown' if receipt_written else 'submission_unknown'
    assert all(r['failure']['type'] == expected and not r['admitted'] for r in receipts)
    assert sum(args[0] == 'run' for args in calls) == 1
    assert [args[0] for args in calls] == (['run', 'observe'] if receipt_written else ['run'])
    assert all(r['elapsed_seconds'] >= 0 for r in receipts)
    assert receipts[0]['failure']['detail'] == 'retained CLI ambiguity'
    if receipt_written:
        assert all(r['raw_measurements']['submission']['handle'] ==
                   'remote:test-target:job:original' for r in receipts)
