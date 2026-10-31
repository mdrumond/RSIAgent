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
