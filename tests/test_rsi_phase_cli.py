"""Phase inheritance requests preserve one service selector and frozen resumes."""
import json
import uuid

import pytest

from test_rsi_jobs import instance
from rsi import jobs
from rsi.cli import main
from rsi.configuration import _bytes, _publish


def previous(instance, mode='phase2'):
    _, directory = jobs._directory(instance)
    job_id = str(uuid.uuid4())
    request = dict(action='start', mode=mode, cycle_id='old', from_cycle='research',
                   inheritance='parent', input_overrides=['kdb'])
    record = dict(job_id=job_id, state='completed', process=None, request=request)
    _publish(jobs._path(directory, job_id), _bytes(record))
    return job_id, request


def test_start_forwards_exact_inheritance_fields_and_components(instance, monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, 'start', lambda root, request: calls.append(request) or {})
    assert main(['--instance', str(instance), 'phase2', 'start', '--cycle', 'application',
        '--from-cycle', 'research', '--inheritance', 'parent', '--input-override', 'memory',
        '--input-override', 'skills', '--input-override', 'memory']) == 0
    assert calls == [dict(action='start', mode='phase2', cycle_id='application',
        from_cycle='research', inheritance='parent', input_overrides=['memory', 'skills'])]


def test_inference_project_and_recovery_flag_do_not_drop_parent_policy(instance, tmp_path, monkeypatch):
    path = tmp_path / 'problem.json'
    path.write_text(json.dumps(dict(project_id='sum', goal='accurate sum', command=['python', 'check.py'])))
    calls = []
    monkeypatch.setattr(jobs, 'start', lambda root, request: calls.append(request) or {})
    assert main(['--instance', str(instance), 'inference', 'start', '--project', str(path),
        '--no-recovery', '--from-cycle', 'validated', '--inheritance', 'auto',
        '--input-override', 'kdb']) == 0
    assert calls[0]['recover'] is False
    assert calls[0]['projects'][0]['project_id'] == 'sum'
    assert calls[0]['from_cycle'] == 'validated' and calls[0]['inheritance'] == 'auto'
    assert calls[0]['input_overrides'] == ['kdb']


@pytest.mark.parametrize('flags', [[], ['--inheritance', 'baseline']])
def test_phase1_defaults_to_baseline(instance, monkeypatch, flags):
    calls = []
    monkeypatch.setattr(jobs, 'start', lambda root, request: calls.append(request) or {})
    assert main(['--instance', str(instance), 'phase1', 'start', *flags]) == 0
    assert calls[0]['inheritance'] == 'baseline' and calls[0]['from_cycle'] is None


@pytest.mark.parametrize('mode,flags', [
    ('phase1', ['--inheritance', 'baseline', '--from-cycle', 'old']),
    ('phase2', ['--inheritance', 'baseline', '--from-cycle', 'old'])])
def test_invalid_inheritance_rejected_before_dispatch(instance, monkeypatch, mode, flags):
    calls = []
    monkeypatch.setattr(jobs, 'start', lambda *args: calls.append(args))
    assert main(['--instance', str(instance), mode, 'start', *flags]) == 2
    assert calls == []


def test_revisit_selects_new_inputs_but_resume_preserves_original_request(instance, monkeypatch):
    identity, old = previous(instance)
    calls = []
    monkeypatch.setattr(jobs, 'start', lambda root, request: calls.append(request) or {})
    assert main(['--instance', str(instance), 'phase2', 'revisit', '--job', identity,
        '--cycle', 'new', '--from-cycle', 'new-research', '--inheritance', 'parent',
        '--input-override', 'skills']) == 0
    assert calls[-1]['from_cycle'] == 'new-research'
    assert calls[-1]['input_overrides'] == ['skills']
    assert calls[-1]['previous_cycle'] == 'old'
    assert main(['--instance', str(instance), 'phase2', 'revisit', '--job', identity,
        '--cycle', 'automatic']) == 0
    assert calls[-1]['from_cycle'] == 'old' and calls[-1]['inheritance'] == 'parent'
    assert calls[-1]['input_overrides'] == []
    assert main(['--instance', str(instance), 'phase2', 'resume', '--job', identity]) == 0
    assert calls[-1] == dict(old, action='resume', previous_job=identity)
    with pytest.raises(SystemExit) as error:
        main(['--instance', str(instance), 'phase2', 'resume', '--job', identity, '--from-cycle', 'new'])
    assert error.value.code == 2 and len(calls) == 3


@pytest.mark.parametrize('mode', ['phase1', 'phase2', 'inference'])
def test_revisit_preserves_previous_accepted_heads_unless_explicit_baseline(instance, monkeypatch, mode):
    identity, _ = previous(instance, mode=mode)
    calls = []
    monkeypatch.setattr(jobs, 'start', lambda root, request: calls.append(request) or {})
    assert main(['--instance', str(instance), mode, 'revisit', '--job', identity, '--cycle', 'again']) == 0
    assert calls[-1]['from_cycle'] == 'old' and calls[-1]['inheritance'] == 'parent'
    assert calls[-1]['input_overrides'] == []
    assert main(['--instance', str(instance), mode, 'revisit', '--job', identity,
        '--cycle', 'baseline-again', '--inheritance', 'baseline']) == 0
    assert calls[-1]['from_cycle'] is None and calls[-1]['inheritance'] == 'baseline'
    assert calls[-1]['input_overrides'] == []


def test_phase1_explicit_parent_is_forwarded_without_restricting_source_mode(instance, monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, 'start', lambda root, request: calls.append(request) or {})
    assert main(['--instance', str(instance), 'phase1', 'start', '--from-cycle', 'accepted-inference',
        '--inheritance', 'parent', '--input-override', 'kdb']) == 0
    assert calls[-1]['from_cycle'] == 'accepted-inference'
    assert calls[-1]['inheritance'] == 'parent' and calls[-1]['input_overrides'] == ['kdb']


def test_revisit_does_not_turn_later_resume_into_old_manual_tuning(instance, monkeypatch):
    identity, request = previous(instance)
    request.update(action='resume', origin_action='tune', steering={'user_guidance': 'old request'})
    path = jobs._path(instance / 'runs/jobs', identity)
    value = json.loads(path.read_text())
    value['request'] = request
    _publish(path, _bytes(value))
    calls = []
    monkeypatch.setattr(jobs, 'start', lambda root, request: calls.append(request) or {})
    assert main(['--instance', str(instance), 'phase2', 'revisit', '--job', identity, '--cycle', 'new']) == 0
    assert 'origin_action' not in calls[0] and 'steering' not in calls[0]
    assert calls[0]['from_cycle'] == 'old' and calls[0]['inheritance'] == 'parent'
