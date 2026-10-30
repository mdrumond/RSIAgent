"""Removed run formats cannot start compute or mutate existing roots."""
import json
from pathlib import Path

import pytest

from benchmarks.ascend.subject_campaign import SubjectCampaign
from benchmarks.ascend.optimizer_execution import OptimizerTaskExecutor
from optimizer.memory import freeze_source


def test_old_subject_run_is_rejected_before_memory_or_scheduler_creation(tmp_path):
    state = tmp_path / 'subject-campaign.json'
    original = b'{"schema":"ascend-subject-campaign-v1","projects":{}}'
    state.write_bytes(original)
    with pytest.raises(ValueError, match='fresh root'):
        SubjectCampaign(tmp_path, [], b'', select_wave=None, execute=None, verifier=None, complete=None)
    assert list(tmp_path.iterdir()) == [state]
    assert state.read_bytes() == original


@pytest.mark.parametrize('phase', ['adaptive-phase2', 'target-driven-phase2', 'optimizer-phase2'])
def test_removed_memory_roots_are_rejected_without_writes(tmp_path, phase):
    state = tmp_path / 'state.json'
    original = json.dumps(dict(phase=phase, terminal=dict(status='READY'))).encode()
    state.write_bytes(original)
    with pytest.raises(ValueError, match='unsupported'):
        freeze_source(tmp_path)
    assert list(tmp_path.iterdir()) == [state]
    assert state.read_bytes() == original


def test_old_task_journal_is_rejected_before_new_job_creation(tmp_path):
    journal = tmp_path / 'jobs/old/events.jsonl'
    journal.parent.mkdir(parents=True)
    original = b'{"kind":"phase2-attempt-request","payload":{"attempt_id":"old"}}\n'
    journal.write_bytes(original)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob('*'))
    with pytest.raises(ValueError, match='unsupported task journal'):
        OptimizerTaskExecutor(tmp_path, {}, {}, {})
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob('*')) == before
    assert journal.read_bytes() == original
