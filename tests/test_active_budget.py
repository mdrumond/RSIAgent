import json

import pytest

from core.active_budget import ActiveTimeBudget, current_budget


class Clock:
    now = 0
    def __call__(self):
        return self.now


def test_active_time_survives_resume_and_excludes_pause_and_downtime(tmp_path):
    clock = Clock()
    path = tmp_path / 'time.json'
    budget = ActiveTimeBudget(path, 10, clock=clock)
    with budget.active():
        assert current_budget() is budget
        clock.now += 3
        assert budget.remaining_seconds == 7
        with budget.pause():
            clock.now += 100
            with budget.pause():
                clock.now += 100
        clock.now += 2
        assert budget.elapsed_seconds == 5
    assert current_budget() is None
    clock.now += 1000
    resumed = ActiveTimeBudget(path, clock=clock)
    assert resumed.remaining_seconds == 5
    with resumed.active():
        clock.now += 5
        assert resumed.expired
    assert json.loads(path.read_text())['elapsed_seconds'] == 10


def test_checkpoint_survives_abandoned_process_and_rejects_allowance_changes(tmp_path):
    clock = Clock()
    path = tmp_path / 'time.json'
    budget = ActiveTimeBudget(path, 20, clock=clock)
    with budget.active():
        clock.now += 4
        budget.checkpoint()
        # A new owner sees only committed active time, never the old start stamp.
        clock.now += 500
        resumed = ActiveTimeBudget(path, clock=clock)
        assert resumed.elapsed_seconds == 4
    with pytest.raises(ValueError, match='changed'):
        ActiveTimeBudget(path, 21)
    with pytest.raises(ValueError, match='explicit'):
        ActiveTimeBudget(tmp_path / 'legacy.json')


def test_rejected_provider_duration_can_be_excluded(tmp_path):
    clock = Clock()
    budget = ActiveTimeBudget(tmp_path / 'time.json', 10, clock=clock)
    with budget.active():
        clock.now += 7
        budget.exclude(5)
        assert budget.elapsed_seconds == 2
