"""Time allocations survive resume and do not use action counters."""
import json

import pytest

from benchmarks.ascend.time_budget import AttemptTimeBudget, deadline_after


@pytest.mark.parametrize('seconds', [0, -1, True, float('inf'), float('nan')])
def test_time_allocation_requires_finite_positive_seconds(seconds):
    with pytest.raises(ValueError, match='max_seconds'):
        deadline_after(seconds)


def test_clean_resume_preserves_elapsed_and_excludes_idle_time(tmp_path):
    now, wall = [10.0], [100.0]
    path = tmp_path / 'time.json'
    budget = AttemptTimeBudget(path, 20, clock=lambda: now[0], wall_clock=lambda: wall[0])
    now[0] += 7
    budget.close()
    wall[0] += 1000
    now[0] = 90  # Another process has a different monotonic origin.
    resumed = AttemptTimeBudget(path, 20, clock=lambda: now[0], wall_clock=lambda: wall[0])
    assert resumed.deadline-now[0] == 13
    now[0] += 13
    resumed.close()
    expired = AttemptTimeBudget(path, 20, clock=lambda: now[0], wall_clock=lambda: wall[0])
    assert expired.deadline == now[0]
    expired.close()


def test_interrupted_operation_is_charged_on_resume(tmp_path):
    now, wall = [0.0], [100.0]
    path = tmp_path / 'time.json'
    AttemptTimeBudget(path, 10, clock=lambda: now[0], wall_clock=lambda: wall[0])
    wall[0] += 12  # Process disappeared without its final checkpoint.
    recovered = AttemptTimeBudget(path, 10, clock=lambda: now[0], wall_clock=lambda: wall[0])
    assert recovered.deadline == now[0]
    recovered.close()
    assert json.loads(path.read_text())['elapsed_seconds'] == 12
    with pytest.raises(ValueError, match='allocation changed'):
        AttemptTimeBudget(path, 20)
