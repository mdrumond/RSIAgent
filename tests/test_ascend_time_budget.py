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



def test_fractional_grants_reopen_original_allocation_without_losing_elapsed_or_receipts(tmp_path):
    from benchmarks.ascend.actor_control import grant_elapsed
    now, wall = [0.0], [100.0]
    path = tmp_path / 'time.json'
    budget = AttemptTimeBudget(path, 10, clock=lambda: now[0], wall_clock=lambda: wall[0])
    now[0] += 4.25
    budget.close()
    grant_elapsed(path, 'first', .1)
    retained_max = grant_elapsed(path, 'second', .2)
    before = path.read_bytes()
    assert grant_elapsed(path, 'second', .2) == retained_max
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='different seconds'):
        grant_elapsed(path, 'second', .3)
    assert path.read_bytes() == before
    wall[0] += 1000  # A settled budget does not charge idle time.
    now[0] = 90
    resumed = AttemptTimeBudget(path, 10, clock=lambda: now[0], wall_clock=lambda: wall[0])
    assert resumed.seconds == retained_max
    assert resumed.elapsed == 4.25
    assert resumed.deadline - now[0] == pytest.approx(6.05)
    assert resumed.grants == {'first': .1, 'second': .2}
    now[0] += .75
    resumed.close()
    retained = json.loads(path.read_bytes())
    assert retained['elapsed_seconds'] == 5
    assert retained['max_seconds'] == retained_max
    assert retained['grants'] == {'first': .1, 'second': .2}
    with pytest.raises(ValueError, match='allocation changed'):
        AttemptTimeBudget(path, 10.001, clock=lambda: now[0], wall_clock=lambda: wall[0])
    assert json.loads(path.read_bytes()) == retained
